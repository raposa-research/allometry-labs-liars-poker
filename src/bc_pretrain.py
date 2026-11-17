"""Behavioral-cloning (BC) pretraining for Liar's Poker RNaD nets.

Initializes an RNaD policy/value net from a hand-crafted *heuristic* policy and
saves the result as an RNaD-loadable checkpoint (the same msgpack format
``load_checkpoint`` / ``restore_state`` reads), so a later RNaD run with
``load_checkpoint=<this>`` starts from the BC policy (a warm start).

Everything on the hot path is JAX-native and jitted (no Python game/step loop):

  * Stars-and-bars hand sampling is a single vectorized op over a batch of games.
  * Rollouts run ``batch_size`` parallel games under ONE jitted ``lax.scan`` over
    a fixed padded game length (``vmap`` over the batch, ``apply_action`` /
    ``legal_actions_mask`` / observations all vmapped), collecting
    ``(obs, target, legal, valid)`` into device arrays.
  * The heuristic target is a fully vectorized ``[B, num_actions]`` tensor
    (belief GATHERED from a precomputed log-survival table -- no per-state
    binomial recompute).
  * BC training is a jitted loss+grad Adam step over the on-device dataset.

The heuristic (per acting player, given public state + THAT player's hand h)
-----------------------------------------------------------------------------
Let ``belief(A) = P(A true | hand)`` for bid ``A=(count c, digit n)``:
    belief(A) = P(others' count of n >= c - h[n])
              = P(Binomial((P-1)*hand_length, 1/num_digits) >= c - h[n])
(=1 when c-h[n] <= 0).  We REUSE the exact closed-form log-survival numerics the
per-digit transformer head uses (``transformer_seq._log_survival_table``, same
clamp), so the belief is byte-consistent with that head.

  * P(challenge) (action 0, only when a bid exists) = 1 - belief(current_bid).
  * weight(A) for a legal bid = belief(A) * exp(-lam * jump(A)), where
    jump(A) = c_A - base_count and base_count = the current bid's count (or
    ``first_bid_base_count`` for the opening).  Same-count higher-digit -> jump 0.
  * Normalise over LEGAL actions only:
        target[challenge] = P(challenge)
        target[bid A]     = (1 - P(challenge)) * weight(A) / sum_legal weight
    If sum(weight)==0 the bid mass is spread UNIFORMLY over the legal bids.
"""

import argparse
import functools
import json

import numpy as np
import jax
import jax.numpy as jnp
import optax

from liars_poker_jax import (
    LiarsPokerConfig,
    LiarsPokerState,
    new_initial_state,
    apply_action,
    legal_actions_mask,
    is_terminal,
)
from transformer_seq import _log_survival_table
from rnad import RNaDConfig, EnvStep
from train_jax import (
    make_network,
    JaxLiarsPokerRNaDSolver,
    _lpt_net_config,
    _lpt_max_seq,
    _build_obs_batch,
    write_bytes,
    read_bytes,
)


# Mirrors the ``transformer_config`` default block in conf/train_jax.yaml.  RNaD
# runs that DON'T override ``transformer_config`` inherit this NON-EMPTY dict, and
# then ``JaxLiarsPokerRNaDSolver._serializable_state()`` INCLUDES a top-level
# ``transformer_config`` key (even for factored_mlp / mlp, which ignore it in
# ``make_network``).  ``restore_state`` (flax ``from_bytes``) requires the saved
# BC state to carry the SAME top-level keys, so the BC solver MUST be built with a
# matching ``transformer_config`` -- otherwise restore fails with a key mismatch.
DEFAULT_TRANSFORMER_CONFIG = {
    "d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 0,
}


def _effective_transformer_config(transformer_config):
    """The transformer_config the BC solver is built with (matches a default RNaD
    run when the caller passes None)."""
    return (DEFAULT_TRANSFORMER_CONFIG if transformer_config is None
            else transformer_config)


# --------------------------------------------------------------------------- #
# Static per-config tables (belief survival + per-action count/number decode)
# --------------------------------------------------------------------------- #
def belief_survival_table(config: LiarsPokerConfig, clamp: float = 30.0):
    """``P(others_count_d >= t)`` for ``t = 0..pool+1`` -> np [pool+2] float64.

    ``pool = (P-1)*hand_length`` (the OTHER players' unknown cards).  Derived from
    ``transformer_seq._log_survival_table`` (same clamp/log-space) then exp'd, so
    it is byte-consistent with the per-digit head's belief.
    """
    pool = (config.num_players - 1) * config.hand_length
    log_surv = np.asarray(_log_survival_table(pool, config.num_digits, clamp))
    return np.exp(log_surv)


def action_count_number(config: LiarsPokerConfig):
    """Per-action (count, number) decode arrays -> (count[A], number[A]) int.

    Action 0 (challenge) decodes to count=0 (never a legal bid, so unused).
    """
    D = config.num_digits
    a = np.arange(config.num_actions)
    bid_id = a - 1
    count = bid_id // D + 1
    number = bid_id % D + 1
    return count.astype(np.int32), number.astype(np.int32)


# --------------------------------------------------------------------------- #
# Vectorized heuristic target policy (pure jnp, batched over states)
# --------------------------------------------------------------------------- #
def heuristic_targets_batched(states, config: LiarsPokerConfig, belief_surv,
                              count_a, number_a, lam: float,
                              first_bid_base_count: int,
                              fixed_challenge: float = -1.0,
                              bid_suffix_max: bool = False):
    """Vectorized heuristic target -> (target [B, A] float32, legal [B, A] bool).

    ``states`` is a batched ``LiarsPokerState`` (leading dim B).  All arguments
    other than ``states`` are static/device constants; the whole function is
    jittable.  ``fixed_challenge >= 0`` overrides the belief-based P(challenge)
    with that CONSTANT (when challenge is legal); ``< 0`` (default) keeps the
    belief-based ``1 - belief(current bid)``.

    ``bid_suffix_max`` (static bool): apply the SUFFIX-MAX bluff-safety transform
    to the per-bid weights before normalizing -- each legal bid inherits the
    weight of the best HIGHER-OR-EQUAL legal bid (``p'_i = max_{j>=i} p_j`` in the
    (count,digit) == action-id order).  A low bid is a safe bluff if a good higher
    bid exists to escalate to, so its value = max over higher bids.  Makes the bid
    profile MONOTONE NON-INCREASING along (count,digit).  Challenge is untouched.
    """
    D = config.num_digits
    A = config.num_actions
    B = states.current_player.shape[0]
    belief_surv = jnp.asarray(belief_surv, dtype=jnp.float32)
    count_a = jnp.asarray(count_a, dtype=jnp.int32)      # [A]
    number_a = jnp.asarray(number_a, dtype=jnp.int32)    # [A]
    L = belief_surv.shape[0] - 1

    legal = jax.vmap(lambda s: legal_actions_mask(s, config))(states)  # [B, A]
    cp = states.current_player                                          # [B]
    acting_hand = states.hands[jnp.arange(B), cp]                       # [B, H]
    digit_ids = jnp.arange(1, D + 1, dtype=jnp.int32)
    hist = jnp.sum(
        acting_hand[:, :, None] == digit_ids[None, None, :], axis=1
    ).astype(jnp.int32)                                                 # [B, D]

    has_bid = states.current_bid_action >= 0                           # [B]
    cur_bid_id = jnp.where(has_bid, states.current_bid_action - 1, 0)
    cur_count = cur_bid_id // D + 1                                     # [B]
    cur_number = cur_bid_id % D + 1                                     # [B]

    # belief per bid action: gather the acting player's held count of number_a.
    h_of_num = hist[:, number_a - 1]                                   # [B, A]
    thr = jnp.clip(count_a[None, :] - h_of_num, 0, L)                  # [B, A]
    belief_a = belief_surv[thr]                                        # [B, A]

    # belief of the CURRENT bid (for P(challenge)).
    h_cur = hist[jnp.arange(B), cur_number - 1]                       # [B]
    thr_cur = jnp.clip(cur_count - h_cur, 0, L)
    belief_cur = belief_surv[thr_cur]                                 # [B]
    # belief-based P(challenge), or a constant when fixed_challenge >= 0.
    p_chal_val = jnp.where(fixed_challenge >= 0.0,
                           jnp.float32(fixed_challenge), 1.0 - belief_cur)
    p_challenge = jnp.where(has_bid, p_chal_val, 0.0)                # [B]

    base_count = jnp.where(
        has_bid, cur_count, jnp.int32(first_bid_base_count)
    )                                                                 # [B]
    jump = count_a[None, :] - base_count[:, None]                    # [B, A]
    weight = belief_a * jnp.exp(-lam * jump.astype(jnp.float32))     # [B, A]

    action_ids = jnp.arange(A)
    legal_bids = legal & (action_ids[None, :] != 0)                 # [B, A]
    w = jnp.where(legal_bids, weight, 0.0)
    if bid_suffix_max:
        # Suffix-max over action ids (== (count,digit) ascending): each bid
        # inherits the best higher-or-equal legal bid's weight.  Illegal bids are
        # 0 (never win a max), so this equals the max over LEGAL j >= i.  Re-mask
        # to legal so the monotone profile is renormalized over legal bids below.
        w = jnp.where(
            legal_bids, jax.lax.cummax(w, axis=w.ndim - 1, reverse=True), 0.0)
    sum_w = jnp.sum(w, axis=-1)                                      # [B]
    n_bids = jnp.sum(legal_bids, axis=-1)                           # [B]
    safe_n = jnp.maximum(n_bids, 1)
    uniform_w = jnp.where(legal_bids, 1.0 / safe_n[:, None], 0.0)
    bid_dist = jnp.where(
        (sum_w > 0)[:, None], w / jnp.maximum(sum_w, 1e-12)[:, None], uniform_w
    )                                                               # [B, A]

    bid_mass = 1.0 - p_challenge                                     # [B]
    target = bid_dist * bid_mass[:, None]                           # [B, A] (col 0 = 0)
    challenge_target = jnp.where(legal[:, 0], p_challenge, 0.0)     # [B]
    target = target.at[:, 0].set(challenge_target)

    # No legal bids (bids exhausted): all mass on the challenge action.
    no_bids = n_bids == 0
    onehot0 = (action_ids == 0).astype(target.dtype)
    target = jnp.where(no_bids[:, None], onehot0[None, :], target)
    return target.astype(jnp.float32), legal


# --------------------------------------------------------------------------- #
# Scalar reference heuristic (oracle for tests + sanity printing)
# --------------------------------------------------------------------------- #
class HeuristicPolicy:
    """Scalar reference implementation of the heuristic (single state)."""

    def __init__(self, config: LiarsPokerConfig, lam: float = 1.0,
                 first_bid_base_count: int = 1, belief_clamp: float = 30.0,
                 challenge_prob=None, bid_suffix_max: bool = False):
        self.config = config
        self.lam = float(lam)
        self.first_bid_base_count = int(first_bid_base_count)
        self.challenge_prob = challenge_prob   # None = belief-based
        self.bid_suffix_max = bool(bid_suffix_max)
        self._belief_surv = belief_survival_table(config, belief_clamp)
        self._count, self._number = action_count_number(config)

    def belief_prob(self, count, number, hist):
        thr = int(count) - int(hist[int(number) - 1])
        thr = min(max(thr, 0), len(self._belief_surv) - 1)
        return float(self._belief_surv[thr])

    def hand_histogram(self, hand):
        D = self.config.num_digits
        hand = np.asarray(hand)
        return np.array([int(np.sum(hand == d)) for d in range(1, D + 1)],
                        dtype=np.int64)

    def target(self, state):
        config = self.config
        D = config.num_digits
        A = config.num_actions
        player = int(state.current_player)
        legal = np.asarray(legal_actions_mask(state, config)).astype(bool)
        hist = self.hand_histogram(np.asarray(state.hands)[player])
        has_bid = int(state.current_bid_action) >= 0
        target = np.zeros(A, dtype=np.float64)

        if has_bid:
            cur_bid_id = int(state.current_bid_action) - 1
            cur_count = cur_bid_id // D + 1
            cur_number = cur_bid_id % D + 1
            base_count = cur_count
            p_challenge = (float(self.challenge_prob)
                           if self.challenge_prob is not None
                           else 1.0 - self.belief_prob(cur_count, cur_number, hist))
        else:
            base_count = self.first_bid_base_count
            p_challenge = 0.0

        legal_bids = np.where(legal)[0]
        legal_bids = legal_bids[legal_bids != 0]
        if legal_bids.size == 0:
            target[0] = 1.0
            return target, legal

        weights = np.zeros(A, dtype=np.float64)
        for a in legal_bids:
            c, n = int(self._count[a]), int(self._number[a])
            weights[a] = self.belief_prob(c, n, hist) * np.exp(
                -self.lam * (c - base_count))
        if self.bid_suffix_max:
            # Suffix-max over action ids (illegal = 0), then re-mask to legal.
            sm = np.maximum.accumulate(weights[::-1])[::-1]
            weights = np.where(legal & (np.arange(A) != 0), sm, 0.0)
        sum_w = weights[legal_bids].sum()
        bid_mass = 1.0 - p_challenge
        if sum_w <= 0.0:
            target[legal_bids] = bid_mass / legal_bids.size
        else:
            target[legal_bids] = bid_mass * weights[legal_bids] / sum_w
        if legal[0]:
            target[0] = p_challenge
        return target, legal


# --------------------------------------------------------------------------- #
# Uniform hand sampling (stars-and-bars), batched + vectorized
# --------------------------------------------------------------------------- #
def sample_histograms_batched(rng_key, batch, num_players, hand_length,
                              num_digits):
    """Uniform compositions of ``hand_length`` into ``num_digits`` bins.

    Returns int32 ``[batch, num_players, num_digits]``.  Uniform over DISTINCT
    histograms via stars-and-bars: pick num_digits-1 bar positions uniformly
    among hand_length+num_digits-1 slots (argsort of random keys = a uniform
    random subset), the gaps are the per-digit counts.  Vectorized over the whole
    ``batch*num_players`` axis (no Python loop).
    """
    H, D, P = int(hand_length), int(num_digits), int(num_players)
    n = batch * P
    if D == 1:
        return jnp.full((batch, P, 1), H, dtype=jnp.int32)
    slots = H + D - 1
    keys = jax.random.uniform(rng_key, (n, slots))
    order = jnp.argsort(keys, axis=-1)                       # [n, slots]
    bars = jnp.sort(order[:, :D - 1], axis=-1)               # [n, D-1]
    left = jnp.full((n, 1), -1, dtype=bars.dtype)
    right = jnp.full((n, 1), slots, dtype=bars.dtype)
    pos = jnp.concatenate([left, bars, right], axis=-1)      # [n, D+1]
    hist = jnp.diff(pos, axis=-1) - 1                        # [n, D]
    return hist.reshape(batch, P, D).astype(jnp.int32)


def histograms_to_hands_batched(hist, hand_length):
    """Composition histograms ``[..., D]`` -> hands ``[..., hand_length]`` (1..D).

    Vectorized ``repeat``: slot ``j`` belongs to digit index
    ``sum_d (cumsum(hist)[d] <= j)`` -> digit value ``+1``.
    """
    H = int(hand_length)
    cum = jnp.cumsum(hist, axis=-1)                          # [..., D]
    positions = jnp.arange(H)                                # [H]
    digit_idx = jnp.sum(
        cum[..., None, :] <= positions[..., :, None], axis=-1
    )                                                        # [..., H]
    return (digit_idx + 1).astype(jnp.int32)


def initial_states_batched(config: LiarsPokerConfig, hands):
    """Batched post-deal ``LiarsPokerState`` from hands ``[B, P, H]`` (1..D)."""
    B = hands.shape[0]
    P = config.num_players
    return LiarsPokerState(
        hands=jnp.asarray(hands, dtype=jnp.int32),
        deal_step=jnp.full((B,), config.total_cards, dtype=jnp.int32),
        bid_history=jnp.zeros((B, config.max_bids, P), dtype=jnp.int32),
        challenge_history=jnp.zeros((B, config.max_bids, P), dtype=jnp.int32),
        current_player=jnp.zeros((B,), dtype=jnp.int32),
        bid_originator=jnp.full((B,), -1, dtype=jnp.int32),
        current_bid_action=jnp.full((B,), -1, dtype=jnp.int32),
        num_challenges=jnp.zeros((B,), dtype=jnp.int32),
        is_rebid=jnp.zeros((B,), dtype=jnp.bool_),
        winner=jnp.full((B,), -1, dtype=jnp.int32),
        loser=jnp.full((B,), -1, dtype=jnp.int32),
    )


def sample_histogram(rng: np.random.RandomState, hand_length, num_digits):
    """Scalar uniform composition of ``hand_length`` into ``num_digits`` bins.

    Numpy reference for the batched ``sample_histograms_batched`` (used by tests
    and convenience callers).
    """
    n, k = int(hand_length), int(num_digits)
    if k == 1:
        return np.array([n], dtype=np.int64)
    slots = n + k - 1
    bars = np.sort(rng.choice(slots, size=k - 1, replace=False))
    pos = np.concatenate([[-1], bars, [slots]])
    return (np.diff(pos) - 1).astype(np.int64)


def histogram_to_hand(hist, hand_length):
    """Scalar composition histogram [D] -> hand [hand_length] of digits 1..D."""
    hand = []
    for d, c in enumerate(hist, start=1):
        hand.extend([d] * int(c))
    hand = np.array(hand, dtype=np.int32)
    assert hand.shape[0] == int(hand_length), (hand.shape, hand_length)
    return hand


def make_dealt_state(config: LiarsPokerConfig, hands):
    """Single post-deal state (values 1..D) -- convenience for tests."""
    base = new_initial_state(config)
    return base.replace(
        hands=jnp.asarray(hands, dtype=jnp.int32),
        deal_step=jnp.asarray(config.total_cards, dtype=jnp.int32),
    )


# --------------------------------------------------------------------------- #
# Vectorized rollout collection (one jitted lax.scan over a batch of games)
# --------------------------------------------------------------------------- #
@functools.partial(
    jax.jit,
    static_argnums=(0, 2, 3, 4, 5, 6, 7, 8, 9, 15, 18),
)
def _collect_scan(config, rng_key, batch, num_players, hand_length, num_digits,
                  network_type, history_encoding, max_seq, traj_max,
                  belief_surv, count_a, number_a, lam, first_bid_base_count,
                  return_states=False, explore_eps=0.0, fixed_challenge=-1.0,
                  bid_suffix_max=False):
    """Roll out ``batch`` parallel games for ``traj_max`` steps under one scan.

    Returns ``(obs, target, legal, valid, is_random)`` with leading dims
    ``[traj_max, batch]`` (plus the per-step ``states`` pytree when
    ``return_states``).  ``obs`` is a flat ``[T, B, obs_dim]`` array
    (mlp/factored) or a batched ``LPInputs`` pytree (lpt).  ``valid[t, b]`` is
    True iff game ``b`` was live (non-terminal) at step ``t``.

    EPSILON EXPLORATION: with prob ``explore_eps`` (per game, per step) the actor
    advances the game with a UNIFORM-RANDOM legal action instead of the heuristic
    sample -- this reaches the rare forced-decision states (impossible/certain
    current bid) that pure on-policy rollouts miss.  ``is_random[t, b]`` flags
    those steps so the BC loss can EXCLUDE them (we don't train on the deviating
    step itself, but DO train on the heuristic-action steps reached afterward).
    ``explore_eps == 0.0`` is byte-identical to the on-policy rollout.
    """
    hand_key, scan_key = jax.random.split(rng_key)
    hist0 = sample_histograms_batched(
        hand_key, batch, num_players, hand_length, num_digits)
    hands0 = histograms_to_hands_batched(hist0, hand_length)
    states0 = initial_states_batched(config, hands0)

    net_config = config  # BC disallows lpt capacity (asserted in run_pipeline)

    def step(carry, _):
        states, rng = carry
        rng, heur_key, rand_key, bern_key = jax.random.split(rng, 4)

        obs = _build_obs_batch(
            states, config, network_type, history_encoding, max_seq,
            lambda s: s.current_player, net_config,
        )
        target, legal = heuristic_targets_batched(
            states, config, belief_surv, count_a, number_a, lam,
            first_bid_base_count, fixed_challenge, bid_suffix_max)
        valid = ~jax.vmap(is_terminal)(states)

        # Heuristic action (gumbel-max over the teacher target).
        gh = jax.random.gumbel(heur_key, target.shape, dtype=jnp.float32)
        act_heur = jnp.argmax(jnp.log(target + 1e-45) + gh, axis=-1)
        # Uniform-random LEGAL action (gumbel-max over the legal mask).
        gr = jax.random.gumbel(rand_key, target.shape, dtype=jnp.float32)
        legal_f = legal.astype(jnp.float32)
        act_rand = jnp.argmax(jnp.log(legal_f + 1e-45) + gr, axis=-1)
        # Per-game bernoulli: explore vs exploit.
        is_random = jax.random.bernoulli(bern_key, explore_eps, (batch,))
        actions = jnp.where(is_random, act_rand, act_heur).astype(jnp.int32)

        new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(
            states, actions)
        outs = (obs, target, legal, valid, is_random)
        if return_states:
            outs = outs + (states,)
        return (new_states, rng), outs

    _, outs = jax.lax.scan(
        step, (states0, scan_key), xs=None, length=traj_max)
    return outs


def collect_dataset(config, network_type, history_encoding, max_seq, *,
                    n_rollout_games, batch_size, traj_max, lam,
                    first_bid_base_count, seed, verbose=True,
                    return_states=False, explore_eps=0.0, challenge_prob=None,
                    bid_suffix_max=False):
    """Collect a BC dataset by running batched rollouts under jitted scans.

    Runs ``ceil(n_rollout_games / batch_size)`` jitted batches; keeps only the
    valid (live-decision) states that were NOT random-exploration steps.  Returns
    ``(obs, targets, legals)`` with leading dim N (device arrays for obs; numpy
    for targets/legals slicing).  When ``return_states`` also returns the batched
    numpy ``LiarsPokerState`` for the kept decisions (worst-case decode).

    ``explore_eps > 0`` injects uniform-random legal actions into the rollout for
    coverage of the rare forced-decision states; the steps where the actor
    deviated randomly are EXCLUDED from the returned dataset (loss valid-mask =
    live AND NOT is_random), while the heuristic-action states reached afterwards
    are kept and trained with their teacher target.
    """
    belief_surv = jnp.asarray(belief_survival_table(config))
    count_a, number_a = action_count_number(config)
    count_a = jnp.asarray(count_a)
    number_a = jnp.asarray(number_a)

    n_batches = max(1, (n_rollout_games + batch_size - 1) // batch_size)
    key = jax.random.PRNGKey(seed)

    obs_parts, tgt_parts, leg_parts, state_parts = [], [], [], []
    total, total_random = 0, 0
    for bi in range(n_batches):
        key, sub = jax.random.split(key)
        out = _collect_scan(
            config, sub, batch_size, config.num_players, config.hand_length,
            config.num_digits, network_type, history_encoding, max_seq,
            traj_max, belief_surv, count_a, number_a, float(lam),
            int(first_bid_base_count), return_states, float(explore_eps),
            float(challenge_prob) if challenge_prob is not None else -1.0,
            bool(bid_suffix_max))
        if return_states:
            obs, target, legal, valid, is_random, states = out
        else:
            obs, target, legal, valid, is_random = out
        # Flatten [T, B, ...] -> [T*B, ...] and keep valid, NON-random rows
        # (host mask; one gather per batch, NOT per state/step).
        valid_flat = np.asarray(valid).reshape(-1)
        random_flat = np.asarray(is_random).reshape(-1)
        keep = np.where(valid_flat & ~random_flat)[0]
        total_random += int((valid_flat & random_flat).sum())
        T, B = valid.shape
        obs_flat = jax.tree_util.tree_map(
            lambda x: x.reshape((T * B,) + x.shape[2:]), obs)
        obs_parts.append(jax.tree_util.tree_map(lambda x: x[keep], obs_flat))
        tgt_parts.append(np.asarray(target).reshape(T * B, -1)[keep])
        leg_parts.append(np.asarray(legal).reshape(T * B, -1)[keep])
        if return_states:
            st_flat = jax.tree_util.tree_map(
                lambda x: np.asarray(x).reshape((T * B,) + x.shape[2:])[keep],
                states)
            state_parts.append(st_flat)
        total += keep.shape[0]
        if verbose:
            extra = (f", {total_random} random-excluded" if explore_eps > 0
                     else "")
            print(f"  batch {bi + 1}/{n_batches}: +{keep.shape[0]} states "
                  f"(total {total}{extra})")

    obs = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0), *obs_parts)
    targets = np.concatenate(tgt_parts, axis=0)
    legals = np.concatenate(leg_parts, axis=0)
    if return_states:
        states = jax.tree_util.tree_map(
            lambda *xs: np.concatenate(xs, axis=0), *state_parts)
        return obs, targets, legals, states
    return obs, targets, legals


# --------------------------------------------------------------------------- #
# BC training (jitted loss+grad Adam over the on-device dataset)
# --------------------------------------------------------------------------- #
def _make_env_step(obs, legal, num_players):
    B = legal.shape[0]
    return EnvStep(
        obs=obs,
        legal=jnp.asarray(legal).astype(jnp.int8),
        player_id=jnp.zeros(B, dtype=jnp.float32),
        valid=jnp.ones(B, dtype=jnp.float32),
        rewards=jnp.zeros((B, num_players), dtype=jnp.float32),
    )


def _index_obs(obs, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], obs)


def _smooth_target(target, legal, uniform_mix):
    """``target'(a) = (1-m)*target(a) + m*(1/n_legal)`` on legal a (0 illegal).

    The uniform floor (``m/n_legal``) keeps ``log target'`` bounded so a reverse
    KL ``KL(net||target')`` can't blow up on teacher-~0 actions.  ``m == 0`` is a
    no-op (byte-identical to ``target``).  Renormalized over legal actions.
    """
    legal_f = legal.astype(target.dtype)
    n_legal = jnp.sum(legal_f, axis=-1, keepdims=True)
    uni = legal_f / jnp.maximum(n_legal, 1.0)
    sm = (1.0 - uniform_mix) * target + uniform_mix * uni
    sm = sm * legal_f
    return sm / jnp.maximum(jnp.sum(sm, axis=-1, keepdims=True), 1e-12)


def bc_loss(params, network, obs, legal, target, num_players, *,
            loss_type="forward_ce", uniform_mix=0.01):
    """Full-distribution BC loss (masked to legal).

    ``forward_ce``: ``-sum_a target(a) * log net(a)`` (mode-covering; the
    original objective).  ``reverse_kl``: ``KL(net || target')`` where
    ``target' = smooth(target)`` -- ``sum_a net(a)*(log net(a) - log target'(a))``
    (mode-seeking -> drives net mass OFF teacher-unlikely actions).
    """
    env = _make_env_step(obs, legal, num_players)
    pi, _v, log_pi, _logit = network.apply(params, env)
    legal_f = jnp.asarray(legal, dtype=pi.dtype)
    if loss_type == "reverse_kl":
        sm = _smooth_target(target, legal_f, uniform_mix)
        log_sm = jnp.log(jnp.where(legal_f > 0, sm, 1.0))
        # net is 0 on illegal (log_pi masked), so restrict to legal explicitly.
        term = pi * (log_pi - log_sm)
        rkl = jnp.sum(jnp.where(legal_f > 0, term, 0.0), axis=-1)
        return jnp.mean(rkl)
    ce = -jnp.sum(target * log_pi, axis=-1)
    return jnp.mean(ce)


def train_bc(network, params, obs, legals, targets, num_players, *,
             epochs=30, batch_size=1024, lr=1e-3, seed=0, val_frac=0.1,
             loss_type="forward_ce", uniform_mix=0.01, verbose=True):
    """BC-train ``params`` to the heuristic ``targets``; return (params, metrics).

    ``loss_type`` selects ``forward_ce`` (mode-covering) or ``reverse_kl``
    (mode-seeking, with ``uniform_mix`` target smoothing).
    """
    targets = jnp.asarray(targets, dtype=jnp.float32)
    legals = jnp.asarray(legals)
    N = targets.shape[0]

    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx = jnp.asarray(perm[:n_val])
    train_idx = np.asarray(perm[n_val:])

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    loss_and_grad = jax.jit(
        jax.value_and_grad(
            lambda p, o, l, t: bc_loss(p, network, o, l, t, num_players,
                                       loss_type=loss_type,
                                       uniform_mix=uniform_mix)))

    @jax.jit
    def eval_metrics(p, o, l, t):
        env = _make_env_step(o, l, num_players)
        pi, _v, log_pi, _logit = network.apply(p, env)
        legal_f = jnp.asarray(l, dtype=pi.dtype)
        # Forward KL / CE (net covers teacher's modes?).
        ce = -jnp.sum(t * log_pi, axis=-1)
        safe_t = jnp.where(t > 0, t, 1.0)
        fkl = jnp.sum(jnp.where(t > 0, t * (jnp.log(safe_t) - log_pi), 0.0),
                      axis=-1)
        # Reverse KL to the SMOOTHED teacher (the reverse_kl objective).
        sm = _smooth_target(t, legal_f, uniform_mix)
        log_sm = jnp.log(jnp.where(legal_f > 0, sm, 1.0))
        rkl = jnp.sum(jnp.where(legal_f > 0, pi * (log_pi - log_sm), 0.0),
                      axis=-1)
        # Net entropy (over legal) -- watch for reverse-KL over-collapse.
        ent = -jnp.sum(jnp.where(legal_f > 0, pi * log_pi, 0.0), axis=-1)
        return (jnp.mean(ce), jnp.mean(fkl), jnp.mean(rkl), jnp.mean(ent))

    @jax.jit
    def update(params, opt_state, o, l, t):
        loss, grads = loss_and_grad(params, o, l, t)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    def gather(idx):
        idx = jnp.asarray(idx)
        return _index_obs(obs, idx), legals[idx], targets[idx]

    val_o, val_l, val_t = gather(val_idx)
    n_train = train_idx.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)

    for ep in range(epochs):
        ep_perm = train_idx[rng.permutation(n_train)]
        for step in range(steps_per_epoch):
            b = ep_perm[step * batch_size:(step + 1) * batch_size]
            if b.size == 0:
                continue
            b = jnp.asarray(b)
            params, opt_state, _ = update(
                params, opt_state, _index_obs(obs, b), legals[b], targets[b])
        if verbose:
            v_ce, v_fkl, v_rkl, v_ent = eval_metrics(params, val_o, val_l, val_t)
            print(f"  epoch {ep + 1:3d}/{epochs}  val CE={float(v_ce):.4f}  "
                  f"fwdKL={float(v_fkl):.4f}  revKL={float(v_rkl):.4f}  "
                  f"netH={float(v_ent):.4f}")

    v_ce, v_fkl, v_rkl, v_ent = eval_metrics(params, val_o, val_l, val_t)
    t_ce, t_fkl, t_rkl, t_ent = eval_metrics(params, *gather(train_idx))
    metrics = {
        "val_ce": float(v_ce), "val_kl": float(v_fkl),
        "val_reverse_kl": float(v_rkl), "val_net_entropy": float(v_ent),
        "train_ce": float(t_ce), "train_kl": float(t_fkl),
        "n_states": int(N), "n_val": int(n_val),
        "loss_type": loss_type, "uniform_mix": uniform_mix,
    }
    return params, metrics


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
def build_solver(config, network_type, layers, transformer_config,
                 history_encoding, factored_count_encoding, seed,
                 rnad_overrides=None):
    """Instantiate a JaxLiarsPokerRNaDSolver EXACTLY like train_jax builds it.

    ``transformer_config`` defaults to :data:`DEFAULT_TRANSFORMER_CONFIG` (the
    conf default) so the serialized state's top-level ``transformer_config`` key
    matches a default RNaD run.  ``rnad_overrides`` merges extra ``RNaDConfig``
    fields (e.g. ``warmup_steps``, which changes the OPTIMIZER-state structure --
    ``scale_by_schedule`` vs ``scale`` -- and must match the target run's, or
    ``restore_state`` fails on the optimizer tree).
    """
    rc = dict(
        game_name="liars_poker",
        policy_network_layers=tuple(layers),
        seed=seed,
    )
    if rnad_overrides:
        rc.update(rnad_overrides)
    return JaxLiarsPokerRNaDSolver(
        RNaDConfig(**rc),
        config,
        network_type=network_type,
        transformer_config=_effective_transformer_config(transformer_config),
        history_encoding=history_encoding,
        factored_count_encoding=factored_count_encoding,
    )


def run_pipeline(*, num_players, hand_length, num_digits, network_type,
                 layers, lam, history_encoding, transformer_config,
                 n_rollout_games, bc_epochs, bc_batch_size, bc_lr,
                 factored_count_encoding, first_bid_base_count, seed,
                 save_path, rollout_batch_size=4096, traj_max=None,
                 rnad_overrides=None, bc_loss_type="forward_ce",
                 target_uniform_mix=0.01, explore_eps=0.0, max_jump=None,
                 challenge_prob=None, bid_suffix_max=False, verbose=True):
    """Run the full BC pipeline and (optionally) save an RNaD checkpoint.

    ``transformer_config`` is resolved to the effective config the solver is built
    with (defaults to the conf default so the checkpoint restores into a default
    RNaD run).  ``rnad_overrides`` forwards structure-affecting ``RNaDConfig``
    fields (e.g. ``warmup_steps``) so the serialized optimizer tree matches the
    target run.  ``max_jump`` (when set) restricts the next bid's count to within
    ``max_jump`` of the current count (the action-space abstraction) for BOTH the
    heuristic target and the rollout -- ``legal_actions_mask`` already enforces it,
    so the heuristic normalizes over the restricted legal set and the exploration
    only samples restricted-legal actions.
    """
    config = LiarsPokerConfig(
        num_players=num_players, hand_length=hand_length, num_digits=num_digits,
        max_jump=max_jump,
    )
    effective_tc = _effective_transformer_config(transformer_config)

    # lpt capacity ("max game") sizing is NOT supported for BC (it changes the
    # action space so the played-game heuristic target no longer aligns with the
    # net's logits).  Assert we are on the non-capacity path.
    if network_type == "lpt":
        net_config, _gd = _lpt_net_config(config, effective_tc)
        if (net_config.num_digits != config.num_digits
                or net_config.hand_length != config.hand_length):
            raise ValueError(
                "BC does not support lpt net_capacity (capacity action space "
                "!= played-game action space).")

    max_seq = _lpt_max_seq(config, effective_tc)
    if traj_max is None:
        # A generous fixed padded game length: bids strictly increase (<= max_bids
        # placed) plus challenge/rebid slack.  Rare overflow just truncates a game
        # tail (still a valid dataset).
        traj_max = config.max_bids + config.num_players + 2

    if verbose:
        print(f"[1/4] Rollout collection: {n_rollout_games} games "
              f"(batch={rollout_batch_size}, traj_max={traj_max}, "
              f"{num_players}x{hand_length}x{num_digits}, lam={lam}, "
              f"explore_eps={explore_eps}) ...")
    obs, targets, legals = collect_dataset(
        config, network_type, history_encoding, max_seq,
        n_rollout_games=n_rollout_games, batch_size=rollout_batch_size,
        traj_max=traj_max, lam=lam, first_bid_base_count=first_bid_base_count,
        seed=seed, verbose=verbose, explore_eps=explore_eps,
        challenge_prob=challenge_prob, bid_suffix_max=bid_suffix_max)
    if verbose:
        print(f"      -> {targets.shape[0]} decision states")

    if verbose:
        print(f"[2/4] Building solver (network_type={network_type}) ...")
    solver = build_solver(config, network_type, layers, transformer_config,
                          history_encoding, factored_count_encoding, seed,
                          rnad_overrides=rnad_overrides)

    if verbose:
        print(f"[3/4] BC training ({bc_epochs} epochs, loss={bc_loss_type}, "
              f"uniform_mix={target_uniform_mix}) ...")
    bc_params, metrics = train_bc(
        solver.network, solver.params, obs, legals, targets, num_players,
        epochs=bc_epochs, batch_size=bc_batch_size, lr=bc_lr, seed=seed,
        loss_type=bc_loss_type, uniform_mix=target_uniform_mix,
        verbose=verbose)

    # Set ALL four param slots to the BC params (fresh RNaD warm-start).
    solver.params = bc_params
    solver.params_target = bc_params
    solver.params_prev = bc_params
    solver.params_prev_ = bc_params

    if save_path is not None:
        if verbose:
            print(f"[4/4] Saving checkpoint -> {save_path}")
        write_bytes(save_path, solver.state_bytes())

    return {
        "config": config,
        "solver": solver,
        "bc_params": bc_params,
        "metrics": metrics,
        "obs": obs,
        "targets": targets,
        "legals": legals,
        "save_path": save_path,
    }


# --------------------------------------------------------------------------- #
# Leaked-mass diagnostic (is net mass LOW on teacher-unlikely actions?)
# --------------------------------------------------------------------------- #
def _net_policy_chunked(network, params, obs, legals, num_players,
                        chunk=8192):
    """Net policy ``pi`` [N, A] (legal-masked, sums to 1) over the dataset."""
    N = legals.shape[0]
    out = []
    for i in range(0, N, chunk):
        o = _index_obs(obs, jnp.arange(i, min(i + chunk, N)))
        env = _make_env_step(o, legals[i:i + chunk], num_players)
        pi, _v, _lp, _lg = network.apply(params, env)
        out.append(np.asarray(pi))
    return np.concatenate(out, axis=0)


def run_diagnostic(*, checkpoint_path, num_players, hand_length, num_digits,
                   network_type, layers, lam, history_encoding,
                   transformer_config, factored_count_encoding,
                   first_bid_base_count, n_states, rollout_batch_size, seed,
                   rnad_overrides=None, eps_list=(0.01, 0.001),
                   teacher_floor=1e-9, explore_eps=0.0, max_jump=None,
                   traj_max=None, challenge_prob=None, bid_suffix_max=False,
                   verbose=True):
    """Measure how much BC-net mass LANDS ON teacher-unlikely (leaked) actions.

    Forward KL (the BC objective) is mode-COVERING and does not penalize net mass
    where teacher ~ 0.  This computes leaked mass, reverse KL (mode-seeking) and
    the worst-case over-weighted teacher-~0 action over a FRESH sample.
    ``explore_eps>0`` enriches the eval with the rare forced-decision states so
    the heavy tail (forced challenge / forced no-challenge) is measurable.
    """
    config = LiarsPokerConfig(
        num_players=num_players, hand_length=hand_length, num_digits=num_digits,
        max_jump=max_jump)
    effective_tc = _effective_transformer_config(transformer_config)
    max_seq = _lpt_max_seq(config, effective_tc)
    if traj_max is None:
        traj_max = config.max_bids + config.num_players + 2

    solver = build_solver(config, network_type, layers, transformer_config,
                          history_encoding, factored_count_encoding, seed,
                          rnad_overrides=rnad_overrides)
    solver.restore_state(read_bytes(checkpoint_path))

    if verbose:
        print(f"Collecting fresh sample (~{n_states} states, "
              f"explore_eps={explore_eps}) ...")
    obs, targets, legals, states = collect_dataset(
        config, network_type, history_encoding, max_seq,
        n_rollout_games=n_states, batch_size=rollout_batch_size,
        traj_max=traj_max, lam=lam, first_bid_base_count=first_bid_base_count,
        seed=seed + 991, verbose=verbose, return_states=True,
        explore_eps=explore_eps, challenge_prob=challenge_prob,
        bid_suffix_max=bid_suffix_max)

    net = _net_policy_chunked(solver.network, solver.params, obs, legals,
                              num_players)                       # [N, A]
    teacher = np.asarray(targets)                               # [N, A]
    legal = np.asarray(legals).astype(bool)                     # [N, A]
    N, A = teacher.shape

    def pct(x):
        return {q: float(np.percentile(x, q)) for q in (50, 90, 95, 99)}

    report = {"n_states": int(N)}

    # (1) Leaked mass per eps: total net prob on {teacher < eps}.
    report["leaked_mass"] = {}
    leaked_001 = None
    for eps in eps_list:
        near0 = teacher < eps                                   # incl. illegal
        lm = np.sum(net * near0, axis=1)                        # [N]
        report["leaked_mass"][eps] = {
            "mean": float(lm.mean()), "max": float(lm.max()), **pct(lm)}
        if abs(eps - 0.01) < 1e-12:
            leaked_001 = lm

    # (2) Reverse KL = sum net * log(net / teacher_floored), masked to legal.
    tfloor = np.where(legal, np.maximum(teacher, teacher_floor), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rterm = net * (np.log(net + 1e-45) - np.log(tfloor))
    rterm = np.where(legal & (net > 0), rterm, 0.0)
    rkl = rterm.sum(axis=1)                                     # [N]
    report["reverse_kl"] = {"mean": float(rkl.mean()),
                            "max": float(rkl.max()), **pct(rkl)}

    # (3) MAX net(a) over teacher-near-zero actions (eps=0.01), per state.
    near0_01 = teacher < 0.01
    net_on_near0 = np.where(near0_01, net, 0.0)
    max_net_near0 = net_on_near0.max(axis=1)                    # [N]
    report["max_net_on_teacher_near0"] = {
        "mean": float(max_net_near0.mean()),
        "max": float(max_net_near0.max()), **pct(max_net_near0)}

    # (4) Forward KL = sum teacher * log(teacher / net_floored) over legal.
    nfloor = np.where(legal, np.maximum(net, teacher_floor), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fterm = teacher * (np.log(np.where(teacher > 0, teacher, 1.0))
                           - np.log(nfloor))
    fterm = np.where(teacher > 0, fterm, 0.0)
    fkl = fterm.sum(axis=1)
    report["forward_kl"] = {"mean": float(fkl.mean()),
                            "max": float(fkl.max()), **pct(fkl)}

    # Entropy (over legal) of net vs teacher -- over-collapse check: a mode-seeking
    # net that collapsed onto one action would have net_H << teacher_H.
    def entropy(p):
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where((p > 0) & legal, p * np.log(p), 0.0).sum(axis=1)
        return h
    net_H, teacher_H = entropy(net), entropy(teacher)
    report["net_entropy"] = {"mean": float(net_H.mean()), **pct(net_H)}
    report["teacher_entropy"] = {"mean": float(teacher_H.mean()), **pct(teacher_H)}

    # (5) Fraction of states with leaked_mass(eps=0.01) over thresholds.
    report["frac_leaked_over"] = {
        thr: float(np.mean(leaked_001 > thr)) for thr in (0.02, 0.05, 0.1)}

    # (6) Worst-case: the (state, action) with teacher<0.001 that net most
    # over-weights.  Restrict to legal teacher-~0 actions.
    cand = np.where(legal & (teacher < 0.001), net, -1.0)       # [N, A]
    flat = int(np.argmax(cand))
    wi, wa = flat // A, flat % A
    D = num_digits
    st_i = jax.tree_util.tree_map(lambda x: x[wi], states)
    acting = int(st_i.current_player)
    hand_hist = [int(np.sum(np.asarray(st_i.hands)[acting] == d))
                 for d in range(1, D + 1)]
    has_bid = int(st_i.current_bid_action) >= 0
    if has_bid:
        cb = int(st_i.current_bid_action) - 1
        cur = (cb // D + 1, cb % D + 1)
    else:
        cur = None
    if wa == 0:
        act_desc = "CHALLENGE"
    else:
        act_desc = f"bid(count={(wa - 1) // D + 1}, digit={(wa - 1) % D + 1})"
    report["worst_case"] = {
        "state_idx": wi, "acting_player": acting,
        "acting_hand_histogram": hand_hist,
        "current_bid": cur, "action": act_desc, "action_id": wa,
        "teacher_prob": float(teacher[wi, wa]),
        "net_prob": float(net[wi, wa]),
    }

    # (7) FORCED-DECISION TAIL: the states the exploration coverage targets.
    #   * forced CHALLENGE: challenge legal AND teacher[challenge] > 0.99 (the
    #     current bid is (near) impossible given the actor's hand). Correct action
    #     = challenge; leak = net puts mass elsewhere -> net[challenge] should be
    #     high. Failure = net[challenge] < 0.5.
    #   * forced NO-CHALLENGE: challenge legal AND teacher[challenge] < 0.01 (the
    #     actor's own hand GUARANTEES the bid true). Correct = never challenge;
    #     failure = net[challenge] > 0.5 (net wrongly challenges).
    challenge_legal = legal[:, 0]
    net_ch = net[:, 0]
    teach_ch = teacher[:, 0]
    fc = challenge_legal & (teach_ch > 0.99)          # forced challenge
    fnc = challenge_legal & (teach_ch < 0.01)         # forced no-challenge

    def _fd_stats(mask, correct_is_challenge):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        p_ch = net_ch[mask]
        if correct_is_challenge:
            fail = float(np.mean(p_ch < 0.5))         # failed to challenge
            wrong_mass = float(np.mean(1.0 - p_ch))   # mass off the challenge
        else:
            fail = float(np.mean(p_ch > 0.5))         # wrongly challenged
            wrong_mass = float(np.mean(p_ch))         # mass on the challenge
        return {"n": n, "mean_net_challenge": float(p_ch.mean()),
                "frac_wrong_action": fail, "mean_wrong_mass": wrong_mass}

    report["forced_challenge"] = _fd_stats(fc, correct_is_challenge=True)
    report["forced_no_challenge"] = _fd_stats(fnc, correct_is_challenge=False)

    # Concrete worst forced-NO-challenge (e.g. holds-five-3s vs bid(5,3)): the
    # forced-true state where the net most wrongly challenges.
    def _fd_worst(mask, by_challenge_prob_high):
        if not mask.any():
            return None
        score = np.where(mask, net_ch if by_challenge_prob_high else -net_ch,
                         -1e9)
        j = int(np.argmax(score))
        s = jax.tree_util.tree_map(lambda x: x[j], states)
        ap = int(s.current_player)
        hh = [int(np.sum(np.asarray(s.hands)[ap] == d)) for d in range(1, D + 1)]
        hb = int(s.current_bid_action) >= 0
        cb2 = (int(s.current_bid_action) - 1) if hb else None
        curbid = (cb2 // D + 1, cb2 % D + 1) if hb else None
        return {"acting_hand_histogram": hh, "current_bid": curbid,
                "net_challenge_prob": float(net_ch[j]),
                "teacher_challenge_prob": float(teach_ch[j])}

    report["forced_no_challenge_worst"] = _fd_worst(fnc, True)   # net wrongly challenges
    report["forced_challenge_worst"] = _fd_worst(fc, False)      # net fails to challenge

    # (8) HIGH-JUMP-BID MASS: the crutch max_jump replaces.  jump(A) = count_A -
    # base_count (base = current bid's count, or first_bid_base_count at opening).
    # The heuristic downweights big jumps via exp(-lam*jump); the BC net should
    # inherit that -> LOW net mass on high-jump bids WITHOUT a hard restriction.
    cba = np.asarray(states.current_bid_action)                # [N]
    has_bid_v = cba >= 0
    cur_count_v = np.where(has_bid_v, (cba - 1) // D + 1, 0)
    base_count_v = np.where(has_bid_v, cur_count_v, first_bid_base_count)  # [N]
    a_ids = np.arange(A)
    count_a = (a_ids - 1) // D + 1                             # [A] (a=0 -> 0)
    is_bid = (a_ids != 0)[None, :]
    jump_a = count_a[None, :] - base_count_v[:, None]          # [N, A]
    report["high_jump"] = {}
    for J in (2, 3, 5):
        hj = legal & is_bid & (jump_a >= J)                    # [N, A]
        net_hj = np.sum(net * hj, axis=1)                      # [N]
        teach_hj = np.sum(teacher * hj, axis=1)                # [N]
        # fraction of states where such a high-jump bid is even legal.
        avail = float(np.mean(hj.any(axis=1)))
        report["high_jump"][J] = {
            "net_mean": float(net_hj.mean()), "net_p99": float(np.percentile(net_hj, 99)),
            "net_max": float(net_hj.max()),
            "teacher_mean": float(teach_hj.mean()),
            "teacher_p99": float(np.percentile(teach_hj, 99)),
            "frac_states_available": avail,
        }
    return report


def _fmt_dist(d):
    return (f"mean={d['mean']:.4f} p50={d[50]:.4f} p90={d[90]:.4f} "
            f"p95={d[95]:.4f} p99={d[99]:.4f} max={d['max']:.4f}")


def print_diagnostic(report):
    print("\n" + "=" * 70)
    print(f"LEAKED-MASS DIAGNOSTIC  (n_states={report['n_states']})")
    print("=" * 70)
    print("(4) FORWARD KL  KL(teacher||net) [BC objective, mode-covering]:")
    print("     " + _fmt_dist(report["forward_kl"]))
    print("(1) LEAKED MASS  sum net(a)*1{teacher(a)<eps}  [KEY METRIC]:")
    for eps, d in report["leaked_mass"].items():
        print(f"   eps={eps}:  " + _fmt_dist(d))
    print("(2) REVERSE KL  KL(net||teacher) [mode-seeking, leak-sensitive]:")
    print("     " + _fmt_dist(report["reverse_kl"]))
    print("(3) MAX net(a) on teacher-near-zero (<0.01) actions, per state:")
    print("     " + _fmt_dist(report["max_net_on_teacher_near0"]))
    ne, te = report["net_entropy"], report["teacher_entropy"]
    print("    ENTROPY (over-collapse check): "
          f"net mean={ne['mean']:.4f} (p50={ne[50]:.4f})  vs  "
          f"teacher mean={te['mean']:.4f} (p50={te[50]:.4f})")
    print("(5) FRACTION of states with leaked_mass(eps=0.01) > thr:")
    for thr, f in report["frac_leaked_over"].items():
        print(f"     > {thr}:  {f:.4f}")
    wc = report["worst_case"]
    print("(6) WORST-CASE over-weighted teacher-~0 action:")
    print(f"     acting hand histogram (d1..dD) = {wc['acting_hand_histogram']}")
    print(f"     current bid (count,digit) = {wc['current_bid']}")
    print(f"     action = {wc['action']}  ->  teacher={wc['teacher_prob']:.2e}"
          f"  net={wc['net_prob']:.4f}")
    print("(7) FORCED-DECISION TAIL (the exploration-targeted states):")
    fc, fnc = report["forced_challenge"], report["forced_no_challenge"]
    if fc.get("n", 0):
        print(f"   forced CHALLENGE (impossible bid): n={fc['n']}  "
              f"mean net[challenge]={fc['mean_net_challenge']:.4f}  "
              f"frac FAIL-to-challenge={fc['frac_wrong_action']:.4f}")
    else:
        print("   forced CHALLENGE: n=0 (none sampled)")
    if fnc.get("n", 0):
        print(f"   forced NO-challenge (certain bid): n={fnc['n']}  "
              f"mean net[challenge]={fnc['mean_net_challenge']:.4f}  "
              f"frac WRONGLY-challenge={fnc['frac_wrong_action']:.4f}")
    else:
        print("   forced NO-challenge: n=0 (none sampled)")
    fcw = report.get("forced_challenge_worst")
    fncw = report.get("forced_no_challenge_worst")
    if fcw:
        print(f"   worst forced-CHALLENGE: hand={fcw['acting_hand_histogram']} "
              f"bid={fcw['current_bid']} -> net[challenge]="
              f"{fcw['net_challenge_prob']:.4f} (teacher="
              f"{fcw['teacher_challenge_prob']:.4f})")
    if fncw:
        print(f"   worst forced-NO-challenge: hand={fncw['acting_hand_histogram']} "
              f"bid={fncw['current_bid']} -> net[challenge]="
              f"{fncw['net_challenge_prob']:.4f} (teacher="
              f"{fncw['teacher_challenge_prob']:.4f})")
    if "high_jump" in report:
        print("(8) HIGH-JUMP-BID MASS (the max_jump crutch we replace) "
              "-- net mass on bids with jump>=J:")
        for J, d in report["high_jump"].items():
            print(f"   jump>={J}: net mean={d['net_mean']:.4f} p99={d['net_p99']:.4f} "
                  f"max={d['net_max']:.4f}  |  teacher mean={d['teacher_mean']:.4f} "
                  f"p99={d['teacher_p99']:.4f}  (avail {d['frac_states_available']:.2f})")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Equilibrium-suppression diagnostic
#   Does the BC init place LOW mass on actions the CONVERGED from-scratch RNaD
#   agent plays with HIGH mass?  (Warm-start risk: suppressing equilibrium
#   actions so RNaD can't recover them.)  Evaluated ON THE EQUILIBRIUM's OWN
#   state distribution: NATURAL (multinomial) deal + self-play under the eq net.
# --------------------------------------------------------------------------- #
def collect_states_under_net(solver, config, network_type, history_encoding,
                             max_seq, *, n_games, batch_size, traj_max, seed):
    """Self-play rollout under ``solver``'s net (NATURAL deal); return states.

    The NATURAL per-card uniform deal (not the stars-and-bars composition used
    for BC training) reproduces the equilibrium agent's ACTUAL state
    distribution.  Returns a batched numpy ``LiarsPokerState`` of the valid
    (live-decision) states.
    """
    network = solver.network
    params = solver.params
    P, H, D = config.num_players, config.hand_length, config.num_digits

    @jax.jit
    def scan_batch(params, rng_key):
        hand_key, scan_key = jax.random.split(rng_key)
        hands0 = jax.random.randint(
            hand_key, (batch_size, P, H), 1, D + 1).astype(jnp.int32)
        states0 = initial_states_batched(config, hands0)

        def step(carry, _):
            states, rng = carry
            rng, ak = jax.random.split(rng)
            obs = _build_obs_batch(
                states, config, network_type, history_encoding, max_seq,
                lambda s: s.current_player, config)
            legal = jax.vmap(lambda s: legal_actions_mask(s, config))(states)
            valid = ~jax.vmap(is_terminal)(states)
            env = EnvStep(
                obs=obs, legal=legal.astype(jnp.int8),
                player_id=jnp.zeros(batch_size, jnp.float32),
                valid=jnp.ones(batch_size, jnp.float32),
                rewards=jnp.zeros((batch_size, P), jnp.float32))
            pi = network.apply(params, env)[0]
            g = jax.random.gumbel(ak, pi.shape, jnp.float32)
            actions = jnp.argmax(jnp.log(pi + 1e-45) + g, axis=-1).astype(jnp.int32)
            new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(
                states, actions)
            return (new_states, rng), (states, valid)

        _, (states, valid) = jax.lax.scan(
            step, (states0, scan_key), xs=None, length=traj_max)
        return states, valid

    n_batches = max(1, (n_games + batch_size - 1) // batch_size)
    key = jax.random.PRNGKey(seed)
    parts, total = [], 0
    for _ in range(n_batches):
        key, sub = jax.random.split(key)
        states, valid = scan_batch(params, sub)
        valid_flat = np.asarray(valid).reshape(-1)
        keep = np.where(valid_flat)[0]
        T, B = valid.shape
        st_flat = jax.tree_util.tree_map(
            lambda x: np.asarray(x).reshape((T * B,) + x.shape[2:])[keep], states)
        parts.append(st_flat)
        total += keep.shape[0]
    return jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, 0), *parts), total


def _decode_state_action(states, si, ai, D):
    """(hand histogram, current bid, action description) for one (state, action)."""
    s = jax.tree_util.tree_map(lambda x: x[si], states)
    ap = int(s.current_player)
    hh = [int(np.sum(np.asarray(s.hands)[ap] == d)) for d in range(1, D + 1)]
    hb = int(s.current_bid_action) >= 0
    cur = None
    if hb:
        cb = int(s.current_bid_action) - 1
        cur = (cb // D + 1, cb % D + 1)
    act = "CHALLENGE" if ai == 0 else \
        f"bid(count={(ai - 1) // D + 1},digit={(ai - 1) % D + 1})"
    return hh, cur, act


def equilibrium_suppression_report(eq_pi, bc_pi, legal, states, D,
                                   bc_floor=0.005, n_worst=6):
    """Metrics: does ``bc_pi`` suppress actions ``eq_pi`` plays with high mass?"""
    eq_pi = np.asarray(eq_pi); bc_pi = np.asarray(bc_pi)
    legal = np.asarray(legal).astype(bool)
    N, A = eq_pi.shape

    def pct(x):
        return {q: float(np.percentile(x, q)) for q in (50, 90, 95, 99)}

    report = {"n_states": int(N)}

    # (1) SUPPRESSED-ACTION MASS: eq mass on actions bc nearly zeroes (bc<floor)
    # that the equilibrium plays with mass > T.
    supp = bc_pi < bc_floor                               # [N, A]
    report["suppressed_mass"] = {}
    for T in (0.05, 0.1, 0.2):
        mask = supp & (eq_pi > T)                         # eq_pi>T => legal
        m = np.sum(eq_pi * mask, axis=1)                  # [N]
        report["suppressed_mass"][T] = {
            "mean": float(m.mean()), "max": float(m.max()),
            "frac_states_any": float(np.mean(mask.any(axis=1))), **pct(m)}

    # (1b) Split the suppressed (bc<floor, eq>0.1) instances by CHALLENGE vs BID:
    # eq wrongly challenging a certainly-true low bid is a dominated/off-path eq
    # imperfection (BC suppressing it is arguably CORRECT), whereas a suppressed
    # high-count BID is a genuine strategic (bluff) action the heuristic misses.
    sup_hi = supp & (eq_pi > 0.1)                          # [N, A]
    ch_mass = np.sum(eq_pi * sup_hi * (np.arange(A) == 0)[None, :], axis=1)
    bid_mass = np.sum(eq_pi * sup_hi * (np.arange(A) != 0)[None, :], axis=1)
    report["suppressed_split_eq0.1"] = {
        "challenge_mean": float(ch_mass.mean()),
        "bid_mean": float(bid_mass.mean()),
        "frac_states_challenge": float(np.mean(ch_mass > 0)),
        "frac_states_bid": float(np.mean(bid_mass > 0))}

    # (2) WORST GAP eq(a)-bc(a) over legal actions (eq plays it, bc suppresses).
    gap = np.where(legal, eq_pi - bc_pi, -np.inf)
    order = np.argsort(-gap.reshape(-1))[:n_worst]
    worst = []
    for idx in order:
        si, ai = int(idx) // A, int(idx) % A
        hh, cur, act = _decode_state_action(states, si, ai, D)
        worst.append({"hand_histogram": hh, "current_bid": cur, "action": act,
                      "eq_prob": float(eq_pi[si, ai]),
                      "bc_prob": float(bc_pi[si, ai])})
    report["worst_gap"] = worst

    # (3) bc mass on the eq's ARGMAX (favored) action per state.
    eq_arg = np.argmax(np.where(legal, eq_pi, -1.0), axis=1)
    bc_on = bc_pi[np.arange(N), eq_arg]
    eq_on = eq_pi[np.arange(N), eq_arg]
    report["bc_on_eq_argmax"] = {
        "mean": float(bc_on.mean()), **pct(bc_on),
        "frac_bc_lt_0.005": float(np.mean(bc_on < 0.005)),
        "frac_bc_lt_0.05": float(np.mean(bc_on < 0.05)),
        "eq_argmax_mean": float(eq_on.mean())}

    # (5) COVERAGE: fraction of eq mass on actions where bc(a) > 0.01.
    cover = np.sum(eq_pi * (bc_pi > 0.01), axis=1)
    report["eq_support_coverage"] = {"mean": float(cover.mean()), **pct(cover)}
    return report


def print_eq_suppression(name, report):
    def fd(d):
        return (f"mean={d['mean']:.4f} p90={d[90]:.4f} p95={d[95]:.4f} "
                f"p99={d[99]:.4f} max={d.get('max', float('nan')):.4f}")
    print("\n" + "=" * 74)
    print(f"EQUILIBRIUM-SUPPRESSION: {name}  (n_states={report['n_states']})")
    print("=" * 74)
    print("(1) SUPPRESSED-ACTION MASS: eq mass on actions bc<0.005 & eq>T:")
    for T, d in report["suppressed_mass"].items():
        print(f"   eq>{T}: mean={d['mean']:.5f} p99={d[99]:.5f} max={d['max']:.4f} "
              f" frac states w/ any = {d['frac_states_any']:.4f}")
    sp = report["suppressed_split_eq0.1"]
    print("(1b) suppressed (bc<0.005, eq>0.1) mass split -- CHALLENGE (often "
          "dominated/off-path eq) vs BID (genuine strategic):")
    print(f"     challenge: mean={sp['challenge_mean']:.5f} "
          f"(states {sp['frac_states_challenge']:.4f})   "
          f"bid: mean={sp['bid_mean']:.5f} (states {sp['frac_states_bid']:.4f})")
    b = report["bc_on_eq_argmax"]
    print("(3) bc mass on the eq's FAVORED (argmax) action per state:")
    print(f"     bc mean={b['mean']:.4f} p50={b[50]:.4f} p90={b[90]:.4f} "
          f"(eq argmax mean={b['eq_argmax_mean']:.4f})")
    print(f"     frac states bc<0.005 = {b['frac_bc_lt_0.005']:.4f}   "
          f"bc<0.05 = {b['frac_bc_lt_0.05']:.4f}")
    c = report["eq_support_coverage"]
    print(f"(5) EQ-SUPPORT COVERAGE (eq mass on bc>0.01 actions): mean={c['mean']:.4f} "
          f"p50={c[50]:.4f} p90={c[90]:.4f}")
    print("(2) WORST eq-minus-bc gaps (eq plays, bc suppresses):")
    for w in report["worst_gap"]:
        print(f"     hand={w['hand_histogram']} bid={w['current_bid']} "
              f"{w['action']}: eq={w['eq_prob']:.4f} bc={w['bc_prob']:.4f} "
              f"(gap {w['eq_prob'] - w['bc_prob']:.4f})")
    print("=" * 74)


def run_eq_suppression(*, eq_checkpoint, bc_checkpoints, num_players, hand_length,
                       num_digits, network_type, layers, history_encoding,
                       factored_count_encoding, n_states, rollout_batch_size,
                       seed, rnad_overrides=None, transformer_config=None,
                       eq_max_jump=None, bc_max_jump=None, traj_max=None,
                       verbose=True):
    """Compare each BC init against the converged (eq) agent on eq states.

    ``eq_max_jump`` / ``bc_max_jump`` are the RESTORE configs (they must match
    each checkpoint's stored jax_config -- a maxjump agent stores max_jump, a
    full-space BC does not, so their jax_config keys differ and restore needs the
    right one).  The network params + obs are max_jump-INDEPENDENT (same tree);
    only the legal mask differs.  The PRIMARY comparison uses the EQ's (possibly
    restricted) legal mask for BOTH eq and bc (way 1: bc renormalized over the
    same action set the eq plays on).  When the two max_jumps differ we ALSO
    report way 2 (bc scored over ITS OWN full legal set) to check it doesn't
    change the conclusion.
    """
    eq_config = LiarsPokerConfig(
        num_players=num_players, hand_length=hand_length, num_digits=num_digits,
        max_jump=eq_max_jump)
    bc_config = LiarsPokerConfig(
        num_players=num_players, hand_length=hand_length, num_digits=num_digits,
        max_jump=bc_max_jump)
    effective_tc = _effective_transformer_config(transformer_config)
    max_seq = _lpt_max_seq(eq_config, effective_tc)
    if traj_max is None:
        traj_max = eq_config.max_bids + eq_config.num_players + 2

    def _load(cfg, path):
        s = build_solver(cfg, network_type, layers, transformer_config,
                         history_encoding, factored_count_encoding, seed,
                         rnad_overrides=rnad_overrides)
        s.restore_state(read_bytes(path))
        return s

    if verbose:
        print(f"Loading equilibrium agent (max_jump={eq_max_jump}): {eq_checkpoint}")
    eq = _load(eq_config, eq_checkpoint)

    if verbose:
        print(f"Rolling out ~{n_states} eq-self-play states (natural deal, "
              f"eq's restricted mask) ...")
    states, n = collect_states_under_net(
        eq, eq_config, network_type, history_encoding, max_seq,
        n_games=n_states, batch_size=rollout_batch_size, traj_max=traj_max,
        seed=seed + 4242)
    if verbose:
        print(f"  -> {n} decision states")

    states_j = jax.tree_util.tree_map(jnp.asarray, states)
    obs = _build_obs_batch(states_j, eq_config, network_type, history_encoding,
                           max_seq, lambda s: s.current_player, eq_config)
    # EQ's (restricted) legal mask = the comparison action set (way 1).
    eq_legals = np.asarray(
        jax.vmap(lambda s: legal_actions_mask(s, eq_config))(states_j))
    # BC's own (full) legal mask, for the way-2 cross-check.
    bc_full_legals = np.asarray(
        jax.vmap(lambda s: legal_actions_mask(s, bc_config))(states_j))
    different_masks = not np.array_equal(eq_legals, bc_full_legals)
    eq_pi = _net_policy_chunked(eq.network, eq.params, obs, eq_legals, num_players)

    reports = {}
    for path in bc_checkpoints:
        if verbose:
            print(f"Scoring BC init: {path}")
        bc = _load(bc_config, path)
        # Way 1 (PRIMARY): bc over the eq's restricted action set.
        bc_pi = _net_policy_chunked(bc.network, bc.params, obs, eq_legals,
                                    num_players)
        rep = equilibrium_suppression_report(eq_pi, bc_pi, eq_legals, states,
                                             num_digits)
        reports[path] = rep
        print_eq_suppression(path.split("/")[-2], rep)
        # Way 2 (cross-check): bc over ITS OWN full legal set (mass may leak to
        # high-jump actions the eq can't play -> slightly lower low-jump mass).
        if different_masks:
            bc_pi_full = _net_policy_chunked(bc.network, bc.params, obs,
                                             bc_full_legals, num_players)
            supp = (bc_pi_full < 0.005) & (eq_pi > 0.1)
            sm = float(np.mean(np.sum(eq_pi * supp, axis=1)))
            cover = float(np.mean(np.sum(eq_pi * (bc_pi_full > 0.01), axis=1)))
            leaked_hi = float(np.mean(np.sum(
                bc_pi_full * (bc_full_legals.astype(bool) & ~eq_legals.astype(bool)),
                axis=1)))
            print(f"   [way 2: bc over FULL mask] suppressed(eq>0.1) mean={sm:.5f}"
                  f"  eq-support coverage={cover:.4f}  "
                  f"bc mass on eq-illegal (high-jump) actions={leaked_hi:.4f}")
    return reports


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-type", default="factored_mlp",
                        choices=["factored_mlp", "mlp", "lpt"])
    # Defaults match the project "5x5x2" convention: num_digits=5, hand_length=5,
    # num_players=2 (a 2-player, 5-digit game), and the winning bigMLP layers.
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--hand-length", type=int, default=5)
    parser.add_argument("--num-digits", type=int, default=5)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[512, 512, 512, 256, 256])
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--challenge-prob", type=float, default=None,
                        help="Fixed P(challenge) constant when challenge is legal "
                             "(default None = belief-based 1-belief(current bid)). "
                             "e.g. 0.1 puts a constant 0.1 on challenge, 0.9 on "
                             "the belief*exp(-lam*jump) bid distribution.")
    parser.add_argument("--bid-suffix-max", action="store_true",
                        help="Apply the suffix-max bluff-safety transform to the "
                             "teacher's bid distribution: each legal bid inherits "
                             "the best higher-or-equal bid's weight (monotone "
                             "non-increasing along (count,digit)). Challenge "
                             "untouched. Raises mass on low/high-jump bids that "
                             "escalate to a strong higher bid.")
    parser.add_argument("--first-bid-base-count", type=int, default=1)
    parser.add_argument("--max-jump", type=int, default=None,
                        help="Action-space abstraction: restrict the next bid's "
                             "count to within max_jump of the current (base) "
                             "count, for both the heuristic and the rollout. "
                             "None = unrestricted (byte-identical to the full "
                             "game). MUST match the target RNaD run's max_jump.")
    parser.add_argument("--history-encoding", default="sparse",
                        choices=["sparse", "compact"])
    parser.add_argument("--factored-count-encoding", default="linear",
                        choices=["linear", "tailprob"])
    parser.add_argument("--transformer-config", default=None,
                        help="JSON string for the lpt transformer_config.")
    parser.add_argument("--n-rollout-games", type=int, default=16384)
    parser.add_argument("--rollout-batch-size", type=int, default=4096)
    parser.add_argument("--traj-max", type=int, default=None,
                        help="Fixed padded rollout length (games truncated past "
                             "it). Default = max_bids+P+2. Set to the target RNaD "
                             "run's trajectory_max (e.g. 100 for the 8x10 mj1 "
                             "baseline) to match its state distribution + speed.")
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-batch-size", type=int, default=4096)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-loss", default="forward_ce",
                        choices=["forward_ce", "reverse_kl"],
                        help="forward_ce = mode-covering CE (default); reverse_kl "
                             "= mode-seeking KL(net||smoothed_teacher), drives net "
                             "mass OFF teacher-unlikely actions.")
    parser.add_argument("--target-uniform-mix", type=float, default=0.01,
                        help="Uniform floor mixed into the teacher for reverse_kl "
                             "(bounds log teacher', avoids revKL blowups).")
    parser.add_argument("--explore-eps", type=float, default=0.0,
                        help="Per-step prob of a UNIFORM-RANDOM legal action in "
                             "the rollout (coverage of forced-decision states); "
                             "those steps are masked from the BC loss.")
    parser.add_argument("--diag-explore-eps", type=float, default=0.0,
                        help="Exploration eps for the DIAGNOSTIC eval sample "
                             "(enriches the forced-decision tail so it is "
                             "measurable). Use the same value for both ckpts.")
    parser.add_argument("--seed", type=int, default=42)
    # MUST match the target RNaD run's warmup_steps: it changes the serialized
    # optimizer-state structure (scale_by_schedule vs scale), so a mismatch makes
    # restore_state fail on the optimizer tree.  The factored-linear-5x5x2 run
    # uses 2000.
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--rnad-overrides", default=None,
                        help="JSON string of extra RNaDConfig fields to match the "
                             "target run (merged over warmup_steps).")
    parser.add_argument("--save-path", default=None,
                        help="Local path or s3:// URI for the checkpoint.")
    parser.add_argument("--mode", default="train",
                        choices=["train", "diagnostic", "eq_suppression"],
                        help="'train' = BC-pretrain + save; 'diagnostic' = leaked "
                             "mass on teacher-~0 actions (needs --checkpoint); "
                             "'eq_suppression' = does a BC init suppress actions a "
                             "converged agent plays (needs --eq-checkpoint + "
                             "--bc-checkpoints).")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to diagnose (local or s3://).")
    parser.add_argument("--eq-checkpoint", default=None,
                        help="Converged (equilibrium proxy) checkpoint for "
                             "--mode eq_suppression.")
    parser.add_argument("--bc-checkpoints", default=None,
                        help="Comma-separated BC-init checkpoints to score "
                             "against the eq agent (--mode eq_suppression).")
    parser.add_argument("--eq-max-jump", type=int, default=None,
                        help="max_jump of the EQ checkpoint (eq_suppression); "
                             "must match its stored jax_config (e.g. 1 for a "
                             "maxjump agent). Defines the comparison legal set.")
    parser.add_argument("--bc-max-jump", type=int, default=None,
                        help="max_jump of the BC checkpoints (eq_suppression); "
                             "None for a full-space BC init.")
    parser.add_argument("--n-diag-states", type=int, default=100000)
    args = parser.parse_args()

    transformer_config = (
        json.loads(args.transformer_config) if args.transformer_config else None
    )
    rnad_overrides = {"warmup_steps": args.warmup_steps}
    if args.rnad_overrides:
        rnad_overrides.update(json.loads(args.rnad_overrides))

    if args.mode == "diagnostic":
        if not args.checkpoint:
            parser.error("--mode diagnostic requires --checkpoint")
        report = run_diagnostic(
            checkpoint_path=args.checkpoint, num_players=args.num_players,
            hand_length=args.hand_length, num_digits=args.num_digits,
            network_type=args.network_type, layers=args.layers, lam=args.lam,
            history_encoding=args.history_encoding,
            transformer_config=transformer_config,
            factored_count_encoding=args.factored_count_encoding,
            first_bid_base_count=args.first_bid_base_count,
            n_states=args.n_diag_states,
            rollout_batch_size=args.rollout_batch_size, seed=args.seed,
            rnad_overrides=rnad_overrides, explore_eps=args.diag_explore_eps,
            max_jump=args.max_jump, traj_max=args.traj_max,
            challenge_prob=args.challenge_prob,
            bid_suffix_max=args.bid_suffix_max)
        print_diagnostic(report)
        return

    if args.mode == "eq_suppression":
        if not args.eq_checkpoint or not args.bc_checkpoints:
            parser.error("--mode eq_suppression requires --eq-checkpoint and "
                         "--bc-checkpoints")
        run_eq_suppression(
            eq_checkpoint=args.eq_checkpoint,
            bc_checkpoints=[p.strip() for p in args.bc_checkpoints.split(",")],
            num_players=args.num_players, hand_length=args.hand_length,
            num_digits=args.num_digits, network_type=args.network_type,
            layers=args.layers, history_encoding=args.history_encoding,
            factored_count_encoding=args.factored_count_encoding,
            n_states=args.n_diag_states,
            rollout_batch_size=args.rollout_batch_size, seed=args.seed,
            rnad_overrides=rnad_overrides, transformer_config=transformer_config,
            eq_max_jump=args.eq_max_jump, bc_max_jump=args.bc_max_jump,
            traj_max=args.traj_max)
        return

    out = run_pipeline(
        num_players=args.num_players, hand_length=args.hand_length,
        num_digits=args.num_digits, network_type=args.network_type,
        layers=args.layers, lam=args.lam,
        history_encoding=args.history_encoding,
        transformer_config=transformer_config,
        n_rollout_games=args.n_rollout_games, bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size, bc_lr=args.bc_lr,
        factored_count_encoding=args.factored_count_encoding,
        first_bid_base_count=args.first_bid_base_count, seed=args.seed,
        save_path=args.save_path, rollout_batch_size=args.rollout_batch_size,
        traj_max=args.traj_max,
        rnad_overrides=rnad_overrides, bc_loss_type=args.bc_loss,
        target_uniform_mix=args.target_uniform_mix,
        explore_eps=args.explore_eps, max_jump=args.max_jump,
        challenge_prob=args.challenge_prob,
        bid_suffix_max=args.bid_suffix_max,
    )
    m = out["metrics"]
    print("\nBC done.")
    print(f"  states={m['n_states']}  loss={m['loss_type']}  "
          f"val CE={m['val_ce']:.4f}  fwdKL={m['val_kl']:.4f}  "
          f"revKL={m['val_reverse_kl']:.4f}  netH={m['val_net_entropy']:.4f}")
    if out["save_path"]:
        print(f"  checkpoint: {out['save_path']}")


if __name__ == "__main__":
    main()
