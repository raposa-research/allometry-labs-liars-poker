#!/usr/bin/env python
"""Train RNaD on Liar's Poker using a fully JAX-based trajectory rollout,
interleaved with best-response (PPO) training.

Loop structure
--------------
For each phase:
  1. Run rnad_steps_per_phase RNaD gradient steps (GPU-accelerated JAX rollout).
  2. Run br_collect_per_phase PPO collect+update cycles, one per player position,
     with the BR agent playing against a frozen copy of the current RNaD policy.
  3. Every eval_every_phases phases: evaluate BR vs main agent and log metrics.

Metrics logged to wandb
-----------------------
  rnad/loss           — RNaD training loss (every learner step)
  br/reward_p{i}      — mean reward of BR agent as player i vs frozen main
  br/agent_score      — mean reward of main agent across all BR configurations
                        (lower = more exploitable by trained BR)
"""

import collections
import contextlib
import dataclasses
import functools
import io
import os
import re
from typing import Any, Optional

try:
    import nvtx
    _nvtx_available = True
except ImportError:
    _nvtx_available = False


@contextlib.contextmanager
def nvtx_range(name: str, color: str = "blue"):
    """Context manager for NVTX range annotations (no-op if nvtx not installed)."""
    if _nvtx_available:
        nvtx.push_range(name, color=color)
    try:
        yield
    finally:
        if _nvtx_available:
            nvtx.pop_range()

import chex
import flax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
import wandb

from rnad import (
    RNaDSolver,
    RNaDConfig,
    NerdConfig,
    RNaDNetwork,
    FactoredMLPNetwork,
    TimeStep as RNaDTimeStep,
    EnvStep,
    ActorStep,
    EntropySchedule,
    optax_optimizer,
    lr_scale_transform,
    FineTuning,
    _legal_policy,
    legal_log_policy,
)
from transformer_network import TransformerPolicyNetwork
from transformer_seq import LPTransformer, LPInputs
from liars_poker_jax import (
    LiarsPokerConfig,
    new_initial_state,
    apply_action,
    is_terminal,
    current_player_id,
    legal_actions_mask,
    observation_tensor,
    observation_size,
    structured_observation,
    cap_to_game_action_map,
    decode_bid,
    returns as jax_returns,
)

# ---------------------------------------------------------------------------
# S3 helpers (boto3 imported lazily; respects AWS_ENDPOINT_URL for R2)
# ---------------------------------------------------------------------------


def _is_s3(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _s3_client():
    import boto3
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client("s3", endpoint_url=endpoint_url)


def _parse_s3(path: str) -> tuple:
    """Return (bucket, key) from an s3://bucket/key path."""
    without_scheme = path[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def _join_path(base: str, name: str) -> str:
    """Join a directory/prefix with a filename for local or s3:// paths."""
    if _is_s3(base):
        return base.rstrip("/") + "/" + name
    return os.path.join(base, name)


def read_bytes(path: str) -> bytes:
    """Read a file from local disk or S3."""
    if _is_s3(path):
        bucket, key = _parse_s3(path)
        buf = io.BytesIO()
        _s3_client().download_fileobj(bucket, key, buf)
        return buf.getvalue()
    with open(path, "rb") as f:
        return f.read()


def write_bytes(path: str, data: bytes) -> None:
    """Write bytes to local disk or S3."""
    if _is_s3(path):
        bucket, key = _parse_s3(path)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


# ---------------------------------------------------------------------------
# Network factory
# ---------------------------------------------------------------------------


class LPTNetwork(nn.Module):
    """Adapts the standalone LPTransformer to the RNaDNetwork contract.

    ``__call__`` takes an :class:`EnvStep` whose ``.obs`` field holds a SINGLE
    (unbatched) :class:`LPInputs` (as produced by ``structured_observation``).
    It runs the FULL-PARALLEL forward (STAGE 1 reference: no KV-cache / no
    causal batching) and returns ``(pi, v, log_pi, logit)`` exactly like
    :class:`RNaDNetwork` / :class:`FactoredMLPNetwork`, so every existing call
    site and R-NaD / PPO loss works unchanged.
    """
    d_model: int
    n_layers: int
    n_heads: int
    num_digits: int
    max_count: int
    num_players: int
    total_cards: int
    num_actions: int
    max_seq: int
    number_encoding: str = "categorical"  # digit-axis: categorical|relative_bias
    digit_rope_origin: str = "max"  # rope_digit only: 'max' (from-max) | 'zero'
    head_digit_equivariant: bool = False  # param-free digit head (transfer)
    game_num_digits: int = 0  # played-game D (capacity mode); 0 => = num_digits
    head_mode: str = "legacy"  # 'legacy' | 'perdigit_tailprob' (per-digit head)
    game_hand_length: int = 0  # played-game hand_length (count/belief pools)
    dtype: Any = jnp.float32  # trunk compute dtype (bf16 mixed precision).
    remat: Optional[bool] = None  # None => remat on CPU only; True/False override.

    @nn.compact
    def __call__(self, env_step: EnvStep):
        core = LPTransformer(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            num_digits=self.num_digits,
            max_count=self.max_count,
            num_players=self.num_players,
            total_cards=self.total_cards,
            num_actions=self.num_actions,
            max_seq=self.max_seq,
            number_encoding=self.number_encoding,
            digit_rope_origin=self.digit_rope_origin,
            head_digit_equivariant=self.head_digit_equivariant,
            game_num_digits=self.game_num_digits,
            head_mode=self.head_mode,
            game_hand_length=self.game_hand_length,
            dtype=self.dtype,
            remat=self.remat,
        )
        # env_step.obs is an LPInputs whose leaves may be UNBATCHED (leading dims
        # (), as at single-vmap collect sites) or carry leading dims (e.g. (B,)
        # when the R-NaD loss vmaps only over the T axis).  Flatten any leading
        # dims into one batch axis, run the batched full-forward, then reshape
        # back -- this makes the adapter behave like RNaDNetwork under either
        # calling convention.
        inputs = env_step.obs
        lead = inputs.hand_hist.shape[:-1]  # () or (B,) -- static under jit
        bflat = 1
        for s in lead:
            bflat *= s
        flat = LPInputs(
            hand_hist=inputs.hand_hist.reshape(bflat, self.num_digits),
            bids=inputs.bids.reshape(bflat, self.max_seq, 3),
            bid_valid=inputs.bid_valid.reshape(bflat, self.max_seq),
            observer=inputs.observer.reshape(bflat),
            globals_=inputs.globals_.reshape(bflat, -1),
            legal_mask=inputs.legal_mask.reshape(bflat, -1),
        )
        logits, value = core(flat, decode=False)  # [bflat, A], [bflat, 1]
        logit = logits.reshape(lead + (self.num_actions,))
        v = value.reshape(lead + (1,))
        legal = env_step.legal  # same leading dims as logit
        pi = _legal_policy(logit, legal)
        log_pi = legal_log_policy(logit, legal)
        return pi, v, log_pi, logit


def _lpt_config(transformer_config):
    """Hyperparameters for the LPTransformer (network_type='lpt').

    Read from a DEDICATED nested ``transformer_config['lpt']`` dict rather than
    the flat ``transformer_config`` keys, because the flat keys configure the
    *other* ``transformer`` architecture (its default d_model=128 both has a
    different meaning here and triggers an XLA CPU compile limit when fused with
    long trajectories).  Absent an explicit ``lpt`` section, lpt uses its own
    reasonable defaults (d_model=64, n_layers=3, n_heads=4).
    """
    tc = transformer_config or {}
    return tc.get("lpt", {}) or {}


def _lpt_net_config(jax_config, transformer_config):
    """Return ``(net_config, game_num_digits)`` for the LPTransformer.

    ``net_config`` is the CAPACITY ("max game") ``LiarsPokerConfig`` the network
    is SIZED for (hand length, num_digits, num_actions, count normalization),
    while the PLAYED game stays ``jax_config``.  Configured via
    ``transformer_config['lpt']['net_capacity'] = {num_digits, hand_length}``
    (default: the 8x10 max).  When unset -- or equal to the played game --
    ``net_config is jax_config`` and ``game_num_digits == jax_config.num_digits``,
    so every derived path is byte-identical (no capacity).

    Building the net at capacity makes the params + I/O FIXED-SIZE across games
    (num_actions=161 for 8x10, count normalization /16, hand_hist length 10), so
    a checkpoint trained on a smaller game (6x6) loads with NO shape mismatch
    into a larger game's net (6x8) for curriculum warm-start.  Because the
    rope_digit + digit-equivariant head make the digit path parameter-free, the
    trainable params are already num_digits-independent; capacity fixes the
    remaining STATIC sizing + count normalization so transfer is seamless.
    """
    lc = _lpt_config(transformer_config)
    cap = lc.get("net_capacity", None)
    if not cap:
        return jax_config, jax_config.num_digits
    Dcap = int(cap.get("num_digits", jax_config.num_digits))
    Hcap = int(cap.get("hand_length", jax_config.hand_length))
    if Dcap < jax_config.num_digits or Hcap < jax_config.hand_length:
        raise ValueError(
            f"net_capacity (num_digits={Dcap}, hand_length={Hcap}) must be >= "
            f"the played game (num_digits={jax_config.num_digits}, "
            f"hand_length={jax_config.hand_length})."
        )
    if Dcap == jax_config.num_digits and Hcap == jax_config.hand_length:
        return jax_config, jax_config.num_digits
    net_config = LiarsPokerConfig(
        num_players=jax_config.num_players,
        hand_length=Hcap,
        num_digits=Dcap,
    )
    return net_config, jax_config.num_digits


_DTYPE_MAP = {
    "bfloat16": jnp.bfloat16,
    "bf16": jnp.bfloat16,
    "float32": jnp.float32,
    "fp32": jnp.float32,
    "float16": jnp.float16,
    "fp16": jnp.float16,
}


def _lpt_dtype(transformer_config):
    """Trunk compute dtype for the LPTransformer (network_type='lpt').

    Configurable via ``transformer_config['lpt']['dtype']`` (a string). Defaults
    to ``bfloat16``: the lpt trunk runs its matmuls in bf16 (params stay fp32),
    roughly halving activation memory and speeding up the forward/backward, while
    the head logits/value are cast back to fp32 for the R-NaD loss.
    """
    lc = _lpt_config(transformer_config)
    name = str(lc.get("dtype", "bfloat16")).lower()
    if name not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown lpt dtype {name!r}; expected one of {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[name]


def _lpt_max_seq(jax_config, transformer_config):
    """Static bid-sequence length for the LPTransformer.

    Configurable via ``transformer_config['lpt']['max_seq']``. Defaults to a
    realistic bound of ``min(max_bids, 24)`` rather than the (loose) tight upper
    bound ``max_bids``: this shrinks the per-token sequence (and thus every
    transformer forward + the KV cache) with no correctness loss, because real
    games rarely place many bids (at 6x6 they seldom exceed ~15, and max_bids is
    a much larger combinatorial cap). If a game DOES exceed ``max_seq``, the
    OLDEST bids are dropped (see ``structured_observation``); pick a value large
    enough that this effectively never happens for your game size.
    """
    lc = _lpt_config(transformer_config)
    net_config, _ = _lpt_net_config(jax_config, transformer_config)
    return int(lc.get("max_seq", min(net_config.max_bids, 24)))


def _dummy_network_input(
    jax_config, network_type, history_encoding, max_seq, net_config=None
):
    """A single (unbatched) network input used only for parameter init.

    ``net_config`` (defaults to ``jax_config``) sizes the lpt input to the
    CAPACITY dims (hand_hist length + legal_mask width) so init produces the
    fixed-size param tree.
    """
    if net_config is None:
        net_config = jax_config
    if network_type == "lpt":
        return LPInputs(
            hand_hist=jnp.zeros(net_config.num_digits, dtype=jnp.int32),
            bids=jnp.zeros((max_seq, 3), dtype=jnp.int32),
            bid_valid=jnp.zeros(max_seq, dtype=jnp.bool_),
            observer=jnp.int32(0),
            globals_=jnp.zeros(4, dtype=jnp.float32),
            legal_mask=jnp.ones(net_config.num_actions, dtype=jnp.bool_),
        )
    return jnp.zeros(
        observation_size(jax_config, history_encoding=history_encoding)
    )


def _build_obs_batch(
    states, jax_config, network_type, history_encoding, max_seq, observer_fn,
    net_config=None,
):
    """Batched network inputs for `states` (vmapped over the leading batch dim).

    Returns a flat ``[B, obs_dim]`` array for mlp/factored_mlp/transformer, or a
    batched :class:`LPInputs` pytree (leaves have leading dim B) for ``lpt``.
    ``observer_fn(state) -> player id`` selects the observing player per state.
    ``net_config`` (defaults to ``jax_config``) sizes the lpt obs to CAPACITY
    (padded hand_hist + capacity-space legal_mask); the game logic uses
    ``jax_config``.
    """
    if network_type == "lpt":
        return jax.vmap(
            lambda s: structured_observation(
                s, jax_config, observer_fn(s), max_seq, net_config
            )
        )(states)
    return jax.vmap(
        lambda s: observation_tensor(
            s, jax_config, observer_fn(s), history_encoding=history_encoding
        )
    )(states)


def _make_lpt_core(
    jax_config, transformer_config, max_seq, net_config=None,
    game_num_digits=0,
):
    """Build the standalone ``LPTransformer`` with the SAME hyperparameters that
    ``LPTNetwork`` instantiates internally.

    Used by the Stage-3 KV-cache rollout, which drives the inner transformer's
    ``prefill_fixed`` / ``append_bid_fixed`` / ``decision_logits_fixed`` methods
    DIRECTLY (via sliced sub-params ``params['params']['LPTransformer_0']``)
    rather than through the ``LPTNetwork`` full-forward wrapper. Keeping the
    config in one place guarantees the standalone core and the ``LPTransformer_0``
    submodule share an identical parameter tree, so no re-init is needed.
    """
    lc = _lpt_config(transformer_config)
    if net_config is None:
        net_config = jax_config
    return LPTransformer(
        d_model=lc.get("d_model", 64),
        n_layers=lc.get("n_layers", 3),
        n_heads=lc.get("n_heads", 4),
        num_digits=net_config.num_digits,
        max_count=net_config.total_cards,
        num_players=net_config.num_players,
        total_cards=net_config.total_cards,
        num_actions=net_config.num_actions,
        max_seq=max_seq,
        number_encoding=str(lc.get("number_encoding", "categorical")),
        digit_rope_origin=str(lc.get("digit_rope_origin", "max")),
        head_digit_equivariant=bool(lc.get("head_digit_equivariant", False)),
        game_num_digits=game_num_digits,
        head_mode=str(lc.get("head_mode", "legacy")),
        game_hand_length=jax_config.hand_length,
        dtype=_lpt_dtype(transformer_config),
        remat=lc.get("remat", None),
    )


# ----------------------------------------------------------------------------
# P-observer-folded KV-cache helpers for the Stage-3 lpt rollout.
#
# The fixed-shape cache API (`prefill_fixed` / `append_bid_fixed` /
# `decision_logits_fixed`) is per-row: each cache row is one observing player's
# incremental KV state.  To hold ALL ``P = num_players`` observer perspectives
# with a SINGLE call to each method (instead of P separate calls), we fold the
# player axis into the leading batch dim as ``row index = p * B + b`` -- every
# per-row leaf in the cache then has leading dim ``P*B``.  The fold ordering is
# the invariant that keeps everything aligned:
#   * ``hands`` stacked over p -> [P, B, D]; ``hands.reshape(P*B, D)`` puts
#     observer p's env-b hand at row ``p*B + b``.
#   * ``observer_fold = repeat(arange(P), B)`` has value p at row ``p*B + b``.
#   * decision gather ``idx = current_player*B + arange(B)`` selects row
#     ``cp*B + b`` = the acting player's own observer cache for env b.
#   * append tiling ``tile(x, (P,1))`` / ``tile(x, P)`` puts x[b] at row
#     ``p*B + b`` for every p (the same public bid is written to all observers).
# At P=2 this is numerically identical to the old two-cache ``jnp.where`` select
# (the gather picks the exact same leaf rows; per-row ops are independent).
# Kept module-level (not methods) so other rollout paths (e.g. the BR agent) can
# reuse them later.
def _lpt_cache_prefill(core, core_params, batch_states, config, net_config,
                       max_seq, num_players):
    """Prefill ALL P observer perspectives into one [P*B]-row fixed cache.

    Returns ``(cache, hands, observer_fold)`` where ``hands`` is [P, B, D] (the
    per-observer hand histograms, constant post-deal) and ``observer_fold`` is
    [P*B] (value p at row p*B+b).
    """
    hands = jnp.stack([
        jax.vmap(
            lambda s: structured_observation(
                s, config, jnp.int32(p), max_seq, net_config
            ).hand_hist
        )(batch_states)
        for p in range(num_players)
    ])  # [P, B, D]
    p, b, d = hands.shape
    cache = core.apply(
        core_params, hands.reshape(p * b, d), max_seq, method="prefill_fixed"
    )
    observer_fold = jnp.repeat(jnp.arange(num_players, dtype=jnp.int32), b)
    return cache, hands, observer_fold


def _lpt_cache_decision(core, core_params, cache, current_player, globals_,
                        legal_mask, counter, max_seq, batch_size, hands=None):
    """DECISION query for the acting player of each env against the folded cache.

    Gathers row ``current_player*B + b`` (the acting player's own observer cache)
    for each env b, then runs ONE ``decision_logits_fixed`` over the [B] rows.
    ``hands`` [P, B, D] is threaded only for the 'perdigit_tailprob' head, which
    needs the ACTING observer's hand histogram (row ``current_player*B + b``) for
    its belief feature; ignored (``None``) by the legacy head.
    """
    idx = current_player * batch_size + jnp.arange(batch_size, dtype=jnp.int32)
    cache_act = [
        (jnp.take(k, idx, axis=0), jnp.take(v, idx, axis=0)) for (k, v) in cache
    ]
    hand_act = None
    if hands is not None:
        p, b, d = hands.shape
        hand_act = jnp.take(hands.reshape(p * b, d), idx, axis=0)  # [B, D]
    logits, _ = core.apply(
        core_params, cache_act, globals_, legal_mask, counter, max_seq,
        hand_act, method="decision_logits_fixed",
    )
    return logits


def _lpt_cache_append(core, core_params, cache, bid_row, counter, observer_fold,
                      hands, bid_active, max_seq, digit_ids, num_players):
    """Append the new bid into ALL P observer caches in one ``append_bid_fixed``.

    The [B]-shaped per-env inputs are tiled to [P*B] (the same public bid is
    written to every observer's cache); ``observer_fold`` supplies the per-row
    observing player and ``hands`` supplies each row's observer hand histogram.
    ``counter`` / ``digit_ids`` stay [B]-shaped for the caller and are only tiled
    for this call.
    """
    p = num_players
    bid_row_f = jnp.tile(bid_row, (p, 1))
    counter_f = jnp.tile(counter, p)
    bid_active_f = jnp.tile(bid_active, p)
    digit_ids_f = jnp.tile(digit_ids, (p, 1))
    hands_f = hands.reshape(-1, hands.shape[-1])
    return core.apply(
        core_params, cache, bid_row_f, counter_f, observer_fold, hands_f,
        bid_active_f, max_seq, digit_ids_f, method="append_bid_fixed",
    )


def make_network(network_type, jax_config, policy_network_layers,
                 transformer_config=None, factored_count_encoding="linear",
                 temperature: float = 1.0):
    """Create the policy/value network based on type.

    Args:
        network_type: "mlp" or "transformer".
        jax_config: LiarsPokerConfig.
        policy_network_layers: MLP layer sizes (used for "mlp" type).
        transformer_config: Dict with keys d_model, n_heads, n_layers, d_ff
                            (used for "transformer" type).
        factored_count_encoding: 'linear' (default, byte-identical normalized
                            count) or 'tailprob' (game-size-invariant closed-form
                            log-survival q(c)); only used for "factored_mlp".

    Returns:
        A Flax Module with the RNaDNetwork __call__ signature.
    """
    if network_type == "lpt":
        lc = _lpt_config(transformer_config)
        net_config, game_num_digits = _lpt_net_config(
            jax_config, transformer_config
        )
        return LPTNetwork(
            d_model=lc.get("d_model", 64),
            n_layers=lc.get("n_layers", 3),
            n_heads=lc.get("n_heads", 4),
            num_digits=net_config.num_digits,
            max_count=net_config.total_cards,
            num_players=net_config.num_players,
            total_cards=net_config.total_cards,
            num_actions=net_config.num_actions,
            max_seq=_lpt_max_seq(jax_config, transformer_config),
            number_encoding=str(lc.get("number_encoding", "categorical")),
            digit_rope_origin=str(lc.get("digit_rope_origin", "max")),
            head_digit_equivariant=bool(lc.get("head_digit_equivariant", False)),
            game_num_digits=game_num_digits,
            head_mode=str(lc.get("head_mode", "legacy")),
            game_hand_length=jax_config.hand_length,
            dtype=_lpt_dtype(transformer_config),
            remat=lc.get("remat", None),
        )
    if network_type == "mlp":
        return RNaDNetwork(
            policy_network_layers=policy_network_layers,
            num_actions=jax_config.num_actions,
            temperature=temperature,
        )
    elif network_type == "factored_mlp":
        return FactoredMLPNetwork(
            policy_network_layers=policy_network_layers,
            num_actions=jax_config.num_actions,
            num_digits=jax_config.num_digits,
            total_cards=jax_config.total_cards,
            num_players=jax_config.num_players,
            count_encoding=factored_count_encoding,
            temperature=temperature,
        )
    elif network_type == "transformer":
        tc = transformer_config or {}
        return TransformerPolicyNetwork(
            num_players=jax_config.num_players,
            hand_length=jax_config.hand_length,
            num_digits=jax_config.num_digits,
            num_actions=jax_config.num_actions,
            d_model=tc.get("d_model", 128),
            n_heads=tc.get("n_heads", 4),
            n_layers=tc.get("n_layers", 2),
            d_ff=tc.get("d_ff", 0),
        )
    else:
        raise ValueError(f"Unknown network_type: {network_type!r}")


# ---------------------------------------------------------------------------
# Minimal game shim  (replaces pyspiel.Game for RNaDSolver internals)
# ---------------------------------------------------------------------------


class _MockGame:
    """Provides the two methods that RNaDSolver.loss() calls on self._game."""

    def __init__(self, num_players: int, num_distinct_actions: int):
        self._n_players = num_players
        self._n_actions = num_distinct_actions

    def num_players(self) -> int:
        return self._n_players

    def num_distinct_actions(self) -> int:
        return self._n_actions


# ---------------------------------------------------------------------------
# JAX-based RNaD solver
# ---------------------------------------------------------------------------


class JaxLiarsPokerRNaDSolver(RNaDSolver):
    """RNaD solver whose trajectory collection is fully JAX/GPU-based.

    Parameters
    ----------
    config:     Standard RNaDConfig.  ``game_name`` is ignored.
    jax_config: LiarsPokerConfig that defines the game.
    network_type: "mlp" or "transformer".
    transformer_config: Dict with d_model, n_heads, n_layers, d_ff
                        (only used when network_type="transformer").
    """

    def __init__(
        self,
        config: RNaDConfig,
        jax_config: LiarsPokerConfig,
        network_type: str = "mlp",
        transformer_config: dict = None,
        br_state_pool_size: int = 0,
        br_state_ratio: float = 0.0,
        history_encoding: str = "sparse",
        factored_count_encoding: str = "linear",
    ):
        self._jax_config = jax_config
        self._network_type = network_type
        self._factored_count_encoding = factored_count_encoding
        self._transformer_config = transformer_config or {}
        self._max_seq = _lpt_max_seq(jax_config, transformer_config)
        self._history_encoding = history_encoding
        self._br_state_pool_size = br_state_pool_size
        self._br_state_ratio = br_state_ratio
        # Pool storage: list of numpy arrays per LiarsPokerState leaf (lazy init).
        self._state_pool = None        # list of np arrays, shape [pool_max, ...]
        self._state_pool_treedef = None  # jax treedef for reconstructing state
        self._state_pool_write_idx = 0   # next circular write position
        self._state_pool_fill = 0        # number of valid entries (≤ pool_max)
        # Replicate parent __init__ without calling pyspiel.load_game.
        self.config = config
        self.learner_steps = 0
        self.actor_steps = 0
        self.init()

    # ------------------------------------------------------------------
    # Initialisation (no pyspiel dependency)
    # ------------------------------------------------------------------

    def init(self):
        """Initialise networks and optimisers from the JAX config."""
        self._rngkey = jax.random.PRNGKey(self.config.seed)
        self._np_rng = np.random.RandomState(self.config.seed)

        jax_config = self._jax_config
        # Capacity ("max game") sizing for lpt: the network is built for
        # ``net_config`` (a >= max game) while the played game stays
        # ``jax_config``.  In non-capacity mode ``net_config is jax_config`` and
        # all derived quantities are byte-identical to before.
        if self._network_type == "lpt":
            self._net_config, self._game_num_digits = _lpt_net_config(
                jax_config, self._transformer_config
            )
        else:
            # Capacity is an lpt-only feature; other archs stay byte-identical.
            self._net_config, self._game_num_digits = (
                jax_config, jax_config.num_digits
            )
        self._num_actions_net = self._net_config.num_actions
        # Static capacity-action -> played-game-action map (identity when no
        # capacity).  A sampled network action (capacity space) is mapped through
        # this before it is applied to the game env.
        self._cap_to_game = jnp.asarray(
            cap_to_game_action_map(jax_config, self._net_config)
        )
        self._game = _MockGame(jax_config.num_players, jax_config.num_actions)

        if getattr(self.config, "advantage_estimator", "vtrace") == "grpo":
            G = self.config.grpo_group_size
            if self.config.batch_size % G != 0:
                raise ValueError(
                    f"GRPO requires batch_size ({self.config.batch_size}) to be "
                    f"a multiple of grpo_group_size ({G})."
                )

        # NashPG (policy_update=nashpg) regularizes the POLICY (a KL penalty
        # toward the iteratively-refined magnet) instead of the REWARD, so the
        # RNaD reward transform is switched off. eta_reward_transform keeps its
        # default (0.4) in the yaml, so say out loud that it is inert here —
        # NashPG's regularization strength is nashpg_mag_coef.
        if getattr(self.config, "policy_update", "neurd") == "nashpg":
            print(
                "[nashpg] reward transform OFF (eta_reward_transform="
                f"{self.config.eta_reward_transform} is IGNORED). Regularization"
                f" = {self.config.nashpg_mag_coef} * "
                f"{self.config.nashpg_mag_divergence.upper()}(pi || pi_reg), "
                f"entropy bonus {self.config.nashpg_ent_coef}, magnet refreshed "
                f"every {self.config.entropy_schedule_size} learner steps."
            )

        # Per-action NeuRD teacher-floor (nerd_teacher_floor): precompute the
        # game-size belief / action-decode tables once so the collection scan can
        # compute the heuristic teacher per state with a pure vectorized gather.
        # Lazy import of bc_pretrain avoids a module-load import cycle
        # (bc_pretrain imports from train_jax).
        self._nerd_teacher_floor = bool(
            getattr(self.config, "nerd_teacher_floor", False))
        if self._nerd_teacher_floor:
            if self._network_type == "lpt":
                raise ValueError(
                    "nerd_teacher_floor is only wired for the flat "
                    "(mlp/factored_mlp) collection path, not lpt.")
            from bc_pretrain import (
                belief_survival_table, action_count_number,
                heuristic_targets_batched)
            self._teacher_belief_surv = jnp.asarray(
                belief_survival_table(jax_config))
            _tca, _tna = action_count_number(jax_config)
            self._teacher_count_a = jnp.asarray(_tca)
            self._teacher_number_a = jnp.asarray(_tna)
            self._teacher_fn = heuristic_targets_batched

        # A representative (unbatched) env_step used for network init.  For lpt
        # the obs field is a single LPInputs pytree; otherwise a flat vector.
        dummy_env_step = EnvStep(
            obs=_dummy_network_input(
                jax_config, self._network_type, self._history_encoding,
                self._max_seq, self._net_config,
            ),
            legal=np.ones(self._num_actions_net, dtype=np.int8),
            player_id=0.0,
            valid=1.0,
            rewards=np.zeros(jax_config.num_players),
        )

        # Network definition.
        self.network = make_network(
            self._network_type,
            self._jax_config,
            self.config.policy_network_layers,
            self._transformer_config,
            factored_count_encoding=self._factored_count_encoding,
        )
        # Stage-3: standalone LPTransformer core for the KV-cache rollout. Shares
        # the LPTNetwork's inner `LPTransformer_0` params (sliced at collect time),
        # so it is a view onto the SAME weights, not a second network.
        self._lpt_core = (
            _make_lpt_core(
                self._jax_config, self._transformer_config, self._max_seq,
                self._net_config, self._game_num_digits,
            )
            if self._network_type == "lpt"
            else None
        )

        self._entropy_schedule = EntropySchedule(
            sizes=self.config.entropy_schedule_size,
            repeats=self.config.entropy_schedule_repeats,
        )
        self._loss_and_grad = jax.value_and_grad(self.loss, has_aux=False)

        key = self._next_rng_key()
        self.params = self.network.init(key, dummy_env_step)
        self.params_target = self.network.init(key, dummy_env_step)
        self.params_prev = self.network.init(key, dummy_env_step)
        self.params_prev_ = self.network.init(key, dummy_env_step)

        self.optimizer = optax_optimizer(
            self.params,
            optax.chain(
                optax.scale_by_adam(eps_root=0.0, **self.config.adam),
                lr_scale_transform(
                    self.config.learning_rate, self.config.warmup_steps),
                optax.clip(self.config.clip_gradient),
            ),
        )
        self.optimizer_target = optax_optimizer(
            self.params_target,
            optax.sgd(self.config.target_network_avg),
        )

    # ------------------------------------------------------------------
    # Serialisation  (must persist _jax_config alongside parent fields)
    # ------------------------------------------------------------------

    def restore_params_only(self, encoded: bytes) -> None:
        """Restore the four parameter trees + step counters, nothing else.

        Ported from the main worktree so eval_br_vs_agent.py can run HERE --
        wt is the only tree whose game supports dummy_mult/dummy_mode, so
        dummy-probe checkpoints can only be replayed from it.

        For read-only uses (BR training, serving, evaluation) the optax state
        is dead weight, and restoring it couples us to the exact optimizer
        CHAIN the checkpoint was written with: a checkpoint from a variant with
        an LR schedule carries an extra ``count`` state and blows up
        ``from_bytes`` even though its policy loads fine. flax ignores
        state-dict keys absent from the template, so a params-only template
        sidesteps that.
        """
        template = dict(
            params=self.params,
            params_target=self.params_target,
            params_prev=self.params_prev,
            params_prev_=self.params_prev_,
            learner_steps=self.learner_steps,
            actor_steps=self.actor_steps,
        )
        state = flax.serialization.from_bytes(template, encoded)
        self.params = state["params"]
        self.params_target = state["params_target"]
        self.params_prev = state["params_prev"]
        self.params_prev_ = state["params_prev_"]
        self.learner_steps = int(state["learner_steps"])
        self.actor_steps = int(state["actor_steps"])

    def _serializable_state(self) -> dict:
        state = super()._serializable_state()
        # Embed the game config so checkpoints are self-contained.
        state["jax_config"] = {
            "num_players": self._jax_config.num_players,
            "hand_length": self._jax_config.hand_length,
            "num_digits": self._jax_config.num_digits,
        }
        # Persist the private-hand encoding only when it is not the default,
        # so every pre-existing checkpoint still deserialises byte-identically.
        # Checkpoints trained against OpenSpiel's python_liars_poker need
        # "raw"; with hand_length == num_digits the two encodings have the SAME
        # length, so a mismatch is silent -- the net just cannot read its hand.
        if getattr(self._jax_config, "hand_encoding", "histogram") != "histogram":
            state["jax_config"]["hand_encoding"] = str(
                self._jax_config.hand_encoding)
        # Persist max_jump only when set so older-format checkpoints (and the
        # unrestricted game) stay byte-identical.
        if self._jax_config.max_jump is not None:
            state["jax_config"]["max_jump"] = int(self._jax_config.max_jump)
        # Same rule for the action-space probe: persist it only when
        # active, so every pre-existing checkpoint still deserialises
        # byte-identically.
        if getattr(self._jax_config, "dummy_mult", 1) != 1:
            state["jax_config"]["dummy_mult"] = int(
                self._jax_config.dummy_mult)
            state["jax_config"]["dummy_mode"] = str(
                getattr(self._jax_config, "dummy_mode", "random"))
        state["policy_network_layers"] = list(
            self.config.policy_network_layers
        )
        state["network_type"] = self._network_type
        state["history_encoding"] = self._history_encoding
        if self._transformer_config:
            state["transformer_config"] = dict(self._transformer_config)
        return state

    def restore_state(self, encoded: bytes) -> None:
        super().restore_state(encoded)
        # jax_config and policy_network_layers are metadata only; the agent
        # was already constructed with the correct config before restore_state
        # is called, so no action is needed here.

    # ------------------------------------------------------------------
    # JAX trajectory collection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # State pool management (curriculum: start from mid-game states)
    # ------------------------------------------------------------------

    def add_states_to_pool(self, np_states) -> None:
        """Add numpy LiarsPokerState entries to the pool (circular buffer).

        np_states: LiarsPokerState pytree with leading dim N (flattened).
        Terminal states (winner >= 0) are filtered out before insertion.
        """
        if self._br_state_pool_size <= 0:
            return
        non_term_mask = np.array(np_states.winner) < 0
        if not non_term_mask.any():
            return

        leaves, treedef = jax.tree_util.tree_flatten(np_states)
        filtered_leaves = [leaf[non_term_mask] for leaf in leaves]
        N_new = filtered_leaves[0].shape[0]

        M = self._br_state_pool_size
        if self._state_pool is None:
            self._state_pool = [
                np.zeros((M,) + leaf.shape[1:], dtype=leaf.dtype)
                for leaf in filtered_leaves
            ]
            self._state_pool_treedef = treedef

        indices = (
            np.arange(self._state_pool_write_idx, self._state_pool_write_idx + N_new)
            % M
        )
        for pool_arr, new_arr in zip(self._state_pool, filtered_leaves):
            pool_arr[indices] = new_arr

        self._state_pool_write_idx = int(
            (self._state_pool_write_idx + N_new) % M
        )
        self._state_pool_fill = min(self._state_pool_fill + N_new, M)

    def _sample_pool_batch(self):
        """Sample batch_size states from the pool; return as JAX LiarsPokerState."""
        B = self.config.batch_size
        idx = self._np_rng.randint(0, self._state_pool_fill, size=B)
        sampled_leaves = [pool_arr[idx] for pool_arr in self._state_pool]
        sampled_np = jax.tree_util.tree_unflatten(
            self._state_pool_treedef, sampled_leaves
        )
        return jax.tree.map(jnp.array, sampled_np)

    def _env_actions(self, actions):
        """Map sampled network (capacity-space) actions to played-game action ids.

        Identity when no capacity (``self._cap_to_game`` is ``arange``), so this
        is byte-identical in the non-capacity case.  In capacity mode a sampled
        action lives in the fixed 161-wide space and must be decoded back to the
        played game's action id before ``apply_action``.
        """
        return self._cap_to_game[actions]

    def _legal_from_obs(self, obs, states, config):
        """env_step legal mask.  For lpt reuse the (capacity-space)
        ``obs.legal_mask`` so the mask width matches the network's logits; for
        other archs compute the flat game mask.  In non-capacity mode
        ``obs.legal_mask == legal_actions_mask`` so this is byte-identical.
        """
        if self._network_type == "lpt":
            return obs.legal_mask
        return jax.vmap(lambda s: legal_actions_mask(s, config))(states)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _jax_collect_from_pool(self, params, rng_key, start_states) -> RNaDTimeStep:
        """Like _jax_collect but skips the deal phase, starts from start_states."""
        config = self._jax_config

        # A5: ``rewards`` (jax_returns of the CURRENT states) at step t is exactly
        # ``next_rewards`` (jax_returns of new_states) computed at step t-1. Thread
        # it through the carry so jax_returns runs once per state instead of twice.
        init_rewards = jax.vmap(lambda s: jax_returns(s, config))(start_states)

        def traj_step(carry, _):
            states, rng, rewards = carry
            rng, action_rng = jax.random.split(rng)

            obs = _build_obs_batch(
                states, config, self._network_type, self._history_encoding,
                self._max_seq, lambda s: s.current_player, self._net_config,
            )
            legal = self._legal_from_obs(obs, states, config)
            player_id = jax.vmap(lambda s: current_player_id(s, config))(states)
            terminal = jax.vmap(is_terminal)(states)
            valid = ~terminal

            env_step = EnvStep(
                obs=obs,
                legal=legal.astype(jnp.int8),
                player_id=player_id,
                valid=valid,
                rewards=rewards,
            )

            pi, _, _, _ = jax.vmap(self.network.apply, (None, 0), 0)(
                params, env_step
            )

            gumbels = jax.random.gumbel(action_rng, pi.shape, dtype=jnp.float32)
            actions = jnp.argmax(jnp.log(pi + 1e-45) + gumbels, axis=-1).astype(
                jnp.int32
            )
            action_oh = jax.nn.one_hot(actions, self._num_actions_net)

            new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(
                states, self._env_actions(actions)
            )
            next_rewards = jax.vmap(lambda s: jax_returns(s, config))(new_states)

            timestep = RNaDTimeStep(
                env=env_step,
                actor=ActorStep(
                    action_oh=action_oh,
                    policy=pi,
                    rewards=next_rewards,
                ),
            )
            return (new_states, rng, next_rewards), timestep

        _, trajectory = jax.lax.scan(
            traj_step,
            (start_states, rng_key, init_rewards),
            xs=None,
            length=self.config.trajectory_max,
        )
        return trajectory

    def collect_batch_trajectory(self) -> RNaDTimeStep:
        """Collect a batch of trajectories entirely on the accelerator.

        Stochastically starts from pool states (br_state_ratio) or from
        the root (deal phase + trajectory), depending on pool availability.
        """
        rng_key = self._next_rng_key()
        with nvtx_range("rnad/collect_trajectory", color="green"):
            if getattr(self.config, "advantage_estimator", "vtrace") == "grpo":
                # GRPO needs GROUPED (shared-deal) rollouts; the state-pool
                # curriculum would break the group structure, so it is bypassed.
                trajectory = self._jax_collect_grpo(self.params, rng_key)
            elif (
                self._state_pool_fill > 0
                and self._np_rng.random() < self._br_state_ratio
            ):
                start_states = self._sample_pool_batch()
                trajectory = self._jax_collect_from_pool(
                    self.params, rng_key, start_states
                )
            else:
                trajectory = self._jax_collect(self.params, rng_key)
        # Approximate actor_steps (terminal steps also counted for simplicity).
        self.actor_steps += self.config.batch_size * self.config.trajectory_max
        return trajectory

    def step(self):
        """One RNaD learner step: trajectory collection + gradient update."""
        with nvtx_range("rnad/step", color="yellow"):
            # collect_batch_trajectory is annotated with "rnad/collect_trajectory";
            # the remaining time within this range is gradient computation.
            return super().step()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _eval_collect(self, params_curr, params_prev, curr_player_pos, rng_key):
        """Collect batch_size games: curr_player_pos uses params_curr, others use params_prev.

        Returns [B, num_players] per-episode reward arrays (summed over time).
        """
        config = self._jax_config
        batch_size = self.config.batch_size

        batch_states = jax.vmap(lambda _: new_initial_state(config))(
            jnp.zeros(batch_size, dtype=jnp.int32)
        )

        def deal_step(carry, _):
            states, rng = carry
            rng, sub = jax.random.split(rng)
            digits = jax.random.randint(
                sub, (batch_size,), 0, config.num_digits, dtype=jnp.int32
            )
            return (
                jax.vmap(lambda s, a: apply_action(s, config, a))(states, digits),
                rng,
            ), None

        rng_key, deal_rng = jax.random.split(rng_key)
        (batch_states, rng_key), _ = jax.lax.scan(
            deal_step,
            (batch_states, deal_rng),
            xs=None,
            length=config.total_cards,
        )

        def traj_step(carry, _):
            states, rng = carry
            rng, curr_rng, prev_rng = jax.random.split(rng, 3)

            cur_player = jax.vmap(lambda s: current_player_id(s, config))(states)
            terminal = jax.vmap(is_terminal)(states)
            obs = _build_obs_batch(
                states, config, self._network_type, self._history_encoding,
                self._max_seq, lambda s: s.current_player, self._net_config,
            )
            legal = self._legal_from_obs(obs, states, config)

            env_step = EnvStep(
                obs=obs,
                legal=legal.astype(jnp.int8),
                player_id=cur_player,
                valid=~terminal,
                rewards=jnp.zeros((batch_size, config.num_players)),
            )

            pi_curr, _, _, _ = jax.vmap(self.network.apply, (None, 0), 0)(
                params_curr, env_step
            )
            pi_prev, _, _, _ = jax.vmap(self.network.apply, (None, 0), 0)(
                params_prev, env_step
            )

            gumbels_curr = jax.random.gumbel(curr_rng, pi_curr.shape, dtype=jnp.float32)
            gumbels_prev = jax.random.gumbel(prev_rng, pi_prev.shape, dtype=jnp.float32)

            action_curr = jnp.argmax(
                jnp.log(pi_curr + 1e-45) + gumbels_curr, axis=-1
            ).astype(jnp.int32)
            action_prev = jnp.argmax(
                jnp.log(pi_prev + 1e-45) + gumbels_prev, axis=-1
            ).astype(jnp.int32)

            is_curr_turn = cur_player == curr_player_pos
            action = jnp.where(is_curr_turn, action_curr, action_prev)

            new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(
                states, self._env_actions(action)
            )

            new_terminal = jax.vmap(is_terminal)(new_states)
            just_terminated = new_terminal & (~terminal)
            all_ret = jax.vmap(lambda s: jax_returns(s, config))(new_states)
            all_rewards = jnp.where(just_terminated[:, None], all_ret, 0.0)

            return (new_states, rng), all_rewards  # [B, num_players]

        _, all_rewards_traj = jax.lax.scan(
            traj_step,
            (batch_states, rng_key),
            xs=None,
            length=self.config.trajectory_max,
        )
        return all_rewards_traj.sum(axis=0)  # [B, num_players]

    def _jax_collect(self, params, rng_key) -> RNaDTimeStep:
        """Jitted rollout: deal phase via scan, then trajectory_max steps.

        Returns a TimeStep with arrays of shape [T, B, ...] where
          T = trajectory_max
          B = batch_size

        For ``network_type=='lpt'`` this dispatches to the Stage-3 KV-cache
        rollout (`_jax_collect_lpt`), which appends ONE bid token per turn to a
        per-player cache instead of re-encoding the whole bid prefix each step
        (the O(L^2) full-forward bottleneck). The mlp / factored_mlp / transformer
        path below is UNCHANGED.
        """
        if self._network_type == "lpt":
            return self._jax_collect_lpt(params, rng_key, True)
        return self._jax_collect_generic(params, rng_key)

    def _jax_collect_grpo(self, params, rng_key) -> RNaDTimeStep:
        """GRPO rollout: GROUPED deals + independent action sampling.

        Deals ``num_groups = batch_size // grpo_group_size`` distinct hands and
        replicates each ``G = grpo_group_size`` times consecutively (see
        ``_deal_batch(grouped=True)``), then reuses the SAME trajectory scan as
        the normal collect — the KV-cache lpt rollout for ``network_type=='lpt'``
        (so rope_digit / bf16 / the cached decision path all flow through
        unchanged) or the full-forward generic scan otherwise. Within a group the
        deal (both hands) is held fixed and only the per-trajectory Gumbel action
        sampling differs; that shared-deal variance cancels in the leave-one-out
        group baseline in ``rnad.RNaDSolver._loss_grpo``.

        Output layout is byte-identical in shape to ``_jax_collect`` ([T, B, ...]
        with B = batch_size = num_groups * G), so the downstream loss is
        unchanged.
        """
        if self._network_type == "lpt":
            return self._jax_collect_lpt(params, rng_key, True, True)
        return self._jax_collect_generic(params, rng_key, True)

    def _deal_batch(self, rng_key, grouped):
        """Deal the initial batch of states.

        ``grouped=False`` (default): deal ``batch_size`` INDEPENDENT hands.
        ``grouped=True`` (GRPO): deal ``num_groups = batch_size // G`` distinct
        hands then replicate each ``G = grpo_group_size`` times *consecutively*
        (``jnp.repeat`` yields [s0]*G, [s1]*G, ...), so trajectory ``i`` belongs
        to group ``i // G`` and all G members of a group share the SAME chance
        node (both players' hands) — matching the ``reshape(num_groups, G)`` in
        ``rnad.RNaDSolver._loss_grpo``. Returns ``(batch_states, rng_key)`` with
        the batch axis of length ``batch_size`` either way, so the downstream
        trajectory scan is identical.
        """
        config = self._jax_config
        if grouped:
            n_deal = self.config.batch_size // self.config.grpo_group_size
        else:
            n_deal = self.config.batch_size

        states = jax.vmap(lambda _: new_initial_state(config))(
            jnp.zeros(n_deal, dtype=jnp.int32)
        )

        def deal_step(carry, _):
            states, rng = carry
            rng, subkey = jax.random.split(rng)
            actions = jax.random.randint(
                subkey, (n_deal,), 0, config.num_digits, dtype=jnp.int32
            )
            # Chance/deal actions (dealt digits), NOT policy actions -> applied
            # directly to the game (no capacity remap).
            new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(
                states, actions
            )
            return (new_states, rng), None

        rng_key, deal_rng = jax.random.split(rng_key)
        (states, rng_key), _ = jax.lax.scan(
            deal_step,
            (states, deal_rng),
            xs=None,
            length=config.total_cards,
        )
        if grouped:
            G = self.config.grpo_group_size
            states = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, G, axis=0), states
            )
        return states, rng_key

    def _teacher_centered(self, states, legal, config):
        """Heuristic-teacher mean-centered logit [B, A] for the NeuRD floor.

        Teacher = ``belief*exp(-nerd_teacher_lam*jump)`` bids + fixed
        ``nerd_teacher_challenge_prob`` challenge (the softmj1 teacher).  Centered
        by the SAME mean-over-all-A the network logits get in ``get_loss_nerd``
        (``logit - sum(logit*legal)/A``); teacher probs are floored at 1e-9 before
        the log so zero-mass (heuristic-suppressed) actions get a very negative
        centered logit -> their per-action floor drops well below -beta.
        """
        p_teacher, _ = self._teacher_fn(
            states, config, self._teacher_belief_surv,
            self._teacher_count_a, self._teacher_number_a,
            float(self.config.nerd_teacher_lam), 1,
            float(self.config.nerd_teacher_challenge_prob), False)
        logp = jnp.log(jnp.maximum(p_teacher, 1e-9))
        legal_f = legal.astype(jnp.float32)
        A = self._num_actions_net
        centered = logp - jnp.sum(logp * legal_f, axis=-1, keepdims=True) / A
        return centered.astype(jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def _jax_collect_generic(self, params, rng_key, grouped=False) -> RNaDTimeStep:
        """Full-forward rollout (STAGE-1 reference for lpt; production for others).

        For lpt, ``self.network.apply`` re-encodes the entire prefix every step;
        this is the O(L^2) path the KV-cache rollout replaces. It stays here as
        the equivalence reference (`_jax_collect_lpt(use_cache=False)` reuses the
        same scan) and as the byte-identical path for mlp / factored / transformer.

        ``grouped=True`` deals GROUPED (shared-deal) states for GRPO; the
        trajectory scan below is otherwise unchanged (independent Gumbel per
        (trajectory, step), so within a group only the action sampling differs).
        """
        config = self._jax_config

        # --- Deal phase (grouped for GRPO, independent otherwise) ---------
        batch_states, rng_key = self._deal_batch(rng_key, grouped)

        # A5: jax_returns of the CURRENT states at step t == jax_returns of
        # new_states at step t-1. Thread it through the carry (computed once for
        # the dealt states) to avoid the duplicate per-step vmap.
        init_rewards = jax.vmap(lambda s: jax_returns(s, config))(batch_states)

        # --- Trajectory phase: trajectory_max player-action steps ---------
        def traj_step(carry, _):
            states, rng, rewards = carry
            rng, action_rng = jax.random.split(rng)

            obs = _build_obs_batch(
                states, config, self._network_type, self._history_encoding,
                self._max_seq, lambda s: s.current_player, self._net_config,
            )
            legal = self._legal_from_obs(obs, states, config)
            player_id = jax.vmap(lambda s: current_player_id(s, config))(states)
            terminal = jax.vmap(is_terminal)(states)
            valid = ~terminal

            env_step = EnvStep(
                obs=obs,
                legal=legal.astype(jnp.int8),
                player_id=player_id,
                valid=valid,
                rewards=rewards,
            )

            # Per-action NeuRD teacher-floor: compute the heuristic teacher's
            # mean-centered logit per state and attach it to the EnvStep so the
            # jitted loss can use it as the per-action lower trust-region bound.
            # Uses the SAME centering the network logits get in get_loss_nerd
            # (subtract mean over ALL A of logit*legal). Byte-identical when off
            # (teacher_centered stays the () default -> no pytree leaf).
            if self._nerd_teacher_floor:
                env_step = env_step.replace(
                    teacher_centered=self._teacher_centered(states, legal, config))

            pi, _, _, _ = jax.vmap(self.network.apply, (None, 0), 0)(
                params, env_step
            )

            # Gumbel-max categorical sampling (GPU-friendly).
            gumbels = jax.random.gumbel(action_rng, pi.shape, dtype=jnp.float32)
            actions = jnp.argmax(jnp.log(pi + 1e-45) + gumbels, axis=-1).astype(
                jnp.int32
            )
            action_oh = jax.nn.one_hot(actions, self._num_actions_net)

            new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(
                states, self._env_actions(actions)
            )
            next_rewards = jax.vmap(lambda s: jax_returns(s, config))(new_states)

            timestep = RNaDTimeStep(
                env=env_step,
                actor=ActorStep(
                    action_oh=action_oh,
                    policy=pi,
                    rewards=next_rewards,
                ),
            )
            return (new_states, rng, next_rewards), timestep

        _, trajectory = jax.lax.scan(
            traj_step,
            (batch_states, rng_key, init_rewards),
            xs=None,
            length=self.config.trajectory_max,
        )
        return trajectory

    @functools.partial(jax.jit, static_argnums=(0, 3, 4))
    def _jax_collect_lpt(self, params, rng_key, use_cache, grouped=False) -> RNaDTimeStep:
        """Stage-3 KV-cache self-play rollout for network_type=='lpt'.

        Identical to ``_jax_collect_generic`` in every observable way (same deal
        phase, same rng threading, same stored ``env_step`` obs / policy /
        rewards, same gumbel sampling) EXCEPT how the per-step policy ``pi`` is
        obtained:

          * ``use_cache=True`` (production): the acting policy is read from a
            PER-PLAYER incremental KV cache. Each player observes the public bids
            from THEIR perspective (player-relative encoding + own hand), so ALL
            ``P = num_players`` observer perspectives are held in ONE fixed cache
            whose leading dim is folded to `P*B` (row = observer*B + env); it
            lives in the scan carry along with a shared per-env bid `counter`.
            All P hands are prefilled ONCE after the deal (single
            `prefill_fixed`). Per step the acting player's cache rows are gathered
            per-env (`idx = current_player*B + arange(B)`) and a length-1 DECISION
            query (`decision_logits_fixed`) yields the logits -> pi. After
            sampling, the new bid is `append_bid_fixed`-ed into ALL P observer
            caches in one call (masked so only actual BID actions write) and
            `counter` is bumped. This appends ONE bid token per turn instead of
            re-encoding the whole prefix (the O(L^2) full-forward bottleneck) and,
            unlike the old two-cache `jnp.where` select, is correct for P>2 (each
            player reads its OWN observer cache). See `_lpt_cache_prefill` /
            `_lpt_cache_decision` / `_lpt_cache_append` for the fold invariant.

          * ``use_cache=False``: identical scan but ``pi`` comes from the
            full-forward ``self.network.apply``. This is the equivalence
            reference for the correctness test (same rng => byte-identical
            trajectory iff cached logits == full-forward logits).

        The R-NaD LOSS is untouched: it still consumes ``env_step.obs`` (the full
        LPInputs stored here) via ``LPTNetwork.__call__`` (full-forward). Only the
        rollout uses the cache.
        """
        config = self._jax_config
        batch_size = self.config.batch_size
        max_seq = self._max_seq
        core = self._lpt_core
        # The cache methods drive the inner LPTransformer directly, so slice out
        # its sub-params (LPTNetwork wraps it as the `LPTransformer_0` submodule).
        core_params = {"params": params["params"]["LPTransformer_0"]}

        # --- Deal phase (grouped for GRPO, independent otherwise) ---------
        # ``grouped=True`` yields B = num_groups * G states where consecutive G
        # share the same dealt hands; the KV-cache prefill / trajectory scan then
        # operate over the full B batch exactly as in the ungrouped case, so only
        # the action-sampling stochasticity varies within a group.
        batch_states, rng_key = self._deal_batch(rng_key, grouped)

        # --- Prefill ALL P players' hands ONCE (hands fixed post-deal) ----
        # ``cache`` folds the P observer perspectives into a [P*B]-row fixed
        # cache (row = observer*B + env); ``hands`` is [P,B,D] and
        # ``observer_fold`` is [P*B] (value p at row p*B+b).
        num_players = config.num_players
        cache, hands, observer_fold = _lpt_cache_prefill(
            core, core_params, batch_states, config, self._net_config,
            max_seq, num_players,
        )
        counter = jnp.zeros((batch_size,), dtype=jnp.int32)
        # Per-token digit ids of the fixed cache slots (relative_bias mode).
        # Slots 0..D-1 = hand digits 1..D; bid slots start as 0 (null/padding)
        # and get the placed bid's public NUMBER written in as bids arrive.
        # Shared across both player caches (the bid number is public), threaded
        # in the scan carry alongside `counter`.  Harmless no-op in categorical.
        # Sized to the CAPACITY digit count so the D hand slots match the cache
        # (padded slots > D_game are masked by the network's hand-slot mask).
        num_digits = self._net_config.num_digits
        digit_ids = jnp.concatenate([
            jnp.broadcast_to(jnp.arange(1, num_digits + 1),
                             (batch_size, num_digits)),
            jnp.zeros((batch_size, max_seq), dtype=jnp.int32),
        ], axis=1).astype(jnp.int32)  # [B, D+max_seq]

        # A5: jax_returns of the CURRENT states at step t == jax_returns of
        # new_states at step t-1. Thread it through the carry (computed once for
        # the dealt states) to avoid the duplicate per-step vmap.
        init_rewards = jax.vmap(lambda s: jax_returns(s, config))(batch_states)

        # --- Trajectory phase --------------------------------------------
        def traj_step(carry, _):
            states, rng, cache, counter, digit_ids, rewards = carry
            rng, action_rng = jax.random.split(rng)

            obs = _build_obs_batch(
                states, config, self._network_type, self._history_encoding,
                self._max_seq, lambda s: s.current_player, self._net_config,
            )
            legal = self._legal_from_obs(obs, states, config)
            player_id = jax.vmap(lambda s: current_player_id(s, config))(states)
            terminal = jax.vmap(is_terminal)(states)
            valid = ~terminal

            env_step = EnvStep(
                obs=obs,
                legal=legal.astype(jnp.int8),
                player_id=player_id,
                valid=valid,
                rewards=rewards,
            )

            if use_cache:
                # Gather the acting player's own observer-cache rows per env
                # (row = current_player*B + b) and run ONE decision query.
                logits = _lpt_cache_decision(
                    core, core_params, cache, states.current_player,
                    obs.globals_, obs.legal_mask, counter, max_seq, batch_size,
                    hands,
                )
                pi = _legal_policy(logits, env_step.legal)
            else:
                pi, _, _, _ = jax.vmap(self.network.apply, (None, 0), 0)(
                    params, env_step
                )

            # Gumbel-max categorical sampling (GPU-friendly).
            gumbels = jax.random.gumbel(action_rng, pi.shape, dtype=jnp.float32)
            actions = jnp.argmax(jnp.log(pi + 1e-45) + gumbels, axis=-1).astype(
                jnp.int32
            )
            action_oh = jax.nn.one_hot(actions, self._num_actions_net)

            # Append the new bid into ALL P observer caches in one call (masked
            # so only actual BID actions write); challenges / terminal envs are
            # no-ops.
            cur = states.current_player
            bid_active = valid & (actions >= 1)
            # actions are in the CAPACITY action space -> decode with net_config
            # (D_cap) to recover the TRUE (count, number) for the cache bid token.
            bcount, bnumber = decode_bid(actions - 1, self._net_config)
            bid_row = jnp.stack([bcount, bnumber, cur], axis=-1).astype(jnp.int32)
            cache = _lpt_cache_append(
                core, core_params, cache, bid_row, counter, observer_fold,
                hands, bid_active, max_seq, digit_ids, num_players,
            )
            # Write the new bid's public NUMBER into the shared digit-id array at
            # slot D+counter (same masked-write pattern as the cache), then bump.
            slot = num_digits + counter
            total = num_digits + max_seq
            dwrite = (jnp.arange(total)[None, :] == slot[:, None]) & bid_active[:, None]
            digit_ids = jnp.where(dwrite, bnumber[:, None].astype(jnp.int32), digit_ids)
            counter = counter + bid_active.astype(jnp.int32)

            new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(
                states, self._env_actions(actions)
            )
            next_rewards = jax.vmap(lambda s: jax_returns(s, config))(new_states)

            timestep = RNaDTimeStep(
                env=env_step,
                actor=ActorStep(
                    action_oh=action_oh,
                    policy=pi,
                    rewards=next_rewards,
                ),
            )
            return (new_states, rng, cache, counter, digit_ids, next_rewards), timestep

        _, trajectory = jax.lax.scan(
            traj_step,
            (batch_states, rng_key, cache, counter, digit_ids, init_rewards),
            xs=None,
            length=self.config.trajectory_max,
        )
        return trajectory

    # ------------------------------------------------------------------
    # Stage-B: LM-style R-NaD loss for network_type=='lpt'
    #
    # The base RNaDSolver.loss applies each of the 4 networks PER TIMESTEP
    # (jax.vmap over the T axis => T tiny full-forwards per net, each re-encoding
    # that turn's whole bid prefix -- the O(L^2) x T bottleneck the transformer
    # hates).  Here we override ONLY the network-application hook
    # (`_loss_rollouts`); the v-trace / neurd recurrence in the base loss is
    # untouched.  For lpt we replace the T per-timestep forwards with just TWO
    # full-trajectory CAUSAL passes PER NET (one per player), reading out a
    # policy/value for EVERY decision turn at once via
    # ``LPTransformer.causal_readout_pd`` (per-decision generalisation of Stage
    # A's causal_readout, exact-equivalent to per-turn full_forward: needed
    # because challenge decisions share a bid prefix but differ in globals), then
    # SELECT each turn's output from the ACTING player's pass.  8 big causal
    # GEMMs total instead of 4*T tiny ones.
    # ------------------------------------------------------------------
    def _loss_rollouts(self, params, params_target, params_prev, params_prev_, ts):
        if self._network_type != "lpt":
            return super()._loss_rollouts(
                params, params_target, params_prev, params_prev_, ts
            )
        # Param-independent trajectory reorganisation, computed ONCE and reused
        # for all four nets (bids / globals / legal / hands / gather indices do
        # not depend on the network weights).
        prep = self._lpt_prepare_loss_inputs(ts)
        legal = ts.env.legal  # [T, B, A] -- the mask _legal_policy re-applies.

        def ro(p):
            core_params = {"params": p["params"]["LPTransformer_0"]}
            return self._lpt_causal_readout(core_params, prep, legal)

        pi, v, log_pi, logit = ro(params)
        _, v_target, _, _ = ro(params_target)
        _, _, log_pi_prev, _ = ro(params_prev)
        _, _, log_pi_prev_, _ = ro(params_prev_)
        return pi, v, log_pi, logit, v_target, log_pi_prev, log_pi_prev_

    def _lpt_prepare_loss_inputs(self, ts):
        """Reorganise the stored [T, B, ...] lpt trajectory into the per-batch
        arrays the per-decision causal readout needs.  ONE decision-readout slot
        per timestep (decision slot j == turn j), so no bid-count gather is
        needed -- each slot is fed ITS OWN turn's (bid-prefix-length, globals,
        legal).  Returns a dict of param-independent tensors."""
        obs = ts.env.obs  # LPInputs, leaves [T, B, ...]
        P = self._jax_config.num_players
        T = obs.hand_hist.shape[0]
        B = obs.hand_hist.shape[1]

        # --- [T, B, ...] -> [B, T, ...] ------------------------------------
        # observer[t] == acting player at turn t (collect uses s.current_player
        # as the observer), so it doubles as the per-turn actor selector.
        observer_bt = jnp.swapaxes(obs.observer, 0, 1)          # [B, T] int
        # per-decision globals / legal: turn j's own -> decision slot j.
        per_dec_globals = jnp.swapaxes(obs.globals_, 0, 1)      # [B, T, G]
        per_dec_legal = jnp.swapaxes(obs.legal_mask, 0, 1)      # [B, T, A] bool
        hand_bt = jnp.swapaxes(obs.hand_hist, 0, 1)             # [B, T, D]
        # dec_prefix_len[b, j] = #valid bids turn j has seen (its bid-prefix len).
        dec_prefix_len = jnp.swapaxes(
            obs.bid_valid.sum(axis=-1).astype(jnp.int32), 0, 1
        )                                                       # [B, T]

        # --- full bid sequence (the whole game) ----------------------------
        # Bids accumulate monotonically (bid i is always stored at row i; no
        # eviction under the max_seq >= max-bids sizing this path assumes) then
        # freeze once terminal, so the LAST turn's stored bids == the whole
        # game's bid sequence.  Each decision attends only its own prefix via
        # dec_prefix_len, so a single shared bid sequence is exact for all turns.
        full_bids = obs.bids[-1]                                # [B, L, 3]
        full_bid_valid = obs.bid_valid[-1]                      # [B, L] bool

        # --- both players' hands (constant per game) -----------------------
        # hand_bt[t] is the ACTING player's hand at turn t.  Player p's hand is
        # read from the first turn p acted (both players always act in a game --
        # the shortest game is p0 bids, p1 challenges).
        hands = []
        arangeB = jnp.arange(B)
        for p in range(P):
            is_p = (observer_bt == p).astype(jnp.int32)          # [B, T]
            first_t = jnp.argmax(is_p, axis=1)                   # [B]
            hands.append(hand_bt[arangeB, first_t])              # [B, D]

        return {
            "B": B, "T": T, "P": P,
            "hands": hands,                        # list[P] of [B, D]
            "full_bids": full_bids,                # [B, L, 3]
            "full_bid_valid": full_bid_valid,      # [B, L]
            "dec_prefix_len": dec_prefix_len,      # [B, T]
            "per_dec_globals": per_dec_globals,    # [B, T, G]
            "per_dec_legal": per_dec_legal,        # [B, T, A]
            "actor_per_dec": observer_bt,          # [B, T]
        }

    def _lpt_causal_readout(self, core_params, prep, legal_TBA):
        """Two per-decision causal passes (one per player) + per-slot actor
        select -> per-timestep (pi, v, log_pi, logit) matching the base
        per-timestep rollout shapes.

        Returns pi [T,B,A], v [T,B,1], log_pi [T,B,A], logit [T,B,A].
        """
        core = self._lpt_core
        B, T, P = prep["B"], prep["T"], prep["P"]
        actor = prep["actor_per_dec"]               # [B, T]

        # One causal pass per player; select the ACTING player's readout per turn.
        logits_sel = None
        value_sel = None
        for p in range(P):
            observer_p = jnp.full((B,), p, dtype=jnp.int32)
            logits_p, value_p = core.apply(
                core_params,
                prep["hands"][p],
                prep["full_bids"],
                prep["full_bid_valid"],
                observer_p,
                prep["dec_prefix_len"],
                prep["per_dec_globals"],
                prep["per_dec_legal"],
                method="causal_readout_pd",
            )  # logits_p [B, T, A], value_p [B, T]
            sel = (actor == p)                      # [B, T] bool
            contrib_l = jnp.where(sel[:, :, None], logits_p, 0.0)
            contrib_v = jnp.where(sel, value_p, 0.0)
            logits_sel = contrib_l if logits_sel is None else logits_sel + contrib_l
            value_sel = contrib_v if value_sel is None else value_sel + contrib_v
        # logits_sel [B, T, A] = acting player's logits at each turn.

        logit = jnp.swapaxes(logits_sel, 0, 1)                  # [T, B, A]
        v = jnp.swapaxes(value_sel, 0, 1)[..., None]            # [T, B, 1]
        pi = _legal_policy(logit, legal_TBA)
        log_pi = legal_log_policy(logit, legal_TBA)
        return pi, v, log_pi, logit


# ---------------------------------------------------------------------------
# PPO Best-Response Agent
# ---------------------------------------------------------------------------

# Trajectory data collected during a PPO rollout.
# All fields have leading dims [T, B] (time, batch).
PPOStep = collections.namedtuple(
    "PPOStep",
    [
        "obs",  # [T, B, obs_dim]   br_player's observation
        "legal",  # [T, B, A]         legal action mask
        "action",  # [T, B]            action sampled by br_player
        "log_prob",  # [T, B]            log π(action | obs)
        "value",  # [T, B]            V(s) from br_player's perspective
        "reward",  # [T, B]            br_player's reward (non-zero at terminal)
        "all_rewards",  # [T, B, P]         all players' rewards (non-zero at terminal)
        "done",  # [T, B]            True after the episode ends
        "br_valid",  # [T, B]            True when br_player acted this step
        "state",  # [T, B, ...]       pre-action LiarsPokerState (for pool)
    ],
)


class JaxPPOBRAgent:
    """PPO best-response agent with fully-JAX vectorized rollouts.

    Uses TWO decoupled networks:

    * ``replay_network`` — built from ``network_type`` / ``network_layers``
      (the CHECKPOINT architecture). Applies the frozen main agent's
      ``main_params`` to compute the opponent's action distribution during
      rollout. Never trained.
    * ``br_network`` — built from ``br_network_type`` / ``br_layers``, an
      INDEPENDENT architecture for the best-response's own trainable
      policy/value net. ``self.params`` (what PPO updates) live here.

    Each net consumes the observation matching its OWN architecture: the
    frozen replay_network gets ``self._network_type`` obs (structured
    LPInputs for lpt), the trainable br_network gets ``self.br_network_type``
    obs (flat observation_tensor for mlp). ``_collect`` builds the two
    independently per step, so a MIXED representation (e.g. lpt opponent +
    flat-obs mlp BR) is fully supported. When ``br_network_type ==
    network_type`` and ``br_layers == network_layers`` (the defaults), the two
    obs coincide and behavior is exactly the legacy coupled implementation.

    Parameters
    ----------
    jax_config:      Game configuration.
    br_player:       Player index this agent trains to exploit.
    network_layers:  MLP hidden layer sizes of the frozen main agent
                     (must match the checkpoint's ``policy_network_layers``);
                     drives ``replay_network``.
    br_network_type: Architecture of the BR's own net ("mlp",
                     "factored_mlp", ...). Defaults to ``network_type``.
    br_layers:       Hidden layer sizes of the BR's own net. Defaults to
                     ``network_layers``.
    batch_size:      Number of parallel environments per rollout.
    trajectory_max:  Steps per rollout (should cover at least one game).
    lr:              Adam learning rate.
    clip_epsilon:    PPO clipping parameter ε.
    vf_coef:         Value-loss coefficient.
    ent_coef:        Entropy-bonus coefficient.
    gae_lambda:      GAE λ parameter.
    gamma:           Discount factor (1.0 for undiscounted finite games).
    n_epochs:        PPO update epochs per collected batch.
    minibatch_size:  Minibatch size within each epoch.
    seed:            PRNG seed.
    """

    def __init__(
        self,
        jax_config: LiarsPokerConfig,
        br_player: int,
        network_layers: list,
        batch_size: int = 512,
        trajectory_max: int = 50,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gae_lambda: float = 0.95,
        gamma: float = 1.0,
        n_epochs: int = 4,
        minibatch_size: int = 256,
        seed: int = 42,
        network_type: str = "mlp",
        transformer_config: dict = None,
        finetune: FineTuning = None,
        history_encoding: str = "sparse",
        br_network_type: str = None,
        br_layers: list = None,
        br_transformer_config: dict = None,
        factored_count_encoding: str = "linear",
        br_factored_count_encoding: str = None,
        br_temperature: float = 1.0,
    ):
        self._jax_config = jax_config
        # Hard-mask dummy actions for the BR (see _collect). Set by
        # build_ppo_br_agents / train_br's --br-mask-dummy.
        self._br_mask_dummy = False
        # Decoupled/abstraction-independent BR: the BR's OWN policy must play
        # the FULL, unrestricted game so it can exploit the (max_jump-limited)
        # agent via large count-jumps the agent never trained on. The frozen
        # agent opponent keeps its max_jump restriction (via self._jax_config).
        # When max_jump is None these two configs are identical (byte-identical
        # to the unrestricted game).
        self._jax_config_full = dataclasses.replace(jax_config, max_jump=None)
        self._history_encoding = history_encoding
        self._network_type = network_type
        self._transformer_config = transformer_config or {}
        self._max_seq = _lpt_max_seq(jax_config, transformer_config)
        self.br_player = br_player

        # ---- Capacity ("max game") sizing for the FROZEN replay net ----
        # When the frozen opponent (replay_network) is lpt, ``make_network``
        # builds it at ``net_capacity`` (fixed-size params: num_actions=161,
        # hand_hist length Dcap, count normalization /Dcap). The structured obs
        # it consumes must therefore be padded/masked to that SAME capacity (via
        # ``structured_observation(..., net_config)``), and the capacity-space
        # action it samples must be mapped back to the played-game action id
        # before ``apply_action``. Reuses the exact B2 helpers the MAIN agent
        # uses (``_lpt_net_config`` / ``cap_to_game_action_map``). In
        # non-capacity mode (or a non-lpt opponent) ``net_config is jax_config``
        # and ``_cap_to_game`` is ``arange``, so every path below is
        # byte-identical to the legacy game-sized behaviour.
        if network_type == "lpt":
            self._net_config, self._game_num_digits = _lpt_net_config(
                jax_config, self._transformer_config
            )
        else:
            self._net_config, self._game_num_digits = (
                jax_config, jax_config.num_digits
            )
        # capacity replay-action id -> played-game action id (arange if no cap).
        self._cap_to_game = jnp.asarray(
            cap_to_game_action_map(jax_config, self._net_config)
        )
        self.finetune = finetune if finetune is not None else FineTuning()
        self.batch_size = batch_size
        self.trajectory_max = trajectory_max
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self._np_rng = np.random.RandomState(seed)

        # ---- Two decoupled networks -----------------------------------
        # replay_network: applies the FROZEN main agent's params
        # (main_params) to compute the opponent's action distribution. It
        # MUST match the checkpoint architecture (network_type /
        # network_layers), and it is NEVER trained.
        self.replay_network = make_network(
            network_type,
            jax_config,
            network_layers,
            transformer_config,
            factored_count_encoding=factored_count_encoding,
        )
        # B2: standalone LPTransformer core for KV-caching the FROZEN replay
        # (opponent) net inside ``_collect``.  Shares the replay LPTNetwork's
        # inner ``LPTransformer_0`` params (sliced at collect time from
        # ``main_params``), so it is a view onto the SAME frozen weights -- not a
        # second network.  Built with the replay net's config (mirrors the RNaD
        # solver's ``self._lpt_core`` construction) so its cache API matches the
        # ``self._net_config``-padded obs ``_collect`` already builds.
        self._replay_is_lpt = network_type == "lpt"
        self._replay_core = (
            _make_lpt_core(
                self._jax_config, self._transformer_config, self._max_seq,
                self._net_config, self._game_num_digits,
            )
            if self._replay_is_lpt
            else None
        )

        # br_network: the best-response's OWN trainable policy/value net.
        # It may use an INDEPENDENT architecture (br_network_type /
        # br_layers). Its params (self.params) are what PPO updates. When
        # br_network_type == network_type and br_layers == network_layers,
        # this is architecturally identical to replay_network, preserving
        # exact backward-compatibility with the coupled implementation.
        br_network_type = (
            br_network_type if br_network_type is not None else network_type
        )
        br_layers = br_layers if br_layers is not None else network_layers
        br_transformer_config = (
            br_transformer_config
            if br_transformer_config is not None
            else transformer_config
        )
        # The BR's OWN factored count encoding; defaults to the frozen agent's so
        # a coupled factored BR matches. A decoupled mlp BR ignores this.
        br_factored_count_encoding = (
            br_factored_count_encoding
            if br_factored_count_encoding is not None
            else factored_count_encoding
        )
        # Decoupled BR across the structured/flat observation boundary (e.g. an
        # lpt frozen opponent + a flat-obs mlp BR) IS supported: ``_collect``
        # builds two independent obs per step — ``cur_obs`` from
        # ``self._network_type`` (structured LPInputs for lpt) feeds the frozen
        # replay_network, and ``br_obs`` from ``self.br_network_type`` (flat for
        # mlp) feeds the trainable br_network and is what PPOStep.obs stores.
        # The two nets never share an observation, so any mix is valid.
        self.br_network_type = br_network_type
        self.br_layers = list(br_layers)
        # The BR's OWN net keeps a GAME-sized action/obs space (br_obs below is
        # built without net_config). A capacity ("max game") lpt BR net would
        # need its action space (br_action / PPOStep.action / the game legal
        # mask / one_hot in the loss) remapped to capacity too, which is not
        # wired here — the decoupled interleaved BR always uses a flat mlp BR.
        # Fail loudly rather than silently shape-mismatch.
        if br_network_type == "lpt":
            _br_net_config, _ = _lpt_net_config(jax_config, br_transformer_config)
            if _br_net_config is not jax_config:
                raise NotImplementedError(
                    "A capacity (net_capacity) lpt BR net is not supported; the "
                    "BR's own net must stay game-sized. Use br_network_type=mlp "
                    "(the decoupled interleaved BR) with a capacity lpt opponent."
                )
        # Only the BR's OWN net is tempered; the frozen replay_network must
        # reproduce the exploitee exactly and is left at T=1.
        self.br_temperature = float(br_temperature)
        self.br_network = make_network(
            br_network_type,
            jax_config,
            br_layers,
            br_transformer_config,
            factored_count_encoding=br_factored_count_encoding,
            temperature=self.br_temperature,
        )
        # Backward-compat alias: some external callers still reference
        # ``.network`` — it points at the trainable BR net.
        self.network = self.br_network

        # BR params are initialized from br_network, so the dummy obs MUST match
        # the BR's OWN architecture (br_network_type) — flat for mlp/factored,
        # structured LPInputs for lpt — NOT the frozen opponent's network_type.
        # (replay_network is never init'd here; it consumes external main_params
        # loaded from the checkpoint.)
        dummy_env_step = EnvStep(
            obs=_dummy_network_input(
                jax_config, self.br_network_type, self._history_encoding,
                self._max_seq,
            ),
            legal=jnp.ones(jax_config.num_actions, dtype=jnp.int8),
            player_id=0.0,
            valid=1.0,
            rewards=jnp.zeros(jax_config.num_players),
        )
        # BR params are initialized from br_network (NOT from main_params).
        self.params = self.br_network.init(
            jax.random.PRNGKey(seed), dummy_env_step
        )
        self._optimizer = optax.adam(lr)
        self.opt_state = self._optimizer.init(self.params)
        self._init_seed = seed  # remember for reinit

    def reinit_params(self):
        """Randomly reinitialize network parameters and optimizer state."""
        self._init_seed += 1000
        self.params = self.br_network.init(
            jax.random.PRNGKey(self._init_seed),
            EnvStep(
                obs=_dummy_network_input(
                    self._jax_config, self.br_network_type,
                    self._history_encoding, self._max_seq,
                ),
                legal=jnp.ones(self._jax_config.num_actions, dtype=jnp.int8),
                player_id=0.0,
                valid=1.0,
                rewards=jnp.zeros(self._jax_config.num_players),
            ),
        )
        self.opt_state = self._optimizer.init(self.params)

    # ------------------------------------------------------------------
    # JAX rollout (vectorized, JIT-compiled)
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0, 4, 5))
    def _collect(
        self, ppo_params, main_params, rng_key, collect_states: bool = True,
        use_cache: bool = False,
    ) -> PPOStep:
        """Collect batch_size parallel episodes.

        br_player acts under ppo_params; all other players act under
        main_params (i.e. the frozen RNaD target policy).

        Returns PPOStep with leading dims [T, B]. When ``collect_states`` is
        False the (large) pre-action env state pytree is NOT stacked into the
        scan output — PPOStep.state is None (an empty pytree, legal in
        lax.scan). Skip it whenever the state pool is dead (grpo / br_state_ratio
        0 / pool size 0) to drop the [T, B, state] -> host copy in ``train_phase``.

        ``use_cache`` (B2): when the FROZEN opponent net is lpt, drive its policy
        from a per-player incremental KV cache (``_lpt_cache_*``) instead of a
        full transformer forward every scan step -- exactly the mechanism the
        RNaD self-play collect (``_jax_collect_lpt``) uses. All P observer
        perspectives are prefilled ONCE after the deal into one folded
        ``[P*B]``-row cache carried in the scan; per step the acting player's own
        cache rows yield the logits (``_lpt_cache_decision``) and EVERY played
        bid (from the BR player AND the main agent -- the public bid history) is
        appended (``_lpt_cache_append``). The uncached path (``use_cache=False``)
        is the byte-identical equivalence reference: same rng => same trajectory.
        Only valid for an lpt replay net; the caller passes
        ``use_cache=self._replay_is_lpt``.
        """
        config = self._jax_config
        config_full = self._jax_config_full
        batch_size = self.batch_size
        br_player = self.br_player

        # Initialise a batch of independent games.
        batch_states = jax.vmap(lambda _: new_initial_state(config))(
            jnp.zeros(batch_size, dtype=jnp.int32)
        )

        # Deal phase: assign hands via random chance actions.
        def deal_step(carry, _):
            states, rng = carry
            rng, sub = jax.random.split(rng)
            digits = jax.random.randint(
                sub, (batch_size,), 0, config.num_digits, dtype=jnp.int32
            )
            return (
                jax.vmap(lambda s, a: apply_action(s, config, a))(
                    states, digits
                ),
                rng,
            ), None

        rng_key, deal_rng = jax.random.split(rng_key)
        (batch_states, rng_key), _ = jax.lax.scan(
            deal_step,
            (batch_states, deal_rng),
            xs=None,
            length=config.total_cards,
        )

        # --- B2: prefill the frozen lpt opponent's KV cache (once, post-deal) --
        # Mirrors ``_jax_collect_lpt``: all P observer perspectives folded into a
        # single [P*B]-row cache, plus a shared per-env bid ``counter`` and
        # ``digit_ids`` (public bid-number per cache slot), threaded in the scan
        # carry. Only built when the opponent is lpt AND caching is requested.
        if use_cache:
            core = self._replay_core
            # The cache methods drive the inner LPTransformer directly, so slice
            # its sub-params from the FROZEN main params (LPTNetwork wraps it as
            # the ``LPTransformer_0`` submodule).
            core_params = {"params": main_params["params"]["LPTransformer_0"]}
            num_players = config.num_players
            cache0, hands, observer_fold = _lpt_cache_prefill(
                core, core_params, batch_states, config, self._net_config,
                self._max_seq, num_players,
            )
            counter0 = jnp.zeros((batch_size,), dtype=jnp.int32)
            num_digits = self._net_config.num_digits
            digit_ids0 = jnp.concatenate([
                jnp.broadcast_to(jnp.arange(1, num_digits + 1),
                                 (batch_size, num_digits)),
                jnp.zeros((batch_size, self._max_seq), dtype=jnp.int32),
            ], axis=1).astype(jnp.int32)  # [B, D+max_seq]

        def traj_step(carry, _):
            if use_cache:
                states, rng, cache, counter, digit_ids = carry
            else:
                states, rng = carry
            rng, br_rng, main_rng = jax.random.split(rng, 3)

            cur_player = jax.vmap(lambda s: current_player_id(s, config))(
                states
            )
            terminal = jax.vmap(is_terminal)(states)
            # Two masks under the max_jump abstraction (identical when
            # max_jump is None):
            #   legal_agent -- the frozen RNaD agent (opponent) plays the
            #     max_jump-RESTRICTED game (`config`).
            #   legal_br    -- the BR's OWN policy plays the FULL, unrestricted
            #     game (`config_full`, max_jump=None) so it can exploit large
            #     count-jumps the restricted agent never trained on.
            legal_agent = jax.vmap(lambda s: legal_actions_mask(s, config))(
                states
            )
            legal_br = jax.vmap(
                lambda s: legal_actions_mask(s, config_full)
            )(states)
            # Hard-mask DUMMY actions out of the BR's OWN action space.
            # A best response never wants a strictly dominated action, so
            # removing them cannot weaken the BR as a measuring instrument --
            # it only relieves it of learning to avoid (mult-1)/mult of its
            # action space, which otherwise makes the BR weak exactly where we
            # need it sharp (the dummy probes). Masking legal_br rather than
            # the sampled action keeps pi_br, log_pi_br and the PPO loss
            # mutually consistent, since br_network.apply forms its legal
            # policy from this mask. No-op when dummy_mult == 1.
            if (self._br_mask_dummy
                    and config_full.num_real_actions < config_full.num_actions):
                _real_mask = (jnp.arange(config_full.num_actions)
                              < config_full.num_real_actions)
                legal_br = legal_br * _real_mask.astype(legal_br.dtype)

            # --- BR player's observation (always from br_player's POV) ------
            # Built from the BR's OWN architecture (br_network_type): flat
            # observation_tensor for mlp/factored/transformer, structured
            # LPInputs for lpt. This is what feeds br_network AND is stored in
            # PPOStep.obs for the PPO update — it is independent of the frozen
            # opponent's obs (cur_obs, below). Built with `config_full` so a
            # structured (lpt) BR obs also carries the FULL legal_mask; the flat
            # observation_tensor does not depend on max_jump, so this is
            # byte-identical for the mlp BR.
            br_obs = _build_obs_batch(
                states, config_full, self.br_network_type,
                self._history_encoding,
                self._max_seq, lambda s: jnp.int32(br_player),
            )
            br_env = EnvStep(
                obs=br_obs,
                legal=legal_br.astype(jnp.int8),
                player_id=jnp.full(batch_size, float(br_player)),
                valid=~terminal,
                rewards=jnp.zeros((batch_size, config.num_players)),
            )
            pi_br, v_br_raw, log_pi_br, _ = jax.vmap(
                self.br_network.apply, (None, 0), 0
            )(ppo_params, br_env)
            v_br = v_br_raw[..., 0]  # [B]

            # Sample action from PPO policy (Gumbel-max).
            #
            # No temperature knob here on purpose: it lives in the NETWORK
            # (JaxPPOBRAgent(br_temperature=...) -> make_network), so pi_br is
            # ALREADY the tempered policy. Sampling, the recorded log_prob, the
            # PPO ratio, the entropy bonus and evaluation then all refer to one
            # and the same distribution, and collection stays exactly on-policy
            # (ratio == 1 at collection). Tempering only the SAMPLING would
            # instead make the update silently off-policy.
            gumbels_br = jax.random.gumbel(
                br_rng, pi_br.shape, dtype=jnp.float32
            )
            br_action = jnp.argmax(
                jnp.log(pi_br.astype(jnp.float32) + 1e-45) + gumbels_br, axis=-1
            ).astype(jnp.int32)
            br_log_prob = jnp.sum(
                log_pi_br * jax.nn.one_hot(br_action, config.num_actions),
                axis=-1,
            )

            # --- Main agent action for the current player -------------------
            # Built from the FROZEN opponent's architecture (self._network_type,
            # the checkpoint arch): structured LPInputs for lpt, flat otherwise.
            # Transient — only used to sample the opponent's action; never
            # stored in PPOStep or trained on. Independent of br_obs above.
            #
            # CAPACITY: for lpt the replay_network is built at net_capacity
            # (fixed-size params), so its obs MUST be padded/masked to capacity
            # via ``self._net_config`` (hand_hist -> Dcap + hand-valid mask,
            # bids -> max_seq, from-max with D_game, legal_mask -> 161 with the
            # game->capacity action remap). ``net_config is jax_config`` when no
            # capacity (or non-lpt), so this is byte-identical to before.
            cur_obs = _build_obs_batch(
                states, config, self._network_type, self._history_encoding,
                self._max_seq, lambda s: s.current_player, self._net_config,
            )
            # Legal mask for the replay net. For lpt the network's logits are
            # capacity-wide (161) and it masks via BOTH obs.legal_mask (internal
            # logit mask) AND env_step.legal (its _legal_policy), so BOTH must be
            # the capacity-space mask carried in ``cur_obs.legal_mask``. For the
            # flat archs (mlp/factored/transformer) cur_obs is a plain vector
            # with no legal_mask, so fall back to the game-space ``legal`` (and
            # in non-capacity lpt, cur_obs.legal_mask == legal, so identical).
            # For lpt the restricted mask is already carried in
            # cur_obs.legal_mask (structured_observation applies
            # legal_actions_mask with `config`, i.e. max_jump-restricted); for
            # the flat archs use the game-space legal_agent (also restricted).
            if self._network_type == "lpt":
                main_legal = cur_obs.legal_mask.astype(jnp.int8)
            else:
                main_legal = legal_agent.astype(jnp.int8)
            main_env = EnvStep(
                obs=cur_obs,
                legal=main_legal,
                player_id=cur_player,
                valid=~terminal,
                rewards=jnp.zeros((batch_size, config.num_players)),
            )
            if use_cache:
                # Read the acting player's own observer-cache rows per env
                # (row = current_player*B + b) and run ONE decision query. The
                # cached logits already carry the internal ``legal_mask`` logit
                # mask (``decision_logits_fixed`` -> ``_heads``), exactly as the
                # full forward does; ``_legal_policy`` re-applies ``main_legal``,
                # matching ``LPTNetwork.__call__``'s ``pi``. Byte-identical.
                logits_main = _lpt_cache_decision(
                    core, core_params, cache, states.current_player,
                    cur_obs.globals_, cur_obs.legal_mask, counter,
                    self._max_seq, batch_size, hands,
                )
                pi_main = _legal_policy(logits_main, main_legal)
            else:
                pi_main, _, _, _ = jax.vmap(
                    self.replay_network.apply, (None, 0), 0
                )(main_params, main_env)
            pi_main = jax.vmap(self.finetune.post_process_policy)(
                pi_main, main_legal
            )
            gumbels_main = jax.random.gumbel(
                main_rng, pi_main.shape, dtype=jnp.float32
            )
            main_action = jnp.argmax(
                jnp.log(pi_main.astype(jnp.float32) + 1e-45) + gumbels_main,
                axis=-1,
            ).astype(jnp.int32)
            # Map the sampled action from the replay net's (capacity) action
            # space back to the played-game action id before it is applied to
            # the env. Identity (arange) when no capacity, so byte-identical.
            main_action = self._cap_to_game[main_action]

            # Select action: br_player uses PPO policy (already game-space, since
            # br_network is the flat mlp), others use the (remapped) main policy.
            is_br_turn = cur_player == br_player
            action = jnp.where(is_br_turn, br_action, main_action)

            # --- B2: append EVERY played bid to the frozen opponent's caches ---
            # The cache holds the PUBLIC bid history, so bids from BOTH the BR
            # player and the main agent must be appended (mirrors
            # ``_jax_collect_lpt``, where all bids are appended regardless of who
            # bid). ``action`` is the FINAL applied action in GAME space (BR
            # actions are game-space; ``main_action`` was mapped via
            # ``_cap_to_game``), so decode the TRUE (count, number) with the GAME
            # config. ``digit_ids`` sizing uses the NET config's num_digits.
            if use_cache:
                valid = ~terminal
                bid_active = valid & (action >= 1)
                bcount, bnumber = decode_bid(action - 1, config)
                cur = states.current_player
                bid_row = jnp.stack(
                    [bcount, bnumber, cur], axis=-1
                ).astype(jnp.int32)
                cache = _lpt_cache_append(
                    core, core_params, cache, bid_row, counter, observer_fold,
                    hands, bid_active, self._max_seq, digit_ids, num_players,
                )
                # Write the new bid's public NUMBER into the shared digit-id
                # array at slot D+counter (masked like the cache), then bump.
                slot = num_digits + counter
                total = num_digits + self._max_seq
                dwrite = (
                    jnp.arange(total)[None, :] == slot[:, None]
                ) & bid_active[:, None]
                digit_ids = jnp.where(
                    dwrite, bnumber[:, None].astype(jnp.int32), digit_ids
                )
                counter = counter + bid_active.astype(jnp.int32)

            new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(
                states, action
            )

            # Reward is non-zero only at the step that causes termination,
            # avoiding double-counting across subsequent (invalid) steps.
            new_terminal = jax.vmap(is_terminal)(new_states)
            just_terminated = new_terminal & (~terminal)
            all_ret = jax.vmap(lambda s: jax_returns(s, config))(new_states)
            just_term_bc = just_terminated[:, None]  # [B, 1] for broadcast
            all_rewards_step = jnp.where(just_term_bc, all_ret, 0.0)
            reward_br = all_rewards_step[:, br_player]

            step = PPOStep(
                obs=br_obs,
                legal=legal_br,
                action=br_action,
                log_prob=br_log_prob,
                value=v_br,
                reward=reward_br,
                all_rewards=all_rewards_step,
                done=new_terminal,
                br_valid=is_br_turn & (~terminal),
                # None (empty pytree) when the pool is dead: the state stacking
                # then vanishes from the compiled scan output entirely.
                state=states if collect_states else None,
            )
            if use_cache:
                return (new_states, rng, cache, counter, digit_ids), step
            return (new_states, rng), step

        init_carry = (
            (batch_states, rng_key, cache0, counter0, digit_ids0)
            if use_cache
            else (batch_states, rng_key)
        )
        _, trajectory = jax.lax.scan(
            traj_step,
            init_carry,
            xs=None,
            length=self.trajectory_max,
        )
        return trajectory

    # ------------------------------------------------------------------
    # GAE computation (JIT-compiled backward scan)
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_gae(self, values, rewards, dones):
        """Compute GAE advantages via backward scan.

        Args:
            values:  [T, B] float64 critic estimates
            rewards: [T, B] float64 sparse terminal rewards
            dones:   [T, B] bool    episode-end flags

        Returns:
            advantages: [T, B]
            returns:    [T, B]  (advantage + value, used as critic targets)
        """
        gamma, lam = self.gamma, self.gae_lambda

        def backward_step(carry, x):
            next_val, last_gae = carry
            val, rew, done = x
            not_done = ~done
            delta = rew + gamma * next_val * not_done - val
            gae = delta + gamma * lam * last_gae * not_done
            return (val, gae), gae

        B = values.shape[1]
        init = (jnp.zeros(B), jnp.zeros(B))
        xs = (jnp.flip(values, 0), jnp.flip(rewards, 0), jnp.flip(dones, 0))
        _, adv_rev = jax.lax.scan(backward_step, init, xs)
        advantages = jnp.flip(adv_rev, 0)
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO loss and JIT-compiled update step
    # ------------------------------------------------------------------

    def _loss_fn(
        self,
        params,
        obs,
        legal,
        action,
        old_log_prob,
        advantage,
        return_,
        valid,
    ):
        # `obs` is either a flat [B, F] array (mlp/factored) or a batched
        # LPInputs pytree (lpt); derive B from the first leaf either way.
        B = jax.tree_util.tree_leaves(obs)[0].shape[0]
        env_step = EnvStep(
            obs=obs,
            legal=legal,
            player_id=jnp.full(B, float(self.br_player)),
            valid=valid,
            rewards=jnp.zeros((B, self._jax_config.num_players)),
        )
        pi, v_raw, log_pi, _ = jax.vmap(self.br_network.apply, (None, 0), 0)(
            params, env_step
        )
        v = v_raw[..., 0]  # [B]

        action_oh = jax.nn.one_hot(action, self._jax_config.num_actions)
        log_prob = jnp.sum(log_pi * action_oh, axis=-1)

        ratio = jnp.exp(log_prob - old_log_prob)
        pg1 = ratio * advantage
        pg2 = (
            jnp.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantage
        )
        n_valid = jnp.sum(valid) + 1e-8
        pg_loss = -jnp.sum(jnp.minimum(pg1, pg2) * valid) / n_valid
        v_loss = jnp.sum(((v - return_) ** 2) * valid) / n_valid
        # Safe entropy: avoids 0 * log(0) = NaN for illegal actions.
        ent = -jnp.sum(jnp.where(pi > 0, pi * log_pi, 0.0), axis=-1)
        ent_loss = -jnp.sum(ent * valid) / n_valid

        return pg_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss

    @functools.partial(jax.jit, static_argnums=(0,))
    def _prepare_ppo_batch(
        self, obs, legal, action, log_prob, value, reward, done, valid
    ):
        """Flatten a [T, B] trajectory into device-resident [N, ...] PPO tensors.

        Runs GAE (``_compute_gae``) and the advantage normalisation entirely on
        device, then reshapes every leaf [T, B, ...] -> [T*B, ...]. The
        normalisation matches the old host branch exactly when any step is valid;
        the ``if n_valid > 0`` host branch becomes a branchless ``jnp.where`` and
        ``n_valid`` is clamped to >= 1 for the mean/var denominator to avoid nan
        when no step is valid (the where then falls back to the raw advantage).
        """
        advantages, returns = self._compute_gae(value, reward, done)

        # Normalise advantages over valid (br_player's) steps — same math as the
        # former host branch, made branchless for JIT.
        n_valid = valid.sum()
        denom = jnp.maximum(n_valid, 1)
        mean_adv = (advantages * valid).sum() / denom
        var_adv = ((advantages - mean_adv) ** 2 * valid).sum() / denom
        norm_adv = jnp.where(
            n_valid > 0,
            (advantages - mean_adv) / (jnp.sqrt(var_adv) + 1e-8),
            advantages,
        )

        # Flatten [T, B, ...] -> [N, ...]. `obs` is a flat array (mlp/factored)
        # or an LPInputs pytree (lpt); flatten each leaf the same way.
        T, B = legal.shape[:2]
        N = T * B
        flat = lambda a: a.reshape((N,) + a.shape[2:])
        obs_f = jax.tree_util.tree_map(flat, obs)
        legal_f = legal.reshape((N,) + legal.shape[2:])
        action_f = action.reshape(N)
        log_prob_f = log_prob.reshape(N)
        adv_f = norm_adv.reshape(N)
        ret_f = returns.reshape(N)
        valid_f = valid.reshape(N)
        return obs_f, legal_f, action_f, log_prob_f, adv_f, ret_f, valid_f

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def _update_epoch(
        self,
        params,
        opt_state,
        perm,
        obs,
        legal,
        action,
        old_log_prob,
        advantage,
        return_,
        valid,
    ):
        """Run one PPO epoch (all minibatches) fully on device.

        ``perm`` is [n_mb, mb] int32 — one row per minibatch, threading params /
        opt_state through a lax.scan (same sequential order as the old host
        loop). Each step gathers its minibatch with ``jnp.take`` and applies the
        SAME ``_loss_fn`` + optax update the old ``_update_step`` used. Returns
        the updated params / opt_state and the per-minibatch losses [n_mb].
        """

        def mb_step(carry, idx):
            params, opt_state = carry
            gather = lambda a: jnp.take(a, idx, axis=0)
            obs_mb = jax.tree_util.tree_map(gather, obs)
            loss, grads = jax.value_and_grad(self._loss_fn)(
                params,
                obs_mb,
                gather(legal),
                gather(action),
                gather(old_log_prob),
                gather(advantage),
                gather(return_),
                gather(valid),
            )
            updates, opt_state = self._optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            mb_step, (params, opt_state), perm
        )
        return params, opt_state, losses

    # ------------------------------------------------------------------
    # Training phase: collect → GAE → PPO updates
    # ------------------------------------------------------------------

    def train_phase(
        self, main_params, rng_key, n_collect: int = 1,
        collect_states: bool = False,
    ):
        """Collect n_collect batches of trajectories and run PPO updates.

        Args:
            main_params:    frozen RNaD target params for opponent players
            rng_key:        JAX PRNG key
            n_collect:      number of collect+update cycles
            collect_states: when True, stack + return the pre-action env states
                for the RNaD state pool; when False (the default, and whenever
                the pool is dead) neither ``_collect`` stacks them nor do we copy
                them to host, and ``combined_states`` is None.

        Returns:
            Tuple of (mean PPO loss, numpy LiarsPokerState with shape [N, ...] or
            None). When populated, N is the total number of states across all
            collected trajectories (T * B * n_collect, unfiltered — includes
            terminal states).
        """
        losses = []
        all_traj_states = []
        for _ in range(n_collect):
            rng_key, collect_key = jax.random.split(rng_key)
            with nvtx_range("br/collect_trajectory", color="green"):
                traj = self._collect(
                    self.params, main_params, collect_key, collect_states,
                    self._replay_is_lpt,
                )

            # Flatten states [T, B, ...] → [T*B, ...] and move to numpy — only
            # when the pool is active (otherwise traj.state is None).
            if collect_states:
                T, B = traj.legal.shape[:2]  # legal is a plain [T, B, A] array
                N = T * B
                flat_states = jax.tree.map(
                    lambda arr: np.array(arr).reshape((N,) + arr.shape[2:]),
                    traj.state,
                )
                all_traj_states.append(flat_states)

            # GAE + advantage-norm + [T, B, ...] → [N, ...] flatten, all on device.
            with nvtx_range("br/compute_gae", color="cyan"):
                (
                    obs_f, legal_f, action_f, log_prob_f, adv_f, ret_f, valid_f
                ) = self._prepare_ppo_batch(
                    traj.obs, traj.legal, traj.action, traj.log_prob,
                    traj.value, traj.reward, traj.done, traj.br_valid,
                )

            T, B = traj.legal.shape[:2]
            N = T * B
            # The device epoch runs a fixed [n_mb, mb] lax.scan, so N must tile
            # evenly into minibatches. The old host loop *could* run a short
            # final minibatch, but its per-minibatch ``n_valid`` normalisation
            # meant a ragged tail never matched a full one cleanly anyway; every
            # in-repo config uses N divisible by minibatch_size, so we assert it
            # rather than silently changing semantics.
            assert N % self.minibatch_size == 0, (
                f"trajectory size N={N} (T*B) must be divisible by "
                f"minibatch_size={self.minibatch_size}"
            )
            n_mb = N // self.minibatch_size

            with nvtx_range("br/gradient_updates", color="red"):
                for _ in range(self.n_epochs):
                    # Byte-identical minibatch schedule to the old host loop:
                    # one permutation per epoch, sliced into contiguous chunks.
                    perm = self._np_rng.permutation(N)
                    perm = perm.reshape(n_mb, self.minibatch_size).astype(
                        np.int32
                    )
                    self.params, self.opt_state, mb_losses = self._update_epoch(
                        self.params,
                        self.opt_state,
                        jnp.asarray(perm),
                        obs_f,
                        legal_f,
                        action_f,
                        log_prob_f,
                        adv_f,
                        ret_f,
                        valid_f,
                    )
                    losses.append(mb_losses)  # [n_mb] device array

        combined_states = (
            jax.tree.map(lambda *arrs: np.concatenate(arrs, axis=0), *all_traj_states)
            if all_traj_states
            else None
        )
        # ONE host transfer for the whole phase's loss mean.
        mean_loss = (
            float(jax.device_get(jnp.mean(jnp.concatenate(losses))))
            if losses
            else 0.0
        )
        return mean_loss, combined_states


# ---------------------------------------------------------------------------
# Build BR agents
# ---------------------------------------------------------------------------


def _rnad_config_with_nerd_beta(rnad_config):
    """Build an RNaDConfig, honouring a FLAT `nerd_beta` override.

    `nerd` is a nested NerdConfig, so a yaml/Hydra value cannot address
    `nerd.beta` directly -- RNaDConfig(**cfg) would be handed a dict where a
    NerdConfig is expected. This accepts a top-level `nerd_beta` scalar and
    folds it in, leaving every other field untouched. nerd_beta=None (the
    default) is byte-identical to RNaDConfig(**rnad_config).

    beta is the NeuRD trust region: apply_force_with_threshold freezes
    downward force once a centered logit reaches -beta, so with the best action
    at +beta and the rest on the floor the RETAINED TAIL MASS is
    (n-1)/(exp(2*beta) + n-1). At fixed beta that GROWS with the action count
    (0.104 at |A_legal|=7.3 to 0.513 at 58.4), so action-space scaling studies
    need   beta_m = beta_0 + 0.5*ln((n_m - 1)/(n_1 - 1))   to hold it constant.
    """
    cfg = dict(rnad_config)
    nerd_beta = cfg.pop("nerd_beta", None)
    out = RNaDConfig(**cfg)
    if nerd_beta is not None:
        out = dataclasses.replace(
            out, nerd=NerdConfig(beta=float(nerd_beta), clip=out.nerd.clip))
    return out


def build_ppo_br_agents(
    jax_config: LiarsPokerConfig,
    ppo_config: dict,
    seed: int = 42,
    network_type: str = "mlp",
    transformer_config: dict = None,
    history_encoding: str = "sparse",
    br_network_type: str = None,
    br_layers: list = None,
    br_transformer_config: dict = None,
    factored_count_encoding: str = "linear",
    br_factored_count_encoding: str = None,
    finetune: "FineTuning" = None,
    br_temperature: float = 1.0,
) -> list:
    """Create one JaxPPOBRAgent per player position.

    ``br_network_type`` / ``br_layers`` (when given) select an independent
    architecture for the BR's own trainable net; they default to the main
    agent's ``network_type`` / layers for backward-compatibility.
    """
    return [
        JaxPPOBRAgent(
            jax_config=jax_config,
            br_player=p,
            seed=seed + p,
            network_type=network_type,
            transformer_config=transformer_config,
            history_encoding=history_encoding,
            br_network_type=br_network_type,
            br_layers=br_layers,
            br_transformer_config=br_transformer_config,
            factored_count_encoding=factored_count_encoding,
            br_factored_count_encoding=br_factored_count_encoding,
            # The FROZEN agent's policy is post-processed with this on every
            # BR rollout step (see the post_process_policy call in _collect):
            # actions below policy_threshold are ZEROED and the rest rounded to
            # multiples of 1/policy_discretization. Note that call goes to
            # post_process_policy DIRECTLY, bypassing the from_learner_steps
            # gate, so FineTuning()'s defaults (threshold 0.03, discretization
            # 32) are ALWAYS in force unless overridden here -- the exploitee
            # the BR faces is thresholded, not raw.
            finetune=finetune if finetune is not None else FineTuning(),
            br_temperature=br_temperature,
            **ppo_config,
        )
        for p in range(jax_config.num_players)
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_br_vs_main(
    jax_config: LiarsPokerConfig,
    main_agent: JaxLiarsPokerRNaDSolver,
    br_agents: list,
    rng_key,
    n_episodes: int,
) -> dict:
    """Vectorized evaluation of each PPO BR agent against the frozen main agent.

    For each player position p, runs n_episodes parallel games where
    br_agents[p] plays as player p and main_agent fills all other seats.

    Returns wandb-ready metrics:
      br/reward_p{i}  — mean reward of BR agent i
      br/agent_score  — mean reward of the main agent across all configurations
    """
    rewards_by_pos = []  # rewards_by_pos[p] has shape [num_players]
    batch_size = br_agents[0].batch_size
    n_batches = max(1, (n_episodes + batch_size - 1) // batch_size)

    for p, agent in enumerate(br_agents):
        ep_all_rewards = []  # list of [B, num_players] arrays
        for _ in range(n_batches):
            rng_key, k = jax.random.split(rng_key)
            # Eval only reads traj.all_rewards; skip the state pytree.
            traj = agent._collect(
                agent.params, main_agent.params_target, k, False,
                agent._replay_is_lpt,
            )
            # Sum rewards over time → [B, num_players] per-episode returns.
            ep_all_rewards.append(np.array(traj.all_rewards.sum(axis=0)))
        all_rewards = np.concatenate(ep_all_rewards, axis=0)[:n_episodes]
        rewards_by_pos.append(all_rewards.mean(axis=0))  # [num_players]

    logs = {}
    for i, rewards in enumerate(rewards_by_pos):
        logs[f"br/reward_p{i}"] = float(rewards[i])

    # Main agent score: mean reward across all BR configurations.
    main_rewards = [
        float(rewards_by_pos[br][main])
        for br in range(jax_config.num_players)
        for main in range(jax_config.num_players)
        if main != br
    ]
    logs["br/agent_score"] = float(np.mean(main_rewards))
    return logs


def eval_current_vs_prev(
    jax_config: LiarsPokerConfig,
    agent: JaxLiarsPokerRNaDSolver,
    params_prev,
    rng_key,
    n_games: int = 4096,
) -> dict:
    """Evaluate current agent params_target against a previous checkpoint's params_target.

    For each player position p, runs n_games where the current agent plays as
    player p and the previous checkpoint fills all other seats.  Returns
    wandb-ready metrics:
      ckpt_eval/reward_curr_p{p}  — mean reward of current agent as player p
      ckpt_eval/mean_reward_curr  — mean across all positions
    """
    params_curr = agent.params_target
    batch_size = agent.config.batch_size
    n_batches = max(1, (n_games + batch_size - 1) // batch_size)

    curr_rewards_by_pos = []
    for p in range(jax_config.num_players):
        ep_rewards = []
        for _ in range(n_batches):
            rng_key, k = jax.random.split(rng_key)
            all_rewards = agent._eval_collect(params_curr, params_prev, p, k)
            ep_rewards.append(np.array(all_rewards)[:, p])  # [B] reward for current
        curr_rewards = np.concatenate(ep_rewards)[:n_games]
        curr_rewards_by_pos.append(float(curr_rewards.mean()))

    logs = {}
    for p, r in enumerate(curr_rewards_by_pos):
        logs[f"ckpt_eval/reward_curr_p{p}"] = r
    logs["ckpt_eval/mean_reward_curr"] = float(np.mean(curr_rewards_by_pos))
    return logs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def checkpoint(agent, save_dir, step):
    """Save agent state via flax.serialization (msgpack).

    ``save_dir`` may be a local directory or an ``s3://bucket/prefix`` URI
    (Cloudflare R2 works via the AWS_ENDPOINT_URL env var).
    """
    path = _join_path(save_dir, f"agent_{step}.msgpack")
    print(f"Saving checkpoint to {path}...")
    write_bytes(path, agent.state_bytes())


def train(
    num_digits,
    hand_length,
    num_players,
    num_phases,
    rnad_steps_per_phase,
    br_collect_per_phase,
    br_eval_episodes,
    eval_every_phases,
    checkpoint_every_phases,
    rnad_config,
    ppo_br_config,
    load_checkpoint=None,
    save_dir="checkpoints",
    checkpoint_eval_episodes=4096,
    network_type="mlp",
    transformer_config=None,
    br_state_pool_size=0,
    br_state_ratio=0.0,
    br_reinit_every=0,
    agent_score_window=20,
    history_encoding="sparse",
    br_network_type=None,
    br_layers=None,
    br_transformer_config=None,
    max_jump=None,
    dummy_mult=1,
    dummy_mode="random",
    factored_count_encoding="linear",
    br_factored_count_encoding=None,
):
    """Interleaved RNaD + PPO best-response training loop.

    ``br_network_type`` / ``br_layers`` (when non-None) give the interleaved
    best-response its OWN trainable architecture, decoupled from the frozen
    main agent (whose arch is ``network_type`` and drives the BR's replay
    net). When both are None the BR is architecturally identical to the main
    agent — exactly the legacy coupled behaviour (byte-identical). Typical
    use: ``network_type='lpt'`` (transformer main) + ``br_network_type='mlp'``
    (fast, honest plain-MLP interleaved BR).
    """
    jax_config = LiarsPokerConfig(
        num_players=num_players,
        hand_length=hand_length,
        num_digits=num_digits,
        max_jump=max_jump,
        dummy_mult=dummy_mult,
        dummy_mode=dummy_mode,
    )

    agent = JaxLiarsPokerRNaDSolver(
        _rnad_config_with_nerd_beta(rnad_config),
        jax_config,
        network_type=network_type,
        transformer_config=transformer_config,
        br_state_pool_size=br_state_pool_size,
        br_state_ratio=br_state_ratio,
        history_encoding=history_encoding,
        factored_count_encoding=factored_count_encoding,
    )
    if load_checkpoint is not None:
        agent.restore_state(read_bytes(load_checkpoint))

    br_agents = build_ppo_br_agents(
        jax_config,
        ppo_br_config,
        seed=rnad_config.get("seed", 42),
        network_type=network_type,
        transformer_config=transformer_config,
        history_encoding=history_encoding,
        br_network_type=br_network_type,
        br_layers=br_layers,
        br_transformer_config=br_transformer_config,
        factored_count_encoding=factored_count_encoding,
        br_factored_count_encoding=br_factored_count_encoding,
    )
    rng_key = jax.random.PRNGKey(rnad_config.get("seed", 42))

    # The RNaD state pool is only fed (and hence only worth collecting BR states
    # for) when it is sized > 0, actually sampled from (br_state_ratio > 0), and
    # not bypassed by the grpo advantage estimator (see collect_batch_trajectory).
    advantage_estimator = rnad_config.get("advantage_estimator", "vtrace")
    pool_active = (
        br_state_pool_size > 0
        and br_state_ratio > 0
        and advantage_estimator != "grpo"
    )

    global_step = agent.learner_steps
    prev_checkpoint_params = None  # params_target saved at the previous checkpoint
    # Rolling buffer of recent br/agent_score values. Because a trained BR gives
    # only a *lower bound* on exploitability, the single last-eval value is a
    # noisy underestimate — especially right after a periodic BR reinit, when the
    # BR is undertrained and agent_score spuriously rises. The windowed *min*
    # (most-exploiting BR seen recently) is the honest exploitability estimate.
    agent_score_history = []

    for phase in trange(num_phases, desc="phases"):
        # ---- RNaD phase ------------------------------------------------
        # A3: ``agent.step()`` returns ``logs["loss"]`` as a DEVICE scalar (the
        # jitted ``update_parameters`` output). Calling ``float()`` on it every
        # step forces a host sync that blocks JAX async dispatch. Instead we
        # append the device scalars, run the whole phase, then do ONE
        # ``jax.device_get`` and replay the per-step ``wandb.log`` with the SAME
        # per-step ``step`` values as before (charts unchanged). ``logs``'s other
        # keys (``actor_steps`` / ``learner_steps``) are Python ints — no sync.
        rnad_loss_devs = []
        rnad_step_start = global_step
        for _ in range(rnad_steps_per_phase):
            logs = agent.step()
            rnad_loss_devs.append(logs["loss"])  # device scalar; no host sync
            global_step += 1
        # Single host pull for the whole phase's losses.
        rnad_losses = [float(l) for l in jax.device_get(rnad_loss_devs)]
        for i, loss in enumerate(rnad_losses):
            wandb.log({"rnad/loss": loss}, step=rnad_step_start + i)

        # ---- Evaluation (with BR training immediately before) ----------
        if (phase + 1) % eval_every_phases == 0:
            if br_reinit_every > 0 and (phase + 1) % br_reinit_every == 0:
                for br_agent in br_agents:
                    br_agent.reinit_params()
            for br_agent in br_agents:
                rng_key, br_key = jax.random.split(rng_key)
                _, br_states = br_agent.train_phase(
                    main_params=agent.params_target,
                    rng_key=br_key,
                    n_collect=br_collect_per_phase,
                    collect_states=pool_active,
                )
                if pool_active and br_states is not None:
                    agent.add_states_to_pool(br_states)

            rng_key, eval_key = jax.random.split(rng_key)
            eval_logs = eval_br_vs_main(
                jax_config, agent, br_agents, eval_key, br_eval_episodes
            )
            mean_loss = float(np.mean(rnad_losses))
            if "br/agent_score" in eval_logs:
                agent_score_history.append(eval_logs["br/agent_score"])
                window = agent_score_history[-agent_score_window:]
                # min == strongest exploit found in the window == honest
                # (tightest) lower bound on how exploitable the agent is.
                eval_logs["br/agent_score_windowed_min"] = float(np.min(window))
                eval_logs["br/agent_score_windowed_mean"] = float(np.mean(window))
            wandb.log(
                {"rnad/mean_loss": mean_loss, **eval_logs}, step=global_step
            )
            br_str = "  ".join(f"{k}={v:.3f}" for k, v in eval_logs.items())
            print(f"Phase {phase + 1:4d}  rnad_loss={mean_loss:.4f}  {br_str}")

        # ---- Checkpoint ------------------------------------------------
        if (phase + 1) % checkpoint_every_phases == 0:
            if prev_checkpoint_params is not None:
                rng_key, ckpt_eval_key = jax.random.split(rng_key)
                ckpt_eval_logs = eval_current_vs_prev(
                    jax_config, agent, prev_checkpoint_params,
                    ckpt_eval_key, checkpoint_eval_episodes,
                )
                wandb.log(ckpt_eval_logs, step=global_step)
                ckpt_str = "  ".join(f"{k}={v:.3f}" for k, v in ckpt_eval_logs.items())
                print(f"  ckpt_eval: {ckpt_str}")
            prev_checkpoint_params = agent.params_target
            checkpoint(agent, save_dir, global_step)

    if prev_checkpoint_params is not None:
        rng_key, ckpt_eval_key = jax.random.split(rng_key)
        ckpt_eval_logs = eval_current_vs_prev(
            jax_config, agent, prev_checkpoint_params,
            ckpt_eval_key, checkpoint_eval_episodes,
        )
        wandb.log(ckpt_eval_logs, step=global_step)
        ckpt_str = "  ".join(f"{k}={v:.3f}" for k, v in ckpt_eval_logs.items())
        print(f"  ckpt_eval (final): {ckpt_str}")
    checkpoint(agent, save_dir, global_step)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="conf", config_name="train_jax", version_base=None)
def main(cfg: DictConfig) -> None:
    # A6: persistent XLA compilation cache. Set BEFORE any jit compilation so the
    # first run populates the cache and subsequent runs reuse compiled
    # executables (cuts repeated warm-up recompiles). Guarded in case a jax
    # build lacks the option (jax 0.10 has both).
    try:
        jax.config.update(
            "jax_compilation_cache_dir",
            os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache"),
        )
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception as e:  # pragma: no cover - version-dependent
        print(f"Persistent compilation cache unavailable: {e}")

    job_id = os.environ.get("JOB_ID", "local")
    run_name = HydraConfig.get().job.override_dirname or job_id
    print("Job ID:", job_id)
    print("JAX devices:", jax.devices())

    wandb.init(
        project="liars-poker__train-rnad-jax",
        entity="columbia_gsb",
        config=OmegaConf.to_container(cfg, resolve=True),
        # name=f"{run_name}-{job_id}",
    )

    transformer_config = None
    if cfg.get("transformer_config"):
        transformer_config = OmegaConf.to_container(
            cfg.transformer_config, resolve=True
        )

    # Decoupled interleaved BR: an INDEPENDENT trainable arch for the BR net,
    # while the frozen main agent (network_type) drives the BR's replay net.
    # Both null => coupled/legacy behaviour (BR arch == main arch), exactly
    # backward-compatible. ``br_layers`` (a Hydra list) is materialised to a
    # plain list; when null it defaults to the main agent's layers downstream.
    br_network_type = cfg.get("br_network_type", None)
    br_layers = cfg.get("br_layers", None)
    if br_layers is not None:
        br_layers = OmegaConf.to_container(br_layers, resolve=True)
    br_transformer_config = None
    if cfg.get("br_transformer_config"):
        br_transformer_config = OmegaConf.to_container(
            cfg.br_transformer_config, resolve=True
        )

    train(
        num_digits=min(cfg.num_digits, 10),
        hand_length=min(cfg.hand_length, 10),
        num_players=min(cfg.num_players, 10),
        num_phases=cfg.num_phases,
        rnad_steps_per_phase=cfg.rnad_steps_per_phase,
        br_collect_per_phase=cfg.br_collect_per_phase,
        br_eval_episodes=cfg.br_eval_episodes,
        eval_every_phases=cfg.eval_every_phases,
        checkpoint_every_phases=cfg.checkpoint_every_phases,
        rnad_config=OmegaConf.to_container(cfg.rnad_config, resolve=True),
        ppo_br_config=OmegaConf.to_container(cfg.ppo_br_config, resolve=True),
        load_checkpoint=cfg.load_checkpoint,
        save_dir=cfg.save_dir,
        checkpoint_eval_episodes=cfg.get("checkpoint_eval_episodes", 4096),
        network_type=cfg.get("network_type", "mlp"),
        transformer_config=transformer_config,
        br_state_pool_size=cfg.get("br_state_pool_size", 0),
        br_state_ratio=cfg.get("br_state_ratio", 0.0),
        br_reinit_every=cfg.get("br_reinit_every", 0),
        agent_score_window=cfg.get("agent_score_window", 20),
        history_encoding=cfg.get("history_encoding", "sparse"),
        br_network_type=br_network_type,
        br_layers=br_layers,
        br_transformer_config=br_transformer_config,
        max_jump=cfg.get("max_jump", None),
        dummy_mult=int(cfg.get("dummy_mult", 1)),
        dummy_mode=str(cfg.get("dummy_mode", "random")),
        factored_count_encoding=cfg.get("factored_count_encoding", "linear"),
        br_factored_count_encoding=cfg.get("br_factored_count_encoding", None),
    )


if __name__ == "__main__":
    main()
