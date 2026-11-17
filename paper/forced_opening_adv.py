"""Per-hand advantage of EVERY opening bid, with the opening FORCED.

Conditioning on the policy's own openings cannot answer this: the actions of
interest have pi ~ 0 and are never sampled. So the opening is overridden to
each legal bid in turn, and the rest of the episode is played by the policy on
both seats (self-play), giving

    Q(h, a)  = opener's expected return | hand h, forced opening a
    V(h)     = sum_a pi(a|h) Q(h, a)
    A(a|h)   = Q(h, a) - V(h)

The mechanism under test (boundary freezing): the PG-no-bonus policy should
have a cluster of bids with A > 0 but pi ~ 0 -- positive-advantage actions it
can no longer reach, because the gradient toward them scales as pi. NeuRD at
comparable exploitability should have no such cluster. PG+bonus should also
have few, but mechanically, because the entropy floor keeps pi off zero.

Returns are per-episode from the opener's seat, with exact hand grouping. The
same dealt hands are reused across all forced actions (common random numbers),
so Q differences within a hand are paired and much less noisy than independent
sampling would give.
"""
import collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax, jax.numpy as jnp, numpy as np
from policy_entropy import load_agent
from liars_poker_jax import (new_initial_state, apply_action, current_player_id,
                             is_terminal, legal_actions_mask, observation_tensor,
                             returns as jax_returns)
from rnad import EnvStep
from ckpt_root import require  # checkpoint location; see REPRODUCE.md section 3

import os
GAMES = int(os.environ.get("FOA_GAMES", 3072))
MAXT = 40
ARMS = [("PG, no bonus", "x1-pg-noent"),
        ("PG + bonus", "dummy-3x3x2-ppoent-x1"),
        ("NeuRD (beta=2)", "dummy-3x3x2-neurd-x1")]


def deal(cfg, rng, n):
    st = jax.vmap(lambda _: new_initial_state(cfg))(jnp.zeros(n, jnp.int32))
    for _ in range(cfg.total_cards):
        rng, sub = jax.random.split(rng)
        d = jax.random.randint(sub, (n,), 0, cfg.num_digits, dtype=jnp.int32)
        st = jax.vmap(lambda s, a: apply_action(s, cfg, a))(st, d)
    return st, rng


def rollout(sol, cfg, states, rng, hist):
    """Play both seats with the policy until terminal; return [n, P] rewards."""
    n = states.hands.shape[0]
    done = jax.vmap(is_terminal)(states)
    final = jnp.zeros((n, cfg.num_players))
    for _ in range(MAXT):
        cur = jax.vmap(lambda s: current_player_id(s, cfg))(states)
        obs = jax.vmap(lambda s: observation_tensor(
            s, cfg, s.current_player, history_encoding=hist))(states)
        legal = jax.vmap(lambda s: legal_actions_mask(s, cfg))(states)
        env = EnvStep(obs=obs.astype(jnp.float32), legal=legal.astype(jnp.int8),
                      player_id=cur.astype(jnp.float32),
                      valid=(~done), rewards=jnp.zeros((n, cfg.num_players)))
        pi, _, _, _ = jax.vmap(sol.network.apply, (None, 0), 0)(sol.params, env)
        rng, sub = jax.random.split(rng)
        g = jax.random.gumbel(sub, pi.shape, dtype=jnp.float32)
        act = jnp.argmax(jnp.log(pi.astype(jnp.float32) + 1e-45) + g, -1).astype(jnp.int32)
        new = jax.vmap(lambda s, a: apply_action(s, cfg, a))(states, act)
        nd = jax.vmap(is_terminal)(new)
        just = nd & ~done
        ret = jax.vmap(lambda s: jax_returns(s, cfg))(new)
        final = jnp.where(just[:, None], ret, final)
        states, done = new, nd
        if bool(jnp.all(done)):
            break
    return np.asarray(final), np.asarray(done), rng


def run(run_name, step=2000000):
    sol = load_agent(require(run_name, step),
                     batch_size=256, trajectory_max=30)
    cfg = sol._jax_config
    hist = sol._history_encoding
    rng = jax.random.PRNGKey(0)
    states0, rng = deal(cfg, rng, GAMES)
    opener = np.asarray(jax.vmap(lambda s: current_player_id(s, cfg))(states0))
    hands = np.asarray(states0.hands)                       # [n, P, hand_len]
    hist_key = [tuple(int((hands[i, opener[i]] == d + 1).sum())
                      for d in range(cfg.num_digits)) for i in range(GAMES)]
    legal0 = np.asarray(jax.vmap(lambda s: legal_actions_mask(s, cfg))(states0))

    # policy at the opening (constant per hand)
    obs0 = jax.vmap(lambda s: observation_tensor(
        s, cfg, s.current_player, history_encoding=hist))(states0)
    env0 = EnvStep(obs=obs0.astype(jnp.float32), legal=jnp.asarray(legal0, jnp.int8),
                   player_id=jnp.asarray(opener, jnp.float32),
                   valid=jnp.ones(GAMES, bool),
                   rewards=jnp.zeros((GAMES, cfg.num_players)))
    pi0, _, _, _ = jax.vmap(sol.network.apply, (None, 0), 0)(sol.params, env0)
    pi0 = np.asarray(pi0, np.float64)

    acts = [a for a in range(cfg.num_actions) if legal0[:, a].mean() > 0.99]
    Q = collections.defaultdict(dict)
    for a in acts:
        st = jax.vmap(lambda s: apply_action(s, cfg, jnp.int32(a)))(states0)
        rew, done, rng = rollout(sol, cfg, st, rng, hist)
        for i in range(GAMES):
            if not done[i]:
                continue
            Q[hist_key[i]].setdefault(a, []).append(float(rew[i, opener[i]]))
    out = {}
    for h, per_a in Q.items():
        idx = [i for i in range(GAMES) if hist_key[i] == h]
        p = pi0[idx[0]]
        qs = {a: float(np.mean(v)) for a, v in per_a.items() if len(v) >= 30}
        se = {a: float(np.std(per_a[a], ddof=1) / np.sqrt(len(per_a[a])))
              for a in qs}
        n = {a: int(len(per_a[a])) for a in qs}
        V = sum(p[a] * q for a, q in qs.items())
        # SE of V treating the forced-action rollouts as independent. They
        # actually share the dealt hands (common random numbers), which makes
        # them positively correlated, so this OVERSTATES se(V) a little.
        V_se = float(np.sqrt(sum((p[a] * se[a]) ** 2 for a in qs)))
        out[str(h)] = {"pi": {a: float(p[a]) for a in qs},
                       "Q": qs, "Q_se": se, "n": n,
                       "A": {a: q - V for a, q in qs.items()},
                       "V": V, "V_se": V_se, "n_hand": len(idx)}
    return out, cfg


if __name__ == "__main__":
    allout = {}
    for label, run_name in ARMS:
        o, cfg = run(run_name)
        allout[label] = o
        pts = [(d["pi"][a], d["A"][a]) for d in o.values() for a in d["A"]]
        pi = np.array([p for p, _ in pts]); A = np.array([x for _, x in pts])
        lo = pi < 0.01
        print(f"\n=== {label}   ({len(pts)} (hand,action) cells)")
        print(f"   cells with pi < 0.01 : {lo.sum():3d}   of these A>0: "
              f"{int((A[lo] > 0).sum()):3d}   A>+0.05: {int((A[lo] > 0.05).sum()):3d}")
        if lo.sum():
            print(f"   among pi<0.01:  max A = {A[lo].max():+.3f}   "
                  f"mean A = {A[lo].mean():+.3f}")
        hi = ~lo
        if hi.sum():
            print(f"   among pi>=0.01: max A = {A[hi].max():+.3f}   "
                  f"mean A = {A[hi].mean():+.3f}")
    pathlib.Path(__file__).with_name(os.environ.get("FOA_OUT","forced_opening_adv.json")).write_text(
        json.dumps(allout, indent=1))
    print("\nwrote", os.environ.get("FOA_OUT","forced_opening_adv.json"))
