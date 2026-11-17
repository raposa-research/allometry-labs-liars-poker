"""Bids / challenges per round in SELF-PLAY, with the CORRECT terminal rule.

A challenge does NOT end the hand. liars_poker_jax counts the pot only when
num_challenges reaches num_players (non-rebid) or num_players-1 (rebid), and
`new_num_challenges_final` RESETS the counter to 0 on any bid. So a hand ends
only on a RUN of consecutive challenges long enough to close it out, and a
single challenge mid-hand just passes.

The previous script treated the first challenge as terminal, which measured
"plies until someone first declines to raise" -- not ladder depth. This one
uses the engine's own terminal flag and classifies every ply as bid/challenge.

Per hand it reports: plies, bids, challenges, bidders per round (round r =
plies [r*P,(r+1)*P)), and the length of the final challenge run.
"""
import collections, json, os, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax, jax.numpy as jnp, numpy as np
from policy_entropy import load_agent
from liars_poker_jax import (new_initial_state, apply_action, current_player_id,
                             is_terminal, legal_actions_mask, observation_tensor)
from rnad import EnvStep
from ckpt_root import require  # checkpoint location; see REPRODUCE.md section 3

RUN = sys.argv[1] if len(sys.argv) > 1 else "rnad-ppo-vtrace-8x10x5-ent-eta0"
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
MAXT = int(os.environ.get("BPR_MAXT", 400))
CHALLENGE = 0


def deal(cfg, rng, n):
    st = jax.vmap(lambda _: new_initial_state(cfg))(jnp.zeros(n, jnp.int32))
    for _ in range(cfg.total_cards):
        rng, sub = jax.random.split(rng)
        d = jax.random.randint(sub, (n,), 0, cfg.num_digits, dtype=jnp.int32)
        st = jax.vmap(lambda s, a: apply_action(s, cfg, a))(st, d)
    return st, rng


def hist(x, label, width=70):
    c = collections.Counter(x)
    n = len(x)
    print(f"\n--- {label} ---")
    print(f"  mean {np.mean(x):.3f}  median {np.median(x):.0f}  sd {np.std(x):.3f}"
          f"  min {min(x)}  max {max(x)}")
    cum = 0
    for v in sorted(c):
        cum += c[v]
        if c[v] / n < 0.002 and v not in (min(x), max(x)):
            continue
        print(f"{v:>6}{c[v]:>8}{100*c[v]/n:>8.2f}%{100*cum/n:>9.2f}%")
    return c


def main():
    sol = load_agent(require(RUN, STEP), batch_size=256,
                     trajectory_max=30)
    cfg = sol._jax_config
    henc = sol._history_encoding
    P = cfg.num_players
    print(f"{RUN} @ {STEP}: {cfg.hand_length}x{cfg.num_digits}x{P}, "
          f"{GAMES} hands, MAXT={MAXT}", flush=True)

    rng = jax.random.PRNGKey(0)
    states, rng = deal(cfg, rng, GAMES)
    done = jax.vmap(is_terminal)(states)
    A, W, V = [], [], []
    for _ in range(MAXT):
        cur = jax.vmap(lambda s: current_player_id(s, cfg))(states)
        obs = jax.vmap(lambda s: observation_tensor(
            s, cfg, s.current_player, history_encoding=henc))(states)
        legal = jax.vmap(lambda s: legal_actions_mask(s, cfg))(states)
        env = EnvStep(obs=obs.astype(jnp.float32), legal=legal.astype(jnp.int8),
                      player_id=cur.astype(jnp.float32), valid=(~done),
                      rewards=jnp.zeros((GAMES, P)))
        pi, _, _, _ = jax.vmap(sol.network.apply, (None, 0), 0)(sol.params, env)
        rng, sub = jax.random.split(rng)
        g = jax.random.gumbel(sub, pi.shape, dtype=jnp.float32)
        a = jnp.argmax(jnp.log(pi.astype(jnp.float32) + 1e-45) + g, -1)
        A.append(np.asarray(a, np.int32)); W.append(np.asarray(cur, np.int32))
        V.append(np.asarray(~done))
        states = jax.vmap(lambda s, x: apply_action(s, cfg, x))(states, a.astype(jnp.int32))
        done = jax.vmap(is_terminal)(states)
        if bool(jnp.all(done)):
            break
    A = np.stack(A); W = np.stack(W); V = np.stack(V)
    finished = np.asarray(done)

    plies, nbids, nchal, tailrun, distinct = [], [], [], [], []
    per_round = collections.defaultdict(list)
    unfinished = int((~finished).sum())
    for b in range(GAMES):
        if not finished[b]:
            continue
        idx = np.where(V[:, b])[0]
        acts = A[idx, b]; who = W[idx, b]
        T = len(acts)
        isb = acts != CHALLENGE
        plies.append(T); nbids.append(int(isb.sum()))
        nchal.append(int((~isb).sum()))
        r = 0
        while r < T and acts[T - 1 - r] == CHALLENGE:
            r += 1
        tailrun.append(r)
        distinct.append(len(set(who[isb].tolist())))
        for rr in range((T + P - 1) // P):
            lo, hi = rr * P, min((rr + 1) * P, T)
            per_round[rr].append(int(isb[lo:hi].sum()))

    N = len(plies)
    print(f"\nfinished {N}/{GAMES}   unfinished {unfinished}")
    hist(plies, "PLIES per hand (true episode length)")
    hist(nbids, "BIDS per hand (ladder depth)")
    hist(nchal, "CHALLENGES per hand")
    hist(tailrun, f"final consecutive-challenge run (closes the hand; needs {P} or {P-1})")
    hist(distinct, f"DISTINCT players who bid (max {P})")

    print(f"\n--- BIDS per ROUND (round r = plies [r*{P},(r+1)*{P}) ) ---")
    print(f"{'round':>6}{'hands reaching':>16}{'mean bids':>11}{'mean chal':>11}{'pct full bids':>15}")
    for r in sorted(per_round):
        v = np.array(per_round[r])
        if len(v) / N < 0.001:
            continue
        print(f"{r+1:>6}{len(v):>16}{v.mean():>11.3f}{P - v.mean():>11.3f}"
              f"{100*(v == P).mean():>14.1f}%")

    out = {"run": RUN, "step": STEP, "players": int(P), "games": GAMES,
           "finished": N, "unfinished": unfinished,
           "mean_plies": float(np.mean(plies)), "mean_bids": float(np.mean(nbids)),
           "mean_challenges": float(np.mean(nchal)),
           "mean_distinct_bidders": float(np.mean(distinct)),
           "plies_per_hand": {str(k): int(v) for k, v in sorted(collections.Counter(plies).items())},
           "bids_per_hand": {str(k): int(v) for k, v in sorted(collections.Counter(nbids).items())},
           "challenges_per_hand": {str(k): int(v) for k, v in sorted(collections.Counter(nchal).items())},
           "final_challenge_run": {str(k): int(v) for k, v in sorted(collections.Counter(tailrun).items())},
           "distinct_bidders": {str(k): int(v) for k, v in sorted(collections.Counter(distinct).items())},
           "bids_per_round": {str(r+1): {"hands": len(per_round[r]),
                                         "mean_bids": float(np.mean(per_round[r])),
                                         "pct_full": float(100*np.mean(np.array(per_round[r]) == P))}
                              for r in sorted(per_round)}}
    pathlib.Path(__file__).with_name(os.environ.get("BPR_OUT", "bpr2.json")).write_text(
        json.dumps(out, indent=1))
    print("\nwrote", os.environ.get("BPR_OUT", "bpr2.json"))


if __name__ == "__main__":
    main()
