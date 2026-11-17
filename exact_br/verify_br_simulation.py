"""Empirically checks the exact best-response values by playing the BR policy.

Runs the exact computation (`lpeas`, keeping the best-response policy), then
plays the tabular best response against the checkpoint in OpenSpiel's own
`python_liars_poker` (its rules, payoffs and information-state tensors, sampled
deals). The empirical mean reward of the best responder should match
`br_value[br_player]` to within a few standard errors.

    .venv/bin/python exact_br/verify_br_simulation.py agent.pickle \
        --sibling-src src --games 1000000
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_liars_poker_checkpoint import add_common_args, build_traverser, load_checkpoint_policy, summarize  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--games", type=int, default=1_000_000, help="games per best-responder seat")
    p.add_argument("--sim-batch", type=int, default=20_000, help="games simulated in lock-step")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def hand_index(digits, num_digits):
    # matches lp::Traverser::hand_digit: slot j holds (h // D^j) % D + 1
    return sum((d - 1) * num_digits**j for j, d in enumerate(digits))


def simulate(game, t, cfg, policy_fn, br_player, num_games, batch, rng):
    """Mean reward (and its standard error) of the exact BR vs the checkpoint."""
    rewards = []
    checked_tensors = 0
    for start in range(0, num_games, batch):
        G = min(batch, num_games - start)
        states = [game.new_initial_state() for _ in range(G)]
        hands = np.zeros((G, 2), dtype=np.int64)
        for s in range(G):
            st = states[s]
            dealt = [[], []]
            step = 0
            while st.is_chance_node():
                outcomes, probs = zip(*st.chance_outcomes())
                a = outcomes[rng.choice(len(outcomes), p=probs)]
                dealt[step % 2].append(a)  # cards are dealt to players alternately
                st.apply_action(a)
                step += 1
            hands[s] = [hand_index(dealt[0], cfg.num_digits), hand_index(dealt[1], cfg.num_digits)]
        idx = np.zeros(G, dtype=np.int64)
        active = np.arange(G)
        level = 0
        while active.size:
            p = t.level_player(level)
            if p == br_player:
                actions = t.br_actions(level, idx[active], hands[active, p])
            else:
                obs = np.stack([states[s].information_state_tensor(p) for s in active]).astype(np.uint8)
                legal = np.zeros((active.size, t.num_actions), dtype=bool)
                for r, s in enumerate(active):
                    legal[r, states[s].legal_actions()] = True
                # the opponent's network sees OpenSpiel's tensor; make sure it is
                # the one lpeas evaluated at this node
                if checked_tensors < 1000:
                    for r, s in enumerate(active[:5]):
                        ti, tl = t.node_infostate(level, int(idx[s]), int(hands[s, p]))
                        assert np.array_equal(ti, obs[r]) and np.array_equal(tl, legal[r]), (level, idx[s])
                        checked_tensors += 1
                probs = policy_fn(p, obs, legal)
                # a trailing column is the forfeit mass (dummy_mode="lose"):
                # sampling it ends the game with the opponent losing
                lose_col = probs.shape[1] == t.num_actions + 1
                legal_ext = np.concatenate([legal, np.ones((active.size, 1), bool)], axis=1) if lose_col else legal
                probs = np.where(legal_ext, probs, 0.0)
                probs /= probs.sum(axis=1, keepdims=True)
                u = rng.random(active.size)
                actions = (probs.cumsum(axis=1) < u[:, None]).sum(axis=1)
                actions = np.minimum(actions, probs.shape[1] - 1)
            forfeit = actions == t.num_actions
            nxt = t.child_indices(level, idx[active], np.where(forfeit, 0, actions))
            still = []
            for r, s in enumerate(active):
                if forfeit[r]:
                    rewards.append(1.0)
                    continue
                states[s].apply_action(int(actions[r]))
                if nxt[r] < 0:
                    assert states[s].is_terminal()
                    rewards.append(states[s].returns()[br_player])
                else:
                    assert not states[s].is_terminal()
                    idx[s] = nxt[r]
                    still.append(s)
            active = np.array(still, dtype=np.int64)
            level += 1
        print(f"  seat {br_player}: {len(rewards):,} games, mean={np.mean(rewards):.4f}", flush=True)
    rewards = np.asarray(rewards)
    return rewards.mean(), rewards.std(ddof=1) / np.sqrt(rewards.size)


def main():
    args = parse_args()
    import lpeas
    import pyspiel
    import open_spiel.python.games  # noqa: F401

    agent, cfg, policy_fn = load_checkpoint_policy(
        args.checkpoint, args.sibling_src, args.finetune, args.batch_size, args.hand_encoding, args.dummy_mode
    )
    t = build_traverser(cfg)
    t0 = time.time()
    res = lpeas.ev_and_exploitability(t, policy_fn, chunk_nodes=args.chunk_nodes, store_br=True)
    summary = summarize(args, cfg, t, res, time.time() - t0)
    print(json.dumps(summary, indent=2))

    game = pyspiel.load_game(
        f"python_liars_poker(num_digits={cfg.num_digits},hand_length={cfg.hand_length},players=2)"
    )
    rng = np.random.default_rng(args.seed)
    summary["simulation"] = {}
    for br_player in (0, 1):
        t1 = time.time()
        mean, se = simulate(game, t, cfg, policy_fn, br_player, args.games, args.sim_batch, rng)
        exact = res["br_value"][br_player]
        z = (mean - exact) / se
        print(f"BR as player {br_player}: empirical {mean:.4f} +- {se:.4f} (SE), exact {exact:.4f}, "
              f"z = {z:+.2f}  [{time.time() - t1:.0f}s]")
        summary["simulation"][f"br_player_{br_player}"] = {
            "games": args.games, "mean": float(mean), "se": float(se), "exact": exact, "z": float(z),
        }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
