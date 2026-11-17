#!/usr/bin/env python
"""High-precision exploitability: replay a trained BR checkpoint against its agent.

train_br.py's in-training `br/agent_score` is measured on a few thousand
episodes per seat, so its standard error (~0.016 at 4096 episodes) is the same
size as the effect we care about once an agent is near-converged -- every
converged checkpoint reads as "0.00 +/- noise". This script reloads the SAVED
BR policy and replays it against the frozen agent for as many episodes as you
like, which is far cheaper than more BR training and shrinks the error bar as
1/sqrt(N).

It is a pure evaluation: the BR is frozen too, so this measures the
exploitability that BR *found*, with the training noise removed. It cannot
exceed what the BR learned -- it is a better estimate of the same quantity, not
a stronger attack.

The BR observes [base_obs, exploitee-policy features] (see OppAwareBRNetwork),
which is why this cannot go through eval_agents.py: the rollout has to compute
the exploitee's policy at the BR's information state. JaxPPOBRAgent._collect
already does exactly that, so we reuse it.

Usage
-----
    python eval_br_vs_agent.py \
        --agent s3://.../rnad-ppo-vtrace-8x10x3-ownreg/agent_1900000.msgpack \
        --br-dir s3://.../br-oppfeat-vtrace-8x10x3-ownreg-1900k/ \
        --step 1000 --episodes 100000
"""

import argparse
import os
import sys

import flax
import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rnad import RNaDConfig, FineTuning  # noqa: E402
from train_br import (  # noqa: E402
    read_bytes,
    read_checkpoint_metadata,
    jax_config_from_meta,
    _is_s3,
)
from train_jax import JaxLiarsPokerRNaDSolver, JaxPPOBRAgent  # noqa: E402


def _as_str(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _layers(pnl) -> list:
    """policy_network_layers survives msgpack as either a list or {"0": v0,...}."""
    if isinstance(pnl, dict):
        return [int(pnl[str(i)]) for i in range(len(pnl))]
    return [int(x) for x in pnl]


def load_br_for_agent(br_bytes: bytes, main_meta: dict, jax_config,
                      opp_threshold: float = 0.03,
                      opp_discretization: int = 32,
                      br_layers_override=None,
                      br_network_type_override=None):
    """Rebuild a JaxPPOBRAgent able to play against the agent in `main_meta`.

    The BR checkpoint records its OWN architecture (br_layers / br_network_type)
    and its exploitee-feature settings, but not the frozen agent's -- the BR's
    replay_network has to match the agent it was trained against, so those come
    from the agent checkpoint.
    """
    meta = flax.serialization.msgpack_restore(br_bytes)
    # Build the kwargs, then DROP any this tree's JaxPPOBRAgent does not
    # accept. The opp_features family exists only in the serve-mlp worktree;
    # `wt` -- the only tree whose game supports dummy_mult/dummy_mode, and so
    # the only one that can load a dummy-probe checkpoint -- has no
    # opp-features code at all and raises TypeError on those kwargs. Filtering
    # by signature keeps one script working in both trees.
    import inspect
    kwargs = dict(
        jax_config=jax_config,
        br_player=int(meta["br_player"]),
        # Frozen agent's architecture -> drives replay_network + observation.
        network_layers=main_meta["policy_network_layers"],
        network_type=main_meta["network_type"],
        history_encoding=main_meta["history_encoding"],
        # The BR's own trainable net. NOTE train_br's save_br_checkpoints
        # writes the FROZEN AGENT's layers here, not the BR's --br-layers, so
        # this field is wrong whenever they differ; --br-layers overrides it.
        br_network_type=(br_network_type_override if br_network_type_override
                         else _as_str(meta.get("network_type", "mlp"))),
        br_layers=(br_layers_override if br_layers_override
                   else _layers(meta["policy_network_layers"])),
        opp_features=_as_str(meta.get("opp_features", "none")),
        opp_logit_residual=(
            bool(int(meta["opp_logit_residual"]))
            if "opp_logit_residual" in meta else None
        ),
        opp_gate_init=float(meta.get("opp_gate_init", 0.01)),
        # MUST MATCH the filtering the BR was trained against: the BR rollout
        # post-processes the frozen agent's policy with this every step.
        finetune=FineTuning(from_learner_steps=0,
                            policy_threshold=opp_threshold,
                            policy_discretization=opp_discretization),
    )
    accepted = set(inspect.signature(JaxPPOBRAgent.__init__).parameters)
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        print(f"  (this tree's JaxPPOBRAgent does not take {dropped}; dropping)")
    # Restore the BR's baked-in policy temperature. Omitting it would rebuild
    # the net at T=1 and evaluate a SHARPER policy than the one trained, which
    # silently reports the wrong agent's exploitability. Absent in every
    # pre-temperature checkpoint, where 1.0 is correct.
    _t = float(meta.get("br_temperature", 1.0))
    if _t != 1.0:
        kwargs["br_temperature"] = _t
    agent = JaxPPOBRAgent(**{k: v for k, v in kwargs.items() if k in accepted})
    if _t != 1.0:
        print(f"  BR temperature: {_t:g} (restored from checkpoint)")
    restored = flax.serialization.from_bytes({"params": agent.params}, br_bytes)
    agent.params = restored["params"]
    return agent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, help="Frozen agent checkpoint (the exploitee).")
    p.add_argument("--br-dir", required=True, help="Directory holding br_p<N>_<step>.msgpack.")
    p.add_argument("--step", type=int, required=True, help="BR cycle number to load.")
    p.add_argument("--episodes", type=int, default=100_000, help="Episodes per seat.")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--trajectory-max", type=int, default=40,
                   help="Must exceed the longest game; unfinished episodes are reported.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outcome-stats", action="store_true",
                   help="Also report win/loss percentages and the BR payoff "
                        "histogram. Average equity alone does not pin down the "
                        "win rate once payoffs are not +-1 (the 3-player game "
                        "pays the bidder +-2 and challengers -+1).")
    p.add_argument("--br-mask-dummy", action="store_true",
                   help="Hard-mask dummy actions out of the BR's action "
                        "space during replay. MUST match how the BR was "
                        "trained, or the policy it plays differs from the "
                        "one it learned.")
    p.add_argument("--br-network-type", default=None, metavar="TYPE",
                   help="Override the BR's own architecture (mlp / "
                        "factored_mlp / transformer). Needed only for "
                        "checkpoints written before train_br recorded it; "
                        "otherwise it is read from the checkpoint.")
    p.add_argument("--br-layers", nargs="+", type=int, default=None,
                   metavar="H",
                   help="Override the BR net width. Needed when the "
                        "BR layers differed from the frozen agent's, "
                        "because train_br stores the AGENT layers in "
                        "the BR checkpoint.")
    p.add_argument("--opp-threshold", type=float, default=0.03, metavar="P",
                   help="Frozen-agent policy threshold; MUST match what "
                        "the BR was trained with (default 0.03).")
    p.add_argument("--opp-discretization", type=int, default=32, metavar="D",
                   help="Frozen-agent probability discretization; MUST "
                        "match training (default 32; <=0 disables).")
    return p.parse_args()


def main():
    args = parse_args()

    main_meta = read_checkpoint_metadata(args.agent)
    jax_config = jax_config_from_meta(main_meta["jax_config"])
    print(f"agent      : {args.agent}")
    print(f"  config   : {main_meta['jax_config']}")
    print(f"  network  : {main_meta['network_type']}/{main_meta['history_encoding']} "
          f"{main_meta['policy_network_layers']}")

    solver = JaxLiarsPokerRNaDSolver(
        RNaDConfig(game_name="unused",
                   policy_network_layers=main_meta["policy_network_layers"]),
        jax_config,
        network_type=main_meta["network_type"],
        history_encoding=main_meta["history_encoding"],
    )
    solver.restore_params_only(read_bytes(args.agent))
    print(f"  steps    : {solver.learner_steps}")
    main_params = solver.params_target

    base = args.br_dir.rstrip("/")
    n_batches = max(1, -(-args.episodes // args.batch_size))
    rng = jax.random.PRNGKey(args.seed)

    # rewards_by_seat[p] = [num_players] mean return when the BR holds seat p.
    rewards_by_seat = []
    for p in range(jax_config.num_players):
        path = f"{base}/br_p{p}_{args.step}.msgpack"
        br = load_br_for_agent(read_bytes(path), main_meta, jax_config,
                               opp_threshold=args.opp_threshold,
                               opp_discretization=args.opp_discretization,
                               br_layers_override=args.br_layers,
                               br_network_type_override=args.br_network_type)
        br.batch_size = args.batch_size
        br.trajectory_max = args.trajectory_max
        br._br_mask_dummy = bool(args.br_mask_dummy)
        print(f"\nseat {p}: {path}")
        print(f"  BR net   : {br.br_network_type} {br.br_layers}  "
              f"opp_features={getattr(br, 'opp_features', 'n/a')} "
              f"residual={getattr(br, 'opp_logit_residual', 'n/a')}")

        per_ep, unfinished = [], 0
        for b in range(n_batches):
            rng, k = jax.random.split(rng)
            traj = br._collect(br.params, main_params, k)
            per_ep.append(np.array(traj.all_rewards.sum(axis=0)))
            unfinished += int((~np.array(traj.done).any(axis=0)).sum())
            if (b + 1) % 10 == 0:
                seen = np.concatenate(per_ep)[:, p]
                print(f"    {len(seen):>7d} eps  BR mean {seen.mean():+.4f} "
                      f"(+/- {seen.std()/np.sqrt(len(seen)):.4f})")
        ep = np.concatenate(per_ep)[: args.episodes]
        rewards_by_seat.append(ep)
        se = ep[:, p].std() / np.sqrt(len(ep))
        print(f"  -> BR as p{p}: {ep[:, p].mean():+.4f} (+/- {se:.4f})  "
              f"n={len(ep)}  unfinished={unfinished}")

    print(f"frozen-agent filtering: threshold={args.opp_threshold} "
          f"discretization={args.opp_discretization}")
    print(f"\n{'='*66}")
    print(f"BR vs agent — {len(rewards_by_seat[0])} episodes per seat")
    print(f"{'='*66}")
    for p, ep in enumerate(rewards_by_seat):
        se = ep[:, p].std() / np.sqrt(len(ep))
        print(f"  br/reward_p{p} = {ep[:, p].mean():+.4f} (+/- {se:.4f})")

    # agent_score: the frozen agent's mean return across every seat it occupies,
    # in every configuration. Matches eval_br_vs_main's definition.
    vals = np.concatenate([
        ep[:, m] for br_seat, ep in enumerate(rewards_by_seat)
        for m in range(jax_config.num_players) if m != br_seat
    ])
    se = vals.std() / np.sqrt(len(vals))
    print(f"\n  br/agent_score = {vals.mean():+.4f} (+/- {se:.4f})   n={len(vals)}")
    print("  (lower = more exploitable; 0.0 = unexploitable by THIS BR)")
    if args.outcome_stats:
        import collections as _c
        br_pay = np.concatenate([ep[:, i] for i, ep in enumerate(rewards_by_seat)])
        hist = dict(sorted(_c.Counter(br_pay.astype(int).tolist()).items()))
        print(f"\n  br/win_pct    = {(br_pay > 0).mean()*100:6.2f}%   "
              f"loss = {(br_pay < 0).mean()*100:6.2f}%   n={len(br_pay)}")
        print(f"  agent/win_pct = {(vals > 0).mean()*100:6.2f}%   "
              f"loss = {(vals < 0).mean()*100:6.2f}%   n={len(vals)}")
        print(f"  BR payoff histogram: {hist}")
    sig = "significantly exploitable" if vals.mean() < -2 * se else "not distinguishable from 0"
    print(f"  -> {sig} at 2 s.e.")


if __name__ == "__main__":
    main()
