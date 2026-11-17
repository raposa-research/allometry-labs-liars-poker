"""Exact exploitability of a liars-poker-open-spiel checkpoint (pickle or msgpack).

Runs inside that repo's environment (JAX + flax); `lpeas` must be installed there:

    .venv/bin/pip install exact_br/liars_poker_eas
    .venv/bin/python exact_br/eval_liars_poker_checkpoint.py \
        experiments/3x3_v3/agent_1650000.pickle [--finetune]

The checkpoint's network is fed OpenSpiel `information_state_tensor`s (raw hand
digits + sparse bid/challenge history), which is exactly what the sibling repo's
`observation_tensor(..., hand_encoding="raw", history_encoding="sparse")` emits
and what `python_liars_poker`-trained pickles were trained on.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

DEFAULT_SIBLING_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"
)


def add_common_args(p):
    p.add_argument("checkpoint", help=".pickle or .msgpack checkpoint (local or s3://)")
    p.add_argument("--sibling-src", default=DEFAULT_SIBLING_SRC,
                   help="path to this repo's src/ (for eval_agents.load_agent); defaults to ../src")
    p.add_argument("--finetune", action="store_true", help="apply rnad.FineTuning post-processing to the policy")
    p.add_argument("--hand-encoding", choices=["auto", "raw", "histogram"], default="auto",
                   help="private-hand encoding the network was trained on; auto = raw for .pickle "
                        "(python_liars_poker-trained), the checkpoint's own config for .msgpack")
    p.add_argument("--dummy-mode", choices=["lose", "alias", "random"], default=None,
                   help="override the checkpoint's dummy_mode (dummy_mult > 1 checkpoints only)")
    p.add_argument("--chunk-nodes", type=int, default=20_000, help="public nodes per lpeas chunk")
    p.add_argument("--batch-size", type=int, default=262_144, help="rows per (padded, jitted) network call")
    p.add_argument("--out", default=None, help="write results as JSON here")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--dry-run", action="store_true", help="evaluate the policy on one chunk and exit")
    return p.parse_args()


def resolve_hand_encoding(checkpoint, cfg, requested="auto"):
    """Which private-hand encoding the checkpoint's network expects.

    Old `.pickle` agents were trained against OpenSpiel's `python_liars_poker`,
    whose observer writes the RAW hand digits, but `eval_agents._load_pickle_agent`
    rebuilds their config with the JAX default (`histogram`); `.msgpack` agents
    from train_jax.py embed the encoding they were trained with. With
    hand_length == num_digits both encodings have the same length, so a mix-up
    would not error -- it would silently evaluate a garbage policy.
    """
    if requested != "auto":
        return requested
    if checkpoint.endswith(".pickle"):
        return "raw"
    return getattr(cfg, "hand_encoding", "histogram")


def hand_to_histogram(infostates, num_digits, hand_length):
    """Rewrites OpenSpiel tensors (raw hand digits) into the JAX `histogram` observation."""
    digits = infostates[:, 2 : 2 + hand_length]
    hist = np.stack([(digits == d).sum(axis=1) for d in range(1, num_digits + 1)], axis=1)
    return np.concatenate([infostates[:, :2], hist, infostates[:, 2 + hand_length :]], axis=1)


# --- action-space scaling probe (sibling liars_poker_jax.LiarsPokerConfig.dummy_mult) ---
#
# The sibling game appends (dummy_mult - 1) * num_real_actions "dummy" actions.
# Dummy `n_real + k` aliases real action `k % n_real` for its LEGALITY in every
# mode (in "random" mode dummies are instead legal whenever any real action
# is), and what it DOES depends on dummy_mode:
#   "lose"   -- the taker loses at once: collapsed into one forfeit column that
#               lpeas handles exactly (cost independent of dummy_mult).
#   "alias"  -- plays the aliased real action: its mass is added to that action.
#   "random" -- resolves to a legal real action chosen by a hash of the FULL
#               state (both hands). We treat it as a uniform mixture over the
#               legal real actions, which is the idealisation the probe was
#               designed around; the hash's dependence on the opponent's hand
#               is not modelled, so this mode is approximate.


def network_legal_mask(legal_real, cfg):
    """Legal mask of the network's full action space from the real one."""
    mult = getattr(cfg, "dummy_mult", 1)
    if mult == 1:
        return legal_real
    if getattr(cfg, "dummy_mode", "random") == "random":
        any_real = legal_real.any(axis=1, keepdims=True)
        return np.concatenate([legal_real, np.repeat(any_real, (mult - 1) * legal_real.shape[1], axis=1)], axis=1)
    return np.tile(legal_real, (1, mult))


def collapse_dummy_probs(pi, legal_real, cfg):
    """Maps the network's [n, num_actions] policy onto the game lpeas solves.

    Returns [n, n_real] (alias / random modes) or [n, n_real + 1] with the
    forfeit mass in the last column (lose mode).
    """
    mult = getattr(cfg, "dummy_mult", 1)
    n_real = legal_real.shape[1]
    if mult == 1:
        return pi
    mode = getattr(cfg, "dummy_mode", "random")
    pi = np.asarray(pi, dtype=np.float64)
    real, dummy = pi[:, :n_real], pi[:, n_real:]
    if mode == "lose":
        return np.concatenate([real, dummy.sum(axis=1, keepdims=True)], axis=1)
    if mode == "alias":
        return pi.reshape(pi.shape[0], mult, n_real).sum(axis=1)
    if mode == "random":
        lr = legal_real.astype(np.float64)
        uniform = lr / np.maximum(lr.sum(axis=1, keepdims=True), 1.0)
        return real + dummy.sum(axis=1, keepdims=True) * uniform
    raise ValueError(f"unknown dummy_mode {mode!r}")


def _load_msgpack_agent(checkpoint):
    """Like eval_agents.load_agent for .msgpack, but keeps EVERY jax_config field.

    eval_agents rebuilds LiarsPokerConfig from (num_players, hand_length,
    num_digits) only, dropping dummy_mult / dummy_mode / hand_encoding that
    train_jax.py stored, so a dummy-probe checkpoint would come back with a
    19-wide legal mask against 19 * dummy_mult logits.
    """
    import dataclasses

    import flax.serialization
    from liars_poker_jax import LiarsPokerConfig
    from rnad import RNaDConfig
    from train_jax import JaxLiarsPokerRNaDSolver

    with open(checkpoint, "rb") as f:
        raw = f.read()
    meta = flax.serialization.msgpack_restore(raw)
    if "br_player" in meta:
        raise SystemExit("BR checkpoints are not supported")
    fields = {f.name for f in dataclasses.fields(LiarsPokerConfig)}
    jax_cfg = {k: v for k, v in meta["jax_config"].items() if k in fields}
    unknown = set(meta["jax_config"]) - fields
    if unknown:
        raise SystemExit(f"checkpoint jax_config has fields this sibling checkout lacks: {sorted(unknown)}")
    pnl = meta["policy_network_layers"]
    layers = tuple(pnl[str(i)] for i in range(len(pnl))) if isinstance(pnl, dict) else tuple(int(x) for x in pnl)
    agent = JaxLiarsPokerRNaDSolver(
        RNaDConfig(game_name="liars_poker", policy_network_layers=layers), LiarsPokerConfig(**jax_cfg)
    )
    # params-only restore: the optax state ties a checkpoint to the exact
    # optimizer chain of the trainer variant that wrote it (e.g. PPO-ent runs
    # carry an extra `count` state), and we never need it
    template = dict(params=agent.params, params_target=agent.params_target, params_prev=agent.params_prev,
                    params_prev_=agent.params_prev_, learner_steps=agent.learner_steps, actor_steps=agent.actor_steps)
    state = flax.serialization.from_bytes(template, raw)
    for k, v in state.items():
        setattr(agent, k, v)
    return agent


def load_checkpoint_policy(checkpoint, sibling_src, finetune=False, batch_size=262_144, hand_encoding="auto",
                           dummy_mode=None):
    """Loads a sibling-repo checkpoint and returns (agent, jax_config, policy_fn).

    `policy_fn(player, infostates, legal_mask)` is the `lpeas` policy interface:
    OpenSpiel information-state tensors in, action probabilities out (with a
    trailing forfeit column for dummy_mode="lose" checkpoints).
    """
    sys.path.insert(0, os.path.abspath(sibling_src))
    import dataclasses

    import jax
    import jax.numpy as jnp
    from eval_agents import _make_finetuned_apply, load_agent
    from rnad import EnvStep, FineTuning

    print(f"JAX backend: {jax.default_backend()} ({jax.devices()})")
    print(f"Loading {checkpoint}")
    agent = _load_msgpack_agent(checkpoint) if checkpoint.endswith(".msgpack") else load_agent(checkpoint)
    cfg = agent._jax_config
    if dummy_mode is not None and getattr(cfg, "dummy_mult", 1) > 1:
        cfg = dataclasses.replace(cfg, dummy_mode=dummy_mode)
    n_real = 2 * cfg.num_digits * cfg.hand_length * cfg.num_players // 2 + 1  # bids + challenge, 2 players
    mult = getattr(cfg, "dummy_mult", 1)
    print(f"  game: digits={cfg.num_digits} hand={cfg.hand_length} players={cfg.num_players} "
          f"max_jump={cfg.max_jump} history={getattr(agent, '_history_encoding', 'sparse')} "
          f"network={getattr(agent, '_network_type', 'mlp')} layers={list(agent.config.policy_network_layers)} "
          f"dummy_mult={mult} dummy_mode={getattr(cfg, 'dummy_mode', None) if mult > 1 else None}")
    if cfg.num_players != 2:
        raise SystemExit("lpeas only supports 2-player Liar's Poker")
    if cfg.max_jump is not None:
        raise SystemExit("max_jump restricts the action space: not the game lpeas solves")
    if getattr(agent, "_history_encoding", "sparse") != "sparse":
        raise SystemExit("only history_encoding='sparse' matches the OpenSpiel information state")
    if cfg.num_actions != n_real * mult:
        raise SystemExit(f"unexpected action count {cfg.num_actions} (expected {n_real} x {mult})")
    hand_encoding = resolve_hand_encoding(checkpoint, cfg, hand_encoding)
    print(f"  hand encoding fed to the network: {hand_encoding}")

    apply = agent.network.apply
    if finetune:
        apply = _make_finetuned_apply(apply, FineTuning())
    params = agent.params_target
    B = batch_size
    NA_net = cfg.num_actions

    @jax.jit
    def policy(obs, legal, player):
        env_step = EnvStep(
            obs=obs,
            legal=legal,
            player_id=jnp.full((B,), player, dtype=jnp.float32),
            valid=jnp.ones((B,), dtype=jnp.float32),
            rewards=jnp.zeros((B, cfg.num_players), dtype=jnp.float32),
        )
        pi, _, _, _ = jax.vmap(apply, (None, 0), 0)(params, env_step)
        return pi

    def policy_fn(player, infostates, legal_mask):
        n = infostates.shape[0]
        if hand_encoding == "histogram":
            infostates = hand_to_histogram(infostates, cfg.num_digits, cfg.hand_length)
        legal_net = network_legal_mask(legal_mask, cfg)
        out = np.empty((n, NA_net), dtype=np.float64)
        for k in range(0, n, B):
            m = min(B, n - k)
            obs = np.zeros((B, infostates.shape[1]), dtype=np.float32)
            legal = np.zeros((B, NA_net), dtype=np.int8)
            obs[:m] = infostates[k : k + m]
            legal[:m] = legal_net[k : k + m]
            # pad rows with a legal mask that has at least one legal action
            legal[m:, 0] = 1
            pi = policy(jnp.asarray(obs), jnp.asarray(legal), jnp.float32(player))
            out[k : k + m] = np.asarray(pi[:m], dtype=np.float64)
        return collapse_dummy_probs(out, legal_mask, cfg)

    return agent, cfg, policy_fn


def build_traverser(cfg):
    import lpeas

    t = lpeas.LiarsPokerTraverser(cfg.num_digits, cfg.hand_length)
    assert t.num_actions * getattr(cfg, "dummy_mult", 1) == cfg.num_actions, (t.num_actions, cfg.num_actions)
    print(f"  tree: {t.num_decision_nodes:,} public decision nodes, {t.num_terminals:,} terminals, "
          f"{t.num_infosets(0) + t.num_infosets(1):,} infosets")
    return t


def summarize(args, cfg, t, res, elapsed):
    ev0, (expl0, expl1) = res["ev0"], res["expl"]
    mult = getattr(cfg, "dummy_mult", 1)
    return {
        "checkpoint": args.checkpoint,
        "finetune": args.finetune,
        "hand_encoding": resolve_hand_encoding(args.checkpoint, cfg, args.hand_encoding),
        "dummy_mult": mult,
        "dummy_mode": getattr(cfg, "dummy_mode", None) if mult > 1 else None,
        "num_digits": cfg.num_digits,
        "hand_length": cfg.hand_length,
        "ev0": ev0,
        "expl0": expl0,
        "expl1": expl1,
        "nash_conv": expl0 + expl1,
        "avg_score_response": (expl0 + expl1) / 2,
        "br_value": list(res["br_value"]),
        "num_infosets": t.num_infosets(0) + t.num_infosets(1),
        "seconds": elapsed,
    }


def main():
    args = parse_args()
    import lpeas

    agent, cfg, policy_fn = load_checkpoint_policy(
        args.checkpoint, args.sibling_src, args.finetune, args.batch_size, args.hand_encoding, args.dummy_mode
    )
    t = build_traverser(cfg)

    if args.dry_run:
        t.begin(args.chunk_nodes)
        for _ in range(3):
            player, infostates, legal = t.next_chunk()
            t0 = time.time()
            probs = policy_fn(player, infostates, legal)
            print(f"dry run: level player {player}, {infostates.shape[0]} rows in {time.time() - t0:.2f}s, "
                  f"probs[0]={np.round(probs[0], 3)}")
            t.push_probs(probs)
        return

    t0 = time.time()
    res = lpeas.ev_and_exploitability(t, policy_fn, chunk_nodes=args.chunk_nodes)
    summary = summarize(args, cfg, t, res, time.time() - t0)
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
