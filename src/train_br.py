#!/usr/bin/env python
"""Train a PPO best-response agent against a frozen checkpoint.

Usage
-----
    python train_br.py <checkpoint.msgpack> [options]

The game config (num_players, hand_length, num_digits) and network
architecture are read directly from the checkpoint, so no separate
config flags are needed.
"""

import argparse
import io
import os
import sys

import flax
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from tqdm import trange
import wandb

from rnad import RNaDConfig, FineTuning
from liars_poker_jax import LiarsPokerConfig

# Import from sibling module (same directory).
from train_jax import (
    JaxLiarsPokerRNaDSolver,
    JaxPPOBRAgent,
    build_ppo_br_agents,
    eval_br_vs_main,
)


# ---------------------------------------------------------------------------
# S3 helpers (boto3 imported lazily; respects AWS_ENDPOINT_URL for R2)
# ---------------------------------------------------------------------------


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _s3_client():
    import boto3
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client("s3", endpoint_url=endpoint_url)


def _parse_s3(path: str) -> tuple[str, str]:
    """Return (bucket, key) from an s3://bucket/key path."""
    without_scheme = path[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


# ---------------------------------------------------------------------------
# Checkpoint metadata extraction
# ---------------------------------------------------------------------------


def _decode_bytes(obj):
    """Recursively decode bytes keys/values in a msgpack-decoded dict."""
    if isinstance(obj, dict):
        return {
            (k.decode() if isinstance(k, bytes) else k): _decode_bytes(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_decode_bytes(x) for x in obj]
    if isinstance(obj, bytes):
        # Leaf bytes are JAX array payloads — leave them alone.
        return obj
    return obj


def _kernel_shape(leaf) -> tuple | None:
    """Return the shape of a flax msgpack array leaf, or None.

    Flax serialises numpy arrays as msgpack ExtType objects whose ``.data``
    is itself a msgpack blob encoding ``[shape, dtype, raw_bytes]``.
    """
    raw = getattr(leaf, "data", None) if not isinstance(leaf, bytes) else leaf
    if not isinstance(raw, bytes):
        return None
    try:
        inner = msgpack.unpackb(raw)
        if isinstance(inner, (list, tuple)) and len(inner) >= 1:
            shape = inner[0]
            if isinstance(shape, (list, tuple)):
                return tuple(int(x) for x in shape)
    except Exception:
        pass
    return None


def _try_infer_config_from_raw_msgpack(data: dict, layers: list, jc_dict: dict) -> dict:
    """Infer max_jump and max_bid_count from kernel shapes in the raw msgpack.

    Flax stores arrays as msgpack ExtType whose .data is itself msgpack-encoded
    as [shape, dtype, raw_bytes], so kernel shapes are directly readable.

    Policy head (Dense_N, N = len(layers)): shape (layers[-1], num_actions)
        → max_jump = (num_actions - 1) // num_digits - 1

    Input layer (Dense_0): shape (obs_dim, layers[0])
        → max_bid_count inferred via the observation_size formula

    This is needed because _serializable_state() only stores the 3 base game
    fields, so max_jump and max_bid_count must be recovered before the network
    is constructed and restore_state() is called.
    """
    result = {}
    try:
        num_digits  = int(jc_dict["num_digits"])
        num_players = int(jc_dict["num_players"])
        hand_length = int(jc_dict["hand_length"])
        N = len(layers)

        params_inner = data.get("params", {}).get("params", {})

        # Policy head: Dense_N kernel shape = (layers[-1], num_actions)
        shape = _kernel_shape(params_inner.get(f"Dense_{N}", {}).get("kernel"))
        if shape and len(shape) == 2 and shape[0] == layers[-1]:
            num_actions = shape[1]
            mj = (num_actions - 1) // num_digits - 1
            if mj >= 1:
                result["max_jump"] = mj

        # Input layer: Dense_0 kernel shape = (obs_dim, layers[0])
        shape0 = _kernel_shape(params_inner.get("Dense_0", {}).get("kernel"))
        if shape0 and len(shape0) == 2 and shape0[1] == layers[0]:
            obs_dim = shape0[0]
            hist = obs_dim - num_players - hand_length - 2
            denom = 2 * num_digits * num_players
            if hist > 0 and denom > 0:
                mbc = hist // denom
                if mbc > 0 and hist == mbc * denom:
                    result["max_bid_count"] = mbc

    except (KeyError, AttributeError, TypeError, ZeroDivisionError, IndexError):
        pass
    return result


def read_checkpoint_metadata(path: str) -> dict:
    """Return jax_config and policy_network_layers from a msgpack checkpoint.

    The checkpoint is produced by flax.serialization.to_bytes(), which
    internally uses msgpack.  We unpack it as raw msgpack to read the
    lightweight metadata fields without reconstructing the full pytree.

    In addition to the stored jax_config fields (num_players, hand_length,
    num_digits), max_jump and max_bid_count are inferred from kernel byte
    lengths so the solver can be constructed with the correct network shape
    before restore_state is called.
    """
    raw = read_bytes(path)
    data = msgpack.unpackb(raw, strict_map_key=False)
    data = _decode_bytes(data)

    meta = {}

    jc = data.get("jax_config")
    if jc is None:
        print(
            "WARNING: checkpoint does not contain 'jax_config'; "
            "using default game config (3-digit 3-card 2-player).",
            file=sys.stderr,
        )
        meta["jax_config"] = {"num_players": 2, "hand_length": 3, "num_digits": 3}
    else:
        # Int-coerce the NUMERIC game dimensions only. jax_config now also
        # carries dummy_mode, a STRING ("random"/"alias"/"lose"), and blanket
        # int() on every value raised
        #   ValueError: invalid literal for int() with base 10: 'lose'
        def _coerce(v):
            if isinstance(v, int):
                return v
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, str) and not v.lstrip("-").isdigit():
                return v                      # genuine string field
            return int(v)
        meta["jax_config"] = {k: _coerce(v) for k, v in jc.items()}

    layers = data.get("policy_network_layers")
    if layers is None:
        print(
            "WARNING: checkpoint does not contain 'policy_network_layers'; "
            "using default [256, 256, 256].",
            file=sys.stderr,
        )
        meta["policy_network_layers"] = [256, 256, 256]
    elif isinstance(layers, dict):
        # flax serializes lists as dicts with string integer keys {"0": v0, "1": v1, ...}
        meta["policy_network_layers"] = [int(v) for _, v in sorted(layers.items(), key=lambda kv: int(kv[0]))]
    else:
        meta["policy_network_layers"] = [int(x) for x in layers]

    # Architecture metadata (saved by JaxLiarsPokerRNaDSolver._serializable_state).
    # Absent in older checkpoints -> default to the legacy plain-mlp/sparse agent
    # so those load byte-identically to before.
    meta["network_type"] = data.get("network_type", "mlp")
    meta["history_encoding"] = data.get("history_encoding", "sparse")
    tc = data.get("transformer_config")
    if isinstance(tc, dict):
        tc = {k: (int(v) if isinstance(v, (int, float)) else v) for k, v in tc.items()}
    meta["transformer_config"] = tc

    # Infer max_jump and max_bid_count from kernel byte lengths.
    inferred = _try_infer_config_from_raw_msgpack(data, meta["policy_network_layers"], meta["jax_config"])
    if inferred:
        meta["jax_config"].update(inferred)

    return meta


def jax_config_from_meta(jc_dict: dict) -> LiarsPokerConfig:
    """Construct a LiarsPokerConfig from checkpoint metadata.

    dummy_mult / dummy_mode MUST be carried through. They change num_actions, so
    dropping them builds a 19-action game for a 152-action dummy-probe
    checkpoint and the frozen agent's params fail to apply. Absent in every
    pre-probe checkpoint, where the defaults (1 / "random") are correct and this
    is byte-identical to before.
    """
    return LiarsPokerConfig(
        num_players=jc_dict["num_players"],
        hand_length=jc_dict["hand_length"],
        num_digits=jc_dict["num_digits"],
        dummy_mult=int(jc_dict.get("dummy_mult", 1)),
        dummy_mode=str(jc_dict.get("dummy_mode", "random")),
        # Checkpoints trained against OpenSpiel's python_liars_poker encode the
        # private hand as RAW digit values; this repo's default is a per-digit
        # histogram. When hand_length == num_digits the two have the same
        # length, so a mismatch is silent and the net simply cannot read its
        # own hand. Absent -> "histogram", byte-identical to before.
        hand_encoding=str(jc_dict.get("hand_encoding", "histogram")),
    )


def infer_max_bid_count_from_params(params, num_players: int, hand_length: int, num_digits: int) -> int | None:
    """Infer max_bid_count from the first Dense layer's input dimension.

    Needed for checkpoints that pre-date storing max_bid_count explicitly.
    Returns None if the inference fails.
    """
    try:
        obs_dim = int(params["params"]["Dense_0"]["kernel"].shape[0])
        # obs_dim = num_players + hand_length + 2 + 2 * max_bid_count * num_digits * num_players
        hist = obs_dim - num_players - hand_length - 2
        mbc = hist // (2 * num_digits * num_players)
        if mbc > 0 and hist == 2 * mbc * num_digits * num_players:
            return mbc
    except (KeyError, IndexError, AttributeError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="Path to the msgpack checkpoint file.")
    p.add_argument(
        "--players",
        nargs="+",
        type=int,
        default=None,
        metavar="P",
        help="Player positions to train BR for (default: all players).",
    )
    p.add_argument(
        "--n-cycles",
        type=int,
        default=200,
        metavar="N",
        help="Number of collect+update cycles (default: 200).",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=20,
        metavar="K",
        help="Evaluate every K cycles (default: 20).",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=2048,
        metavar="E",
        help="Episodes per evaluation (default: 2048).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Parallel environments per rollout (default: 512).",
    )
    p.add_argument(
        "--trajectory-max",
        type=int,
        default=30,
        help="Steps per rollout (default: 30).",
    )
    p.add_argument(
        "--br-network-type", default=None, metavar="ARCH",
        help="Architecture for the BR's OWN trainable net (mlp/factored_mlp/lpt). "
             "Default: 'mlp' when the frozen agent is not plain mlp (a decoupled, "
             "honest plain-MLP exploiter -- the recommended setup for a "
             "factored/transformer agent), else the agent's own arch.",
    )
    p.add_argument(
        "--br-layers", nargs="+", type=int, default=None, metavar="H",
        help="Hidden layer sizes for the BR net (default: the agent's "
             "policy_network_layers).",
    )
    p.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate (default: 3e-4).")
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--n-epochs", type=int, default=4, help="PPO update epochs per batch (default: 4).")
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-jump", type=int, default=None, metavar="J",
                   help="Override max_jump (not stored in checkpoint; must match training config).")
    p.add_argument("--max-bid-count", type=int, default=None, metavar="M",
                   help="Override max_bid_count (not stored in checkpoint; must match training config).")
    p.add_argument("--first-bid-base-count", type=int, default=None, metavar="C",
                   help="Override first_bid_base_count (not stored in checkpoint; must match training config).")
    p.add_argument("--save-dir", default="checkpoints/br", metavar="DIR",
                   help="Directory to write BR checkpoints (default: checkpoints/br).")
    p.add_argument("--checkpoint-every", type=int, default=0, metavar="K",
                   help="Save a checkpoint every K cycles. 0 = only save at the end (default: 0).")
    p.add_argument("--opp-threshold", type=float, default=0.03, metavar="P",
                   help="Zero out FROZEN-agent actions with probability below "
                        "P before the BR samples against them. This is R-NaD "
                        "FineTuning's policy_threshold and it is ALWAYS applied "
                        "in the BR rollout (default 0.03); pass 0 for the raw "
                        "softmax policy.")
    p.add_argument("--opp-discretization", type=int, default=32, metavar="D",
                   help="Round frozen-agent probabilities to multiples of 1/D "
                        "(default 32; <=0 disables).")
    p.add_argument("--wandb-project", default="liars-poker__train-br", metavar="PROJECT")
    p.add_argument("--wandb-entity", default=None, metavar="ENTITY")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    p.add_argument("--br-resume-dir", default=None, metavar="DIR",
                   help="Continue a previous BR run: load br_p<seat>_<step>."
                        "msgpack from DIR into the MATCHING seat, restoring "
                        "params and Adam state. Unlike --br-init-from (which "
                        "copies ONE checkpoint into every seat, correct only "
                        "for a symmetric agent), this keeps each seat's own "
                        "trained net. Requires --br-resume-step.")
    p.add_argument("--br-resume-step", type=int, default=None, metavar="N",
                   help="Cycle to resume from. Training then runs cycles "
                        "N+1..--n-cycles, so checkpoint numbering continues "
                        "the original run instead of overwriting it.")
    p.add_argument("--br-resume-no-opt", action="store_true",
                   help="Restore params but NOT the Adam state. Use only when "
                        "changing --lr; a fresh optimiser loses the moment "
                        "estimates and the first steps after resume are noisy.")
    p.add_argument("--br-init-from", default=None, metavar="CKPT",
                   help="Warm-start each BR's own net from this checkpoint's "
                        "params_target instead of random init. Use when a known "
                        "policy already beats the exploitee: PPO then starts at "
                        "that policy's value and can only improve. The "
                        "checkpoint's architecture must match "
                        "--br-network-type / --br-layers. NOTE a warm-started "
                        "BR begins as a COPY of what it is exploiting, so with "
                        "--ent-coef 0 it has nothing pushing it to deviate and "
                        "a near-zero score can mean 'did not search' rather "
                        "than 'nothing to find' -- pair it with "
                        "--br-temperature > 1 and --ent-coef > 0.")
    p.add_argument("--br-temperature", type=float, default=1.0, metavar="T",
                   help="Policy temperature BAKED INTO the BR network: "
                        "pi = softmax(logit / T). The tempered policy IS the "
                        "policy -- sampled, scored, entropy-regularised, "
                        "trained and evaluated -- so collection stays exactly "
                        "on-policy. T>1 gives a warm-started BR a flatter "
                        "STARTING policy (its loaded logits arrive at the "
                        "agent's trained scale) and damps the logit gradient "
                        "by 1/T. It is a reparameterisation, NOT an entropy "
                        "floor: the net can resharpen by growing ||logit||, so "
                        "pair it with --ent-coef > 0. Persisted in the "
                        "checkpoint; evaluation must use the same T.")
    p.add_argument("--br-mask-dummy", action="store_true",
                   help="Hard-mask DUMMY actions out of the BR's own action "
                        "space. A best response never wants a strictly "
                        "dominated action, so this cannot weaken the BR; it "
                        "removes its burden of learning to avoid them, which "
                        "otherwise makes the BR a weak instrument in the dummy "
                        "probes. No-op when dummy_mult == 1.")
    return p.parse_args()


def save_br_checkpoints(br_agents, save_dir: str, step: int, jax_config, network_layers):
    """Save each BR agent's params and opt_state to local disk or S3."""
    if not _is_s3(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for agent in br_agents:
        filename = f"br_p{agent.br_player}_{step}.msgpack"
        path = f"{save_dir.rstrip('/')}/{filename}"
        state = {
            "params": agent.params,
            "opt_state": agent.opt_state,
            "jax_config": {
                "num_players": jax_config.num_players,
                "hand_length": jax_config.hand_length,
                "num_digits":  jax_config.num_digits,
            },
            "policy_network_layers": network_layers,
            # The BR's OWN architecture. Without this the loader cannot tell a
            # factored_mlp BR from an mlp one and builds the wrong net; old
            # checkpoints written by the serve-mlp tree carry it, this tree's
            # did not, so only NEW checkpoints were unloadable.
            "network_type": str(getattr(agent, "br_network_type", "mlp")),
            "br_player": agent.br_player,
            # Evaluation MUST rebuild the net with the same temperature or it
            # scores a different (sharper) policy than the one trained.
            "br_temperature": float(getattr(agent, "br_temperature", 1.0)),
        }
        write_bytes(path, flax.serialization.to_bytes(state))
        print(f"Saved BR checkpoint: {path}")


def restore_frozen_params(main_agent, path: str) -> None:
    """Restore ONLY the param trees (+ step counts) from a checkpoint.

    The frozen opponent never trains, so its optimizer state is irrelevant --
    and restoring it via the full ``restore_state`` fails whenever the training
    optimizer differs from the default (e.g. ``warmup_steps>0`` adds a schedule
    'count' leaf that isn't stored in the checkpoint metadata). Passing a
    PARTIAL target to ``flax.serialization.from_bytes`` restores just the listed
    keys and ignores everything else in the msgpack (optimizer, rngkey, ...).
    """
    encoded = read_bytes(path)
    target = {
        "params": main_agent.params,
        "params_target": main_agent.params_target,
        "params_prev": main_agent.params_prev,
        "params_prev_": main_agent.params_prev_,
        "learner_steps": main_agent.learner_steps,
        "actor_steps": main_agent.actor_steps,
    }
    restored = flax.serialization.from_bytes(target, encoded)
    main_agent.params = restored["params"]
    main_agent.params_target = restored["params_target"]
    main_agent.params_prev = restored["params_prev"]
    main_agent.params_prev_ = restored["params_prev_"]
    main_agent.learner_steps = int(restored["learner_steps"])
    main_agent.actor_steps = int(restored["actor_steps"])


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Read metadata from checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    meta = read_checkpoint_metadata(args.checkpoint)
    jc_dict = meta["jax_config"]
    network_layers = meta["policy_network_layers"]
    network_type = meta["network_type"]
    history_encoding = meta["history_encoding"]
    transformer_config = meta["transformer_config"]
    print(f"  game config    : {jc_dict}")
    print(f"  network layers : {network_layers}")
    print(f"  agent arch     : {network_type} / history={history_encoding}")
    print(f"  JAX devices    : {jax.devices()}")

    jax_config = jax_config_from_meta(jc_dict)

    # ------------------------------------------------------------------
    # 2. Reconstruct main agent (with its REAL arch) and restore state
    # ------------------------------------------------------------------
    rnad_config = RNaDConfig(
        game_name="unused",
        policy_network_layers=network_layers,
    )
    main_agent = JaxLiarsPokerRNaDSolver(
        rnad_config,
        jax_config,
        network_type=network_type,
        transformer_config=transformer_config,
        history_encoding=history_encoding,
    )
    restore_frozen_params(main_agent, args.checkpoint)

    print(f"  learner steps  : {main_agent.learner_steps}")

    # ------------------------------------------------------------------
    # 3. Build BR agents
    # ------------------------------------------------------------------
    target_players = args.players if args.players is not None else list(range(jax_config.num_players))
    print(f"  training BR for player(s): {target_players}")

    # Decoupled BR arch: exploit a non-mlp (factored/transformer) agent with a
    # plain-MLP BR by default -- the honest exploiter (memory: a same-arch BR is
    # a WEAK exploiter, so specialized-arch numbers must be validated by a
    # decoupled mlp BR). For a plain-mlp agent the BR stays same-arch (mlp).
    if args.br_network_type is not None:
        br_network_type = args.br_network_type
    elif network_type != "mlp":
        br_network_type = "mlp"
    else:
        br_network_type = None  # same as main (mlp) — legacy byte-identical path
    br_layers = args.br_layers  # None -> defaults to the agent's layers
    print(f"  BR arch        : "
          f"{br_network_type or network_type} "
          f"layers={br_layers or network_layers}"
          + ("  (decoupled honest exploiter)"
             if br_network_type and br_network_type != network_type else ""))

    ppo_config = dict(
        network_layers=network_layers,
        batch_size=args.batch_size,
        trajectory_max=args.trajectory_max,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        gamma=1.0,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
    )

    def make_br_agents(seed):
        _agents = build_ppo_br_agents(
            jax_config,
            ppo_config,
            seed=seed,
            network_type=network_type,
            transformer_config=transformer_config,
            history_encoding=history_encoding,
            br_network_type=br_network_type,
            br_layers=br_layers,
            finetune=FineTuning(
                from_learner_steps=0,
                policy_threshold=args.opp_threshold,
                policy_discretization=args.opp_discretization,
            ),
            br_temperature=args.br_temperature,
        )
        for _a in (_agents.values() if isinstance(_agents, dict) else _agents):
            _a._br_mask_dummy = bool(args.br_mask_dummy)
        return _agents

    print(f"Frozen-agent policy filtering: threshold={args.opp_threshold} "
          f"discretization={args.opp_discretization}")
    all_br_agents = make_br_agents(args.seed)
    br_agents = [all_br_agents[p] for p in target_players]

    # ------------------------------------------------------------------
    # 3b. Optional warm start of the BR's own net
    # ------------------------------------------------------------------
    if args.br_resume_dir:
        if args.br_resume_step is None:
            raise SystemExit("--br-resume-dir requires --br-resume-step.")
        if args.br_init_from:
            raise SystemExit(
                "--br-resume-dir and --br-init-from are mutually exclusive: "
                "the first continues per-seat training, the second overwrites "
                "every seat with one checkpoint.")
        if args.br_resume_step >= args.n_cycles:
            raise SystemExit(
                f"--br-resume-step {args.br_resume_step} >= --n-cycles "
                f"{args.n_cycles}: nothing left to train.")
        print(f"\nResuming BR from {args.br_resume_dir} at cycle "
              f"{args.br_resume_step}")
        for agent in br_agents:
            path = (f"{args.br_resume_dir.rstrip('/')}/"
                    f"br_p{agent.br_player}_{args.br_resume_step}.msgpack")
            blob = read_bytes(path)
            meta = flax.serialization.msgpack_restore(blob)
            # The seat recorded in the file must be the seat we are loading
            # into. Loading p3's net into p1 would silently train the wrong
            # policy and still produce plausible-looking numbers.
            got_seat = int(meta["br_player"])
            if got_seat != agent.br_player:
                raise SystemExit(
                    f"{path} records br_player={got_seat} but is being loaded "
                    f"into seat {agent.br_player}.")
            target = {"params": agent.params}
            if not args.br_resume_no_opt:
                target["opt_state"] = agent.opt_state
            restored = flax.serialization.from_bytes(target, blob)
            agent.params = restored["params"]
            if args.br_resume_no_opt:
                agent.opt_state = agent._optimizer.init(agent.params)
                note = "params only, fresh optimiser"
            else:
                agent.opt_state = restored["opt_state"]
                note = "params + Adam state"
            print(f"  seat {agent.br_player}: {note}  <- {path}")
        print(f"  will train cycles {args.br_resume_step + 1}..{args.n_cycles} "
              f"({args.n_cycles - args.br_resume_step} more)")

    if args.br_init_from:
        print(f"\nWarm-starting BR nets from: {args.br_init_from}")
        warm_meta = read_checkpoint_metadata(args.br_init_from)
        warm_solver = JaxLiarsPokerRNaDSolver(
            RNaDConfig(game_name="unused",
                       policy_network_layers=warm_meta["policy_network_layers"]),
            jax_config_from_meta(warm_meta["jax_config"]),
            network_type=warm_meta["network_type"],
            history_encoding=warm_meta["history_encoding"],
        )
        warm_solver.restore_params_only(read_bytes(args.br_init_from))
        warm_params = warm_solver.params_target

        # Structural mismatch here means the warm-start net is a different
        # architecture than the BR's; fail loudly rather than silently
        # random-initialising or half-loading.
        want = jax.tree_util.tree_structure(br_agents[0].params)
        got = jax.tree_util.tree_structure(warm_params)
        if want != got:
            raise SystemExit(
                f"--br-init-from architecture mismatch.\n"
                f"  checkpoint: {warm_meta['network_type']} "
                f"{warm_meta['policy_network_layers']}\n"
                f"  BR net    : {br_network_type} {list(br_layers)}\n"
                "Set --br-network-type / --br-layers to match the checkpoint."
            )
        shapes_w = [x.shape for x in jax.tree_util.tree_leaves(warm_params)]
        shapes_b = [x.shape for x in jax.tree_util.tree_leaves(br_agents[0].params)]
        if shapes_w != shapes_b:
            raise SystemExit(
                f"--br-init-from shape mismatch (same tree, different sizes):\n"
                f"  checkpoint: {shapes_w[:3]}...\n  BR net    : {shapes_b[:3]}...\n"
                "history_encoding or layer widths differ."
            )

        for agent in br_agents:
            agent.params = warm_params
            agent.opt_state = agent._optimizer.init(agent.params)
        print(f"  warm start OK -- every BR seat begins as "
              f"{warm_meta['network_type']} {warm_meta['policy_network_layers']} "
              f"({warm_solver.learner_steps} learner steps)")
        if args.br_temperature != 1.0:
            print(f"  the loaded logits are then divided by T="
                  f"{args.br_temperature:g}, so the BR STARTS flatter than the "
                  "policy it was initialised from.")
        else:
            print("  cycle-0 score should equal that policy's head-to-head value.")

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    rng_key = jax.random.PRNGKey(args.seed)
    main_params = main_agent.params_target  # frozen throughout

    # ------------------------------------------------------------------
    # 4b. Initialise wandb
    # ------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "checkpoint": args.checkpoint,
                **jc_dict,
                "network_layers": network_layers,
                "learner_steps": main_agent.learner_steps,
                "target_players": target_players,
                "n_cycles": args.n_cycles,
                **{f"ppo_{k}": v for k, v in ppo_config.items()},
                "seed": args.seed,
            },
        )

    print(f"\nTraining {args.n_cycles} cycles, eval every {args.eval_every}...\n")

    if args.br_temperature != 1.0:
        print(f"BR network temperature: {args.br_temperature:g} "
              f"(pi = softmax(logit / T); ent_coef={args.ent_coef})")

    start_cycle = args.br_resume_step or 0
    for cycle in trange(start_cycle, args.n_cycles, desc="BR cycles",
                        initial=start_cycle, total=args.n_cycles):
        cycle_losses = {}
        for agent in br_agents:
            rng_key, k = jax.random.split(rng_key)
            loss, _states = agent.train_phase(main_params=main_params, rng_key=k,
                                              n_collect=1)
            cycle_losses[f"br/loss_p{agent.br_player}"] = loss

        if use_wandb:
            wandb.log(cycle_losses, step=cycle + 1)

        if args.checkpoint_every > 0 and (cycle + 1) % args.checkpoint_every == 0:
            save_br_checkpoints(br_agents, args.save_dir, cycle + 1, jax_config, network_layers)

        if (cycle + 1) % args.eval_every == 0:
            rng_key, eval_key = jax.random.split(rng_key)
            # Build a list aligned to all player positions for eval_br_vs_main.
            # Wrap in a dummy structure if we only trained a subset of players.
            if target_players == list(range(jax_config.num_players)):
                eval_agents = br_agents
            else:
                # Build fresh (untrained) agents for positions we didn't train,
                # then slot in the trained ones.
                eval_agents = make_br_agents(args.seed)
                for trained_agent, p in zip(br_agents, target_players):
                    eval_agents[p] = trained_agent

            logs = eval_br_vs_main(
                jax_config, main_agent, eval_agents, eval_key, args.eval_episodes
            )
            if use_wandb:
                wandb.log(logs, step=cycle + 1)
            parts = "  ".join(f"{k}={v:+.4f}" for k, v in sorted(logs.items()))
            print(f"\ncycle {cycle + 1:4d}  {parts}")

    # ------------------------------------------------------------------
    # 5. Final evaluation
    # ------------------------------------------------------------------
    print("\n--- Final evaluation ---")
    rng_key, eval_key = jax.random.split(rng_key)
    if target_players == list(range(jax_config.num_players)):
        eval_agents = br_agents
    else:
        eval_agents = make_br_agents(args.seed)
        for trained_agent, p in zip(br_agents, target_players):
            eval_agents[p] = trained_agent

    logs = eval_br_vs_main(
        jax_config, main_agent, eval_agents, eval_key, args.eval_episodes * 4
    )
    for k, v in sorted(logs.items()):
        print(f"  {k} = {v:+.4f}")

    agent_score = logs.get("br/agent_score", float("nan"))
    print(f"\nMain agent score vs BR: {agent_score:+.4f}")
    print("(Lower = more exploitable; 0.0 = Nash for zero-sum games)")

    save_br_checkpoints(br_agents, args.save_dir, args.n_cycles, jax_config, network_layers)

    if use_wandb:
        wandb.log({f"final/{k}": v for k, v in logs.items()}, step=args.n_cycles)
        wandb.finish()


if __name__ == "__main__":
    main()
