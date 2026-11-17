"""Measure policy entropy of saved 3x3x3 agents, to test the determinism hypothesis.

The PPO cells of the 3x3x3 matrix degrade to a BR winning EVERY hand in EVERY
seat, while the NeuRD cells stay flat. The proposed mechanism is that the PPO
surrogate has nothing holding the policy off determinism (NeuRD's beta/clip
trust region does not apply; the ratio clip is inert at one gradient step per
batch; ppo_ent_coef defaults to 0). If that is right, the PPO agent's policy
entropy should be collapsing toward 0 while NeuRD's stays high.

Reports mean entropy over the acting player's visited states, in NATS, next to
the max possible entropy given the number of legal actions at those states
(log |A_legal|), since the two are only comparable relative to that ceiling.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

import flax.serialization as fs  # noqa: E402
from liars_poker_jax import LiarsPokerConfig  # noqa: E402
from rnad import RNaDConfig  # noqa: E402
from train_jax import JaxLiarsPokerRNaDSolver  # noqa: E402


def load_agent(path, batch_size=256, trajectory_max=30, history_encoding=None):
    """Params-only loader.

    eval_agents.load_agent calls restore_state, which also restores the OPTIMIZER
    state -- and that fails here with a flax/optax namedtuple mismatch, because
    the checkpoints were written remotely against a newer optax
    ("field names of the state dict and the named tuple do not match ... 'count'").
    Only the policy params are needed to measure entropy, so rebuild the solver
    from the checkpoint's own metadata (network_type et al -- eval_agents does
    NOT read those, so it would silently build an mlp for a factored_mlp
    checkpoint) and assign params directly.
    """
    # s3:// support so the GPU runs can read checkpoints straight from R2
    # (eval_agents already has the boto3 reader; import it lazily so local use
    # does not need click/boto3).
    if str(path).startswith("s3://"):
        from eval_agents import _read_bytes as _rb
        raw = _rb(str(path))
    else:
        raw = pathlib.Path(path).read_bytes()
    meta = fs.msgpack_restore(raw)
    jc = meta["jax_config"]
    pnl = meta["policy_network_layers"]
    layers = (tuple(pnl[str(i)] for i in range(len(pnl))) if isinstance(pnl, dict)
              else tuple(int(x) for x in pnl))
    agent = JaxLiarsPokerRNaDSolver(
        RNaDConfig(game_name="liars_poker", policy_network_layers=layers,
                   batch_size=batch_size, trajectory_max=trajectory_max),
        # dummy_mult MUST be carried through: it changes num_actions, so
        # omitting it builds a game whose action space is smaller than the
        # checkpoint's policy head and the params fail to apply. Absent in
        # every pre-probe checkpoint, where the default 1 is correct.
        LiarsPokerConfig(num_players=int(jc["num_players"]),
                         hand_length=int(jc["hand_length"]),
                         num_digits=int(jc["num_digits"]),
                         dummy_mult=int(jc.get("dummy_mult", 1)),
                         dummy_mode=str(jc.get("dummy_mode", "random")),
                         hand_encoding=str(
                             jc.get("hand_encoding", "histogram"))),
        network_type=str(meta.get("network_type", "mlp")),
        history_encoding=history_encoding or str(meta.get("history_encoding", "sparse")))
    agent.params = jax.tree_util.tree_map(jnp.asarray, meta["params"])
    return agent


def _states_from(agent, batches):
    """Roll out self-play and return the [T,B,...] env steps + valid masks."""
    return [agent.collect_batch_trajectory() for _ in range(batches)]


def entropy_report(path: str, label: str, batches: int = 3, ref=None, **kw):
    """Entropy of the checkpoint's policy.

    By default the states are the agent's OWN self-play visitation. That is the
    natural distribution but it makes cross-agent comparison unfair: a collapsed
    policy reaches a much narrower set of states, so it is only ever measured
    where it goes, and it could in principle be deterministic on-path yet mixed
    off-path (which is where a best response drives it). Pass `ref` (another
    loaded agent) to evaluate this policy on the REFERENCE agent's visited
    states instead, so every agent is scored on one common distribution.
    """
    agent = load_agent(path, **kw)
    ents, ceils, legals, maxps = [], [], [], []
    trajs = _states_from(ref if ref is not None else agent, batches)
    for ts in trajs:
        if ref is None:
            pi = np.asarray(ts.actor.policy, np.float64)      # [T, B, A] behaviour policy
        else:
            # Apply THIS agent's network to the reference agent's states.
            pi_j, _, _, _ = jax.vmap(agent.network.apply, (None, 0), 0)(
                agent.params, ts.env)
            pi = np.asarray(pi_j, np.float64)
        legal = np.asarray(ts.env.legal, np.float64)          # [T, B, A]
        valid = np.asarray(ts.env.valid, bool)                # [T, B]
        if not valid.any():
            continue
        p = pi[valid]
        lg = legal[valid]
        # entropy over the legal support
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0, np.log(np.maximum(p, 1e-12)), 0.0)
        ents.append(-(p * logp).sum(axis=-1))
        n_legal = lg.sum(axis=-1)
        ceils.append(np.log(np.maximum(n_legal, 1)))
        legals.append(n_legal)
        maxps.append(p.max(axis=-1))
    ent = np.concatenate(ents); ceil = np.concatenate(ceils)
    nl = np.concatenate(legals); mx = np.concatenate(maxps)
    print(f"{label:34s} H={ent.mean():.4f} nats  ceiling(log|A_legal|)={ceil.mean():.4f}  "
          f"H/ceiling={ent.mean()/max(ceil.mean(),1e-9):.3f}  "
          f"max_action_prob={mx.mean():.3f}  |A_legal|={nl.mean():.1f}  n={len(ent)}")


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) < 2:
        print("usage: policy_entropy.py CKPT [CKPT ...] "
              "[--batch N] [--traj N] [--history compact|sparse]")
        raise SystemExit(2)
    args, kw, ref_path = [], {}, None
    it = iter(_s.argv[1:])
    for a in it:
        if a == "--ref":
            ref_path = next(it)
        elif a == "--batch":
            kw["batch_size"] = int(next(it))
        elif a == "--traj":
            kw["trajectory_max"] = int(next(it))
        elif a == "--history":
            kw["history_encoding"] = next(it)
        else:
            args.append(a)
    ref_agent = load_agent(ref_path, **kw) if ref_path else None
    if ref_agent is not None:
        print(f"[states drawn from reference: {pathlib.Path(ref_path).stem}]")
    for path in args:
        entropy_report(path, pathlib.Path(path).stem, batches=1, ref=ref_agent, **kw)
