"""Gradient variance of the NeuRD update vs the PPO update at the SAME checkpoint.

Hypothesis under test: NeuRD training is limited by gradient VARIANCE, not by the
trust region. The concrete mechanism is that v-trace's per-action learning target
carries ``inv_mu = 1/mu(a_t)``:

    q_vr(a_t) = v + eta_log_policy(a_t) + inv_mu * (G_t - v)

NeuRD *needs* that factor (it makes one sampled action an unbiased estimate of
the all-actions Q-vector it centers), but it is ~|A_legal| for a near-uniform
policy -- ~210 at 8x10x3. The PPO path divides it back out and relies on the
ratio pi/mu for the correction instead, so its advantage has no such multiplier.
If variance is the story, NeuRD's gradient should be far noisier at the same
params on the same data.

### Design

Both estimators are evaluated at the SAME params, on the SAME trajectory
batches, with the SAME v-trace machinery -- the only difference is the update
rule. Two solvers are built from one checkpoint (including `params_target` /
`params_prev` / `params_prev_`, which the reward transform and critic need) and
differ only in `policy_update`.

`policy_only=True` drops the critic loss. `loss_v` is identical for both
branches here, so including it would add the same gradient and the same variance
to both and compress any ratio toward 1.

### Metrics (and why)

Raw gradient norms are NOT comparable: the NeuRD loss sums a force over all
legal actions while the PPO loss is a clipped surrogate on one, so they differ
by an arbitrary scale. Every headline number here is scale-invariant -- multiply
either loss by c and the statistic is unchanged:

  * relative variance  R = tr(Cov(g)) / ||E[g]||^2
    Also the standard gradient-noise-scale / critical-batch-size estimator
    B_simple. "How many batches must be averaged before the signal exceeds the
    noise."
  * mean cosine to the mean direction, E[cos(g_i, E[g])]
    How aligned a single batch's update is with the true descent direction.
    1.0 = noiseless, ~0 = pure noise.

CAVEAT ON "UPDATE" vs "GRADIENT": this measures the GRADIENT. The optimizer is
Adam, which rescales per-coordinate by a running second-moment estimate and so
partially normalizes raw scale -- but it does not remove directional noise,
which is what the cosine captures. Read R as a statement about the gradient
signal available, not about the literal parameter delta.

Usage:
  update_variance.py CKPT [--batches N] [--batch-size N] [--traj N]
                          [--history compact] [--out FILE]
"""
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization as fs

from liars_poker_jax import LiarsPokerConfig
from rnad import RNaDConfig
from train_jax import JaxLiarsPokerRNaDSolver


def _read_meta(path):
    if str(path).startswith("s3://"):
        from eval_agents import _read_bytes as _rb
        return fs.msgpack_restore(_rb(str(path)))
    return fs.msgpack_restore(pathlib.Path(path).read_bytes())


def _build(meta, policy_update, batch_size, traj, hist, shape_meta=None,
           seed=42):
    sm = shape_meta if shape_meta is not None else meta
    jc = sm["jax_config"]
    pnl = sm["policy_network_layers"]
    layers = (tuple(pnl[str(i)] for i in range(len(pnl))) if isinstance(pnl, dict)
              else tuple(int(x) for x in pnl))
    cfg = RNaDConfig(game_name="liars_poker", policy_network_layers=layers,
                     batch_size=batch_size, trajectory_max=traj,
                     policy_update=policy_update, advantage_estimator="vtrace",
                     seed=seed,
                     eta_reward_transform=float(
                         sm.get("eta_reward_transform", 0.4)))
    s = JaxLiarsPokerRNaDSolver(
        cfg,
        # dummy_mult changes num_actions, so it MUST be carried through or
        # the policy head and the game disagree on the action-space size.
        LiarsPokerConfig(num_players=int(jc["num_players"]),
                         hand_length=int(jc["hand_length"]),
                         num_digits=int(jc["num_digits"]),
                         dummy_mult=int(jc.get("dummy_mult", 1)),
                         dummy_mode=str(jc.get("dummy_mode", "random"))),
        network_type=str(sm.get("network_type", "mlp")),
        history_encoding=hist or str(sm.get("history_encoding", "sparse")))
    # The reward transform needs prev/prev_ and the critic needs target; a
    # params-only load would leave these at random init and the comparison
    # would be measuring noise from an untrained anchor.
    if meta is not None:
        for attr in ("params", "params_target", "params_prev", "params_prev_"):
            setattr(s, attr, jax.tree_util.tree_map(jnp.asarray, meta[attr]))
    return s


def _share_params(a, b):
    """Force solver b to hold exactly solver a's params (all four trees)."""
    for attr in ("params", "params_target", "params_prev", "params_prev_"):
        setattr(b, attr, getattr(a, attr))


def _flat(g):
    leaves = jax.tree_util.tree_leaves(g)
    return np.concatenate([np.asarray(l, np.float64).ravel() for l in leaves])


def _stats(G):
    """G: [N, D] per-batch gradients. Returns scale-invariant noise stats."""
    mean = G.mean(axis=0)
    nm2 = float(mean @ mean)
    # tr(Cov), unbiased over batches
    dev = G - mean
    tr_cov = float((dev * dev).sum(axis=1).mean() * G.shape[0] / (G.shape[0] - 1))
    cos = [float(g @ mean / (np.linalg.norm(g) * np.sqrt(nm2) + 1e-300)) for g in G]
    return dict(
        grad_norm_mean=float(np.sqrt(nm2)),
        per_batch_norm=float(np.linalg.norm(G, axis=1).mean()),
        tr_cov=tr_cov,
        rel_variance=tr_cov / max(nm2, 1e-300),      # = B_simple
        cos_mean=float(np.mean(cos)),
        cos_sd=float(np.std(cos)),
    )


def inv_mu_stats(solver, ts):
    """1/mu(a_t) on the acting player's valid steps -- the hypothesised culprit."""
    oh = np.asarray(ts.actor.action_oh, np.float64)
    mu = (oh * np.asarray(ts.actor.policy, np.float64)).sum(-1)
    valid = np.asarray(ts.env.valid, bool)
    m = mu[valid]
    m = m[m > 0]
    inv = 1.0 / m
    legal = np.asarray(ts.env.legal, np.float64).sum(-1)[valid]
    return dict(inv_mu_mean=float(inv.mean()), inv_mu_p99=float(np.percentile(inv, 99)),
                inv_mu_max=float(inv.max()), legal_mean=float(legal.mean()))


def synth_meta(game, layers, network, dummy_mult, dummy_mode="random"):
    """Shape metadata for a game specified on the command line.

    Lets random-init variance be measured at any action-space size without a
    checkpoint existing there -- which is the point when sweeping dummy_mult.
    """
    hand, digits, players = (int(x) for x in game.lower().split("x"))
    return {"jax_config": {"num_players": players, "hand_length": hand,
                           "num_digits": digits, "dummy_mult": int(dummy_mult),
                           "dummy_mode": dummy_mode},
            "policy_network_layers": [int(x) for x in layers.split(",")],
            "network_type": network, "history_encoding": "compact",
            "learner_steps": None}


def run(ckpt, n_batches, batch_size, traj, hist, out, ckpt_params=None,
        seed=42, shape_meta=None):
    if shape_meta is None:
        shape_meta = _read_meta(ckpt)
    random_init = str(ckpt_params).lower() == "random"
    meta = None if random_init else shape_meta
    if random_init:
        print(f"params: FRESH RANDOM INIT (shapes/game from {ckpt})")
        print("  NEITHER rule is at its fixed point here, so rel_variance is not\n"
              "  confounded by a vanishing mean gradient -- and a fresh net is the\n"
              "  flat-policy condition at the START of training, which is exactly\n"
              "  where 'variance kills NeuRD' would have to bite.")
    else:
        print(f"checkpoint: {ckpt}\n  learner_steps={shape_meta.get('learner_steps')} "
              f"network={shape_meta.get('network_type')} "
              f"hist={shape_meta.get('history_encoding')}")
    # The solver init is DETERMINISTIC in config.seed: re-running random-init
    # with the same seed reproduces the identical point (verified to 4 dp), so
    # genuine replicates require varying --seed.
    sn = _build(meta, "neurd", batch_size, traj, hist, shape_meta=shape_meta,
                seed=seed)
    sp = _build(meta, "ppo", batch_size, traj, hist, shape_meta=shape_meta,
                seed=seed)
    if random_init:
        _share_params(sn, sp)        # both rules MUST see identical params
    # identical params in both solvers
    for attr in ("params", "params_target", "params_prev", "params_prev_"):
        a = _flat(getattr(sn, attr)); b = _flat(getattr(sp, attr))
        assert np.array_equal(a, b), f"{attr} differs between the two solvers"
    print(f"  both solvers loaded with identical params "
          f"({_flat(sn.params).size:,} policy+value params)")

    gn = jax.jit(jax.grad(lambda p, ts, al: sn.loss(
        p, sn.params_target, sn.params_prev, sn.params_prev_, ts, al, 1_000_000,
        policy_only=True)))
    gp = jax.jit(jax.grad(lambda p, ts, al: sp.loss(
        p, sp.params_target, sp.params_prev, sp.params_prev_, ts, al, 1_000_000,
        policy_only=True)))

    Gn, Gp, imu = [], [], []
    for i in range(n_batches):
        ts = sn.collect_batch_trajectory()      # behaviour policy = the ckpt
        imu.append(inv_mu_stats(sn, ts))
        Gn.append(_flat(gn(sn.params, ts, 1.0)))
        Gp.append(_flat(gp(sp.params, ts, 1.0)))
        if (i + 1) % 4 == 0:
            print(f"  batch {i+1}/{n_batches}")
    Gn = np.stack(Gn); Gp = np.stack(Gp)

    st_n, st_p = _stats(Gn), _stats(Gp)
    im = {k: float(np.mean([d[k] for d in imu])) for k in imu[0]}

    print(f"\n{n_batches} batches of {batch_size} trajectories, traj_max={traj}")
    print(f"inv_mu = 1/mu(a_t) on valid steps: mean={im['inv_mu_mean']:.1f}  "
          f"p99={im['inv_mu_p99']:.1f}  max={im['inv_mu_max']:.1f}  "
          f"|A_legal|={im['legal_mean']:.1f}")
    print(f"\n{'':22s} {'NeuRD':>14s} {'PPO':>14s} {'ratio':>10s}")
    rows = [("||E[g]||", "grad_norm_mean"), ("E||g||", "per_batch_norm"),
            ("tr(Cov)", "tr_cov"),
            ("REL VARIANCE", "rel_variance"), ("cos to E[g]", "cos_mean")]
    for lab, k in rows:
        a, b = st_n[k], st_p[k]
        r = (a / b) if b else float("inf")
        print(f"{lab:22s} {a:14.5g} {b:14.5g} {r:10.3g}")
    print(f"\nscale-invariant headline:")
    print(f"  relative variance tr(Cov)/||E g||^2 : "
          f"NeuRD {st_n['rel_variance']:.4g}  vs  PPO {st_p['rel_variance']:.4g}"
          f"   ({st_n['rel_variance']/max(st_p['rel_variance'],1e-300):.1f}x)")
    print(f"  single-batch cosine to true direction: "
          f"NeuRD {st_n['cos_mean']:.4f} +-{st_n['cos_sd']:.4f}  vs  "
          f"PPO {st_p['cos_mean']:.4f} +-{st_p['cos_sd']:.4f}")

    # NOT meta.get: meta is None in random-init mode, which crashed this line
    # AFTER all the stats had printed (numbers fine, nothing persisted).
    summary = dict(checkpoint=("RANDOM_INIT" if random_init else ckpt),
                   learner_steps=(None if random_init
                                  else shape_meta.get("learner_steps")),
                   n_batches=n_batches, batch_size=batch_size,
                   trajectory_max=traj, seed=seed, inv_mu=im,
                   neurd=st_n, ppo=st_p)
    if out:
        pathlib.Path(out).write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    a = {"batches": "24", "batch-size": "256", "traj": "100",
         "history": "compact", "out": None, "params": None,
         "seed": "42", "game": None, "layers": None,
         "network": None, "dummy-mult": None, "dummy-mode": None}
    pos = []
    it = iter(sys.argv[1:])
    for x in it:
        if x.startswith("--"):
            a[x[2:]] = next(it)
        else:
            pos.append(x)
    sm = None
    if a.get("game"):
        sm = synth_meta(a["game"], a.get("layers") or "256,256,128",
                        a.get("network") or "mlp", a.get("dummy-mult") or 1,
                        a.get("dummy-mode") or "random")
        a["params"] = "random"          # no checkpoint exists for a synth game
        pos = pos or ["<synth>"]
    if not pos:
        print(__doc__); raise SystemExit(2)
    run(pos[0], int(a["batches"]), int(a["batch-size"]), int(a["traj"]),
        a["history"], a["out"], ckpt_params=a.get("params"),
        seed=int(a.get("seed") or 42), shape_meta=sm)
