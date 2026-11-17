"""How much probability mass does an agent waste on DUMMY actions?

The action-space scaling probe (config.dummy_mult) adds dummy actions that each
resolve to a uniformly random legal REAL action. A dummy is therefore a mixture
over existing pure strategies and can never beat the best real action, so a
learner that copes with the enlarged action space drives the mass it puts on
dummies toward 0. A learner defeated by action-space size leaves that mass
spread across them.

This is the direct read on the hypothesis, and it is sharper than the exploit-
ability score: at 16x an agent could score badly for reasons unrelated to the
distractors, whereas dummy mass isolates "did the enlarged action space itself
defeat the learner".

Reference point: a policy that has learnt NOTHING is uniform over the legal
set, which puts

    dummy_mass_uniform = (#legal dummies) / (#legal actions)

on dummies -- at 16x on 3x3x2 that is ~0.98, because dummies are always legal
while most real bids are not. So compare the measured mass against that
baseline, NOT against 0.

Also reports entropy restricted to REAL actions, renormalised: an agent can
have low total entropy simply by concentrating on dummies, so real-action
entropy is what says whether it learnt a strategy.

Usage:
  dummy_mass.py CKPT [CKPT ...] [--batches N] [--traj N] [--history compact]
"""
import sys

import jax
import numpy as np

from policy_entropy import load_agent


def report(path, label, batches=2, **kw):
    agent = load_agent(path, **kw)
    cfg = agent._jax_config
    n_real = cfg.num_real_actions
    mult = getattr(cfg, "dummy_mult", 1)

    dm, du, hr, mx, nl = [], [], [], [], []
    for _ in range(batches):
        ts = agent.collect_batch_trajectory()
        valid = np.asarray(ts.env.valid, bool)
        if not valid.any():
            continue
        pi = np.asarray(ts.actor.policy, np.float64)[valid]
        legal = np.asarray(ts.env.legal, np.float64)[valid]
        dm.append(pi[:, n_real:].sum(-1))                     # mass on dummies
        n_legal = legal.sum(-1)
        n_legal_dummy = legal[:, n_real:].sum(-1)
        du.append(n_legal_dummy / np.maximum(n_legal, 1))     # uniform baseline
        nl.append(n_legal)
        # entropy over REAL actions only, renormalised to a distribution
        pr = pi[:, :n_real]
        z = pr.sum(-1, keepdims=True)
        prn = np.where(z > 1e-12, pr / np.maximum(z, 1e-12), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.where(prn > 0, np.log(np.maximum(prn, 1e-300)), 0.0)
        hr.append(-(prn * lg).sum(-1))
        mx.append(pi.max(-1))

    dm = np.concatenate(dm); du = np.concatenate(du)
    hr = np.concatenate(hr); mx = np.concatenate(mx); nl = np.concatenate(nl)
    n_real_legal = np.maximum(nl - du * nl, 1)
    ceil_real = np.log(n_real_legal)
    print(f"{label:34s} mult={mult:<3d} |A|={cfg.num_actions:<4d} "
          f"dummy_mass={dm.mean():.4f}  (uniform would be {du.mean():.4f})  "
          f"ratio={dm.mean()/max(du.mean(),1e-9):.3f}  "
          f"H_real={hr.mean():.3f}/{ceil_real.mean():.3f}  "
          f"max_p={mx.mean():.3f}  n={len(dm)}")
    return dict(label=label, dummy_mult=mult, num_actions=cfg.num_actions,
                dummy_mass=float(dm.mean()), uniform_mass=float(du.mean()),
                ratio=float(dm.mean() / max(du.mean(), 1e-9)),
                h_real=float(hr.mean()), h_real_ceiling=float(ceil_real.mean()),
                max_prob=float(mx.mean()))


if __name__ == "__main__":
    a = {"batches": "2", "traj": "24", "history": "compact"}
    pos = []
    it = iter(sys.argv[1:])
    for x in it:
        if x.startswith("--"):
            a[x[2:]] = next(it)
        else:
            pos.append(x)
    if not pos:
        print(__doc__); raise SystemExit(2)
    import pathlib
    for p in pos:
        q = pathlib.Path(p)
        report(p, f"{q.parent.name}/{q.stem}", batches=int(a["batches"]),
               trajectory_max=int(a["traj"]), history_encoding=a["history"],
               batch_size=512)
