"""Update variance vs DOMINATED-ACTION MASS, at fixed real-action strategy.

Controlled version of the scaling result. Take the trained 1x 3x3x2 NeuRD
checkpoint, embed its policy in a mult-x `lose` action space, and move a
controlled fraction m of probability mass onto the dominated (dummy) actions.
Everything else -- the real-action strategy, the state distribution, the
per-action returns -- is held fixed, so any change in variance is attributable
to m and mult alone.

Construction per visited state s (states and quantities come from self-play of
the real 1x checkpoint, via enumerate_var.gather):

  legal set   : in `lose` mode a dummy is legal iff its target real action is,
                so n_legal scales exactly by mult.
  policy      : real a  -> (1-m) * pi_1x(a)
                dummy   -> m / (#legal dummies)      [uniform over dummies]
  delta       : dummy   -> -1 - v            EXACT (terminal, taker loses)
                real a  -> delta_1x(a)       estimated from the 1x checkpoint
  eta_a       : set to 0. At convergence pi ~= pi_reg so the reward-transform
                term vanishes; and for a SYNTHETIC policy pi_reg is undefined,
                so assuming the converged value is the only consistent choice.
                With eta=0, u_a - ubar = 0 and the NeuRD update reduces to
                    Xn[t,a] = -delta_t + 1{a=t} * delta_t / pi_t
                whose mean is exactly the advantage delta_a - mean(delta).

Then the exact moments over the action draw t ~ pi, plus the between-state
term, exactly as in total_var.py.

Note m -> 0 is the OPTIMUM (no mass on dominated actions) and is also where
pi_t -> 0 for dummies, so the 1/pi term diverges. That is the point of the
plot, not an artifact.
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from enumerate_var import gather

MULTS = [2, 4, 8]
MASSES = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.6, 0.8]
MIN_VISITS = 12


def base_states(run="dummy-3x3x2-neurd-x1", step=2000000, batches=24):
    """Per-state (pi_real, delta_real, v, weight) from the real 1x checkpoint."""
    G, n_real, _ = gather(run, step, batches=batches)
    out = []
    for _, vs in G.items():
        if len(vs) < MIN_VISITS:
            continue
        P = np.stack([x[0] for x in vs]).mean(0)
        Q = np.stack([x[1] for x in vs])
        V = float(np.mean([x[2] for x in vs]))
        L = vs[0][5].astype(bool)
        Sm = np.stack([x[6] for x in vs])
        idx = np.where(L)[0]
        pi = P[idx]; pi = pi / max(pi.sum(), 1e-12)
        d = np.zeros(len(idx))
        for j, a in enumerate(idx):
            hit = Sm[:, a]
            if hit.any():
                d[j] = float(np.mean((Q[hit, a] - V) * max(P[a], 1e-12)))
        out.append((pi, d, V, float(len(vs))))
    return out


def moments(states, mult, m):
    """Exact (within, between, ||M||^2) for NeuRD-unclipped and PPO."""
    nmax = max(len(s[0]) for s in states) * mult
    accE = {k: [] for k in ("neurd", "softmax_pg")}
    acctr = {k: [] for k in ("neurd", "softmax_pg")}
    ws = []
    for pi_r, d_r, V, w in states:
        nr = len(pi_r)
        nd = nr * (mult - 1)                       # dummy copies of legal reals
        pi = np.concatenate([(1.0 - m) * pi_r,
                             np.full(nd, m / max(nd, 1))])
        delta = np.concatenate([d_r, np.full(nd, -1.0 - V)])
        n = nr + nd
        # eta = 0 -> u_a - ubar = 0
        # Xn : UNCLIPPED NeuRD.  d loss/d logit_a = adv_a, estimated by the
        #      1/pi one-hot scatter -> Xn[t,a] = -delta_t + 1{a=t} delta_t/pi_t
        # Xp : SOFTMAX POLICY GRADIENT (score function / REINFORCE), the
        #      theoretical counterpart.  dJ/d logit_a = pi_a*adv_a, estimated by
        #      A_t * (1{a=t} - pi_a) with A_t = delta_t.  No ratio, no clipping
        #      -- this is vanilla PG, not the PPO surrogate.
        Xn = np.zeros((n, n)); Xp = np.zeros((n, n))
        for t in range(n):
            Xn[t, :] = -delta[t]
            Xn[t, t] += delta[t] / max(pi[t], 1e-300)
            Xp[t, :] = -delta[t] * pi
            Xp[t, t] += delta[t]
        for k, X in (("neurd", Xn), ("softmax_pg", Xp)):
            Ex = pi @ X
            full = np.zeros(nmax); full[:n] = Ex
            accE[k].append(full)
            acctr[k].append(float(pi @ (X * X).sum(1)) - float(Ex @ Ex))
        ws.append(w)
    w = np.array(ws); w = w / w.sum()
    out = {}
    for k in ("neurd", "softmax_pg"):
        E = np.stack(accE[k]); tr = np.array(acctr[k])
        M = (w[:, None] * E).sum(0)
        within = float((w * tr).sum())
        between = float((w * ((E - M) ** 2).sum(1)).sum())
        nM2 = float(M @ M)
        out[k] = {"within": within, "between": between, "normM2": nM2,
                  "relvar": (within + between) / max(nM2, 1e-300)}
    out["ratio"] = out["neurd"]["relvar"] / max(out["softmax_pg"]["relvar"], 1e-300)
    return out


if __name__ == "__main__":
    print("gathering base states from the trained 1x checkpoint ...", flush=True)
    st = base_states()
    print(f"  {len(st)} infostates with >={MIN_VISITS} visits, "
          f"mean |A_legal| = {np.mean([len(s[0]) for s in st]):.1f}", flush=True)
    res = {}
    hdr = f"{'mult':>5}{'mass m':>9}{'|A_leg|':>9}{'NeuRD relvar':>15}{'softmaxPG':>13}{'ratio':>12}"
    print(hdr); print("-" * len(hdr), flush=True)
    import csv, io
    F = ["mult","A_legal_mean","dominated_mass",
         "neurd_unclipped_within","neurd_unclipped_between","neurd_unclipped_normE2",
         "neurd_unclipped_relvar","softmax_pg_within","softmax_pg_between",
         "softmax_pg_normE2","softmax_pg_relvar","ratio_neurd_over_pg","sd_ratio"]
    csv_rows = []
    for mult in MULTS:
        for m in MASSES:
            r = moments(st, mult, m)
            res[f"{mult}|{m}"] = {"mult": mult, "mass": m, **r}
            al = np.mean([len(s[0]) for s in st]) * mult
            print(f"{mult:>5}{m:>9.4f}{al:>9.1f}{r['neurd']['relvar']:>15.5g}"
                  f"{r['softmax_pg']['relvar']:>13.5g}{r['ratio']:>11.4g}x", flush=True)
            nn, pg = r["neurd"], r["softmax_pg"]
            csv_rows.append({"mult": mult, "A_legal_mean": round(al, 2),
                "dominated_mass": m,
                "neurd_unclipped_within": f"{nn['within']:.6g}",
                "neurd_unclipped_between": f"{nn['between']:.6g}",
                "neurd_unclipped_normE2": f"{nn['normM2']:.6g}",
                "neurd_unclipped_relvar": f"{nn['relvar']:.6g}",
                "softmax_pg_within": f"{pg['within']:.6g}",
                "softmax_pg_between": f"{pg['between']:.6g}",
                "softmax_pg_normE2": f"{pg['normM2']:.6g}",
                "softmax_pg_relvar": f"{pg['relvar']:.6g}",
                "ratio_neurd_over_pg": f"{r['ratio']:.6g}",
                "sd_ratio": f"{r['ratio'] ** 0.5:.6g}"})
    pathlib.Path(__file__).with_name("mass_sweep.json").write_text(json.dumps(res, indent=1))
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=F)
    w.writeheader(); w.writerows(csv_rows)
    pathlib.Path(__file__).with_name("mass_sweep.csv").write_text(buf.getvalue())
    print("\nwrote mass_sweep.json and mass_sweep.csv")
