"""Total per-decision update variance: within-state + between-state.

enumerate_all.py reported only E_s[Var(g|s)] -- the action-draw term. The law
of total variance says

    Var(g) = E_s[ Var(g|s) ]  +  Var_s( E[g|s] )
             \_ within _/        \_ between _/

Both are computable from the same data: embed every state's conditional mean
E[g|s] into the FULL action space (zero on illegal actions) so vectors from
states with different legal sets live in one space, then take the variance
across states, weighted by visit count to approximate the self-play
visitation distribution.

    M       = sum_s w_s E[g|s] / sum_s w_s
    within  = sum_s w_s tr(Cov[g|s]) / sum_s w_s
    between = sum_s w_s ||E[g|s] - M||^2 / sum_s w_s
    relvar  = (within + between) / ||M||^2

CAVEAT that does not go away: states are kept only if visited >= min_visits,
because delta for REAL actions is estimated from the visits where they were
drawn. That filter discards rare states, which is exactly where the
between-state term is largest, so `between` here is a LOWER bound. (The within
term is unaffected in kind -- it is computed analytically per state.)

Still logit-space, not parameter-space: no network Jacobian is applied.
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from enumerate_var import gather

KEYS = ("neurd_unclipped", "neurd_clipped", "ppo")


def decompose(G, n_real, beta, mode, min_visits=12):
    acc = {k: {"E": [], "tr": [], "w": []} for k in KEYS}
    A_full = None
    PIMIN, PIDUM = [], []
    for _, vs in G.items():
        if len(vs) < min_visits:
            continue
        w = float(len(vs))                       # visit count = state weight
        P = np.stack([x[0] for x in vs]).mean(0)
        Q = np.stack([x[1] for x in vs])
        V = float(np.mean([x[2] for x in vs]))
        Eta = np.stack([x[3] for x in vs]).mean(0)
        Cen = np.stack([x[4] for x in vs]).mean(0)
        L = vs[0][5].astype(bool)
        Sm = np.stack([x[6] for x in vs])
        A_full = L.shape[0]

        idx = np.where(L)[0]
        pi = P[idx]; pi = pi / max(pi.sum(), 1e-12)
        u = V + Eta[idx]; cen = Cen[idx]; is_d = idx >= n_real

        delta = np.zeros(len(idx)); real_delta = {}
        for j, a in enumerate(idx):
            if is_d[j]:
                continue
            hit = Sm[:, a]
            if hit.any():
                delta[j] = float(np.mean((Q[hit, a] - u[j]) * max(P[a], 1e-12)))
            real_delta[int(a)] = delta[j]
        if mode == "lose":
            delta[is_d] = -1.0 - V
        elif mode == "alias":
            for j, a in enumerate(idx):
                if is_d[j]:
                    delta[j] = real_delta.get(int((a - n_real) % n_real), 0.0)

        PIMIN.append(float(pi.min()))
        if is_d.any(): PIDUM.append(float(pi[is_d].mean()))
        ubar = float(pi @ u); cu = u - ubar; n = len(idx)
        Xn = np.empty((n, n)); Xp = np.empty((n, n))
        for t in range(n):
            Xn[t] = cu - delta[t]
            Xn[t, t] += delta[t] / max(pi[t], 1e-12)
            A_t = Eta[idx][t] + delta[t]
            Xp[t] = -A_t * pi
            Xp[t, t] += A_t
        Xc = (np.where(cen[None, :] > -beta, np.minimum(Xn, 0.0), 0.0)
              + np.where(cen[None, :] < beta, np.maximum(Xn, 0.0), 0.0))

        for k, X in (("neurd_unclipped", Xn), ("neurd_clipped", Xc), ("ppo", Xp)):
            Ex = pi @ X
            full = np.zeros(A_full); full[idx] = Ex          # embed in R^A
            acc[k]["E"].append(full)
            acc[k]["tr"].append(float(pi @ (X * X).sum(1)) - float(Ex @ Ex))
            acc[k]["w"].append(w)

    out = {}
    for k in KEYS:
        E = np.stack(acc[k]["E"]); tr = np.array(acc[k]["tr"])
        w = np.array(acc[k]["w"]); w = w / w.sum()
        M = (w[:, None] * E).sum(0)
        within = float((w * tr).sum())
        between = float((w * ((E - M) ** 2).sum(1)).sum())
        nM2 = float(M @ M)
        out[k] = {"within": within, "between": between, "normM2": nM2,
                  "relvar_within_only": within / max(nM2, 1e-300),
                  "relvar_total": (within + between) / max(nM2, 1e-300),
                  "between_frac": between / max(within + between, 1e-300)}
    out["n_states"] = len(acc["ppo"]["w"])
    out["pi_min_mean"] = float(np.mean(PIMIN))      # mean over states of min legal pi
    out["pi_dummy_mean"] = float(np.mean(PIDUM)) if PIDUM else float("nan")
    return out


SETTINGS = [("none  1x   b=2.0",   "dummy-3x3x2-neurd-x1",    2000000, 2.0,   "none"),
            ("lose  2x   b=2.0",   "lose-s-neurd-b2-x2",      2000000, 2.0,   "lose"),
            ("lose  4x   b=2.0",   "lose-s-neurd-b2-x4",      2000000, 2.0,   "lose"),
            ("lose  8x   b=2.0",   "lose-fix-3x3x2-neurd-x8", 2000000, 2.0,   "lose"),
            ("lose  2x   b=2.385", "lose-s-neurd-badj-x2",    2000000, 2.385, "lose"),
            ("lose  4x   b=2.749", "lose-s-neurd-badj-x4",    2000000, 2.749, "lose"),
            ("lose  8x   b=3.105", "lose-s-neurd-badj-x8",    2000000, 3.105, "lose")]

if __name__ == "__main__":
    res = {}
    hdr = (f"{'setting':18s}{'states':>7s}{'upd':>17s}{'within':>12s}"
           f"{'between':>12s}{'btw%':>7s}{'relvar_tot':>12s}")
    print(hdr); print("-" * len(hdr), flush=True)
    for label, run, step, beta, mode in SETTINGS:
        try:
            G, n_real, _ = gather(run, step)
            d = decompose(G, n_real, beta, mode)
            res[label] = d
            for k in ("neurd_unclipped", "ppo"):
                print(f"{label:18s}{d['n_states']:>7d}{k:>17s}"
                      f"{d[k]['within']:>12.4g}{d[k]['between']:>12.4g}"
                      f"{d[k]['between_frac']*100:>6.1f}%"
                      f"{d[k]['relvar_total']:>12.4g}", flush=True)
            r_w = d["neurd_unclipped"]["relvar_within_only"] / d["ppo"]["relvar_within_only"]
            r_t = d["neurd_unclipped"]["relvar_total"] / d["ppo"]["relvar_total"]
            print(f"{'':18s}{'':>7s}{'RATIO N/P':>17s}"
                  f"{'within-only ' + f'{r_w:.1f}x':>24s}"
                  f"{'   total ' + f'{r_t:.1f}x':>24s}\n", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); res[label] = {"error": str(e)}
    pathlib.Path(__file__).with_name("total_var.json").write_text(json.dumps(res, indent=1))
    print("wrote total_var.json")
