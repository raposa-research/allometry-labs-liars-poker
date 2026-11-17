"""Exact action-draw enumeration across every NeuRD setting.

Same algebra as enumerate_var.py; see that file's docstring. The only thing
that varies per setting is delta_a, the return residual if a were drawn:

  lose  dummies : EXACT.  The action terminates the episode with the taker
                  losing, so delta_a = -1 - v.
  alias dummies : EXACT BY EQUIVALENCE.  A dummy is a deterministic duplicate
                  of real action (a - n_real) mod n_real, so it inherits that
                  action's delta -- which is itself estimated, so the alias
                  rows are only as good as the real-action estimates.
  real actions  : estimated per (state, action) from visits where the action
                  was actually drawn; 0 if never drawn.

So the `lose` rows rest on an exact delta for 7/8 of the action space, the
alias and 1x rows do not. Flagged in the output.
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from enumerate_var import gather

KEYS = ("neurd_unclipped", "neurd_clipped", "ppo")


def moments(G, n_real, beta, mode, min_visits=12):
    per = {k: [] for k in KEYS}
    n_states = 0
    for _, vs in G.items():
        if len(vs) < min_visits:
            continue
        n_states += 1
        P = np.stack([x[0] for x in vs]).mean(0)
        Q = np.stack([x[1] for x in vs])
        V = float(np.mean([x[2] for x in vs]))
        Eta = np.stack([x[3] for x in vs]).mean(0)
        Cen = np.stack([x[4] for x in vs]).mean(0)
        L = vs[0][5].astype(bool)
        Sm = np.stack([x[6] for x in vs])

        idx = np.where(L)[0]
        pi = P[idx]; pi = pi / max(pi.sum(), 1e-12)
        u = V + Eta[idx]
        cen = Cen[idx]
        is_d = idx >= n_real

        # --- delta_a -----------------------------------------------------
        delta = np.zeros(len(idx))
        # real actions first (alias dummies inherit from these)
        real_delta = {}
        for j, a in enumerate(idx):
            if is_d[j]:
                continue
            hit = Sm[:, a]
            if hit.any():
                delta[j] = float(np.mean((Q[hit, a] - u[j]) * max(P[a], 1e-12)))
            real_delta[int(a)] = delta[j]
        if mode == "lose":
            delta[is_d] = -1.0 - V                       # exact
        elif mode == "alias":
            for j, a in enumerate(idx):
                if is_d[j]:
                    delta[j] = real_delta.get(int((a - n_real) % n_real), 0.0)
        # mode "none" -> no dummies at all

        ubar = float(pi @ u)
        cu = u - ubar
        n = len(idx)
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
            nE2 = float(Ex @ Ex)
            per[k].append((float(pi @ (X * X).sum(1)) - nE2, nE2))

    out = {"n_states": n_states,
           "legal_mean": float(np.mean([len(np.where(v[0][5].astype(bool))[0])
                                        for v in G.values() if len(v) >= min_visits]))}
    for k in KEYS:
        tc = np.array([x[0] for x in per[k]]); n2 = np.array([x[1] for x in per[k]])
        out[k] = {"pooled": float(tc.sum() / max(n2.sum(), 1e-300)),
                  "median": float(np.median(tc / np.maximum(n2, 1e-300)))}
    return out


SETTINGS = [
    # (label, run, step, beta, mode, exact_delta?)
    ("none  1x   b=2.0",    "dummy-3x3x2-neurd-x1",     2000000, 2.0,   "none",  False),
    ("lose  2x   b=2.0",    "lose-s-neurd-b2-x2",       2000000, 2.0,   "lose",  True),
    ("lose  2x   b=2.385",  "lose-s-neurd-badj-x2",     2000000, 2.385, "lose",  True),
    ("lose  4x   b=2.0",    "lose-s-neurd-b2-x4",       2000000, 2.0,   "lose",  True),
    ("lose  4x   b=2.749",  "lose-s-neurd-badj-x4",     2000000, 2.749, "lose",  True),
    ("lose  8x   b=2.0",    "lose-fix-3x3x2-neurd-x8",  2000000, 2.0,   "lose",  True),
    ("lose  8x   b=3.105*", "lose-s-neurd-badj-x8",     1700000, 3.105, "lose",  True),
    ("alias 8x   b=2.0",    "alias-fix-3x3x2-neurd-x8", 2000000, 2.0,   "alias", False),
]

if __name__ == "__main__":
    res = {}
    hdr = (f"{'setting':20s}{'|A_leg|':>9s}{'states':>8s}"
           f"{'NeuRD unclip':>14s}{'NeuRD clip':>12s}{'PPO':>9s}"
           f"{'unclip/PPO':>12s}{'clip/PPO':>10s}{'exact d':>9s}")
    print(hdr); print("-" * len(hdr), flush=True)
    for label, run, step, beta, mode, exact in SETTINGS:
        try:
            G, n_real, _ = gather(run, step)
            m = moments(G, n_real, beta, mode)
            res[label] = {**m, "run": run, "step": step, "beta": beta,
                          "mode": mode, "exact_delta": exact}
            u = m["neurd_unclipped"]["pooled"]; c = m["neurd_clipped"]["pooled"]
            p = m["ppo"]["pooled"]
            print(f"{label:20s}{m['legal_mean']:>9.1f}{m['n_states']:>8d}"
                  f"{u:>14.4g}{c:>12.4g}{p:>9.4g}"
                  f"{u/max(p,1e-300):>11.1f}x{c/max(p,1e-300):>9.1f}x"
                  f"{'yes' if exact else 'no':>9s}", flush=True)
        except Exception as e:
            res[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{label:20s} FAILED {type(e).__name__}: {e}", flush=True)
    pathlib.Path(__file__).with_name("enumerate_all.json").write_text(
        json.dumps(res, indent=1))
    print("\n* = run not finished; latest checkpoint used, not final."
          "\n'exact d' = delta exact for dummies (lose: terminal -1). "
          "Rows with 'no' rest on estimated real-action deltas.")
    print("wrote enumerate_all.json")
