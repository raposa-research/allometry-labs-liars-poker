"""How many opening bids have a STATISTICALLY SIGNIFICANT positive advantage,
per training setting?

A(a|h) = Q(h,a) - V(h) with V(h) = sum_b pi(b|h) Q(h,b), both from forced-
opening Monte Carlo rollouts. Because V contains Q_a itself,

    A_a = (1 - pi_a) Q_a  -  sum_{b != a} pi_b Q_b

so, treating the per-action rollout means as independent,

    se(A_a)^2 = (1 - pi_a)^2 se_a^2 + sum_{b != a} pi_b^2 se_b^2

The independence assumption is CONSERVATIVE here: all forced actions reuse the
same dealt hands (common random numbers), so the Q_b are positively correlated
and the true se(A) is smaller than this. Significant counts are therefore a
lower bound.

One-sided test (the question is A > 0, not A != 0). 180 tests per arm means
~9 false positives at raw p<.05, so Benjamini-Hochberg FDR is the headline
number and raw p is shown only for comparison.
"""
import csv, io, json, pathlib, sys
import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).parent
SRC = sys.argv[1] if len(sys.argv) > 1 else "forced_opening_adv_se.json"
D = 3
nm = lambda a: "CHAL" if a == 0 else f"{(a-1)//D+1}x{(a-1)%D+1}"


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


d = json.loads((HERE / SRC).read_text())
rows = []
for arm, o in d.items():
    for hd, v in o.items():
        acts = sorted(v["A"], key=int)
        pi = np.array([v["pi"][a] for a in acts])
        se = np.array([v["Q_se"][a] for a in acts])
        for i, a in enumerate(acts):
            w = -pi.copy()
            w[i] += 1.0
            sA = float(np.sqrt(np.sum((w * se) ** 2)))
            A = float(v["A"][a])
            z = A / sA if sA > 0 else 0.0
            rows.append(dict(arm=arm, hand=hd, bid=nm(int(a)), action=int(a),
                             count=(int(a) - 1) // D + 1,
                             digit=(int(a) - 1) % D + 1, n=int(v["n"][a]),
                             pi=float(v["pi"][a]), Q=float(v["Q"][a]),
                             Q_se=float(v["Q_se"][a]), V=float(v["V"]),
                             V_se=float(v["V_se"]), A=A, se_A=sA, z=z,
                             p_one=float(stats.norm.sf(z))))

arms = list(d)
for arm in arms:
    R = [r for r in rows if r["arm"] == arm]
    for r, x in zip(R, bh([r["p_one"] for r in R])):
        r["q_bh"] = float(x)

print("Statistically significant POSITIVE-advantage openings, per setting")
print("(3x3x2, 2M steps; one-sided z on A = Q - V; conservative se)\n")
hdr = (f"{'setting':>16}{'cells':>7}{'A>0':>6}{'p<.05':>7}{'p<.01':>7}"
       f"{'BH q<.05':>10}{'BH q<.10':>10}{'max z':>8}{'max A':>8}")
print(hdr); print("-" * len(hdr))
for arm in arms:
    R = [r for r in rows if r["arm"] == arm]
    z = np.array([r["z"] for r in R]); p = np.array([r["p_one"] for r in R])
    q = np.array([r["q_bh"] for r in R]); A = np.array([r["A"] for r in R])
    print(f"{arm:>16}{len(R):>7}{int((A > 0).sum()):>6}{int((p < .05).sum()):>7}"
          f"{int((p < .01).sum()):>7}{int((q < .05).sum()):>10}"
          f"{int((q < .10).sum()):>10}{z.max():>8.2f}{A.max():>+8.3f}")
print(f"\n  ~{0.05*len(rows)/len(arms):.0f} raw p<.05 hits expected per arm by chance; "
      f"use the BH columns.")

print("\nSame, split by reachability:")
hdr2 = (f"{'setting':>16}{'group':>11}{'cells':>7}{'A>0':>6}{'p<.05':>7}"
        f"{'BH q<.05':>10}{'max A':>8}")
print(hdr2); print("-" * len(hdr2))
for arm in arms:
    for lab, f in [("reachable", lambda r: r["pi"] >= 0.01),
                   ("frozen", lambda r: r["pi"] < 0.01)]:
        R = [r for r in rows if r["arm"] == arm and f(r)]
        if not R:
            print(f"{arm:>16}{lab:>11}{0:>7}")
            continue
        A = np.array([r["A"] for r in R]); p = np.array([r["p_one"] for r in R])
        q = np.array([r["q_bh"] for r in R])
        print(f"{arm:>16}{lab:>11}{len(R):>7}{int((A > 0).sum()):>6}"
              f"{int((p < .05).sum()):>7}{int((q < .05).sum()):>10}{A.max():>+8.3f}")

print("\nEvery FROZEN cell (pi<0.01) significant at BH q<0.10 -- the actions the "
      "policy cannot reach but should:")
S = [r for r in rows if r["pi"] < 0.01 and r["q_bh"] < 0.10]
if S:
    h3 = (f"{'setting':>16}{'hand':>10}{'bid':>6}{'cnt':>4}{'pi':>10}{'A':>8}"
          f"{'se_A':>7}{'z':>7}{'BH q':>8}{'n':>7}")
    print(h3); print("-" * len(h3))
    for r in sorted(S, key=lambda r: -r["z"]):
        print(f"{r['arm']:>16}{r['hand']:>10}{r['bid']:>6}{r['count']:>4}"
              f"{r['pi']:>10.1e}{r['A']:>+8.3f}{r['se_A']:>7.3f}{r['z']:>7.2f}"
              f"{r['q_bh']:>8.3f}{r['n']:>7}")
else:
    print("  none in any arm")

F = ["arm", "hand", "bid", "action", "count", "digit", "n", "pi", "Q", "Q_se",
     "V", "V_se", "A", "se_A", "z", "p_one", "q_bh"]
b = io.StringIO(); w = csv.DictWriter(b, fieldnames=F); w.writeheader()
for r in rows:
    w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in F})
(HERE / "a_vs_pi_points.csv").write_text(b.getvalue())
print(f"\nwrote a_vs_pi_points.csv ({len(rows)} rows)  <- the scatter itself")

# ---- companion: the per-arm numbers quoted in the figure annotations ----
N_LEGAL, BETA = 18, 2.0
hi = 1 / (1 + (N_LEGAL - 1) * np.exp(-2 * BETA))
lo = 1 / (1 + (N_LEGAL - 1) * np.exp(+2 * BETA))
S = ["arm", "n_cells", "n_reachable", "n_frozen", "min_pi", "max_pi",
     "pi_span_ratio", "frozen_pi_min", "frozen_pi_max", "frozen_span_ratio",
     "n_upper_left", "n_upper_left_sig", "n_sig_total", "n_sig_reachable",
     "n_sig_frozen", "max_A", "mean_A_frozen", "spearman_logpi_A",
     "spearman_p", "beta_pi_floor", "beta_pi_ceiling"]
srows = []
for arm in arms:
    R = [r for r in rows if r["arm"] == arm]
    pi = np.array([r["pi"] for r in R]); A = np.array([r["A"] for r in R])
    q = np.array([r["q_bh"] for r in R])
    fz = pi < 0.01
    ul = fz & (A > 0)
    sig = (q < 0.05) & (A > 0)
    sp = stats.spearmanr(np.log10(np.maximum(pi, 1e-300)), A)
    isn = arm.startswith("NeuRD")
    srows.append({
        "arm": arm, "n_cells": len(R), "n_reachable": int((~fz).sum()),
        "n_frozen": int(fz.sum()), "min_pi": f"{pi.min():.6g}",
        "max_pi": f"{pi.max():.6g}", "pi_span_ratio": f"{pi.max()/pi.min():.6g}",
        "frozen_pi_min": f"{pi[fz].min():.6g}", "frozen_pi_max": f"{pi[fz].max():.6g}",
        "frozen_span_ratio": f"{pi[fz].max()/pi[fz].min():.6g}",
        "n_upper_left": int(ul.sum()), "n_upper_left_sig": int((ul & sig).sum()),
        "n_sig_total": int(sig.sum()), "n_sig_reachable": int((sig & ~fz).sum()),
        "n_sig_frozen": int((sig & fz).sum()), "max_A": f"{A.max():.6g}",
        "mean_A_frozen": f"{A[fz].mean():.6g}",
        "spearman_logpi_A": f"{sp.statistic:.6g}", "spearman_p": f"{sp.pvalue:.3g}",
        "beta_pi_floor": f"{lo:.6g}" if isn else "",
        "beta_pi_ceiling": f"{hi:.6g}" if isn else ""})
b2 = io.StringIO(); w2 = csv.DictWriter(b2, fieldnames=S)
w2.writeheader(); w2.writerows(srows)
(HERE / "a_vs_pi_summary.csv").write_text(b2.getvalue())
print(f"wrote a_vs_pi_summary.csv ({len(srows)} rows)  <- annotation numbers")
