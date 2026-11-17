"""A(a|h) vs pi(a|h) at the FINAL checkpoint, one panel per training setting.

This is the figure for the original prediction: if the policy is abandoning
good actions, there should be a cluster in the UPPER-LEFT -- positive advantage
at probability near zero -- for PG with no entropy bonus, and nothing there for
NeuRD or PG+bonus.

Every (hand, opening bid) cell is plotted; no selection on A or pi. Error bars
are +-1 se(A) with

    se(A_a)^2 = (1 - pi_a)^2 se_a^2 + sum_{b != a} pi_b^2 se_b^2

(conservative: common random numbers correlate the Q's, shrinking the true se).
Points significant at BH q<0.05 against the one-sided null A<=0 are ringed.
"""
import json, pathlib
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).parent
D = 3
FLOOR = 1e-24          # below the smallest observed pi (8.5e-23), so nothing
                       # is clipped and the x-range is the true one
nm = lambda a: "CHAL" if a == 0 else f"{(a-1)//D+1}x{(a-1)%D+1}"


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


d = json.loads((HERE / "forced_opening_adv_se.json").read_text())
arms = list(d)
data = {}
for arm, o in d.items():
    rec = []
    for hd, v in o.items():
        acts = sorted(v["A"], key=int)
        pi = np.array([v["pi"][a] for a in acts])
        se = np.array([v["Q_se"][a] for a in acts])
        for i, a in enumerate(acts):
            w = -pi.copy(); w[i] += 1.0
            sA = float(np.sqrt(np.sum((w * se) ** 2)))
            A = float(v["A"][a])
            rec.append(dict(hand=hd, bid=nm(int(a)), count=(int(a) - 1) // D + 1,
                            pi=float(v["pi"][a]), A=A, se=sA,
                            p=float(stats.norm.sf(A / sA)) if sA > 0 else 1.0))
    for r, q in zip(rec, bh([r["p"] for r in rec])):
        r["q"] = float(q)
    data[arm] = rec

counts = sorted({r["count"] for v in data.values() for r in v})
cmap = plt.get_cmap("viridis")
col = {c: cmap(i / max(len(counts) - 1, 1)) for i, c in enumerate(counts)}

fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.3), sharey=True)
for ax, arm in zip(axes, arms):
    rec = data[arm]
    ax.axhspan(0, 1.2, xmin=0, xmax=1, color="#fdf0f0", zorder=0)
    ax.axvspan(FLOOR / 2, 0.01, color="#e8e8e8", alpha=.55, zorder=0, lw=0)
    ax.axhline(0, color="#333", lw=1.1, zorder=1)
    ax.axvline(0.01, color="#777", ls=":", lw=1.1, zorder=1)
    for r in rec:
        x = max(r["pi"], FLOOR)
        ax.errorbar(x, r["A"], yerr=r["se"], fmt="o", ms=4.6,
                    color=col[r["count"]], ecolor=col[r["count"]],
                    elinewidth=.9, capsize=1.6, alpha=.8, zorder=3)
        if r["q"] < 0.05 and r["A"] > 0:
            ax.plot(x, r["A"], "o", ms=12, mfc="none", mec="#c1121f", mew=1.9,
                    zorder=5)
    sig = sorted([r for r in rec if r["q"] < 0.05 and r["A"] > 0],
                 key=lambda r: -r["A"])[:3]
    for j, r in enumerate(sig):
        ax.annotate(f"{r['hand'][1:-1].replace(', ','')} {r['bid']}",
                    (max(r["pi"], FLOOR), r["A"]), fontsize=7.5,
                    color="#c1121f", xytext=(11, 10 - 11 * j),
                    textcoords="offset points", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#c1121f", lw=.6))
    nsig = sum(1 for r in rec if r["q"] < 0.05 and r["A"] > 0)
    nfroz_sig = sum(1 for r in rec if r["q"] < 0.05 and r["A"] > 0 and r["pi"] < 0.01)
    ax.set_xscale("log"); ax.set_xlim(FLOOR / 2, 2)
    ax.set_title(f"{arm}\n{nsig} cells with A>0 at BH q<.05  "
                 f"({nfroz_sig} of them frozen)", fontsize=11)
    ax.set_xlabel(r"$\pi(a\,|\,h)$   (log)")
    ax.grid(alpha=.25, lw=.5, zorder=0)
axes[0].set_ylabel(r"$A(a\,|\,h) = Q(h,a) - V(h)$   (game units)")
axes[0].set_ylim(-1.15, 0.42)
axes[0].annotate("THE PREDICTED CLUSTER:\ngood bids the policy cannot reach",
                 (1e-22, 0.39), fontsize=9.5, color="#c1121f", style="italic",
                 ha="left", va="top")
axes[0].annotate(r"frozen $\pi$ spans $7.8\times10^{19}$," "\n"
                 r"down to $8.5\times10^{-23}$",
                 (1e-22, -0.95), fontsize=8.5, color="#333", ha="left")
axes[1].annotate(r"frozen $\pi$ spans $1.3\times10^{10}$," "\n"
                 r"down to $1.9\times10^{-13}$",
                 (1e-22, -0.95), fontsize=8.5, color="#333", ha="left")
axes[2].annotate("hard floor from the " r"$\beta=2$" " logit trust region:\n"
                 "all 93 frozen cells pinned in\n"
                 r"$[2.25,\,2.96]\times10^{-3}$ — a 1.3x range —" "\n"
                 "regardless of how bad A is",
                 (1e-22, -0.62), fontsize=9, color="#1b5e20", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f1f8e9", ec="#7cb342"))

leg = [Line2D([], [], color=col[c], marker="o", ls="none", ms=6,
              label=f"bid count {c}") for c in counts]
leg += [Line2D([], [], color="#c1121f", marker="o", mfc="none", mew=1.9,
               ls="none", ms=10, label="BH q<.05, A>0")]
axes[-1].legend(handles=leg, fontsize=8.5, loc="lower right", ncol=2,
                framealpha=.95)
fig.suptitle(
    "Opening-bid advantage vs probability at 2M steps (3x3x2, all 180 "
    "hand x bid cells per arm, no selection).  Grey = frozen (pi<0.01); "
    "pink = A>0.\n"
    "Upper-left quadrant is the signature of abandoning a good action: PG with no "
    "entropy bonus has 6 significant cells there; the other two arms have none.  "
    "NeuRD instead shows a hard probability floor.",
    fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.90))
out = HERE / "a_vs_pi_final.png"
fig.savefig(out, dpi=150)
print("wrote", out)

print(f"\n{'setting':>16}{'cells':>7}{'upper-left (pi<.01, A>0)':>26}{'of those sig':>14}")
print("-" * 63)
for arm in arms:
    rec = data[arm]
    ul = [r for r in rec if r["pi"] < 0.01 and r["A"] > 0]
    s = [r for r in ul if r["q"] < 0.05]
    print(f"{arm:>16}{len(rec):>7}{len(ul):>26}{len(s):>14}")
