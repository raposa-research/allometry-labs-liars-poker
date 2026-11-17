"""Plot update variance vs dominated-action mass, per action-space multiplier."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
d = json.loads((HERE / "mass_sweep.json").read_text())
mults = sorted({v["mult"] for v in d.values()})
cmap = plt.get_cmap("viridis")
col = {m: cmap(i / max(len(mults) - 1, 1)) for i, m in enumerate(mults)}

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
for m in mults:
    rows = sorted([v for v in d.values() if v["mult"] == m], key=lambda r: r["mass"])
    x = np.array([r["mass"] for r in rows])
    n = np.array([r["neurd"]["relvar"] for r in rows])
    p = np.array([r["softmax_pg"]["relvar"] for r in rows])
    ax[0].plot(x, n, "o-", color=col[m], lw=2, ms=5, label=f"{m}x")
    ax[1].plot(x, p, "s-", color=col[m], lw=2, ms=5, label=f"{m}x")
    ax[2].plot(x, n / p, "^-", color=col[m], lw=2, ms=5, label=f"{m}x")

for a, t, yl in (
        (ax[0], "NeuRD update (UNCLIPPED)", r"relative variance  tr(Cov)/$\|\mathbb{E}\|^2$"),
        (ax[1], "softmax policy gradient", r"relative variance  tr(Cov)/$\|\mathbb{E}\|^2$"),
        (ax[2], "ratio NeuRD / softmax PG", "variance ratio")):
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("probability mass on DOMINATED actions")
    a.set_ylabel(yl); a.set_title(t, fontsize=12)
    a.grid(alpha=.3, which="both")
    a.legend(title="action-space\nmultiplier", fontsize=9, title_fontsize=9)
    # the optimum is m -> 0 (left edge)
    a.annotate("optimum\n(m→0)", xy=(0.001, a.get_ylim()[0]), xytext=(0.0012, a.get_ylim()[0]*1.6),
               fontsize=8, color="#b00", ha="left")

fig.suptitle("Update variance vs mass on dominated actions — trained 1x 3x3x2 NeuRD policy "
             "embedded in a mult-x `lose` action space\n"
             "Real-action strategy, states and returns held fixed; only m and mult vary. "
             "NeuRD (unclipped, 1/mu all-actions) diverges as the policy approaches the optimum; softmax PG does not.",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.90))
out = HERE / "mass_sweep.png"
fig.savefig(out, dpi=150)
print("wrote", out)
