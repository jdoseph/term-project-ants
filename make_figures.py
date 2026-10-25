import pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from analyze import load, recovered, IDEAL, q
from trial import make_colony, WORLD, WALL_X, UP_Y, HALF
from ants import seal_gap

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
import os; os.makedirs("figures", exist_ok=True)
d = load()
FLOOR = np.median([r["pre_rate"] for r in d["no_scent"]]); FUNC = 3 * FLOOR
func = lambda r: r["pre_rate"] >= FUNC

# ---------------- Fig 1: scent fields, baseline seed 0 ----------------
import os
_sp = "snaps_baseline_seed0.pkl" if os.path.exists("snaps_baseline_seed0.pkl") else os.path.join("results", "snaps_baseline_seed0.pkl")
snaps = pickle.load(open(_sp, "rb"))
c0 = make_colony(0, True); open_blocked = c0.blocked.copy()
seal_gap(c0, WALL_X, UP_Y, HALF); sealed_blocked = c0.blocked.copy()
panels = [(("pre", 1500), open_blocked, "(a) before seal, tick 1,500"),
          (("pre", 5000), open_blocked, "(b) before seal, tick 5,000"),
          (("post", 500), sealed_blocked, "(c) 500 ticks after seal"),
          (("post", 4000), sealed_blocked, "(d) 4,000 ticks after seal")]
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95))
for ax, (key, blocked, title) in zip(axes, panels):
    f, h = snaps[key]
    fi = np.clip(f / 2.5, 0, 1); hi = np.clip(h / 2.5, 0, 1)
    img = np.stack([1 - hi, 1 - fi - hi, 1 - fi], axis=-1).clip(0, 1)
    img[blocked] = 0.25
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.add_patch(Circle((WORLD["home"][1], WORLD["home"][0]), WORLD["home_radius"], fill=False, ec="green", lw=1.0))
    ax.add_patch(Circle((WORLD["food"][1], WORLD["food"][0]), WORLD["food_radius"], fill=False, ec="darkorange", lw=1.0))
    ax.set_title(title); ax.axis("off")
fig.tight_layout(pad=0.3)
fig.savefig("figures/fig1_fields.pdf"); fig.savefig("figures/fig1_fields.png", dpi=200); plt.close(fig)

# ---------------- Fig 2: recovery curves, baseline ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.35))
x = (np.arange(20) * 250 - 1000)
for r in d["baseline"]:
    y = np.array(r["curve"]) * 4
    ok = recovered(r)
    ax.plot(x, y, lw=0.9, color=("tab:green" if ok else "tab:red"), alpha=0.8)
ax.axhline(FLOOR, ls=":", color="gray", lw=0.9)
ax.text(3980, FLOOR + 90, "random-search floor", ha="right", fontsize=6, color="gray")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set(xlabel="ticks relative to seal", ylabel="food arrivals per 1,000 ticks", xlim=(-1000, 4000), ylim=(-100, 3400))
ax.text(60, 3120, "seal", fontsize=6.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color="tab:green", lw=1.2, label="recovered (4 seeds)"),
                   Line2D([0], [0], color="tab:red", lw=1.2, label="did not recover (6 seeds)")], loc="upper right", frameon=False)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig2_recovery.pdf"); fig.savefig("figures/fig2_recovery.png", dpi=200); plt.close(fig)

# ---------------- Fig 3: route preference by condition ----------------
order = [("control_equal", "equal routes (control)"), ("baseline", "baseline"),
         ("decay_0.008", "deposit decay 0.008"), ("decay_0.03", "deposit decay 0.03"),
         ("abl_frustration", "+ frustration state"), ("abl_single_field", "one scent only"),
         ("abl_adjacent", "adjacent sensing only"), ("decay_0.0", "no deposit decay"),
         ("evap_0.0", "no evaporation"), ("no_scent", "no scent at all")]
fig, ax = plt.subplots(figsize=(3.4, 2.9))
rng = np.random.default_rng(1)
for i, (k, lab) in enumerate(order):
    yy = len(order) - 1 - i
    for r in d[k]:
        jit = rng.uniform(-0.18, 0.18)
        ax.scatter(r["late_up"], yy + jit, s=11, marker="o", facecolor=("k" if func(r) else "none"),
                   edgecolor="k", lw=0.8, zorder=3)
    ax.plot([q([r["late_up"] for r in d[k]], 50)] * 2, [yy - 0.35, yy + 0.35], color="tab:blue", lw=1.6, zorder=2)
ax.axvline(50, color="gray", ls=":", lw=0.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([lab for _, lab in order][::-1])
ax.set(xlabel="% of gap use via the short gap", xlim=(-2, 102))
ax.scatter([], [], s=11, facecolor="k", edgecolor="k", label="working colony"); ax.scatter([], [], s=11, facecolor="none", edgecolor="k", label="not working")
ax.plot([], [], color="tab:blue", lw=1.6, label="median"); ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig3_preference.pdf"); fig.savefig("figures/fig3_preference.png", dpi=200); plt.close(fig)

# ---------------- Fig 4: sweeps ----------------
def sweep(ax, conds, labels, title, xlabel):
    works, recs, prefs, rtxt = [], [], [], []
    for c in conds:
        rs = d[c]; f = [r for r in rs if func(r)]
        works.append(100 * len(f) / len(rs))
        if len(f) >= 5:                      # only trust rates when >=5 colonies worked
            k = sum(recovered(r) for r in f)
            recs.append(100 * k / len(f)); rtxt.append(f"{k}/{len(f)}")
            prefs.append(q([r["late_up"] for r in f], 50))
        else:
            recs.append(np.nan); rtxt.append(""); prefs.append(np.nan)
    xs = np.arange(len(conds))
    ax.bar(xs, works, color="0.86", width=0.62, label="colonies that work (%)")
    ax.plot(xs, prefs, "s-", color="tab:blue", ms=3.5, lw=1.0, label="short-gap use, median (%)")
    ax.plot(xs, recs, "o-", color="tab:red", ms=3.5, lw=1.0, label="recovered after seal (%)")
    for xv, yv, t_ in zip(xs, recs, rtxt):
        if t_: ax.annotate(t_, (xv, yv), textcoords="offset points", xytext=(0, 5 if yv < 15 else -10), ha="center", fontsize=5.8, color="tab:red")
    ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set(ylim=(0, 108), xlabel=xlabel, title=title)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), sharey=True)
sweep(axes[0], ["decay_0.0", "decay_0.003", "decay_0.008", "baseline", "decay_0.03", "decay_0.06"],
      ["0", "0.003", "0.008", "0.015", "0.03", "0.06"], "(a) how fast an ant's scent weakens as it walks", "deposit decay")
sweep(axes[1], ["evap_0.0", "evap_0.005", "evap_0.015", "baseline", "evap_0.06"],
      ["0", "0.005", "0.015", "0.03", "0.06"], "(b) how fast scent evaporates", "evaporation rate")
axes[0].set_ylabel("percent")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
fig.savefig("figures/fig4_sweeps.pdf"); fig.savefig("figures/fig4_sweeps.png", dpi=200); plt.close(fig)
print("figures written")
