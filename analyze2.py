import json, numpy as np
from scipy import stats
from analyze import load, recovered, recovery_time, IDEAL, L_UP, L_LO, q
d = load()
floor = np.median([r["pre_rate"] for r in d["no_scent"]])
FUNC = 3 * floor
print(f"no-scent (random search) median throughput = {floor:.0f}/1000 ticks;  functional threshold = {FUNC:.0f}\n")
def functional(r): return r["pre_rate"] >= FUNC
order = ["baseline","control_equal","no_scent","abl_single_field","abl_adjacent","abl_frustration",
         "decay_0.0","decay_0.003","decay_0.008","decay_0.03","decay_0.06",
         "evap_0.0","evap_0.005","evap_0.015","evap_0.06"]
print(f"{'condition':18s} func  pref_med(all)  pref_med(func)  locked(func)  rec(func)  recTimes(func)   postEff(func,rec)")
rows = {}
for c in order:
    rs = d[c]; f = [r for r in rs if functional(r)]
    prefA = q([r["late_up"] for r in rs], 50)
    prefF = q([r["late_up"] for r in f], 50) if f else float("nan")
    lockF = sum(r["late_up"] >= 80 for r in f)
    if "post_rate" in rs[0]:
        rec = [r for r in f if recovered(r)]
        rt = sorted(recovery_time(r) for r in rec if recovery_time(r) is not None)
        pe = [L_LO / r["post_len"] for r in rec if r["post_len"] == r["post_len"]]
        pes = f"{q(pe,50):.2f}" if pe else "-"
        print(f"{c:18s} {len(f):2d}/10   {prefA:6.1f}        {prefF:6.1f}         {lockF:2d}/{len(f):<2d}       {len(rec):2d}/{len(f):<2d}      {rt}   {pes}")
        rows[c] = (len(f), len(rec))
    else:
        print(f"{c:18s} {len(f):2d}/10   {prefA:6.1f}        {prefF:6.1f}         {lockF:2d}/{len(f):<2d}       (no seal)")
        rows[c] = (len(f), None)

print("\n--- tests ---")
b = [r["late_up"] for r in d["baseline"]]; ce = [r["late_up"] for r in d["control_equal"]]
print("baseline vs equal-route control, short-gap preference: MWU p =", f"{stats.mannwhitneyu(b, ce).pvalue:.4f}")
print("control seeds late_up:", [round(x) for x in ce])
print("deviation from 50% (|x-50|): baseline median", round(np.median(np.abs(np.array(b)-50)),1), " control median", round(np.median(np.abs(np.array(ce)-50)),1))
def fisher(a, na, b2, nb): return stats.fisher_exact([[a, na-a],[b2, nb-b2]]).pvalue
for lo, hi in [("decay_0.008","decay_0.03"), ("evap_0.015","evap_0.06"), ("baseline","abl_frustration"), ("decay_0.008","baseline"), ("evap_0.015","baseline")]:
    (nf1,k1),(nf2,k2) = rows[lo], rows[hi]
    print(f"recovery {lo} {k1}/{nf1} vs {hi} {k2}/{nf2}: Fisher p = {fisher(k1,nf1,k2,nf2):.4f}")
# trend across the functional decay range
xs=[];ys=[]
for c,dv in [("decay_0.008",0.008),("baseline",0.015),("decay_0.03",0.03)]:
    for r in d[c]:
        if functional(r): xs.append(dv); ys.append(int(recovered(r)))
print("decay (0.008,0.015,0.03) vs recovered, Spearman:", stats.spearmanr(xs,ys))
xs=[];ys=[]
for c,ev in [("evap_0.015",0.015),("baseline",0.03),("evap_0.06",0.06)]:
    for r in d[c]:
        if functional(r): xs.append(ev); ys.append(int(recovered(r)))
print("evaporation (0.015,0.03,0.06) vs recovered, Spearman:", stats.spearmanr(xs,ys))
# baseline details
print("\nbaseline pre-seal: median throughput", q([r['pre_rate'] for r in d['baseline']],50),
      " median trip len", q([r['pre_len'] for r in d['baseline']],50), "(optimal 52)")
rec=[r for r in d["baseline"] if recovered(r)]
print("baseline recovered seeds:", [r['seed'] for r in rec], " post median trip len:", [r['post_len'] for r in rec], "(optimal 75)")
print("baseline early_up median (first 1000 ticks):", q([r['early_up'] for r in d['baseline']],50), " control early_up:", q([r['early_up'] for r in d['control_equal']],50))
# A* cost
from trial import make_colony, WORLD, WALL_X, UP_Y, HALF
from ants import seal_gap
from astar import astar
c = make_colony(0, True); L1,n1 = astar(c.blocked, WORLD["home"], WORLD["food"], 10)
old = c.blocked.copy(); seal_gap(c, WALL_X, UP_Y, HALF)
# stale plan: does the old optimal path still exist? it used the upper gap, which is now blocked
L2,n2 = astar(c.blocked, WORLD["home"], WORLD["food"], 10)
print(f"\nA*: before seal {L1} steps ({n1} nodes expanded); after seal with updated map {L2} steps ({n2} nodes)")
print("n ants:", WORLD["n_ants"])
