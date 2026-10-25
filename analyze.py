"""
Summarise results.jsonl. Rules fixed before looking at results:
  optimal route lengths (A*): upper 52, lower 75  -> ideal rate ratio 52/75
  RECOVERED := post_rate >= 0.5 * (52/75) * pre_rate     (half the ideal)
  RECOVERY TIME := first 250-tick bin after the seal that starts two consecutive
                   bins each holding >= 0.5*(52/75)*pre_rate*(250/1000) arrivals
  LOCKED-IN SHORT := late_up >= 80 (percent of gap occupancy via the short gap)
"""
import json, collections
import numpy as np

L_UP, L_LO = 52, 75
IDEAL = L_UP / L_LO
BIN, PREBINS = 250, 4          # curve starts 1000 ticks before the seal

def load(path="results.jsonl"):
    import os
    if not os.path.exists(path):
        path = os.path.join("results", "results.jsonl")
    d = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line); d[r["cond"]].append(r)
    for k in d: d[k].sort(key=lambda r: r["seed"])
    return d

def recovered(r):
    return "post_rate" in r and r["post_rate"] >= 0.5 * IDEAL * r["pre_rate"]

def recovery_time(r):
    if "curve" not in r: return None
    need = 0.5 * IDEAL * r["pre_rate"] * BIN / 1000
    post = r["curve"][PREBINS:]
    for i in range(len(post) - 1):
        if post[i] >= need and post[i + 1] >= need:
            return i * BIN
    return None

def q(a, p): return float(np.percentile(a, p)) if len(a) else float("nan")

def summarize(rs):
    late = [r["late_up"] for r in rs]; pre = [r["pre_rate"] for r in rs]
    eff = [L_UP / r["pre_len"] for r in rs if r["pre_len"] == r["pre_len"]]
    s = dict(n=len(rs),
             late_med=q(late, 50), late_lo=q(late, 25), late_hi=q(late, 75),
             locked=sum(x >= 80 for x in late), lockedlow=sum(x <= 20 for x in late),
             pre_med=q(pre, 50), eff_med=q(eff, 50))
    if "post_rate" in rs[0]:
        rec = [recovered(r) for r in rs]
        s.update(rec=sum(rec),
                 post_med=q([r["post_rate"] for r in rs], 50),
                 ratio_med=q([r["post_rate"] / max(r["pre_rate"], 1) for r in rs], 50),
                 rt=[recovery_time(r) for r in rs])
    return s

if __name__ == "__main__":
    d = load()
    order = ["baseline","control_equal","no_scent","abl_single_field","abl_adjacent","abl_frustration",
             "decay_0.0","decay_0.003","decay_0.008","decay_0.03","decay_0.06",
             "evap_0.0","evap_0.005","evap_0.015","evap_0.06"]
    print(f"A*: upper={L_UP} lower={L_LO}  ideal rate ratio={IDEAL:.3f}\n")
    print(f"{'condition':18s} n  shortpref med [IQR]    lockedS lockedL  preRate  trip-eff  rec  postRate  post/pre")
    for c in order:
        if c not in d: continue
        s = summarize(d[c])
        line = f"{c:18s} {s['n']:2d} {s['late_med']:5.1f} [{s['late_lo']:4.1f},{s['late_hi']:5.1f}]   {s['locked']:2d}      {s['lockedlow']:2d}     {s['pre_med']:6.0f}   {s['eff_med']:5.2f}"
        if "rec" in s: line += f"    {s['rec']:2d}/{s['n']}  {s['post_med']:6.0f}    {s['ratio_med']:5.2f}"
        print(line)
    if "baseline" in d:
        print("\nbaseline per seed (late_up, pre_rate, post_rate, recovered, recovery_time):")
        for r in d["baseline"]:
            print(f"  seed {r['seed']}: {r['late_up']:5.1f}  {r['pre_rate']:5d}  {r['post_rate']:5d}  {recovered(r)}  {recovery_time(r)}")
