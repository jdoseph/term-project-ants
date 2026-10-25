"""
Final experiment suite. Resumable: each finished (condition, seed) is appended
to results.jsonl and skipped on rerun.

    python final_experiments.py            # run everything (about 25 min, 1 CPU)

Evaluation seeds are 0-9. Parameters were tuned on separate seeds (100-103).
"""
import json, os, sys, time, pickle
import numpy as np
from trial import run_trial

SEEDS = range(10)
OUT = "results.jsonl"

# (name, kwargs). Everything not listed uses the baseline defaults in ants.py
# (evaporation 0.03, deposit_decay 0.015).
CONDITIONS = [
    ("baseline",        dict()),
    ("control_equal",   dict(baffle=False, seal=False)),
    ("no_scent",        dict(deposit_q=0.0)),
    ("abl_single_field",dict(single_field=True)),
    ("abl_adjacent",    dict(sensor_dist=1)),
    ("abl_frustration", dict(frustration=0.01)),
    ("decay_0.0",       dict(deposit_decay=0.0)),
    ("evap_0.0",        dict(evaporation=0.0)),
    ("decay_0.003",     dict(deposit_decay=0.003)),
    ("decay_0.008",     dict(deposit_decay=0.008)),
    ("decay_0.03",      dict(deposit_decay=0.03)),
    ("decay_0.06",      dict(deposit_decay=0.06)),
    ("evap_0.005",      dict(evaporation=0.005)),
    ("evap_0.015",      dict(evaporation=0.015)),
    ("evap_0.06",       dict(evaporation=0.06)),
]

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        r = json.loads(line); done.add((r["cond"], r["seed"]))

t0 = time.time()
for name, kw in CONDITIONS:
    for seed in SEEDS:
        if (name, seed) in done:
            continue
        kw2 = dict(kw)
        snaps = {}
        if name == "baseline" and seed == 0:
            kw2.update(snap_pre=(500, 1500, 5000), snap_post=(500, 4000))
        r = run_trial(seed, **kw2)
        if r["snaps"]:
            pickle.dump(r["snaps"], open("snaps_baseline_seed0.pkl", "wb"))
        r.pop("snaps")
        r["cond"] = name
        with open(OUT, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(f"[{time.time()-t0:6.0f}s] {name:18s} seed {seed}: late_up={r['late_up']:5.1f}% "
              f"pre_rate={r['pre_rate']:5d} post_rate={r.get('post_rate','-')}", flush=True)
print("ALL DONE", flush=True)
