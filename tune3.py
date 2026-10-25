import sys, time, numpy as np
import trial
from trial import *
from ants import seal_gap
def run(seed, x1, pre=4000, post=6000, **ov):
    trial.SHELF = (72, 83, x1)
    c = make_colony(seed, True, **ov); c.run(pre)
    a = np.array(c.trip_lengths, float).reshape(-1,2)
    pre_rate = int((a[:,0] > pre-1000).sum())
    seal_gap(c, WALL_X, UP_Y, HALF)
    out=[]
    for k in range(6):
        c.run(1000)
        a = np.array(c.trip_lengths, float).reshape(-1,2)
        m = a[:,0] > c.tick-1000
        out.append((int(m.sum()), int(np.median(a[m,1])) if m.any() else None))
    return pre_rate, out
t0=time.time()
for x1 in (120, 126):
    for ev in (0.015, 0.03):
        for sd in (100,101):
            pr,o = run(sd, x1, evaporation=ev, deposit_decay=0.015)
            print(f"shelf_x1={x1} evap={ev} seed={sd} pre={pr}  post(rate,median_len) per 1000 ticks: {o}", flush=True)
print(f"({time.time()-t0:.0f}s)  optimal lower: x1=120 -> 75, x1=126 -> 85")
