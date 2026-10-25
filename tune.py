import sys, itertools, time, numpy as np
from trial import *
from trial import _count
def quick(seed, pre=4000, **ov):
    c = make_colony(seed, True, **ov)
    s = {"up":0,"lo":0}
    for t in range(1, pre+1):
        c.step()
        if t % 5 == 0 and t > pre-1500: _count(c, s)
    a = np.array(c.trip_lengths, float).reshape(-1,2)
    m = a[:,0] > pre-1000
    rate = int(m.sum()); med = float(np.median(a[m,1])) if m.any() else float('nan')
    p10 = float(np.percentile(a[m,1],10)) if m.any() else float('nan')
    up = 100*s["up"]/max(s["up"]+s["lo"],1)
    return rate, med, p10, up
grid = eval(sys.argv[1])
keys = list(grid.keys())
t0=time.time()
for combo in itertools.product(*grid.values()):
    ov = dict(zip(keys, combo))
    res = np.array([quick(sd, **ov) for sd in (100,101)])
    r = res.mean(axis=0)
    print(f"{ov}  rate/1k={r[0]:6.0f}  median_len={r[1]:6.0f}  p10={r[2]:5.0f}  upper%={r[3]:5.1f}", flush=True)
print(f"({time.time()-t0:.0f}s)  optimal upper route = 52 steps")
