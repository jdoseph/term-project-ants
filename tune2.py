import sys, itertools, time, numpy as np
from trial import *
from ants import seal_gap
def quick2(seed, pre=4000, post=3500, **ov):
    c = make_colony(seed, True, **ov); c.run(pre)
    a = np.array(c.trip_lengths, float).reshape(-1,2)
    pre_rate = int((a[:,0] > pre-1000).sum())
    seal_gap(c, WALL_X, UP_Y, HALF); c.run(post)
    a = np.array(c.trip_lengths, float).reshape(-1,2)
    m = a[:,0] > pre+post-1000
    return pre_rate, int(m.sum()), (float(np.median(a[m,1])) if m.any() else float('nan'))
grid = eval(sys.argv[1]); keys=list(grid)
t0=time.time()
for combo in itertools.product(*grid.values()):
    ov=dict(zip(keys,combo))
    r=[quick2(sd, **ov) for sd in (100,101)]
    print(ov, " pre_rate/post_rate/post_median_len per seed:", [(x[0],x[1],round(x[2]) if x[2]==x[2] else None) for x in r], flush=True)
print(f"({time.time()-t0:.0f}s) optimal lower route = 99 steps; ideal rate ratio ~ 52/99 = 0.53")
