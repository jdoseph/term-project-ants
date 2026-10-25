"""
One experimental trial, shared by tuning and the final experiments.

World: wall with two gaps symmetric about the nest-food axis. With the shelf,
the lower route is longer (A*: 52 vs 99 steps); without it they are equal.
Phase 1 (PRE ticks): colony runs, we record which gap it uses.
Phase 2 (POST ticks): the UPPER (short) gap is sealed; we record recovery.
"""
import numpy as np
from ants import Colony, Config, two_gap_world, seal_gap
from astar import astar

WORLD = dict(height=120, width=160, home=(60, 50), food=(60, 110),
             home_radius=8, food_radius=10, n_ants=500,
             max_steps_from_source=1600)
WALL_X, UP_Y, LO_Y, HALF = 80, 30, 90, 5
SHELF = (72, 83, 120)
PRE, POST = 5000, 4000
BIN = 250


def make_colony(seed, baffle=True, **overrides):
    cfg = Config(seed=seed, **WORLD, **overrides)
    cfg.obstacles = two_gap_world(cfg, baffle=SHELF if baffle else None)
    return Colony(cfg)


def route_lengths(baffle=True):
    c = make_colony(0, baffle)
    b0 = c.blocked.copy()
    seal_gap(c, WALL_X, LO_Y, HALF); up, _ = astar(c.blocked, WORLD["home"], WORLD["food"], 10)
    c.blocked = b0.copy()
    seal_gap(c, WALL_X, UP_Y, HALF); lo, _ = astar(c.blocked, WORLD["home"], WORLD["food"], 10)
    return up, lo


def _count(col, s):
    on = (col.x >= WALL_X - 2) & (col.x <= WALL_X + 5)
    s["up"] += int((on & (np.abs(col.y - UP_Y) <= HALF)).sum())
    s["lo"] += int((on & (np.abs(col.y - LO_Y) <= HALF)).sum())


def run_trial(seed, baffle=True, seal=True, pre=PRE, post=POST, snap_pre=(), snap_post=(), **overrides):
    c = make_colony(seed, baffle, **overrides)
    early = {"up": 0, "lo": 0}; late = {"up": 0, "lo": 0}
    snaps = {}
    for t in range(1, pre + 1):
        c.step()
        if t % 5 == 0:
            if t <= 1000: _count(c, early)
            if t > pre - 2000: _count(c, late)
        if t in snap_pre:
            snaps[("pre", t)] = (c.food_pher.copy(), c.home_pher.copy())
    arr = np.array(c.trip_lengths, float).reshape(-1, 2)
    def window(a, b):
        m = (arr[:, 0] > a) & (arr[:, 0] <= b)
        return int(m.sum()), (float(np.median(arr[m, 1])) if m.any() else float("nan"))
    pre_rate, pre_len = window(pre - 1000, pre)
    out = dict(seed=seed,
               early_up=100 * early["up"] / max(early["up"] + early["lo"], 1),
               late_up=100 * late["up"] / max(late["up"] + late["lo"], 1),
               pre_rate=pre_rate, pre_len=pre_len)
    if seal:
        seal_gap(c, WALL_X, UP_Y, HALF)
        for t in range(pre + 1, pre + post + 1):
            c.step()
            if (t - pre) in snap_post:
                snaps[("post", t - pre)] = (c.food_pher.copy(), c.home_pher.copy())
        arr = np.array(c.trip_lengths, float).reshape(-1, 2)   # refresh BEFORE windowing
        post_rate, post_len = window(pre + post - 1000, pre + post)
        edges = np.arange(pre - 1000, pre + post + BIN, BIN)
        counts, _ = np.histogram(arr[:, 0], bins=edges) if len(arr) else (np.zeros(len(edges) - 1), None)
        out.update(post_rate=post_rate, post_len=post_len, curve=counts.tolist())
    out["snaps"] = snaps
    return out
