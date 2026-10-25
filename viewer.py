"""
Live view of the colony in the two-gap world (the demo).

    python viewer.py                  # shelf world: short gap (top) vs long gap (bottom)
    python viewer.py --seal 5000      # seal the short gap at tick 5000, watch it re-route
    python viewer.py --control        # equal-length routes (no shelf)

Red = food scent (laid by ants carrying food), blue = home scent (laid by
outbound ants). Dark = both. Grey = walls. Green circle = nest, orange = food.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from trial import make_colony, WORLD, WALL_X, UP_Y, HALF
from ants import seal_gap

p = argparse.ArgumentParser()
p.add_argument("--control", action="store_true", help="equal-length routes (no shelf)")
p.add_argument("--seal", type=int, default=0, help="tick at which to seal the short gap")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps-per-frame", type=int, default=5)
args = p.parse_args()

col = make_colony(args.seed, baffle=not args.control)
H, W = col.cfg.height, col.cfg.width

fig, ax = plt.subplots(figsize=(9, 6.8))
im = ax.imshow(np.ones((H, W, 3)), origin="upper", interpolation="nearest")
dots, = ax.plot([], [], ".", ms=2.2, color="black", alpha=0.45)
ax.add_patch(Circle((WORLD["home"][1], WORLD["home"][0]), WORLD["home_radius"], fill=False, ec="green", lw=1.5))
ax.add_patch(Circle((WORLD["food"][1], WORLD["food"][0]), WORLD["food_radius"], fill=False, ec="darkorange", lw=1.5))
title = ax.set_title("")
ax.axis("off")


def frame(_):
    for _ in range(args.steps_per_frame):
        if args.seal and col.tick == args.seal:
            seal_gap(col, WALL_X, UP_Y, HALF)
        col.step()
    f = np.clip(col.food_pher / 2.5, 0, 1)
    h = np.clip(col.home_pher / 2.5, 0, 1)
    img = np.stack([1 - h, 1 - f - h, 1 - f], axis=-1).clip(0, 1)
    img[col.blocked] = 0.25
    im.set_data(img)
    dots.set_data(col.x, col.y)
    recent = sum(1 for t, _ in col.trip_lengths if t > col.tick - 1000)
    title.set_text(f"tick {col.tick}    food arrivals in last 1,000 ticks: {recent}    "
                   f"carrying food: {int(col.carrying.sum())}/{col.cfg.n_ants}")
    return im, dots, title


ani = FuncAnimation(fig, frame, interval=20, blit=False, cache_frame_data=False)
plt.show()
