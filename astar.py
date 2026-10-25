"""
A* baseline: the representational control.

It is handed the complete map (which cells are walls) and returns the shortest
path in steps. Moves are 8-connected and each costs one step, exactly like an
ant's move, so path lengths are directly comparable to the colony's trip
lengths (steps since leaving the nest).
"""
import heapq
import numpy as np

MOVES = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def astar(blocked, start, goal_center, goal_radius):
    """Shortest path from `start` to any cell within `goal_radius` of
    `goal_center`. Returns (steps, nodes_expanded); steps is None if no path.

    Heuristic: Chebyshev distance to the goal centre minus the radius. Every
    goal cell lies inside the disc, so this never overestimates (admissible)."""
    H, W = blocked.shape
    gy, gx = goal_center
    r2 = goal_radius ** 2

    def h(y, x):
        return max(0, max(abs(y - gy), abs(x - gx)) - goal_radius)

    def is_goal(y, x):
        return (y - gy) ** 2 + (x - gx) ** 2 <= r2

    open_heap = [(h(*start), 0, start)]
    best = {start: 0}
    expanded = 0
    while open_heap:
        f, g, (y, x) = heapq.heappop(open_heap)
        if g > best.get((y, x), 1e18):
            continue
        expanded += 1
        if is_goal(y, x):
            return g, expanded
        for dy, dx in MOVES:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not blocked[ny, nx]:
                ng = g + 1
                if ng < best.get((ny, nx), 1e18):
                    best[(ny, nx)] = ng
                    heapq.heappush(open_heap, (ng + h(ny, nx), ng, (ny, nx)))
    return None, expanded
