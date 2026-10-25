"""
Ant colony foraging model.

Dual-pheromone stigmergy: outbound ants lay a "home" trail, returning ants lay a
"food" trail. No ant holds a map, a goal, or any representation of the colony.
The only thing that persists across ants is the pheromone field in the world.

Everything is vectorized over ants so parameter sweeps are cheap.
"""

from dataclasses import dataclass, field
import numpy as np

# 8-connected neighbourhood, ordered so that d+1 is a 45-degree turn.
DY = np.array([0, 1, 1, 1, 0, -1, -1, -1])
DX = np.array([1, 1, 0, -1, -1, -1, 0, 1])
TURNS = np.array([-1, 0, 1])  # sensing arc: left, straight, right


@dataclass
class Config:
    # world
    height: int = 160
    width: int = 200
    home: tuple = (80, 20)
    food: tuple = (80, 180)
    home_radius: int = 6
    food_radius: int = 6

    # colony
    n_ants: int = 400
    max_steps_from_source: int = 700   # give up and respawn at home after this

    # pheromone dynamics
    alpha: float = 2.5         # pheromone exponent: exploitation pressure
    base_weight: float = 0.02  # floor weight -> keeps exploration alive
    forward_bias: float = 14.0  # multiplier on going straight (prevents spinning)
    sensor_dist: int = 2       # ants sample pheromone this far ahead, not adjacent
    deposit_q: float = 1.0
    deposit_decay: float = 0.015  # deposit shrinks with distance from source
    evaporation: float = 0.03
    diffusion: float = 0.02
    max_pheromone: float = 6.0
    frustration: float = 0.0   # trail-trust decays as an ant fails to arrive
    single_field: bool = False  # ablation: one shared scent instead of two

    seed: int = 0
    obstacles: np.ndarray = field(default=None, repr=False)


class Colony:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        H, W = cfg.height, cfg.width

        self.blocked = (
            np.zeros((H, W), dtype=bool) if cfg.obstacles is None
            else cfg.obstacles.copy()
        )

        yy, xx = np.mgrid[0:H, 0:W]
        self.home_mask = (yy - cfg.home[0]) ** 2 + (xx - cfg.home[1]) ** 2 <= cfg.home_radius ** 2
        self.food_mask = (yy - cfg.food[0]) ** 2 + (xx - cfg.food[1]) ** 2 <= cfg.food_radius ** 2
        self.blocked &= ~(self.home_mask | self.food_mask)

        self.home_pher = np.zeros((H, W), dtype=np.float32)
        self.food_pher = np.zeros((H, W), dtype=np.float32)

        n = cfg.n_ants
        self.y = np.full(n, cfg.home[0], dtype=np.int32)
        self.x = np.full(n, cfg.home[1], dtype=np.int32)
        self.d = self.rng.integers(0, 8, size=n).astype(np.int32)
        self.carrying = np.zeros(n, dtype=bool)
        self.since_source = np.zeros(n, dtype=np.int32)

        self.tick = 0
        self.trip_lengths = []   # (tick, steps) for each home -> food arrival
        self.returns = []        # (tick, steps) for each food -> home arrival

    # ---------------------------------------------------------------- sensing

    def _choose_direction(self):
        """Vectorized probabilistic step choice over the 3-cell forward arc."""
        cfg = self.cfg
        cand_d = (self.d[:, None] + TURNS[None, :]) % 8          # (n, 3)
        cand_y = self.y[:, None] + DY[cand_d]
        cand_x = self.x[:, None] + DX[cand_d]

        in_bounds = (
            (cand_y >= 0) & (cand_y < cfg.height) &
            (cand_x >= 0) & (cand_x < cfg.width)
        )
        cy = np.clip(cand_y, 0, cfg.height - 1)
        cx = np.clip(cand_x, 0, cfg.width - 1)
        valid = in_bounds & ~self.blocked[cy, cx]

        # Sense at a distance. Sampling the adjacent cell gives almost no
        # gradient signal; a forward sensor offset is what makes trails followable.
        sy = np.clip(self.y[:, None] + DY[cand_d] * cfg.sensor_dist, 0, cfg.height - 1)
        sx = np.clip(self.x[:, None] + DX[cand_d] * cfg.sensor_dist, 0, cfg.width - 1)

        # Outbound ants follow the food trail; returning ants follow the home trail.
        if cfg.single_field:
            target = self.food_pher[sy, sx]
        else:
            target = np.where(
                self.carrying[:, None],
                self.home_pher[sy, sx],
                self.food_pher[sy, sx],
            )

        # Frustration: the longer an ant has gone without reaching its target,
        # the less it trusts the trail. Without this the colony cannot abandon a
        # route that has been blocked -- ants keep re-reinforcing a dead end.
        frustration = 1.0 + cfg.frustration * self.since_source
        base = cfg.base_weight * frustration[:, None]

        w = (base + target) ** cfg.alpha
        w = w * np.where(TURNS[None, :] == 0, cfg.forward_bias, 1.0)
        w = np.where(valid, w, 0.0)

        total = w.sum(axis=1)
        stuck = total <= 0

        # roulette-wheel sample
        r = self.rng.random(len(total)) * np.where(stuck, 1.0, total)
        pick = (r[:, None] >= np.cumsum(w, axis=1)).sum(axis=1)
        pick = np.clip(pick, 0, 2)

        new_d = cand_d[np.arange(len(pick)), pick]
        new_y = cand_y[np.arange(len(pick)), pick]
        new_x = cand_x[np.arange(len(pick)), pick]

        # cornered ants turn around instead of moving
        new_d = np.where(stuck, (self.d + 4) % 8, new_d)
        new_y = np.where(stuck, self.y, new_y)
        new_x = np.where(stuck, self.x, new_x)
        return new_d, new_y, new_x

    # ------------------------------------------------------------------ step

    def step(self):
        cfg = self.cfg
        self.d, self.y, self.x = self._choose_direction()
        self.since_source += 1

        # Deposit. Strength falls off with distance travelled since the source,
        # which is what creates a usable gradient rather than a uniform smear.
        amount = cfg.deposit_q * np.exp(-cfg.deposit_decay * self.since_source)
        if cfg.single_field:
            np.add.at(self.food_pher, (self.y, self.x), amount)
        else:
            out = ~self.carrying
            if out.any():
                np.add.at(self.home_pher, (self.y[out], self.x[out]), amount[out])
            if self.carrying.any():
                c = self.carrying
                np.add.at(self.food_pher, (self.y[c], self.x[c]), amount[c])

        # Arrivals
        at_food = self.food_mask[self.y, self.x] & ~self.carrying
        if at_food.any():
            for s in self.since_source[at_food]:
                self.trip_lengths.append((self.tick, int(s)))
            self.carrying[at_food] = True
            self.since_source[at_food] = 0
            self.d[at_food] = (self.d[at_food] + 4) % 8

        at_home = self.home_mask[self.y, self.x] & self.carrying
        if at_home.any():
            for s in self.since_source[at_home]:
                self.returns.append((self.tick, int(s)))
            self.carrying[at_home] = False
            self.since_source[at_home] = 0
            self.d[at_home] = (self.d[at_home] + 4) % 8

        # Lost ants respawn at home
        lost = self.since_source > cfg.max_steps_from_source
        if lost.any():
            self.y[lost] = cfg.home[0]
            self.x[lost] = cfg.home[1]
            self.d[lost] = self.rng.integers(0, 8, size=int(lost.sum()))
            self.carrying[lost] = False
            self.since_source[lost] = 0

        self._decay()
        self.tick += 1

    def _decay(self):
        cfg = self.cfg
        for f in (self.home_pher, self.food_pher):
            if cfg.diffusion > 0:
                nb = (
                    np.roll(f, 1, 0) + np.roll(f, -1, 0) +
                    np.roll(f, 1, 1) + np.roll(f, -1, 1)
                ) * 0.25
                f *= (1.0 - cfg.diffusion)
                f += cfg.diffusion * nb
            f *= (1.0 - cfg.evaporation)
            np.clip(f, 0.0, cfg.max_pheromone, out=f)
            f[self.blocked] = 0.0

    def run(self, ticks, callback=None, every=1):
        for i in range(ticks):
            self.step()
            if callback and i % every == 0:
                callback(self)
        return self

    # --------------------------------------------------------------- metrics

    def trip_series(self, window=60):
        """Rolling mean outbound trip length, indexed by tick of arrival."""
        if not self.trip_lengths:
            return np.array([]), np.array([])
        arr = np.array(self.trip_lengths, dtype=float)
        ticks, lens = arr[:, 0], arr[:, 1]
        if len(lens) < window:
            return ticks, lens
        kern = np.ones(window) / window
        smooth = np.convolve(lens, kern, mode="valid")
        return ticks[window - 1:], smooth

    def throughput(self, bin_size=200):
        """Successful food arrivals per bin of ticks."""
        if not self.trip_lengths:
            return np.array([]), np.array([])
        ticks = np.array([t for t, _ in self.trip_lengths])
        edges = np.arange(0, self.tick + bin_size, bin_size)
        counts, _ = np.histogram(ticks, bins=edges)
        return edges[:-1], counts

    def convergence_tick(self, tolerance=0.10, window=60, hold=400):
        """First tick after which smoothed trip length stays within
        `tolerance` of its final value for `hold` ticks."""
        ticks, smooth = self.trip_series(window)
        if len(smooth) < 10:
            return None
        final = np.median(smooth[-max(10, len(smooth) // 10):])
        ok = np.abs(smooth - final) <= tolerance * final
        for i in range(len(ok)):
            if ok[i] and ticks[i] + hold <= self.tick:
                seg = ok[(ticks >= ticks[i]) & (ticks <= ticks[i] + hold)]
                if seg.all():
                    return float(ticks[i])
        return None


# ------------------------------------------------------------------ worlds

def double_bridge(cfg: Config, wall_x=100, short_gap_y=80, long_gap_y=25,
                  gap_half=6, thickness=3):
    """Canonical ACO test: a wall with two openings. The short route is a
    straight shot; the long route is a detour. Correct behaviour is
    convergence on the short one."""
    obs = np.zeros((cfg.height, cfg.width), dtype=bool)
    obs[:, wall_x:wall_x + thickness] = True
    obs[short_gap_y - gap_half:short_gap_y + gap_half, wall_x:wall_x + thickness] = False
    obs[long_gap_y - gap_half:long_gap_y + gap_half, wall_x:wall_x + thickness] = False
    return obs


def seal_gap(colony, wall_x=100, gap_y=80, gap_half=6, thickness=3):
    """Block a gap mid-run and clear pheromone there. Used for the
    re-convergence experiment."""
    colony.blocked[gap_y - gap_half:gap_y + gap_half, wall_x:wall_x + thickness] = True
    colony.home_pher[colony.blocked] = 0.0
    colony.food_pher[colony.blocked] = 0.0


def two_gap_world(cfg: Config, wall_x=80, gap_rows=(30, 90), gap_half=5,
                  thickness=3, baffle=None):
    """A wall with two gaps placed symmetrically about the nest-food axis, so
    neither gap is straighter than the other. If `baffle` = (row, x0, x1) is
    given, a horizontal shelf under the food forces ants that use the lower
    gap to detour around its right end. The detour is what makes one route
    genuinely longer; without it the two routes are equal (the control world).

    (An earlier world put one gap on the nest-food axis and the other far off
    it. On an 8-connected grid a vertical detour is absorbed into diagonal
    moves, so both routes had the same optimal length and the test did not
    isolate route length.)"""
    obs = np.zeros((cfg.height, cfg.width), dtype=bool)
    obs[:, wall_x:wall_x + thickness] = True
    for g in gap_rows:
        obs[g - gap_half:g + gap_half, wall_x:wall_x + thickness] = False
    if baffle is not None:
        row, x0, x1 = baffle          # horizontal shelf below the food
        obs[row:row + 2, x0:x1] = True
    return obs
