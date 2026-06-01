import numpy as np
from gymnasium.spaces import Discrete
from simenv.pvz_sim import (
    Scene, Move, config, WaveZombieSpawner,
    Sunflower, Peashooter, Wallnut, Potatomine,
)

MAX_ZOMBIE_HP = 10000
MAX_SUN = 10000
SUN_NORM = 200.0


class SimPVZEnv:
    """
    Simplified PVZ simulation environment with DDQN-compatible interface.

    Replaces both PVZEnv and DDQNEnvAdapter — directly outputs flat state
    vectors and provides mask_available_actions().

    State vector (95 dims for 5x9 grid with 4 plants):
      [plant_grid(45), zombie_hp_grid(45), plant_availability(4), sun_norm(1)]
    """

    def __init__(self):
        self.plant_deck = {
            "sunflower": Sunflower,
            "peashooter": Peashooter,
            "wall-nut": Wallnut,
            "potatomine": Potatomine,
        }
        self.rows = config.N_LANES       # 5
        self.cols = config.LANE_LENGTH   # 9
        self.num_cards = len(self.plant_deck)  # 4
        self.grid_size = self.rows * self.cols  # 45

        self.action_space = Discrete(
            self.num_cards * self.rows * self.cols + 1)  # 181
        self.action_space.n = self.action_space.n  # handy attribute

        self._plant_names = list(self.plant_deck)
        self._plant_classes = [
            self.plant_deck[n].__name__ for n in self.plant_deck]
        self._plant_no = {
            self._plant_classes[i]: i for i in range(self.num_cards)}

        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._steps = 0
        self._last_mask = None
        self._collect_render = False
        self._render_data = []  # stored per-frame render info for last episode

    @property
    def steps(self):
        return self._steps

    def enable_render_collection(self):
        self._collect_render = True

    def disable_render_collection(self):
        self._collect_render = False

    @property
    def render_data(self):
        return self._render_data

    def reset(self, **kwargs):
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._steps = 0
        self._last_mask = self.mask_available_actions()
        if self._collect_render:
            self._render_data = [self._capture_frame()]
        return self._build_state()

    def step(self, action):
        # Execute action
        self._take_action(action)

        # Advance simulation until player can act or game ends
        self._scene.step()
        if self._collect_render:
            self._render_data.append(self._capture_frame())
        reward = self._scene.score
        episode_over = self._scene._chrono > config.MAX_FRAMES
        while (not self._scene.move_available()) and (not episode_over):
            self._scene.step()
            if self._collect_render:
                self._render_data.append(self._capture_frame())
            episode_over = self._scene._chrono > config.MAX_FRAMES
            reward += self._scene.score

        episode_over = episode_over or (self._scene.lives <= 0)
        state = self._build_state()
        self._last_mask = self.mask_available_actions()
        self._steps += 1
        return state, float(reward), bool(episode_over), {}

    def mask_available_actions(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True  # no-op always available
        empty_cells, available_plants = self._scene.get_available_moves()
        if len(empty_cells[0]) == 0:
            return mask
        base = (empty_cells[0] + self.rows * empty_cells[1]) * self.num_cards
        for plant in available_plants:
            idx = base + self._plant_no[plant.__name__] + 1
            mask[idx] = True
        return mask

    def close(self):
        pass

    def _build_state(self):
        """Raw state vector matching original PVZEnv_V2._get_obs().

        Returns int64[95]:
          [0:45]   plant_grid   — categorical 0=empty, 1=SF, 2=Pea, 3=Wall, 4=Mine
          [45:90]  zombie_grid  — raw zombie HP sum per cell
          [90]     sun          — raw sun (capped at MAX_SUN)
          [91:95]  action_avail — 0/1 mask per card
        """
        plant_grid = np.zeros(self.grid_size, dtype=int)
        zombie_grid = np.zeros(self.grid_size, dtype=int)
        for plant in self._scene.plants:
            idx = plant.lane * self.cols + plant.pos
            plant_grid[idx] = self._plant_no[plant.__class__.__name__] + 1
        for zombie in self._scene.zombies:
            idx = zombie.lane * self.cols + zombie.pos
            zombie_grid[idx] += int(zombie.hp)

        action_avail = np.array([
            self._scene.plant_cooldowns[name] <= 0
            for name in self._plant_names
        ], dtype=bool)
        action_avail *= np.array([
            self._scene.sun >= self.plant_deck[name].COST
            for name in self._plant_names
        ], dtype=bool)

        return np.concatenate(
            [plant_grid, zombie_grid,
             np.array([min(self._scene.sun, MAX_SUN)], dtype=int),
             action_avail.astype(int)]).astype(np.int64)

    def _take_action(self, action):
        if action > 0:
            action -= 1
            plant_idx = action % self.num_cards
            grid_idx = action // self.num_cards
            lane = grid_idx % self.rows
            pos = grid_idx // self.rows
            move = Move(self._plant_names[plant_idx], lane, pos)
            if move.is_valid(self._scene):
                move.apply_move(self._scene)

    def _capture_frame(self):
        """Capture current scene state for later visualization."""
        zombies = [[] for _ in range(config.N_LANES)]
        plants = [[] for _ in range(config.N_LANES)]
        projectiles = [[] for _ in range(config.N_LANES)]
        for z in self._scene.zombies:
            zombies[z.lane].append((z.__class__.__name__, int(z.pos), z.get_offset(), z.hp))
        for p in self._scene.plants:
            plants[p.lane].append((p.__class__.__name__, p.pos, p.hp))
        for proj in self._scene.projectiles:
            if hasattr(proj, '_render') and proj._render():
                offset = getattr(proj, '_offset', 0)
                pos = getattr(proj, '_pos', proj.pos if hasattr(proj, 'pos') else 0)
                projectiles[proj.lane].append((proj.__class__.__name__, int(pos), float(offset)))
        return {
            "zombies": zombies,
            "plants": plants,
            "projectiles": projectiles,
            "sun": self._scene.sun,
            "score": self._scene.score,
            "cooldowns": {n: int(self._scene.plant_cooldowns[n] / config.FPS) + 1
                          for n in self._plant_names},
            "time": int(self._scene._chrono / config.FPS),
            "lives": self._scene.lives,
        }
