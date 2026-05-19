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

    @property
    def steps(self):
        return self._steps

    def reset(self, **kwargs):
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._steps = 0
        self._last_mask = self.mask_available_actions()
        return self._build_state()

    def step(self, action):
        # Execute action
        self._take_action(action)

        # Advance simulation until player can act or game ends
        self._scene.step()
        reward = self._scene.score
        episode_over = self._scene._chrono > config.MAX_FRAMES
        while (not self._scene.move_available()) and (not episode_over):
            self._scene.step()
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
        """Build flat state vector matching DDQN adapter format."""
        plant_grid = np.zeros(self.grid_size, dtype=np.float32)
        zombie_grid = np.zeros(self.grid_size, dtype=np.float32)
        for plant in self._scene.plants:
            idx = plant.lane * self.cols + plant.pos
            plant_grid[idx] = 1.0  # binary: plant exists
        for zombie in self._scene.zombies:
            idx = zombie.lane * self.cols + zombie.pos
            zombie_grid[idx] += min(zombie.hp, MAX_ZOMBIE_HP)

        # Plant availability from cooldowns + sun
        plant_avail = np.array([
            (self._scene.plant_cooldowns[name] <= 0
             and self._scene.sun >= self.plant_deck[name].COST)
            for name in self._plant_names
        ], dtype=np.float32)

        sun_norm = np.array(
            [min(self._scene.sun, MAX_SUN) / SUN_NORM], dtype=np.float32)

        return np.concatenate(
            [plant_grid, zombie_grid, plant_avail, sun_norm], axis=0)

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
