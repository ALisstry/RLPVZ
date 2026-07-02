"""
Heuristic baseline agents for SimPVZ.

- SimpleHeuristicAgent: always plants the cheapest affordable plant.
- SmartHeuristicAgent: domain-knowledge-driven strategy balancing economy,
  offense, and defense.
"""

import numpy as np
from simenv.pvz_sim import config


def _action_to_card_and_cell(action: int, num_cards: int, rows: int,
                              cols: int) -> tuple[int | None, int | None,
                                                    int | None]:
    """Decode a discrete action index into (card_idx, row, col).
    Returns (None, None, None) for the wait action.
    """
    grid_size = rows * cols
    if action == num_cards * grid_size:
        return None, None, None
    plant_idx = action // grid_size
    grid_idx = action % grid_size
    lane = grid_idx // cols
    pos = grid_idx % cols
    return int(plant_idx), int(lane), int(pos)


def _card_and_cell_to_action(card_idx: int, row: int, col: int,
                             num_cards: int, rows: int, cols: int) -> int:
    """Encode (card_idx, row, col) to card-major action index."""
    return int(card_idx * (rows * cols) + row * cols + col)


def _zombies_per_lane(env, max_col: int | None = None) -> list[int]:
    """Count zombies in each lane, optionally only up to *max_col* (exclusive)."""
    scene = env._scene
    counts = [0] * config.N_LANES
    for z in scene.zombies:
        if z and (max_col is None or z.pos < max_col):
            counts[z.lane] += 1
    return counts


def _zombie_hp_per_lane(env) -> list[int]:
    """Total zombie HP in each lane."""
    scene = env._scene
    hp = [0] * config.N_LANES
    for z in scene.zombies:
        if z:
            hp[z.lane] += max(0, z.hp)
    return hp


def _plants_per_lane(env, plant_class) -> list[int]:
    """Count plants of a given class in each lane."""
    scene = env._scene
    counts = [0] * config.N_LANES
    for p in scene.plants:
        if isinstance(p, plant_class):
            counts[p.lane] += 1
    return counts


def _empty_cells_in_columns(env, col_start: int, col_end: int
                            ) -> list[tuple[int, int]]:
    """Return list of (row, col) for empty cells in column range [start, end)."""
    scene = env._scene
    cells = []
    for lane in range(config.N_LANES):
        for pos in range(col_start, col_end):
            if scene.grid.is_empty(lane, pos):
                cells.append((lane, pos))
    return cells


def _plant_and_pos_in_columns(env, plant_class, col_start: int, col_end: int
                              ) -> list[tuple[int, int]]:
    """Return (lane, pos) of plants of *plant_class* in [col_start, col_end)."""
    scene = env._scene
    result = []
    for p in scene.plants:
        if isinstance(p, plant_class) and col_start <= p.pos < col_end:
            result.append((p.lane, p.pos))
    return result


class SimpleHeuristicAgent:
    """Always plants the cheapest affordable plant in a random empty cell.
    Falls back to waiting if nothing is affordable.
    """

    @property
    def name(self) -> str:
        return "SimpleHeuristic"

    def select_action(self, env) -> int:
        mask = np.asarray(env.mask_available_actions(), dtype=bool)
        plant_mask = mask.copy()
        plant_mask[env.wait_action] = False

        if not plant_mask.any():
            return env.wait_action

        scene = env._scene
        num_cards = env.num_cards
        rows = env.rows
        cols = env.cols

        plant_names = list(env.plant_deck.keys())
        plant_classes = [env.plant_deck[n] for n in plant_names]

        affordable = []
        for i, cls in enumerate(plant_classes):
            cd_remaining = scene.plant_cooldowns.get(plant_names[i], 0)
            if cd_remaining <= 0 and scene.sun >= cls.COST:
                affordable.append((cls.COST, i))
        affordable.sort(key=lambda x: x[0])

        if not affordable:
            return env.wait_action

        for _cost, card_idx in affordable:
            for lane in range(rows):
                for pos in range(cols):
                    if scene.grid.is_empty(lane, pos):
                        action = _card_and_cell_to_action(
                            card_idx, lane, pos, num_cards, rows, cols)
                        if action < len(mask) and mask[action]:
                            return action
        return env.wait_action


class SmartHeuristicAgent:
    """Domain-knowledge-driven strategy for SimPVZ."""

    def __init__(self):
        self._target_sunflowers = 8
        self._sunflower_cols = (0, 3)
        self._pea_cols = (3, 6)
        self._wall_cols = (6, 8)

    @property
    def name(self) -> str:
        return "SmartHeuristic"

    def _current_sunflower_count(self, env) -> int:
        from simenv.pvz_sim.entities.plants.sunflower import Sunflower
        return sum(isinstance(p, Sunflower) for p in env._scene.plants)

    def _lacks_peashooter(self, env, lane: int) -> bool:
        from simenv.pvz_sim.entities.plants.peashooter import Peashooter
        for p in env._scene.plants:
            if isinstance(p, Peashooter) and p.lane == lane:
                return False
        return True

    def _can_afford(self, env, cost: int) -> bool:
        return env._scene.sun >= cost

    def _is_ready(self, env, plant_name: str) -> bool:
        return env._scene.plant_cooldowns.get(plant_name, 999) <= 0

    def _plant_if_possible(self, env, card_idx: int, candidates: list[tuple[int, int]]
                           ) -> int | None:
        num_cards = env.num_cards
        rows = env.rows
        cols = env.cols
        mask = np.asarray(env.mask_available_actions(), dtype=bool)
        for lane, pos in candidates:
            action = _card_and_cell_to_action(
                card_idx, lane, pos, num_cards, rows, cols)
            if action < len(mask) and mask[action]:
                return action
        return None

    def select_action(self, env) -> int:
        mask = np.asarray(env.mask_available_actions(), dtype=bool)
        plant_mask = mask.copy()
        plant_mask[env.wait_action] = False
        if not plant_mask.any():
            return env.wait_action

        hp_lanes = _zombie_hp_per_lane(env)
        zombie_lanes = _zombies_per_lane(env)
        rows = env.rows
        cols = env.cols
        num_cards = env.num_cards

        n_sf = self._current_sunflower_count(env)
        if (n_sf < self._target_sunflowers
                and self._can_afford(env, 50)
                and self._is_ready(env, "sunflower")):
            empty = _empty_cells_in_columns(env, *self._sunflower_cols)
            action = self._plant_if_possible(env, 0, empty)
            if action is not None:
                return action

        if self._can_afford(env, 100) and self._is_ready(env, "peashooter"):
            lane_order = sorted(
                range(rows),
                key=lambda ln: (-hp_lanes[ln], ln),
            )
            for lane in lane_order:
                if zombie_lanes[lane] > 0 and self._lacks_peashooter(env, lane):
                    for pos in range(self._pea_cols[1] - 1,
                                     self._pea_cols[0] - 1, -1):
                        if env._scene.grid.is_empty(lane, pos):
                            action = _card_and_cell_to_action(
                                1, lane, pos, num_cards, rows, cols)
                            if action < len(mask) and mask[action]:
                                return action
            empty_pea = _empty_cells_in_columns(env, *self._pea_cols)
            action = self._plant_if_possible(env, 1, empty_pea)
            if action is not None:
                return action

        if self._can_afford(env, 50) and self._is_ready(env, "wall-nut"):
            lane_order = sorted(
                range(rows),
                key=lambda ln: (-hp_lanes[ln], ln),
            )
            for lane in lane_order:
                if hp_lanes[lane] > 300:
                    for pos in range(self._wall_cols[1] - 1,
                                     self._wall_cols[0] - 1, -1):
                        if env._scene.grid.is_empty(lane, pos):
                            action = _card_and_cell_to_action(
                                4, lane, pos, num_cards, rows, cols)
                            if action < len(mask) and mask[action]:
                                return action

        plant_names = list(env.plant_deck.keys())
        plant_classes = [env.plant_deck[n] for n in plant_names]
        affordable = []
        for i, cls in enumerate(plant_classes):
            cd_remaining = env._scene.plant_cooldowns.get(plant_names[i], 0)
            if cd_remaining <= 0 and env._scene.sun >= cls.COST:
                affordable.append((cls.COST, i))
        affordable.sort(key=lambda x: x[0])

        for _cost, card_idx in affordable:
            empty_cells = _empty_cells_in_columns(env, 0, cols)
            action = self._plant_if_possible(env, card_idx, empty_cells)
            if action is not None:
                return action

        return env.wait_action
