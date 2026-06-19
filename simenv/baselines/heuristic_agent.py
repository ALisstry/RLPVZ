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
    Returns (None, None, None) for the NO-OP / wait action.
    """
    if action == 0:
        return None, None, None          # wait / no-op
    action -= 1                          # remove no-op offset
    plant_idx = action % num_cards
    grid_idx = action // num_cards
    lane = grid_idx % rows
    pos = grid_idx // rows
    return int(plant_idx), int(lane), int(pos)


def _card_and_cell_to_action(card_idx: int, row: int, col: int,
                             num_cards: int, rows: int, cols: int) -> int:
    """Encode (card_idx, row, col) → discrete action index (0 = wait)."""
    # Action encoding matches SimPVZEnv._take_action:
    #   action = card_idx + num_cards * (row + rows * col) + 1
    return int(card_idx + num_cards * (row + rows * col) + 1)


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


# ──────────────────────────────────────────────────────────────────────────────
# Simple Heuristic Agent
# ──────────────────────────────────────────────────────────────────────────────

class SimpleHeuristicAgent:
    """Always plants the cheapest affordable plant in a random empty cell.
    Falls back to waiting if nothing is affordable.
    """

    @property
    def name(self) -> str:
        return "SimpleHeuristic"

    def select_action(self, env) -> int:
        mask = np.asarray(env.mask_available_actions(), dtype=bool)
        # mask[0] = wait, mask[1:] = plant actions
        plant_mask = mask.copy()
        plant_mask[0] = False

        if not plant_mask.any():
            return 0    # wait

        scene = env._scene
        num_cards = env.num_cards
        rows = env.rows
        cols = env.cols

        # Find the cheapest affordable plant
        plant_names = list(env.plant_deck.keys())
        plant_classes = [env.plant_deck[n] for n in plant_names]

        # Sort plant indices by cost (cheapest first)
        affordable = []
        for i, cls in enumerate(plant_classes):
            cd_remaining = scene.plant_cooldowns.get(plant_names[i], 0)
            if cd_remaining <= 0 and scene.sun >= cls.COST:
                affordable.append((cls.COST, i))
        affordable.sort(key=lambda x: x[0])

        if not affordable:
            return 0    # wait

        # Try each affordable plant type in empty cells
        for _cost, card_idx in affordable:
            for lane in range(rows):
                for pos in range(cols):
                    if scene.grid.is_empty(lane, pos):
                        action = _card_and_cell_to_action(
                            card_idx, lane, pos, num_cards, rows, cols)
                        if action < len(mask) and mask[action]:
                            return action
        return 0    # wait (should not normally reach here)


# ──────────────────────────────────────────────────────────────────────────────
# Smart Heuristic Agent (Tactical Agent)
# ──────────────────────────────────────────────────────────────────────────────

class SmartHeuristicAgent:
    """Domain-knowledge-driven strategy for SimPVZ.

    Decision priorities (checked in order):
    1. ECONOMY — maintain ≥8 sunflowers in back columns (0–2)
    2. OFFENSE — plant peashooters in lanes with zombies (columns 3–5)
    3. DEFENSE — plant wall-nuts against high-HP lanes (columns 6–7)
    4. TRAP    — plant potato mines ahead of tough zombies (columns 7–8)
    5. WAIT    — nothing useful to do
    """

    def __init__(self):
        self._target_sunflowers = 8
        self._sunflower_cols = (0, 3)     # [0, 3) → cols 0,1,2
        self._pea_cols = (3, 6)           # [3, 6) → cols 3,4,5
        self._wall_cols = (6, 8)          # [6, 8) → cols 6,7
        self._mine_cols = (7, 9)          # [7, 9) → cols 7,8

    @property
    def name(self) -> str:
        return "SmartHeuristic"

    # ── helpers that access env internals ─────────────────────────────────

    def _current_sunflower_count(self, env) -> int:
        from simenv.pvz_sim.entities.plants.sunflower import Sunflower
        return sum(isinstance(p, Sunflower) for p in env._scene.plants)

    def _current_peashooter_count(self, env) -> int:
        from simenv.pvz_sim.entities.plants.peashooter import Peashooter
        return sum(isinstance(p, Peashooter) for p in env._scene.plants)

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
        """Try the candidate cells in order; return action or None."""
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

    # ── main decision logic ──────────────────────────────────────────────

    def select_action(self, env) -> int:
        mask = np.asarray(env.mask_available_actions(), dtype=bool)
        plant_mask = mask.copy()
        plant_mask[0] = False    # exclude wait
        if not plant_mask.any():
            return 0

        hp_lanes = _zombie_hp_per_lane(env)
        zombie_lanes = _zombies_per_lane(env)
        rows = env.rows
        cols = env.cols
        num_cards = env.num_cards

        # ── 1. ECONOMY: sunflower in back columns ─────────────────────
        n_sf = self._current_sunflower_count(env)
        if (n_sf < self._target_sunflowers
                and self._can_afford(env, 50)
                and self._is_ready(env, "sunflower")):
            empty = _empty_cells_in_columns(env, *self._sunflower_cols)
            action = self._plant_if_possible(env, 0, empty)  # card_idx 0=sunflower
            if action is not None:
                return action

        # ── 2. OFFENSE: peashooter in lanes with zombies ──────────────
        if self._can_afford(env, 100) and self._is_ready(env, "peashooter"):
            # Prioritise lanes with the most zombie HP, then by column
            lane_order = sorted(
                range(rows),
                key=lambda ln: (-hp_lanes[ln], ln),
            )
            for lane in lane_order:
                if zombie_lanes[lane] > 0 and self._lacks_peashooter(env, lane):
                    # Find empty cell in pea column range, prefer closer
                    # to zombies (right side)
                    for pos in range(self._pea_cols[1] - 1,
                                     self._pea_cols[0] - 1, -1):
                        if env._scene.grid.is_empty(lane, pos):
                            action = _card_and_cell_to_action(
                                1, lane, pos, num_cards, rows, cols)   # card_idx 1=peashooter
                            if action < len(mask) and mask[action]:
                                return action
            # Fallback: any empty cell in pea columns if no specific lane
            # was matched
            empty_pea = _empty_cells_in_columns(env, *self._pea_cols)
            action = self._plant_if_possible(env, 1, empty_pea)
            if action is not None:
                return action

        # ── 3. DEFENSE: wall-nut in high-threat lanes ─────────────────
        if self._can_afford(env, 50) and self._is_ready(env, "wall-nut"):
            lane_order = sorted(
                range(rows),
                key=lambda ln: (-hp_lanes[ln], ln),
            )
            for lane in lane_order:
                if hp_lanes[lane] > 300:   # substantial threat
                    for pos in range(self._wall_cols[1] - 1,
                                     self._wall_cols[0] - 1, -1):
                        if env._scene.grid.is_empty(lane, pos):
                            action = _card_and_cell_to_action(
                                2, lane, pos, num_cards, rows, cols)   # card_idx 2=wall-nut
                            if action < len(mask) and mask[action]:
                                return action

        # ── 4. TRAP: potato mine in front of tough zombies ────────────
        if self._can_afford(env, 25) and self._is_ready(env, "potatomine"):
            for lane in sorted(range(rows),
                               key=lambda ln: (-hp_lanes[ln], ln)):
                if zombie_lanes[lane] > 0:
                    for pos in range(self._mine_cols[1] - 1,
                                     self._mine_cols[0] - 1, -1):
                        if env._scene.grid.is_empty(lane, pos):
                            action = _card_and_cell_to_action(
                                3, lane, pos, num_cards, rows, cols)   # card_idx 3=potato mine
                            if action < len(mask) and mask[action]:
                                return action

        # ── 5. FALLBACK: cheapest affordable plant anywhere ───────────
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

        return 0   # wait
