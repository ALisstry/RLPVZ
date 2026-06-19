import numpy as np

SUN_NORM = 200.0


class _DiscreteActionSpace:
    def __init__(self, n):
        self.n = int(n)


class DDQNSpaceSpec:
    """
    Env-like shape adapter for constructing DDQN networks without opening PVZ.
    """

    def __init__(self, env_spec, scenario_spec=None,
                 use_paper_observation: bool = False):
        self.rows = env_spec.rows
        self.cols = env_spec.cols
        self.num_cards = env_spec.plant_types
        self.action_space = _DiscreteActionSpace(env_spec.action_space_size)
        self.plant_deck = (
            list(scenario_spec.cards) if scenario_spec is not None else []
        )
        self.use_paper_observation = use_paper_observation


def _extract_action_mask(obs, env):
    if isinstance(obs, dict) and "action_mask" in obs:
        return np.asarray(obs["action_mask"], dtype=np.int8)
    if hasattr(env, "_get_action_mask") and hasattr(env, "pvz"):
        try:
            game_state = env.pvz.get_game_state() if env.pvz else None
            return np.asarray(env._get_action_mask(game_state), dtype=np.int8)
        except Exception:
            pass
    return np.ones(env.action_space.n, dtype=np.int8)


def _extract_sun(obs, info):
    if isinstance(info, dict) and "sun" in info:
        return float(info["sun"])
    if isinstance(obs, dict) and "global_features" in obs:
        return float(obs["global_features"][0]) * 9999.0
    return 0.0


def _plant_availability_from_mask(action_mask, num_cards, grid_size):
    mask = np.asarray(action_mask, dtype=bool)
    avail = np.zeros(num_cards, dtype=np.float32)
    for i in range(num_cards):
        start = i * grid_size
        end = start + grid_size
        if end <= mask.size and np.any(mask[start:end]):
            avail[i] = 1.0
    return avail


# ─────────────────────────────────────────────────────────────────────────
# Legacy flat vector (used by the original small DDQN adapter)
# ─────────────────────────────────────────────────────────────────────────

def build_state_vector(obs, action_mask, sun, rows, cols, num_cards):
    """Original simplified state: [plant_grid | zombie_hp | plant_avail | sun]."""
    if not isinstance(obs, dict) or "grid" not in obs:
        raise ValueError("DDQN requires dict observation with 'grid'.")

    grid = obs["grid"]
    if grid.shape[0] != rows or grid.shape[1] != cols:
        raise ValueError(
            f"Grid shape mismatch: expected ({rows},{cols},C), got {grid.shape}."
        )

    plant_grid = (grid[:, :, 0] > 0).astype(np.float32)
    zombie_grid = grid[:, :, 4].astype(np.float32)

    grid_size = rows * cols
    plant_avail = _plant_availability_from_mask(action_mask, num_cards, grid_size)

    sun_norm = float(sun) / float(SUN_NORM)

    state = np.concatenate(
        [
            plant_grid.reshape(-1),
            zombie_grid.reshape(-1),
            plant_avail,
            np.array([sun_norm], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return state


# ─────────────────────────────────────────────────────────────────────────
# Paper-format observation (596-dim flat vector, matching paper Section 4.1)
# Matches the paper spec: 11 (global) + 495 (plant one-hot) + 45 (HP) + 45 (zombie HP) = 596
# ─────────────────────────────────────────────────────────────────────────

# Mapping from raw plant_type_id (grid channel 0) → observation class index (0-10)
# Class 0-9 = plant type as card slot index, 10 = empty / unknown
_PLANT_ID_TO_CLASS: dict[int, int] = {}


def _build_plant_onehot(grid_obs, rows, cols, num_cards,
                        card_plant_ids: list[int]) -> np.ndarray:
    """Build one-hot plant-type encoding: 45 cells × (num_cards + 1) classes.

    Args:
        grid_obs: shape (rows, cols, C) numpy array from env.
        rows, cols: grid dimensions.
        num_cards: number of card slots (= plant types in deck).
        card_plant_ids: list of PlantType IDs ordered by card slot.

    Returns:
        float32 array of shape (rows * cols * (num_cards + 1),).
    """
    global _PLANT_ID_TO_CLASS
    n_classes = num_cards + 1  # +1 for empty
    n_cells = rows * cols

    # Build mapping: raw plant_type_id → class index (0-based)
    if not _PLANT_ID_TO_CLASS:
        for cls_idx, pid in enumerate(card_plant_ids):
            _PLANT_ID_TO_CLASS[int(pid)] = cls_idx

    onehot = np.zeros((rows, cols, n_classes), dtype=np.float32)

    # Channel 0 carries (plant_type + 1) / 50.0 normalised.
    # Recover the approximate plant_type_id: round(50 * val) - 1
    raw_vals = grid_obs[:, :, 0]  # shape (rows, cols), values in [0, 1]
    # The original encoding in PVZEnv is: grid[row, col, 0] = (plant.type + 1) / 50.0
    # So type_id ≈ round(50 * val) - 1, clamped to valid range
    type_ids = np.round(raw_vals * 50.0).astype(int) - 1
    type_ids = np.clip(type_ids, -1, 255)

    for r in range(rows):
        for c in range(cols):
            tid = type_ids[r, c]
            cls_idx = _PLANT_ID_TO_CLASS.get(int(tid), n_classes - 1)
            onehot[r, c, cls_idx] = 1.0

    return onehot.reshape(-1)


def _build_plant_hp(grid_obs, rows, cols, plant_data) -> np.ndarray:
    """Extract plant HP per cell, normalised by each plant type's max HP.

    Returns shape (rows * cols,).
    """
    hp_grid = np.zeros((rows, cols), dtype=np.float32)
    raw_type = np.round(grid_obs[:, :, 0] * 50.0).astype(int) - 1
    raw_hp_ratio = grid_obs[:, :, 1]  # already normalised in env

    plant_hp_max = {}
    if hasattr(plant_data, 'PLANT_HP'):
        plant_hp_max = plant_data.PLANT_HP

    for r in range(rows):
        for c in range(cols):
            tid = raw_type[r, c]
            if tid < 0:
                continue
            hp_ratio = float(raw_hp_ratio[r, c])
            if hp_ratio > 0:
                hp_grid[r, c] = hp_ratio

    return hp_grid.reshape(-1)


def _build_zombie_hp(grid_obs, rows, cols) -> np.ndarray:
    """Zombie aggregate HP per cell (already normalised in env channel 4).

    Returns shape (rows * cols,).
    """
    return grid_obs[:, :, 4].astype(np.float32).reshape(-1)


def _extract_card_cooldowns(env, num_cards: int) -> np.ndarray | None:
    """Read continuous cooldown progress from game state seeds.

    Returns float32[num_cards] where:
      0.0 = fully recharged (ready to use)
      (0, 1) = cooldown in progress (fraction remaining)
      1.0 = still full cooldown / seed unavailable

    Returns None if game state cannot be accessed.
    """
    try:
        if not hasattr(env, "pvz") or env.pvz is None:
            return None
        if not hasattr(env.pvz, "is_attached") or not env.pvz.is_attached():
            return None
        game_state = env.pvz.get_game_state()
        if game_state is None or not game_state.seeds:
            return None

        cooldowns = np.ones(num_cards, dtype=np.float32)
        for i, seed in enumerate(game_state.seeds):
            if i >= num_cards:
                break
            if seed.recharge_time > 0:
                # fraction of cooldown remaining → 1.0 = just used, 0.0 = ready
                cooldowns[i] = (
                    float(seed.recharge_countdown) / float(seed.recharge_time)
                )
            elif seed.is_ready:
                cooldowns[i] = 0.0
        return cooldowns
    except Exception:
        return None


def _build_card_cooldowns(action_mask, rows, cols, num_cards) -> np.ndarray:
    """Binary fallback: 0 = ready, 1 = on cooldown (derived from action mask)."""
    mask = np.asarray(action_mask, dtype=bool)
    grid_size = rows * cols
    cooldowns = np.ones(num_cards, dtype=np.float32)
    for i in range(num_cards):
        start = i * grid_size
        end = start + grid_size
        if end <= mask.size and np.any(mask[start:end]):
            cooldowns[i] = 0.0
    return cooldowns


def build_paper_state_vector(obs, action_mask, sun, rows, cols, num_cards,
                             card_plant_ids: list[int],
                             cooldowns: np.ndarray | None = None) -> np.ndarray:
    """Build the 596-dim paper-format observation vector.

    Composition (matching paper Table 2):
        sun count           :  1 dim
        card cooldowns      : 10 dim  — continuous [0,1] when game state is
                              available (0=ready, →1=full cooldown), binary
                              fallback from action mask otherwise.
        plant type one-hot  : rows×cols×(num_cards+1) = 45×11 = 495 dim
        plant HP grid       : rows×cols = 45 dim
        zombie HP grid      : rows×cols = 45 dim
        ─────────────────────────────────────────────────────
        Total               : 596

    Returns:
        float32 numpy array of length 596.
    """
    if not isinstance(obs, dict) or "grid" not in obs:
        raise ValueError("DDQN paper-format requires dict observation with 'grid'.")

    from data import plants as plant_data

    grid = obs["grid"]
    if grid.shape[0] != rows or grid.shape[1] != cols:
        raise ValueError(
            f"Grid shape mismatch: expected ({rows},{cols},C), got {grid.shape}."
        )

    sun_norm = np.array([float(sun) / 9999.0], dtype=np.float32)

    if cooldowns is not None:
        cd = np.asarray(cooldowns, dtype=np.float32)
        if cd.shape[0] != num_cards:
            cd = _build_card_cooldowns(action_mask, rows, cols, num_cards)
    else:
        cd = _build_card_cooldowns(action_mask, rows, cols, num_cards)

    plant_onehot = _build_plant_onehot(grid, rows, cols, num_cards, card_plant_ids)
    plant_hp = _build_plant_hp(grid, rows, cols, plant_data)
    zombie_hp = _build_zombie_hp(grid, rows, cols)

    state = np.concatenate(
        [
            sun_norm,
            cd,
            plant_onehot,
            plant_hp,
            zombie_hp,
        ],
        axis=0,
    ).astype(np.float32)

    expected = 1 + num_cards + rows * cols * (num_cards + 1) + 2 * rows * cols
    if state.shape[0] != expected:
        raise RuntimeError(
            f"Paper state vector size mismatch: {state.shape[0]} != {expected}"
        )
    return state


def paper_state_dim(rows: int, cols: int, num_cards: int) -> int:
    """Return expected dimension of the paper-format observation."""
    return 1 + num_cards + rows * cols * (num_cards + 1) + 2 * rows * cols


# ─────────────────────────────────────────────────────────────────────────
# DDQNEnvAdapter
# ─────────────────────────────────────────────────────────────────────────

class DDQNEnvAdapter:
    """Wrap PVZEnv to provide DDQN-compatible observations and action masks.

    Two observation modes:
    - Legacy (default): plant_grid + zombie_grid + availability + sun
    - Paper: full one-hot encoding + HP per cell (596-dim for 5×9×10)
    """

    def __init__(self, env, env_spec=None, scenario_spec=None,
                 use_paper_observation: bool = False):
        self.env = env
        self.action_space = env.action_space
        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.plant_deck = list(getattr(env, "card_plant_ids", []))
        self._last_action_mask = None
        self.use_paper_observation = use_paper_observation

        expected_rows = env_spec.rows if env_spec is not None else self.rows
        expected_cols = env_spec.cols if env_spec is not None else self.cols
        expected_plants = (
            env_spec.plant_types if env_spec is not None else self.num_cards
        )
        if self.rows != expected_rows or self.cols != expected_cols:
            raise ValueError(
                f"DDQN expects grid {expected_rows}x{expected_cols}, "
                f"but env has {self.rows}x{self.cols}."
            )
        if self.num_cards != expected_plants:
            raise ValueError(
                f"DDQN expects {expected_plants} plant types, "
                f"but env has {self.num_cards}."
            )
        if env_spec is not None and self.action_space.n != env_spec.action_space_size:
            raise ValueError(
                f"DDQN expects {env_spec.action_space_size} actions, "
                f"but env has {self.action_space.n}."
            )
        if (scenario_spec is not None
                and tuple(self.plant_deck) != scenario_spec.cards):
            raise ValueError(
                f"DDQN expects cards {scenario_spec.cards}, "
                f"but env has {tuple(self.plant_deck)}."
            )

    @property
    def steps(self):
        return getattr(self.env, "steps", 0)

    def _build_state(self, obs, info):
        action_mask = _extract_action_mask(obs, self.env)
        self._last_action_mask = action_mask
        sun = _extract_sun(obs, info)

        if self.use_paper_observation:
            cooldowns = _extract_card_cooldowns(self.env, self.num_cards)
            return build_paper_state_vector(
                obs, action_mask, sun,
                self.rows, self.cols, self.num_cards,
                self.plant_deck,
                cooldowns=cooldowns,
            )
        return build_state_vector(
            obs, action_mask, sun, self.rows, self.cols, self.num_cards,
        )

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}
        return self._build_state(obs, info)

    def step(self, action):
        step_out = self.env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out
        state = self._build_state(obs, info)
        return state, reward, done, info

    def mask_available_actions(self):
        if self._last_action_mask is None:
            return np.ones(self.action_space.n, dtype=bool)
        return self._last_action_mask.astype(bool)

    def close(self):
        return self.env.close()
