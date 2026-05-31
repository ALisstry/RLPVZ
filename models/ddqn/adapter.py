import numpy as np

SUN_NORM = 200.0


class _DiscreteActionSpace:
    def __init__(self, n):
        self.n = int(n)


class DDQNSpaceSpec:
    """
    Env-like shape adapter for constructing DDQN networks without opening PVZ.
    """

    def __init__(self, env_spec, scenario_spec=None):
        self.rows = env_spec.rows
        self.cols = env_spec.cols
        self.num_cards = env_spec.plant_types
        self.action_space = _DiscreteActionSpace(env_spec.action_space_size)
        self.plant_deck = list(scenario_spec.cards) if scenario_spec is not None else []


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
        # global_features[0] = sun / 9999.0
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


def build_state_vector(obs, action_mask, sun, rows, cols, num_cards):
    if not isinstance(obs, dict) or "grid" not in obs:
        raise ValueError("DDQN requires dict observation with 'grid'.")

    grid = obs["grid"]
    if grid.shape[0] != rows or grid.shape[1] != cols:
        raise ValueError(
            f"Grid shape mismatch: expected ({rows},{cols},C), got {grid.shape}."
        )

    # Plant grid: 1 if plant exists, 0 if empty
    plant_grid = (grid[:, :, 0] > 0).astype(np.float32)
    # Zombie HP sum per tile (already normalized in env)
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


class DDQNEnvAdapter:
    """
    Wrap PVZEnv to provide DDQN-compatible observations and action masks.

    Observation vector:
    - plant grid (rows x cols)
    - zombie HP grid (rows x cols)
    - plant availability (10)
    - sun (1)
    """

    def __init__(self, env, env_spec=None, scenario_spec=None):
        self.env = env
        self.action_space = env.action_space
        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.plant_deck = list(getattr(env, "card_plant_ids", []))
        self._last_action_mask = None
        expected_rows = env_spec.rows if env_spec is not None else self.rows
        expected_cols = env_spec.cols if env_spec is not None else self.cols
        expected_plants = env_spec.plant_types if env_spec is not None else self.num_cards
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
        if scenario_spec is not None and tuple(self.plant_deck) != scenario_spec.cards:
            raise ValueError(
                f"DDQN expects cards {scenario_spec.cards}, "
                f"but env has {tuple(self.plant_deck)}."
            )

    @property
    def steps(self):
        return getattr(self.env, "steps", 0)

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}

        self._last_action_mask = _extract_action_mask(obs, self.env)
        sun = _extract_sun(obs, info)
        state = build_state_vector(
            obs, self._last_action_mask, sun, self.rows, self.cols, self.num_cards
        )
        return state

    def step(self, action):
        step_out = self.env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        self._last_action_mask = _extract_action_mask(obs, self.env)
        sun = _extract_sun(obs, info)
        state = build_state_vector(
            obs, self._last_action_mask, sun, self.rows, self.cols, self.num_cards
        )
        return state, reward, done, info

    def mask_available_actions(self):
        if self._last_action_mask is None:
            return np.ones(self.action_space.n, dtype=bool)
        return self._last_action_mask.astype(bool)

    def close(self):
        return self.env.close()
