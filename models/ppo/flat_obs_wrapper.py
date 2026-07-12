"""Observation wrapper that converts PVZEnv Dict obs to the same 596-dim
flat vector used by the DDQN baseline (paper-format).

This unifies the PPO and DDQN observation spaces so both baselines use
identical input representations:

    sun(1) + cooldowns(10) + plant_onehot(495) + plant_hp(45) + zombie_hp(45) = 596
"""

import numpy as np
from gymnasium import ObservationWrapper, spaces


class FlatPaperObsWrapper(ObservationWrapper):
    """Convert Dict observation to 596-dim paper-format flat vector.

    Must wrap a PVZEnv (or its ActionMasker-wrapped variant) that produces:
        obs["grid"]         : (rows, cols, 13)  float32
        obs["action_mask"]  : (n_actions,)       int8
        obs["global_features"] : (global_dim,)   float32 (unused for paper format)
        obs["card_attributes"] : (num_cards, 7)  float32 (unused for paper format)

    The wrapper reads the raw env for cooldowns / sun, then calls
    build_paper_state_vector to produce the flat 596-dim output.
    """

    def __init__(self, env):
        super().__init__(env)

        # Resolve the inner PVZEnv through ActionMasker wrappers
        raw = env
        while hasattr(raw, "env"):
            raw = raw.env
        self._pvz_env = raw

        rows = self._pvz_env.rows
        cols = self._pvz_env.cols
        num_cards = self._pvz_env.num_cards
        self._card_plant_ids = list(self._pvz_env.card_plant_ids)

        n_obs = 1 + num_cards + rows * cols * (num_cards + 1) + 2 * rows * cols
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_obs,), dtype=np.float32,
        )

    def observation(self, obs: dict) -> np.ndarray:
        from models.ddqn.adapter import (
            build_typed_onehot_state_vector,
            _extract_card_cooldowns,
        )

        action_mask = np.asarray(obs.get("action_mask", np.ones(self.action_space.n, dtype=bool)), dtype=bool)
        cooldowns = _extract_card_cooldowns(self._pvz_env, self._pvz_env.num_cards)
        if cooldowns is None:
            from models.ddqn.adapter import _build_card_cooldowns
            cooldowns = _build_card_cooldowns(
                action_mask, self._pvz_env.rows, self._pvz_env.cols,
                self._pvz_env.num_cards,
            )

        game_state = (
            self._pvz_env.pvz.get_game_state()
            if self._pvz_env.pvz and self._pvz_env.pvz.is_attached()
            else None
        )
        sun = game_state.sun if game_state else 50

        return build_typed_onehot_state_vector(
            obs=obs,
            action_mask=action_mask,
            sun=sun,
            rows=self._pvz_env.rows,
            cols=self._pvz_env.cols,
            num_cards=self._pvz_env.num_cards,
            card_plant_ids=self._card_plant_ids,
            cooldowns=cooldowns,
        )
