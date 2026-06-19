"""
Random baseline agent — uniformly samples a legal action at each step.
"""

import numpy as np


class RandomAgent:
    """Selects a uniformly random legal action at every decision step."""

    @property
    def name(self) -> str:
        return "Random"

    def select_action(self, env) -> int:
        """
        Args:
            env: SimPVZEnv instance (must have mask_available_actions()).

        Returns:
            int action index.
        """
        mask = np.asarray(env.mask_available_actions(), dtype=bool)
        valid_actions = np.arange(env.action_space.n)[mask]
        return int(np.random.choice(valid_actions))
