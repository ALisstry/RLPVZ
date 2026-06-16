from stable_baselines3.common.callbacks import BaseCallback


class PPOCurriculumCallback(BaseCallback):
    def __init__(self, context, verbose=0):
        super().__init__(verbose)
        self.context = context
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue

            self.episode_count += 1
            changed, scenario = self.context.update_curriculum(
                {
                    "episode_reward": float(episode["r"]),
                    "episode_success": bool(info.get("win") is True),
                    "episode_count": self.episode_count,
                    "step": self.num_timesteps,
                }
            )
            if changed:
                self.training_env.env_method("set_pending_scenario", scenario)
        return True
