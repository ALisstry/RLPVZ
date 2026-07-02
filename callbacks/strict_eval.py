from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from models.ppo.env import get_env
from models.ppo.evaluate import evaluate_ppo_model
from training.evaluation import BestEvaluationCheckpoint, EvaluationScheduler
from utils.train_utils import get_current_stage_name


class StrictEvalCallback(BaseCallback):
    def __init__(self, context, verbose=0):
        super().__init__(verbose)
        self.context = context
        self.scheduler = EvaluationScheduler(context.eval_config)
        self.episode_count = 0
        self.eval_env = None
        self.best_eval_checkpoint = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info.get("episode"):
                continue
            self.episode_count += 1
            if self.scheduler.should_run(self.episode_count):
                self._run_eval()
        return True

    def _run_eval(self):
        if not self.context.eval_game_instances:
            return
        if self.eval_env is None:
            self.eval_env = get_env(
                self.context.args,
                self.context.eval_game_instances,
                env_spec=self.context.env_spec,
                scenario_spec=self.context.scenario_spec,
                load_path=self.context.checkpoint.resolve_load_path(),
            )
        if hasattr(self.eval_env, "env_method"):
            self.eval_env.env_method("set_pending_scenario", self.context.scenario_spec)
        try:
            sync_envs_normalization(self.training_env, self.eval_env)
        except Exception:
            pass
        result = evaluate_ppo_model(
            model=self.model,
            env=self.eval_env,
            scenario_spec=self.context.scenario_spec,
            episodes=self.context.eval_config.episodes,
            episode=self.episode_count,
            step=self.num_timesteps,
        )
        result = _with_stage_name(result, get_current_stage_name(self.context.curriculum))
        self.context.evaluation_writer.write(result)
        if self.best_eval_checkpoint is None:
            self.best_eval_checkpoint = BestEvaluationCheckpoint(
                self.context.evaluation_writer.output_dir,
                model_filename="best_model.zip",
            )
        saved_path = self.best_eval_checkpoint.maybe_save(
            result,
            lambda path: self.model.save(path),
        )
        if saved_path is not None:
            print(f"[Eval] New best model saved to {saved_path}", flush=True)
        print(
            f"\n[Eval] Episode {result.episode} | "
            f"stage={result.stage_name} | "
            f"reward={result.reward_mean:.2f} | "
            f"survival={result.survival_mean:.2f} | "
            f"win_rate={result.win_rate:.2%}",
            flush=True,
        )

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None


def _with_stage_name(result, stage_name):
    from dataclasses import replace

    return replace(result, stage_name=stage_name)
