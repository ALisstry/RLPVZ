"""Train MaskablePPO (stable-baselines3 + sb3-contrib) on SimPVZEnv."""

import csv
import os
import time
from collections import deque
from datetime import datetime
from itertools import islice

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback

from simenv import SimPVZEnv
from simenv.pvz_sim import config
from simenv.model import transform_observation


# ═══════════════════════════════════════════════════════════════════════════
# Gymnasium wrapper
# ═══════════════════════════════════════════════════════════════════════════

class SimPVZGymEnv(gym.Env):
    """Wrap SimPVZEnv with action masking for sb3 MaskablePPO."""

    def __init__(self, stage=None):
        super().__init__()
        self._env = SimPVZEnv(stage=stage)
        self.action_space = spaces.Discrete(self._env.action_space.n)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(int(self._env.state_dim),),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = transform_observation(self._env.reset())
        mask = self._env.mask_available_actions()
        return obs.astype(np.float32), {"action_mask": mask.astype(bool)}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        obs = transform_observation(obs).astype(np.float32)
        mask = info.get("mask", self._env.mask_available_actions())
        truncated = self._env._scene._chrono >= config.MAX_FRAMES
        return obs, float(reward), bool(done), truncated, \
            {"action_mask": mask.astype(bool)}

    def close(self):
        self._env.close()


# ═══════════════════════════════════════════════════════════════════════════
# Training callback
# ═══════════════════════════════════════════════════════════════════════════

class TrainCallback(BaseCallback):
    """Periodic eval, dashboard plot, model save, console output."""

    def __init__(self, save_path, plot_callback=None,
                 plot_freq=1000, save_freq=10000,
                 eval_freq=2500, n_eval_episodes=100):
        super().__init__(verbose=0)
        self._save_path = save_path
        self._plot_callback = plot_callback
        self._plot_freq = plot_freq
        self._save_freq = save_freq
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._t_start = time.perf_counter()
        self._last_plot_ep = 0
        self._last_save_ep = 0
        self._last_eval_ep = 0
        self._ep = 0
        self._reward_history = []
        self._length_history = []

    def _init_callback(self) -> None:
        # Expand sb3's ep_info_buffer so we don't lose history
        buf = self.model.ep_info_buffer
        if buf is None:
            self.model.ep_info_buffer = deque(maxlen=100000)
        else:
            self.model.ep_info_buffer = deque(buf, maxlen=100000)

    @staticmethod
    def _est_ep(num_timesteps: int) -> int:
        """Rough episode estimate from total timesteps."""
        return num_timesteps // 50  # ~50 steps per episode on average

    def _on_step(self) -> bool:
        ep = self._est_ep(self.num_timesteps)
        if ep != self._ep:
            self._ep = ep
            self._on_episode(ep)
        return True

    def _on_episode(self, ep):
        # Sync episode history from sb3 buffer (only copy new entries)
        n_model_eps = len(self.model.ep_info_buffer)
        n_my_eps = len(self._reward_history)
        if n_model_eps > n_my_eps:
            for info in islice(self.model.ep_info_buffer, n_my_eps, None):
                if 'r' in info:
                    self._reward_history.append(info['r'])
                if 'l' in info:
                    self._length_history.append(info['l'])

        # Console log (every 100 ep to avoid spam)
        if ep % 100 == 0:
            elapsed = time.perf_counter() - self._t_start
            steps = self.num_timesteps
            print(f"[PPO] ep~{ep:6d}  steps={steps:8d}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

        # Periodic eval
        if (self._n_eval_episodes > 0
                and ep - self._last_eval_ep >= self._eval_freq):
            self._last_eval_ep = ep
            self._run_eval(ep)

        # Dashboard plot
        if (self._plot_callback is not None
                and ep - self._last_plot_ep >= self._plot_freq):
            self._last_plot_ep = ep
            self._do_plot(ep)

        # Periodic save
        if ep - self._last_save_ep >= self._save_freq:
            self._last_save_ep = ep
            self.model.save(self._save_path)
            print(f"[PPO] Saved model (ep~{ep})", flush=True)

    def _do_plot(self, ep):
        """Generate dashboard with accumulated episode history."""
        rewards = np.array(self._reward_history) if self._reward_history else np.zeros(1)
        iterations = np.array(self._length_history) if self._length_history else np.zeros(1)
        self._plot_callback(
            self._save_path.replace(".zip", ".pt"),
            rewards, iterations, np.array([]),
        )

    def _run_eval(self, ep):
        """Run eval episodes and write to eval.csv."""
        env = SimPVZEnv()
        stage = getattr(
            self.training_env.envs[0].unwrapped._env,
            '_stage', None)
        if stage is not None:
            env.apply_stage(stage)
        rewards = []
        survivals = []
        max_frames = config.MAX_FRAMES

        for _ in range(self._n_eval_episodes):
            state = transform_observation(env.reset())
            done = False
            total_reward = 0.0
            while not done:
                mask = env.mask_available_actions()
                obs_t = np.expand_dims(state, 0)
                mask_t = np.expand_dims(mask, 0)
                action, _ = self.model.predict(obs_t, action_masks=mask_t,
                                               deterministic=True)
                state, reward, done, _ = env.step(int(action[0]))
                state = transform_observation(state)
                total_reward += float(reward)

            rewards.append(total_reward)
            survivals.append(min(max_frames, env._scene._chrono))

        rewards = np.array(rewards)
        survivals = np.array(survivals)
        print(f"\n[PPO Eval] ep={ep}  "
              f"reward: {rewards.mean():.1f}±{rewards.std():.1f}  "
              f"survival: {survivals.mean():.0f}±{survivals.std():.0f}  "
              f"max={rewards.max():.0f}  min={rewards.min():.0f}\n",
              flush=True)

        # Write eval.csv
        output_dir = os.path.dirname(self._save_path) or "."
        eval_path = os.path.join(output_dir, "eval.csv")
        file_exists = os.path.isfile(eval_path)
        with open(eval_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["episode", "reward_mean", "survival_mean"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "episode": ep,
                "reward_mean": float(rewards.mean()),
                "survival_mean": float(survivals.mean()),
            })
        env.close()


# ═══════════════════════════════════════════════════════════════════════════
# Training entry point
# ═══════════════════════════════════════════════════════════════════════════

def train_ppo(
    max_episodes=100000,
    network_type="deepmlp",
    lr=3e-4,
    n_epochs=10,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    save_path=None,
    eval_episodes=100,
    eval_freq_episodes=2500,
    plot_callback=None,
    plot_freq=1000,
    n_steps=2048,
):
    if save_path is None:
        tag = f"ppo_{network_type}"
        save_path = _default_save_path("ppo", "sim_ppo.zip", tag=tag)

    total_timesteps = max_episodes * 50  # ~50 env steps per episode

    # Policy architecture
    if network_type == "mlp":
        net_arch = dict(pi=[128, 64], vf=[128, 64])
        policy, pk = "MlpPolicy", dict(net_arch=net_arch)
    elif network_type == "deepmlp":
        net_arch = dict(pi=[2048, 2048], vf=[2048, 2048])
        policy, pk = "MlpPolicy", dict(net_arch=net_arch)
    else:  # cnn
        policy, pk = "CnnPolicy", {}

    # Create env
    env = SimPVZGymEnv()

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    def _mask_fn(env_instance):
        return env_instance.unwrapped._env.mask_available_actions()
    env = ActionMasker(env, _mask_fn)

    model = MaskablePPO(
        policy, env,
        learning_rate=lr, n_steps=n_steps, batch_size=batch_size,
        n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
        clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, policy_kwargs=pk,
        verbose=0, tensorboard_log=None,
    )

    callback = TrainCallback(
        save_path=save_path,
        plot_callback=plot_callback, plot_freq=plot_freq,
        save_freq=10000, eval_freq=eval_freq_episodes,
        n_eval_episodes=eval_episodes,
    )

    _print_sb3_config(locals())
    model.learn(total_timesteps=total_timesteps, callback=callback,
                progress_bar=False)

    # Final eval
    final_ep = callback._est_ep(model.num_timesteps)
    if eval_episodes > 0:
        callback._run_eval(final_ep)

    # Final save
    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    model.save(save_path)
    print(f"[PPO] Final model saved to {save_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _default_save_path(algo, filename, tag=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = timestamp if tag is None else f"{timestamp}_{tag}"
    return os.path.join("saved", algo, folder, filename)


def _print_sb3_config(loc):
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  PPO (sb3-contrib MaskablePPO)")
    print(f"{sep}")
    print(f"  Network:       {loc['network_type']}")
    print(f"  Episodes:      {loc['max_episodes']} (~{loc['total_timesteps']} steps)")
    print(f"  n_steps:       {loc['n_steps']}")
    print(f"  Batch:         {loc['batch_size']}  x  {loc['n_epochs']} epochs")
    print(f"  Gamma:         {loc['gamma']}  λ={loc['gae_lambda']}")
    print(f"  Clip:          {loc['clip_range']}  ent={loc['ent_coef']}  vf={loc['vf_coef']}")
    print(f"  LR:            {loc['lr']}  grad_norm={loc['max_grad_norm']}")
    print(f"  Eval:          every {loc['eval_freq_episodes']} ep x {loc['eval_episodes']}")
    print(f"  Grid:          {config.N_LANES}x{config.LANE_LENGTH}")
    print(f"{sep}\n")
