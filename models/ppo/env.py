import os
import random
import time
from typing import Tuple

import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecFrameStack,
    VecNormalize,
)

from envs import PVZEnv


def mask_fn(env):
    """获取动作掩码的回调函数"""
    return env.unwrapped._get_action_mask(
        env.unwrapped.pvz.get_game_state() if env.unwrapped.pvz else None
    )


class DiversifiedPVZEnv(PVZEnv):
    """
    多样化训练环境

    策略:
    - 随机跳过开局若干帧，让 AI 学会应对各种中途状态
    - 随机初始阳光 (模拟不同经济状况)
    - 随机选择是否预先放置一些植物
    """

    def __init__(
        self,
        diversify_prob: float = 0.3,  # 30% 概率使用多样化
        skip_frames_range: Tuple[int, int] = (0, 500),  # 跳过帧数范围
        random_sun_range: Tuple[int, int] = (50, 500),  # 随机阳光范围
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diversify_prob = diversify_prob
        self.skip_frames_range = skip_frames_range
        self.random_sun_range = random_sun_range
        self._diversify_enabled = True

    def set_diversify(self, enabled: bool):
        """动态开关多样化训练"""
        self._diversify_enabled = enabled

    def reset(self, seed=None, options=None):
        """重置环境，有概率进行多样化"""
        obs, info = super().reset(seed=seed, options=options)

        # 决定是否多样化
        if self._diversify_enabled and random.random() < self.diversify_prob:
            self._apply_diversification()
            # 重新获取观测
            obs = self._get_observation()

        # 标记需要重置记忆
        info["needs_memory_reset"] = True

        return obs, info

    def _apply_diversification(self):
        """应用多样化策略"""
        if not self.hook_client or not self.hook_client.connected:
            return

        # 策略1: 随机跳过若干帧 (模拟中途状态)
        skip_frames = random.randint(*self.skip_frames_range)
        if skip_frames > 0:
            print(f"  🎲 多样化: 跳过 {skip_frames} 帧，模拟中途状态")
            # 执行若干次等待，让游戏自然推进
            wait_iterations = min(
                skip_frames // max(1, self.frame_skip), 50
            )  # 最多50次
            for i in range(wait_iterations):
                # 自动收集阳光
                if i % 5 == 0:  # 每5次收集一次
                    self.hook_client.collect()
                time.sleep(0.002)  # 2ms延迟，配合高速游戏

            # 再次收集阳光确保资源
            self.hook_client.collect()

            # 更新缓存的游戏状态
            if self.pvz:
                self._cached_game_state = self.pvz.get_game_state()


def _validate_env_spec(env, env_spec, scenario_spec):
    if env_spec is None:
        return
    actual_cards = tuple(getattr(env, "card_plant_ids", ()))
    if env.rows != env_spec.rows or env.cols != env_spec.cols:
        raise ValueError(
            f"EnvSpec grid mismatch: expected {env_spec.rows}x{env_spec.cols}, "
            f"got {env.rows}x{env.cols}."
        )
    if env.num_cards != env_spec.plant_types:
        raise ValueError(
            f"EnvSpec plant type mismatch: expected {env_spec.plant_types}, "
            f"got {env.num_cards}."
        )
    if env.action_space.n != env_spec.action_space_size:
        raise ValueError(
            f"EnvSpec action size mismatch: expected {env_spec.action_space_size}, "
            f"got {env.action_space.n}."
        )
    grid_shape = env.observation_space["grid"].shape
    expected_grid_shape = (env_spec.rows, env_spec.cols, env_spec.grid_channels)
    if grid_shape != expected_grid_shape:
        raise ValueError(
            f"EnvSpec grid observation mismatch: expected {expected_grid_shape}, "
            f"got {grid_shape}."
        )
    global_shape = env.observation_space["global_features"].shape
    if global_shape != (env_spec.global_feature_dim,):
        raise ValueError(
            "EnvSpec global feature mismatch: "
            f"expected {(env_spec.global_feature_dim,)}, got {global_shape}."
        )
    card_shape = env.observation_space["card_attributes"].shape
    if card_shape != env_spec.card_attribute_shape:
        raise ValueError(
            "EnvSpec card attribute mismatch: "
            f"expected {env_spec.card_attribute_shape}, got {card_shape}."
        )
    if scenario_spec is not None and actual_cards != scenario_spec.cards:
        raise ValueError(
            f"ScenarioSpec cards mismatch: expected {scenario_spec.cards}, "
            f"got {actual_cards}."
        )


def make_single_env(args, instance, env_spec=None, scenario_spec=None):
    if args.no_diversify:
        env = PVZEnv(
            config_path=args.training_config,
            hook_port=instance["port"],
            target_pid=instance["pid"],
            game_speed=args.speed,
            frame_skip=args.frameskip,
            verbose=args.env_console_log_level,
            log_verbose=args.file_log_level,
        )
    else:
        env = DiversifiedPVZEnv(
            config_path=args.training_config,
            hook_port=instance["port"],
            target_pid=instance["pid"],
            game_speed=args.speed,
            frame_skip=args.frameskip,
            diversify_prob=args.diversify,
            verbose=args.env_console_log_level,
            log_verbose=args.file_log_level,
        )
    _validate_env_spec(env, env_spec, scenario_spec)
    env = ActionMasker(env, mask_fn)
    return env


def _make_env_factory(args, instance, env_spec=None, scenario_spec=None):
    def _factory():
        return make_single_env(args, instance, env_spec, scenario_spec)

    return _factory


def get_env(args, instances, env_spec=None, scenario_spec=None, load_path=None):
    factories = [
        _make_env_factory(args, instance, env_spec, scenario_spec)
        for instance in instances
    ]
    if len(factories) == 1:
        env = DummyVecEnv(factories)
    else:
        env = SubprocVecEnv(factories, start_method="spawn")

    # 1. 监控 (记录原始奖励)
    env = VecMonitor(env)

    # 2. 归一化 (关键优化: 稳定 PPO 训练)
    # 归一化观测值和奖励，防止梯度爆炸/消失
    if load_path:
        path_no_ext = os.path.splitext(load_path)[0]
        vec_norm_path = path_no_ext + "_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            print(f"加载归一化统计: {vec_norm_path}")
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
            env.norm_reward = True
            env.norm_obs = True
        else:
            print(f"未找到归一化统计文件，创建新的归一化层")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 3. 帧堆叠 (赋予时间感知)
    # 堆叠最近4帧，让AI能感知僵尸的移动速度和波次节奏
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    return env
