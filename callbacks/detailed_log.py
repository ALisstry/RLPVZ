import os
import time
import threading
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class DetailedLogCallback(BaseCallback):
    """
    详细日志记录器 - 记录动作、注意力和游戏状态的关系到 log 文件
    """

    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_log_step = 0
        self.episode_num = 0
        self.step_in_episode = 0
        self.reward_breakdown = {}

    def _on_step(self) -> bool:
        from utils.logger import log_attention_debug, log_game_state_debug

        self.step_in_episode += 1

        # 每 log_freq 步记录一次详细信息
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_attention_coherence(self.locals.get("actions")[0])
            self._log_game_state()
            self.last_log_step = self.num_timesteps

        # 检测 episode 结束
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_num += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                is_win = ep_reward > 0

                # 记录 episode 摘要
                from utils.logger import log_episode_summary

                log_episode_summary(
                    episode=self.episode_num,
                    steps=int(ep_length),
                    reward=ep_reward,
                    reward_breakdown=self.reward_breakdown,
                    win=is_win,
                )

                # 重置
                self.step_in_episode = 0
                self.reward_breakdown = {}

                # 如果失败，记录失败分析
                if not is_win:
                    self._log_failure_analysis(info)

        return True

    def _log_attention_coherence(self, action):
        from utils.logger import log_attention_debug

        obs = self.locals.get("new_obs")
        grid_shape = self._infer_grid_shape(obs)
        if grid_shape is None:
            return
        rows, cols = grid_shape

        # 获取注意力权重
        attn_weights = None
        if hasattr(self.model.policy.features_extractor, "last_attn_weights"):
            weights_tensor = self.model.policy.features_extractor.last_attn_weights
            if weights_tensor is not None:
                try:
                    attn_weights = (
                        weights_tensor[0].detach().cpu().numpy().reshape(rows, cols)
                    )
                except ValueError:
                    pass

        if attn_weights is None:
            return

        # 计算统计量
        max_attn_pos = np.unravel_index(attn_weights.argmax(), attn_weights.shape)

        # 获取最大威胁位置 (从 grid 的 threat channel)
        threat_pos = None
        if obs and "grid" in obs:
            grid = obs["grid"][0]  # (rows, cols, channels)
            if grid.shape[2] > 9:
                threat_map = grid[:, :, 9]
                threat_pos = np.unravel_index(threat_map.argmax(), threat_map.shape)

        # 记录到文件
        log_attention_debug(
            episode=self.episode_num,
            step=self.step_in_episode,
            attn_weights=attn_weights,
            action=action,
            max_pos=max_attn_pos,
            threat_pos=threat_pos,
        )

    def _log_game_state(self):
        from utils.logger import log_game_state_debug

        # 从 info 获取游戏状态
        infos = self.locals.get("infos", [])
        if not infos or len(infos) == 0:
            return

        info = infos[0]
        log_game_state_debug(
            step=self.step_in_episode,
            sun=info.get("sun", 0),
            wave=info.get("wave", 0),
            zombies=info.get("zombie_count", 0),
            plants=info.get("plant_count", 0),
            lawnmowers=info.get("lawnmowers", []),
            is_paused=info.get("is_paused", False),
        )

    def _log_failure_analysis(self, info):
        from utils.logger import log_failure_analysis

        # 获取注意力分布
        attn_distribution = None
        if hasattr(self.model.policy.features_extractor, "last_attn_weights"):
            weights_tensor = self.model.policy.features_extractor.last_attn_weights
            if weights_tensor is not None:
                obs = self.locals.get("new_obs")
                grid_shape = self._infer_grid_shape(obs)
                try:
                    if grid_shape is None:
                        return
                    rows, cols = grid_shape
                    attn_weights = (
                        weights_tensor[0].detach().cpu().numpy().reshape(rows, cols)
                    )
                    # 按行求和得到每行的注意力分配
                    attn_distribution = attn_weights.sum(axis=1).tolist()
                except ValueError:
                    pass

        log_failure_analysis(
            episode=self.episode_num,
            step=self.step_in_episode,
            reason="Zombie reached house" if info["episode"]["r"] < 0 else "Unknown",
            attn_distribution=attn_distribution,
        )

    def _infer_grid_shape(self, obs):
        if not obs or "grid" not in obs:
            return None
        grid = obs["grid"]
        if len(grid.shape) == 4:
            return int(grid.shape[1]), int(grid.shape[2])
        if len(grid.shape) == 3:
            return int(grid.shape[0]), int(grid.shape[1])
        return None


# =============================================================================
# 自动收集回调 (使用代码 Patch 方式)
# =============================================================================
