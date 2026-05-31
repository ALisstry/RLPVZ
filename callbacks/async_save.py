import os
import time
import threading
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

# =============================================================================
# 异步模型保存回调 (不阻塞主线程)
# =============================================================================
class AsyncSingleModelCallback(BaseCallback):
    """每隔一定步数异步保存模型，只保留一个文件 (覆盖)，避免阻塞训练"""

    def __init__(
        self,
        save_freq: int,
        save_path: str | None = None,
        checkpoint=None,
        verbose=0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.checkpoint = checkpoint
        self.save_thread = None

    def _save_model_thread(self, model, save_path, env):
        """后台保存线程"""
        try:
            if self.checkpoint is not None:
                self.checkpoint.save(model=model, env=env)
                return

            if save_path is None:
                raise ValueError("save_path is required when checkpoint is not provided")
            model.save(save_path)

            # 保存 VecNormalize 统计信息
            path_no_ext = os.path.splitext(save_path)[0]
            vec_norm_path = path_no_ext + "_vecnormalize.pkl"
            if hasattr(self.training_env, "save"):
                try:
                    self.training_env.save(vec_norm_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"\n[警告] 异步保存模型失败: {e}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 如果上一次保存还没结束，跳过本次保存
            if self.save_thread and self.save_thread.is_alive():
                if self.verbose > 0:
                    print(
                        f"\r⚠️ 上次保存尚未完成，跳过本次保存 (Step {self.num_timesteps})",
                        end="",
                        flush=True,
                    )
                return True

            # 启动新线程保存
            # 注意：SB3 的 save 方法通常是线程安全的，但为了绝对安全，
            # 最好是保存参数的副本，不过对于大模型这会消耗双倍内存。
            # 考虑到 PPO 训练时参数变化，直接 save 可能会有极小概率导致保存的模型参数不一致，
            # 但通常是可以接受的，且比阻塞好得多。
            self.save_thread = threading.Thread(
                target=self._save_model_thread,
                args=(self.model, self.save_path, self.training_env),
            )
            self.save_thread.start()

            if self.verbose > 0:
                print(
                    f"\r💾 正在后台保存模型: {self.save_path} (Step {self.num_timesteps})",
                    end="",
                    flush=True,
                )
        return True


# =============================================================================
# 热力图可视化回调
# =============================================================================
