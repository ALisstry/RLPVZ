import os
import time
import threading
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class HeatmapCallback(BaseCallback):
    """
    生成实时热力图 HTML
    """

    def __init__(self, save_path="heatmap.html", refresh_rate=10, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.refresh_rate = refresh_rate

    def _on_step(self) -> bool:
        if self.n_calls % self.refresh_rate == 0:
            try:
                # 获取最新的观测 (VecEnv -> Dict -> grid)
                obs = self.locals.get("new_obs")
                if obs and "grid" in obs:
                    obs_grid = obs["grid"][0]
                    grid_rows, grid_cols = obs_grid.shape[:2]
                    # 触发一次预测以更新注意力权重 (predict 会自动切换到 eval 模式)
                    # 这对于我们在 attention_extractor.py 中添加的钩子是必须的
                    self.model.predict(obs, deterministic=True)

                    # 获取注意力权重
                    attn_weights = None
                    if hasattr(
                        self.model.policy.features_extractor, "last_attn_weights"
                    ):
                        # last_attn_weights 是 (B, rows * cols)
                        weights_tensor = (
                            self.model.policy.features_extractor.last_attn_weights
                        )
                        if weights_tensor is not None:
                            # 动态计算形状
                            flat_size = weights_tensor.shape[1]
                            try:
                                attn_weights = (
                                    weights_tensor[0]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .reshape(grid_rows, grid_cols)
                                )
                            except ValueError:
                                pass

                            # Attention weights extracted (silent mode)

                    # 取第一个环境的观测
                    grid = obs_grid
                    self.generate_html(grid, attn_weights)
            except Exception as e:
                pass  # 忽略错误，不影响训练
        return True

    def generate_html(self, grid, attn_map=None):
        # grid shape: (rows, cols, channels)
        rows, cols, channels = grid.shape

        # 通道 8: DPS (Blue)
        # 通道 9: Threat (Red)

        # 检查通道数，兼容旧配置
        if channels > 9:
            dps_map = grid[:, :, 8]
            threat_map = grid[:, :, 9]
        else:
            # 尝试从通道 2 和 4 提取 (旧版混合通道)
            # 但这很难分离，这里假设是新版配置
            dps_map = np.zeros((rows, cols))
            threat_map = np.zeros((rows, cols))

        # 准备 Attention HTML
        attn_html = ""
        if attn_map is not None:
            # 归一化以便显示 (0-1)
            # 注意力权重通常和为1，但单个值可能很小，或者如果用了 sigmoid 可能会不同
            # 这里我们做一个简单的最大值归一化来增强对比度
            max_val = attn_map.max()
            if max_val > 0:
                display_map = attn_map / max_val
            else:
                display_map = attn_map

            attn_html = f"""
            <div class="container">
                <h2 style="color: #ffff00;">Attention (AI关注点)</h2>
                <table>
                    {self._generate_table_rows(display_map, 'yellow')}
                </table>
            </div>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="refresh" content="1"> <!-- 每1秒自动刷新 -->
            <title>PVZ AI Heatmap</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #222; color: #fff; display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }}
                .container {{ text-align: center; margin: 10px; }}
                table {{ border-collapse: collapse; margin: 10px auto; }}
                td {{ width: 50px; height: 50px; border: 1px solid #444; text-align: center; font-size: 10px; color: rgba(255,255,255,0.8); }}
                h2 {{ margin-bottom: 5px; font-size: 18px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2 style="color: #4da6ff;">DPS Heatmap (火力覆盖)</h2>
                <table>
                    {self._generate_table_rows(dps_map, 'blue')}
                </table>
            </div>
            <div class="container">
                <h2 style="color: #ff4d4d;">Threat Heatmap (僵尸威胁)</h2>
                <table>
                    {self._generate_table_rows(threat_map, 'red')}
                </table>
            </div>
            {attn_html}
        </body>
        </html>
        """

        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _generate_table_rows(self, matrix, color_theme):
        rows_html = ""
        for r in range(matrix.shape[0]):
            rows_html += "<tr>"
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                # 颜色计算
                if color_theme == "blue":
                    # 蓝色: rgba(0, 100, 255, alpha)
                    bg_color = f"rgba(0, 120, 255, {val:.2f})"
                elif color_theme == "red":
                    # 红色: rgba(255, 50, 50, alpha)
                    bg_color = f"rgba(255, 50, 50, {val:.2f})"
                elif color_theme == "yellow":
                    # 黄色: rgba(255, 255, 0, alpha)
                    bg_color = f"rgba(255, 255, 0, {val:.2f})"
                else:
                    bg_color = "rgba(0,0,0,0)"

                fmt = ".3f" if color_theme == "yellow" else ".2f"
                rows_html += (
                    f'<td style="background-color: {bg_color}">{val:{fmt}}</td>'
                )
            rows_html += "</tr>"
        return rows_html
