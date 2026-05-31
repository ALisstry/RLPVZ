# AGENTS.md

## 项目定位
- 本仓库的目标是训练一个Agent挑战PVZ无尽模式。
- 目前包含 PPO 和 DDQN。

## 工作原则
- 非用户明确要求，不直接修改文件。
- 优先局部修复；只有在现有结构无法满足需求时才新增抽象。
- 代码风格优先级：最小改动、保持兼容、最佳实践。
- 训练入口统一从 `train.py` 进入。
- Python 相关命令默认使用 `.venv\Scripts\python.exe`。
- 代码风格保持与现有仓库一致，避免顺手清理无关代码。

## 仓库重点
- `train.py`：训练总入口。
- `train_args.py`：命令行参数。
- `training/`：训练生命周期、日志、checkpoint、metrics 和运行准备。
- `envs/`：环境实现。
- `models/ppo/`：PPO 环境、模型构建和训练入口。
- `models/ddqn/`：DDQN 相关实现。
- `callbacks/`：PPO/训练回调。
- `hook/` / `hook_client/`：DLL 注入与 Hook 通信。

## Git 约定
- 原始来源仓库的默认分支是 `master`。
- 主分支为 `main`。
- 提交信息遵守约定式提交，使用英语。
