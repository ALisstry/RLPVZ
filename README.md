# RLPVZ

这是一个基于强化学习的 Plants VS Zombies 游戏 Agent 仓库。本仓库基于原作者 @AlanRuskin6 的 `TransformerPVZ` 项目（已删库）进行二次开发，保留了原项目中的 PVZ Hook、PPO 训练环境、动作掩码和内存读取等基础能力，并在此基础上扩展了 DDQN、多进程训练、日志、checkpoint 和训练曲线观测。

当前仓库仍处于训练框架开发和实验阶段。我们尚未复现或达到原作者 README 中曾声明的胜率或通关成绩，因此本 README 不再保留相关 benchmark claim。

## 当前状态

- PPO 路径仍保留，训练入口为 `train.py --algo ppo`。
- DDQN 路径已接入当前环境，训练入口为 `train.py --algo ddqn`。
- DDQN 使用异步 actor-learner 架构：每个 PVZ 进程对应一个 worker，主进程作为 learner 统一更新网络。
- 训练支持自动保存、异常保存、按 episode 周期 checkpoint 和自动恢复。
- 训练日志写入 `logs/`，模型、metrics 和训练曲线写入 `models_output/`。
- 每次训练都会创建 `models_output/{algo}/runs/{timestamp}/`，记录本次 run 的 metadata、metrics 和训练曲线。
- Hook 注入时会尝试开启 `background_running`，使游戏失焦后仍继续运行。

## 环境要求

- 操作系统：Windows 10/11
- Python：3.13
- 游戏版本：`v1.0.0.1051`
- GPU：可使用 CUDA
- 需要能够运行多个 PVZ 进程时，请确保端口、进程和 DLL 注入权限正常

本项目不提供游戏本体。请自行准备合法游戏文件，并在 `training_config.yaml` 或相关参数中配置路径。

## 安装

建议使用项目内 `.venv`：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 快速开始

默认从 `train.py` 进入训练：

```powershell
.\.venv\Scripts\python.exe train.py
```

指定 PPO：

```powershell
.\.venv\Scripts\python.exe train.py --algo ppo
```

指定 DDQN：

```powershell
.\.venv\Scripts\python.exe train.py --algo ddqn
```

DDQN 多进程示例：

```powershell
.\.venv\Scripts\python.exe train.py --algo ddqn --num_envs 6 --base_port 12345
```

如果你已经手动启动并注入游戏，可使用：

```powershell
.\.venv\Scripts\python.exe train.py --algo ddqn --no_auto_start
```

## 仿真环境（SimPVZEnv）

为了快速验证算法思路和超参数，本项目内置了一个**简化版 PVZ 仿真环境**，不依赖游戏进程、Hook DLL 和内存读取。所有游戏逻辑（植物、僵尸、飞行物、阳光、冷却、波次生成）均由纯 Python 模拟。

### 特点

- **零依赖启动**：无需安装游戏、无需注入 DLL、无需配置进程 PID
- **高速运行**：无 gym wrapper 校验开销，无渲染数据采集，每步仅做游戏帧计算
- **与真实环境一致的博弈规则**：相同的植物/僵尸/飞行物行为、相同的波次递增难度、相同的动作空间和奖励结构
- **兼容 DDQN 训练接口**：`reset()` → 状态向量，`step()` → (state, reward, done, info)，`mask_available_actions()` → 动作掩码

### 快速开始

```powershell
.\.venv\Scripts\python.exe train_sim_ddqn.py
```

训练脚本默认参数：5 万 episode、batch size 200、buffer 容量 10 万。模型和训练曲线保存在 `saved/` 目录。

### 仿真环境参数

| 参数 | 值 | 说明 |
|---|---|---|
| 网格 | 5×9 | 行×列 |
| 植物 | 向日葵/豌豆射手/坚果墙/土豆雷 | 4 种 |
| 僵尸 | 普通/路障/铁桶/旗帜 | 4 种，含护甲蜕变机制 |
| 动作空间 | 181 (4×5×9 + 1) | Discrete |
| 观测维度 | 95 | `[plant_grid(45), zombie_hp(45), plant_avail(4), sun_norm(1)]` |
| 生成器 | WaveZombieSpawner | 波次递增难度 |
| 最大帧数 | 400 | 单局上限 |

### 真机环境 vs 仿真环境

| 方面 | 真机环境 | 仿真环境 |
|---|---|---|
| 启动速度 | 需启动游戏 × N 个实例 | 瞬间 |
| 网格 | 6×9（含泳池） | 5×9 |
| 植物数 | 10 种 | 4 种 |
| 观测维度 | 119 | 95 |
| 动作数 | 451+ | 181 |
| 并行 | 多进程（每 worker 一个游戏） | 单进程 |
| 适用场景 | 最终训练、评估 | 快速实验、超参调试 |

### Python API

```python
from simenv import SimPVZEnv

env = SimPVZEnv()
state = env.reset()          # float32[95]
mask  = env.mask_available_actions()  # bool[181]
next_state, reward, done, info = env.step(action)
```

## 配置

主要配置文件：

- `training_config.yaml`：游戏模式、地图、卡组、训练默认参数等配置。

参数优先级：

```text
CLI 显式输入 > `training_config.yaml`
```

常用参数：

- `--algo ppo|ddqn`：选择训练算法。
- `--num_envs`：PVZ 进程数量。
- `--base_port`：多进程 Hook 起始端口。
- `--pids`：显式指定 PVZ 进程 PID 列表。
- `--ports`：显式指定 Hook 端口列表。
- `--no_auto_start`：禁用自动启动和注入。
- `--no_auto_resume`：禁用自动恢复模型。
- `--env_console_log_level`：控制台环境日志等级。
- `--file_log_level`：文件日志等级。
- `--execution`：训练执行策略。当前 PPO 使用 `sb3_vec_env`，DDQN 使用 `async_worker_pool`，`auto` 会选择算法默认策略。
- `--curriculum`：课程学习扩展点（待实现）。

## 输出文件

训练过程中会产生以下文件：

```text
logs/
  training_*.log

models_output/
  ddqn/
    latest_model.pt
    episode_*.pt
    runs/
      YYYYMMDD_HHMMSS/
        run_metadata.json
        metrics.jsonl
        metrics.csv
        metrics_snapshot.json
        training_curve.png
  ppo/
    latest_model.zip
    latest_model_vecnormalize.pkl
    runs/
      YYYYMMDD_HHMMSS/
        run_metadata.json
        metrics.jsonl
        metrics.csv
        metrics_snapshot.json
        training_curve.png
        heatmap.html
```

DDQN 默认每 `500` 个 episode 保存一次周期 checkpoint，可通过 `--ddqn_checkpoint_freq` 调整。

## 项目结构

```text
callbacks/             PPO 和训练回调
pvz_interface/         PVZ 高层接口
data/                  植物、僵尸、偏移量等数据
envs/                  Gymnasium 环境
game/                  游戏对象状态建模
gameobj/               游戏文件目录
hook/                  C++ Hook 动态库和对应源码
hook_client/           Python Hook 客户端和 DLL 注入逻辑
memory/                进程附加与内存读写
models/                模型实现
  models/ddqn/         DDQN、异步训练器和环境适配器
  models/ppo/          PPO 环境、模型构建和训练入口
models_output/         模型、checkpoint 和训练曲线输出
simenv/                仿真环境（纯 Python PVZ 模拟）
  simenv/pvz_sim/      仿真游戏引擎（实体、场景、网格、移动）
tools/                 独立辅助工具
utils/                 日志、绘图、坐标、伤害和训练辅助工具
train.py               统一训练入口
train_sim_ddqn.py      仿真环境 DDQN 训练脚本
training/              训练生命周期、日志、checkpoint、metrics 和运行准备
AGENTS.md              Agents 开发协作规则
```

## 训练框架

训练入口统一从 `train.py` 进入。公共训练框架位于 `training/`，负责：

- `TrainRunner`：统一训练生命周期、异常保存和收尾。
- `AlgorithmSpec`：声明算法类型、执行策略、动作掩码和课程学习能力。
- `EnvSpec` / `ScenarioSpec`：集中表达模型输入输出规格和训练场景。
- `CheckpointManager`：统一恢复、缓存模型、周期 checkpoint 和异常保存入口。
- `MetricsPipeline`：统一写入 `jsonl`、`csv`、snapshot 和训练曲线。
- `utils/train_utils.py`：集中放置训练配置读取、Torch 运行时设置、GPU 显存打印和 run metadata 写入。

PPO 和 DDQN 保留各自算法本质差异，不强行共用同一个训练循环。

重构静态验收至少应检查 Python 编译、`train.py --help` 和算法 registry。静态检查不能替代真实 PVZ 训练回归。

## DDQN 训练架构

DDQN 当前使用异步 actor-learner 结构：

- worker 独占一个 PVZ 进程，负责 reset、step、动作执行和采样。
- learner 位于主进程，维护 online network、target network、loss 和 update。
- async trainer 位于主进程，负责 replay buffer、worker queue、update/sync 调度、checkpoint 触发。
- worker 将 transition 发送给 learner。
- learner 定期更新网络，并向 worker 广播最新权重。
- episode 统计、worker 状态、metrics 输出和控制台报告统一收敛在 `models/ddqn/monitoring.py`。

这种结构用于避免单个游戏进程 reset 或 UI 准备耗时阻塞其他进程。

## 已知限制

- 尚无稳定胜率或通关成绩。
- Hook 和内存偏移依赖 PVZ `v1.0.0.1051`，请严格确保版本号正确。
- 如果某个 worker 失效，当前逻辑会将其移除并继续训练；新启动的 PVZ 进程不会被自动接管。
- 游戏窗口、Hook 通信和 Python worker 会显著占用 CPU；可能会限制 CUDA 利用率。
- 当前推荐以 `.venv` 和 `requirements.txt` 管理 Python 依赖。

## 常见问题

**DLL 注入失败**

请确认终端权限、游戏版本、进程名、端口占用和 Hook DLL 路径。

**训练中断或 worker 失效**

查看 `logs/training_*.log`。DDQN 会定期保存 `models_output/ddqn/latest_model.pt` 和 `episode_*.pt`，可在下次启动时自动恢复。

**CUDA OOM**

降低 DDQN batch size、减少并行 worker 数，或确认显卡显存是否被其他程序占用。

**看不到训练曲线**

训练曲线默认按 `--ddqn_plot_freq` 或 `--ppo_plot_freq` 周期刷新，输出在 `models_output/{algo}/runs/{timestamp}/`。

## 致谢

本仓库基于原作者的 TransformerPVZ 项目继续开发。原项目提供了重要的 Hook、环境、PPO 和游戏交互基础。

同时参考或使用了以下项目和工具：

- `re-plants-vs-zombies`
- `pvzclass`
- `AsmVsZombies`
- `pvztools / pvztoolkit`

## 免责声明

本项目仅用于学习、研究和技术交流。项目涉及游戏内存读取、DLL 注入和自动化训练，请自行承担运行风险。请勿用于商业用途或破坏游戏公平性的场景。Plants vs. Zombies 版权归 PopCap Games / Electronic Arts 所有。
