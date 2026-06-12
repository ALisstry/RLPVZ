# models 目录说明

`models/` 存放算法相关代码。训练入口仍统一由根目录 `train.py` 进入，`training.registry` 根据 `--algo` 选择 `models/ppo` 或 `models/ddqn`。

这里的“interface”是逻辑约定，不是必须新增一个真实 Python interface 文件。每个算法包只需要对训练框架暴露同一组入口，让 `training/` 能用一致方式调度。

## 算法包约定

每个算法目录应尽量保持以下文件边界：

- `args.py`：注册该算法自己的命令行参数。
- `train_entry.py`：算法主入口，提供 `create_algorithm(args)`。
- `checkpoint.py`：该算法的加载、自动恢复和保存逻辑。
- `metrics.py`：该算法额外的 metrics writer。

`train_entry.py` 中的算法类应包含：

- `spec`：`AlgorithmSpec`，声明算法能力和支持的执行策略。
- `describe_config()`：返回用于启动时打印的关键配置。
- `train(context)`：接收 `TrainContext`，完成该算法自己的环境、模型、trainer 构建和训练。

`training/` 不应该知道 PPO 或 DDQN 的内部训练细节；它只负责构建 `TrainContext`、准备进程、checkpoint、metrics 和 run lifecycle。

## 通用数据流

```text
train.py
  -> training.registry.create_algorithm(args.algo, args)
  -> TrainRunner.run()
  -> build TrainContext
  -> algorithm.train(context)
  -> algorithm writes checkpoint targets into context.artifacts
  -> TrainRunner final save / metadata / close
```

`TrainContext` 是算法入口的数据边界，包含：

- `args`
- `device`
- `execution`
- `env_spec`
- `scenario_spec`
- `game_instances`
- `metrics`
- `checkpoint`
- `run_paths`
- `artifacts`

算法内部需要保存的运行时对象应写入 `context.artifacts`：

- PPO 写入 `artifacts.env` 和 `artifacts.model`
- DDQN 写入 `artifacts.network`，必要时写入 `artifacts.env`

## PPO 结构

PPO 目前主要依赖 Stable-Baselines3 / sb3-contrib：

- `train_entry.py`：PPO 算法入口。
- `env.py`：构建 SB3 VecEnv、ActionMasker、VecNormalize、VecFrameStack。
- `model.py`：构建或加载 MaskablePPO。
- `attention_extractor.py`：PPO 自定义特征抽取器。
- `callbacks.py`：组装 PPO 训练 callback。
- `checkpoint.py`：保存/恢复 PPO 模型和 VecNormalize。
- `metrics.py`：PPO metrics writer 扩展点。

PPO 的主流程比较短，核心数据流是：

```text
TrainContext
  -> get_env(...)
  -> get_model(...)
  -> build_callbacks(...)
  -> model.learn(...)
```

## DDQN 结构

DDQN 当前使用异步 actor-learner 架构，文件相对更多。现状分工：

- `train_entry.py`：DDQN 算法入口，创建网络和 `AsyncDDQNTrainer`。
- `adapter.py`：把 `PVZEnv` 的 dict observation 转为 DDQN 使用的一维状态，并提供 action mask。
- `ddqn.py`：QNetwork、replay buffer 和 state dict 工具。
- `learner.py`：主进程 learner，负责 loss、optimizer step、target network sync。
- `async_trainer.py`：主进程训练调度，消费 worker transition，驱动 learner 更新。
- `worker_pool.py`：启动 worker 进程，每个 worker 独占一个 PVZ 实例采样。
- `checkpoint.py`：保存/恢复 DDQN state dict。
- `metrics.py`：DDQN metrics writer 扩展点。
- `monitoring.py`：DDQN 训练统计、metrics 事件转换、控制台输出、worker 状态。
- `threshold.py`：epsilon 衰减。

DDQN 的主流程是：

```text
TrainContext
  -> QNetwork(space spec)
  -> AsyncDDQNTrainer
  -> DDQNWorkerPool starts N workers
  -> workers: env reset/step, choose action, send transition
  -> trainer: replay buffer, learner update, target sync, metrics, checkpoint
```

## DDQN 后续整理方向

DDQN 文件多的主要原因是异步训练天然包含 worker、learner、调度、统计、日志、checkpoint 多个角色。后续减少认知负担时，优先合并辅助文件，而不是先动核心训练流程。

建议后续考虑：

- 将 `threshold.py` 并入 `worker_pool.py`，因为它目前只服务 worker 的 epsilon。
- 保留 `adapter.py`，它是环境和 DDQN 状态空间之间的边界。
- 暂时保留 `async_trainer.py`、`learner.py`、`worker_pool.py` 三分结构，它们分别对应 trainer、learner、actor worker，是真实架构边界。

整理原则：先减少辅助文件数量，再考虑核心训练文件合并；不要为了减少文件数牺牲 actor-learner 数据流的可读性。
