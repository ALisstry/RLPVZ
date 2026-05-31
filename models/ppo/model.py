import os
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from models.attention_extractor import PVZAttentionExtractor


def linear_schedule(initial_value: float):
    """
    线性学习率衰减函数
    :param initial_value: 初始学习率
    :return: schedule function
    """

    def func(progress_remaining: float):
        """
        Progress remaining decreases from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value

    return func


def cosine_schedule(initial_value: float, warmup_steps: int = 0, total_steps: int = 1):
    """
    余弦退火学习率调度 (含热身)
    :param initial_value: 初始学习率
    :param warmup_steps: 热身步数 (占总步数的比例，0-1)
    :param total_steps: 总步数 (用于计算进度)
    :return: schedule function
    """

    def func(progress_remaining: float):
        """
        Progress remaining decreases from 1 (beginning) to 0.
        Current progress = 1 - progress_remaining
        """
        current_progress = 1.0 - progress_remaining

        # 热身阶段
        if current_progress < warmup_steps:
            return initial_value * (current_progress / warmup_steps)

        # 余弦退火阶段
        decay_progress = (current_progress - warmup_steps) / (1.0 - warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_progress))
        return initial_value * cosine_decay

    return func


def get_model(args, env, device, load_path=None):
    # 网络配置 - 激进版本
    net_configs = {
        "small": dict(pi=[256, 256], vf=[256, 256]),
        "medium": dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        "large": dict(pi=[1024, 512, 256], vf=[1024, 512, 256]),
        "xlarge": dict(pi=[2048, 1024, 512, 256], vf=[2048, 1024, 512, 256]),
        "huge": dict(
            pi=[4096, 2048, 1024, 512], vf=[4096, 2048, 1024, 512]
        ),  # 巨型网络
    }
    net_arch = net_configs[args.net]

    policy_kwargs = dict(net_arch=net_arch)
    if not args.no_attn:
        policy_kwargs.update(
            features_extractor_class=PVZAttentionExtractor,
            features_extractor_kwargs=dict(
                hidden_size=128,  # 精简维度：GPU 不是瓶颈，优先减少计算量
                attn_heads=4,  # 精简注意力头数
                ff_dim=256,  # 精简前馈维度
                dropout=0.0,
                num_layers=2,  # 精简层数：2层足够，推理快 2 倍
            ),
        )

    load_path = load_path if load_path is not None else args.load
    if load_path:
        print(f"加载模型: {load_path}")
        try:
            model = MaskablePPO.load(load_path, env=env, device=device)
            # 更新超参数 - 使用线性衰减学习率
            model.learning_rate = linear_schedule(args.lr)

            # 关键修复: 如果加载的模型 n_steps 与当前参数不一致，需要调整 buffer 大小
            if model.n_steps != args.n_steps:
                print(
                    f"模型 n_steps ({model.n_steps}) 与参数 ({args.n_steps}) 不一致，正在调整..."
                )
                model.n_steps = args.n_steps
                model.rollout_buffer.buffer_size = args.n_steps
                model.rollout_buffer.reset()

            model.batch_size = args.batch
            model.n_epochs = args.n_epochs
        except ValueError as exc:
            # 观测空间不匹配（例如通道数从8改为11）时，自动从头训练
            if "Observation spaces do not match" in str(exc):
                print("观测空间已变更（例如网格通道数从8 -> 11），将从零开始重新训练")
                load_path = None
            else:
                raise

    if not load_path:
        # 使用余弦退火学习率 (10% 热身)
        lr_schedule = cosine_schedule(args.lr, warmup_steps=0.1)

        # 增强策略网络配置
        policy_kwargs.update(
            dict(
                activation_fn=torch.nn.GELU,  # 使用 GELU 激活函数 (比 ReLU 更平滑)
                optimizer_class=torch.optim.AdamW,  # 使用 AdamW 优化器 (更好的权重衰减)
                optimizer_kwargs=dict(weight_decay=1e-5),
            )
        )

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch,
            n_epochs=args.n_epochs,
            gamma=0.995,  # 提高 Gamma (0.99 -> 0.995) 以关注更长远的未来
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.start_ent,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,  # 新增: 目标 KL 散度 (防止策略更新过猛)
            policy_kwargs=policy_kwargs,
            verbose=1,  # 关闭SB3日志，用自己的输出
            device=device,
        )

    return model


