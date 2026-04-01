import os
import torch
from models.ddqn.train_entry import train_ddqn

# reduce TensorFlow logs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


torch.set_num_threads(8)


from train_args import get_args
from train_env import get_env
from train_model import get_model
from train_utils import (
    print_metadata,
    build_callbacks,
    train_model as run_ppo,
)
import train_utils


def main():
    args = get_args()
    train_utils.setup_logging(args)
    train_utils.prepare_resume_paths(args)

    print_metadata(args)
    device = train_utils.setup_device()

    if not train_utils.ensure_game_ready(args):
        return

    training_metrics_callback = train_utils.build_training_metrics_callback(args)

    if args.algo == "ddqn":
        train_ddqn(args, metrics_callback=training_metrics_callback)
        return
    else:  # ppo
        # 创建环境
        print(f"建环境...")
        env = get_env(args)

        # 创建/加载模型
        print(f"创建模型...")
        model = get_model(args, env, device)

        # GPU 显存
        train_utils.print_gpu_memory()

        # 设置回调
        callbacks = build_callbacks(args)

        # 开始训练
        run_ppo(model, env, args, callbacks)


if __name__ == "__main__":
    main()

