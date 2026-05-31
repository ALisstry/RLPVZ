import glob
import os

from stable_baselines3.common.vec_env import VecNormalize

from training.paths import (
    build_checkpoint_paths,
    get_cached_model_path,
)


LEGACY_MODEL_PATH = "models/latest_model.zip"


def find_latest_model():
    if os.path.exists(LEGACY_MODEL_PATH):
        return LEGACY_MODEL_PATH

    patterns = [
        "models/advanced_*/final_model.zip",
        "models/*/final_model.zip",
        "models/*.zip",
    ]

    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))

    if not all_models:
        return None

    return max(all_models, key=os.path.getmtime)


def prepare_resume(args, run_paths=None):
    if args.load:
        print(f"使用参数指定模型路径：{args.load}")
        return

    if args.no_auto_resume:
        print("自动恢复已禁用，从零开始训练")
        return

    cached_path = (
        run_paths.cached_model_path if run_paths else get_cached_model_path("ppo")
    )
    if os.path.exists(cached_path):
        args.load = cached_path
        print(f"自动恢复 PPO: {cached_path}")
        return

    load_path = find_latest_model()
    if load_path:
        args.load = load_path
        print(f"自动恢复: 找到最新模型 {load_path}")
    else:
        print("未找到已有模型，从零开始训练")


def resolve_load_path(args, run_paths=None):
    return args.load


def save_checkpoint(args, payload=None, run_paths=None, **_kwargs):
    model = payload.model if payload is not None else None
    env = payload.env if payload is not None else None
    tag = payload.tag if payload is not None else None
    if model is None:
        return None

    paths = build_checkpoint_paths(
        "ppo",
        run_paths=run_paths,
        explicit_path=getattr(args, "save_path", None),
        tag=tag,
    )

    if paths.explicit_path:
        os.makedirs(os.path.dirname(paths.explicit_path) or ".", exist_ok=True)
        model.save(paths.explicit_path)
    model.save(paths.cached_path)
    if paths.tagged_path:
        model.save(paths.tagged_path)

    vec_normalize = _find_vec_normalize(env)
    if vec_normalize is not None:
        try:
            if paths.explicit_path:
                explicit_vec_path = (
                    os.path.splitext(paths.explicit_path)[0] + "_vecnormalize.pkl"
                )
                vec_normalize.save(explicit_vec_path)
            vec_normalize.save(
                os.path.splitext(paths.cached_path)[0] + "_vecnormalize.pkl"
            )
            if paths.tagged_path:
                tagged_vec_path = (
                    os.path.splitext(paths.tagged_path)[0] + "_vecnormalize.pkl"
                )
                vec_normalize.save(tagged_vec_path)
        except Exception:
            pass

    print(f"\n模型已保存: {paths.cached_path}")
    return paths.cached_path


def _find_vec_normalize(current_env):
    if current_env is None:
        return None
    if isinstance(current_env, VecNormalize):
        return current_env
    for attr in ("venv", "env"):
        nested = getattr(current_env, attr, None)
        if nested is not None:
            found = _find_vec_normalize(nested)
            if found is not None:
                return found
    return None
