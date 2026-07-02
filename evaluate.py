import argparse
import csv
import json
import os
import shutil
from datetime import datetime

from models.ddqn.evaluate import evaluate_ddqn
from models.ppo.evaluate import evaluate_ppo
from training.args import get_args
from training.evaluation import EvaluationWriter, load_evaluation_config
from training.game_instances import prepare_game_instances
from training.paths import get_cached_model_path
from training.registry import available_algorithms
from training.specs import build_base_eval_specs
from utils.train_utils import load_training_config


def main(argv=None):
    eval_args, train_argv = _parse_eval_args(argv)
    args = get_args(train_argv)
    eval_config = _load_eval_config(args)
    _apply_eval_instance_config(args, eval_args, eval_config)

    model_path = eval_args.model or get_cached_model_path(args.algo)
    output_dir = eval_args.eval_output or _default_eval_output(args.algo)
    episodes = eval_args.eval_episodes or eval_config.episodes

    env_spec, scenario_spec = build_base_eval_specs(args)
    _print_eval_metadata(
        args=args,
        model_path=model_path,
        output_dir=output_dir,
        episodes=episodes,
        env_spec=env_spec,
        scenario_spec=scenario_spec,
    )
    instances = prepare_game_instances(args)
    if instances is None:
        return
    _print_instances(instances)

    if args.algo == "ddqn":
        result = evaluate_ddqn(
            args=args,
            model_path=model_path,
            instances=instances,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            episodes=episodes,
        )
    elif args.algo == "ppo":
        result = evaluate_ppo(
            args=args,
            model_path=model_path,
            instances=instances,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            episodes=episodes,
            device="auto",
        )
    else:
        raise NotImplementedError(
            f"Offline evaluate is not implemented for algo: {args.algo}"
        )

    writer = EvaluationWriter(
        output_dir,
        save_episode_details=eval_config.save_episode_details,
    )
    writer.write(result)
    copied_model_path = _copy_model_to_eval_output(model_path, output_dir)
    plant_stats_path = _write_plant_stats(result, output_dir)
    diagnostics_paths = _write_diagnostics(result, output_dir)
    _print_eval_result(result)
    print(f"Saved eval summary to {writer.csv_path}")
    if plant_stats_path:
        print(f"Saved plant stats to {plant_stats_path}")
    if diagnostics_paths:
        print(f"Saved diagnostics to {', '.join(diagnostics_paths)}")
    if copied_model_path:
        print(f"Copied eval model to {copied_model_path}")


def _parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained real-env model")
    parser.add_argument(
        "--algo",
        type=str,
        choices=available_algorithms(),
        default=None,
        help="Algorithm to evaluate",
    )
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=None,
        help="Number of independent eval episodes",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Directory for eval.jsonl/eval.csv/eval_snapshot.json",
    )
    eval_args, train_argv = parser.parse_known_args(argv)
    if eval_args.algo:
        train_argv.extend(["--algo", eval_args.algo])
    return eval_args, train_argv


def _load_eval_config(args):
    config = load_training_config(args.training_config)
    return load_evaluation_config(
        config.get("training", {}).get("eval", {})
    )


def _apply_eval_instance_config(args, eval_args, eval_config):
    args.num_envs = 1
    if eval_config.real_base_port is not None:
        args.base_port = eval_config.real_base_port
        args.port = eval_config.real_base_port


def _default_eval_output(algo):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("eval_output", algo, timestamp)


def _copy_model_to_eval_output(model_path, output_dir):
    if not model_path:
        return None
    if not os.path.isfile(model_path):
        return None
    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, os.path.basename(model_path))
    if os.path.abspath(model_path) == os.path.abspath(destination):
        return destination
    shutil.copy2(model_path, destination)
    return destination


def _write_plant_stats(result, output_dir):
    plant_stats = (result.extra or {}).get("plant_stats") or {}
    if not plant_stats:
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "plant_stats.csv")
    fieldnames = [
        "plant_id",
        "name",
        "count_total",
        "survival_steps_mean",
        "survival_steps_total",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(
            plant_stats.values(),
            key=lambda value: int(value.get("plant_id", 0)),
        ):
            writer.writerow({field: item.get(field) for field in fieldnames})
    return path


def _write_diagnostics(result, output_dir):
    diagnostics = (result.extra or {}).get("diagnostics") or {}
    if not diagnostics:
        return None

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "diagnostics.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "diagnostics.csv")
    fieldnames = [
        "episode_index",
        "wait_actions",
        "plant_actions",
        "shovel_actions",
        "invalid_actions",
        "zombies_killed",
        "plants_lost",
        "final_sun",
        "max_sun",
        "mean_sun",
        "sun_gained",
        "sun_spent",
        "wait_with_high_sun",
        "reward_breakdown",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in diagnostics.get("per_episode", []):
            row = {field: item.get(field) for field in fieldnames}
            row["reward_breakdown"] = json.dumps(
                row.get("reward_breakdown") or {},
                ensure_ascii=False,
            )
            writer.writerow(row)
    return [json_path, csv_path]


def _print_eval_metadata(args, model_path, output_dir, episodes,
                         env_spec, scenario_spec):
    sep = "-" * 58
    print(f"\n{sep}")
    print("  Evaluation Configuration")
    print(f"{sep}")
    print(f"  {'Algorithm:':24s} {args.algo}")
    print(f"  {'Model path:':24s} {model_path}")
    print(f"  {'Output dir:':24s} {output_dir}")
    print(f"  {'Episodes:':24s} {episodes}")
    print(f"  {'Eval envs:':24s} {args.num_envs}")
    print(f"  {'Base port:':24s} {getattr(args, 'base_port', args.port)}")
    print(f"  {'Grid:':24s} {env_spec.rows}x{env_spec.cols}")
    print(f"  {'Actions:':24s} {env_spec.action_space_size}")
    print(f"  {'Cards:':24s} {list(scenario_spec.cards)}")
    print(f"  {'Game mode:':24s} {scenario_spec.game_mode_id}")
    print(f"  {'Initial sun:':24s} {scenario_spec.initial_sun}")
    print(f"  {'Win condition:':24s} {scenario_spec.win_condition}")
    print(f"  {'Target sublevels:':24s} {scenario_spec.target_sublevels}")
    print(f"{sep}\n")


def _print_instances(instances):
    print("[Eval] Instances: " + ", ".join(
        f"pid={item['pid']} port={item['port']}" for item in instances
    ))


def _print_eval_result(result):
    sep = "-" * 58
    print(f"\n{sep}")
    print("  Evaluation Result")
    print(f"{sep}")
    print(f"  {'Reward:':20s} mean={result.reward_mean:8.2f}  "
          f"std={result.reward_std:8.2f}  min={result.reward_min:8.2f}  "
          f"max={result.reward_max:8.2f}")
    print(f"  {'Survival:':20s} mean={result.survival_mean:8.2f}  "
          f"std={result.survival_std:8.2f}  min={result.survival_min:8.0f}  "
          f"max={result.survival_max:8.0f}")
    print(f"  {'Win rate:':20s} {result.win_count}/{result.episodes} "
          f"({100 * result.win_rate:.1f}%)")
    print(f"  {'Duration:':20s} {result.duration_sec:.2f}s")
    _print_eval_diagnostics(result)
    _print_plant_stats(result)
    print(f"{sep}\n")


def _print_eval_diagnostics(result):
    diagnostics = (result.extra or {}).get("diagnostics") or {}
    if not diagnostics:
        return

    rows = diagnostics.get("per_episode", [])
    if not rows:
        return

    print(f"  {'Episode diagnostics:':20s}")
    for item in rows:
        print(
            f"    episode={int(item.get('episode_index', 0)):3d}  "
            f"actions(wait={int(item.get('wait_actions', 0))}, "
            f"plant={int(item.get('plant_actions', 0))}, "
            f"shovel={int(item.get('shovel_actions', 0))}, "
            f"invalid={int(item.get('invalid_actions', 0))})  "
            f"killed={int(item.get('zombies_killed', 0))}  "
            f"lost={int(item.get('plants_lost', 0))}  "
            f"rewards={_format_top_items(item.get('reward_breakdown', {}), precision=1)}"
        )


def _print_plant_stats(result):
    plant_stats = (result.extra or {}).get("plant_stats") or {}
    if not plant_stats:
        return

    print(f"  {'Plant stats:':20s}")
    rows = sorted(
        plant_stats.values(),
        key=lambda item: (-int(item.get("count_total", 0)), item.get("name", "")),
    )
    for item in rows:
        print(
            f"    {item.get('name', item.get('plant_id')):18s} "
            f"count={int(item.get('count_total', 0)):4d}  "
            f"mean_survival_steps={float(item.get('survival_steps_mean', 0.0)):8.2f}  "
            f"total_survival_steps={float(item.get('survival_steps_total', 0.0)):8.2f}"
        )


def _format_top_items(items, limit=4, precision=0):
    if not items:
        return "none"
    top = sorted(items.items(), key=lambda item: abs(float(item[1])), reverse=True)
    values = []
    for key, value in top[:limit]:
        if precision:
            values.append(f"{key}={float(value):.{precision}f}")
        else:
            values.append(f"{key}={int(value)}")
    return ", ".join(values)


if __name__ == "__main__":
    main()
