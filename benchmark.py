import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import shutil
from datetime import datetime

from benchmark_plots import generate_benchmark_plots
from models.ddqn.evaluate import evaluate_ddqn
from models.ddqn.worker_pool import build_ddqn_env
from models.ppo.evaluate import evaluate_ppo
from training.args import get_args
from training.evaluation import (
    EpisodeEvalResult,
    EvaluationWriter,
    elapsed_since,
    load_evaluation_config,
    new_eval_id,
    summarize_diagnostics,
    summarize_eval_results,
    summarize_plant_stats,
    time_eval_run,
)
from training.game_instances import prepare_game_instances, terminate_pvz_processes
from training.paths import get_cached_model_path
from training.registry import available_algorithms
from training.specs import build_base_eval_specs
from utils.train_utils import load_training_config


def _split_episodes(total, num_workers):
    """Split *total* episodes evenly across *num_workers* workers.

    Returns a list of ``(start_index, count)`` tuples.  Workers with zero
    episodes are omitted.
    """
    base = total // num_workers
    remainder = total % num_workers
    splits = []
    start = 0
    for i in range(num_workers):
        count = base + (1 if i < remainder else 0)
        if count > 0:
            splits.append((start, count))
            start += count
    return splits


def _random_worker_run(
    args,
    instance,
    env_spec,
    scenario_spec,
    num_episodes,
    start_index,
    total_episodes,
    worker_id,
    stop_event=None,
):
    """Run random-eval episodes on a single game instance (used by workers)."""
    env = None
    details = []
    try:
        env = build_ddqn_env(
            args,
            instance,
            worker_id=f"eval-random-w{worker_id}",
            env_spec=env_spec,
            scenario_spec=scenario_spec,
        )
        _wait_idx = env.action_space.n - 1
        for i in range(num_episodes):
            if stop_event and stop_event.is_set():
                break
            state = env.reset()
            done = False
            total_reward = 0.0
            actions = 0
            info = {}
            _wait_choice = 0   # 能种却选了 wait
            _wait_forced = 0   # 只能 wait
            _ep_step_times = []  # 记录每步耗时用于诊断
            _ep_gaps = []        # 记录相邻 step 完成的时间间隔
            _prev_done_ts = __import__("time").time()  # 上一轮 step 完成的时间戳
            while not done:
                if stop_event and stop_event.is_set():
                    break
                mask = env.mask_available_actions()
                valid_actions = [j for j, m in enumerate(mask) if m > 0]
                if valid_actions:
                    action = random.choice(valid_actions)
                else:
                    # mask 全为零时的安全兜底：使用 wait 动作（最后一个动作）
                    action = len(mask) - 1
                if action == _wait_idx:
                    _has_nonwait = any(mask[:-1]) if len(mask) > 1 else False
                    if _has_nonwait:
                        _wait_choice += 1
                    else:
                        _wait_forced += 1
                    print(
                        f"[Benchmark][Random][W{worker_id}] WARNING: "
                        f"empty action mask at episode {start_index + i + 1} "
                        f"step {actions}, falling back to wait action={action}",
                        flush=True,
                    )
                _t0 = __import__("time").time()
                state, reward, done, info = env.step(action)
                _dt = __import__("time").time() - _t0
                _ep_step_times.append(_dt)

                # 距离上次 step 完成的时间差（包含本步耗时 + 动作选择开销）
                _now = __import__("time").time()
                _gap = _now - _prev_done_ts
                _ep_gaps.append(_gap)
                _prev_done_ts = _now

                if _dt > 1.0:
                    print(
                        f"[Benchmark][Random][W{worker_id}] SLOW STEP "
                        f"ep={start_index + i + 1}/{total_episodes} "
                        f"step={actions} dt={_dt:.2f}s gap={_gap:.2f}s "
                        f"action={action} n_valid={len(valid_actions)}",
                        flush=True,
                    )
                total_reward += float(reward)
                actions += 1

            episode_index = start_index + i + 1
            _step_timing = {
                "total_sec": float(sum(_ep_step_times)),
                "max_sec": float(max(_ep_step_times)) if _ep_step_times else 0.0,
                "max_gap_sec": float(max(_ep_gaps)) if _ep_gaps else 0.0,
                "slow_steps": sum(1 for t in _ep_step_times if t > 1.0),
                "mean_sec": float(sum(_ep_step_times) / len(_ep_step_times))
                if _ep_step_times else 0.0,
                "wait_choice": _wait_choice,
                "wait_forced": _wait_forced,
            }
            _diag = dict(info.get("diagnostics", {}))
            _diag["step_timing"] = _step_timing
            details.append(
                EpisodeEvalResult(
                    eval_id="",
                    episode_index=episode_index,
                    reward=float(total_reward),
                    survival=float(
                        info.get("steps", getattr(env, "steps", actions))
                    ),
                    win=bool(info.get("win") is True),
                    game_ended=bool(info.get("game_ended", done)),
                    completed_sublevels=_optional_int(
                        info.get("completed_sublevels")
                    ),
                    zombies_killed=_optional_int(info.get("zombies_killed")),
                    plants_lost=_optional_int(info.get("plants_lost")),
                    actions=actions,
                    extra={
                        "current_sublevel_index": info.get(
                            "current_sublevel_index"
                        ),
                        "sublevel_cleared_this_step": info.get(
                            "sublevel_cleared_this_step"
                        ),
                        "plant_stats": info.get("plant_stats", {}),
                        "diagnostics": _diag,
                    },
                )
            )
            _st = _step_timing
            print(
                f"[Benchmark][Random][W{worker_id}] episode {episode_index}/{total_episodes} | "
                f"reward={total_reward:.2f} | "
                f"survival={details[-1].survival:.0f} | "
                f"win={details[-1].win} | "
                f"actions={actions} | "
                f"wait(choice={_wait_choice}, forced={_wait_forced}) | "
                f"max_gap={_st['max_gap_sec']:.2f}s | "
                f"step_time(total={_st['total_sec']:.1f}s, max={_st['max_sec']:.2f}s, "
                f"slow={_st['slow_steps']}, mean={_st['mean_sec']*1000:.0f}ms)",
                flush=True,
            )
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
    return details


def _random_worker_proc(
    result_queue,
    stop_event,
    args,
    instance,
    env_spec,
    scenario_spec,
    num_episodes,
    start_index,
    total_episodes,
    worker_id,
):
    """Multiprocessing worker target — wraps _random_worker_run.

    Puts ``("ok", details)`` or ``("error", (worker_id, msg))`` into
    *result_queue* so the main process can collect results.
    """
    try:
        details = _random_worker_run(
            args=args,
            instance=instance,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            num_episodes=num_episodes,
            start_index=start_index,
            total_episodes=total_episodes,
            worker_id=worker_id,
            stop_event=stop_event,
        )
        result_queue.put(("ok", details))
    except KeyboardInterrupt:
        pass  # 被主进程终止，静默退出
    except Exception as exc:
        result_queue.put(("error", (worker_id, repr(exc))))


def evaluate_random(
    args,
    instances,
    env_spec,
    scenario_spec,
    episodes,
    num_workers=1,
):
    """Evaluate with uniformly random actions (respecting action masks).

    Uses *multiprocessing.Process* workers (same pattern as training) so
    that Ctrl+C can terminate stuck workers immediately via
    ``Process.terminate()``.
    """
    if not instances:
        raise ValueError("Random eval requires at least one game instance")

    num_workers = max(1, min(num_workers, len(instances)))
    eval_id = new_eval_id("real_random")
    start_time = time_eval_run()

    episode_splits = _split_episodes(episodes, num_workers)
    actual_workers = len(episode_splits)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    result_queue = ctx.Queue()

    print(
        f"[Benchmark][Random] Dispatching {episodes} episodes "
        f"across {actual_workers} workers (multiprocessing)"
    )

    processes = []
    for worker_id, (start_idx, count) in enumerate(episode_splits):
        p = ctx.Process(
            target=_random_worker_proc,
            args=(
                result_queue,
                stop_event,
                args,
                instances[worker_id],
                env_spec,
                scenario_spec,
                count,
                start_idx,
                episodes,
                worker_id,
            ),
        )
        p.start()
        processes.append(p)
        print(
            f"[Benchmark][Random] Worker {worker_id} started: "
            f"pid={instances[worker_id]['pid']} port={instances[worker_id]['port']}"
        )

    all_details = []
    completed = 0
    try:
        while completed < len(processes):
            try:
                status, data = result_queue.get(timeout=1.0)
            except queue.Empty:
                # 检查是否有进程异常退出
                for p in processes:
                    if p.exitcode is not None and p.exitcode != 0:
                        print(
                            f"[Benchmark][Random] Worker exited with "
                            f"code {p.exitcode}",
                            flush=True,
                        )
                continue
            completed += 1
            if status == "ok":
                all_details.extend(data)
            else:
                wid, error = data
                print(
                    f"[Benchmark][Random] Worker {wid} error: {error}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print(
            "\n[Benchmark][Random] Interrupted, stopping workers...",
            flush=True,
        )
        stop_event.set()
    finally:
        # 对齐训练 AsyncWorkerPool.stop() 的终止流程
        for p in processes:
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

    all_details.sort(key=lambda d: d.episode_index)

    return summarize_eval_results(
        eval_id=eval_id,
        algo="random",
        env_kind="real",
        episode=None,
        step=None,
        stage_name="base",
        win_condition=scenario_spec.win_condition,
        target_sublevels=scenario_spec.target_sublevels,
        details=all_details,
        duration_sec=elapsed_since(start_time),
        model_path=None,
        extra={
            "game_mode_id": scenario_spec.game_mode_id,
            "rows": scenario_spec.rows,
            "cols": scenario_spec.cols,
            "initial_sun": scenario_spec.initial_sun,
            "cards": list(scenario_spec.cards),
            "plant_stats": summarize_plant_stats(all_details),
            "diagnostics": summarize_diagnostics(all_details),
        },
    )


def main(argv=None):
    eval_args, train_argv = _parse_eval_args(argv)
    args = get_args(train_argv)
    if not hasattr(args, "speed"):
        args.speed = 5.0
    eval_config = _load_eval_config(args)
    _apply_eval_instance_config(args, eval_args, eval_config)

    model_path = (
        None if eval_args.random
        else eval_args.model or get_cached_model_path(args.algo)
    )
    algo_label = "random" if eval_args.random else args.algo
    output_dir = eval_args.eval_output or _default_benchmark_output(algo_label)
    episodes = eval_args.eval_episodes or 100

    env_spec, scenario_spec = build_base_eval_specs(args)
    _print_eval_metadata(
        args=args,
        model_path=model_path,
        output_dir=output_dir,
        episodes=episodes,
        env_spec=env_spec,
        scenario_spec=scenario_spec,
        random_mode=eval_args.random,
    )
    instances = prepare_game_instances(args)
    if instances is None:
        return
    _print_instances(instances)

    try:
        if eval_args.random:
            result = evaluate_random(
                args=args,
                instances=instances,
                env_spec=env_spec,
                scenario_spec=scenario_spec,
                episodes=episodes,
                num_workers=eval_args.eval_workers,
            )
        elif args.algo == "ddqn":
            result = evaluate_ddqn(
                args=args,
                model_path=model_path,
                instances=instances,
                env_spec=env_spec,
                scenario_spec=scenario_spec,
                episodes=episodes,
                num_workers=eval_args.eval_workers,
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
                num_workers=eval_args.eval_workers,
            )
        else:
            raise NotImplementedError(
                f"Offline evaluate is not implemented for algo: {args.algo}"
            )
    except KeyboardInterrupt:
        print("\n[Benchmark] Interrupted by user", flush=True)
        return
    finally:
        terminate_pvz_processes(instances, auto_start=args.auto_start)

    writer = EvaluationWriter(
        output_dir,
        save_episode_details=eval_config.save_episode_details,
    )
    writer.write(result)
    if model_path:
        copied_model_path = _copy_model_to_eval_output(model_path, output_dir)
    else:
        copied_model_path = None
    plant_stats_path = _write_plant_stats(result, output_dir)
    diagnostics_paths = _write_diagnostics(result, output_dir)

    # ── Benchmark analysis plots ──
    plot_paths = generate_benchmark_plots(
        result=result,
        output_dir=output_dir,
        rows=scenario_spec.rows,
        cols=scenario_spec.cols,
    )

    _print_eval_result(result)
    print(f"Saved eval summary to {writer.csv_path}")
    if plant_stats_path:
        print(f"Saved plant stats to {plant_stats_path}")
    if diagnostics_paths:
        print(f"Saved diagnostics to {', '.join(diagnostics_paths)}")
    if copied_model_path:
        print(f"Copied eval model to {copied_model_path}")
    if plot_paths:
        print(f"Saved benchmark plots to {', '.join(plot_paths)}")


def _parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark a trained model on 困难草地第一关 (game_mode_id=6)")
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
    parser.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="Use random actions instead of a trained model (baseline evaluation)",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=1,
        help="Number of parallel workers for evaluation (requires multiple game instances)",
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
    num_workers = max(1, int(getattr(eval_args, "eval_workers", 1)))
    args.num_envs = num_workers
    # base_port is already set via get_args() from training_config.yaml.
    # EvaluationConfig does not carry port overrides — ports come from
    # the training config's training.args.base_port field.


def _default_benchmark_output(algo):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("benchmark_output", algo, timestamp)


def _optional_int(value):
    if value is None:
        return None
    return int(value)


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

    # Filter to only dict entries (skip internal keys like "_placements")
    items = [
        v for v in (plant_stats.values() if isinstance(plant_stats, dict) else [])
        if isinstance(v, dict)
    ]
    if not items:
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
        for item in sorted(items, key=lambda v: int(v.get("plant_id", 0))):
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
        "step_total_sec",
        "step_max_sec",
        "step_max_gap_sec",
        "step_slow_count",
        "step_mean_sec",
        "wait_choice",
        "wait_forced",
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
                         env_spec, scenario_spec, random_mode=False):
    sep = "-" * 58
    print(f"\n{sep}")
    print("  Benchmark Configuration")
    print(f"{sep}")
    algo_label = "random (baseline)" if random_mode else args.algo
    print(f"  {'Algorithm:':24s} {algo_label}")
    model_label = model_path or "(none — random mode)"
    print(f"  {'Model path:':24s} {model_label}")
    print(f"  {'Output dir:':24s} {output_dir}")
    print(f"  {'Episodes:':24s} {episodes}")
    print(f"  {'Eval envs:':24s} {args.num_envs}")
    num_workers = max(1, int(getattr(args, "num_envs", 1)))
    print(f"  {'Parallel workers:':24s} {num_workers}")
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
    print("[Benchmark] Instances: " + ", ".join(
        f"pid={item['pid']} port={item['port']}" for item in instances
    ))


def _print_eval_result(result):
    sep = "-" * 58
    print(f"\n{sep}")
    print("  Benchmark Result")
    print(f"{sep}")
    print(f"  {'Reward:':20s} mean={result.reward_mean:8.2f}  "
          f"median={result.reward_median:8.2f}  "
          f"std={result.reward_std:8.2f}  "
          f"min={result.reward_min:8.2f}  max={result.reward_max:8.2f}")
    print(f"  {'Survival:':20s} mean={result.survival_mean:8.2f}  "
          f"median={result.survival_median:8.2f}  "
          f"std={result.survival_std:8.2f}  "
          f"min={result.survival_min:8.0f}  max={result.survival_max:8.0f}")
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
            f"step_time(total={float(item.get('step_total_sec', 0)):.1f}s, "
            f"max={float(item.get('step_max_sec', 0)):.2f}s, "
            f"gap={float(item.get('step_max_gap_sec', 0)):.2f}s, "
            f"slow={int(item.get('step_slow_count', 0))})  "
            f"rewards={_format_top_items(item.get('reward_breakdown', {}), precision=1)}"
        )


def _print_plant_stats(result):
    plant_stats = (result.extra or {}).get("plant_stats") or {}
    if not plant_stats:
        return

    # Filter to only dict entries (skip internal keys like "_placements")
    items = [
        v for v in (plant_stats.values() if isinstance(plant_stats, dict) else [])
        if isinstance(v, dict)
    ]
    if not items:
        return

    print(f"  {'Plant stats:':20s}")
    rows = sorted(items, key=lambda v: (-int(v.get("count_total", 0)), v.get("name", "")))
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
