"""Training loop for the simulation environment."""

import gc
import csv
import json
import os
import signal
import shutil
import subprocess
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from simenv import SimPVZEnv
from simenv.config import CURRICULUM
from simenv.curriculum import build_curriculum
from simenv.pvz_sim import config
from simenv.model import (
    ReplayBuffer, DDQNNetwork, DifferentialDDQNNetwork,
    transform_observation, calculate_loss, LossResult,
)
from training.evaluation import (
    BestEvaluationCheckpoint,
    EpisodeEvalResult,
    EvaluationConfig,
    EvaluationScheduler,
    EvaluationWriter,
    elapsed_since,
    new_eval_id,
    summarize_diagnostics,
    summarize_eval_results,
    summarize_plant_stats,
    time_eval_run,
)


class EpsilonSchedule:
    def __init__(self, seq_length, start_epsilon=1.0, end_epsilon=0.05):
        self.seq_length = max(1, int(seq_length))
        self.start_epsilon = float(start_epsilon)
        self.end_epsilon = float(end_epsilon)

    def epsilon(self, index):
        ratio = min(1.0, max(0.0, index / self.seq_length))
        return self.end_epsilon + (
            self.start_epsilon - self.end_epsilon
        ) * np.exp(-5.0 * ratio)


def train_sim(
    max_episodes=200000,
    buffer_size=100000,
    burn_in=10000,
    batch_size=512,
    gamma=0.99,
    lr=1e-4,
    network_update_freq=32,
    network_sync_freq=2000,
    save_path=None,
    eval_episodes=100,
    eval_freq_episodes=2500,
    max_grad_norm=0.5,
    visualize=False,
    plot_freq=100,
    plot_callback=None,
    use_differential=False,
):
    if save_path is None:
        save_path = _default_save_path("ddqn", "sim_ddqn.pt")
    output_dir = os.path.dirname(save_path) or "."
    eval_config = EvaluationConfig(
        enabled=eval_freq_episodes > 0 and eval_episodes > 0,
        freq_episodes=eval_freq_episodes,
        episodes=eval_episodes,
        deterministic=True,
        save_episode_details=True,
    )
    eval_scheduler = EvaluationScheduler(eval_config)
    eval_writer = EvaluationWriter(
        output_dir,
        save_episode_details=eval_config.save_episode_details,
    )
    best_eval_checkpoint = BestEvaluationCheckpoint(
        output_dir,
        model_filename="best_model.pt",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimPVZEnv()
    curriculum = build_curriculum(
        CURRICULUM,
        rows=env.rows,
        plant_ids=env.card_plant_ids,
    )
    if getattr(curriculum, "enabled", True):
        env.apply_stage(curriculum.current_stage)
    NetworkCls = DifferentialDDQNNetwork if use_differential else DDQNNetwork
    network = NetworkCls(env, learning_rate=lr, device=device)
    target_network = deepcopy(network)
    buffer = ReplayBuffer(memory_size=buffer_size, burn_in=burn_in)
    threshold = EpsilonSchedule(
        seq_length=max_episodes,
        start_epsilon=1.0,
        end_epsilon=0.05,
    )

    _print_config(
        device=device,
        network_type="ddqn_differential" if use_differential else "ddqn",
        network_params=sum(p.numel() for p in network.parameters()),
        max_episodes=max_episodes,
        buffer_size=buffer_size,
        burn_in=burn_in,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        max_grad_norm=max_grad_norm,
        network_update_freq=network_update_freq,
        network_sync_freq=network_sync_freq,
        eval_episodes=eval_episodes,
        eval_freq_episodes=eval_freq_episodes,
        epsilon_decay=f"{threshold.start_epsilon} -> {threshold.end_epsilon} (exponential)",
        curriculum_enabled=getattr(curriculum, "enabled", False),
        curriculum_stage=curriculum.current_stage_name,
        env_config={
            "rows": config.N_LANES,
            "cols": config.LANE_LENGTH,
            "fps": config.FPS,
            "max_frames": config.MAX_FRAMES,
            "plants": list(env.plant_deck.keys()),
        },
    )
    _save_run_metadata(
        output_dir,
        network=network,
        env=env,
        device=device,
        max_episodes=max_episodes,
        buffer_size=buffer_size,
        burn_in=burn_in,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        max_grad_norm=max_grad_norm,
        network_update_freq=network_update_freq,
        network_sync_freq=network_sync_freq,
        eval_config=eval_config,
        curriculum=curriculum,
    )

    training_rewards = []
    training_iterations = []
    # Per‑training‑step diagnostics (one entry per optimizer step)
    training_loss = []
    training_advantage = []
    training_entropy = []
    training_mean_q = []
    training_max_q = []
    training_td_error = []
    training_grad_norm = []
    training_q_wait = []
    training_delta_mean = []
    training_delta_max = []
    step_count = 0
    window = 100
    last_eval_episode = None
    curriculum_completed_printed = False
    saved_on_interrupt = False
    stop_requested = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def _handle_interrupt(signum, frame):
        nonlocal saved_on_interrupt, stop_requested
        stop_requested = True
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if not saved_on_interrupt:
            saved_on_interrupt = True
            print("\nTraining interrupted. Saving current model...")
            _save_training_checkpoint(
                save_path,
                network,
                training_rewards,
                training_iterations,
                training_loss,
                training_advantage=training_advantage,
                training_entropy=training_entropy,
                training_mean_q=training_mean_q,
                training_max_q=training_max_q,
                training_td_error=training_td_error,
                training_grad_norm=training_grad_norm,
                training_q_wait=training_q_wait,
                training_delta_mean=training_delta_mean,
                training_delta_max=training_delta_max,
                plot_callback=plot_callback,
            )

    signal.signal(signal.SIGINT, _handle_interrupt)

    s_0, mask, step_count, stopped = _burn_in_stage(
        env,
        buffer,
        step_count=step_count,
        stage_name=curriculum.current_stage_name,
        stop_requested=lambda: stop_requested,
    )
    if stopped:
        print(f"Training stopped during burn-in at {step_count} steps.")
        signal.signal(signal.SIGINT, previous_sigint_handler)
        return

    ep = 0
    print(f"Training {max_episodes} episodes...")

    while ep < max_episodes and not stop_requested:
        rewards = 0
        done = False
        info = {}
        while not done and not stop_requested:
            if getattr(curriculum, "enabled", False):
                epsilon = curriculum.epsilon()
            else:
                epsilon = threshold.epsilon(ep)
            action = network.decide_action(s_0, mask, epsilon=epsilon)
            s_1, r, done, info = env.step(action)
            s_1 = transform_observation(s_1)
            next_mask = info.get("mask", env.mask_available_actions())
            rewards += r
            buffer.append(s_0, action, r, done, s_1, mask, next_mask)
            s_0 = s_1               # transform_observation already copied
            mask = next_mask         # carry forward to next step
            step_count += 1

            if step_count % network_update_freq == 0:
                network.optimizer.zero_grad(set_to_none=True)
                batch = buffer.sample_batch(batch_size=batch_size)
                result = calculate_loss(network, target_network, batch, gamma)
                result.loss.backward()
                # ── Gradient norm (before optimizer step) ──
                total_norm = 0.0
                for p in network.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = total_norm ** 0.5
                nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
                network.optimizer.step()
                training_loss.append(result.loss.detach().item())
                training_advantage.append(result.diagnostics["advantage"])
                training_entropy.append(result.diagnostics["entropy"])
                training_mean_q.append(result.diagnostics["mean_q"])
                training_max_q.append(result.diagnostics["max_q"])
                training_td_error.append(result.diagnostics["td_error"])
                training_grad_norm.append(grad_norm)
                training_q_wait.append(result.diagnostics["q_wait"])
                training_delta_mean.append(result.diagnostics["delta_mean"])
                training_delta_max.append(result.diagnostics["delta_max"])

            if step_count % network_sync_freq == 0:
                target_network.load_state_dict(network.state_dict())

            if done:
                ep += 1
                training_rewards.append(rewards)
                training_iterations.append(float(info.get("steps", env._scene._chrono)))

                old_stage = curriculum.current_stage_name
                curriculum.record_episode()
                stage_changed = False

                if _should_run_eval(ep, curriculum, eval_config, eval_scheduler):
                    eval_result = _run_and_save_eval(
                        network,
                        eval_writer,
                        eval_config,
                        best_eval_checkpoint=(
                            best_eval_checkpoint
                            if _should_save_best(curriculum)
                            else None
                        ),
                        episode=ep,
                        step=step_count,
                        stage=curriculum.current_stage,
                    )
                    last_eval_episode = ep
                    if (
                        getattr(curriculum, "enabled", False)
                        and eval_result is not None
                    ):
                        stage_changed = curriculum.advance(eval_result)
                        if stage_changed:
                            new_stage = curriculum.current_stage_name
                            print(f"[Curriculum] stage changed: {old_stage} -> {new_stage}")
                            _append_curriculum_event(
                                output_dir,
                                ep,
                                old_stage,
                                new_stage,
                                eval_result,
                            )
                            last_eval_episode = None
                if (
                    getattr(curriculum, "enabled", False)
                    and curriculum.completed
                    and not curriculum_completed_printed
                ):
                    curriculum_completed_printed = True
                    print(
                        f"[Curriculum] completed at episode={ep} "
                        f"stage_episode={curriculum.stage_episode}"
                    )

                if ep % 100 == 0:
                    gc.collect()
                    step_win = max(100, window * network_update_freq)
                    mean_r = np.mean(training_rewards[-window:])
                    mean_i = np.mean(training_iterations[-window:])
                    mean_l = np.nanmean(training_loss[-step_win:]) if training_loss else 0
                    mean_adv = np.nanmean(training_advantage[-step_win:]) if training_advantage else 0
                    mean_ent = np.nanmean(training_entropy[-step_win:]) if training_entropy else 0
                    mean_q = np.nanmean(training_mean_q[-step_win:]) if training_mean_q else 0
                    mean_td = np.nanmean(training_td_error[-step_win:]) if training_td_error else 0
                    mean_gn = np.nanmean(training_grad_norm[-step_win:]) if training_grad_norm else 0
                    print(f"Episode {ep:5d}/{max_episodes}  "
                          f"Stage {curriculum.current_stage_name}  "
                          f"stage_episode={curriculum.stage_episode}  "
                          f"Steps {step_count:7d}  "
                          f"Mean R {mean_r:8.2f}  Mean I {mean_i:.2f}  Mean L {mean_l:.2f}  "
                          f"Adv {mean_adv:+.3f}  Ent {mean_ent:.3f}  "
                          f"Qmean {mean_q:+.2f}  |TD| {mean_td:.3f}  |grad| {mean_gn:.3f}")

                if plot_freq and ep % plot_freq == 0:
                    _save_training_artifacts(
                        save_path,
                        training_rewards,
                        training_iterations,
                        training_loss,
                        training_advantage=training_advantage,
                        training_entropy=training_entropy,
                        training_mean_q=training_mean_q,
                        training_max_q=training_max_q,
                        training_td_error=training_td_error,
                        training_grad_norm=training_grad_norm,
                        training_q_wait=training_q_wait,
                        training_delta_mean=training_delta_mean,
                        training_delta_max=training_delta_max,
                        plot_callback=plot_callback,
                    )

                if ep >= max_episodes:
                    print(f"\nEpisode limit reached ({max_episodes} episodes, {step_count} steps).")
                    break

                if stage_changed:
                    env.apply_stage(curriculum.current_stage)
                    target_network.load_state_dict(network.state_dict())
                    stage_burn_in = _stage_burn_in(curriculum.current_stage, burn_in)
                    buffer = ReplayBuffer(
                        memory_size=buffer_size,
                        burn_in=stage_burn_in,
                    )
                    s_0, mask, step_count, stopped = _burn_in_stage(
                        env,
                        buffer,
                        step_count=step_count,
                        stage_name=curriculum.current_stage_name,
                        stop_requested=lambda: stop_requested,
                    )
                    if stopped:
                        break
                else:
                    s_0 = transform_observation(env.reset())
                    mask = np.array(env.mask_available_actions())

    if stop_requested:
        print(f"Training stopped at episode {ep}, step {step_count}.")
        signal.signal(signal.SIGINT, previous_sigint_handler)
        return

    _save_training_checkpoint(
        save_path,
        network,
        training_rewards,
        training_iterations,
        training_loss,
        training_advantage=training_advantage,
        training_entropy=training_entropy,
        training_mean_q=training_mean_q,
        training_max_q=training_max_q,
        training_td_error=training_td_error,
        training_grad_norm=training_grad_norm,
        training_q_wait=training_q_wait,
        training_delta_mean=training_delta_mean,
        training_delta_max=training_delta_max,
        plot_callback=plot_callback,
    )
    print("Training complete.")

    if eval_episodes > 0 and ep != last_eval_episode:
        _run_and_save_eval(
            network,
            eval_writer,
            eval_config,
            best_eval_checkpoint=(
                best_eval_checkpoint
                if _should_save_best(curriculum)
                else None
            ),
            episode=ep,
            step=step_count,
            stage=curriculum.current_stage,
            force=True,
        )
        _save_training_artifacts(
            save_path,
            training_rewards,
            training_iterations,
            training_loss,
            training_advantage=training_advantage,
            training_entropy=training_entropy,
            training_mean_q=training_mean_q,
            training_max_q=training_max_q,
            training_td_error=training_td_error,
            training_grad_norm=training_grad_norm,
            plot_callback=plot_callback,
        )

    if visualize:
        _visualize_episode(env, network)

    signal.signal(signal.SIGINT, previous_sigint_handler)


def _print_config(**cfg):
    """Pretty-print the training configuration before starting."""
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  Training Configuration")
    print(f"{sep}")
    print(f"  {'Device:':24s} {cfg['device'].upper()}")
    print(f"  {'Network:':24s} {cfg['network_type']} ({cfg['network_params']:,} params)")
    print(f"  {'Max episodes:':24s} {cfg['max_episodes']}")
    print(f"  {'Buffer size:':24s} {cfg['buffer_size']}")
    print(f"  {'Burn-in steps:':24s} {cfg['burn_in']}")
    print(f"  {'Batch size:':24s} {cfg['batch_size']}")
    print(f"  {'Gamma:':24s} {cfg['gamma']}")
    print(f"  {'Learning rate:':24s} {cfg['lr']}")
    print(f"  {'Max grad norm:':24s} {cfg['max_grad_norm']}")
    print(f"  {'Network update freq:':24s} {cfg['network_update_freq']} steps")
    print(f"  {'Network sync freq:':24s} {cfg['network_sync_freq']} steps")
    print(f"  {'Epsilon decay:':24s} {cfg['epsilon_decay']}")
    print(f"  {'Eval episodes:':24s} {cfg['eval_episodes']}")
    print(f"  {'Eval frequency:':24s} {cfg['eval_freq_episodes']} episodes")
    print(f"  {'Curriculum:':24s} {cfg['curriculum_enabled']}")
    print(f"  {'Curriculum stage:':24s} {cfg['curriculum_stage']}")
    print(f"{sep}")
    ec = cfg["env_config"]
    print(f"  Environment")
    print(f"{sep}")
    print(f"  {'Grid:':24s} {ec['rows']}x{ec['cols']} (rows x cols)")
    print(f"  {'FPS:':24s} {ec['fps']}")
    print(f"  {'Max frames:':24s} {ec['max_frames']} ({ec['max_frames'] // ec['fps']}s game time)")
    print(f"  {'Plant deck:':24s} {', '.join(ec['plants'])}")
    print(f"{sep}\n")


def _save_run_metadata(
    output_dir,
    *,
    network,
    env,
    device,
    max_episodes,
    buffer_size,
    burn_in,
    batch_size,
    gamma,
    lr,
    max_grad_norm,
    network_update_freq,
    network_sync_freq,
    eval_config,
    curriculum,
):
    os.makedirs(output_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    _copy_if_exists(
        os.path.join(project_root, "simenv", "config.py"),
        os.path.join(output_dir, "simenv_config.py"),
    )
    _copy_if_exists(
        os.path.join(project_root, "simenv", "consts.py"),
        os.path.join(output_dir, "simenv_consts.py"),
    )

    hidden_sizes = _network_hidden_sizes(network)
    git_info = _git_info(project_root)
    diff_text = git_info.pop("_diff_text", "")
    metadata = {
        "algo": "ddqn",
        "env_kind": "sim",
        "device": device,
        "network": {
            "class": network.__class__.__name__,
            "input_dim": int(network.n_inputs),
            "output_dim": int(network.n_outputs),
            "hidden_sizes": hidden_sizes,
            "params": int(sum(p.numel() for p in network.parameters())),
        },
        "training": {
            "max_episodes": int(max_episodes),
            "buffer_size": int(buffer_size),
            "burn_in": int(burn_in),
            "batch_size": int(batch_size),
            "gamma": float(gamma),
            "lr": float(lr),
            "max_grad_norm": float(max_grad_norm),
            "network_update_freq": int(network_update_freq),
            "network_sync_freq": int(network_sync_freq),
        },
        "eval": {
            "enabled": bool(eval_config.enabled),
            "freq_episodes": int(eval_config.freq_episodes),
            "episodes": int(eval_config.episodes),
            "deterministic": bool(eval_config.deterministic),
            "save_episode_details": bool(eval_config.save_episode_details),
        },
        "env": {
            "rows": int(env.rows),
            "cols": int(env.cols),
            "state_dim": int(env.state_dim),
            "action_dim": int(env.action_space.n),
            "card_order": list(env.card_plant_ids),
            "plant_names": list(env._plant_names),
        },
        "curriculum": _jsonable(CURRICULUM),
        "current_stage": getattr(curriculum, "current_stage_name", None),
        "git": git_info,
        "snapshots": {
            "config": "simenv_config.py",
            "consts": "simenv_consts.py",
            "diff": "git_diff.patch" if git_info.get("dirty") else None,
        },
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    if diff_text:
        with open(os.path.join(output_dir, "git_diff.patch"), "w", encoding="utf-8") as file:
            file.write(diff_text)


def _copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copyfile(src, dst)


def _network_hidden_sizes(network):
    linears = [
        module for module in network.network
        if isinstance(module, torch.nn.Linear)
    ]
    return [int(layer.out_features) for layer in linears[:-1]]


def _git_info(project_root):
    commit = _run_git(project_root, "rev-parse", "HEAD").strip()
    status = _run_git(project_root, "status", "--porcelain")
    diff_text = ""
    if status.strip():
        diff_text = _run_git(project_root, "diff", "HEAD", "--")
    return {
        "commit": commit or None,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "_diff_text": diff_text,
    }


def _run_git(project_root, *args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _burn_in_stage(env, buffer, *, step_count, stage_name, stop_requested):
    print(f"Burn-in ({buffer.burn_in} steps, stage={stage_name})...")
    state = transform_observation(env.reset())
    mask = np.array(env.mask_available_actions())  # initial mask
    while buffer.burn_in_capacity() < 1 and not stop_requested():
        if np.random.random() < 0.5:
            action = env.wait_action
        else:
            action = np.random.choice(np.arange(env.action_space.n)[mask])
        next_state, reward, done, info = env.step(action)
        next_state = transform_observation(next_state)
        next_mask = info.get("mask", env.mask_available_actions())
        buffer.append(state, action, reward, done, next_state, mask, next_mask)
        state = next_state  # transform_observation already copied
        mask = next_mask      # carry forward to next iteration
        if done:
            state = transform_observation(env.reset())
            mask = np.array(env.mask_available_actions())  # fresh mask after reset
        step_count += 1
    print(f"Burn-in done. Buffer: {len(buffer)}  "
          f"(steps so far: {step_count})")
    return state, mask, step_count, bool(stop_requested())


def _stage_burn_in(stage, fallback):
    value = getattr(stage, "burn_in", None)
    if value is None:
        return max(1, int(fallback))
    return max(1, int(value))


def _should_run_eval(ep, curriculum, eval_config, eval_scheduler):
    if not getattr(curriculum, "enabled", False):
        return eval_scheduler.should_run(ep)
    if not eval_config.enabled:
        return False
    freq = int(eval_config.freq_episodes)
    return (
        freq > 0
        and curriculum.stage_episode > 0
        and curriculum.stage_episode % freq == 0
    )


def _append_curriculum_event(output_dir, episode, old_stage, new_stage, eval_result):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sim_curriculum_events.csv")
    has_header = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "from_stage",
                "to_stage",
                "eval_id",
                "reward_mean",
                "win_rate",
            ],
        )
        if not has_header:
            writer.writeheader()
        writer.writerow({
            "episode": int(episode),
            "from_stage": old_stage,
            "to_stage": new_stage,
            "eval_id": eval_result.eval_id,
            "reward_mean": float(eval_result.reward_mean),
            "win_rate": float(eval_result.win_rate),
        })


def _should_save_best(curriculum):
    return (
        not getattr(curriculum, "enabled", False)
        or bool(getattr(curriculum, "is_final_stage", False))
    )


def _save_training_checkpoint(
    save_path,
    network,
    training_rewards,
    training_iterations,
    training_loss,
    training_advantage=None,
    training_entropy=None,
    training_mean_q=None,
    training_max_q=None,
    training_td_error=None,
    training_grad_norm=None,
    training_q_wait=None,
    training_delta_mean=None,
    training_delta_max=None,
    plot_callback=None,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(network.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    _save_training_artifacts(
        save_path,
        training_rewards,
        training_iterations,
        training_loss,
        training_advantage=training_advantage,
        training_entropy=training_entropy,
        training_mean_q=training_mean_q,
        training_max_q=training_max_q,
        training_td_error=training_td_error,
        training_grad_norm=training_grad_norm,
        training_q_wait=training_q_wait,
        training_delta_mean=training_delta_mean,
        training_delta_max=training_delta_max,
        plot_callback=plot_callback,
    )


def _save_training_artifacts(
    save_path,
    training_rewards,
    training_iterations,
    training_loss,
    training_advantage=None,
    training_entropy=None,
    training_mean_q=None,
    training_max_q=None,
    training_td_error=None,
    training_grad_norm=None,
    training_q_wait=None,
    training_delta_mean=None,
    training_delta_max=None,
    plot_callback=None,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    rewards = np.array(training_rewards)
    iterations = np.array(training_iterations)
    loss = np.array(training_loss)
    np.save(save_path.replace(".pt", "_rewards.npy"), rewards)
    np.save(save_path.replace(".pt", "_iterations.npy"), iterations)
    np.save(save_path.replace(".pt", "_loss.npy"), loss)
    for name, data in [
        ("advantage", training_advantage),
        ("entropy", training_entropy),
        ("mean_q", training_mean_q),
        ("max_q", training_max_q),
        ("td_error", training_td_error),
        ("grad_norm", training_grad_norm),
        ("q_wait", training_q_wait),
        ("delta_mean", training_delta_mean),
        ("delta_max", training_delta_max),
    ]:
        if data is not None and len(data) > 0:
            np.save(
                save_path.replace(".pt", f"_{name}.npy"),
                np.array(data),
            )
    if plot_callback is not None:
        plot_callback(
            save_path,
            rewards, iterations, loss,
            advantage=np.array(training_advantage) if training_advantage else None,
            entropy=np.array(training_entropy) if training_entropy else None,
            mean_q=np.array(training_mean_q) if training_mean_q else None,
            max_q=np.array(training_max_q) if training_max_q else None,
            td_error=np.array(training_td_error) if training_td_error else None,
            grad_norm=np.array(training_grad_norm) if training_grad_norm else None,
            q_wait=np.array(training_q_wait) if training_q_wait else None,
            delta_mean=np.array(training_delta_mean) if training_delta_mean else None,
            delta_max=np.array(training_delta_max) if training_delta_max else None,
        )


def _default_save_path(algo, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("saved", algo, timestamp, filename)


def _run_and_save_eval(
    network,
    eval_writer,
    eval_config,
    best_eval_checkpoint=None,
    episode=None,
    step=None,
    stage=None,
    force=False,
):
    if not force and not eval_config.enabled:
        return None
    result = _evaluate(network, n_episodes=eval_config.episodes,
                       episode=episode, step=step, stage=stage)
    eval_writer.write(result)
    if best_eval_checkpoint is not None:
        saved_path = best_eval_checkpoint.maybe_save(
            result,
            lambda path: torch.save(network.state_dict(), path),
        )
        if saved_path is not None:
            print(f"[Eval] New best model saved to {saved_path}")
    return result


def _evaluate(network, n_episodes=20, episode=None, step=None, stage=None):
    """Run N independent sim episodes with greedy policy."""
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  Evaluation ({n_episodes} episodes, greedy policy)")
    print(f"{sep}")

    eval_id = new_eval_id("sim_ddqn")
    start_time = time_eval_run()
    eval_env = SimPVZEnv(stage=stage)
    details = []
    max_frames = (
        int(stage.timeout_frames)
        if stage is not None
        else config.MAX_FRAMES
    )
    stage_name = stage.stage_name if stage is not None else "sim"
    target_flag_waves = (
        int(stage.target_flag_waves)
        if stage is not None
        else None
    )

    for index in range(n_episodes):
        state = transform_observation(eval_env.reset())
        done = False
        total_reward = 0.0
        steps = 0
        info = {}
        while not done:
            mask = eval_env.mask_available_actions()
            with torch.no_grad():
                qvals = network.get_qvals(state)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
            qvals = qvals.clone()
            qvals[~mask_t] = qvals.min()
            action = torch.max(qvals, dim=-1)[1].item()
            state, reward, done, info = eval_env.step(action)
            state = transform_observation(state)
            total_reward += reward
            steps += 1

        survival = float(info.get("steps", min(max_frames, eval_env._scene._chrono)))
        details.append(
            EpisodeEvalResult(
                eval_id=eval_id,
                episode_index=index + 1,
                reward=float(total_reward),
                survival=survival,
                win=bool(info.get("win", survival >= max_frames)),
                game_ended=True,
                completed_sublevels=info.get("completed_sublevels"),
                actions=steps,
                extra={
                    "max_frames": max_frames,
                    "fps": config.FPS,
                    "stage_name": info.get("stage_name", stage_name),
                    "target_frames": info.get("target_frames"),
                    "target_flag_waves": info.get("target_flag_waves"),
                    "timeout_frames": info.get("timeout_frames"),
                    "current_wave_index": info.get("current_wave_index"),
                    "is_flag_wave": info.get("is_flag_wave"),
                    "diagnostics": info.get("diagnostics", {}),
                },
            )
        )

    fps = config.FPS
    result = summarize_eval_results(
        eval_id=eval_id,
        algo="ddqn",
        env_kind="sim",
        episode=episode,
        step=step,
        stage_name=stage_name,
        win_condition="stage_objective" if stage is not None else "max_frames",
        target_sublevels=target_flag_waves,
        details=details,
        duration_sec=elapsed_since(start_time),
        extra={
            "max_frames": max_frames,
            "fps": fps,
            "stage_name": stage_name,
            "target_frames": getattr(stage, "target_frames", None),
            "target_flag_waves": target_flag_waves,
            "timeout_frames": getattr(stage, "timeout_frames", None),
            "diagnostics": summarize_diagnostics(details),
            "plant_stats": summarize_plant_stats(details),
        },
    )
    diagnostics = result.extra.get("diagnostics", {})
    action_stats = diagnostics.get("action_stats", {})
    plant_success = action_stats.get("plant_success_by_type", {})
    print(f"  {'Reward:':20s} mean={result.reward_mean:8.2f}  std={result.reward_std:8.2f}  "
          f"min={result.reward_min:8.2f}  max={result.reward_max:8.2f}")
    print(f"  {'Survival (frames):':20s} mean={result.survival_mean:8.2f}  std={result.survival_std:8.2f}  "
          f"min={result.survival_min:8.0f}  max={result.survival_max:8.0f}")
    print(f"  {'Survival (sec):':20s} mean={result.survival_mean / fps:8.2f}  std={result.survival_std / fps:8.2f}  "
          f"min={result.survival_min / fps:8.2f}  max={result.survival_max / fps:8.2f}")
    print(f"  {'Actions taken:':20s} mean={result.actions_mean or 0:8.2f}")
    print(f"  {'Action stats:':20s} wait={int(action_stats.get('wait', 0))}  "
          f"plant={int(action_stats.get('plant', 0))}  "
          f"invalid={int(action_stats.get('invalid', 0))}")
    if plant_success:
        plant_text = ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                plant_success.items(),
                key=lambda item: (-int(item[1]), item[0]),
            )
        )
        print(f"  {'Plant success:':20s} {plant_text}")
    print(f"  {'Stage wins:':20s} {result.win_count}/{n_episodes} ({100 * result.win_rate:.1f}%)")
    print(f"{sep}\n")
    return result


def _visualize_episode(env, network):
    """Play one episode with render collection and show replay."""
    from simenv.render import replay_episode
    env.enable_render_collection()
    state = transform_observation(env.reset())
    done = False
    total_reward = 0.0
    while not done:
        mask = env.mask_available_actions()
        with torch.no_grad():
            qvals = network.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        action = torch.max(qvals, dim=-1)[1].item()
        state, reward, done, _ = env.step(action)
        state = transform_observation(state)
        total_reward += reward
    env.disable_render_collection()
    print(f"\nReplay: {len(env.render_data)} frames, reward={total_reward:.0f}")
    replay_episode(env.render_data, fps=15,
                   title=f"SimPVZ Trained Agent - Reward: {total_reward:.0f}")
