"""Training loop for the simulation environment."""

import gc
import os
import signal
from datetime import datetime
import numpy as np
import torch
from copy import deepcopy

from simenv import SimPVZEnv
from simenv.pvz_sim import config
from simenv.model import (
    ReplayBuffer, DDQNNetwork, transform_observation, calculate_loss,
)
from training.evaluation import (
    BestEvaluationCheckpoint,
    EpisodeEvalResult,
    EvaluationConfig,
    EvaluationScheduler,
    EvaluationWriter,
    elapsed_since,
    new_eval_id,
    summarize_eval_results,
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
    max_episodes=100000,
    buffer_size=100000,
    burn_in=10000,
    batch_size=200,
    gamma=0.99,
    lr=1e-3,
    network_update_freq=32,
    network_sync_freq=2000,
    save_path=None,
    eval_episodes=20,
    eval_freq_episodes=2500,
    visualize=False,
    plot_freq=100,
    plot_callback=None,
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
    network = DDQNNetwork(env, learning_rate=lr, device=device)
    target_network = deepcopy(network)
    buffer = ReplayBuffer(memory_size=buffer_size, burn_in=burn_in)
    threshold = EpsilonSchedule(
        seq_length=max_episodes,
        start_epsilon=1.0,
        end_epsilon=0.05,
    )

    _print_config(
        device=device,
        network_type="ddqn",
        network_params=sum(p.numel() for p in network.parameters()),
        max_episodes=max_episodes,
        buffer_size=buffer_size,
        burn_in=burn_in,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        network_update_freq=network_update_freq,
        network_sync_freq=network_sync_freq,
        eval_episodes=eval_episodes,
        eval_freq_episodes=eval_freq_episodes,
        epsilon_decay=f"{threshold.start_epsilon} -> {threshold.end_epsilon} (exponential)",
        env_config={
            "rows": config.N_LANES,
            "cols": config.LANE_LENGTH,
            "fps": config.FPS,
            "max_frames": config.MAX_FRAMES,
            "plants": list(env.plant_deck.keys()),
        },
    )

    training_rewards = []
    training_loss = []
    training_iterations = []
    update_loss = []
    step_count = 0
    window = 100
    last_eval_episode = None
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
                plot_callback=plot_callback,
            )

    signal.signal(signal.SIGINT, _handle_interrupt)

    print(f"Burn-in ({burn_in} steps)...")
    s_0 = transform_observation(env.reset())
    while buffer.burn_in_capacity() < 1 and not stop_requested:
        mask = np.array(env.mask_available_actions())
        if np.random.random() < 0.5:
            action = env.wait_action
        else:
            action = np.random.choice(np.arange(env.action_space.n)[mask])
        s_1, reward, done, _ = env.step(action)
        s_1 = transform_observation(s_1)
        next_mask = np.array(env.mask_available_actions())
        buffer.append(s_0, action, reward, done, s_1, mask, next_mask)
        s_0 = s_1.copy()
        if done:
            s_0 = transform_observation(env.reset())
        step_count += 1
    if stop_requested:
        print(f"Training stopped during burn-in at {step_count} steps.")
        signal.signal(signal.SIGINT, previous_sigint_handler)
        return
    print(f"Burn-in done. Buffer: {len(buffer.replay_memory)}  "
          f"(steps so far: {step_count})")

    ep = 0
    s_0 = transform_observation(env.reset())
    print(f"Training {max_episodes} episodes...")

    while ep < max_episodes and not stop_requested:
        rewards = 0
        done = False
        while not done and not stop_requested:
            epsilon = threshold.epsilon(ep)
            mask = np.array(env.mask_available_actions())
            action = network.decide_action(s_0, mask, epsilon=epsilon)
            s_1, r, done, _ = env.step(action)
            s_1 = transform_observation(s_1)
            next_mask = np.array(env.mask_available_actions())
            rewards += r
            buffer.append(s_0, action, r, done, s_1, mask, next_mask)
            s_0 = s_1.copy()
            step_count += 1

            if step_count % network_update_freq == 0:
                network.optimizer.zero_grad(set_to_none=True)
                batch = buffer.sample_batch(batch_size=batch_size)
                loss = calculate_loss(network, target_network, batch, gamma)
                loss.backward()
                network.optimizer.step()
                update_loss.append(loss.detach().item())

            if step_count % network_sync_freq == 0:
                target_network.load_state_dict(network.state_dict())

            if done:
                ep += 1
                training_rewards.append(rewards)
                training_iterations.append(min(config.MAX_FRAMES, env._scene._chrono))
                if update_loss:
                    training_loss.append(np.mean(update_loss))
                update_loss = []

                if ep % 100 == 0:
                    gc.collect()
                    mean_r = np.mean(training_rewards[-window:])
                    mean_i = np.mean(training_iterations[-window:])
                    mean_l = np.mean(training_loss[-window:]) if training_loss else 0
                    print(f"Episode {ep:5d}/{max_episodes}  "
                          f"Steps {step_count:7d}  "
                          f"Mean R {mean_r:8.2f}  Mean I {mean_i:.2f}  Mean L {mean_l:.2f}")

                if eval_scheduler.should_run(ep):
                    _run_and_save_eval(
                        network,
                        eval_writer,
                        eval_config,
                        best_eval_checkpoint=best_eval_checkpoint,
                        episode=ep,
                        step=step_count,
                    )
                    last_eval_episode = ep

                if plot_freq and ep % plot_freq == 0:
                    _save_training_artifacts(
                        save_path,
                        training_rewards,
                        training_iterations,
                        training_loss,
                        plot_callback=plot_callback,
                    )

                if ep >= max_episodes:
                    print(f"\nEpisode limit reached ({max_episodes} episodes, {step_count} steps).")
                    break

                s_0 = transform_observation(env.reset())

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
        plot_callback=plot_callback,
    )
    print("Training complete.")

    if eval_episodes > 0 and ep != last_eval_episode:
        _run_and_save_eval(
            network,
            eval_writer,
            eval_config,
            best_eval_checkpoint=best_eval_checkpoint,
            episode=ep,
            step=step_count,
            force=True,
        )
        _save_training_artifacts(
            save_path,
            training_rewards,
            training_iterations,
            training_loss,
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
    print(f"  {'Network update freq:':24s} {cfg['network_update_freq']} steps")
    print(f"  {'Network sync freq:':24s} {cfg['network_sync_freq']} steps")
    print(f"  {'Epsilon decay:':24s} {cfg['epsilon_decay']}")
    print(f"  {'Eval episodes:':24s} {cfg['eval_episodes']}")
    print(f"  {'Eval frequency:':24s} {cfg['eval_freq_episodes']} episodes")
    print(f"{sep}")
    ec = cfg["env_config"]
    print(f"  Environment")
    print(f"{sep}")
    print(f"  {'Grid:':24s} {ec['rows']}x{ec['cols']} (rows x cols)")
    print(f"  {'FPS:':24s} {ec['fps']}")
    print(f"  {'Max frames:':24s} {ec['max_frames']} ({ec['max_frames'] // ec['fps']}s game time)")
    print(f"  {'Plant deck:':24s} {', '.join(ec['plants'])}")
    print(f"{sep}\n")


def _save_training_checkpoint(
    save_path,
    network,
    training_rewards,
    training_iterations,
    training_loss,
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
        plot_callback=plot_callback,
    )


def _save_training_artifacts(
    save_path,
    training_rewards,
    training_iterations,
    training_loss,
    plot_callback=None,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    rewards = np.array(training_rewards)
    iterations = np.array(training_iterations)
    loss = np.array(training_loss)
    np.save(save_path.replace(".pt", "_rewards.npy"), rewards)
    np.save(save_path.replace(".pt", "_iterations.npy"), iterations)
    np.save(save_path.replace(".pt", "_loss.npy"), loss)
    if plot_callback is not None:
        plot_callback(save_path, rewards, iterations, loss)


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
    force=False,
):
    if not force and not eval_config.enabled:
        return None
    result = _evaluate(network, n_episodes=eval_config.episodes,
                       episode=episode, step=step)
    eval_writer.write(result)
    if best_eval_checkpoint is not None:
        saved_path = best_eval_checkpoint.maybe_save(
            result,
            lambda path: torch.save(network.state_dict(), path),
        )
        if saved_path is not None:
            print(f"[Eval] New best model saved to {saved_path}")
    return result


def _evaluate(network, n_episodes=20, episode=None, step=None):
    """Run N independent sim episodes with greedy policy."""
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  Evaluation ({n_episodes} episodes, greedy policy)")
    print(f"{sep}")

    eval_id = new_eval_id("sim_ddqn")
    start_time = time_eval_run()
    eval_env = SimPVZEnv()
    details = []
    max_frames = config.MAX_FRAMES

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

        survival = min(max_frames, eval_env._scene._chrono)
        details.append(
            EpisodeEvalResult(
                eval_id=eval_id,
                episode_index=index + 1,
                reward=float(total_reward),
                survival=float(survival),
                win=survival >= max_frames,
                game_ended=True,
                completed_sublevels=info.get("completed_sublevels"),
                actions=steps,
                extra={
                    "max_frames": max_frames,
                    "fps": config.FPS,
                    "current_wave_index": info.get("current_wave_index"),
                    "is_flag_wave": info.get("is_flag_wave"),
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
        stage_name="sim",
        win_condition="max_frames",
        target_sublevels=None,
        details=details,
        duration_sec=elapsed_since(start_time),
        extra={
            "max_frames": max_frames,
            "fps": fps,
        },
    )
    print(f"  {'Reward:':20s} mean={result.reward_mean:8.2f}  std={result.reward_std:8.2f}  "
          f"min={result.reward_min:8.2f}  max={result.reward_max:8.2f}")
    print(f"  {'Survival (frames):':20s} mean={result.survival_mean:8.2f}  std={result.survival_std:8.2f}  "
          f"min={result.survival_min:8.0f}  max={result.survival_max:8.0f}")
    print(f"  {'Survival (sec):':20s} mean={result.survival_mean / fps:8.2f}  std={result.survival_std / fps:8.2f}  "
          f"min={result.survival_min / fps:8.2f}  max={result.survival_max / fps:8.2f}")
    print(f"  {'Actions taken:':20s} mean={result.actions_mean or 0:8.2f}")
    print(f"  {'Full survival:':20s} {result.win_count}/{n_episodes} ({100 * result.win_rate:.1f}%)")
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
