"""Training loop for the simulation environment."""

import gc
import os
import numpy as np
import torch
from copy import deepcopy

from simenv import SimPVZEnv
from simenv.pvz_sim import config
from simenv.model import (
    ReplayBuffer, SimQNetwork, SimDeepMLPNetwork, SimCNNQNetwork,
    transform_observation, calculate_loss,
)
from models.ddqn.threshold import Threshold


def train_sim(
    max_episodes=100000,
    buffer_size=100000,
    burn_in=10000,
    batch_size=200,
    gamma=0.99,
    lr=1e-3,
    network_update_freq=32,
    network_sync_freq=2000,
    eval_freq=5000,
    eval_n_iter=200,
    save_path="saved/sim_ddqn.pt",
    network_type="cnn",
    eval_episodes=100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimPVZEnv()
    if network_type == "cnn":
        network = SimCNNQNetwork(env, learning_rate=lr, device=device)
    elif network_type == "deepmlp":
        network = SimDeepMLPNetwork(env, learning_rate=lr, device=device)
    else:
        network = SimQNetwork(env, learning_rate=lr, device=device,
                              use_zombienet=False, use_gridnet=False)
    target_network = deepcopy(network)
    buffer = ReplayBuffer(memory_size=buffer_size, burn_in=burn_in)
    threshold = Threshold(
        seq_length=max_episodes,
        start_epsilon=1.0,
        interpolation="exponential",
        end_epsilon=0.05,
    )

    # ── Print training configuration ──
    _print_config(
        device=device,
        network_type=network_type,
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
        epsilon_decay=f"{threshold.start_epsilon} -> {threshold.end_epsilon} ({threshold.interpolation})",
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

    # ── Burn-in ──
    print(f"Burn-in ({burn_in} steps)...")
    s_0 = transform_observation(env.reset())
    while buffer.burn_in_capacity() < 1:
        mask = np.array(env.mask_available_actions())
        if np.random.random() < 0.5:
            action = 0
        else:
            action = np.random.choice(np.arange(env.action_space.n)[mask])
        s_1, reward, done, _ = env.step(action)
        s_1 = transform_observation(s_1)
        buffer.append(s_0, action, reward, done, s_1)
        s_0 = s_1.copy()
        if done:
            s_0 = transform_observation(env.reset())
        step_count += 1
    print(f"Burn-in done. Buffer: {len(buffer.replay_memory)}")

    # ── Training loop ──
    ep = 0
    s_0 = transform_observation(env.reset())
    print(f"Training {max_episodes} episodes...")

    while ep < max_episodes:
        rewards = 0
        done = False
        while not done:
            epsilon = threshold.epsilon(ep)
            mask = np.array(env.mask_available_actions())
            action = network.decide_action(s_0, mask, epsilon=epsilon)
            s_1, r, done, _ = env.step(action)
            s_1 = transform_observation(s_1)
            rewards += r
            buffer.append(s_0, action, r, done, s_1)
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
                    gc.collect()  # force GC periodically to avoid memory creep
                    mean_r = np.mean(training_rewards[-window:])
                    mean_i = np.mean(training_iterations[-window:])
                    mean_l = np.mean(training_loss[-window:]) if training_loss else 0
                    print(f"Episode {ep:5d} Mean Rewards {mean_r:8.2f}\t\t "
                          f"Mean Iterations {mean_i:.2f}\t Mean Loss {mean_l:.2f}")

                if ep >= max_episodes:
                    print("\nEpisode limit reached.")
                    break

                s_0 = transform_observation(env.reset())

    # ── Save ──
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(network.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    np.save(save_path.replace(".pt", "_rewards.npy"), np.array(training_rewards))
    np.save(save_path.replace(".pt", "_iterations.npy"), np.array(training_iterations))
    np.save(save_path.replace(".pt", "_loss.npy"), np.array(training_loss))
    print("Training complete.")

    # ── Evaluation ──
    _evaluate(env, network, n_episodes=eval_episodes)

    _visualize_episode(env, network)


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
    print(f"{sep}")
    ec = cfg["env_config"]
    print(f"  Environment")
    print(f"{sep}")
    print(f"  {'Grid:':24s} {ec['rows']}x{ec['cols']} (rows x cols)")
    print(f"  {'FPS:':24s} {ec['fps']}")
    print(f"  {'Max frames:':24s} {ec['max_frames']} ({ec['max_frames'] // ec['fps']}s game time)")
    print(f"  {'Plant deck:':24s} {', '.join(ec['plants'])}")
    print(f"{sep}\n")


def _evaluate(env, network, n_episodes=100):
    """Run N episodes with greedy policy (epsilon=0) and report statistics."""
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  Evaluation ({n_episodes} episodes, greedy policy)")
    print(f"{sep}")

    rewards = []
    survivals = []
    actions_taken = []
    max_frames = config.MAX_FRAMES

    for _ in range(n_episodes):
        state = transform_observation(env.reset())
        done = False
        total_reward = 0.0
        steps = 0
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
            steps += 1

        rewards.append(total_reward)
        survivals.append(min(max_frames, env._scene._chrono))
        actions_taken.append(steps)

    rewards = np.array(rewards)
    survivals = np.array(survivals)
    actions = np.array(actions_taken)

    # Convert frames to seconds for readability
    fps = config.FPS
    print(f"  {'Reward:':20s} mean={rewards.mean():8.2f}  std={rewards.std():8.2f}  "
          f"min={rewards.min():8.2f}  max={rewards.max():8.2f}")
    print(f"  {'Survival (frames):':20s} mean={survivals.mean():8.2f}  std={survivals.std():8.2f}  "
          f"min={survivals.min():8.0f}  max={survivals.max():8.0f}")
    print(f"  {'Survival (sec):':20s} mean={survivals.mean() / fps:8.2f}  std={survivals.std() / fps:8.2f}  "
          f"min={survivals.min() / fps:8.2f}  max={survivals.max() / fps:8.2f}")
    print(f"  {'Actions taken:':20s} mean={actions.mean():8.2f}  std={actions.std():8.2f}  "
          f"min={actions.min():8.0f}  max={actions.max():8.0f}")

    # Survival histogram (coarse bins)
    survived_full = (survivals >= max_frames).sum()
    print(f"  {'Full survival:':20s} {survived_full}/{n_episodes} ({100 * survived_full / n_episodes:.1f}%)")
    print(f"{sep}\n")


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
