"""Training loop for the simulation environment."""

import os
import numpy as np
import torch
from copy import deepcopy

from simenv import SimPVZEnv
from simenv.pvz_sim import config
from simenv.model import (
    ReplayBuffer, SimQNetwork, transform_observation, calculate_loss,
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
):
    env = SimPVZEnv()
    network = SimQNetwork(env, learning_rate=lr, device="cpu",
                          use_zombienet=False, use_gridnet=False)
    target_network = deepcopy(network)
    buffer = ReplayBuffer(memory_size=buffer_size, burn_in=burn_in)
    threshold = Threshold(
        seq_length=max_episodes,
        start_epsilon=1.0,
        interpolation="exponential",
        end_epsilon=0.05,
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

    _visualize_episode(env, network)


def _visualize_episode(env, network):
    """Play one episode with render collection and show replay."""
    from simenv.render import replay_episode
    env.enable_render_collection()
    state = transform_observation(env.reset())
    done = False
    total_reward = 0.0
    while not done:
        mask = env.mask_available_actions()
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
