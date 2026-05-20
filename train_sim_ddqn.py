"""
DDQN training with the simplified PVZ simulation environment.

Usage: python train_sim_ddqn.py
"""
import sys
import os
import numpy as np
import torch
from collections import namedtuple, deque
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simenv import SimPVZEnv
from models.ddqn.ddqn import QNetwork, experienceReplayBuffer
from models.threshold import Threshold


def evaluate(env, network, n_iter=100, verbose=True):
    """Evaluate the agent for n_iter episodes, return avg score and iterations."""
    sum_score = 0.0
    sum_iter = 0
    for ep in range(n_iter):
        state = env.reset()
        done = False
        ep_score = 0.0
        ep_steps = 0
        while not done:
            mask = env.mask_available_actions()
            qvals = network.get_qvals(state)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
            qvals = qvals.clone()
            qvals[~mask_t] = qvals.min()
            action = torch.max(qvals, dim=-1)[1].item()
            state, reward, done, _ = env.step(action)
            ep_score += reward
            ep_steps += 1
        sum_score += ep_score
        sum_iter += ep_steps
        if verbose and ep % 20 == 0:
            print(f"\r  eval {ep}/{n_iter}", end="")
    return sum_score / n_iter, sum_iter / n_iter


def train_sim_ddqn(
    max_episodes=100000,
    buffer_size=100000,
    burn_in=1000,
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
    network = QNetwork(env, learning_rate=lr, device="cpu")
    target_network = deepcopy(network)
    buffer = experienceReplayBuffer(memory_size=buffer_size, burn_in=burn_in)
    threshold = Threshold(
        seq_length=100000,
        start_epsilon=1.0,
        interpolation="exponential",
        end_epsilon=0.05,
    )

    # Metrics
    training_rewards = []
    training_loss = []
    training_iterations = []
    real_rewards = []
    real_iterations = []
    update_loss = []
    step_count = 0

    # Burn-in
    print(f"Burn-in ({burn_in} steps)...")
    state = env.reset()
    while buffer.burn_in_capacity() < 1:
        mask = env.mask_available_actions()
        if np.random.random() < 0.5:
            action = 0
        else:
            valid = np.arange(env.action_space.n)[mask]
            action = np.random.choice(valid)
        next_state, reward, done, _ = env.step(action)
        buffer.append(state, action, reward, done, next_state, mask, env.mask_available_actions())
        state = next_state.copy()
        if done:
            state = env.reset()
        step_count += 1
    print(f"Burn-in done. Buffer: {len(buffer.replay_memory)}")

    # Training loop
    ep = 0
    state = env.reset()
    print(f"Training {max_episodes} episodes...")

    while ep < max_episodes:
        ep_reward = 0.0
        done = False
        while not done:
            epsilon = threshold.epsilon(ep)
            mask = env.mask_available_actions()
            action = network.decide_action(state, mask, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            next_mask = env.mask_available_actions()
            buffer.append(state, action, reward, done, next_state, mask, next_mask)
            step_count += 1
            ep_reward += reward
            state = next_state.copy()
            mask = next_mask

            # Update network
            if step_count % network_update_freq == 0:
                network.optimizer.zero_grad(set_to_none=True)
                batch = buffer.sample_batch(batch_size=batch_size)
                loss = _calculate_loss(
                    network, target_network, batch, gamma, env)
                loss.backward()
                network.optimizer.step()
                update_loss.append(loss.detach().item())

            # Sync target network
            if step_count % network_sync_freq == 0:
                target_network.load_state_dict(network.state_dict())

        ep += 1
        training_rewards.append(ep_reward)
        training_iterations.append(env.steps)
        if update_loss:
            training_loss.append(np.mean(update_loss))
        update_loss = []

        # Print progress
        if ep % 100 == 0:
            mean_r = np.mean(training_rewards[-100:])
            mean_i = np.mean(training_iterations[-100:])
            avg_loss = np.mean(training_loss[-100:]) if training_loss else 0
            print(f"Ep {ep:5d} | reward={mean_r:8.1f} | iter={mean_i:6.1f} | "
                  f"loss={avg_loss:.4f} | epsilon={epsilon:.3f}")

        # Evaluate
        if ep % eval_freq == 0:
            avg_score, avg_iter = evaluate(env, network, n_iter=eval_n_iter, verbose=False)
            real_rewards.append(avg_score)
            real_iterations.append(avg_iter)
            print(f"  >>> Eval @ ep {ep}: score={avg_score:.1f}, iter={avg_iter:.1f}")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(network.state_dict(), save_path)
    print(f"Saved model to {save_path}")

    # Save metrics
    np.save(save_path.replace(".pt", "_rewards.npy"), np.array(training_rewards))
    np.save(save_path.replace(".pt", "_iterations.npy"), np.array(training_iterations))
    np.save(save_path.replace(".pt", "_eval_rewards.npy"), np.array(real_rewards))
    print("Training complete.")

    # --- Visualize one episode with the trained agent ---
    _visualize_episode(env, network)


def _calculate_loss(network, target_network, batch, gamma, env):
    states, actions, rewards, dones, next_states, masks, next_masks = [
        item for item in batch
    ]

    rewards_t = torch.FloatTensor(rewards).reshape(-1, 1)
    actions_t = torch.LongTensor(np.array(actions)).reshape(-1, 1)
    dones_t = torch.as_tensor(dones, dtype=torch.bool)

    qvals = torch.gather(network.get_qvals(states), 1, actions_t)

    with torch.no_grad():
        next_masks_t = torch.as_tensor(
            np.array(next_masks), dtype=torch.bool)
        qvals_next_pred = network.get_qvals(next_states)
        qvals_next_pred = qvals_next_pred.clone()
        qvals_next_pred[~next_masks_t] = qvals_next_pred.min()
        next_actions = torch.max(qvals_next_pred, dim=-1)[1]
        next_actions_t = next_actions.reshape(-1, 1)
        target_qvals = target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t)
    qvals_next[dones_t] = 0
    expected_qvals = gamma * qvals_next + rewards_t
    return torch.nn.MSELoss()(qvals, expected_qvals)


def _visualize_episode(env, network):
    """Play one episode with render collection and show replay."""
    from simenv.render import replay_episode

    env.enable_render_collection()
    state = env.reset()
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
        total_reward += reward
    env.disable_render_collection()

    print(f"\nReplay episode: {len(env.render_data)} frames, "
          f"reward={total_reward:.0f}")
    replay_episode(env.render_data, fps=15,
                   title=f"SimPVZ Trained Agent — Reward: {total_reward:.0f}")


if __name__ == "__main__":
    train_sim_ddqn(
        max_episodes=50000,
        buffer_size=100000,
        burn_in=1000,
        batch_size=200,
    )
