"""
DDQN training with the simplified PVZ simulation environment.

Architecture, observation, reward, and training logic are byte-identical
to the original pvz_rl project.

Usage: python train_sim_ddqn.py
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import namedtuple, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simenv import SimPVZEnv
from simenv.pvz_sim import config
from models.threshold import Threshold


# ── Hyperparameters (same as pvz_rl) ──────────────────────────────────
HP_NORM = 1
SUN_NORM = 200


# ── 5-field replay buffer (same as pvz_rl) ────────────────────────────
class ReplayBuffer:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple(
            "Buffer", field_names=["state", "action", "reward", "done", "next_state"])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


# ── Q-Network (identical to pvz_rl QNetwork + ZombieNet + GridNet) ────
class ZombieNet(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(config.LANE_LENGTH, output_size)

    def forward(self, x):
        return self.fc1(x)


class SimQNetwork(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 use_zombienet=True, use_gridnet=True):
        super().__init__()
        self.device = device
        self._grid_size = config.N_LANES * config.LANE_LENGTH  # 45
        self.n_outputs = env.action_space.n                     # 181
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        # ── Feature extractors ──
        self.use_zombienet = use_zombienet
        if use_zombienet:
            self.zombienet_output_size = 1
            self.zombienet = ZombieNet(output_size=self.zombienet_output_size)

        self.use_gridnet = use_gridnet
        if use_gridnet:
            self.gridnet_output_size = 4
            self.gridnet = nn.Linear(self._grid_size, self.gridnet_output_size)

        # ── Compute combined input size (same formula as pvz_rl) ──
        n_inputs = self._grid_size + config.N_LANES + len(env.plant_deck) + 1  # 55
        if use_zombienet:
            n_inputs += (self.zombienet_output_size - 1) * config.N_LANES       # +0
        if use_gridnet:
            n_inputs += self.gridnet_output_size - self._grid_size              # 4-45 = -41
        self.n_inputs = n_inputs  # = 14

        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 50, bias=True),
            nn.LeakyReLU(),
            nn.Linear(50, self.n_outputs, bias=True),
        )

        if self.device == "cuda":
            self.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

    # ── Forward pass (identical to pvz_rl QNetwork.get_qvals) ─────────
    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.FloatTensor(state).to(device=self.device)
            zombie_grid = state_t[:, self._grid_size:(2 * self._grid_size)].reshape(-1, config.LANE_LENGTH)
            plant_grid = state_t[:, :self._grid_size]
            if self.use_zombienet:
                zombie_grid = self.zombienet(zombie_grid).view(-1, self.zombienet_output_size * config.N_LANES)
            else:
                zombie_grid = torch.sum(zombie_grid, axis=1).view(-1, config.N_LANES)
            if self.use_gridnet:
                plant_grid = self.gridnet(plant_grid)
            state_t = torch.cat([plant_grid, zombie_grid, state_t[:, 2 * self._grid_size:]], axis=1)
        else:
            state_t = torch.FloatTensor(state).to(device=self.device)
            zombie_grid = state_t[self._grid_size:(2 * self._grid_size)].reshape(-1, config.LANE_LENGTH)
            plant_grid = state_t[:self._grid_size]
            if self.use_zombienet:
                zombie_grid = self.zombienet(zombie_grid).view(-1)
            else:
                zombie_grid = torch.sum(zombie_grid, axis=1)
            if self.use_gridnet:
                plant_grid = self.gridnet(plant_grid)
            state_t = torch.cat([plant_grid, zombie_grid, state_t[2 * self._grid_size:]])
        return self.network(state_t)

    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions[mask])
        return self.get_greedy_action(state, mask)

    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    # ── Mask recalculation (identical to pvz_rl _get_mask) ────────────
    def _get_mask(self, observation):
        empty_cells = np.nonzero(
            (observation[:self._grid_size] == 0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(self.n_outputs, dtype=bool)
        mask[0] = True
        empty_cells_flat = (empty_cells[0] + config.N_LANES * empty_cells[1]) * 4  # num_cards=4
        available_plants = observation[-4:]  # last 4 = action_avail
        for i in range(4):
            if available_plants[i]:
                idx = empty_cells_flat + i + 1
                mask[idx] = True
        return mask


# ── Observation normalization (identical to pvz_rl _transform_observation)
def _transform_observation(observation):
    obs = observation.astype(np.float64)
    obs[45:90] /= HP_NORM    # no-op (HP_NORM=1)
    obs[90] /= SUN_NORM       # /200
    return obs


# ── DDQN loss (identical to pvz_rl calculate_loss) ────────────────────
def _calculate_loss(network, target_network, batch, gamma):
    states, actions, rewards, dones, next_states = [i for i in batch]
    rewards_t = torch.FloatTensor(rewards).to(device=network.device).reshape(-1, 1)
    actions_t = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(device=network.device)
    dones_t = torch.BoolTensor(dones).to(device=network.device)

    qvals = torch.gather(network.get_qvals(states), 1, actions_t)

    with torch.no_grad():
        next_masks = np.array([network._get_mask(s) for s in next_states])
        next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool, device=network.device)
        qvals_next_pred = network.get_qvals(next_states)
        qvals_next_pred = qvals_next_pred.clone()
        qvals_next_pred[~next_masks_t] = qvals_next_pred.min()
        next_actions = torch.max(qvals_next_pred, dim=-1)[1]
        next_actions_t = next_actions.reshape(-1, 1).to(device=network.device)
        target_qvals = target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t)
    qvals_next[dones_t] = 0
    expected_qvals = gamma * qvals_next + rewards_t
    return nn.MSELoss()(qvals, expected_qvals)


# ── Training ──────────────────────────────────────────────────────────
def train_sim_ddqn(
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
    real_rewards = []
    real_iterations = []
    update_loss = []
    step_count = 0
    window = 100

    # ── Burn-in (identical to pvz_rl) ──
    print(f"Burn-in ({burn_in} steps)...")
    s_0 = _transform_observation(env.reset())
    while buffer.burn_in_capacity() < 1:
        mask = np.array(env.mask_available_actions())
        if np.random.random() < 0.5:
            action = 0
        else:
            action = np.random.choice(np.arange(env.action_space.n)[mask])
        s_1, reward, done, _ = env.step(action)
        s_1 = _transform_observation(s_1)
        buffer.append(s_0, action, reward, done, s_1)
        s_0 = s_1.copy()
        if done:
            s_0 = _transform_observation(env.reset())
        step_count += 1
    print(f"Burn-in done. Buffer: {len(buffer.replay_memory)}")

    # ── Training loop (identical to pvz_rl) ──
    ep = 0
    s_0 = _transform_observation(env.reset())
    print(f"Training {max_episodes} episodes...")

    while ep < max_episodes:
        rewards = 0
        done = False
        while not done:
            epsilon = threshold.epsilon(ep)
            mask = np.array(env.mask_available_actions())
            action = network.decide_action(s_0, mask, epsilon=epsilon)
            s_1, r, done, _ = env.step(action)
            s_1 = _transform_observation(s_1)
            rewards += r
            buffer.append(s_0, action, r, done, s_1)
            s_0 = s_1.copy()
            step_count += 1

            if step_count % network_update_freq == 0:
                network.optimizer.zero_grad(set_to_none=True)
                batch = buffer.sample_batch(batch_size=batch_size)
                loss = _calculate_loss(network, target_network, batch, gamma)
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

                s_0 = _transform_observation(env.reset())

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
    state = _transform_observation(env.reset())
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
        state = _transform_observation(state)
        total_reward += reward
    env.disable_render_collection()
    print(f"\nReplay: {len(env.render_data)} frames, reward={total_reward:.0f}")
    replay_episode(env.render_data, fps=15,
                   title=f"SimPVZ Trained Agent - Reward: {total_reward:.0f}")


if __name__ == "__main__":
    train_sim_ddqn(
        max_episodes=100000,
        buffer_size=100000,
        burn_in=10000,
        batch_size=200,
    )
