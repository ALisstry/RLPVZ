"""DDQN model, replay buffer, and loss for the simulation environment."""

import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque

from simenv.pvz_sim import config

HP_NORM = 1
SUN_NORM = 200


# ── Replay Buffer ───────────────────────────────────────────────────────
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


# ── Feature extractors ───────────────────────────────────────────────────
class ZombieNet(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(config.LANE_LENGTH, output_size)

    def forward(self, x):
        return self.fc1(x)


# ── Q-Network ────────────────────────────────────────────────────────────
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

        # ── Compute combined input size ──
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

    # ── Forward pass ─────────────────────────────────────────────────
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

    # ── Mask recalculation ───────────────────────────────────────────
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


# ── Observation normalization ────────────────────────────────────────────
def transform_observation(observation):
    obs = observation.astype(np.float64)
    obs[45:90] /= HP_NORM    # no-op (HP_NORM=1)
    obs[90] /= SUN_NORM       # /200
    return obs


# ── DDQN loss ────────────────────────────────────────────────────────────
def calculate_loss(network, target_network, batch, gamma):
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
