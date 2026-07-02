"""DDQN model, replay buffer, and loss for the simulation environment."""

from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple(
            "Buffer",
            field_names=[
                "state",
                "action",
                "reward",
                "done",
                "next_state",
                "mask",
                "next_mask",
            ],
        )
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state, mask, next_mask):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state, mask, next_mask)
        )

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


class DDQNNetwork(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None):
        super().__init__()
        self.device = device
        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.grid_size = self.rows * self.cols
        self.n_inputs = int(env.state_dim)
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        if hidden_sizes is None:
            hidden_sizes = [2048, 1024]

        layers = []
        prev_size = self.n_inputs
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size, bias=True))
            layers.append(nn.LeakyReLU())
            prev_size = h_size
        layers.append(nn.Linear(prev_size, self.n_outputs, bias=True))
        self.network = nn.Sequential(*layers)

        if self.device == "cuda":
            self.network.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

    @torch.no_grad()
    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            valid_actions = self.actions[np.asarray(mask, dtype=bool)]
            return np.random.choice(valid_actions)
        return self.get_greedy_action(state, mask)

    @torch.no_grad()
    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.network(state_t)


def transform_observation(observation):
    return observation.astype(np.float32)


_mse_loss = nn.MSELoss()


def calculate_loss(network, target_network, batch, gamma):
    states, actions, rewards, dones, next_states, masks, next_masks = [
        item for item in batch
    ]
    rewards_t = torch.FloatTensor(rewards).to(device=network.device).reshape(-1, 1)
    actions_t = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(device=network.device)
    dones_t = torch.BoolTensor(dones).to(device=network.device)

    qvals = torch.gather(network.get_qvals(states), 1, actions_t)

    next_masks = np.array(next_masks, dtype=bool)
    with torch.no_grad():
        qvals_next_pred = network.get_qvals(next_states)
        next_masks_t = torch.as_tensor(
            next_masks, dtype=torch.bool, device=qvals_next_pred.device)
        qvals_next_pred = qvals_next_pred.clone()
        qvals_next_pred[~next_masks_t] = qvals_next_pred.min()
        next_actions = torch.max(qvals_next_pred, dim=-1)[1]
        next_actions_t = next_actions.reshape(-1, 1).to(device=network.device)

        target_qvals = target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t)
    qvals_next[dones_t] = 0
    expected_qvals = gamma * qvals_next + rewards_t
    return _mse_loss(qvals, expected_qvals)
