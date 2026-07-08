"""DDQN model, replay buffer, and loss for the simulation environment."""

from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.replay_memory = [None] * memory_size  # ring buffer — O(1) indexing
        self._write_ptr = 0
        self._size = 0

    def __len__(self):
        return self._size

    def sample_batch(self, batch_size=32):
        n = min(self._size, self.memory_size)
        samples = np.random.choice(n, batch_size, replace=False)
        entries = [self.replay_memory[i] for i in samples]
        return zip(*entries)

    def append(self, state, action, reward, done, next_state, mask, next_mask):
        self.replay_memory[self._write_ptr] = self.Buffer(
            state, action, reward, done, next_state, mask, next_mask,
        )
        self._write_ptr = (self._write_ptr + 1) % self.memory_size
        self._size = min(self._size + 1, self.memory_size)

    def burn_in_capacity(self):
        return self._size / max(1, self.burn_in)


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
            hidden_sizes = [2048, 2048]

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


_loss_fn = nn.SmoothL1Loss(beta=1.0)

# Return type for calculate_loss: loss scalar + diagnostics
LossResult = namedtuple("LossResult", ["loss", "diagnostics"])


def calculate_loss(network, target_network, batch, gamma):
    """Double-DQN loss with training diagnostics.

    Returns a :class:`LossResult` containing the scalar Huber loss and a
    ``diagnostics`` dict with the following keys (all Python floats):

    * ``advantage`` — Q(s,a) − mean(Q(s,·))  (how much better the chosen
      action is vs the mean)
    * ``entropy`` — −Σ softmax(Q) · log_softmax(Q)  (policy spread)
    * ``mean_q`` — mean Q-value over the batch
    * ``max_q`` — mean of per-sample max Q-values
    * ``td_error`` — mean |Q(s,a) − target|
    """
    states, actions, rewards, dones, next_states, masks, next_masks = [
        item for item in batch
    ]
    rewards_t = torch.FloatTensor(rewards).to(device=network.device).reshape(-1, 1)
    actions_t = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(device=network.device)
    dones_t = torch.BoolTensor(dones).to(device=network.device)

    # ── Current Q-values ─────────────────────────────────────────────
    all_qvals = network.get_qvals(states)                 # (B, n_actions)
    qvals = torch.gather(all_qvals, 1, actions_t)         # (B, 1)

    # ── Diagnostics (detached) ───────────────────────────────────────
    with torch.no_grad():
        mean_q = float(qvals.mean().cpu().item())
        max_q = float(all_qvals.max(dim=1).values.mean().cpu().item())

        probs = F.softmax(all_qvals, dim=1)               # (B, n_actions)
        log_probs = torch.log(probs + 1e-12)
        entropy = float(-(probs * log_probs).sum(dim=1).mean().cpu().item())

        advantage = float(
            (qvals - all_qvals.mean(dim=1, keepdim=True)).mean().cpu().item()
        )

    # ── Double-DQN target ────────────────────────────────────────────
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

    # ── TD error (detached) ──────────────────────────────────────────
    with torch.no_grad():
        td_error = float((expected_qvals - qvals).abs().mean().cpu().item())

    diagnostics = {
        "advantage": advantage,
        "entropy": entropy,
        "mean_q": mean_q,
        "max_q": max_q,
        "td_error": td_error,
    }

    return LossResult(
        loss=_loss_fn(qvals, expected_qvals),
        diagnostics=diagnostics,
    )
