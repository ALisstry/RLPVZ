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


class FactoredDDQNNetwork(DDQNNetwork):
    """3-Factor Q-Network: Q(card,row,col) = q_card + q_row + q_col + q_wait.

    Decomposes the 451-action space into independent factors:
        q_card ∈ R^{10}   — card-type preference
        q_row  ∈ R^{5}    — row preference
        q_col  ∈ R^{9}    — column preference
        q_wait ∈ R^{1}    — "do nothing" value

    Full Q-values are rebuilt by explicitly enumerating all
    10 × 5 × 9 = 450 plant-action combinations at query time.
    Output: 10 + 5 + 9 + 1 = 25 dims (vs 451, ~18× compression).
    """

    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None):
        super().__init__(env, learning_rate=learning_rate, device=device,
                         hidden_sizes=hidden_sizes)
        if hidden_sizes is None:
            hidden_sizes = [2048, 2048]

        # ── Rebuild trunk (all hidden layers) ──
        trunk_layers = []
        prev = self.n_inputs
        for h in hidden_sizes:
            trunk_layers.append(nn.Linear(prev, h, bias=True))
            trunk_layers.append(nn.LeakyReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        trunk_out = prev if hidden_sizes else self.n_inputs

        # ── Factored heads (direct Linear — free-range Q values) ──
        self.head_card = nn.Linear(trunk_out, self.num_cards, bias=True)  # 10
        self.head_row  = nn.Linear(trunk_out, self.rows, bias=True)       #  5
        self.head_col  = nn.Linear(trunk_out, self.cols, bias=True)       #  9
        self.head_wait = nn.Linear(trunk_out, 1, bias=True)               #  1

        self.network = None

        if self.device == "cuda":
            self.to("cuda")

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        single = state_t.dim() == 1
        if single:
            state_t = state_t.unsqueeze(0)

        feat = self.trunk(state_t)
        q_card = self.head_card(feat)   # (B, 10)
        q_row  = self.head_row(feat)    # (B,  5)
        q_col  = self.head_col(feat)    # (B,  9)
        q_wait = self.head_wait(feat)   # (B,  1)

        # Explicit enumeration: Q(card_i, row_j, col_k) = q_card[i] + q_row[j] + q_col[k]
        # Broadcasting: (B,10,1,1) + (B,1,5,1) + (B,1,1,9) = (B,10,5,9)
        q_plant = (
              q_card.unsqueeze(-1).unsqueeze(-1)
            + q_row.unsqueeze(1).unsqueeze(-1)
            + q_col.unsqueeze(1).unsqueeze(1)
        ).reshape(state_t.shape[0], -1)                    # (B, 450)

        qvals = torch.cat([q_plant, q_wait], dim=-1)      # (B, 451)
        if single:
            qvals = qvals.squeeze(0)
        return qvals


class DifferentialDDQNNetwork(DDQNNetwork):
    """Differential Q-Network: Q(s,a) = Q(s,wait) + Δ(s,a),  Δ(s,wait) ≡ 0.

    Splits after a shared trunk into two heads:
        wait_head(s)  → Q(s, wait) ∈ R
        delta_head(s) → Δ(s, ·) ∈ R^{n_actions}

    Q(s, a) = Q(s, wait) + Δ(s, a)
    """

    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None):
        super().__init__(env, learning_rate=learning_rate, device=device,
                         hidden_sizes=hidden_sizes)
        if hidden_sizes is None:
            hidden_sizes = [2048, 2048]

        if len(hidden_sizes) >= 2:
            trunk_sizes = hidden_sizes[:-1]
            branch_in = hidden_sizes[-1]
        else:
            trunk_sizes = []
            branch_in = hidden_sizes[0]

        # ── Rebuild trunk ──
        trunk_layers = []
        prev = self.n_inputs
        for h in trunk_sizes:
            trunk_layers.append(nn.Linear(prev, h, bias=True))
            trunk_layers.append(nn.LeakyReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        trunk_out = prev if trunk_sizes else self.n_inputs

        # ── Wait baseline head ──
        self.wait_head = nn.Sequential(
            nn.Linear(trunk_out, branch_in, bias=True),
            nn.LeakyReLU(),
            nn.Linear(branch_in, 1, bias=True),
        )

        # ── Delta head ──
        self.delta_head = nn.Sequential(
            nn.Linear(trunk_out, branch_in, bias=True),
            nn.LeakyReLU(),
            nn.Linear(branch_in, self.n_outputs, bias=True),
        )

        self.network = None
        self._wait_idx = self.n_outputs - 1

        if self.device == "cuda":
            self.to("cuda")

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        single = state_t.dim() == 1
        if single:
            state_t = state_t.unsqueeze(0)

        shared = self.trunk(state_t)
        q_wait = self.wait_head(shared)
        delta = self.delta_head(shared)
        delta[:, self._wait_idx] = 0.0

        qvals = q_wait + delta
        if single:
            qvals = qvals.squeeze(0)
        return qvals


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

    # ── Wait-baseline / differential statistics (always computed) ─────
    with torch.no_grad():
        q_wait = all_qvals[:, -1]
        delta_all = all_qvals - q_wait.unsqueeze(-1)
        q_wait_mean = float(q_wait.mean().cpu().item())
        delta_mean = float(delta_all.mean().cpu().item())
        delta_max = float(delta_all.max(dim=1).values.mean().cpu().item())

    diagnostics = {
        "advantage": advantage,
        "entropy": entropy,
        "mean_q": mean_q,
        "max_q": max_q,
        "td_error": td_error,
        "q_wait": q_wait_mean,
        "delta_mean": delta_mean,
        "delta_max": delta_max,
    }

    return LossResult(
        loss=_loss_fn(qvals, expected_qvals),
        diagnostics=diagnostics,
    )
