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


class SumTree:
    """Binary sum-tree for O(log N) priority-based sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._ptr = 0
        self.n_entries = 0

    def total(self) -> float:
        return float(self.tree[0])

    def max_priority(self) -> float:
        leaves = self.tree[self.capacity - 1:self.capacity - 1 + self.n_entries]
        return float(np.max(leaves)) if self.n_entries > 0 else 1.0

    def add(self, priority: float) -> int:
        idx = self._ptr + self.capacity - 1
        self.update(idx, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get_leaf(self, value: float) -> int:
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx

    def sample(self, n: int) -> np.ndarray:
        indices = np.empty(n, dtype=np.int64)
        total = self.total()
        if total <= 0:
            offset = self.capacity - 1
            indices[:] = offset + np.random.randint(0, self.n_entries, size=n)
            return indices
        segment = total / n
        for i in range(n):
            value = np.random.uniform(i * segment, (i + 1) * segment)
            indices[i] = self.get_leaf(min(value, total - 1e-12))
        return indices


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (Schaul et al., 2016).

    Stores transitions in a fixed-size ring buffer and samples them
    proportional to ``priority ** alpha`` via a SumTree.
    """

    def __init__(self, memory_size=50000, burn_in=10000,
                 alpha: float = 0.6, epsilon: float = 1e-6):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.alpha = alpha
        self.epsilon = epsilon

        self.Buffer = namedtuple(
            "Buffer",
            field_names=[
                "state", "action", "reward", "done",
                "next_state", "mask", "next_mask",
            ],
        )
        self.replay_memory = [None] * memory_size
        self.sum_tree = SumTree(memory_size)
        self._write_ptr = 0

    def __len__(self):
        return self.sum_tree.n_entries

    def burn_in_capacity(self):
        return self.sum_tree.n_entries / max(1, self.burn_in)

    def append(self, state, action, reward, done, next_state, mask, next_mask):
        self.replay_memory[self._write_ptr] = self.Buffer(
            state, action, reward, done, next_state, mask, next_mask,
        )
        max_p = self.sum_tree.max_priority()
        priority = max(max_p, 1.0)
        self.sum_tree.add(priority ** self.alpha)
        self._write_ptr = (self._write_ptr + 1) % self.memory_size

    def sample_batch(self, batch_size=32, beta: float = 0.4):
        tree_indices = self.sum_tree.sample(batch_size)
        data_indices = tree_indices - (self.sum_tree.capacity - 1)

        probs = self.sum_tree.tree[tree_indices] / max(self.sum_tree.total(), 1e-12)
        n = self.sum_tree.n_entries
        is_weights = (n * probs) ** (-beta)
        is_weights /= max(is_weights.max(), 1e-12)

        entries = [self.replay_memory[i] for i in data_indices]
        batch = zip(*entries)
        return batch, tree_indices, is_weights.astype(np.float32).reshape(-1, 1)

    def update_priorities(self, tree_indices, td_errors):
        flat_errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        for idx, td_err in zip(tree_indices, flat_errors):
            priority = float(abs(td_err)) + self.epsilon
            self.sum_tree.update(int(idx), priority ** self.alpha)


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
LossResult = namedtuple("LossResult", ["loss", "diagnostics", "td_errors"])


def calculate_loss(network, target_network, batch, gamma, is_weights=None):
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

    # Apply PER importance-sampling weights if provided
    with torch.no_grad():
        td_errors = (expected_qvals - qvals).detach().cpu().numpy()
    if is_weights is not None:
        is_weights_t = torch.as_tensor(
            is_weights, dtype=torch.float32, device=qvals.device,
        )
        elementwise = F.smooth_l1_loss(qvals, expected_qvals, beta=1.0, reduction='none')
        loss = (elementwise * is_weights_t).mean()
    else:
        loss = _loss_fn(qvals, expected_qvals)

    return LossResult(loss=loss, diagnostics=diagnostics, td_errors=td_errors)


class RowFirstCNNQNetwork(nn.Module):
    """Row-First CNN for DDQN — single serial backbone, PvZ-mechanic-driven.

    Design rationale:
    1. PvZ rows are independent battle lanes → the first conv aggregates
       horizontal context **before** any cross-row spatial reasoning.
    2. A single serial backbone avoids the FC-fusion overhead of dual-branch
       designs.
    3. Global Average Pooling (AdaptiveAvgPool2d) at the network neck
       eliminates a large grid_proj FC layer.

    Architecture:
      grid (B,13,5,9)
        Conv2d(13→64, k=(1,5), p=(0,2)) → BN → ReLU    (5,9)
          ↑ 1×5 kernel: half-row horizontal aggregation
        Conv2d(64→128, k=3, p=1) → BN → ReLU
          + MaxPool2d(2,2,ceil) → (3,5)
          ↑ 2D spatial reasoning on horizontal features
        Conv2d(128→256, k=3, p=1) → BN → ReLU
          + AdaptiveAvgPool2d((1,1)) → GAP → 256d
          ↑ deeper features, collapse to vector
      global (B,11)
        Linear(11→64) → ReLU → 64d
      shared = 256 + 64 = 320
        head: Linear(320→128) → ReLU → Linear(128→451)

    Parameter count: ~0.47M (standard), ~0.52M (factored).
    """

    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None, use_factored: bool = False):
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
        self._use_factored = use_factored
        self._n_cells = self.grid_size       # 45
        self._n_cards = self.num_cards        # 10

        # ── derived dims ──────────────────────────────────────────
        n_grid_channels = self.num_cards + 1 + 2  # one-hot(11) + plantHP(1) + zombieHP(1)
        n_global = 1 + self.num_cards              # sun(1) + cooldowns(10)
        self._n_grid_channels = n_grid_channels
        self._n_global = n_global

        # ── Row Encoder ───────────────────────────────────────────
        # 1×5 kernel with padding=(0,2): crosses ~half the row width
        self.row_encoder = nn.Sequential(
            nn.Conv2d(n_grid_channels, 64, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Spatial Encoder ───────────────────────────────────────
        # Input  (5, 9)
        # Conv1 + MaxPool(2,2) → (3, 5)   [ceil_mode]
        # Conv2 + AdaptiveAvgPool2d((1,1)) → (1, 1)  [GAP]
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        _grid_feat = 256  # GAP output: 1 × 1 × 256

        # ── global branch ─────────────────────────────────────────
        self.global_proj = nn.Sequential(
            nn.Linear(n_global, 64),
            nn.ReLU(inplace=True),
        )

        # ── output head ───────────────────────────────────────────
        shared_dim = _grid_feat + 64  # 256 + 64 = 320
        if use_factored:
            head_hidden = 256
            shared_head = nn.Sequential(
                nn.Linear(shared_dim, head_hidden),
                nn.ReLU(inplace=True),
            )
            self.head_wait = nn.Sequential(
                shared_head,
                nn.Linear(head_hidden, 1),
            )
            self.head_pos = nn.Sequential(
                nn.Linear(shared_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self._n_cells),   # 45
            )
            self.head_card = nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self._n_cards),    # 10
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(shared_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.n_outputs),  # 451
            )

        if device == "cuda":
            self.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 596) flat observation vector."""
        bsz = x.shape[0]
        n_cells = self.rows * self.cols  # 45
        n_onehot = self.num_cards + 1    # 11

        glob_ = x[:, :self._n_global]                     # (B, 11)
        grid_flat = x[:, self._n_global:]                 # (B, 585)

        # Split feature-major blocks
        split_1 = n_cells * n_onehot                       # 495
        split_2 = split_1 + n_cells                        # 540

        onehot = grid_flat[:, :split_1]                    # (B, 495)
        plant_hp = grid_flat[:, split_1:split_2]           # (B, 45)
        zombie_hp = grid_flat[:, split_2:]                 # (B, 45)

        # Interleave into per-cell channels
        onehot = onehot.view(bsz, n_cells, n_onehot)       # (B, 45, 11)
        plant_hp = plant_hp.view(bsz, n_cells, 1)          # (B, 45, 1)
        zombie_hp = zombie_hp.view(bsz, n_cells, 1)        # (B, 45, 1)

        grid = torch.cat([onehot, plant_hp, zombie_hp], dim=-1)  # (B, 45, 13)
        grid = grid.view(bsz, self.rows, self.cols, self._n_grid_channels)  # (B, 5, 9, 13)
        grid = grid.permute(0, 3, 1, 2).contiguous()       # (B, 13, 5, 9)

        # Serial backbone: row-first → spatial
        grid = self.row_encoder(grid)                       # (B, 64, 5, 9)
        grid_feat = self.spatial_encoder(grid).reshape(bsz, -1)  # (B, 256)

        glob_feat = self.global_proj(glob_)                  # (B, 64)
        shared = torch.cat([grid_feat, glob_feat], dim=1)    # (B, 320)

        if self._use_factored:
            q_wait = self.head_wait(shared)                      # (B, 1)
            q_pos  = self.head_pos(shared)                       # (B, 45)
            q_card = self.head_card(shared)                      # (B, 10)

            q_plant = q_card.unsqueeze(-1) + q_pos.unsqueeze(-2)  # (B, 10, 45)
            q_plant = q_plant.reshape(bsz, 450)                   # (B, 450)
            return torch.cat([q_plant, q_wait], dim=-1)           # (B, 451)
        else:
            return self.head(shared)                              # (B, 451)

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
        single = state_t.dim() == 1
        if single:
            state_t = state_t.unsqueeze(0)
        qvals = self.forward(state_t)
        if single:
            qvals = qvals.squeeze(0)
        return qvals
