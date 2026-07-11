"""CNN-based Q-Network for DDQN with dual-branch grid processing.

3x3 branch: captures local spatial patterns (adjacent cell interactions).
1x9 branch: captures full-row patterns (entire row defence/offence balance).
Both branches use MaxPool2d after every conv to reduce spatial dims and
shift parameters from FC layers into deeper CNN channels.
Global features (sun + cooldowns) bypass CNN and merge after grid encoding.
"""

import numpy as np
import torch
import torch.nn as nn


class CNNQNetwork(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None, n_inputs_override=None,
                 create_optimizer=True):
        super().__init__()
        self.device = device
        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        # ── derived dims ──────────────────────────────────────────
        n_grid_channels = self.num_cards + 1 + 2   # one-hot(11) + plantHP(1) + zombieHP(1)
        n_global = 1 + self.num_cards               # sun(1) + cooldowns(10)
        self._n_grid_channels = n_grid_channels
        self._n_global = n_global

        # ── 3x3 spatial branch ────────────────────────────────────
        # Input  (5, 9)
        # Conv1 + MaxPool(2,2) → (3, 5)   [ceil_mode]
        # Conv2 + MaxPool(2,2) → (2, 3)
        # Conv3 + MaxPool(2,2) → (1, 2)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(n_grid_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )
        _3x3_flat = 1 * 2 * 512  # 1024

        # ── 1x9 row branch ────────────────────────────────────────
        # Input  (5, 9)
        # Conv(1,9) → (5, 1)  then MaxPool(2,1) → (3, 1)
        # Conv(1,1) → (3, 1)  then MaxPool(2,1) → (2, 1)
        # Conv(1,1) → (2, 1)  then MaxPool(2,1) → (1, 1)
        self.branch_1x9 = nn.Sequential(
            nn.Conv2d(n_grid_channels, 128, kernel_size=(1, 9), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), ceil_mode=True),

            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), ceil_mode=True),

            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), ceil_mode=True),
        )
        _1x9_flat = 1 * 1 * 256  # 256

        # ── grid merge ────────────────────────────────────────────
        grid_out = _3x3_flat + _1x9_flat  # 1280
        self.grid_proj = nn.Sequential(
            nn.Linear(grid_out, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # ── global branch ─────────────────────────────────────────
        self.global_proj = nn.Sequential(
            nn.Linear(n_global, 128),
            nn.ReLU(inplace=True),
        )

        # ── output head ───────────────────────────────────────────
        shared_dim = 1024 + 128  # 1152
        self.head = nn.Sequential(
            nn.Linear(shared_dim, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, self.n_outputs),
        )

        if device == "cuda":
            self.cuda()

        self.optimizer = None
        if create_optimizer:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 596) flat observation vector.

        The paper-format vector (build_paper_state_vector) stores the 585 grid
        elements in *feature-major* blocks::

            [ onehot_all(495) | plant_hp_all(45) | zombie_hp_all(45) ]

        This method rearranges them into channel-last interleaved layout
        ``(B, 5, 9, 13)`` so the 3×3 / 1×9 convolutions see coherent
        per-cell channel vectors.
        """
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

        # Reshape each block → (B, cells, features) and interleave
        onehot = onehot.view(bsz, n_cells, n_onehot)       # (B, 45, 11)
        plant_hp = plant_hp.view(bsz, n_cells, 1)          # (B, 45, 1)
        zombie_hp = zombie_hp.view(bsz, n_cells, 1)        # (B, 45, 1)

        grid = torch.cat([onehot, plant_hp, zombie_hp], dim=-1)  # (B, 45, 13)
        grid = grid.view(bsz, self.rows, self.cols, self._n_grid_channels)  # (B, 5, 9, 13)
        grid = grid.permute(0, 3, 1, 2).contiguous()       # (B, 13, 5, 9)

        feat_3x3 = self.branch_3x3(grid).reshape(bsz, -1)
        feat_1x9 = self.branch_1x9(grid).reshape(bsz, -1)

        grid_feat = self.grid_proj(torch.cat([feat_3x3, feat_1x9], dim=1))
        glob_feat = self.global_proj(glob_)
        shared = torch.cat([grid_feat, glob_feat], dim=1)  # (B, 1152)

        return self.head(shared)                              # (B, 451)

    # ── DDQN interface ─────────────────────────────────────────────
    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            valid_actions = self.actions[np.asarray(mask, dtype=bool)]
            return np.random.choice(valid_actions)
        return self.get_greedy_action(state, mask)

    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        if state.ndim == 1:
            state = state[np.newaxis, :]
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        qvals = self.forward(state_t)
        if qvals.shape[0] == 1:
            qvals = qvals.squeeze(0)
        return qvals


class RowFirstCNNQNetwork(nn.Module):
    """Row-First CNN for DDQN — single serial backbone, PvZ-mechanic-driven.

    Design rationale (in order of priority):
    1. PvZ rows are independent battle lanes → the first conv must aggregate
       horizontal context **before** any cross-row spatial reasoning.
    2. A single serial backbone avoids the FC-fusion overhead that dual-branch
       designs pay (grid_proj concat + linear is ~35% of params in V1/V2).
    3. Global Average Pooling (AdaptiveAvgPool2d((1,1))) at the network neck
       eliminates the grid_proj FC entirely — the CNN hands a compact vector
       straight to the head.

    Architecture (Row-First CNN):
      grid (B,13,5,9) ────────────────────────────────────────────┐
        Conv2d(13→64, k=(1,5), p=(0,2)) → BN → ReLU    (5,9)     │
          ↑ 1×5 kernel: ~half-row horizontal aggregation          │
        Conv2d(64→128, k=3, p=1) → BN → ReLU                      │
          + MaxPool2d(2,2,ceil) → (3,5)                           │
          ↑ first 2D spatial reasoning on horizontal features     │
        Conv2d(128→256, k=3, p=1) → BN → ReLU                     │
          + AdaptiveAvgPool2d((1,1)) → GAP → 256d                 │
          ↑ deeper features, collapse to vector — no grid_proj    │
      global (B,11) ──────────────────────────────────────────────┤
        Linear(11→64) → ReLU → 64d                                │
      shared = 256 + 64 = 320 ────────────────────────────────────┤
        head: Linear(320→128) → ReLU → Linear(128→451)            │
        factored: shared(256) → wait(1), pos(128→45), card(64→10) │

    Parameter count: ~0.47M (standard), ~0.48M (factored).
    """

    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None, n_inputs_override=None,
                 create_optimizer=True, use_factored: bool = False):
        super().__init__()
        self.device = device
        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        self._use_factored = use_factored
        self._n_cells = self.rows * self.cols  # 45
        self._n_cards = self.num_cards          # 10

        # ── derived dims ──────────────────────────────────────────
        n_grid_channels = self.num_cards + 1 + 2   # one-hot(11) + plantHP(1) + zombieHP(1)
        n_global = 1 + self.num_cards               # sun(1) + cooldowns(10)
        self._n_grid_channels = n_grid_channels
        self._n_global = n_global

        # ── Row Encoder ───────────────────────────────────────────
        # 1×5 kernel with padding=(0,2): crosses ~half the row width
        # without mixing rows — the PvZ-mechanic-aligned first step.
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
        glob_proj_dim = 64
        self.global_proj = nn.Sequential(
            nn.Linear(n_global, glob_proj_dim),
            nn.ReLU(inplace=True),
        )

        # ── output head ───────────────────────────────────────────
        shared_dim = _grid_feat + glob_proj_dim  # 256 + 64 = 320
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
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Linear(shared_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.n_outputs),  # 451
            )

        if device == "cuda":
            self.cuda()

        self.optimizer = None
        if create_optimizer:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 596) flat observation vector.

        Splits into global (11) and grid (585), rearranges grid into
        ``(B, 13, 5, 9)``, then passes through the serial backbone:
        row_encoder → spatial_encoder → GAP → merge with global → head.
        """
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

    # ── DDQN interface ─────────────────────────────────────────────
    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            valid_actions = self.actions[np.asarray(mask, dtype=bool)]
            return np.random.choice(valid_actions)
        return self.get_greedy_action(state, mask)

    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        if state.ndim == 1:
            state = state[np.newaxis, :]
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        qvals = self.forward(state_t)
        if qvals.shape[0] == 1:
            qvals = qvals.squeeze(0)
        return qvals
