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

    @torch.no_grad()
    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions[mask])
        return self.get_greedy_action(state, mask)

    @torch.no_grad()
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


# ── Deep MLP Q-Network ────────────────────────────────────────────────────
class SimDeepMLPNetwork(nn.Module):
    """Deep MLP with parameter count matching SimCNNQNetwork (~848K).

    Uses the same 55-dim input as SimQNetwork (flat plant grid + zombie HP
    summed per lane + sun + availability), then passes through deep FC layers:

        55 -> 1024 -> 512 -> 384 -> 181

    This keeps the existing MLP while providing a parameter-matched alternative
    for fair comparison against the CNN.
    """

    def __init__(self, env, learning_rate=1e-3, device="cpu"):
        super().__init__()
        self.device = device
        self._grid_size = config.N_LANES * config.LANE_LENGTH  # 45
        self.n_outputs = env.action_space.n                     # 181
        self.actions = np.arange(self.n_outputs)
        self.learning_rate = learning_rate
        self._num_cards = len(env.plant_deck)                   # 4

        self.network = nn.Sequential(
            nn.Linear(55, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 384, bias=True),
            nn.ReLU(),
            nn.Linear(384, self.n_outputs, bias=True),
        )

        if device == "cuda":
            self.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    # ── Forward pass ─────────────────────────────────────────────────
    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        plant_grid = state_t[:, :self._grid_size]                       # (B, 45)
        zombie_grid = state_t[:, self._grid_size:2 * self._grid_size]   # (B, 45)
        zombie_by_lane = torch.sum(
            zombie_grid.reshape(-1, config.N_LANES, config.LANE_LENGTH), dim=-1)  # (B, 5)
        state_t = torch.cat([plant_grid, zombie_by_lane,
                             state_t[:, 2 * self._grid_size:]], dim=1)   # (B, 55)

        out = self.network(state_t)
        if squeeze:
            out = out.squeeze(0)
        return out

    @torch.no_grad()
    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions[mask])
        return self.get_greedy_action(state, mask)

    @torch.no_grad()
    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def _get_mask(self, observation):
        empty_cells = np.nonzero(
            (observation[:self._grid_size] == 0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(self.n_outputs, dtype=bool)
        mask[0] = True
        empty_cells_flat = (empty_cells[0] + config.N_LANES * empty_cells[1]) * self._num_cards
        available_plants = observation[-self._num_cards:]
        for i in range(self._num_cards):
            if available_plants[i]:
                idx = empty_cells_flat + i + 1
                mask[idx] = True
        return mask


# ── CNN Q-Network ──────────────────────────────────────────────────────────
class SimCNNQNetwork(nn.Module):
    """CNN-based Q-Network with separate feature extractors for plant and zombie
    grids, using two kernel types:

    - **3×3 kernels**: capture local spatial patterns (plant clusters, adjacency)
    - **1×9 (full-row) kernels**: capture lane-wide context (each row is an
      independent lane in PvZ)

    Plant grid (categorical) is embedded first, then fed to both CNN branches.
    Zombie grid (HP values) is processed directly as a single channel.
    """

    def __init__(self, env, learning_rate=1e-3, device="cpu"):
        super().__init__()
        self.device = device
        self._rows = config.N_LANES               # 5
        self._cols = config.LANE_LENGTH            # 9
        self._grid_size = self._rows * self._cols  # 45
        self.n_outputs = env.action_space.n        # 181
        self.actions = np.arange(self.n_outputs)
        self.learning_rate = learning_rate
        self._num_cards = len(env.plant_deck)      # 4
        self._num_plant_types = self._num_cards + 1  # 5 (0=empty)

        embed_dim = 8

        # Plant type embedding: categorical index → dense feature vector
        self.plant_embed = nn.Embedding(self._num_plant_types, embed_dim)

        # ── 3×3 Conv branch ─────────────────────────────────────────────
        self.plant_conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.zombie_conv3 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ── 1×9 (full-row) Conv branch ──────────────────────────────────
        self.plant_conv_row = nn.Sequential(
            nn.Conv2d(embed_dim, 16, kernel_size=(1, self._cols)),
            nn.ReLU(),
        )
        self.zombie_conv_row = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, self._cols)),
            nn.ReLU(),
        )

        # Flattened feature dimensions:
        #   conv3:  32 ch × 5 × 9 = 1440  (×2 for plant+zombie)
        #   row:    16 ch × 5 × 1 =   80  (×2 for plant+zombie)
        cnn_dim = 2 * (32 * self._rows * self._cols + 16 * self._rows)
        extra_dim = 1 + self._num_cards  # sun + cooldown availability

        self.fc = nn.Sequential(
            nn.Linear(cnn_dim + extra_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_outputs),
        )

        if device == "cuda":
            self.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    # ── CNN feature extraction ─────────────────────────────────────────
    def _cnn_features(self, state_t):
        """Extract CNN features.  Handles both (B, 95) and (95,) tensors."""
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B = state_t.shape[0]
        R, C, G = self._rows, self._cols, self._grid_size

        # Decompose state vector
        plant_flat = state_t[:, :G].long()          # (B, 45) int indices (copy — needed for Embedding)
        zombie_flat = state_t[:, G:2 * G]            # (B, 45) already float32, no copy needed
        extra = state_t[:, 2 * G:]                   # (B, 5) sun + avail (view)

        # Plant: embed → (B, embed_dim, R, C)
        p = self.plant_embed(plant_flat)           # (B, 45, embed_dim)
        p = p.permute(0, 2, 1).reshape(B, -1, R, C)  # (B, embed_dim, R, C)

        # Zombie: reshape → (B, 1, R, C)
        z = zombie_flat.view(B, 1, R, C)

        # 3×3 conv features
        pf3 = self.plant_conv3(p).reshape(B, -1)   # (B, 1440)
        zf3 = self.zombie_conv3(z).reshape(B, -1)  # (B, 1440)

        # 1×9 row conv features
        pfr = self.plant_conv_row(p).reshape(B, -1)  # (B, 80)
        zfr = self.zombie_conv_row(z).reshape(B, -1)  # (B, 80)

        out = torch.cat([pf3, zf3, pfr, zfr, extra], dim=1)

        if squeeze:
            out = out.squeeze(0)
        return out

    # ── Q-value interface (compatible with SimQNetwork) ─────────────────
    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        features = self._cnn_features(state_t)
        return self.fc(features)

    @torch.no_grad()
    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions[mask])
        return self.get_greedy_action(state, mask)

    @torch.no_grad()
    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def _get_mask(self, observation):
        """Reconstruct action mask from raw observation."""
        empty_cells = np.nonzero(
            (observation[:self._grid_size] == 0).reshape(self._rows, self._cols))
        mask = np.zeros(self.n_outputs, dtype=bool)
        mask[0] = True
        empty_cells_flat = (empty_cells[0] + self._rows * empty_cells[1]) * self._num_cards
        available_plants = observation[-self._num_cards:]
        for i in range(self._num_cards):
            if available_plants[i]:
                idx = empty_cells_flat + i + 1
                mask[idx] = True
        return mask


# ── Observation normalization ────────────────────────────────────────────
def transform_observation(observation):
    obs = observation.astype(np.float32)
    obs[45:90] /= HP_NORM    # no-op (HP_NORM=1)
    obs[90] /= SUN_NORM       # /200
    return obs


# ── DDQN loss ────────────────────────────────────────────────────────────
_mse_loss = nn.MSELoss()  # pre-created to avoid per-step module allocation


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
    return _mse_loss(qvals, expected_qvals)
