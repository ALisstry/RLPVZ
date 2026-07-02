import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple


class QNetwork(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None, n_inputs_override=None,
                 create_optimizer=True):
        """Deep Q-Network with configurable hidden layers.

        Args:
            env: Environment-like object with .rows, .cols, .num_cards, .action_space.n.
            learning_rate: Adam learning rate.
            device: "cpu" or "cuda".
            hidden_sizes: List of hidden layer sizes, e.g. [2048, 2048].
                Default [256, 128] for backward compatibility.
            n_inputs_override: Override the auto-computed n_inputs.
                Used when the adapter produces a different observation size.
            create_optimizer: If False, skip Adam creation (worker/inference only).
        """
        super().__init__()
        self.device = device

        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.grid_size = self.rows * self.cols

        if n_inputs_override is not None:
            self.n_inputs = int(n_inputs_override)
        else:
            self.n_inputs = self.grid_size * 2 + self.num_cards + 1

        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

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

        self.optimizer = None
        if create_optimizer:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
            )

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
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.network(state_t)


class SumTree:
    """Binary sum-tree for O(log N) priority-based sampling.

    Tree is stored as a flat numpy array.  Leaves begin at index ``capacity - 1``
    and each internal node stores the sum of its two children.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._ptr = 0       # next leaf to write
        self.n_entries = 0

    # ── properties ────────────────────────────────────────────────────
    def total(self) -> float:
        return float(self.tree[0])

    def max_priority(self) -> float:
        leaves = self.tree[self.capacity - 1 : self.capacity - 1 + self.n_entries]
        return float(np.max(leaves)) if self.n_entries > 0 else 1.0

    # ── ops ───────────────────────────────────────────────────────────
    def add(self, priority: float) -> int:
        """Append a leaf with *priority*, return its tree index."""
        idx = self._ptr + self.capacity - 1
        self.update(idx, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def update(self, idx: int, priority: float):
        """Set priority at tree index *idx* and propagate the delta."""
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get_leaf(self, value: float) -> int:
        """Find the leaf index whose cumulative range contains *value*."""
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
        """Sample *n* leaf indices proportional to their priorities."""
        indices = np.empty(n, dtype=np.int64)
        total = self.total()
        if total <= 0:
            # All priorities zero — fall back to uniform
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
    proportional to ``priority ** alpha`` via a :class:`SumTree`.
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
        self.replay_memory = [None] * memory_size   # ring buffer (indexed)
        self.sum_tree = SumTree(memory_size)
        self._write_ptr = 0

    # ── properties ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self.sum_tree.n_entries

    def burn_in_capacity(self) -> float:
        return self.sum_tree.n_entries / max(1, self.burn_in)

    # ── store ─────────────────────────────────────────────────────────
    def append(self, state, action, reward, done, next_state, mask, next_mask):
        transition = self.Buffer(
            state, action, reward, done, next_state, mask, next_mask,
        )
        self.replay_memory[self._write_ptr] = transition

        # New transitions get max priority (or 1.0 if tree is still filling)
        max_p = self.sum_tree.max_priority()
        priority = max(max_p, 1.0)
        self.sum_tree.add(priority ** self.alpha)

        self._write_ptr = (self._write_ptr + 1) % self.memory_size

    # ── sample ────────────────────────────────────────────────────────
    def sample_batch(self, batch_size=32, beta: float = 0.4):
        """Sample a batch with importance-sampling weights.

        Returns:
            (batch, tree_indices, is_weights) where *batch* is the usual
            tuple-of-arrays, *tree_indices* are SumTree leaf positions,
            and *is_weights* is a ``(batch_size, 1)`` float32 array for
            loss correction.
        """
        tree_indices = self.sum_tree.sample(batch_size)

        # Convert tree index → data index
        data_indices = tree_indices - (self.sum_tree.capacity - 1)

        # Importance-sampling weights
        probs = self.sum_tree.tree[tree_indices] / max(self.sum_tree.total(), 1e-12)
        n = self.sum_tree.n_entries
        is_weights = (n * probs) ** (-beta)
        is_weights /= max(is_weights.max(), 1e-12)   # normalise so max = 1

        # Gather batch
        entries = [self.replay_memory[i] for i in data_indices]
        batch = tuple(zip(*entries))
        return batch, tree_indices, is_weights.astype(np.float32).reshape(-1, 1)

    # ── update ────────────────────────────────────────────────────────
    def update_priorities(self, tree_indices, td_errors):
        """Set new priorities from (absolute) TD-errors."""
        for idx, td_err in zip(tree_indices, td_errors):
            priority = float(abs(td_err)) + self.epsilon
            self.sum_tree.update(int(idx), priority ** self.alpha)


# Legacy alias kept for backward-compatible imports
experienceReplayBuffer = PrioritizedReplayBuffer


def copy_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu() for key, value in state_dict.items()}
