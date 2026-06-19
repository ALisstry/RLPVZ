import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple


class QNetwork(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None, n_inputs_override=None):
        """Deep Q-Network with configurable hidden layers.

        Args:
            env: Environment-like object with .rows, .cols, .num_cards, .action_space.n.
            learning_rate: Adam learning rate.
            device: "cpu" or "cuda".
            hidden_sizes: List of hidden layer sizes, e.g. [2048, 2048].
                Default [256, 128] for backward compatibility.
            n_inputs_override: Override the auto-computed n_inputs.
                Used when the adapter produces a different observation size.
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

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
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


class experienceReplayBuffer:
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


def copy_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu() for key, value in state_dict.items()}
