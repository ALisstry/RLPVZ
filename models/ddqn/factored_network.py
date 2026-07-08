"""3-Factor Action-Space Q-Network for DDQN.

Decomposes the 451-action space into independent factors:

    Q(card, row, col, wait) = q_card[card] + q_row[row] + q_col[col] + q_wait

where:
    q_card ∈ R^{10}   — card-type preference   (Sunflower, Peashooter, …)
    q_row  ∈ R^{5}    — row preference          (back row vs front row)
    q_col  ∈ R^{9}    — column preference       (left vs right)
    q_wait ∈ R^{1}    — "do nothing" value

The full 451-dim Q-vector is reconstructed by **explicitly enumerating**
all 450 (= 10 × 5 × 9) plant-action combinations at query time:

    for i in 0..9, j in 0..4, k in 0..8:
        Q[plant_action(i,j,k)] = q_card[i] + q_row[j] + q_col[k]
    Q[wait] = q_wait

This costs B×450 additions (negligible) but guarantees that action-mask
filtering and argmax operate on the correct, mask-aware Q-values.

Output: 10 + 5 + 9 + 1 = 25 dims  (vs 451 for flat MLP, ~18× compression).
"""

import numpy as np
import torch
import torch.nn as nn


class FactoredQNetwork(nn.Module):
    """Q-Network with 3-factor action-space decomposition.

    Parameters
    ----------
    env : DDQNSpaceSpec or PVZEnv
        Must have ``rows``, ``cols``, ``num_cards``, ``action_space.n``.
    learning_rate : float
        Adam learning rate.
    device : str
        ``"cpu"`` or ``"cuda"``.
    hidden_sizes : list[int] | None
        Hidden layer sizes for the shared trunk, e.g. ``[2048, 2048]``.
    n_inputs_override : int | None
        Override the auto-computed observation dimension.
    create_optimizer : bool
        If False, skip Adam creation (worker / eval use).
    """

    def __init__(self, env, learning_rate=1e-3, device="cpu",
                 hidden_sizes=None, n_inputs_override=None,
                 create_optimizer=True):
        super().__init__()
        self.device = device
        self._use_factored = True  # marker for _print_network_summary

        self.rows = env.rows                     # 5
        self.cols = env.cols                     # 9
        self.num_cards = env.num_cards           # 10
        self.n_cells = self.rows * self.cols     # 45
        self.n_plant_actions = self.num_cards * self.n_cells  # 450
        self.n_outputs = env.action_space.n      # 451  (= 450 + wait)
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        if n_inputs_override is not None:
            self.n_inputs = int(n_inputs_override)
        else:
            # Correct fallback: typed-onehot = 1(sun) + num_cards(cooldown)
            #   + n_cells*(num_cards+1)(plant onehot) + 2*n_cells(plantHP+zombieHP)
            self.n_inputs = (1 + self.num_cards
                             + self.n_cells * (self.num_cards + 1)
                             + 2 * self.n_cells)

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        # ── Shared trunk ──
        trunk_layers = []
        prev = self.n_inputs
        for h in hidden_sizes:
            trunk_layers.append(nn.Linear(prev, h, bias=True))
            trunk_layers.append(nn.LeakyReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        trunk_out = prev if hidden_sizes else self.n_inputs

        # ── Factored heads (direct Linear, no activation → free-range Q) ──
        self.head_card = nn.Linear(trunk_out, self.num_cards, bias=True)   # 10
        self.head_row  = nn.Linear(trunk_out, self.rows, bias=True)        #  5
        self.head_col  = nn.Linear(trunk_out, self.cols, bias=True)        #  9
        self.head_wait = nn.Linear(trunk_out, 1, bias=True)                #  1

        if self.device == "cuda":
            self.to("cuda")

        self.optimizer = None
        if create_optimizer:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
            )

    # ── DDQN interface ──────────────────────────────────────────────────

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

    # ── Core: factored forward + explicit enumeration ───────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return raw factor logits: (q_card, q_row, q_col, q_wait)."""
        feat = self.trunk(x)                     # (B, trunk_out)
        return (
            self.head_card(feat),                # (B, 10)
            self.head_row(feat),                 # (B,  5)
            self.head_col(feat),                 # (B,  9)
            self.head_wait(feat),                # (B,  1)
        )

    @staticmethod
    def _enumerate_plant_q(q_card: torch.Tensor,
                           q_row: torch.Tensor,
                           q_col: torch.Tensor) -> torch.Tensor:
        """Explicitly build all 450 plant-action Q-values.

        For each (card_i, row_j, col_k):
            Q[i,j,k] = q_card[:,i] + q_row[:,j] + q_col[:,k]

        Args:
            q_card: (B, 10)
            q_row:  (B,  5)
            q_col:  (B,  9)

        Returns:
            q_plant: (B, 450)  — actions are ordered card-major:
                card0_row0_col0, card0_row0_col1, …, card9_row4_col8
        """
        # Broadcasting: (B, 10, 1, 1) + (B, 1, 5, 1) + (B, 1, 1, 9) = (B, 10, 5, 9)
        q_3d = (
              q_card.unsqueeze(-1).unsqueeze(-1)
            + q_row.unsqueeze(1).unsqueeze(-1)
            + q_col.unsqueeze(1).unsqueeze(1)
        )
        # Flatten card-major: card0→(row0,col0..col8), card0→(row1,col0..col8), …
        # reshape keeps the last 3 dims contiguous in the order they appear
        bsz = q_3d.shape[0]
        return q_3d.reshape(bsz, -1)            # (B, 450)

    def get_qvals(self, state):
        """Return full (…, 451) Q-values by enumerating all 450 plant combos."""
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        single = state_t.dim() == 1
        if single:
            state_t = state_t.unsqueeze(0)            # (D,) → (1, D)

        q_card, q_row, q_col, q_wait = self.forward(state_t)

        q_plant = self._enumerate_plant_q(q_card, q_row, q_col)  # (B, 450)
        qvals = torch.cat([q_plant, q_wait], dim=-1)             # (B, 451)

        if single:
            qvals = qvals.squeeze(0)                             # (451,)
        return qvals
