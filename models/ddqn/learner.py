from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from .ddqn import copy_state_dict_to_cpu


class DDQNLearner:
    def __init__(self, network, batch_size: int, gamma: float):
        self.network = network
        self.target_network = deepcopy(network)
        self.target_network.eval()  # always use running stats (BatchNorm-safe)
        # Target network never trains — drop the copied optimizer to save memory
        if hasattr(self.target_network, 'optimizer'):
            self.target_network.optimizer = None
        self.batch_size = batch_size
        self.gamma = gamma

    def state_dict_cpu(self):
        return copy_state_dict_to_cpu(self.network.state_dict())

    def sync_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        return self.state_dict_cpu()

    def calculate_loss(self, batch, is_weights=None):
        """Double-DQN loss, optionally with PER importance-sampling weights.

        Returns:
            (loss, td_errors, q_stats) — *loss* is scalar, *td_errors* is
            (B, 1) numpy array, *q_stats* is a dict with mean_q, max_q, entropy.
        """
        states, actions, rewards, dones, next_states, masks, next_masks = [
            item for item in batch
        ]

        rewards_t = (
            torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1, 1)
        )
        actions_t = (
            torch.LongTensor(np.array(actions))
            .reshape(-1, 1)
            .to(device=self.network.device)
        )
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.network.device)

        all_qvals = self.network.get_qvals(states)                # (B, n_actions)
        qvals = torch.gather(all_qvals, 1, actions_t)             # (B, 1)

        # ── Q-value statistics (detached, no gradient) ──
        with torch.no_grad():
            mean_q = float(qvals.mean().cpu().item())
            max_q = float(all_qvals.max(dim=1).values.mean().cpu().item())
            # Entropy of softmax over Q-values — measures policy certainty
            probs = F.softmax(all_qvals, dim=1)                   # (B, n_actions)
            log_probs = torch.log(probs + 1e-12)
            entropy = float(-(probs * log_probs).sum(dim=1).mean().cpu().item())
            # Advantage: how much better the chosen action is vs mean
            advantage = float(
                (qvals - all_qvals.mean(dim=1, keepdim=True)).mean().cpu().item()
            )
        q_stats = {
            "mean_q": mean_q, "max_q": max_q,
            "entropy": entropy, "advantage": advantage,
        }

        next_masks = np.array(next_masks, dtype=bool)
        with torch.no_grad():
            qvals_next_pred = self.network.get_qvals(next_states)
            next_masks_t = torch.as_tensor(
                next_masks, dtype=torch.bool, device=qvals_next_pred.device
            )
            qvals_next_pred = qvals_next_pred.clone()
            qvals_next_pred[~next_masks_t] = qvals_next_pred.min()
            next_actions = torch.max(qvals_next_pred, dim=-1)[1]
            next_actions_t = torch.as_tensor(
                next_actions, dtype=torch.long, device=self.network.device
            ).reshape(-1, 1)

            target_qvals = self.target_network.get_qvals(next_states)
            qvals_next = torch.gather(target_qvals, 1, next_actions_t)
        qvals_next[dones_t] = 0
        expected_qvals = self.gamma * qvals_next + rewards_t

        # Element-wise TD errors for PER priority updates
        td_errors = (expected_qvals - qvals).detach()

        # PER importance-sampling correction
        elementwise_loss = F.mse_loss(qvals, expected_qvals, reduction='none')
        if is_weights is not None:
            is_weights_t = torch.as_tensor(
                is_weights, dtype=torch.float32, device=elementwise_loss.device,
            )
            loss = (elementwise_loss * is_weights_t).mean()
        else:
            loss = elementwise_loss.mean()

        return loss, td_errors.cpu().numpy(), q_stats

    def update(self, replay_buffer, beta: float = 0.4):
        """Single gradient step.  Returns ``(loss, tree_indices, td_errors, q_stats)``
        so the caller can update PER priorities, or ``None`` on OOM skip."""
        self.network.optimizer.zero_grad(set_to_none=True)
        batch, tree_indices, is_weights = replay_buffer.sample_batch(
            batch_size=self.batch_size, beta=beta,
        )

        try:
            loss, td_errors, q_stats = self.calculate_loss(batch, is_weights)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=10.0,
            )
            grad_norm = float(total_norm.item()) if torch.is_tensor(total_norm) else float(total_norm)
            self.network.optimizer.step()
            return (
                float(loss.detach().cpu().item())
                if self.network.device == "cuda"
                else float(loss.detach().item())
            ), tree_indices, td_errors, q_stats, grad_norm
        except (RuntimeError, torch.AcceleratorError) as exc:
            message = str(exc).lower()
            is_oom = "out of memory" in message or "cudaerrormemoryallocation" in message
            if not is_oom:
                raise

            self.network.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                "\n[DDQN] CUDA OOM，已清理缓存并跳过本次 update。"
                f" batch_size={self.batch_size}",
                flush=True,
            )
            return None
