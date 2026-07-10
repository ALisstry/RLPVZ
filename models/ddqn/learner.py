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

    def calculate_loss(self, batch, is_weights=None, target_shift=0.0):
        """Double-DQN loss, optionally with PER importance-sampling weights.

        Args:
            batch: tuple of (states, actions, rewards, dones, next_states, masks, next_masks)
            is_weights: optional PER importance-sampling weights, shape (B, 1)
            target_shift: optional scalar added to expected_qvals for gradient-flow check.
                Default 0.0 (no shift).  Set to e.g. 1.0 to verify the network can chase
                a moving target.

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
            # ── Differential / wait-baseline statistics ──
            # Always meaningful: Q(s, wait) is the value of doing nothing,
            # Δ(s,a) = Q(s,a) - Q(s,wait) is how much better action a is.
            q_wait = all_qvals[:, -1]                              # (B,)
            delta_all = all_qvals - q_wait.unsqueeze(-1)           # (B, n_actions)
            # delta_all[:, -1] ≡ 0 by construction for DifferentialQNetwork
            # and ≈ 0 for other networks (Q(s,wait) - Q(s,wait) = 0)
            q_wait_mean = float(q_wait.mean().cpu().item())
            delta_mean = float(delta_all.mean().cpu().item())
            delta_max = float(delta_all.max(dim=1).values.mean().cpu().item())
        q_stats = {
            "mean_q": mean_q, "max_q": max_q,
            "entropy": entropy, "advantage": advantage,
            "q_wait": q_wait_mean, "delta_mean": delta_mean,
            "delta_max": delta_max,
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

        # Optional target shift for gradient-flow diagnostic
        if target_shift != 0.0:
            expected_qvals = expected_qvals + target_shift

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

    def overfit_test(self, batch, is_weights=None, n_iterations: int = 100,
                     log_prefix: str = "[Overfit]"):
        """单步过拟合诊断：对同一批数据反复训练，观察 loss 是否下降。

        用于验证网络 + 损失函数是否正常：
          - loss 下降到接近 0 → 算法实现正确，问题在数据分布/探索
          - loss 不下降或震荡 → 网络架构或损失计算存在 bug

        Phase 2 (gradient-flow check):
          收敛后将 target 偏移 +1.0，验证 Q 值能否追到新目标。
          能追上 → 计算图活着，梯度正常流动。
          追不上 → 权重被冻住或梯度断裂。

        Returns:
            losses: list of float, 每次迭代的 loss 值（仅 phase 1）
        """
        import time

        # =================================================================
        # Phase 1: 标准过拟合 — loss → 0
        # =================================================================
        losses = []
        start = time.perf_counter()
        for i in range(n_iterations):
            self.network.optimizer.zero_grad(set_to_none=True)
            loss, td_errors, q_stats = self.calculate_loss(batch, is_weights)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=10.0,
            )
            self.network.optimizer.step()
            loss_val = float(loss.detach().cpu().item())
            losses.append(loss_val)
            if i == 0 or i == n_iterations - 1 or (i + 1) % 20 == 0:
                print(
                    f"{log_prefix} iter {i + 1:3d}/{n_iterations} | "
                    f"loss={loss_val:.6f} | "
                    f"mean_q={q_stats['mean_q']:.4f} | "
                    f"max_q={q_stats['max_q']:.4f} | "
                    f"td_err_mean={float(td_errors.mean()):.6f} | "
                    f"grad_norm={float(total_norm):.4f}",
                    flush=True,
                )
        elapsed = time.perf_counter() - start
        loss_start = losses[0]
        loss_end = losses[-1]
        loss_min = min(losses)
        converged = loss_end < loss_start * 0.1 or loss_end < 0.01
        verdict = "CONVERGED OK" if converged else "FAILED TO CONVERGE - check network/loss!"
        print(
            f"{log_prefix} done | "
            f"loss: {loss_start:.6f} -> {loss_end:.6f} (min={loss_min:.6f}) | "
            f"{verdict} | "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )

        # =================================================================
        # Phase 2: 梯度流验证 — 偏移 target 看网络能否追上
        #
        # 关键：只检查被采取动作的 Q(s,a)，因为 loss = MSE(Q(s,a), target)。
        # mean_q 包含所有 451 个动作，不受 loss 直接约束。
        # =================================================================
        TARGET_SHIFT = 1.0
        N_SHIFT_ITERS = 30

        states, actions, _, _, _, _, _ = [item for item in batch]
        action_idx = actions[0] if isinstance(actions, (list, tuple)) else int(actions)

        # 保存收敛后该动作的 Q 值
        with torch.no_grad():
            all_q = self.network.get_qvals(states)
            if all_q.dim() == 1:
                q_before_action = float(all_q[action_idx].cpu().item())
            else:
                q_before_action = float(all_q[0, action_idx].cpu().item())

        print(
            f"\n{log_prefix} [GradFlow] shifting target by +{TARGET_SHIFT:.1f}, "
            f"Q(s, a={action_idx}) before = {q_before_action:.4f}",
            flush=True,
        )

        shift_losses = []
        for i in range(N_SHIFT_ITERS):
            self.network.optimizer.zero_grad(set_to_none=True)
            loss, td_errors, q_stats = self.calculate_loss(
                batch, is_weights, target_shift=TARGET_SHIFT,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=10.0,
            )
            self.network.optimizer.step()
            loss_val = float(loss.detach().cpu().item())
            shift_losses.append(loss_val)
            if i == 0 or i == N_SHIFT_ITERS - 1 or (i + 1) % 10 == 0:
                # Show Q(s,a) where 'a' is the taken action
                with torch.no_grad():
                    all_q = self.network.get_qvals(states)
                    if all_q.dim() == 1:
                        q_action = float(all_q[action_idx].cpu().item())
                    else:
                        q_action = float(all_q[0, action_idx].cpu().item())
                print(
                    f"{log_prefix} [GradFlow] iter {i + 1:3d}/{N_SHIFT_ITERS} | "
                    f"loss={loss_val:.6f} | "
                    f"Q(s,a={action_idx})={q_action:.4f} | "
                    f"td_err_mean={float(td_errors.mean()):.6f}",
                    flush=True,
                )

        with torch.no_grad():
            all_q = self.network.get_qvals(states)
            if all_q.dim() == 1:
                q_after_action = float(all_q[action_idx].cpu().item())
            else:
                q_after_action = float(all_q[0, action_idx].cpu().item())
        q_delta = q_after_action - q_before_action

        shift_start = shift_losses[0]
        shift_end = shift_losses[-1]
        graph_alive = (
            shift_end < shift_start * 0.5
            and abs(q_delta - TARGET_SHIFT) < 0.3
        )
        gv = "GRAPH ALIVE - gradients flowing correctly" if graph_alive else \
            "GRAPH DEAD? - Q(s,a) did not chase the shifted target, check gradients!"
        print(
            f"{log_prefix} [GradFlow] done | "
            f"loss: {shift_start:.6f} -> {shift_end:.6f} | "
            f"Q(s,a={action_idx}): {q_before_action:.4f} -> {q_after_action:.4f} "
            f"(delta={q_delta:+.4f}, target=+{TARGET_SHIFT:.1f}) | "
            f"{gv}",
            flush=True,
        )

        return losses

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
