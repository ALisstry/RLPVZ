from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from .ddqn import copy_state_dict_to_cpu

_mseloss = nn.MSELoss()


class DDQNLearner:
    def __init__(self, network, batch_size: int, gamma: float):
        self.network = network
        self.target_network = deepcopy(network)
        self.batch_size = batch_size
        self.gamma = gamma

    def state_dict_cpu(self):
        return copy_state_dict_to_cpu(self.network.state_dict())

    def sync_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        return self.state_dict_cpu()

    def calculate_loss(self, batch):
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

        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t)

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
        return _mseloss(qvals, expected_qvals)

    def update(self, replay_buffer):
        self.network.optimizer.zero_grad(set_to_none=True)
        batch = replay_buffer.sample_batch(batch_size=self.batch_size)

        try:
            loss = self.calculate_loss(batch)
            loss.backward()
            self.network.optimizer.step()
            return (
                float(loss.detach().cpu().item())
                if self.network.device == "cuda"
                else float(loss.detach().item())
            )
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
