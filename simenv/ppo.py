"""Maskable PPO (Proximal Policy Optimization) for SimPVZ environment.

Key features:
- Actor-Critic with shared feature extractor (supports mlp / deepmlp / cnn)
- **Action masking**: invalid-action logits set to -inf before softmax
- GAE (Generalized Advantage Estimation) for stable advantage computation
- Clipped surrogate objective + value clipping + entropy bonus
"""

import gc
import os
import signal
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

from simenv import SimPVZEnv
from simenv.pvz_sim import config
from simenv.model import transform_observation


# ═══════════════════════════════════════════════════════════════════════════
# PPO Actor-Critic Network
# ═══════════════════════════════════════════════════════════════════════════

class PPONetwork(nn.Module):
    """Actor-Critic with shared feature extractor and maskable policy head.

    Uses the same 596-dim typed-onehot observation and 451-action space
    as the DDQN implementation for fair comparison.
    """

    def __init__(self, env, network_type="cnn", device="cpu"):
        super().__init__()
        self.device = device
        self._rows = config.N_LANES               # 5
        self._cols = config.LANE_LENGTH            # 9
        self._grid_size = self._rows * self._cols  # 45
        self.n_outputs = env.action_space.n        # 451
        self.actions = np.arange(self.n_outputs)
        self._num_cards = env.num_cards            # 10
        self._n_inputs = int(env.state_dim)        # 596
        self._network_type = network_type

        # One-hot grid channels: num_cards+1 (plant types) + 1 (plant HP) + 1 (zombie HP)
        self._n_grid_channels = self._num_cards + 3  # 13

        if network_type == "cnn":
            self._build_cnn()
        elif network_type == "deepmlp":
            self._build_deep_mlp()
        else:
            self._build_mlp()

        if device == "cuda":
            self.cuda()

    # ── MLP architectures ──────────────────────────────────────────────

    def _build_mlp(self):
        """Small MLP: 596→128→64 shared trunk."""
        self.shared = nn.Sequential(
            nn.Linear(self._n_inputs, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, self.n_outputs)
        self.value_head = nn.Linear(64, 1)

    def _build_deep_mlp(self):
        """Deep MLP: 596→2048→2048 shared trunk (same as DDQN default)."""
        self.shared = nn.Sequential(
            nn.Linear(self._n_inputs, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
        )
        self.policy_head = nn.Linear(2048, self.n_outputs)
        self.value_head = nn.Linear(2048, 1)

    # ── CNN architecture ───────────────────────────────────────────────

    def _build_cnn(self):
        """CNN feature extractor with dual-kernel (3×3 + 1×9 row).

        Parses the 596-dim typed-onehot observation into 13-channel grid
        tensors, mirroring the DDQN CNNQNetwork input layout.
        """
        embed_dim = 32

        self.plant_embed = nn.Sequential(
            nn.Linear(self._num_cards + 1, embed_dim, bias=False),
            nn.ReLU(),
        )

        self.plant_conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.hp_conv3 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.plant_conv_row = nn.Sequential(
            nn.Conv2d(embed_dim, 32, (1, self._cols)), nn.ReLU(),
        )
        self.hp_conv_row = nn.Sequential(
            nn.Conv2d(2, 16, (1, self._cols)), nn.ReLU(),
        )

        cnn_dim = 64 * self._rows * self._cols + 32 * self._rows * self._cols \
                + 32 * self._rows + 16 * self._rows  # 4560
        extra_dim = 1 + self._num_cards               # 11  (sun + cooldowns)

        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_dim + extra_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, self.n_outputs)
        self.value_head = nn.Linear(128, 1)

    # ── Observation parsing ────────────────────────────────────────────

    @staticmethod
    def _parse_596_to_grid(state_t: torch.Tensor, rows: int, cols: int,
                           num_cards: int, n_grid_channels: int):
        """Parse 596-dim typed-onehot into grid tensor + global features.

        596 = sun(1) + cooldowns(num_cards) + plant_onehot(rows*cols*(num_cards+1))
              + plantHP(rows*cols) + zombieHP(rows*cols)

        Returns:
            grid: (B, n_grid_channels, rows, cols)  — 13 channels
            glob_: (B, 1 + num_cards)  — sun + cooldowns
        """
        bsz = state_t.shape[0]
        n_cells = rows * cols
        n_onehot = num_cards + 1

        glob_ = state_t[:, :1 + num_cards]               # (B, 11)
        grid_flat = state_t[:, 1 + num_cards:]            # (B, 585)

        split_1 = n_cells * n_onehot                      # 495
        split_2 = split_1 + n_cells                       # 540

        onehot = grid_flat[:, :split_1].view(bsz, n_cells, n_onehot)
        plant_hp = grid_flat[:, split_1:split_2].view(bsz, n_cells, 1)
        zombie_hp = grid_flat[:, split_2:].view(bsz, n_cells, 1)

        grid = torch.cat([onehot, plant_hp, zombie_hp], dim=-1)  # (B, 45, 13)
        grid = grid.view(bsz, rows, cols, n_grid_channels)
        grid = grid.permute(0, 3, 1, 2).contiguous()      # (B, 13, rows, cols)
        return grid, glob_

    # ── Feature extraction ─────────────────────────────────────────────

    def _extract_features(self, state_t):
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if self._network_type in ("mlp", "deepmlp"):
            features = self.shared(state_t)
        else:  # cnn
            B = state_t.shape[0]
            grid, glob_ = self._parse_596_to_grid(
                state_t, self._rows, self._cols, self._num_cards,
                self._n_grid_channels)

            # Split grid channels: onehot(11) + plantHP(1) + zombieHP(1)
            onehot_grid = grid[:, :self._num_cards + 1, :, :]  # (B, 11, R, C)
            hp_grid     = grid[:, self._num_cards + 1:, :, :]  # (B, 2, R, C)

            p_embed = self.plant_embed(
                onehot_grid.permute(0, 2, 3, 1).reshape(-1, self._num_cards + 1)
            ).reshape(B, self._rows, self._cols, -1).permute(0, 3, 1, 2)

            pf3 = self.plant_conv3(p_embed).reshape(B, -1)
            zf3 = self.hp_conv3(hp_grid).reshape(B, -1)
            pfr = self.plant_conv_row(p_embed).reshape(B, -1)
            zfr = self.hp_conv_row(hp_grid).reshape(B, -1)

            features = self.shared_fc(
                torch.cat([pf3, zf3, pfr, zfr, glob_], dim=1))

        if squeeze:
            features = features.squeeze(0)
        return features

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, state, action_mask=None):
        """Return (masked_logits, values)."""
        if isinstance(state, (list, tuple)):
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(self.device)
        features = self._extract_features(state_t)

        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)

        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            logits = logits.clone()
            logits[~mask_t] = float("-inf")

        return logits, values

    @torch.no_grad()
    def get_action(self, state, action_mask):
        """Sample action.  Returns (action, log_prob, value)."""
        logits, value = self.forward(state, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate(self, state, action, action_mask):
        """Evaluate actions for PPO update.  Returns (log_probs, entropy, values)."""
        logits, values = self.forward(state, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return log_probs, entropy, values


# ═══════════════════════════════════════════════════════════════════════════
# Rollout Buffer (on-policy)
# ═══════════════════════════════════════════════════════════════════════════

class PPORolloutBuffer:
    """Stores on-policy trajectories for one PPO update cycle."""

    def __init__(self, horizon):
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.masks = []

    def add(self, state, action, reward, done, log_prob, value, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(mask)

    def is_full(self):
        return len(self.states) >= self.horizon

    def get_data(self):
        """Return all stored data as numpy arrays."""
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=bool),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.masks, dtype=bool),
        )


# ═══════════════════════════════════════════════════════════════════════════
# PPO Training
# ═══════════════════════════════════════════════════════════════════════════

def train_ppo(
    max_episodes=100000,
    horizon=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=3e-4,
    network_type="cnn",
    save_path=None,
    eval_episodes=100,
    plot_callback=None,
    plot_freq=100,
):
    """Train a maskable PPO agent on SimPVZ.

    Parameters
    ----------
    max_episodes : int
        Total episodes to train for.
    horizon : int
        Steps collected per rollout before each PPO update.
    batch_size : int
        Mini-batch size for PPO updates.
    n_epochs : int
        Number of epochs per PPO update (passes over the rollout data).
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda parameter for advantage estimation.
    clip_epsilon : float
        PPO clipping range.
    vf_coef : float
        Value function loss coefficient.
    ent_coef : float
        Entropy bonus coefficient.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    lr : float
        Learning rate.
    network_type : str
        "mlp", "deepmlp", or "cnn".
    save_path : str
        Path to save the trained model.
    """
    if save_path is None:
        tag = f"ppo_{network_type}"
        save_path = _default_save_path("ppo", "sim_ppo.pt", tag=tag)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimPVZEnv()
    network = PPONetwork(env, network_type=network_type, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    n_params = sum(p.numel() for p in network.parameters())
    _print_ppo_config(locals())

    buffer = PPORolloutBuffer(horizon)
    mse_loss = nn.MSELoss()

    state = transform_observation(env.reset())
    ep = 0
    total_steps_done = 0
    episode_rewards = []
    episode_lengths = []
    episode_reward = 0.0
    episode_steps = 0
    update_count = 0
    update_losses = []
    update_policy_losses = []
    update_value_losses = []
    update_entropies = []

    # Periodic save + interrupt handler
    _save_freq = 10000
    _saved_on_interrupt = False
    _stop_requested = False
    _prev_sigint = signal.getsignal(signal.SIGINT)

    def _save_model():
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(network.state_dict(), save_path)
        np.save(save_path.replace(".pt", "_rewards.npy"), np.array(episode_rewards))
        print(f"[PPO] Saved model to {save_path} (ep={ep})", flush=True)

    def _handle_interrupt(signum, frame):
        nonlocal _saved_on_interrupt, _stop_requested
        _stop_requested = True
        signal.signal(signal.SIGINT, _prev_sigint)
        if not _saved_on_interrupt:
            _saved_on_interrupt = True
            print("\n[PPO] Interrupted, saving...", flush=True)
            _save_model()

    signal.signal(signal.SIGINT, _handle_interrupt)

    while ep < max_episodes and not _stop_requested:
        # Phase 1: Rollout — collect on-policy trajectories
        buffer.reset()
        episode_reward = 0.0

        while not buffer.is_full() and ep < max_episodes and not _stop_requested:
            mask = env.mask_available_actions()
            action, log_prob, value = network.get_action(state, mask)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_observation(next_state)

            buffer.add(state, action, reward, done, log_prob, value, mask)
            episode_reward += reward
            episode_steps += 1
            total_steps_done += 1

            state = next_state
            if done:
                ep += 1
                episode_lengths.append(episode_steps)
                state = transform_observation(env.reset())
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                episode_steps = 0

        # Flush partial episode reward
        if episode_reward > 0:
            episode_rewards.append(episode_reward)

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Compute GAE advantages and returns
        # ═══════════════════════════════════════════════════════════════
        (states, actions, rewards, dones,
         old_log_probs, old_values, masks) = buffer.get_data()

        n = len(states)  # may be < horizon if training ends early

        # Bootstrap from last state
        last_mask = env.mask_available_actions()
        with torch.no_grad():
            _, last_value = network(state, last_mask)
        last_value = last_value.item()

        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = old_values[t + 1]

            if dones[t]:
                next_value = 0.0
                # Also reset GAE at episode boundary
                gae = 0.0

            delta = rewards[t] + gamma * next_value - old_values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + old_values[t]

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ═══════════════════════════════════════════════════════════════
        # Phase 3: PPO update — K epochs of mini-batch SGD
        # ═══════════════════════════════════════════════════════════════
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(device)
        old_values_t = torch.FloatTensor(old_values).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        masks_t = torch.BoolTensor(masks).to(device)

        indices = np.arange(n)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]

                new_log_probs, entropy, new_values = network.evaluate(
                    states_t[batch_idx],
                    actions_t[batch_idx],
                    masks_t[batch_idx],
                )

                # Clipped policy loss
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = (torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                         * advantages_t[batch_idx])
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_pred = new_values
                value_target = returns_t[batch_idx]
                value_loss_unclipped = (value_pred - value_target) ** 2
                value_clipped = (old_values_t[batch_idx]
                                 + torch.clamp(value_pred - old_values_t[batch_idx],
                                               -clip_epsilon, clip_epsilon))
                value_loss_clipped = (value_clipped - value_target) ** 2
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus (we want to maximize entropy → negative in loss)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
                optimizer.step()

        update_count += 1
        update_losses.append(loss.item())
        update_policy_losses.append(policy_loss.item())
        update_value_losses.append(value_loss.item())
        update_entropies.append(-entropy_loss.item())

        # ═══════════════════════════════════════════════════════════════
        # Plot
        # ═══════════════════════════════════════════════════════════════
        if plot_callback is not None and ep % plot_freq == 0 and ep > 0:
            plot_callback(
                save_path,
                np.array(episode_rewards),
                np.array(episode_lengths),
                np.array(update_losses),
                advantage=np.array(update_policy_losses) if update_policy_losses else None,
                entropy=np.array(update_entropies) if update_entropies else None,
            )

        # ═══════════════════════════════════════════════════════════════
        # Logging
        # ═══════════════════════════════════════════════════════════════
        if update_count % 5 == 0 or ep >= max_episodes:
            gc.collect()
            recent = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            mean_r = np.mean(recent) if recent else 0.0
            print(f"Ep {ep:5d}/{max_episodes}  "
                  f"Steps {total_steps_done:7d}  "
                  f"Mean R {mean_r:8.2f}  "
                  f"Policy L {policy_loss.item():.4f}  "
                  f"Value L {value_loss.item():.4f}  "
                  f"Entropy {entropy.mean().item():.4f}")

        # Periodic save
        if ep > 0 and ep % _save_freq == 0:
            _save_model()

    signal.signal(signal.SIGINT, _prev_sigint)
    _save_model()

    print("Training complete.")

    # Evaluation
    _evaluate_ppo(env, network, n_episodes=eval_episodes)

    # Visualize
    _visualize_ppo_episode(env, network)


def _default_save_path(algo, filename, tag=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = timestamp if tag is None else f"{timestamp}_{tag}"
    return os.path.join("saved", algo, folder, filename)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_ppo(env, network, n_episodes=100):
    """Run N episodes with greedy policy and report statistics."""
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  PPO Evaluation ({n_episodes} episodes)")
    print(f"{sep}")

    rewards = []
    survivals = []
    actions_taken = []
    max_frames = config.MAX_FRAMES

    for _ in range(n_episodes):
        state = transform_observation(env.reset())
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            mask = env.mask_available_actions()
            action, _, _ = network.get_action(state, mask)  # already @torch.no_grad()
            state, reward, done, _ = env.step(action)
            state = transform_observation(state)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        survivals.append(min(max_frames, env._scene._chrono))
        actions_taken.append(steps)

    rewards = np.array(rewards)
    survivals = np.array(survivals)
    actions = np.array(actions_taken)

    fps = config.FPS
    print(f"  {'Reward:':20s} mean={rewards.mean():8.2f}  std={rewards.std():8.2f}  "
          f"min={rewards.min():8.2f}  max={rewards.max():8.2f}")
    print(f"  {'Survival (frames):':20s} mean={survivals.mean():8.2f}  std={survivals.std():8.2f}  "
          f"min={survivals.min():8.0f}  max={survivals.max():8.0f}")
    print(f"  {'Survival (sec):':20s} mean={survivals.mean() / fps:8.2f}  std={survivals.std() / fps:8.2f}  "
          f"min={survivals.min() / fps:8.2f}  max={survivals.max() / fps:8.2f}")
    print(f"  {'Actions taken:':20s} mean={actions.mean():8.2f}  std={actions.std():8.2f}  "
          f"min={actions.min():8.0f}  max={actions.max():8.0f}")

    survived_full = (survivals >= max_frames).sum()
    print(f"  {'Full survival:':20s} {survived_full}/{n_episodes} ({100 * survived_full / n_episodes:.1f}%)")
    print(f"{sep}\n")


def _print_ppo_config(loc):
    """Pretty-print PPO training configuration."""
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  PPO Training Configuration")
    print(f"{sep}")
    print(f"  {'Device:':28s} {loc['device'].upper()}")
    print(f"  {'Network:':28s} {loc['network_type']} ({loc['n_params']:,} params)")
    print(f"  {'Max episodes:':28s} {loc['max_episodes']}")
    print(f"  {'Horizon:':28s} {loc['horizon']}")
    print(f"  {'Batch size:':28s} {loc['batch_size']}")
    print(f"  {'Epochs per update:':28s} {loc['n_epochs']}")
    print(f"  {'Gamma:':28s} {loc['gamma']}")
    print(f"  {'GAE lambda:':28s} {loc['gae_lambda']}")
    print(f"  {'Clip epsilon:':28s} {loc['clip_epsilon']}")
    print(f"  {'Value coeff:':28s} {loc['vf_coef']}")
    print(f"  {'Entropy coeff:':28s} {loc['ent_coef']}")
    print(f"  {'Max grad norm:':28s} {loc['max_grad_norm']}")
    print(f"  {'Learning rate:':28s} {loc['lr']}")
    print(f"  {'Eval episodes:':28s} {loc['eval_episodes']}")
    print(f"  {'Grid:':28s} {config.N_LANES}x{config.LANE_LENGTH} (rows x cols)")
    print(f"{sep}\n")


def _visualize_ppo_episode(env, network):
    """Play one episode with trained PPO agent and show replay."""
    from simenv.render import replay_episode
    env.enable_render_collection()
    state = transform_observation(env.reset())
    done = False
    total_reward = 0.0
    while not done:
        mask = env.mask_available_actions()
        action, _, _ = network.get_action(state, mask)
        state, reward, done, _ = env.step(action)
        state = transform_observation(state)
        total_reward += reward
    env.disable_render_collection()
    print(f"\nReplay: {len(env.render_data)} frames, reward={total_reward:.0f}")
    replay_episode(env.render_data, fps=15,
                   title=f"SimPVZ PPO Agent - Reward: {total_reward:.0f}")
