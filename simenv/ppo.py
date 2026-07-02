"""Maskable PPO (Proximal Policy Optimization) for SimPVZ environment.

Key features:
- Actor-Critic with shared feature extractor (supports mlp / deepmlp / cnn)
- **Action masking**: invalid-action logits set to -inf before softmax
- GAE (Generalized Advantage Estimation) for stable advantage computation
- Clipped surrogate objective + value clipping + entropy bonus
"""

import gc
import os
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
    """Actor-Critic with shared feature extractor and maskable policy head."""

    def __init__(self, env, network_type="cnn", device="cpu"):
        super().__init__()
        self.device = device
        self._rows = config.N_LANES               # 5
        self._cols = config.LANE_LENGTH            # 9
        self._grid_size = self._rows * self._cols  # 45
        self.n_outputs = env.action_space.n        # 181
        self.actions = np.arange(self.n_outputs)
        self._num_cards = len(env.plant_deck)      # 4
        self._network_type = network_type

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
        """Small MLP: 55→128→64 shared trunk."""
        n_inputs = self._grid_size + self._rows + 1 + self._num_cards  # 55

        self.shared = nn.Sequential(
            nn.Linear(n_inputs, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, self.n_outputs)
        self.value_head = nn.Linear(64, 1)

    def _build_deep_mlp(self):
        """Deep MLP: 55→512→256→128 shared trunk (~848K params)."""
        n_inputs = self._grid_size + self._rows + 1 + self._num_cards  # 55

        self.shared = nn.Sequential(
            nn.Linear(n_inputs, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, self.n_outputs)
        self.value_head = nn.Linear(128, 1)

    # ── CNN architecture ───────────────────────────────────────────────

    def _build_cnn(self):
        """CNN feature extractor with dual-kernel (3×3 + 1×9 row)."""
        embed_dim = 8
        self._num_plant_types = self._num_cards + 1  # 0=empty + 4 plants

        self.plant_embed = nn.Embedding(self._num_plant_types, embed_dim)

        self.plant_conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.zombie_conv3 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.plant_conv_row = nn.Sequential(
            nn.Conv2d(embed_dim, 16, (1, self._cols)), nn.ReLU(),
        )
        self.zombie_conv_row = nn.Sequential(
            nn.Conv2d(1, 16, (1, self._cols)), nn.ReLU(),
        )

        cnn_dim = 2 * (32 * self._rows * self._cols + 16 * self._rows)  # 3040
        extra_dim = 1 + self._num_cards                                 # 5

        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_dim + extra_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, self.n_outputs)
        self.value_head = nn.Linear(128, 1)

    # ── Feature extraction ─────────────────────────────────────────────

    def _extract_features(self, state_t):
        """Extract features.  Handles (B, 95) and (95,) tensors."""
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if self._network_type in ("mlp", "deepmlp"):
            plant_grid = state_t[:, :self._grid_size]
            zombie_grid = state_t[:, self._grid_size:2 * self._grid_size]
            zombie_by_lane = torch.sum(
                zombie_grid.reshape(-1, self._rows, self._cols), dim=-1)
            state_t = torch.cat(
                [plant_grid, zombie_by_lane, state_t[:, 2 * self._grid_size:]], dim=1)
            features = self.shared(state_t)
        else:  # cnn
            B = state_t.shape[0]
            R, C, G = self._rows, self._cols, self._grid_size

            plant_flat = state_t[:, :G].long()
            zombie_flat = state_t[:, G:2 * G]
            extra = state_t[:, 2 * G:]

            p = self.plant_embed(plant_flat)
            p = p.permute(0, 2, 1).reshape(B, -1, R, C)
            z = zombie_flat.view(B, 1, R, C)

            pf3 = self.plant_conv3(p).reshape(B, -1)
            zf3 = self.zombie_conv3(z).reshape(B, -1)
            pfr = self.plant_conv_row(p).reshape(B, -1)
            zfr = self.zombie_conv_row(z).reshape(B, -1)

            features = self.shared_fc(
                torch.cat([pf3, zf3, pfr, zfr, extra], dim=1))

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
        save_path = _default_save_path("ppo", "sim_ppo.pt")

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
    episode_reward = 0.0
    update_count = 0

    while ep < max_episodes:
        # Phase 1: Rollout — collect on-policy trajectories
        buffer.reset()
        episode_reward = 0.0

        while not buffer.is_full() and ep < max_episodes:
            mask = env.mask_available_actions()
            action, log_prob, value = network.get_action(state, mask)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_observation(next_state)

            buffer.add(state, action, reward, done, log_prob, value, mask)
            episode_reward += reward
            total_steps_done += 1

            state = next_state
            if done:
                ep += 1
                state = transform_observation(env.reset())
                episode_rewards.append(episode_reward)
                episode_reward = 0.0

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

    # ═══════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(network.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    np.save(save_path.replace(".pt", "_rewards.npy"), np.array(episode_rewards))
    print("Training complete.")

    # Evaluation
    _evaluate_ppo(env, network, n_episodes=eval_episodes)

    # Visualize
    _visualize_ppo_episode(env, network)


def _default_save_path(algo, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("saved", algo, timestamp, filename)


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
