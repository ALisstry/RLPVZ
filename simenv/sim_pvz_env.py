import numpy as np
from gymnasium.spaces import Discrete
from simenv.config import REWARDS
from simenv.pvz_sim import (
    Scene, Move, config, WaveZombieSpawner,
    Sunflower, Peashooter, SnowPea, Repeater, Wallnut, Potatomine,
    Squash, CherryBomb, Spikeweed, KernelPult, MelonPult,
)

MAX_SUN = 9999.0
ZOMBIE_HP_NORM = 3000.0


CARD_SPECS = (
    ("sunflower", "Sunflower", 1, Sunflower),
    ("peashooter", "Peashooter", 0, Peashooter),
    ("snow-pea", "Snow Pea", 5, SnowPea),
    ("repeater", "Repeater", 7, Repeater),
    ("wall-nut", "Wall-nut", 3, Wallnut),
    ("squash", "Squash", 17, Squash),
    ("cherry-bomb", "Cherry Bomb", 2, CherryBomb),
    ("spikeweed", "Spikeweed", 21, Spikeweed),
    ("kernel-pult", "Kernel-pult", 34, KernelPult),
    ("melon-pult", "Melon-pult", 39, MelonPult),
)


class SimPVZEnv:
    """
    Simplified PVZ simulation environment with DDQN-compatible interface.

    Replaces both PVZEnv and DDQNEnvAdapter — directly outputs flat state
    vectors and provides mask_available_actions().

    State vector (95 dims for 5x9 grid with 4 plants):
      [plant_grid(45), zombie_hp_grid(45), plant_availability(4), sun_norm(1)]
    """

    def __init__(self):
        self.card_specs = CARD_SPECS
        self.plant_deck = {
            key: plant_cls
            for key, _, _, plant_cls in self.card_specs
            if plant_cls is not None
        }
        self.rows = config.N_LANES       # 5
        self.cols = config.LANE_LENGTH   # 9
        self.num_cards = len(self.card_specs)  # 10
        self.grid_size = self.rows * self.cols  # 45
        self.wait_action = self.num_cards * self.grid_size
        self.state_dim = (
            1
            + self.num_cards
            + self.grid_size * (self.num_cards + 1)
            + self.grid_size
            + self.grid_size
        )

        self.action_space = Discrete(self.wait_action + 1)  # 451
        self.action_space.n = self.action_space.n  # handy attribute

        self._plant_names = [key for key, _, _, _ in self.card_specs]
        self.card_plant_ids = [plant_id for _, _, plant_id, _ in self.card_specs]
        self._implemented_plant_names = list(self.plant_deck)
        self._plant_classes = [
            self.plant_deck[n].__name__ for n in self._implemented_plant_names]
        self._plant_no = {
            self._plant_classes[i]: self._plant_names.index(self._implemented_plant_names[i])
            for i in range(len(self._implemented_plant_names))}

        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._steps = 0
        self._last_mask = None
        self._collect_render = False
        self._render_data = []  # stored per-frame render info for last episode
        self.rewards = REWARDS
        self._last_sun = 0
        self._last_plants = {}
        self._last_wave_index = 0
        self._last_potential = 0.0
        self._reward_details = {}

    @property
    def steps(self):
        return self._steps

    def enable_render_collection(self):
        self._collect_render = True

    def disable_render_collection(self):
        self._collect_render = False

    @property
    def render_data(self):
        return self._render_data

    def reset(self, **kwargs):
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._steps = 0
        self._last_mask = self.mask_available_actions()
        self._last_sun = self._scene.sun
        self._last_plants = self._snapshot_plants()
        self._last_wave_index = self._current_wave_index()
        self._last_potential = self._calculate_potential()
        self._reward_details = {}
        if self._collect_render:
            self._render_data = [self._capture_frame()]
        return self._build_state()

    def step(self, action):
        # Execute action
        action_success, planted_name = self._take_action(action)
        self._scene.killed_zombies = []
        self._scene.escaped_zombies = []

        # Advance simulation until player can act or game ends
        self._scene.step()
        if self._collect_render:
            self._render_data.append(self._capture_frame())
        episode_over = self._scene._chrono > config.MAX_FRAMES
        while (not self._scene.move_available()) and (not episode_over):
            self._scene.step()
            if self._collect_render:
                self._render_data.append(self._capture_frame())
            episode_over = self._scene._chrono > config.MAX_FRAMES

        episode_over = episode_over or (self._scene.lives <= 0)
        reward, details = self._calculate_reward(
            action,
            action_success,
            planted_name,
            episode_over,
        )
        state = self._build_state()
        self._last_mask = self.mask_available_actions()
        self._steps += 1
        self._reward_details = details
        info = self._build_info(episode_over, details)
        return state, float(reward), bool(episode_over), info

    def mask_available_actions(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[self.wait_action] = True
        empty_cells, available_plants = self._scene.get_available_moves()
        if len(empty_cells[0]) == 0:
            return mask
        grid_indices = empty_cells[0] * self.cols + empty_cells[1]
        for plant in available_plants:
            card_idx = self._plant_no[plant.__name__]
            for row, col in zip(empty_cells[0], empty_cells[1]):
                mask[self.action_index(card_idx, row, col)] = True
        return mask

    def action_index(self, card_idx, row, col):
        return int(card_idx * self.grid_size + row * self.cols + col)

    def decode_action(self, action):
        if action == self.wait_action:
            return None
        if action < 0 or action >= self.wait_action:
            return None
        card_idx = action // self.grid_size
        grid_idx = action % self.grid_size
        row = grid_idx // self.cols
        col = grid_idx % self.cols
        return int(card_idx), int(row), int(col)

    def close(self):
        pass

    def _build_state(self):
        empty_unknown_class = self.num_cards
        plant_onehot = np.zeros(
            (self.grid_size, self.num_cards + 1), dtype=np.float32)
        plant_onehot[:, empty_unknown_class] = 1.0
        plant_hp = np.zeros(self.grid_size, dtype=np.float32)
        zombie_hp = np.zeros(self.grid_size, dtype=np.float32)

        for plant in self._scene.plants:
            idx = plant.lane * self.cols + plant.pos
            cls_idx = self._plant_no.get(plant.__class__.__name__, empty_unknown_class)
            plant_onehot[idx, :] = 0.0
            plant_onehot[idx, cls_idx] = 1.0
            max_hp = max(1.0, float(getattr(plant, "MAX_HP", 1)))
            plant_hp[idx] = max(0.0, min(1.0, float(plant.hp) / max_hp))

        for zombie in self._scene.zombies:
            idx = zombie.lane * self.cols + zombie.pos
            zombie_hp[idx] = min(
                1.0, zombie_hp[idx] + float(zombie.hp) / ZOMBIE_HP_NORM)

        cooldowns = np.ones(self.num_cards, dtype=np.float32)
        for i, name in enumerate(self._plant_names):
            if name not in self.plant_deck:
                continue
            plant_cls = self.plant_deck[name]
            full_cd = max(1.0, float(plant_cls.COOLDOWN * config.FPS - 1))
            cooldowns[i] = max(
                0.0, min(1.0, self._scene.plant_cooldowns[name] / full_cd))

        return np.concatenate(
            [
                np.array([min(float(self._scene.sun), MAX_SUN) / MAX_SUN],
                         dtype=np.float32),
                cooldowns,
                plant_onehot.reshape(-1),
                plant_hp,
                zombie_hp,
            ]
        ).astype(np.float32)

    def _take_action(self, action):
        if action == self.wait_action:
            return True, None
        if action < 0 or action >= self.action_space.n:
            return False, None
        decoded = self.decode_action(action)
        if decoded is None:
            return False, None
        plant_idx, lane, pos = decoded
        plant_name = self._plant_names[plant_idx]
        if plant_name not in self.plant_deck:
            return False, None
        move = Move(plant_name, lane, pos)
        if move.is_valid(self._scene):
            move.apply_move(self._scene)
            return True, plant_name
        return False, None

    def _calculate_reward(self, action, action_success, planted_name, episode_over):
        reward = 0.0
        details = {}
        current_plants = self._snapshot_plants()
        current_wave_index = self._current_wave_index()

        if not action_success:
            r_invalid = float(self.rewards.get("invalid_action", -0.01))
            reward += r_invalid
            details["invalid"] = r_invalid
        elif action == self.wait_action:
            threshold = self.rewards.get("wait_sun_threshold", 300)
            if self._last_sun >= threshold:
                r_wait = float(self.rewards.get("wait_with_sun", -0.02))
                reward += r_wait
                details["wait_with_sun"] = r_wait
        elif planted_name is not None:
            r_plant = self._plant_reward(planted_name)
            if r_plant:
                reward += r_plant
                details["plant"] = r_plant

        r_survival = float(self.rewards.get("survival_per_step", 0.0))
        if r_survival:
            reward += r_survival
            details["survival"] = r_survival

        sun_diff = self._scene.sun - self._last_sun
        if sun_diff > 0:
            r_sun = sun_diff * float(self.rewards.get("sun_collect", 0.01))
            reward += r_sun
            details["sun"] = r_sun

        killed = list(getattr(self._scene, "killed_zombies", []))
        if killed:
            r_kill = sum(self._zombie_kill_reward(z) for z in killed)
            reward += r_kill
            details["kill"] = r_kill
            flag_kills = sum(1 for z in killed if z["name"] == "Zombie_flag")
            if flag_kills:
                r_wave = flag_kills * float(self.rewards.get("wave_complete", 4.0))
                reward += r_wave
                details["wave"] = r_wave

        lost_plants = [
            plant for entity_id, plant in self._last_plants.items()
            if entity_id not in current_plants
        ]
        if lost_plants:
            r_lost = len(lost_plants) * float(self.rewards.get("plant_lost", -0.25))
            reward += r_lost
            details["plant_lost"] = r_lost
            sunflower_lost = sum(
                1 for plant in lost_plants
                if plant["name"] == "Sunflower"
            )
            if sunflower_lost:
                r_sf = sunflower_lost * float(
                    self.rewards.get("sunflower_lost", -0.80)
                )
                reward += r_sf
                details["sunflower_lost"] = r_sf

        potential = self._calculate_potential()
        delta = max(-5.0, min(5.0, potential - self._last_potential))
        delta_scale = float(
            self.rewards.get("potential", {}).get("delta_scale", 0.18)
        )
        r_potential = delta * delta_scale
        if abs(r_potential) > 1e-6:
            reward += r_potential
            details["potential_delta"] = r_potential

        if episode_over:
            if self._scene.lives <= 0:
                r_lose = float(self.rewards.get("game_lose", -12.0))
                reward += r_lose
                details["lose"] = r_lose
            else:
                r_win = float(self.rewards.get("game_win", 18.0))
                reward += r_win
                details["win"] = r_win

        self._last_sun = self._scene.sun
        self._last_plants = current_plants
        self._last_wave_index = current_wave_index
        self._last_potential = potential
        return reward, details

    def _plant_reward(self, plant_name):
        plant_cls = self.plant_deck.get(plant_name)
        if plant_cls is Sunflower:
            return float(self.rewards.get("plant_sunflower", 0.10))
        if plant_cls is Wallnut:
            return float(self.rewards.get("plant_wall", 0.18))
        if plant_cls in (
            Peashooter,
            SnowPea,
            Repeater,
            Squash,
            CherryBomb,
            Spikeweed,
            KernelPult,
            MelonPult,
        ):
            return float(self.rewards.get("plant_attacker", 0.35))
        return float(self.rewards.get("plant_other", 0.30))

    def _zombie_kill_reward(self, zombie):
        rewards = self.rewards.get("zombie_kill", {})
        default = float(rewards.get("default", 0.30))
        if not rewards.get("use_type_rewards", False):
            return default
        key_by_class = {
            "Zombie": "zombie",
            "Zombie_flag": "flag",
            "Zombie_cone": "conehead",
            "Zombie_bucket": "buckethead",
        }
        return float(rewards.get(key_by_class.get(zombie["name"], "zombie"), default))

    def _calculate_potential(self):
        cfg = self.rewards.get("potential", {})
        sun_cap = max(1.0, float(cfg.get("sun_cap", 300.0)))
        sun_potential = float(cfg.get("sun_scale", 0.06)) * (
            self._scene.sun / (self._scene.sun + sun_cap)
        )

        plant_potential = 0.0
        covered_rows = set()
        for plant in self._scene.plants:
            max_hp = max(1.0, float(getattr(plant, "MAX_HP", 1)))
            hp_ratio = max(0.0, min(1.0, float(plant.hp) / max_hp))
            base_value = self._plant_potential_value(plant)
            col_factor = 1.0 + 0.3 * (1.0 - plant.pos / max(1, self.cols - 1))
            plant_potential += base_value * hp_ratio * col_factor
            covered_rows.add(plant.lane)
        coverage = (
            len(covered_rows) / max(1, self.rows)
        ) * float(cfg.get("spread_bonus", 0.06))

        zombie_threat = 0.0
        for zombie in self._scene.zombies:
            hp_ratio = max(0.0, float(zombie.hp) / ZOMBIE_HP_NORM)
            distance = 1.0 - max(0.0, min(1.0, zombie.pos / max(1, self.cols - 1)))
            base_threat = 0.35 + distance * float(
                cfg.get("zombie_distance_bonus", 0.75)
            )
            zombie_threat += (
                float(cfg.get("zombie_threat_scale", 0.35))
                * base_threat
                * self._zombie_threat_multiplier(zombie)
                * hp_ratio
            )

        wave_potential = self._current_wave_index() * float(cfg.get("wave_scale", 0.05))
        return (
            sun_potential
            + plant_potential * float(cfg.get("plant_scale", 0.22))
            + coverage
            + wave_potential
            - zombie_threat
        )

    def _plant_potential_value(self, plant):
        if isinstance(plant, Sunflower):
            return 0.45
        if isinstance(plant, Wallnut):
            return 0.55
        if isinstance(plant, (SnowPea, KernelPult)):
            return 0.65
        if isinstance(plant, (Repeater, MelonPult)):
            return 0.80
        if isinstance(plant, (Squash, CherryBomb, Spikeweed)):
            return 0.45
        return 0.50

    def _zombie_threat_multiplier(self, zombie):
        name = zombie.__class__.__name__
        if name == "Zombie_bucket":
            return 1.8
        if name == "Zombie_cone":
            return 1.4
        if name == "Zombie_flag":
            return 1.1
        return 1.0

    def _snapshot_plants(self):
        return {
            plant.entity_id: {
                "name": plant.__class__.__name__,
                "lane": plant.lane,
                "pos": plant.pos,
                "hp": plant.hp,
            }
            for plant in self._scene.plants
        }

    def _snapshot_zombies(self):
        return {
            zombie.entity_id: {
                "name": zombie.__class__.__name__,
                "lane": zombie.lane,
                "pos": zombie.pos,
                "hp": zombie.hp,
            }
            for zombie in self._scene.zombies
        }

    def _current_wave_index(self):
        return int(getattr(self._scene._zombie_spawner, "wave_index", 0))

    def _build_info(self, episode_over, reward_details):
        current_wave = self._current_wave_index()
        spawner = self._scene._zombie_spawner
        return {
            "steps": min(config.MAX_FRAMES, self._scene._chrono),
            "win": bool(episode_over and self._scene.lives > 0),
            "game_ended": bool(episode_over),
            "completed_sublevels": int(
                getattr(spawner, "completed_sublevels", 0)
            ),
            "current_wave_index": current_wave,
            "is_flag_wave": bool(getattr(spawner, "last_wave_was_flag", False)),
            "zombie_count": len(self._scene.zombies),
            "plant_count": len(self._scene.plants),
            "sun": self._scene.sun,
            "lives": self._scene.lives,
            "reward_details": dict(reward_details),
        }

    def _capture_frame(self):
        """Capture current scene state for later visualization."""
        zombies = [[] for _ in range(config.N_LANES)]
        plants = [[] for _ in range(config.N_LANES)]
        projectiles = [[] for _ in range(config.N_LANES)]
        for z in self._scene.zombies:
            zombies[z.lane].append((z.__class__.__name__, int(z.pos), z.get_offset(), z.hp))
        for p in self._scene.plants:
            plants[p.lane].append((p.__class__.__name__, p.pos, p.hp))
        for proj in self._scene.projectiles:
            if hasattr(proj, '_render') and proj._render():
                offset = getattr(proj, '_offset', 0)
                pos = getattr(proj, '_pos', proj.pos if hasattr(proj, 'pos') else 0)
                projectiles[proj.lane].append((proj.__class__.__name__, int(pos), float(offset)))
        return {
            "zombies": zombies,
            "plants": plants,
            "projectiles": projectiles,
            "sun": self._scene.sun,
            "score": self._scene.score,
            "cooldowns": {n: int(self._scene.plant_cooldowns[n] / config.FPS) + 1
                          for n in self._plant_names},
            "time": int(self._scene._chrono / config.FPS),
            "lives": self._scene.lives,
        }
