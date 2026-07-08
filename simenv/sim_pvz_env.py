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
    ("potatomine", "Potato Mine", 4, Potatomine),
    ("wall-nut", "Wall-nut", 3, Wallnut),
    ("repeater", "Repeater", 7, Repeater),
    ("squash", "Squash", 17, Squash),
    ("cherry-bomb", "Cherry Bomb", 2, CherryBomb),
    ("snow-pea", "Snow Pea", 5, SnowPea),
    ("melon-pult", "Melon-pult", 39, MelonPult),
    ("spikeweed", "Spikeweed", 21, Spikeweed),
)


class SimPVZEnv:
    """
    Simplified PVZ simulation environment with DDQN-compatible interface.

    Replaces both PVZEnv and DDQNEnvAdapter — directly outputs flat state
    vectors and provides mask_available_actions().

    State vector (95 dims for 5x9 grid with 4 plants):
      [plant_grid(45), zombie_hp_grid(45), plant_availability(4), sun_norm(1)]
    """

    def __init__(self, stage=None):
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

        self._all_rows = tuple(range(self.rows))
        self._all_plant_ids = tuple(self.card_plant_ids)
        self.current_stage = None
        self.stage_name = "sim"
        self.enabled_rows = set(self._all_rows)
        self.enabled_plant_ids = set(self._all_plant_ids)
        self.initial_sun = config.INITIAL_SUN_AMOUNT
        self.target_frames = config.MAX_FRAMES
        self.target_flag_waves = 0
        self.timeout_frames = config.MAX_FRAMES
        self._uses_stage_objective = False
        if stage is not None:
            self.apply_stage(stage)

        self._scene = self._new_scene()
        self._steps = 0
        self._collect_render = False
        self._render_data = []  # stored per-frame render info for last episode
        self.rewards = REWARDS
        self._use_shaped = bool(REWARDS.get("use_shaped", False))
        self._last_sun = 0
        self._last_plants = {}
        self._last_zombies = {}
        self._last_wave_index = 0
        self._last_potential = 0.0
        self._reward_details = {}
        self._episode_diagnostics = self._new_episode_diagnostics()

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

    def apply_stage(self, stage):
        self.current_stage = stage
        self.stage_name = stage.stage_name
        self.enabled_rows = set(int(row) for row in stage.enabled_rows)
        self.enabled_plant_ids = set(int(plant) for plant in stage.enabled_plants)
        self.initial_sun = int(stage.initial_sun)
        self.target_frames = int(stage.target_frames)
        self.target_flag_waves = int(stage.target_flag_waves)
        self.timeout_frames = int(stage.timeout_frames)
        self._uses_stage_objective = True

    def reset(self, **kwargs):
        self._scene = self._new_scene()
        self._steps = 0
        self._last_sun = self._scene.sun
        self._last_plants = self._snapshot_plants() if self._use_shaped else {}
        self._last_zombies = self._snapshot_zombies()
        self._last_wave_index = self._current_wave_index()
        self._last_potential = self._calculate_potential() if self._use_shaped else 0.0
        self._reward_details = {}
        self._episode_diagnostics = self._new_episode_diagnostics()
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
        episode_over, episode_win = self._episode_status()
        while (not self._move_available()) and (not episode_over):
            self._scene.step()
            if self._collect_render:
                self._render_data.append(self._capture_frame())
            episode_over, episode_win = self._episode_status()

        reward, details, step_diagnostics = self._calculate_reward(
            action,
            action_success,
            planted_name,
            episode_over,
            episode_win,
        )
        self._record_step_diagnostics(
            action,
            action_success,
            planted_name,
            details,
            step_diagnostics,
        )
        state = self._build_state()
        mask = self.mask_available_actions()
        self._steps += 1
        self._reward_details = details
        info = self._build_info(episode_over, episode_win, details)
        info["mask"] = mask
        return state, float(reward), bool(episode_over), info

    def mask_available_actions(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[self.wait_action] = True
        empty_cells, available_plants = self._scene.get_available_moves()
        if len(empty_cells[0]) == 0:
            return mask
        for plant in available_plants:
            card_idx = self._plant_no[plant.__name__]
            if self.card_plant_ids[card_idx] not in self.enabled_plant_ids:
                continue
            for row, col in zip(empty_cells[0], empty_cells[1]):
                if row not in self.enabled_rows:
                    continue
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
            if plant.lane not in self.enabled_rows:
                continue
            idx = plant.lane * self.cols + plant.pos
            cls_idx = self._plant_no.get(plant.__class__.__name__, empty_unknown_class)
            plant_onehot[idx, :] = 0.0
            plant_onehot[idx, cls_idx] = 1.0
            max_hp = max(1.0, float(getattr(plant, "MAX_HP", 1)))
            plant_hp[idx] = max(0.0, min(1.0, float(plant.hp) / max_hp))

        for zombie in self._scene.zombies:
            if zombie.lane not in self.enabled_rows:
                continue
            idx = zombie.lane * self.cols + zombie.pos
            zombie_hp[idx] = min(
                1.0, zombie_hp[idx] + float(zombie.hp) / ZOMBIE_HP_NORM)

        cooldowns = np.ones(self.num_cards, dtype=np.float32)
        for i, name in enumerate(self._plant_names):
            if name not in self.plant_deck:
                continue
            if self.card_plant_ids[i] not in self.enabled_plant_ids:
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
        if lane not in self.enabled_rows:
            return False, None
        if self.card_plant_ids[plant_idx] not in self.enabled_plant_ids:
            return False, None
        plant_name = self._plant_names[plant_idx]
        if plant_name not in self.plant_deck:
            return False, None
        move = Move(plant_name, lane, pos)
        if move.is_valid(self._scene):
            move.apply_move(self._scene)
            return True, plant_name
        return False, None

    def _calculate_reward(
        self,
        action,
        action_success,
        planted_name,
        episode_over,
        episode_win,
    ):
        reward = 0.0
        details = {}
        step_diagnostics = {
            "zombies_killed": 0,
            "plants_lost": 0,
            "sun_diff": 0,
        }
        use_shaped = self._use_shaped
        current_plants = self._snapshot_plants() if use_shaped else {}
        current_zombies = self._snapshot_zombies()
        current_wave_index = self._current_wave_index()
        damage = self._zombie_damage(current_zombies)
        reward += damage
        details["zombie_damage"] = damage

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
            if use_shaped:
                r_plant = self._plant_reward(planted_name)
                if r_plant:
                    reward += r_plant
                    details["plant"] = r_plant

        if use_shaped:
            r_survival = float(self.rewards.get("survival_per_step", 0.0))
            if r_survival:
                reward += r_survival
                details["survival"] = r_survival

        sun_diff = self._scene.sun - self._last_sun
        step_diagnostics["sun_diff"] = int(sun_diff)
        if use_shaped and sun_diff > 0:
            r_sun = sun_diff * float(self.rewards.get("sun_collect", 0.01))
            reward += r_sun
            details["sun"] = r_sun

        killed = list(getattr(self._scene, "killed_zombies", []))
        if killed:
            step_diagnostics["zombies_killed"] = len(killed)
            if use_shaped:
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
            step_diagnostics["plants_lost"] = len(lost_plants)
            if use_shaped:
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

        potential = self._calculate_potential() if use_shaped else 0.0
        if use_shaped:
            delta = max(-5.0, min(5.0, potential - self._last_potential))
            delta_scale = float(
                self.rewards.get("potential", {}).get("delta_scale", 0.18)
            )
            r_potential = delta * delta_scale
            if abs(r_potential) > 1e-6:
                reward += r_potential
                details["potential_delta"] = r_potential

        if episode_over:
            if episode_win:
                r_win = float(self.rewards.get("game_win", 18.0))
                reward += r_win
                details["win"] = r_win
            else:
                r_lose = float(self.rewards.get("game_lose", -12.0))
                reward += r_lose
                details["lose"] = r_lose

        self._last_sun = self._scene.sun
        self._last_plants = current_plants
        self._last_zombies = current_zombies
        self._last_wave_index = current_wave_index
        self._last_potential = potential
        return reward, details, step_diagnostics

    def _zombie_damage(self, current_zombies):
        damage = 0.0
        current_ids = set(current_zombies)
        for entity_id, zombie in current_zombies.items():
            previous = self._last_zombies.get(entity_id)
            if previous is not None:
                damage += max(0.0, float(previous["hp"]) - float(zombie["hp"]))
        for entity_id, zombie in self._last_zombies.items():
            if entity_id not in current_ids:
                damage += max(0.0, float(zombie["hp"]))
        return damage

    def _new_episode_diagnostics(self):
        return {
            "action_stats": {
                "wait": 0,
                "plant": 0,
                "shovel": 0,
                "invalid": 0,
                "plant_success_by_type": {},
            },
            "reward_breakdown": {},
            "zombies_killed": 0,
            "plants_lost": 0,
            "sun_stats": {
                "final_sun": int(getattr(self, "_scene", None).sun) if hasattr(self, "_scene") else 0,
                "max_sun": int(getattr(self, "_scene", None).sun) if hasattr(self, "_scene") else 0,
                "sun_gained": 0,
                "sun_spent": 0,
                "wait_with_high_sun": 0,
                "_sun_total": 0.0,
                "_sun_samples": 0,
            },
        }

    def _record_step_diagnostics(
        self,
        action,
        action_success,
        planted_name,
        reward_details,
        step_diagnostics,
    ):
        diagnostics = self._episode_diagnostics
        action_stats = diagnostics["action_stats"]
        if action == self.wait_action:
            action_stats["wait"] += 1
            threshold = self.rewards.get("wait_sun_threshold", 300)
            if self._last_sun >= threshold:
                diagnostics["sun_stats"]["wait_with_high_sun"] += 1
        elif not action_success:
            action_stats["invalid"] += 1
        elif planted_name is not None:
            action_stats["plant"] += 1
            by_type = action_stats["plant_success_by_type"]
            by_type[planted_name] = int(by_type.get(planted_name, 0)) + 1

        for key, value in reward_details.items():
            diagnostics["reward_breakdown"][key] = (
                float(diagnostics["reward_breakdown"].get(key, 0.0))
                + float(value)
            )

        diagnostics["zombies_killed"] += int(step_diagnostics.get("zombies_killed", 0))
        diagnostics["plants_lost"] += int(step_diagnostics.get("plants_lost", 0))

        sun_stats = diagnostics["sun_stats"]
        sun_diff = int(step_diagnostics.get("sun_diff", 0))
        if sun_diff > 0:
            sun_stats["sun_gained"] += sun_diff
        elif sun_diff < 0:
            sun_stats["sun_spent"] += abs(sun_diff)
        current_sun = int(self._scene.sun)
        sun_stats["final_sun"] = current_sun
        sun_stats["max_sun"] = max(int(sun_stats["max_sun"]), current_sun)
        sun_stats["_sun_total"] += current_sun
        sun_stats["_sun_samples"] += 1

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

    def _new_scene(self):
        scene = Scene(
            self.plant_deck,
            WaveZombieSpawner(enabled_rows=tuple(sorted(self.enabled_rows))),
        )
        scene.sun = int(self.initial_sun)
        return scene

    def _move_available(self):
        """Fast check: is at least one curriculum-valid action available?

        Only iterates the plant deck (≤10 items) and checks cooldown/sun/grid
        constraints — does NOT build the full 451-dim action mask.  The full
        mask is computed once in ``mask_available_actions()`` when the agent
        actually needs to choose an action.
        """
        if self._scene.grid.is_full():
            return False
        empty_rows = set(self._scene.grid.empty_cells()[0])
        if not (empty_rows & self.enabled_rows):
            return False
        for plant_name, plant_cls in self.plant_deck.items():
            cls_name = plant_cls.__name__
            if cls_name not in self._plant_no:
                continue
            card_idx = self._plant_no[cls_name]
            if self.card_plant_ids[card_idx] not in self.enabled_plant_ids:
                continue
            if (self._scene.plant_cooldowns[plant_name] <= 0
                    and plant_cls.COST <= self._scene.sun):
                return True
        return False

    def _episode_status(self):
        if self._scene.lives <= 0:
            return True, False
        if not self._uses_stage_objective:
            over = self._scene._chrono > config.MAX_FRAMES
            return over, bool(over)

        if self._stage_win_reached():
            return True, True
        if self._scene._chrono >= self.timeout_frames:
            return True, False
        return False, False

    def _stage_win_reached(self):
        return (
            self._scene._chrono >= self.target_frames
            and self.completed_flag_waves >= self.target_flag_waves
        )

    @property
    def completed_flag_waves(self):
        spawner = self._scene._zombie_spawner
        return int(getattr(spawner, "completed_flag_waves", 0))

    def _build_info(self, episode_over, episode_win, reward_details):
        current_wave = self._current_wave_index()
        spawner = self._scene._zombie_spawner
        return {
            "steps": min(self.timeout_frames, self._scene._chrono),
            "win": bool(episode_win),
            "game_ended": bool(episode_over),
            "completed_sublevels": int(
                getattr(spawner, "completed_flag_waves", 0)
            ),
            "completed_flag_waves": int(
                getattr(spawner, "completed_flag_waves", 0)
            ),
            "stage_name": self.stage_name,
            "target_frames": self.target_frames,
            "target_flag_waves": self.target_flag_waves,
            "timeout_frames": self.timeout_frames,
            "current_wave_index": current_wave,
            "is_flag_wave": bool(getattr(spawner, "last_wave_was_flag", False)),
            "zombie_count": len(self._scene.zombies),
            "plant_count": len(self._scene.plants),
            "sun": self._scene.sun,
            "lives": self._scene.lives,
            "reward_details": dict(reward_details),
            "diagnostics": self._episode_diagnostics_snapshot(),
        }

    def _episode_diagnostics_snapshot(self):
        diagnostics = self._episode_diagnostics
        sun_stats = dict(diagnostics["sun_stats"])
        samples = int(sun_stats.pop("_sun_samples", 0))
        total = float(sun_stats.pop("_sun_total", 0.0))
        sun_stats["mean_sun"] = total / samples if samples else 0.0
        return {
            "action_stats": {
                "wait": int(diagnostics["action_stats"]["wait"]),
                "plant": int(diagnostics["action_stats"]["plant"]),
                "shovel": int(diagnostics["action_stats"]["shovel"]),
                "invalid": int(diagnostics["action_stats"]["invalid"]),
                "plant_success_by_type": dict(
                    diagnostics["action_stats"]["plant_success_by_type"]
                ),
            },
            "reward_breakdown": dict(diagnostics["reward_breakdown"]),
            "zombies_killed": int(diagnostics["zombies_killed"]),
            "plants_lost": int(diagnostics["plants_lost"]),
            "sun_stats": sun_stats,
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
