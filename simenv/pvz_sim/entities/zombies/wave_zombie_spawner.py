from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from .zombie_cone import Zombie_cone
from .zombie_bucket import Zombie_bucket
from .zombie_flag import Zombie_flag
from ... import config
from simenv.config import ZOMBIE_SPAWN
import random

FIRST_WAVE_DELAY = ZOMBIE_SPAWN["first_wave_delay_sec"]
WAVE_INTERVAL = ZOMBIE_SPAWN["wave_interval_sec"]
POST_CLEAR_WAVE_DELAY = ZOMBIE_SPAWN["post_clear_wave_delay_sec"]
FLAG_WAVE_FOLLOWUP_INTERVAL = ZOMBIE_SPAWN["flag_wave_followup_interval_sec"]


class WaveZombieSpawner(ZombieSpawner):
    def __init__(self):
        self._next_wave_frame = FIRST_WAVE_DELAY * config.FPS
        self.p = ZOMBIE_SPAWN["base_advanced_probability"]
        self.wave_index = 0
        self.completed_sublevels = 0
        self.last_wave_was_flag = False

    def spawn(self, scene):
        current_frame = int(getattr(scene, "_chrono", 0))
        if self.wave_index > 0 and len(scene.zombies) == 0:
            self._next_wave_frame = min(
                self._next_wave_frame,
                current_frame + POST_CLEAR_WAVE_DELAY * config.FPS,
            )
        if current_frame < self._next_wave_frame:
            if self.wave_index > 0 and len(scene.zombies) == 0:
                self.last_wave_was_flag = False
            return

        self.last_wave_was_flag = (
            self.wave_index % ZOMBIE_SPAWN["flag_wave_modulo"]
            == ZOMBIE_SPAWN["flag_wave_remainder"]
        )
        if self.last_wave_was_flag:
            scene.add_zombie(Zombie_flag(0))
            self.completed_sublevels += 1
        lane = random.choice(range(config.N_LANES))
        s = random.random()
        if s < self.p:
            scene.add_zombie(Zombie_bucket(lane))
        elif s < ZOMBIE_SPAWN["cone_probability_multiplier"] * self.p:
            scene.add_zombie(Zombie_cone(lane))
        else:
            scene.add_zombie(Zombie(lane))
        self.wave_index += 1
        if self.last_wave_was_flag:
            self.p = min(
                self.p * ZOMBIE_SPAWN["advanced_probability_growth"],
                ZOMBIE_SPAWN["max_advanced_probability"],
            )
            self._next_wave_frame = (
                current_frame + FLAG_WAVE_FOLLOWUP_INTERVAL * config.FPS
            )
        else:
            self._next_wave_frame = current_frame + WAVE_INTERVAL * config.FPS
