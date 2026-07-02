from .pea import Pea
from ... import config


class SnowPeaProjectile(Pea):
    SLOW_DURATION = 10 * config.FPS

    def _attack_zombies(self, zombies, scene=None):
        zombie_hit = min(zombies, key=lambda z: (z.pos, z._offset))
        zombie_hit.hp -= self._attack
        zombie_hit.slow(self.SLOW_DURATION)
        self.hp = 0
