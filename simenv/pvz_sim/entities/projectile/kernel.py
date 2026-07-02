from .pea import Pea
from ... import config


class Kernel(Pea):
    BUTTER_STUN_DURATION = 4 * config.FPS

    def __init__(self, speed, attack, lane, pos, butter=False):
        super().__init__(speed, attack, lane, pos)
        self._butter = bool(butter)

    def _attack_zombies(self, zombies, scene=None):
        zombie_hit = min(zombies, key=lambda z: (z.pos, z._offset))
        zombie_hit.hp -= self._attack
        if self._butter:
            zombie_hit.stun(self.BUTTER_STUN_DURATION)
        self.hp = 0
