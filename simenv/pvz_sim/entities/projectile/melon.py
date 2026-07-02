from .pea import Pea


class Melon(Pea):
    SPLASH_ATTACK = 60
    SPLASH_RADIUS = 1

    def _attack_zombies(self, zombies, scene=None):
        zombie_hit = min(zombies, key=lambda z: (z.pos, z._offset))
        zombie_hit.hp -= self._attack
        if scene is not None:
            for zombie in scene.zombies:
                if zombie is zombie_hit:
                    continue
                if (zombie.lane == zombie_hit.lane
                        and abs(zombie.pos - zombie_hit.pos) <= self.SPLASH_RADIUS):
                    zombie.hp -= self.SPLASH_ATTACK
        self.hp = 0
