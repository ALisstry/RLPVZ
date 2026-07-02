from .peashooter import Peashooter
from ..projectile.pea import Pea
from ... import config

REPEATER_COST = 200
REPEATER_COOLDOWN = 5
REPEATER_MAX_HP = 300
REPEATER_ATTACK = 20
REPEATER_ATTACK_COOLDOWN = 1.41 / 2
REPEATER_PROJECTILE_SPEED = 4.625


class Repeater(Peashooter):
    MAX_HP = REPEATER_MAX_HP
    COOLDOWN = REPEATER_COOLDOWN
    COST = REPEATER_COST
    ATTACK = REPEATER_ATTACK
    ATTACK_COOLDOWN = REPEATER_ATTACK_COOLDOWN
    PROJECTILE_SPEED = REPEATER_PROJECTILE_SPEED

    def step(self, scene):
        if self.attack_cooldown <= 0:
            if scene.grid.is_attacked(self.lane):
                scene.projectiles.append(
                    Pea(self.PROJECTILE_SPEED, self.ATTACK, self.lane, self.pos))
                self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        else:
            self.attack_cooldown -= 1
