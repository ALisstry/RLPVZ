from .plant import Plant
from ..projectile.pea import Pea
from ... import config

PEASHOOTER_COST = 100
PEASHOOTER_COOLDOWN = 5
PEASHOOTER_MAX_HP = 300
PEASHOOTER_ATTACK = 20
PEASHOOTER_ATTACK_COOLDOWN = 1.5
PEASHOOTER_PROJECTILE_SPEED = 5


class Peashooter(Plant):
    MAX_HP = PEASHOOTER_MAX_HP
    COOLDOWN = PEASHOOTER_COOLDOWN
    COST = PEASHOOTER_COST
    ATTACK = PEASHOOTER_ATTACK
    ATTACK_COOLDOWN = PEASHOOTER_ATTACK_COOLDOWN
    PROJECTILE_SPEED = PEASHOOTER_PROJECTILE_SPEED

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1

    def step(self, scene):
        if self.attack_cooldown <= 0:
            if scene.grid.is_attacked(self.lane):
                scene.projectiles.append(
                    Pea(self.PROJECTILE_SPEED, self.ATTACK, self.lane, self.pos))
                self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        else:
            self.attack_cooldown -= 1
