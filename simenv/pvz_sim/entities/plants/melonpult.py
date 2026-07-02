from .peashooter import Peashooter
from ..projectile.melon import Melon
from ... import config

MELON_PULT_COST = 300
MELON_PULT_COOLDOWN = 7.5
MELON_PULT_MAX_HP = 300
MELON_PULT_ATTACK = 80
MELON_PULT_ATTACK_COOLDOWN = 3.0
MELON_PULT_PROJECTILE_SPEED = 3.75


class MelonPult(Peashooter):
    MAX_HP = MELON_PULT_MAX_HP
    COOLDOWN = MELON_PULT_COOLDOWN
    COST = MELON_PULT_COST
    ATTACK = MELON_PULT_ATTACK
    ATTACK_COOLDOWN = MELON_PULT_ATTACK_COOLDOWN
    PROJECTILE_SPEED = MELON_PULT_PROJECTILE_SPEED

    def step(self, scene):
        if self.attack_cooldown <= 0:
            if scene.grid.is_attacked(self.lane):
                scene.projectiles.append(
                    Melon(self.PROJECTILE_SPEED, self.ATTACK, self.lane, self.pos))
                self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        else:
            self.attack_cooldown -= 1
