from .peashooter import Peashooter
from ..projectile.snowpea import SnowPeaProjectile
from ... import config

SNOW_PEA_COST = 175
SNOW_PEA_COOLDOWN = 5
SNOW_PEA_MAX_HP = 300
SNOW_PEA_ATTACK = 20
SNOW_PEA_ATTACK_COOLDOWN = 1.41
SNOW_PEA_PROJECTILE_SPEED = 4.625


class SnowPea(Peashooter):
    MAX_HP = SNOW_PEA_MAX_HP
    COOLDOWN = SNOW_PEA_COOLDOWN
    COST = SNOW_PEA_COST
    ATTACK = SNOW_PEA_ATTACK
    ATTACK_COOLDOWN = SNOW_PEA_ATTACK_COOLDOWN
    PROJECTILE_SPEED = SNOW_PEA_PROJECTILE_SPEED

    def step(self, scene):
        if self.attack_cooldown <= 0:
            if scene.grid.is_attacked(self.lane):
                scene.projectiles.append(
                    SnowPeaProjectile(
                        self.PROJECTILE_SPEED,
                        self.ATTACK,
                        self.lane,
                        self.pos,
                    )
                )
                self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        else:
            self.attack_cooldown -= 1
