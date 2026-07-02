import random

from .peashooter import Peashooter
from ..projectile.kernel import Kernel
from ... import config

KERNEL_PULT_COST = 100
KERNEL_PULT_COOLDOWN = 7.5
KERNEL_PULT_MAX_HP = 300
KERNEL_PULT_ATTACK = 20
KERNEL_PULT_BUTTER_ATTACK = 40
KERNEL_PULT_ATTACK_COOLDOWN = 3.0
KERNEL_PULT_PROJECTILE_SPEED = 3.75
KERNEL_PULT_BUTTER_PROBABILITY = 0.25


class KernelPult(Peashooter):
    MAX_HP = KERNEL_PULT_MAX_HP
    COOLDOWN = KERNEL_PULT_COOLDOWN
    COST = KERNEL_PULT_COST
    ATTACK = KERNEL_PULT_ATTACK
    BUTTER_ATTACK = KERNEL_PULT_BUTTER_ATTACK
    ATTACK_COOLDOWN = KERNEL_PULT_ATTACK_COOLDOWN
    PROJECTILE_SPEED = KERNEL_PULT_PROJECTILE_SPEED
    BUTTER_PROBABILITY = KERNEL_PULT_BUTTER_PROBABILITY

    def step(self, scene):
        if self.attack_cooldown <= 0:
            if scene.grid.is_attacked(self.lane):
                butter = random.random() < self.BUTTER_PROBABILITY
                scene.projectiles.append(
                    Kernel(
                        self.PROJECTILE_SPEED,
                        self.BUTTER_ATTACK if butter else self.ATTACK,
                        self.lane,
                        self.pos,
                        butter=butter,
                    )
                )
                self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        else:
            self.attack_cooldown -= 1
