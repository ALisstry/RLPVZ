from .plant import Plant
from ... import config

SQUASH_COST = 50
SQUASH_COOLDOWN = 30
SQUASH_MAX_HP = 300
SQUASH_ATTACK = 1800
SQUASH_ATTACK_COOLDOWN = 1


class Squash(Plant):
    MAX_HP = SQUASH_MAX_HP
    COOLDOWN = SQUASH_COOLDOWN
    COST = SQUASH_COST
    ATTACK = SQUASH_ATTACK
    ATTACK_COOLDOWN = SQUASH_ATTACK_COOLDOWN

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1

    def step(self, scene):
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            return
        for zombie in scene.zombies:
            if zombie.lane == self.lane and abs(zombie.pos - self.pos) <= 1:
                zombie.hp -= self.ATTACK
                self.hp = 0
                break
