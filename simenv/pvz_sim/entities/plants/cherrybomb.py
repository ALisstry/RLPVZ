from .plant import Plant
from ... import config

CHERRY_BOMB_COST = 150
CHERRY_BOMB_COOLDOWN = 50
CHERRY_BOMB_MAX_HP = 300
CHERRY_BOMB_ATTACK = 1800
CHERRY_BOMB_ATTACK_COOLDOWN = 1


class CherryBomb(Plant):
    MAX_HP = CHERRY_BOMB_MAX_HP
    COOLDOWN = CHERRY_BOMB_COOLDOWN
    COST = CHERRY_BOMB_COST
    ATTACK = CHERRY_BOMB_ATTACK
    ATTACK_COOLDOWN = CHERRY_BOMB_ATTACK_COOLDOWN

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1

    def step(self, scene):
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            return
        for zombie in scene.zombies:
            if abs(zombie.lane - self.lane) <= 1 and abs(zombie.pos - self.pos) <= 1:
                zombie.hp -= self.ATTACK
        self.hp = 0
