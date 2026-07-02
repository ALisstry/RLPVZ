from .plant import Plant

SPIKEWEED_COST = 100
SPIKEWEED_COOLDOWN = 7.5
SPIKEWEED_MAX_HP = 300
SPIKEWEED_ATTACK = 20


class Spikeweed(Plant):
    MAX_HP = SPIKEWEED_MAX_HP
    COOLDOWN = SPIKEWEED_COOLDOWN
    COST = SPIKEWEED_COST
    ATTACK = SPIKEWEED_ATTACK
    BLOCKS_ZOMBIE = False

    def __init__(self, lane, pos):
        super().__init__(lane, pos)

    def step(self, scene):
        for zombie in scene.zombies:
            if zombie.lane == self.lane and zombie.pos == self.pos:
                zombie.hp -= self.ATTACK
