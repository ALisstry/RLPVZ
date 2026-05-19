from .plant import Plant

WALLNUT_COST = 50
WALLNUT_COOLDOWN = 20
WALLNUT_MAX_HP = 4000


class Wallnut(Plant):
    MAX_HP = WALLNUT_MAX_HP
    COOLDOWN = WALLNUT_COOLDOWN
    COST = WALLNUT_COST

    def __init__(self, lane, pos):
        super().__init__(lane, pos)

    def step(self, scene):
        pass
