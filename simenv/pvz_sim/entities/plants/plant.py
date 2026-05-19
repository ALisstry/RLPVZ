from ..entity import Entity
from ... import config


class Plant(Entity):
    COOLDOWN = None
    COST = None

    def __init__(self, lane, pos):
        super().__init__(lane)
        assert 0 <= pos < config.LANE_LENGTH
        self.pos = pos
