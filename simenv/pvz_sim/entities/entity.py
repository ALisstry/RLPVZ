from .. import config


class Entity:
    _next_entity_id = 0
    MAX_HP = None

    def __init__(self, lane):
        self.entity_id = Entity._next_entity_id
        Entity._next_entity_id += 1
        self.hp = self.MAX_HP
        assert 0 <= lane < config.N_LANES
        self.lane = lane

    def step(self, scene):
        raise NotImplementedError

    def __bool__(self):
        return self.hp > 0
