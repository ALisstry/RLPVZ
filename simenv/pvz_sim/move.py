from . import config


class Move:
    def __init__(self, plant_name, lane, pos):
        self.plant_name = plant_name
        assert 0 <= lane < config.N_LANES
        assert 0 <= pos < config.LANE_LENGTH
        self.lane = lane
        self.pos = pos

    def is_valid(self, scene):
        assert self.plant_name in scene.plant_deck
        return (scene.plant_cooldowns[self.plant_name] <= 0
                and scene.grid.is_empty(self.lane, self.pos)
                and scene.sun >= scene.plant_deck[self.plant_name].COST)

    def apply_move(self, scene):
        plant = scene.plant_deck[self.plant_name](self.lane, self.pos)
        scene.plants.append(plant)
        scene.grid.add_obj(
            self.lane,
            self.pos,
            blocks=getattr(plant, "BLOCKS_ZOMBIE", True),
        )
        scene.plant_cooldowns[self.plant_name] = (
            scene.plant_deck[self.plant_name].COOLDOWN * config.FPS - 1)
        scene.sun -= scene.plant_deck[self.plant_name].COST
