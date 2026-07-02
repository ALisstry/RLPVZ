from ..entity import Entity
from ... import config


class Zombie(Entity):
    MAX_HP = 270
    WALKING_SPEED = 80 / 23
    ATTACK_SPEED = 100
    SCORE = 120

    def __init__(self, lane, pos=config.LANE_LENGTH - 1):
        super().__init__(lane)
        self.pos = pos
        self._attack = self.ATTACK_SPEED // config.FPS
        self._cell_length = self.WALKING_SPEED * config.FPS
        self._offset = self._cell_length - 1
        self._slow_timer = 0
        self._stun_timer = 0

    def step(self, scene):
        if self._stun_timer > 0:
            self._stun_timer -= 1
            return

        if not scene.grid.is_blocked(self.lane, self.pos):
            movement = 0.5 if self._slow_timer > 0 else 1.0
            if self._slow_timer > 0:
                self._slow_timer -= 1
            if self._offset <= 0:
                self.pos -= 1
                self._offset = self._cell_length - 1
                if self.pos < 0:
                    scene.zombie_reach(self.lane)
                    self.hp = 0
            else:
                self._offset -= movement
        else:
            for plant in scene.plants:
                if (plant.lane == self.lane
                        and plant.pos == self.pos
                        and getattr(plant, "BLOCKS_ZOMBIE", True)):
                    self.attack(plant)
                    break

    def attack(self, plant):
        plant.hp -= self._attack

    def slow(self, duration):
        self._slow_timer = max(self._slow_timer, int(duration))

    def stun(self, duration):
        self._stun_timer = max(self._stun_timer, int(duration))

    def get_offset(self):
        return self._offset / self._cell_length

    def __str__(self):
        return (f"Lane: {self.lane} Pos: {self.pos} Health: {self.hp}")
