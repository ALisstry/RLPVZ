from . import config
from .grid import Grid
import numpy as np
from .entities.projectile.mower import Mower


class Scene:
    def __init__(self, plant_deck, zombie_spawner):
        self.plants = []
        self.zombies = []
        self.projectiles = []
        self.sun = config.INITIAL_SUN_AMOUNT
        self.plant_deck = plant_deck
        self.plant_cooldowns = {plant: 0 for plant in plant_deck}
        self.grid = Grid()
        self._zombie_spawner = zombie_spawner
        self._timer = config.NATURAL_SUN_PRODUCTION_COOLDOWN * config.FPS - 1
        self._chrono = 0
        self.score = 0
        self.lives = 1

    def step(self):
        for plant in self.plants:
            plant.step(self)
        for zombie in self.zombies:
            zombie.step(self)
        for projectile in self.projectiles:
            projectile.step(self)

        self._chrono += 1
        self.score = (config.SURVIVAL *
                      int((self._chrono + 1) % (config.FPS * config.SURVIVAL_STEP) == 0)
                      + self.grid._mowers.sum())

        self._zombie_spawner.spawn(self)
        self._remove_dead_objects()
        self._timed_events()
        self._timer -= 1

    def add_zombie(self, zombie):
        self.zombies.append(zombie)
        self.grid.zombie_entrance(zombie.lane)

    def zombie_reach(self, lane):
        if self.grid.is_mower(lane):
            self.grid.remove_mower(lane)
            self.projectiles.append(Mower(lane))
        else:
            self.lives -= 1

    def _remove_dead_objects(self):
        alive_plants = []
        for plant in self.plants:
            if plant:
                alive_plants.append(plant)
                self.score += config.SCORE_ALIVE_PLANT
            else:
                self.grid.remove_obj(plant.lane, plant.pos)
        self.plants = alive_plants

        alive_zombies = []
        for zombie in self.zombies:
            if zombie:
                alive_zombies.append(zombie)
            else:
                self.grid.zombie_death(zombie.lane)
                self.score += zombie.SCORE
        self.zombies = alive_zombies

        alive_projectiles = []
        for projectile in self.projectiles:
            if projectile:
                alive_projectiles.append(projectile)
        self.projectiles = alive_projectiles

    def _timed_events(self):
        for plant in self.plant_cooldowns:
            self.plant_cooldowns[plant] = max(0, self.plant_cooldowns[plant] - 1)
        if self._timer <= 0:
            self.sun += config.NATURAL_SUN_PRODUCTION
            self._timer = config.NATURAL_SUN_PRODUCTION_COOLDOWN * config.FPS - 1

    def move_available(self):
        if not self.grid.is_full():
            for plant_name in self.plant_deck:
                if (self.plant_cooldowns[plant_name] <= 0
                        and self.plant_deck[plant_name].COST <= self.sun):
                    return True
        return False

    def get_available_moves(self):
        empty_cells = self.grid.empty_cells()
        available_plants = [
            self.plant_deck[plant_name]
            for plant_name in self.plant_deck
            if (self.sun >= self.plant_deck[plant_name].COST
                and self.plant_cooldowns[plant_name] <= 0)
        ]
        return (empty_cells, available_plants)
