"""
PVZ game mode ids.

Reference: ConstEnums.h - GameMode
"""

from enum import IntEnum


class GameMode(IntEnum):
    """游戏模式 ID。"""

    SURVIVAL_NORMAL_DAY = 1
    SURVIVAL_NORMAL_NIGHT = 2
    SURVIVAL_NORMAL_POOL = 3
    SURVIVAL_NORMAL_FOG = 4
    SURVIVAL_NORMAL_ROOF = 5
    SURVIVAL_HARD_DAY = 6
    SURVIVAL_HARD_NIGHT = 7
    SURVIVAL_HARD_POOL = 8
    SURVIVAL_HARD_FOG = 9
    SURVIVAL_HARD_ROOF = 10
    SURVIVAL_ENDLESS_DAY = 11
    SURVIVAL_ENDLESS_NIGHT = 12
    SURVIVAL_ENDLESS_POOL = 13
    SURVIVAL_ENDLESS_FOG = 14
    SURVIVAL_ENDLESS_ROOF = 15


POOL_GAME_MODE_IDS = frozenset(
    {
        int(GameMode.SURVIVAL_NORMAL_POOL),
        int(GameMode.SURVIVAL_HARD_POOL),
        int(GameMode.SURVIVAL_ENDLESS_POOL),
    }
)
SURVIVAL_GAME_MODE_IDS = frozenset(int(mode) for mode in GameMode)
POOL_WATER_ROWS = (2, 3)
