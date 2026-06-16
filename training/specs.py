from dataclasses import dataclass
from typing import Any

from utils.train_utils import load_training_config


@dataclass(frozen=True)
class EnvSpec:
    rows: int
    cols: int
    grid_channels: int
    plant_types: int
    action_space_size: int
    global_feature_dim: int
    card_attribute_shape: tuple[int, int]
    use_action_mask: bool
    num_envs: int
    base_port: int


@dataclass(frozen=True)
class ScenarioSpec:
    map_id: str
    game_mode: str
    cards: tuple[int, ...]
    enabled_rows: tuple[int, ...]
    enabled_plants: tuple[int, ...]
    difficulty_stage: int = 0


def build_specs(args: Any) -> tuple[EnvSpec, ScenarioSpec]:
    config = load_training_config(getattr(args, "training_config", None))
    game_cfg = config.get("game", {})
    cards_cfg = config.get("cards", {})
    action_cfg = config.get("action_space", {})
    obs_cfg = config.get("observation_space", {})

    rows = int(game_cfg.get("rows", 6))
    cols = int(game_cfg.get("cols", 9))
    grid_shape = _parse_shape(obs_cfg.get("grid", {}).get("shape"), (rows, cols, 13))
    if grid_shape[0] != rows or grid_shape[1] != cols:
        raise ValueError(
            "observation_space.grid.shape 与 game.rows/game.cols 不一致: "
            f"expected ({rows}, {cols}, C), got {grid_shape}"
        )
    global_feature_dim = int(obs_cfg.get("global", {}).get("total_dim", 0))
    card_attribute_shape = _parse_shape(
        obs_cfg.get("card_attributes", {}).get("shape"), (10, 7)
    )
    cards = tuple(int(card["id"]) for card in cards_cfg.get("plants", []))
    plant_types = int(cards_cfg.get("slot_count", len(cards) or 10))
    action_structure = action_cfg.get("structure", {})
    plant_actions = plant_types * rows * cols
    shovel_actions = int(action_structure.get("shovel_actions", 0))
    wait_actions = int(action_structure.get("wait_action", 1))
    action_space_size = plant_actions + shovel_actions + wait_actions
    _validate_action_config(action_cfg, plant_actions, action_space_size)

    env_spec = EnvSpec(
        rows=rows,
        cols=cols,
        grid_channels=grid_shape[2],
        plant_types=plant_types,
        action_space_size=action_space_size,
        global_feature_dim=global_feature_dim,
        card_attribute_shape=card_attribute_shape,
        use_action_mask=True,
        num_envs=max(1, int(getattr(args, "num_envs", 1))),
        base_port=int(getattr(args, "base_port", getattr(args, "port", 12345))),
    )
    scenario_spec = ScenarioSpec(
        map_id=str(game_cfg.get("map", "pool")),
        game_mode=str(game_cfg.get("mode", "")),
        cards=cards,
        enabled_rows=tuple(range(rows)),
        enabled_plants=cards,
    )
    return env_spec, scenario_spec


def _parse_shape(raw_shape, default: tuple[int, ...]) -> tuple[int, ...]:
    shape = raw_shape if raw_shape is not None else default
    return tuple(int(value) for value in shape)


def _validate_action_config(
    action_cfg: dict,
    plant_actions: int,
    action_space_size: int,
) -> None:
    structure = action_cfg.get("structure", {})
    configured_plant_actions = structure.get("plant_actions")
    if (
        configured_plant_actions is not None
        and int(configured_plant_actions) != plant_actions
    ):
        raise ValueError(
            "action_space.structure.plant_actions 与行列/植物数量不一致: "
            f"expected {plant_actions}, got {configured_plant_actions}"
        )

    configured_size = action_cfg.get("size")
    if configured_size is not None and int(configured_size) != action_space_size:
        raise ValueError(
            "action_space.size 与 action_space.structure 不一致: "
            f"expected {action_space_size}, got {configured_size}"
        )
