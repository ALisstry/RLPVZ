from dataclasses import dataclass
from typing import Any

from data.game_modes import SURVIVAL_GAME_MODE_IDS
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
    game_mode_id: int
    level: int | None
    rows: int
    cols: int
    cards: tuple[int, ...]
    enabled_rows: tuple[int, ...]
    enabled_plants: tuple[int, ...]
    initial_sun: int | None
    win_condition: str
    target_sublevels: int


def build_specs(args: Any) -> tuple[EnvSpec, ScenarioSpec]:
    config = load_training_config(getattr(args, "training_config", None))
    curriculum_name = getattr(args, "curriculum", "none")
    base_scenario, stage_scenarios = build_scenario_specs(config, curriculum_name)
    all_scenarios = (base_scenario, *stage_scenarios)

    game_cfg = config.get("game", {})
    cards_cfg = config.get("cards", {})
    action_cfg = config.get("action_space", {})
    obs_cfg = config.get("observation_space", {})

    rows = max(scenario.rows for scenario in all_scenarios)
    cols = max(scenario.cols for scenario in all_scenarios)
    configured_grid_shape = _parse_shape(
        obs_cfg.get("grid", {}).get("shape"), (rows, cols, 13)
    )
    if configured_grid_shape[0] != rows or configured_grid_shape[1] != cols:
        raise ValueError(
            "observation_space.grid.shape 必须匹配所有课程阶段的最大行列数: "
            f"expected ({rows}, {cols}, C), got {configured_grid_shape}"
        )
    grid_shape = configured_grid_shape
    global_feature_dim = int(obs_cfg.get("global", {}).get("total_dim", 0))
    card_attribute_shape = _parse_shape(
        obs_cfg.get("card_attributes", {}).get("shape"), (10, 7)
    )
    cards = base_scenario.cards
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
    if curriculum_name == "stage_gate" and stage_scenarios:
        scenario_spec = stage_scenarios[0]
    else:
        scenario_spec = base_scenario
    return env_spec, scenario_spec


def build_scenario_specs(
    config: dict[str, Any],
    curriculum_name: str = "none",
) -> tuple[ScenarioSpec, tuple[ScenarioSpec, ...]]:
    game_cfg = config.get("game", {})
    cards = _parse_cards(config.get("cards", {}))
    config_rows = int(game_cfg.get("rows", 6))
    config_cols = int(game_cfg.get("cols", 9))
    initial_sun = game_cfg.get("initial_sun", 50)
    win_condition = str(game_cfg.get("win_condition", "level_end"))
    target_sublevels = int(game_cfg.get("target_sublevels", 1))
    level = game_cfg.get("level")
    base_scenario = ScenarioSpec(
        game_mode_id=int(game_cfg.get("game_mode_id", 13)),
        level=None if level is None else int(level),
        rows=config_rows,
        cols=config_cols,
        cards=cards,
        enabled_rows=tuple(range(config_rows)),
        enabled_plants=cards,
        initial_sun=None if initial_sun is None else int(initial_sun),
        win_condition=win_condition,
        target_sublevels=target_sublevels,
    )
    validate_scenario_spec(base_scenario)

    if curriculum_name != "stage_gate":
        return base_scenario, ()

    stage_cfgs = (
        config.get("curriculum", {})
        .get("stage_gate", {})
        .get("stages", [])
        or []
    )
    stage_scenarios = tuple(
        _build_stage_scenario(stage_cfg, base_scenario)
        for stage_cfg in stage_cfgs
    )
    return base_scenario, stage_scenarios


def _build_stage_scenario(
    stage_cfg: dict[str, Any],
    base_scenario: ScenarioSpec,
) -> ScenarioSpec:
    stage_cards = _parse_stage_cards(stage_cfg)
    if stage_cards is not None and stage_cards != base_scenario.cards:
        raise ValueError("课程阶段不能改变 cards，避免动作索引和 checkpoint 语义变化")

    rows = int(stage_cfg.get("rows", base_scenario.rows))
    cols = int(stage_cfg.get("cols", base_scenario.cols))
    raw_enabled_rows = stage_cfg.get("enabled_rows")
    raw_enabled_plants = stage_cfg.get("enabled_plants")
    raw_level = stage_cfg.get("level", base_scenario.level)
    raw_initial_sun = stage_cfg.get("initial_sun", base_scenario.initial_sun)
    win_condition = str(stage_cfg.get("win_condition", base_scenario.win_condition))
    target_sublevels = int(
        stage_cfg.get("target_sublevels", base_scenario.target_sublevels)
    )
    enabled_rows = (
        tuple(range(rows))
        if raw_enabled_rows is None
        else tuple(int(row) for row in raw_enabled_rows)
    )
    enabled_plants = (
        base_scenario.cards
        if raw_enabled_plants is None
        else tuple(int(plant) for plant in raw_enabled_plants)
    )
    scenario = ScenarioSpec(
        game_mode_id=int(stage_cfg.get("game_mode_id", base_scenario.game_mode_id)),
        level=None if raw_level is None else int(raw_level),
        rows=rows,
        cols=cols,
        cards=base_scenario.cards,
        enabled_rows=enabled_rows,
        enabled_plants=enabled_plants,
        initial_sun=None if raw_initial_sun is None else int(raw_initial_sun),
        win_condition=win_condition,
        target_sublevels=target_sublevels,
    )
    validate_scenario_spec(scenario)
    return scenario


def _parse_cards(cards_cfg: dict[str, Any]) -> tuple[int, ...]:
    cards = tuple(int(card["id"]) for card in cards_cfg.get("plants", []))
    slot_count = int(cards_cfg.get("slot_count", len(cards) or 10))
    if cards and len(cards) != slot_count:
        raise ValueError(
            "cards.slot_count 必须与 cards.plants 数量一致，才能固定动作索引语义"
        )
    return cards


def _parse_stage_cards(stage_cfg: dict[str, Any]) -> tuple[int, ...] | None:
    if "cards" not in stage_cfg:
        return None
    raw_cards = stage_cfg["cards"]
    if isinstance(raw_cards, dict):
        raw_cards = raw_cards.get("plants", [])
    cards = []
    for item in raw_cards or []:
        if isinstance(item, dict):
            cards.append(int(item["id"]))
        else:
            cards.append(int(item))
    return tuple(cards)


def validate_scenario_spec(
    scenario: ScenarioSpec,
    *,
    expected_cards: tuple[int, ...] | None = None,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> None:
    if scenario.rows <= 0 or scenario.cols <= 0:
        raise ValueError("ScenarioSpec.rows/cols 必须为正数")
    if scenario.win_condition not in {"level_end", "survival_sublevels"}:
        raise ValueError(
            "ScenarioSpec.win_condition 只支持 level_end 或 survival_sublevels"
        )
    if scenario.target_sublevels < 1:
        raise ValueError("ScenarioSpec.target_sublevels 必须 >= 1")
    if (
        scenario.win_condition == "survival_sublevels"
        and int(scenario.game_mode_id) not in SURVIVAL_GAME_MODE_IDS
    ):
        raise ValueError("survival_sublevels 仅支持 survival 类 game_mode_id")
    if expected_cards is not None and tuple(expected_cards) != tuple(scenario.cards):
        raise ValueError(
            f"ScenarioSpec cards mismatch: expected {tuple(expected_cards)}, "
            f"got {scenario.cards}"
        )
    if max_rows is not None and scenario.rows > max_rows:
        raise ValueError(
            "ScenarioSpec rows 不能超过 run-level EnvSpec: "
            f"scenario={scenario.rows}, env={max_rows}"
        )
    if max_cols is not None and scenario.cols > max_cols:
        raise ValueError(
            "ScenarioSpec cols 不能超过 run-level EnvSpec: "
            f"scenario={scenario.cols}, env={max_cols}"
        )
    invalid_rows = [
        row for row in scenario.enabled_rows if row < 0 or row >= scenario.rows
    ]
    if invalid_rows:
        raise ValueError(
            f"enabled_rows 必须在当前 scenario.rows 内: {invalid_rows}"
        )
    missing_plants = [
        plant for plant in scenario.enabled_plants if plant not in scenario.cards
    ]
    if missing_plants:
        raise ValueError(
            f"enabled_plants 必须能由固定 cards 表达: {missing_plants}"
        )


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
