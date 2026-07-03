from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StageEpsilon:
    start: float = 0.35
    end: float = 0.05
    decay_episodes: int = 20000

    def value(self, stage_episode):
        ratio = min(1.0, max(0.0, stage_episode / max(1, self.decay_episodes)))
        return self.end + (self.start - self.end) * math.exp(-5.0 * ratio)


@dataclass(frozen=True)
class SimCurriculumStage:
    stage_name: str
    enabled_rows: tuple[int, ...]
    enabled_plants: tuple[int, ...]
    initial_sun: int
    target_frames: int
    target_flag_waves: int
    timeout_frames: int
    min_episodes: int
    mean_reward_threshold: float
    mean_success_rate_threshold: float
    epsilon: StageEpsilon
    burn_in: int | None = None


class SimStageGateCurriculum:
    enabled = True

    def __init__(self, stages):
        if not stages:
            raise ValueError("CURRICULUM stage_gate.stages must not be empty")
        self.stages = list(stages)
        self.current_stage_index = 0
        self.stage_episode = 0
        self.completed = False

    @property
    def current_stage(self):
        return self.stages[self.current_stage_index]

    @property
    def current_stage_name(self):
        return self.current_stage.stage_name

    @property
    def is_final_stage(self):
        return self.current_stage_index >= len(self.stages) - 1

    def epsilon(self):
        return self.current_stage.epsilon.value(self.stage_episode)

    def record_episode(self):
        self.stage_episode += 1

    def advance(self, eval_result):
        if not self._meets_current_gate(eval_result):
            return False

        if self.is_final_stage:
            self.completed = True
            return False

        self.current_stage_index += 1
        self.stage_episode = 0
        return True

    def _meets_current_gate(self, eval_result):
        stage = self.current_stage
        min_episodes = max(1, int(stage.min_episodes))
        if self.stage_episode < min_episodes:
            return False
        return (
            float(eval_result.reward_mean) >= stage.mean_reward_threshold
            and float(eval_result.win_rate) >= stage.mean_success_rate_threshold
        )


class DisabledCurriculum:
    enabled = False
    current_stage = None
    current_stage_name = "sim"
    current_stage_index = 0
    stage_episode = 0
    is_final_stage = True
    completed = False

    def epsilon(self):
        return None

    def record_episode(self):
        self.stage_episode += 1
        return False

    def advance(self, eval_result):
        return False


def build_curriculum(raw_config, *, rows, plant_ids):
    raw_config = raw_config or {}
    if not bool(raw_config.get("enabled", False)):
        return DisabledCurriculum()

    strategy = raw_config.get("strategy", "stage_gate")
    if strategy != "stage_gate":
        raise ValueError(f"Unsupported sim curriculum strategy: {strategy}")

    default_epsilon = _build_epsilon(
        raw_config.get("default_stage_epsilon", {})
    )
    default_burn_in = raw_config.get("default_burn_in")
    if default_burn_in is not None:
        default_burn_in = int(default_burn_in)

    raw_stages = (
        raw_config.get("stage_gate", {})
        .get("stages", [])
    )
    stages = [
        _build_stage(
            item,
            default_epsilon=default_epsilon,
            default_burn_in=default_burn_in,
            rows=rows,
            plant_ids=plant_ids,
        )
        for item in raw_stages
    ]
    return SimStageGateCurriculum(stages=stages)


def _build_stage(raw, *, default_epsilon, default_burn_in, rows, plant_ids):
    stage = SimCurriculumStage(
        stage_name=str(raw.get("stage_name", "")),
        enabled_rows=tuple(int(row) for row in raw.get("enabled_rows", range(rows))),
        enabled_plants=tuple(int(plant) for plant in raw.get("enabled_plants", plant_ids)),
        initial_sun=int(raw.get("initial_sun", 50)),
        target_frames=int(raw.get("target_frames", 0)),
        target_flag_waves=int(raw.get("target_flag_waves", 0)),
        timeout_frames=int(raw.get("timeout_frames", 0)),
        min_episodes=int(raw.get("min_episodes", 0)),
        mean_reward_threshold=float(raw.get("mean_reward_threshold", -math.inf)),
        mean_success_rate_threshold=float(raw.get("mean_success_rate_threshold", 0.0)),
        epsilon=_build_epsilon(raw.get("epsilon"), default_epsilon),
        burn_in=(
            int(raw["burn_in"])
            if raw.get("burn_in") is not None
            else default_burn_in
        ),
    )
    _validate_stage(stage, rows=rows, plant_ids=set(plant_ids))
    return stage


def _build_epsilon(raw, default=None):
    if raw is None and default is not None:
        return default
    raw = raw or {}
    return StageEpsilon(
        start=float(raw.get("start", 0.35)),
        end=float(raw.get("end", 0.05)),
        decay_episodes=int(raw.get("decay_episodes", 20000)),
    )


def _validate_stage(stage, *, rows, plant_ids):
    if not stage.stage_name:
        raise ValueError("Sim curriculum stage_name must not be empty")
    if not stage.enabled_rows:
        raise ValueError("Sim curriculum enabled_rows must not be empty")
    if not stage.enabled_plants:
        raise ValueError("Sim curriculum enabled_plants must not be empty")
    invalid_rows = [row for row in stage.enabled_rows if row < 0 or row >= rows]
    if invalid_rows:
        raise ValueError(
            f"Sim curriculum enabled_rows out of range: {invalid_rows}"
        )
    missing_plants = [
        plant for plant in stage.enabled_plants if plant not in plant_ids
    ]
    if missing_plants:
        raise ValueError(
            f"Sim curriculum enabled_plants not in CARD_SPECS: {missing_plants}"
        )
    if stage.target_frames < 0 or stage.target_flag_waves < 0:
        raise ValueError("Sim curriculum targets must be >= 0")
    if stage.timeout_frames <= 0:
        raise ValueError("Sim curriculum timeout_frames must be > 0")
