from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = False
    freq_episodes: int = 500
    episodes: int = 20
    deterministic: bool = True
    save_episode_details: bool = True
    real_num_envs: int = 1
    real_base_port: int | None = None


@dataclass(frozen=True)
class EpisodeEvalResult:
    eval_id: str
    episode_index: int
    reward: float
    survival: float
    win: bool
    game_ended: bool = True
    completed_sublevels: int | None = None
    zombies_killed: int | None = None
    plants_lost: int | None = None
    actions: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    eval_id: str
    algo: str
    env_kind: str
    episode: int | None
    step: int | None
    stage_name: str
    win_condition: str | None
    target_sublevels: int | None
    episodes: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    survival_mean: float
    survival_std: float
    survival_min: float
    survival_max: float
    win_rate: float
    win_count: int
    completed_sublevels_mean: float | None = None
    zombies_killed_mean: float | None = None
    plants_lost_mean: float | None = None
    actions_mean: float | None = None
    duration_sec: float = 0.0
    model_path: str | None = None
    checkpoint_tag: str | None = None
    error: str | None = None
    details: list[EpisodeEvalResult] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class EvaluationScheduler:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def should_run(self, episode: int) -> bool:
        if not self.config.enabled:
            return False
        freq = int(self.config.freq_episodes)
        return episode > 0 and freq > 0 and episode % freq == 0


class EvaluationWriter:
    summary_fieldnames = [
        "eval_id",
        "algo",
        "env_kind",
        "episode",
        "step",
        "stage_name",
        "win_condition",
        "target_sublevels",
        "episodes",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "survival_mean",
        "survival_std",
        "survival_min",
        "survival_max",
        "win_rate",
        "win_count",
        "completed_sublevels_mean",
        "zombies_killed_mean",
        "plants_lost_mean",
        "actions_mean",
        "duration_sec",
        "model_path",
        "checkpoint_tag",
        "error",
        "extra",
    ]

    detail_fieldnames = [
        "eval_id",
        "episode_index",
        "reward",
        "survival",
        "win",
        "game_ended",
        "completed_sublevels",
        "zombies_killed",
        "plants_lost",
        "actions",
        "error",
        "extra",
    ]

    def __init__(self, output_dir: str, save_episode_details: bool = True):
        self.output_dir = output_dir
        self.save_episode_details = save_episode_details
        self.jsonl_path = os.path.join(output_dir, "eval.jsonl")
        self.csv_path = os.path.join(output_dir, "eval.csv")
        self.snapshot_path = os.path.join(output_dir, "eval_snapshot.json")
        self.details_path = os.path.join(output_dir, "eval_details.csv")
        os.makedirs(output_dir, exist_ok=True)

    def write(self, result: EvaluationResult) -> None:
        summary = _summary_row(result)
        with open(self.jsonl_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        _append_csv(self.csv_path, self.summary_fieldnames, summary)
        with open(self.snapshot_path, "w", encoding="utf-8") as file:
            json.dump(asdict(result), file, ensure_ascii=False, indent=2)
        if self.save_episode_details:
            for detail in result.details:
                _append_csv(
                    self.details_path,
                    self.detail_fieldnames,
                    _detail_row(detail),
                )


class BestEvaluationCheckpoint:
    def __init__(self, output_dir: str, model_filename: str = "best_model.pt"):
        self.output_dir = output_dir
        self.model_path = os.path.join(output_dir, model_filename)
        self.metadata_path = os.path.join(output_dir, "best_eval.json")
        self.best_score = self._load_best_score()

    def maybe_save(self, result: EvaluationResult, save_model) -> str | None:
        score = evaluation_score(result)
        if self.best_score is not None and score <= self.best_score:
            return None

        os.makedirs(self.output_dir, exist_ok=True)
        save_model(self.model_path)
        self.best_score = score
        metadata = _summary_row(result)
        metadata["best_score"] = list(score)
        metadata["saved_model_path"] = self.model_path
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        return self.model_path

    def _load_best_score(self) -> tuple[float, float, float] | None:
        if not os.path.exists(self.metadata_path):
            return None
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            score = metadata.get("best_score")
            if not isinstance(score, list) or len(score) != 3:
                return None
            return (float(score[0]), float(score[1]), float(score[2]))
        except Exception:
            return None


def evaluation_score(result: EvaluationResult) -> tuple[float, float, float]:
    return (
        float(result.win_rate),
        float(result.survival_mean),
        float(result.reward_mean),
    )


def load_evaluation_config(raw: dict[str, Any] | None) -> EvaluationConfig:
    raw = raw or {}
    return EvaluationConfig(
        enabled=bool(raw.get("enabled", False)),
        freq_episodes=int(raw.get("freq_episodes", 500)),
        episodes=int(raw.get("episodes", 20)),
        deterministic=bool(raw.get("deterministic", True)),
        save_episode_details=bool(raw.get("save_episode_details", True)),
        real_num_envs=max(1, int(raw.get("real_num_envs", 1))),
        real_base_port=(
            None
            if raw.get("real_base_port") in (None, "")
            else int(raw.get("real_base_port"))
        ),
    )


def new_eval_id(prefix: str = "eval") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def summarize_eval_results(
    *,
    eval_id: str,
    algo: str,
    env_kind: str,
    episode: int | None,
    step: int | None,
    stage_name: str = "",
    win_condition: str | None = None,
    target_sublevels: int | None = None,
    details: list[EpisodeEvalResult],
    duration_sec: float,
    model_path: str | None = None,
    checkpoint_tag: str | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> EvaluationResult:
    rewards = [item.reward for item in details]
    survivals = [item.survival for item in details]
    wins = [1 if item.win else 0 for item in details]
    return EvaluationResult(
        eval_id=eval_id,
        algo=algo,
        env_kind=env_kind,
        episode=episode,
        step=step,
        stage_name=stage_name,
        win_condition=win_condition,
        target_sublevels=target_sublevels,
        episodes=len(details),
        reward_mean=_mean(rewards),
        reward_std=_std(rewards),
        reward_min=min(rewards) if rewards else 0.0,
        reward_max=max(rewards) if rewards else 0.0,
        survival_mean=_mean(survivals),
        survival_std=_std(survivals),
        survival_min=min(survivals) if survivals else 0.0,
        survival_max=max(survivals) if survivals else 0.0,
        win_rate=_mean(wins),
        win_count=sum(wins),
        completed_sublevels_mean=_optional_mean(
            item.completed_sublevels for item in details
        ),
        zombies_killed_mean=_optional_mean(item.zombies_killed for item in details),
        plants_lost_mean=_optional_mean(item.plants_lost for item in details),
        actions_mean=_optional_mean(item.actions for item in details),
        duration_sec=duration_sec,
        model_path=model_path,
        checkpoint_tag=checkpoint_tag,
        error=error,
        details=details,
        extra=extra or {},
    )


def summarize_plant_stats(details: Iterable[EpisodeEvalResult]) -> dict[str, Any]:
    details = list(details)
    totals: dict[str, dict[str, Any]] = {}
    action_counts: dict[str, int] = {}
    for detail in details:
        diagnostics = detail.extra.get("diagnostics") or {}
        action_stats = diagnostics.get("action_stats") or {}
        _merge_count_map(
            action_counts,
            action_stats.get("plant_success_by_type") or {},
        )
        plant_stats = detail.extra.get("plant_stats") or {}
        for key, stat in plant_stats.items():
            plant_id = str(stat.get("plant_id", key))
            item = totals.setdefault(
                plant_id,
                {
                    "plant_id": int(stat.get("plant_id", plant_id)),
                    "name": stat.get("name", plant_id),
                    "count_total": 0,
                    "observed_count_total": 0,
                    "survival_steps_total": 0.0,
                },
            )
            item["observed_count_total"] += int(stat.get("count", 0))
            item["survival_steps_total"] += float(
                stat.get("survival_steps_total", 0.0)
            )

    for item in totals.values():
        action_count = action_counts.pop(str(item["name"]), None)
        if action_count is None:
            action_count = item["observed_count_total"]
        item["count_total"] = int(action_count)
        count = item["count_total"]
        item["survival_steps_mean"] = (
            item["survival_steps_total"] / count if count > 0 else 0.0
        )

    for name, count in action_counts.items():
        totals[f"action:{name}"] = {
            "plant_id": -1,
            "name": name,
            "count_total": int(count),
            "observed_count_total": 0,
            "survival_steps_total": 0.0,
            "survival_steps_mean": 0.0,
        }

    return {
        key: totals[key]
        for key in sorted(totals, key=_plant_stats_sort_key)
    }


def summarize_diagnostics(details: Iterable[EpisodeEvalResult]) -> dict[str, Any]:
    details = list(details)
    summary: dict[str, Any] = {
        "action_stats": {
            "wait": 0,
            "plant": 0,
            "shovel": 0,
            "invalid": 0,
        },
        "reward_breakdown": {},
        "zombies_killed": 0,
        "plants_lost": 0,
        "per_episode": [],
    }
    for detail in details:
        diagnostics = detail.extra.get("diagnostics") or {}
        summary["per_episode"].append(_episode_diagnostics_row(detail, diagnostics))
        action_stats = diagnostics.get("action_stats") or {}
        summary["action_stats"]["wait"] += int(action_stats.get("wait", 0))
        summary["action_stats"]["plant"] += int(action_stats.get("plant", 0))
        summary["action_stats"]["shovel"] += int(action_stats.get("shovel", 0))
        summary["action_stats"]["invalid"] += int(action_stats.get("invalid", 0))
        _merge_float_map(
            summary["reward_breakdown"],
            diagnostics.get("reward_breakdown") or {},
        )
        summary["zombies_killed"] += int(diagnostics.get("zombies_killed", 0))
        summary["plants_lost"] += int(diagnostics.get("plants_lost", 0))

    return summary


def _episode_diagnostics_row(
    detail: EpisodeEvalResult,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    action_stats = diagnostics.get("action_stats") or {}
    sun_stats = diagnostics.get("sun_stats") or {}
    return {
        "episode_index": detail.episode_index,
        "wait_actions": int(action_stats.get("wait", 0)),
        "plant_actions": int(action_stats.get("plant", 0)),
        "shovel_actions": int(action_stats.get("shovel", 0)),
        "invalid_actions": int(action_stats.get("invalid", 0)),
        "reward_breakdown": dict(diagnostics.get("reward_breakdown") or {}),
        "zombies_killed": int(diagnostics.get("zombies_killed", 0)),
        "plants_lost": int(diagnostics.get("plants_lost", 0)),
        "final_sun": int(sun_stats.get("final_sun", 0)),
        "max_sun": int(sun_stats.get("max_sun", 0)),
        "mean_sun": float(sun_stats.get("mean_sun", 0.0)),
        "sun_gained": int(sun_stats.get("sun_gained", 0)),
        "sun_spent": int(sun_stats.get("sun_spent", 0)),
        "wait_with_high_sun": int(sun_stats.get("wait_with_high_sun", 0)),
    }


def _plant_stats_sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _merge_count_map(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)


def _merge_float_map(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = float(target.get(str(key), 0.0)) + float(value)


def time_eval_run():
    return time.perf_counter()


def elapsed_since(start_time: float) -> float:
    return time.perf_counter() - start_time


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _optional_mean(values: Iterable[int | float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return _mean(present) if present else None


def _append_csv(path: str, fieldnames: list[str], row: dict[str, Any]) -> None:
    has_header = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def _summary_row(result: EvaluationResult) -> dict[str, Any]:
    row = asdict(result)
    row.pop("details", None)
    row["extra"] = json.dumps(row.get("extra") or {}, ensure_ascii=False)
    return row


def _detail_row(detail: EpisodeEvalResult) -> dict[str, Any]:
    row = asdict(detail)
    row["extra"] = json.dumps(row.get("extra") or {}, ensure_ascii=False)
    return row
