"""Configuration for scenario authoring, curriculum, and cache-backed reset."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ScenarioCacheMode = Literal["fallback", "require", "disabled"]


DEFAULT_SCENARIO_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "scenario_authoring.small.json"
)


@dataclass(frozen=True)
class ScenarioAuthorConfig:
    provider: str = "huggingface"
    model_id: str = "deepseek-ai/DeepSeek-V4-Pro"
    thinking_mode: str = "thinking"
    reasoning_effort: str = "high"
    temperature: float = 1.0
    top_p: float = 1.0
    max_context_tokens: int = 131072


@dataclass(frozen=True)
class CurriculumCacheConfig:
    difficulty_bucket_count: int = 4
    difficulty_labels: list[str] = field(default_factory=lambda: ["D0", "D1", "D2", "D3"])
    train_scenarios_per_bucket: int = 25
    validation_scenarios_per_bucket: int = 10
    heldout_eval_scenarios_per_bucket: int = 10
    target_cache_hit_rate: float = 0.95
    target_reset_latency_ms: int = 200
    scenario_refresh_rate_per_epoch: float = 0.05
    difficulty_calibration_strategy: str = "baseline_agent_pass_rate"
    pass_rate_thresholds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "D0": (0.8, 1.0),
            "D1": (0.6, 0.8),
            "D2": (0.4, 0.6),
            "D3": (0.2, 0.4),
        }
    )

    def minimum_for_split(self, split: str) -> int:
        if split == "hidden_eval":
            return self.heldout_eval_scenarios_per_bucket
        if split == "validation":
            return self.validation_scenarios_per_bucket
        return self.train_scenarios_per_bucket


@dataclass(frozen=True)
class ScenarioRuntimeConfig:
    cache_mode: ScenarioCacheMode = "fallback"
    cache_dir: str = "scenario_cache"
    generator_version: str = "scenario_generator_v1"
    verifier_version: str = "verifier_v1"


@dataclass(frozen=True)
class ScenarioAuthoringSettings:
    scenario_author: ScenarioAuthorConfig = field(default_factory=ScenarioAuthorConfig)
    curriculum: CurriculumCacheConfig = field(default_factory=CurriculumCacheConfig)
    runtime: ScenarioRuntimeConfig = field(default_factory=ScenarioRuntimeConfig)
    source_path: str = ""


def load_scenario_authoring_config(path: str | Path | None = None) -> ScenarioAuthoringSettings:
    """Load and validate the small scenario-authoring config with env overrides."""

    configured_path = Path(
        path
        or os.getenv("CYBERSECURITY_OWASP_SCENARIO_CONFIG", "")
        or DEFAULT_SCENARIO_CONFIG_PATH
    )
    raw = json.loads(configured_path.read_text(encoding="utf-8"))
    raw = _apply_env_overrides(raw)
    settings = ScenarioAuthoringSettings(
        scenario_author=ScenarioAuthorConfig(**raw.get("scenario_author", {})),
        curriculum=_curriculum_from_raw(raw.get("curriculum", {})),
        runtime=ScenarioRuntimeConfig(**raw.get("runtime", {})),
        source_path=str(configured_path),
    )
    _validate_settings(settings)
    return settings


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(raw))
    author = data.setdefault("scenario_author", {})
    curriculum = data.setdefault("curriculum", {})
    runtime = data.setdefault("runtime", {})

    _set_if_present(author, "model_id", "CYBERSECURITY_OWASP_SCENARIO_AUTHOR_MODEL")
    _set_if_present(author, "provider", "CYBERSECURITY_OWASP_SCENARIO_AUTHOR_PROVIDER")
    _set_if_present(author, "thinking_mode", "CYBERSECURITY_OWASP_SCENARIO_THINKING_MODE")
    _set_if_present(author, "reasoning_effort", "CYBERSECURITY_OWASP_SCENARIO_REASONING_EFFORT")
    _set_if_present(author, "temperature", "CYBERSECURITY_OWASP_SCENARIO_TEMPERATURE", float)
    _set_if_present(author, "top_p", "CYBERSECURITY_OWASP_SCENARIO_TOP_P", float)
    _set_if_present(author, "max_context_tokens", "CYBERSECURITY_OWASP_SCENARIO_MAX_CONTEXT", int)

    _set_if_present(curriculum, "difficulty_bucket_count", "CYBERSECURITY_OWASP_DIFFICULTY_BUCKETS", int)
    _set_if_present(curriculum, "train_scenarios_per_bucket", "CYBERSECURITY_OWASP_TRAIN_SCENARIOS_PER_BUCKET", int)
    _set_if_present(curriculum, "validation_scenarios_per_bucket", "CYBERSECURITY_OWASP_VALIDATION_SCENARIOS_PER_BUCKET", int)
    _set_if_present(curriculum, "heldout_eval_scenarios_per_bucket", "CYBERSECURITY_OWASP_HELDOUT_SCENARIOS_PER_BUCKET", int)
    _set_if_present(curriculum, "target_cache_hit_rate", "CYBERSECURITY_OWASP_TARGET_CACHE_HIT_RATE", float)
    _set_if_present(curriculum, "target_reset_latency_ms", "CYBERSECURITY_OWASP_TARGET_RESET_LATENCY_MS", int)
    _set_if_present(curriculum, "scenario_refresh_rate_per_epoch", "CYBERSECURITY_OWASP_SCENARIO_REFRESH_RATE", float)
    _set_if_present(curriculum, "difficulty_calibration_strategy", "CYBERSECURITY_OWASP_DIFFICULTY_CALIBRATION")

    _set_if_present(runtime, "cache_dir", "CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR")
    _set_if_present(runtime, "cache_mode", "CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE")
    _set_if_present(runtime, "generator_version", "CYBERSECURITY_OWASP_SCENARIO_GENERATOR_VERSION")
    _set_if_present(runtime, "verifier_version", "CYBERSECURITY_OWASP_SCENARIO_VERIFIER_VERSION")
    return data


def _set_if_present(
    target: dict[str, Any],
    key: str,
    env_name: str,
    caster: type | None = None,
) -> None:
    value = os.getenv(env_name)
    if value is None:
        return
    target[key] = caster(value) if caster else value


def _curriculum_from_raw(raw: dict[str, Any]) -> CurriculumCacheConfig:
    values = dict(raw)
    bucket_count = int(values.get("difficulty_bucket_count", 4))
    labels = list(values.get("difficulty_labels") or [])
    if len(labels) < bucket_count:
        labels.extend(f"D{index}" for index in range(len(labels), bucket_count))
    values["difficulty_labels"] = labels
    thresholds = values.get("pass_rate_thresholds") or {}
    values["pass_rate_thresholds"] = {
        str(key): tuple(float(item) for item in value)
        for key, value in thresholds.items()
    }
    return CurriculumCacheConfig(**values)


def _validate_settings(settings: ScenarioAuthoringSettings) -> None:
    author = settings.scenario_author
    curriculum = settings.curriculum
    runtime = settings.runtime

    if not author.model_id:
        raise ValueError("scenario_author.model_id is required")
    if author.temperature <= 0.0 or author.top_p <= 0.0:
        raise ValueError("scenario author sampling values must be positive")
    if author.max_context_tokens < 4096:
        raise ValueError("scenario author max_context_tokens is too small")
    if curriculum.difficulty_bucket_count <= 0:
        raise ValueError("difficulty_bucket_count must be positive")
    if len(curriculum.difficulty_labels) < curriculum.difficulty_bucket_count:
        raise ValueError("difficulty_labels must cover every configured bucket")
    for attr in (
        "train_scenarios_per_bucket",
        "validation_scenarios_per_bucket",
        "heldout_eval_scenarios_per_bucket",
        "target_reset_latency_ms",
    ):
        if int(getattr(curriculum, attr)) <= 0:
            raise ValueError(f"{attr} must be positive")
    if not 0.0 < curriculum.target_cache_hit_rate <= 1.0:
        raise ValueError("target_cache_hit_rate must be in (0, 1]")
    if not 0.0 <= curriculum.scenario_refresh_rate_per_epoch <= 1.0:
        raise ValueError("scenario_refresh_rate_per_epoch must be in [0, 1]")
    if runtime.cache_mode not in {"fallback", "require", "disabled"}:
        raise ValueError("runtime.cache_mode must be fallback, require, or disabled")
