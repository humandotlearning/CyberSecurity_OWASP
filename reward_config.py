"""Configurable reward shaping settings for CyberSecurity_OWASP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_GRPO_CONFIG_PATH = (
    Path(__file__).resolve().parent / "training" / "configs" / "grpo_small.yaml"
)
REWARD_MODES = {"dense_train", "sparse_eval"}
REWARD_STAGES = {"early", "middle", "late", "final"}


@dataclass(frozen=True)
class RewardSettings:
    """Loaded reward settings with stage-aware helpers."""

    mode: str
    training_mode: str
    stage: str
    raw: dict[str, Any]
    source_path: str

    @property
    def dense_train(self) -> bool:
        return self.mode == "dense_train"

    @property
    def shaping_weight(self) -> float:
        override = os.getenv("CYBERSECURITY_OWASP_SHAPING_WEIGHT")
        if override is not None:
            return float(override)
        return self.value("shaping_weight", 0.0)

    def entry(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}

    def value(self, name: str, default: float = 0.0) -> float:
        entry = self.entry(name)
        if self.stage in entry:
            return float(entry[self.stage])
        if "value" in entry:
            return float(entry["value"])
        return float(default)

    def cap(self, name: str, default: float | None = None) -> float | None:
        entry = self.entry(name)
        if "cap" not in entry:
            return default
        return float(entry["cap"])

    def int_value(self, name: str, key: str, default: int) -> int:
        entry = self.entry(name)
        return int(entry.get(key, default))

    def terminate(self, name: str) -> bool:
        return bool(self.entry(name).get("terminate", False))


def load_reward_settings(path: str | Path | None = None) -> RewardSettings:
    """Load reward settings from the GRPO YAML config with env overrides."""

    configured_path = Path(
        path
        or os.getenv("CYBERSECURITY_OWASP_REWARD_CONFIG", "")
        or DEFAULT_GRPO_CONFIG_PATH
    )
    raw = yaml.safe_load(configured_path.read_text(encoding="utf-8")) or {}
    reward = dict(raw.get("reward") or {})
    mode = os.getenv("CYBERSECURITY_OWASP_REWARD_MODE", str(reward.get("mode", "sparse_eval")))
    training_mode = str(reward.get("training_mode", "dense_train"))
    stage = os.getenv("CYBERSECURITY_OWASP_REWARD_STAGE", str(reward.get("stage", "early")))
    settings = RewardSettings(
        mode=mode,
        training_mode=training_mode,
        stage=stage,
        raw=reward,
        source_path=str(configured_path),
    )
    validate_reward_settings(settings)
    return settings


def validate_reward_settings(settings: RewardSettings) -> None:
    if settings.mode not in REWARD_MODES:
        raise ValueError("reward.mode must be dense_train or sparse_eval")
    if settings.training_mode not in REWARD_MODES:
        raise ValueError("reward.training_mode must be dense_train or sparse_eval")
    if settings.stage not in REWARD_STAGES:
        raise ValueError("reward.stage must be early, middle, late, or final")

    for key, value in settings.raw.items():
        if not isinstance(value, dict):
            continue
        if not str(value.get("description", "")).strip():
            raise ValueError(f"reward.{key}.description is required")


def compute_token_penalty(
    completion_tokens: int,
    settings: RewardSettings | None = None,
) -> float:
    """Return the trainer-side token penalty for a completion."""

    settings = settings or load_reward_settings()
    if not settings.dense_train:
        return 0.0
    target = settings.int_value("token_penalty", "target_tokens", 350)
    excess = max(0, int(completion_tokens) - target)
    penalty = settings.value("token_penalty", 0.0) * excess
    cap = settings.cap("token_penalty", -0.5)
    return max(penalty, cap if cap is not None else penalty)
