"""Configurable reward shaping settings for CyberSecurity_OWASP."""

from __future__ import annotations

import hashlib
import json
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
    raw = _load_yaml_with_extends(configured_path)
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


def _load_yaml_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML file, recursively merging an optional relative `extends` file."""

    resolved_path = path.expanduser().resolve()
    seen = seen or set()
    if resolved_path in seen:
        chain = " -> ".join(str(item) for item in [*seen, resolved_path])
        raise ValueError(f"reward config extends cycle detected: {chain}")
    seen.add(resolved_path)

    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"reward config must be a YAML mapping: {resolved_path}")

    extends = raw.get("extends")
    if not extends:
        return raw
    if not isinstance(extends, str):
        raise ValueError("reward config extends must be a string path")

    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = resolved_path.parent / base_path
    child = {key: value for key, value in raw.items() if key != "extends"}
    return _deep_merge(_load_yaml_with_extends(base_path, seen), child)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def flatten_reward_config(
    settings: RewardSettings | None = None,
) -> list[dict[str, Any]]:
    """Return display-friendly reward config rows for tracking dashboards."""

    settings = settings or load_reward_settings()
    rows: list[dict[str, Any]] = []
    for key in sorted(settings.raw):
        entry = settings.raw[key]
        if not isinstance(entry, dict):
            continue
        has_resolved_value = "value" in entry or settings.stage in entry
        rows.append(
            {
                "key": key,
                "value": _empty_if_missing(entry.get("value")),
                "stage_value": _empty_if_missing(entry.get(settings.stage)),
                "resolved": settings.value(key, 0.0) if has_resolved_value else "",
                "cap": _empty_if_missing(entry.get("cap")),
                "threshold": _empty_if_missing(
                    entry.get("threshold", entry.get("threshold_lines"))
                ),
                "severe_threshold": _empty_if_missing(
                    entry.get("severe_threshold", entry.get("severe_threshold_lines"))
                ),
                "terminate": bool(entry.get("terminate", False)),
                "description": str(entry.get("description", "")),
            }
        )
    return rows


def reward_config_hash(settings: RewardSettings | None = None) -> str:
    """Return a deterministic hash for the effective reward configuration."""

    settings = settings or load_reward_settings()
    payload = {
        "mode": settings.mode,
        "training_mode": settings.training_mode,
        "stage": settings.stage,
        "shaping_weight": settings.shaping_weight,
        "raw": _strip_descriptions(settings.raw),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reward_config_summary(settings: RewardSettings | None = None) -> dict[str, Any]:
    """Return reward config identity and flattened rows for run metadata."""

    settings = settings or load_reward_settings()
    config_hash = reward_config_hash(settings)
    source = Path(settings.source_path)
    return {
        "reward_config_id": (
            f"{source.stem}-{settings.mode}-{settings.stage}-{config_hash[:12]}"
        ),
        "reward_config_hash": config_hash,
        "reward_config_source": str(source),
        "reward_config_source_name": source.name,
        "reward_mode": settings.mode,
        "reward_training_mode": settings.training_mode,
        "reward_stage": settings.stage,
        "reward_shaping_weight": settings.shaping_weight,
        "reward_entries": flatten_reward_config(settings),
    }


def reward_config_run_config(settings: RewardSettings | None = None) -> dict[str, Any]:
    """Return compact reward config fields safe to store in Trackio run config."""

    summary = reward_config_summary(settings)
    reward_values = {
        str(row["key"]): {
            key: value
            for key, value in row.items()
            if key != "key" and value != ""
        }
        for row in summary["reward_entries"]
    }
    config = {
        "reward_config_id": summary["reward_config_id"],
        "reward_config_hash": summary["reward_config_hash"],
        "reward_config_source": summary["reward_config_source"],
        "reward_config_source_name": summary["reward_config_source_name"],
        "reward_variant": os.getenv("CYBERSECURITY_OWASP_REWARD_VARIANT", "default") or "default",
        "reward_mode": summary["reward_mode"],
        "reward_training_mode": summary["reward_training_mode"],
        "reward_stage": summary["reward_stage"],
        "reward_shaping_weight": summary["reward_shaping_weight"],
        "reward_config_values": reward_values,
        "reward_config_values_json": json.dumps(reward_values, sort_keys=True),
    }
    for reward_key, values in reward_values.items():
        safe_reward_key = _config_key_safe(reward_key)
        for field, value in values.items():
            if isinstance(value, (int, float, bool)):
                config[f"reward_config__{safe_reward_key}__{field}"] = value
    return config


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


def _empty_if_missing(value: Any) -> Any:
    return "" if value is None else value


def _strip_descriptions(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_descriptions(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_strip_descriptions(item) for item in value]
    return value


def _config_key_safe(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value).strip("_")


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
