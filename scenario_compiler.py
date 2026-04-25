"""Compatibility facade for deterministic scenario compilation."""

from __future__ import annotations

from typing import Any

try:
    from .server.scenario_factory import ScenarioFactory
except ImportError:  # pragma: no cover
    from server.scenario_factory import ScenarioFactory


def compile_scenario(
    seed: int,
    split: str = "train",
    difficulty: int = 0,
    curriculum_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile one isolated authorization-repair scenario."""

    return ScenarioFactory().compile_scenario(
        seed,
        split=split,
        difficulty=difficulty,
        curriculum_profile=curriculum_profile,
    )
