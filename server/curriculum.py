"""Runtime curriculum controller for closed-loop scenario selection."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

try:
    from ..models import CyberSecurityOWASPState
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPState


DIFFICULTY_TIERS = ("warmup", "beginner", "intermediate", "advanced", "expert")
WEAKNESS_TARGETS = (
    "same_role_cross_object",
    "cross_tenant_boundary",
    "public_route_overlock",
    "alternate_route_same_service",
    "visible_test_edge_case",
)


@dataclass
class CurriculumController:
    """Tracks episode outcomes and picks the next bounded weakness target."""

    window_size: int = 10
    reward_trend: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    outcomes_by_target: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list))
    failures_by_target: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    episodes_seen: int = 0

    def select_profile(
        self,
        *,
        seed: int,
        split: str = "train",
        requested_difficulty: int = 0,
    ) -> dict[str, Any]:
        difficulty = self._difficulty_for_split(split, requested_difficulty)
        target = self._target_for_seed(seed, split)
        if self.failures_by_target:
            target = max(
                WEAKNESS_TARGETS,
                key=lambda item: (self.failures_by_target.get(item, 0), -WEAKNESS_TARGETS.index(item)),
            )
        return {
            "difficulty": difficulty,
            "difficulty_tier": DIFFICULTY_TIERS[min(difficulty, len(DIFFICULTY_TIERS) - 1)],
            "target_weakness": target,
            "split": split,
            "episodes_seen": self.episodes_seen,
            "recent_reward_mean": self._recent_reward_mean(),
            "mastery": self.mastery_snapshot(),
        }

    def record_episode(self, state: CyberSecurityOWASPState) -> dict[str, Any]:
        target = state.target_weakness or "same_role_cross_object"
        success = bool(state.success)
        self.episodes_seen += 1
        self.outcomes_by_target[target].append(success)
        if not success:
            self.failures_by_target[target] += 1
        self.reward_trend.append(float(state.last_reward or 0.0))
        return self.mastery_snapshot()

    def mastery_snapshot(self) -> dict[str, Any]:
        target_mastery = {}
        for target in WEAKNESS_TARGETS:
            outcomes = self.outcomes_by_target.get(target, [])
            target_mastery[target] = {
                "episodes": len(outcomes),
                "success_rate": sum(1 for item in outcomes if item) / max(1, len(outcomes)),
                "failures": self.failures_by_target.get(target, 0),
            }
        return {
            "episodes_seen": self.episodes_seen,
            "recent_reward_mean": self._recent_reward_mean(),
            "target_mastery": target_mastery,
        }

    def _difficulty_for_split(self, split: str, requested_difficulty: int) -> int:
        difficulty = max(0, min(int(requested_difficulty), len(DIFFICULTY_TIERS) - 1))
        if split == "hidden_eval":
            return max(3, difficulty)
        if self.episodes_seen >= self.window_size and self._recent_reward_mean() > 10.0:
            return min(difficulty + 1, len(DIFFICULTY_TIERS) - 1)
        return difficulty

    def _target_for_seed(self, seed: int, split: str) -> str:
        offset = 2 if split == "hidden_eval" else 0
        return WEAKNESS_TARGETS[(int(seed) + offset) % len(WEAKNESS_TARGETS)]

    def _recent_reward_mean(self) -> float:
        if not self.reward_trend:
            return 0.0
        return sum(self.reward_trend) / len(self.reward_trend)
