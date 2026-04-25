"""Scenario grouping and adaptive curriculum helpers for GRPO training."""

from __future__ import annotations

import random
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdaptiveDifficultyCurriculum:
    min_level: int = 0
    max_level: int = 3
    current_level: int = 0
    promote_after: int = 50
    promote_threshold: float = 0.70
    demote_threshold: float = 0.35
    ema_alpha: float = 0.10
    rng_seed: int = 0
    counts: dict[int, int] = field(default_factory=dict)
    ema_success: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.min_level = int(self.min_level)
        self.max_level = int(self.max_level)
        self.current_level = max(self.min_level, min(int(self.current_level), self.max_level))
        self._rng = random.Random(int(self.rng_seed))

    def sample_difficulty(self, available_difficulties: Iterable[int]) -> int:
        available = {int(item) for item in available_difficulties}
        if not available:
            raise ValueError("No cached difficulties are available for GRPO curriculum sampling.")

        candidates = [
            max(self.min_level, self.current_level - 1),
            self.current_level,
            min(self.max_level, self.current_level + 1),
        ]
        weights = [0.20, 0.65, 0.15]
        weighted: dict[int, float] = {}
        for level, weight in zip(candidates, weights):
            if level in available:
                weighted[level] = weighted.get(level, 0.0) + weight

        if not weighted:
            nearest = min(available, key=lambda level: (abs(level - self.current_level), level))
            return nearest
        levels = list(weighted)
        return int(self._rng.choices(levels, weights=[weighted[level] for level in levels], k=1)[0])

    def update(self, difficulty: int, success: float | bool) -> dict[str, Any]:
        level = int(difficulty)
        value = max(0.0, min(1.0, float(success)))
        self.counts[level] = self.counts.get(level, 0) + 1
        old = self.ema_success.get(level, 0.0)
        self.ema_success[level] = (1.0 - self.ema_alpha) * old + self.ema_alpha * value

        if level == self.current_level and self.counts[level] >= self.promote_after:
            if self.ema_success[level] >= self.promote_threshold:
                self.current_level = min(self.max_level, self.current_level + 1)
            elif self.ema_success[level] <= self.demote_threshold:
                self.current_level = max(self.min_level, self.current_level - 1)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_level": self.current_level,
            "counts": {str(key): value for key, value in sorted(self.counts.items())},
            "ema_success": {
                str(key): value for key, value in sorted(self.ema_success.items())
            },
            "current_level_ema_success": self.ema_success.get(self.current_level, 0.0),
        }


class ScenarioGroupRegistry:
    """Assign each GRPO group to exactly one cached scenario."""

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        split: str = "train",
        initial_difficulty: int = 0,
        rng_seed: int = 0,
        max_level: int | None = None,
    ) -> None:
        self.split = split
        self._rng = random.Random(int(rng_seed))
        self._lock = threading.Lock()
        self._assignments: dict[int, dict[str, Any]] = {}
        self._completed_groups: set[int] = set()
        self._entries_by_difficulty: dict[int, list[dict[str, Any]]] = {}
        self._cursors: dict[int, int] = {}

        for entry in entries:
            if entry.get("validated") is not True or entry.get("split") != split:
                continue
            difficulty = int(entry.get("difficulty", 0))
            self._entries_by_difficulty.setdefault(difficulty, []).append(dict(entry))

        for difficulty, items in self._entries_by_difficulty.items():
            items.sort(key=lambda item: (int(item.get("seed", 0)), str(item.get("scenario_hash", ""))))
            self._rng.shuffle(items)
            self._cursors[difficulty] = 0

        if not self._entries_by_difficulty:
            raise ValueError(f"No validated cached scenarios are available for split={split!r}.")

        available = sorted(self._entries_by_difficulty)
        resolved_max = max_level if max_level is not None else max(available)
        self.curriculum = AdaptiveDifficultyCurriculum(
            min_level=min(available),
            max_level=int(resolved_max),
            current_level=int(initial_difficulty),
            rng_seed=int(rng_seed),
        )

    @property
    def available_difficulties(self) -> list[int]:
        return sorted(self._entries_by_difficulty)

    def assignment_for(
        self,
        *,
        scenario_group_id: int,
        requested_seed: int | None = None,
        requested_difficulty: int | None = None,
        split: str | None = None,
        difficulty_policy: str = "adaptive",
    ) -> dict[str, Any]:
        group_id = int(scenario_group_id)
        with self._lock:
            if group_id in self._assignments:
                return dict(self._assignments[group_id])

            if difficulty_policy == "fixed":
                difficulty = int(
                    requested_difficulty
                    if requested_difficulty is not None
                    else self.curriculum.current_level
                )
                entry = self._find_entry(
                    seed=requested_seed,
                    split=split or self.split,
                    difficulty=difficulty,
                ) or self._next_entry(difficulty)
            else:
                difficulty = self.curriculum.sample_difficulty(self.available_difficulties)
                entry = self._next_entry(difficulty)

            assignment = self._assignment_from_entry(group_id, entry)
            self._assignments[group_id] = assignment
            return dict(assignment)

    def record_group_outcome(self, scenario_group_id: int, success_rate: float) -> dict[str, Any] | None:
        group_id = int(scenario_group_id)
        with self._lock:
            if group_id in self._completed_groups:
                return None
            self._completed_groups.add(group_id)
            assignment = self._assignments.get(group_id)
            if not assignment:
                return self.curriculum.snapshot()
            return self.curriculum.update(
                int(assignment["difficulty"]),
                max(0.0, min(1.0, float(success_rate))),
            )

    def metrics(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        unique_trace_count: int,
        duplicate_trace_suppressed_count: int,
    ) -> dict[str, float]:
        scenario_hashes = {
            str(record.get("scenario_hash") or record.get("scenario_id_hash") or "")
            for record in records
            if record.get("scenario_hash") or record.get("scenario_id_hash")
        }
        seeds = {
            int(record.get("scenario/seed", record.get("seed", 0)) or 0)
            for record in records
        }
        total = max(1, len(records))
        snapshot = self.curriculum.snapshot()
        return {
            "train/unique_trace_count": float(unique_trace_count),
            "train/duplicate_trace_suppressed_count": float(duplicate_trace_suppressed_count),
            "train/unique_trace_rate": float(unique_trace_count) / total,
            "train/unique_seed_count": float(len(seeds)),
            "train/unique_scenario_hash_count": float(len(scenario_hashes)),
            "train/curriculum_level": float(snapshot["current_level"]),
            "train/curriculum_ema_success": float(snapshot["current_level_ema_success"]),
        }

    def _find_entry(
        self,
        *,
        seed: int | None,
        split: str,
        difficulty: int,
    ) -> dict[str, Any] | None:
        if seed is None or split != self.split:
            return None
        for entry in self._entries_by_difficulty.get(int(difficulty), []):
            if int(entry.get("seed", -1)) == int(seed):
                return dict(entry)
        return None

    def _next_entry(self, difficulty: int) -> dict[str, Any]:
        level = int(difficulty)
        items = self._entries_by_difficulty.get(level)
        if not items:
            nearest = min(
                self.available_difficulties,
                key=lambda item: (abs(item - level), item),
            )
            items = self._entries_by_difficulty[nearest]
            level = nearest
        cursor = self._cursors.get(level, 0)
        self._cursors[level] = cursor + 1
        return dict(items[cursor % len(items)])

    def _assignment_from_entry(self, group_id: int, entry: Mapping[str, Any]) -> dict[str, Any]:
        cache_key = entry.get("cache_key") if isinstance(entry.get("cache_key"), Mapping) else {}
        return {
            "scenario_group_id": int(group_id),
            "seed": int(entry.get("seed", 0)),
            "split": str(entry.get("split", self.split)),
            "difficulty": int(entry.get("difficulty", 0)),
            "scenario_hash": str(entry.get("scenario_hash", "")),
            "template_id": str(entry.get("template_id") or cache_key.get("app_family", "")),
            "bug_family": str(entry.get("bug_family") or cache_key.get("authz_bug_type", "")),
        }


def build_scenario_group_rows(
    *,
    dataset_size: int,
    training_prompt: str,
    seed_start: int = 0,
    split: str = "train",
    difficulty: int = 0,
    difficulty_policy: str = "adaptive",
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": [{"role": "user", "content": training_prompt}],
            "scenario_group_id": int(seed_start) + index,
            "seed": int(seed_start) + index,
            "difficulty": int(difficulty),
            "split": split,
            "difficulty_policy": difficulty_policy,
        }
        for index in range(int(dataset_size))
    ]
