"""Episode artifact logging for training, debugging, and demos."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from ..models import CyberSecurityOWASPState
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPState


class EpisodeArtifactLogger:
    """Appends compact JSONL episode transcripts under outputs/rollouts."""

    def __init__(self, output_path: str | Path | None = None):
        configured = output_path or os.getenv("CYBERSECURITY_OWASP_EPISODE_LOG")
        self.output_path = Path(configured) if configured else Path("outputs/rollouts/episodes.jsonl")

    def log_episode(
        self,
        state: CyberSecurityOWASPState,
        *,
        final_observation: dict[str, Any] | None = None,
    ) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "episode_id": state.episode_id,
            "task_id": state.task_id,
            "seed": state.seed,
            "split": state.split,
            "difficulty": state.difficulty,
            "difficulty_tier": state.difficulty_tier,
            "template_id": state.template_id,
            "scenario_family": state.scenario_family,
            "domain": state.domain,
            "bug_family": state.bug_family,
            "target_weakness": state.target_weakness,
            "agent_actions": state.action_history,
            "observations": state.observation_history,
            "final_observation": final_observation or {},
            "patch_diff": state.patch_diff,
            "visible_test_result": self._verifier_layer(state, "visible"),
            "hidden_test_result": self._verifier_layer(state, "hidden_tests"),
            "oracle_result": self._verifier_layer(state, "oracle_matrix"),
            "regression_result": self._verifier_layer(state, "regression"),
            "reward_breakdown": state.reward_history[-1] if state.reward_history else {},
            "reward_breakdown_by_step": state.reward_history,
            "total_reward": state.accumulated_reward,
            "final_reward_breakdown": state.reward_history[-1] if state.reward_history else {},
            "progress_reward_total": state.progress_reward_total,
            "completion_tokens": state.completion_tokens,
            "diagnosis_submitted": state.diagnosis_submitted,
            "diagnosis": state.diagnosis,
            "request_trace": state.request_trace,
            "final_status": "resolved" if state.success else "failed",
            "failure_reason": state.failure_reason,
            "safety_violations": [
                flag for flag in state.anti_cheat_flags if "network" in flag or "unsafe" in flag
            ],
            "anti_cheat_flags": state.anti_cheat_flags,
            "metrics": state.metrics,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        state.episode_artifact_path = str(self.output_path)
        return self.output_path

    def _verifier_layer(self, state: CyberSecurityOWASPState, key: str) -> Any:
        return (state.verification_summary or {}).get(key)
