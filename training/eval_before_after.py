"""Baseline-vs-trained evaluation scaffold for CyberSecurity_OWASP."""

from __future__ import annotations

import json
from pathlib import Path

from training.trackio_utils import log_eval_summary


def summarize_runs(baseline: list[dict], trained: list[dict], heldout: list[dict]) -> dict:
    def mean(items: list[dict], key: str) -> float:
        return sum(float(item.get(key, 0.0)) for item in items) / max(1, len(items))

    return {
        "baseline_success_rate": mean(baseline, "success"),
        "trained_success_rate": mean(trained, "success"),
        "absolute_success_improvement": mean(trained, "success") - mean(baseline, "success"),
        "baseline_mean_reward": mean(baseline, "reward_total"),
        "trained_mean_reward": mean(trained, "reward_total"),
        "absolute_reward_improvement": mean(trained, "reward_total") - mean(baseline, "reward_total"),
        "heldout_success_rate": mean(heldout, "success"),
        "heldout_mean_reward": mean(heldout, "reward_total"),
        "exploit_block_rate": mean(trained, "exploit_blocked"),
        "regression_preservation_rate": mean(trained, "regression_preserved"),
        "public_route_preservation_rate": mean(trained, "public_routes_preserved"),
        "anti_cheat_pass_rate": mean(trained, "anti_cheat_pass"),
        "invalid_action_rate": mean(trained, "invalid_action_rate"),
        "timeout_rate": mean(trained, "timeout"),
        "safety_violation_rate": mean(trained, "safety_violation"),
        "mean_episode_length": mean(trained, "episode_length"),
    }


def save_eval_summary(
    run_name: str,
    summary: dict,
    *,
    track: bool = True,
    trackio_config: dict | None = None,
) -> Path:
    output = Path("outputs/evals") / f"{run_name}_eval_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if track:
        log_eval_summary(run_name, summary, config=trackio_config)
    return output
