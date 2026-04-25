"""Baseline-vs-trained evaluation scaffold for CyberSecurity_OWASP."""

from __future__ import annotations

import json
from pathlib import Path


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
    }


def save_eval_summary(run_name: str, summary: dict) -> Path:
    output = Path("outputs/evals") / f"{run_name}_eval_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output
