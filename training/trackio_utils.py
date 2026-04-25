"""Trackio helpers used by training and evaluation scripts."""

from __future__ import annotations

from datetime import datetime


TRAIN_METRICS = [
    "train/reward_total_mean",
    "train/reward_discovery_mean",
    "train/reward_security_mean",
    "train/reward_regression_mean",
    "train/reward_public_routes_mean",
    "train/reward_patch_quality_mean",
    "train/reward_visible_tests_mean",
    "train/reward_safety_mean",
    "train/reward_anti_cheat_mean",
    "train/success_rate",
    "train/exploit_block_rate",
    "train/regression_preservation_rate",
    "train/public_route_preservation_rate",
    "train/invalid_action_rate",
    "train/timeout_rate",
    "train/safety_violation_rate",
    "train/reward_hacking_suspected_rate",
    "train/episode_length_mean",
    "train/episode_length_p95",
    "train/rollouts_per_second",
    "train/tokens_per_second",
    "train/loss",
    "train/learning_rate",
    "train/kl",
    "train/grad_norm",
]


def build_run_name(model: str, algo: str, difficulty: int, git_sha: str = "nogit") -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
    model_slug = model.replace("/", "-")
    return f"CyberSecurity_OWASP-{model_slug}-{algo}-level{difficulty}-{stamp}-{git_sha[:8]}"
