"""Trackio helpers used by training and evaluation scripts."""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


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


EVAL_METRICS = [
    "eval/baseline_success_rate",
    "eval/trained_success_rate",
    "eval/absolute_success_improvement",
    "eval/baseline_mean_reward",
    "eval/trained_mean_reward",
    "eval/absolute_reward_improvement",
    "eval/heldout_success_rate",
    "eval/heldout_mean_reward",
    "eval/exploit_block_rate",
    "eval/regression_preservation_rate",
    "eval/public_route_preservation_rate",
    "eval/anti_cheat_pass_rate",
    "eval/invalid_action_rate",
    "eval/timeout_rate",
    "eval/safety_violation_rate",
    "eval/mean_episode_length",
]


def build_run_name(model: str, algo: str, difficulty: int, git_sha: str = "nogit") -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_slug = model.replace("/", "-")
    return f"CyberSecurity_OWASP-{model_slug}-{algo}-level{difficulty}-{stamp}-{git_sha[:8]}"


def get_git_sha(default: str = "nogit") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return default
    return result.stdout.strip() or default


def _load_trackio():
    os.environ.setdefault("TRACKIO_DIR", str((Path.cwd() / "outputs" / "trackio").resolve()))
    try:
        import trackio
    except ImportError as exc:
        raise RuntimeError(
            "Trackio is required for CyberSecurity_OWASP runs. Install dependencies "
            "with `uv sync` and set TRACKIO_SPACE_ID when you want remote HF Spaces tracking."
        ) from exc
    return trackio


def init_trackio_run(
    *,
    run_name: str,
    run_type: str,
    config: dict[str, Any] | None = None,
    project: str | None = None,
    space_id: str | None = None,
    group: str | None = None,
):
    trackio = _load_trackio()
    project = project or os.getenv("TRACKIO_PROJECT", "CyberSecurity_OWASP")
    space_id = space_id if space_id is not None else os.getenv("TRACKIO_SPACE_ID", "")
    run_config = {
        "environment": "CyberSecurity_OWASP",
        "run_type": run_type,
        **(config or {}),
    }
    kwargs: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "config": run_config,
    }
    if space_id:
        kwargs["space_id"] = space_id
    if group:
        kwargs["group"] = group
    return trackio.init(**kwargs)


def log_trackio_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    trackio = _load_trackio()
    numeric = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool))
    }
    if step is None:
        trackio.log(numeric)
    else:
        trackio.log(numeric, step=step)


def finish_trackio_run() -> None:
    trackio = _load_trackio()
    trackio.finish()


@contextmanager
def trackio_run(
    *,
    run_name: str,
    run_type: str,
    config: dict[str, Any] | None = None,
    project: str | None = None,
    space_id: str | None = None,
    group: str | None = None,
) -> Iterator[Any]:
    run = init_trackio_run(
        run_name=run_name,
        run_type=run_type,
        config=config,
        project=project,
        space_id=space_id,
        group=group,
    )
    try:
        yield run
    finally:
        finish_trackio_run()


def log_eval_summary(run_name: str, summary: dict[str, Any], config: dict[str, Any] | None = None) -> None:
    metrics = {
        f"eval/{key}": float(value)
        for key, value in summary.items()
        if isinstance(value, (int, float, bool))
    }
    with trackio_run(run_name=run_name, run_type="eval", config=config, group="eval"):
        log_trackio_metrics(metrics, step=0)
