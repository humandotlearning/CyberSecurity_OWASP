"""Ephemeral Modal Labs launcher for CyberSecurity_OWASP training smoke runs.

Run from the repo root:

    modal run scripts/modal_ephemeral_train.py --mode smoke --episodes 4

This intentionally stays separate from ``training/train_grpo.py``. It packages
the local repo into a temporary Modal app and returns compact JSON artifacts to
the local process, so the run disappears when ``modal run`` exits.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = "CyberSecurity_OWASP-ephemeral-training"
SECRET_NAME = "CyberSecurity_OWASP-secrets"
SCENARIO_CACHE_VOLUME_NAME = "CyberSecurity_OWASP-scenario-cache"
SCENARIO_CACHE_DIR = Path("/scenario-cache")
REMOTE_PROJECT = "/root/CyberSecurity_OWASP"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)
scenario_cache_volume = modal.Volume.from_name(SCENARIO_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("openenv-core[core]>=0.2.2", "trackio>=0.22.0")
    .add_local_dir(
        PROJECT_ROOT,
        remote_path=REMOTE_PROJECT,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            ".env",
            ".env.*",
            "__pycache__",
            ".pytest_cache",
            "outputs",
            "*.pyc",
        ],
    )
    .run_commands(f"pip install --no-deps -e {REMOTE_PROJECT}")
    .workdir(REMOTE_PROJECT)
)


def _configure_scenario_cache_env(*, required: bool = True) -> None:
    SCENARIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR"] = str(SCENARIO_CACHE_DIR)
    os.environ["CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE"] = "require" if required else "fallback"


class NoopTrainer:
    """Deterministic placeholder policy for cheap Modal smoke runs."""

    def generate_rollout_completions(self, prompts: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "text": '{"tool_name":"noop","arguments":{}}',
                "prompt_ids": [],
                "completion_ids": [],
                "logprobs": [],
            }
            for _ in prompts
        ]


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={SCENARIO_CACHE_DIR: scenario_cache_volume},
)
def prepare_ephemeral_scenario_cache(
    seed_start: int = 0,
    difficulty_buckets: int = 0,
    train_per_bucket: int = 0,
    validation_per_bucket: int = 0,
    heldout_per_bucket: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    import os

    if difficulty_buckets:
        os.environ["CYBERSECURITY_OWASP_DIFFICULTY_BUCKETS"] = str(difficulty_buckets)
    if train_per_bucket:
        os.environ["CYBERSECURITY_OWASP_TRAIN_SCENARIOS_PER_BUCKET"] = str(train_per_bucket)
    if validation_per_bucket:
        os.environ["CYBERSECURITY_OWASP_VALIDATION_SCENARIOS_PER_BUCKET"] = str(validation_per_bucket)
    if heldout_per_bucket:
        os.environ["CYBERSECURITY_OWASP_HELDOUT_SCENARIOS_PER_BUCKET"] = str(heldout_per_bucket)
    _configure_scenario_cache_env(required=False)
    from CyberSecurity_OWASP.config import load_scenario_authoring_config
    from CyberSecurity_OWASP.server.scenario_cache import prepare_scenario_cache

    settings = load_scenario_authoring_config()
    result = prepare_scenario_cache(
        cache_dir=SCENARIO_CACHE_DIR,
        settings=settings,
        seed_start=seed_start,
        force=force,
    )
    scenario_cache_volume.commit()
    result["scenario_cache_volume"] = SCENARIO_CACHE_VOLUME_NAME
    return result


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={SCENARIO_CACHE_DIR: scenario_cache_volume},
    secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN"])],
)
def run_ephemeral_smoke(
    episodes: int = 4,
    seed_start: int = 0,
    trackio_space_id: str = "",
    trackio_project: str = "CyberSecurity_OWASP-smoke",
) -> dict[str, Any]:
    _configure_scenario_cache_env(required=True)
    from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
    from CyberSecurity_OWASP.config import load_scenario_authoring_config
    from CyberSecurity_OWASP.reward_config import load_reward_settings
    from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
        CybersecurityOwaspEnvironment,
    )
    from CyberSecurity_OWASP.server.scenario_cache import ScenarioCache
    from training.rollout import rollout_once
    from training.trackio_utils import (
        aggregate_episode_metrics,
        episode_record_from_state,
        log_episode_batch,
        log_reward_config,
        log_trackio_metrics,
        reward_config_trackio_config,
        trace_table_rows,
        trackio_run,
    )

    scenario_cache_volume.reload()
    settings = load_scenario_authoring_config()
    cache_coverage = ScenarioCache(SCENARIO_CACHE_DIR, settings=settings).assert_coverage(
        split="validation",
        difficulty=0,
    )
    available_scenarios = int(
        cache_coverage.get("counts", {}).get("validation", {}).get("0", 0)
    )
    if available_scenarios < episodes:
        raise RuntimeError(
            "Scenario cache does not cover this smoke run. Run prepare-cache "
            f"with a larger validation count. available={available_scenarios}, episodes={episodes}"
        )

    baseline = []
    oracle = []
    reward_settings = load_reward_settings()
    reward_tracking_config = reward_config_trackio_config(reward_settings)
    run_context = {
        "algo": "modal_ephemeral_smoke",
        "reward_version": "reward_v2",
        "env_version": "0.1.0",
        **reward_tracking_config,
    }

    for offset in range(episodes):
        seed = seed_start + offset

        baseline_env = CybersecurityOwaspEnvironment()
        baseline_rollout = rollout_once(
            NoopTrainer(),
            baseline_env,
            max_steps=5,
            reset_kwargs={"seed": seed, "split": "validation", "difficulty": 0},
        )
        baseline_record = episode_record_from_state(
            baseline_env.state,
            run_context={**run_context, "base_model": "noop"},
        )
        baseline_record.update(
            {
                "reward_total": baseline_rollout.get("reward_total", 0.0),
                "success": baseline_rollout.get("success", False),
                "episode_length": baseline_rollout.get("episode_length", 0),
            }
        )
        baseline.append(baseline_record)

        oracle_env = CybersecurityOwaspEnvironment()
        oracle_env.reset(seed=seed, split="validation")
        hidden = oracle_env.state.hidden_facts
        evidence = oracle_env.step(
            CyberSecurityOWASPAction(
                tool_name="send_local_request",
                arguments={
                    "method": "GET",
                    "path": f"/invoices/{hidden['other_invoice_id']}",
                    "user_id": hidden["owner_user_id"],
                },
            )
        )
        trace_id = "req_001"
        if '"trace_id": "req_' in evidence.last_tool_result:
            trace_id = evidence.last_tool_result.split('"trace_id": "', 1)[1].split('"', 1)[0]
        oracle_env.step(
            CyberSecurityOWASPAction(
                tool_name="submit_diagnosis",
                arguments={
                    "bug_class": "idor_ownership_bug",
                    "route": "GET /invoices/{invoice_id}",
                    "violated_policy_rule": "Only owner or billing_admin in same tenant may read invoices.",
                    "evidence_trace_ids": [trace_id],
                    "fix_plan": "Add tenant and owner/admin checks before returning invoice data.",
                },
            )
        )
        source = (
            Path(hidden["workspace"]) / "app/routes/invoices.py"
        ).read_text(encoding="utf-8")
        fixed = source.replace(
            "    # BUG: this only checks that the caller is authenticated. It forgets the\n"
            "    # owner/admin and tenant policy checks required by the policy graph.\n"
            "    return {\"status\": 200, \"body\": invoice}\n",
            "    if invoice[\"tenant_id\"] != actor[\"tenant_id\"]:\n"
            "        return {\"status\": 403, \"body\": {\"detail\": \"forbidden\"}}\n"
            "    if invoice[\"owner_user_id\"] != actor[\"user_id\"] and not is_billing_admin(actor):\n"
            "        return {\"status\": 403, \"body\": {\"detail\": \"forbidden\"}}\n"
            "    return {\"status\": 200, \"body\": invoice}\n",
        )
        oracle_env.step(
            CyberSecurityOWASPAction(
                tool_name="patch_file",
                arguments={"path": "app/routes/invoices.py", "content": fixed},
            )
        )
        oracle_env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
        final = oracle_env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))
        oracle_record = episode_record_from_state(
            oracle_env.state,
            run_context={**run_context, "base_model": "oracle"},
            final_observation=final.model_dump(),
        )
        oracle_record.update(
            {
                "reward_total": final.reward_breakdown.get("total", 0.0),
                "success": oracle_env.state.success,
            }
        )
        oracle.append(oracle_record)

    def mean(items: list[dict[str, Any]], key: str) -> float:
        return sum(float(item.get(key, 0.0)) for item in items) / max(1, len(items))

    run_name = f"{APP_NAME}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    episode_records = [*baseline, *oracle]
    tracking_metrics = aggregate_episode_metrics(episode_records)
    result = {
        "run_name": run_name,
        "mode": "smoke",
        "episodes": episodes,
        "seed_start": seed_start,
        "baseline_mean_reward": mean(baseline, "reward_total"),
        "oracle_mean_reward": mean(oracle, "reward_total"),
        "oracle_success_rate": mean(oracle, "success"),
        "scenario_cache_volume": SCENARIO_CACHE_VOLUME_NAME,
        "scenario_cache_mode": "require",
        "scenario_cache_coverage": cache_coverage,
        "tracking_metrics": tracking_metrics,
        "tracking_trace_rows": trace_table_rows(episode_records),
        "baseline": baseline,
        "oracle": oracle,
        **reward_tracking_config,
    }
    with trackio_run(
        run_name=run_name,
        run_type="modal_ephemeral_smoke",
        project=trackio_project,
        space_id=trackio_space_id,
        config={
            "episodes": episodes,
            "seed_start": seed_start,
            "mode": "smoke",
            **reward_tracking_config,
        },
        group="smoke",
    ):
        log_reward_config(reward_settings, step=0)
        logged_metrics = log_episode_batch(episode_records, step=0)
        log_trackio_metrics(
            {
                **logged_metrics,
                "smoke/baseline_mean_reward": result["baseline_mean_reward"],
                "smoke/oracle_mean_reward": result["oracle_mean_reward"],
                "smoke/oracle_success_rate": result["oracle_success_rate"],
                "smoke/episodes": episodes,
            },
            step=0,
        )
    return result


@app.function(image=image, timeout=60 * 10)
def run_grpo_config_check() -> str:
    from training.train_grpo import build_grpo_config

    return str(build_grpo_config())


@app.function(
    image=image,
    timeout=60 * 10,
    secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN"])],
)
def verify_trackio_run(
    run_name: str,
    trackio_space_id: str = "Humanlearning/CyberSecurity_OWASP-trackio",
    trackio_project: str = "CyberSecurity_OWASP-smoke",
) -> dict[str, Any]:
    import os
    from training.trackio_utils import (
        REQUIRED_SMOKE_TRACKIO_ITEMS,
        missing_required_trackio_items,
    )

    hf_token = os.environ["HF_TOKEN"]
    cmd = [
        "trackio",
        "get",
        "run",
        "--project",
        trackio_project,
        "--run",
        run_name,
        "--space",
        trackio_space_id,
        "--hf-token",
        hf_token,
        "--json",
    ]
    metrics_cmd = [
        "trackio",
        "list",
        "metrics",
        "--project",
        trackio_project,
        "--run",
        run_name,
        "--space",
        trackio_space_id,
        "--hf-token",
        hf_token,
        "--json",
    ]
    last_result: dict[str, Any] = {}
    for attempt in range(1, 4):
        completed = subprocess.run(cmd, capture_output=True, text=True)
        metrics_completed = subprocess.run(metrics_cmd, capture_output=True, text=True)
        last_result = {
            "attempt": attempt,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "metrics_returncode": metrics_completed.returncode,
            "metrics_stdout": metrics_completed.stdout[-4000:],
            "metrics_stderr": metrics_completed.stderr[-4000:],
        }
        if completed.returncode == 0:
            data = json.loads(completed.stdout)
            if metrics_completed.returncode == 0:
                metrics_data = json.loads(metrics_completed.stdout)
                if isinstance(metrics_data.get("metrics"), list):
                    data["metrics"] = metrics_data["metrics"]
            missing = missing_required_trackio_items(data, REQUIRED_SMOKE_TRACKIO_ITEMS)
            return {
                "ok": not missing,
                "trackio_space_id": trackio_space_id,
                "trackio_project": trackio_project,
                "run_name": run_name,
                "required_items": list(REQUIRED_SMOKE_TRACKIO_ITEMS),
                "missing_required_items": missing,
                "run": data,
            }
        time.sleep(10)
    return {
        "ok": False,
        "trackio_space_id": trackio_space_id,
        "trackio_project": trackio_project,
        "run_name": run_name,
        "last_result": last_result,
    }


@app.function(
    image=image,
    timeout=60 * 10,
    secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN"])],
)
def inspect_trackio_space(
    trackio_space_id: str = "Humanlearning/CyberSecurity_OWASP-trackio",
) -> dict[str, Any]:
    import os

    hf_token = os.environ["HF_TOKEN"]

    def run_trackio(args: list[str]) -> dict[str, Any]:
        completed = subprocess.run(
            ["trackio", *args, "--space", trackio_space_id, "--hf-token", hf_token, "--json"],
            capture_output=True,
            text=True,
        )
        result = {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-4000:],
        }
        if completed.returncode == 0:
            result["json"] = json.loads(completed.stdout)
        return result

    projects_result = run_trackio(["list", "projects"])
    projects = (projects_result.get("json") or {}).get("projects", [])
    runs_by_project = {
        project: run_trackio(["list", "runs", "--project", project])
        for project in projects
    }
    return {
        "trackio_space_id": trackio_space_id,
        "projects": projects_result,
        "runs_by_project": runs_by_project,
    }


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    episodes: int = 4,
    seed_start: int = 0,
    trackio_space_id: str = "",
    trackio_project: str = "CyberSecurity_OWASP-smoke",
    run_name: str = "",
    cache_difficulty_buckets: int = 0,
    cache_train_per_bucket: int = 0,
    cache_validation_per_bucket: int = 0,
    cache_heldout_per_bucket: int = 0,
    cache_force: bool = False,
) -> None:
    if mode == "prepare-cache":
        result = prepare_ephemeral_scenario_cache.remote(
            seed_start=seed_start,
            difficulty_buckets=cache_difficulty_buckets,
            train_per_bucket=cache_train_per_bucket,
            validation_per_bucket=cache_validation_per_bucket,
            heldout_per_bucket=cache_heldout_per_bucket,
            force=cache_force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif mode == "smoke":
        result = run_ephemeral_smoke.remote(
            episodes=episodes,
            seed_start=seed_start,
            trackio_space_id=trackio_space_id,
            trackio_project=trackio_project,
        )
        output_dir = PROJECT_ROOT / "outputs" / "rollouts"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{result['run_name']}.json"
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"saved": str(output_path), **result}, indent=2, sort_keys=True))
    elif mode == "grpo-config":
        print(run_grpo_config_check.remote())
    elif mode == "verify-trackio":
        if not run_name:
            raise ValueError("--run-name is required for verify-trackio mode")
        result = verify_trackio_run.remote(
            run_name=run_name,
            trackio_space_id=trackio_space_id
            or "Humanlearning/CyberSecurity_OWASP-trackio",
            trackio_project=trackio_project,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif mode == "inspect-trackio":
        result = inspect_trackio_space.remote(
            trackio_space_id=trackio_space_id
            or "Humanlearning/CyberSecurity_OWASP-trackio",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        raise ValueError(
            "mode must be 'prepare-cache', 'smoke', 'grpo-config', 'verify-trackio', or 'inspect-trackio'"
        )
