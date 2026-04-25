"""Ephemeral Modal Labs launcher for CyberSecurity_OWASP training smoke runs.

Run from the repo root:

    modal run scripts/modal_ephemeral_train.py --mode smoke --episodes 4

This intentionally stays separate from ``training/train_grpo.py``. It packages
the local repo into a temporary Modal app and returns compact JSON artifacts to
the local process, so the run disappears when ``modal run`` exits.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = "CyberSecurity_OWASP-ephemeral-training"
REMOTE_PROJECT = "/root/CyberSecurity_OWASP"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_dir(
        PROJECT_ROOT,
        remote_path=REMOTE_PROJECT,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "outputs",
            "*.pyc",
        ],
    )
    .run_commands(f"pip install -e {REMOTE_PROJECT}")
    .workdir(REMOTE_PROJECT)
)


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


@app.function(image=image, timeout=60 * 30)
def run_ephemeral_smoke(
    episodes: int = 4,
    seed_start: int = 0,
    trackio_space_id: str = "",
    trackio_project: str = "CyberSecurity_OWASP-smoke",
) -> dict[str, Any]:
    from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
    from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
        CybersecurityOwaspEnvironment,
    )
    from training.rollout import rollout_once
    from training.trackio_utils import log_trackio_metrics, trackio_run

    baseline = []
    oracle = []

    for offset in range(episodes):
        seed = seed_start + offset

        baseline_env = CybersecurityOwaspEnvironment()
        baseline_env.reset(seed=seed, split="validation")
        baseline.append(rollout_once(NoopTrainer(), baseline_env, max_steps=5))

        oracle_env = CybersecurityOwaspEnvironment()
        oracle_env.reset(seed=seed, split="validation")
        hidden = oracle_env.state.hidden_facts
        oracle_env.step(
            CyberSecurityOWASPAction(
                tool_name="submit_finding",
                arguments={
                    "summary": "BOLA/IDOR authorization bug in invoice read route.",
                    "evidence": (
                        f"user {hidden['owner_user_id']} can request invoice "
                        f"{hidden['other_invoice_id']} despite the owner/admin policy"
                    ),
                    "policy_rule": "Only owner or billing_admin in same tenant may read invoices.",
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
        oracle.append(
            {
                "seed": seed,
                "success": oracle_env.state.success,
                "reward_total": final.reward_breakdown.get("total", 0.0),
                "reward_breakdown": final.reward_breakdown,
            }
        )

    def mean(items: list[dict[str, Any]], key: str) -> float:
        return sum(float(item.get(key, 0.0)) for item in items) / max(1, len(items))

    run_name = f"{APP_NAME}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    result = {
        "run_name": run_name,
        "mode": "smoke",
        "episodes": episodes,
        "seed_start": seed_start,
        "baseline_mean_reward": mean(baseline, "reward_total"),
        "oracle_mean_reward": mean(oracle, "reward_total"),
        "oracle_success_rate": mean(oracle, "success"),
        "baseline": baseline,
        "oracle": oracle,
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
        },
        group="smoke",
    ):
        log_trackio_metrics(
            {
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


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    episodes: int = 4,
    seed_start: int = 0,
    trackio_space_id: str = "",
    trackio_project: str = "CyberSecurity_OWASP-smoke",
) -> None:
    if mode == "smoke":
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
    else:
        raise ValueError("mode must be 'smoke' or 'grpo-config'")
