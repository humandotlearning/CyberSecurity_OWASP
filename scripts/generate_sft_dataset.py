"""Generate verifier-gated SFT data for CyberSecurity_OWASP.

The default path asks a larger Hugging Face-hosted teacher model for one JSON
action at a time, executes those actions in the real environment, and keeps
only trajectories that pass the local deterministic verifier.  The
``--dry-run-oracle`` path is intentionally network-free and exists for CI and
smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from CyberSecurity_OWASP.models import CyberSecurityOWASPAction, CyberSecurityOWASPObservation
from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
    CybersecurityOwaspEnvironment,
)
from CyberSecurity_OWASP.validators import detect_cheating


DEFAULT_TEACHER_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
DEFAULT_TARGET_MODEL = "unsloth/gemma-4-E2B-it"
TRAINING_SYSTEM_PROMPT = (
    "You are a defensive AppSec repair agent in the local CyberSecurity_OWASP "
    "OpenEnv environment. Use only the listed local tools. Do not target real "
    "systems. Work step by step: inspect policy and generated code, reproduce "
    "the authorization issue locally, submit a policy-tied diagnosis, patch the "
    "generated app, run visible tests, then submit the fix. Return exactly one "
    "JSON action object and no markdown."
)
BANNED_PROMPT_MARKERS = (
    "hidden_facts",
    "oracle_hidden_focus",
    "reward_engine",
    "validators.py",
    "rewards.py",
    "tests/hidden",
    "hidden tests",
    ".git",
)
RISKY_ARGUMENT_MARKERS = (
    "hidden",
    "oracle",
    "reward_engine",
    "validators.py",
    "rewards.py",
    ".git",
    "..",
)


@dataclass
class DatasetConfig:
    teacher_model: str = DEFAULT_TEACHER_MODEL
    target_model: str = DEFAULT_TARGET_MODEL
    split: str = "train"
    difficulty: int = 0
    seed_start: int = 0
    episodes: int = 100
    validation_episodes: int = 0
    out_dir: Path = Path("outputs/sft")
    max_steps: int = 40
    max_teacher_retries: int = 2
    max_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.95
    dry_run_oracle: bool = False
    workers: int = 0
    min_terminal_reward: float = 12.0
    difficulty_levels: tuple[int, ...] = ()
    difficulty_buckets: int = 0
    push_to_hub: bool = False
    dataset_repo_id: str = "Humanlearning/CyberSecurity_OWASP-sft-dataset"
    hub_private: bool = False
    progress: bool = False


class HuggingFaceTeacher:
    """Small wrapper around Hugging Face chat completion."""

    def __init__(
        self,
        *,
        model: str,
        token: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - dependency smoke checked separately
            raise RuntimeError(
                "huggingface_hub is required for teacher generation. Install project "
                "dependencies or use --dry-run-oracle for local CI."
            ) from exc

        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.client = InferenceClient(token=token)

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat_completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return _chat_response_content(response)


def _chat_response_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content", ""))
    return str(response)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from raw teacher text."""

    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        for part in stripped.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            candidates.append(candidate)

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded

    start = stripped.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        loaded = json.loads(stripped[start : index + 1])
                    except Exception:
                        break
                    if isinstance(loaded, dict):
                        return loaded
        start = stripped.find("{", start + 1)
    return None


def parse_action_text(text: str) -> CyberSecurityOWASPAction:
    data = extract_first_json_object(text)
    if data is None:
        raise ValueError("teacher did not return a JSON object")
    return CyberSecurityOWASPAction(**data)


def action_to_json(action: CyberSecurityOWASPAction) -> str:
    return json.dumps(action.model_dump(), separators=(",", ":"), sort_keys=True)


def _safe_observation_payload(
    observation: CyberSecurityOWASPObservation,
    recent_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "phase": observation.phase,
        "task_brief": observation.task_brief,
        "scenario_prompt": observation.scenario_prompt,
        "available_actions": observation.available_actions,
        "last_tool_result": observation.last_tool_result,
        "last_action_valid": observation.last_action_valid,
        "last_action_error": observation.last_action_error,
        "visible_test_result": observation.visible_test_result,
        "done_reason": observation.done_reason,
        "recent_actions": recent_actions[-8:],
    }


def build_user_prompt(
    observation: CyberSecurityOWASPObservation,
    recent_actions: list[dict[str, Any]],
    retry_error: str | None = None,
) -> str:
    payload = _safe_observation_payload(observation, recent_actions)
    prompt = (
        "Current CyberSecurity_OWASP observation, containing only information "
        "available to the agent:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n\n"
        "Choose the next action. Output exactly one JSON object with keys "
        "`tool_name` and `arguments`. Do not include markdown or commentary."
    )
    if retry_error:
        prompt += f"\nPrevious candidate was rejected safely: {retry_error}"
    _assert_prompt_is_safe(prompt)
    return prompt


def _assert_prompt_is_safe(prompt: str) -> None:
    lowered = prompt.lower()
    leaked = [marker for marker in BANNED_PROMPT_MARKERS if marker.lower() in lowered]
    if leaked:
        raise ValueError(f"prompt contains blocked marker(s): {', '.join(leaked)}")


def build_chat_messages(
    observation: CyberSecurityOWASPObservation,
    recent_actions: list[dict[str, Any]],
    retry_error: str | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(observation, recent_actions, retry_error)},
    ]


def make_chat_row(
    *,
    messages: list[dict[str, str]],
    action: CyberSecurityOWASPAction,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messages": [
            *messages,
            {"role": "assistant", "content": action_to_json(action)},
        ],
        "metadata": metadata,
    }


def preflight_action(
    env: CybersecurityOwaspEnvironment,
    observation: CyberSecurityOWASPObservation,
    action: CyberSecurityOWASPAction,
) -> tuple[bool, str]:
    if action.tool_name not in observation.available_actions:
        return False, f"{action.tool_name} is not allowed during {observation.phase}"
    args = action.arguments or {}
    flags = detect_cheating(env.state, action)
    if flags:
        return False, f"action triggered safety flags: {', '.join(flags)}"
    arg_text = json.dumps(args, sort_keys=True, default=str).lower()
    if any(marker in arg_text for marker in RISKY_ARGUMENT_MARKERS):
        return False, "arguments reference blocked files or paths"
    if action.tool_name == "read_file" and not args.get("path"):
        return False, "read_file requires path"
    if action.tool_name == "search_code" and not args.get("query"):
        return False, "search_code requires query"
    if action.tool_name == "patch_file":
        path = str(args.get("path", ""))
        if not path:
            return False, "patch_file requires path"
        if path.replace("\\", "/").startswith("tests/"):
            return False, "patch_file cannot modify tests"
        if not args.get("content") and not args.get("diff"):
            return False, "patch_file requires content or diff"
    if action.tool_name == "send_local_request":
        path = str(args.get("path", ""))
        if not path.startswith("/"):
            return False, "send_local_request requires a local route path"
    if action.tool_name == "compare_identities":
        path = str(args.get("path", ""))
        if not path.startswith("/"):
            return False, "compare_identities requires a local route path"
        if not args.get("first_user_id") or not args.get("second_user_id"):
            return False, "compare_identities requires two user ids"
    if action.tool_name == "submit_diagnosis":
        required = ("bug_class", "route", "violated_policy_rule", "evidence_trace_ids", "fix_plan")
        missing = [key for key in required if not args.get(key)]
        if missing:
            return False, f"submit_diagnosis missing: {', '.join(missing)}"
    return True, ""


def _trace_id_from_observation(observation: CyberSecurityOWASPObservation) -> str:
    try:
        payload = json.loads(observation.last_tool_result)
    except Exception:
        return "req_001"
    return str(payload.get("trace_id", "req_001"))


def _secure_invoice_source(env: CybersecurityOwaspEnvironment) -> str:
    source = (Path(env.state.hidden_facts["workspace"]) / "app/routes/invoices.py").read_text(
        encoding="utf-8"
    )
    return source.replace(
        "    # BUG: this only checks that the caller is authenticated. It forgets the\n"
        "    # owner/admin and tenant policy checks required by the policy graph.\n"
        "    return {\"status\": 200, \"body\": invoice}\n",
        "    if invoice[\"tenant_id\"] != actor[\"tenant_id\"]:\n"
        "        return {\"status\": 403, \"body\": {\"detail\": \"forbidden\"}}\n"
        "    if invoice[\"owner_user_id\"] != actor[\"user_id\"] and not is_billing_admin(actor):\n"
        "        return {\"status\": 403, \"body\": {\"detail\": \"forbidden\"}}\n"
        "    return {\"status\": 200, \"body\": invoice}\n",
    )


def oracle_actions_for_state(
    env: CybersecurityOwaspEnvironment,
    evidence_trace_id: str | None = None,
) -> list[CyberSecurityOWASPAction]:
    hidden = env.state.hidden_facts
    trace_id = evidence_trace_id or "req_001"
    return [
        CyberSecurityOWASPAction(tool_name="inspect_policy_graph", arguments={}),
        CyberSecurityOWASPAction(tool_name="list_routes", arguments={}),
        CyberSecurityOWASPAction(
            tool_name="read_file",
            arguments={"path": "app/routes/invoices.py"},
        ),
        CyberSecurityOWASPAction(
            tool_name="send_local_request",
            arguments={
                "method": "GET",
                "path": f"/invoices/{hidden['other_invoice_id']}",
                "user_id": hidden["owner_user_id"],
            },
        ),
        CyberSecurityOWASPAction(
            tool_name="submit_diagnosis",
            arguments={
                "bug_class": "idor_ownership_bug",
                "route": "GET /invoices/{invoice_id}",
                "violated_policy_rule": "Only the owner or a billing_admin in the same tenant may read invoices.",
                "evidence_trace_ids": [trace_id],
                "fix_plan": "Add tenant and owner/admin checks before returning invoice data.",
            },
        ),
        CyberSecurityOWASPAction(
            tool_name="patch_file",
            arguments={"path": "app/routes/invoices.py", "content": _secure_invoice_source(env)},
        ),
        CyberSecurityOWASPAction(tool_name="run_visible_tests", arguments={}),
        CyberSecurityOWASPAction(tool_name="submit_fix", arguments={}),
    ]


def _teacher_action(
    *,
    teacher: HuggingFaceTeacher,
    env: CybersecurityOwaspEnvironment,
    observation: CyberSecurityOWASPObservation,
    recent_actions: list[dict[str, Any]],
    config: DatasetConfig,
) -> tuple[CyberSecurityOWASPAction, list[dict[str, str]]]:
    retry_error: str | None = None
    for _ in range(config.max_teacher_retries + 1):
        messages = build_chat_messages(observation, recent_actions, retry_error)
        raw = teacher.complete(messages)
        try:
            action = parse_action_text(raw)
        except Exception as exc:
            retry_error = str(exc)
            continue
        ok, error = preflight_action(env, observation, action)
        if ok:
            return action, messages
        retry_error = error
    raise ValueError(retry_error or "teacher did not produce a usable action")


def _oracle_action(
    *,
    env: CybersecurityOwaspEnvironment,
    observation: CyberSecurityOWASPObservation,
    recent_actions: list[dict[str, Any]],
    oracle_actions: list[CyberSecurityOWASPAction],
    step_index: int,
) -> tuple[CyberSecurityOWASPAction, list[dict[str, str]]]:
    action = oracle_actions[step_index]
    messages = build_chat_messages(observation, recent_actions)
    ok, error = preflight_action(env, observation, action)
    if not ok:
        raise ValueError(error)
    return action, messages


def _terminal_checks_passed(env: CybersecurityOwaspEnvironment) -> bool:
    verifier = env.state.verification_summary or {}
    required = ("visible", "security", "regression", "public_routes", "patch_quality")
    return all(bool((verifier.get(key) or {}).get("passed", False)) for key in required)


def _episode_reward(env: CybersecurityOwaspEnvironment) -> float:
    if env.state.reward_history:
        return float(env.state.reward_history[-1].get("terminal_total", 0.0))
    return 0.0


def run_episode(
    *,
    seed: int,
    split: str,
    difficulty: int,
    config: DatasetConfig,
    teacher: HuggingFaceTeacher | None,
) -> dict[str, Any]:
    env = CybersecurityOwaspEnvironment()
    rows: list[dict[str, Any]] = []
    trajectory_steps: list[dict[str, Any]] = []
    recent_actions: list[dict[str, Any]] = []
    try:
        observation = env.reset(seed=seed, split=split, difficulty=difficulty)
        oracle_actions = oracle_actions_for_state(env) if config.dry_run_oracle else []
        for step_index in range(config.max_steps):
            if observation.done:
                break
            if config.dry_run_oracle:
                if step_index >= len(oracle_actions):
                    raise ValueError("oracle action script ended before terminal state")
                if step_index == 4 and env.state.request_trace:
                    trace_id = _trace_id_from_observation(observation)
                    oracle_actions = oracle_actions_for_state(env, evidence_trace_id=trace_id)
                action, messages = _oracle_action(
                    env=env,
                    observation=observation,
                    recent_actions=recent_actions,
                    oracle_actions=oracle_actions,
                    step_index=step_index,
                )
            else:
                if teacher is None:
                    raise RuntimeError("teacher is required unless --dry-run-oracle is set")
                action, messages = _teacher_action(
                    teacher=teacher,
                    env=env,
                    observation=observation,
                    recent_actions=recent_actions,
                    config=config,
                )

            step_number = step_index + 1
            action_record = action.model_dump()
            row = make_chat_row(
                messages=messages,
                action=action,
                metadata={
                    "target_model": config.target_model,
                    "teacher_model": config.teacher_model,
                    "seed": seed,
                    "split": split,
                    "difficulty": difficulty,
                    "step": step_number,
                    "tool_name": action.tool_name,
                    "task_id": env.state.task_id,
                    "episode_id": env.state.episode_id,
                    "scenario_hash": env.state.scenario_hash,
                },
            )
            next_observation = env.step(action)
            trajectory_steps.append(
                {
                    "step": step_number,
                    "prompt_messages": messages,
                    "action": action_record,
                    "observation": next_observation.model_dump(),
                    "reward_breakdown": dict(next_observation.reward_breakdown or {}),
                }
            )
            if not next_observation.last_action_valid:
                raise ValueError(next_observation.last_action_error or "invalid action")
            if env.state.anti_cheat_flags:
                raise ValueError(f"anti-cheat flags: {env.state.anti_cheat_flags}")
            rows.append(row)
            recent_actions.append(action_record)
            observation = next_observation
            if observation.done:
                break

        if not env.state.done:
            raise ValueError("episode did not reach a terminal state")
        if not env.state.success:
            raise ValueError(env.state.failure_reason or "terminal verifier failed")
        if env.state.step_count > config.max_steps:
            raise ValueError("episode exceeded max steps")
        if env.state.anti_cheat_flags:
            raise ValueError("episode has anti-cheat flags")
        if not _terminal_checks_passed(env):
            raise ValueError("terminal verifier checks did not all pass")

        final_reward = _episode_reward(env)
        final_breakdown = dict(env.state.reward_history[-1]) if env.state.reward_history else {}
        for row in rows:
            row["metadata"].update(
                {
                    "final_success": True,
                    "terminal_total": final_reward,
                    "total_reward": float(env.state.accumulated_reward),
                    "anti_cheat_flags": list(env.state.anti_cheat_flags),
                    "final_reward_breakdown": final_breakdown,
                }
            )
        return {
            "accepted": True,
            "seed": seed,
            "split": split,
            "difficulty": difficulty,
            "rows": rows,
            "trajectory": {
                "episode_id": env.state.episode_id,
                "task_id": env.state.task_id,
                "seed": seed,
                "split": split,
                "difficulty": difficulty,
                "domain": env.state.domain,
                "bug_family": env.state.bug_family,
                "scenario_hash": env.state.scenario_hash,
                "actions": [step["action"] for step in trajectory_steps],
                "steps": trajectory_steps,
                "reward_breakdown_by_step": list(env.state.reward_history),
                "final_reward_breakdown": final_breakdown,
                "total_reward": float(env.state.accumulated_reward),
                "terminal_total": final_reward,
                "success": True,
                "failure_reason": None,
                "anti_cheat_flags": list(env.state.anti_cheat_flags),
                "verification_summary": env.state.verification_summary,
            },
        }
    except Exception as exc:
        return {
            "accepted": False,
            "seed": seed,
            "split": split,
            "difficulty": difficulty,
            "reason": str(exc),
            "rows": [],
            "trajectory": {
                "seed": seed,
                "split": split,
                "difficulty": difficulty,
                "steps": trajectory_steps,
                "actions": [step["action"] for step in trajectory_steps],
                "success": bool(env.state.success),
                "failure_reason": env.state.failure_reason or str(exc),
                "anti_cheat_flags": list(env.state.anti_cheat_flags),
            },
        }
    finally:
        env.close()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_dataset_card(out_dir: Path, manifest: dict[str, Any], dataset_repo_id: str) -> Path:
    card_path = out_dir / "README.md"
    difficulty_levels = manifest.get("difficulty_levels", [])
    reward_verification = manifest.get("reward_verification", {})
    card = f"""---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- cybersecurity
- owasp
- openenv
- tool-use
- sft
pretty_name: CyberSecurity_OWASP SFT Dataset
---

# CyberSecurity_OWASP SFT Dataset

This dataset contains verifier-gated supervised fine-tuning examples for the
`CyberSecurity_OWASP` OpenEnv environment. Each row teaches one step of the
defensive local AppSec workflow: inspect policy/code, reproduce a local
authorization failure, submit a policy-tied diagnosis, patch the generated app,
run visible tests, and submit the fix.

Every kept trajectory is executed against the real local environment and must
pass the deterministic reward verifier before rows are written.

## Intended Use

- Target SFT model: `{manifest.get("target_model", "")}`
- Teacher model: `{manifest.get("teacher_model", "")}`
- Dataset repo: `{dataset_repo_id}`
- Format: chat JSONL with `messages` and verifier metadata
- Dry-run oracle: `{manifest.get("dry_run_oracle", False)}`

## Curriculum Coverage

- Difficulty levels: `{difficulty_levels}`
- Episodes attempted: `{manifest.get("episodes_attempted", 0)}`
- Episodes accepted: `{manifest.get("episodes_accepted", 0)}`
- Acceptance rate: `{manifest.get("acceptance_rate", 0.0):.4f}`
- Rows by split: `{json.dumps(manifest.get("rows_by_split", {}), sort_keys=True)}`
- Rows by difficulty: `{json.dumps(manifest.get("rows_by_difficulty", {}), sort_keys=True)}`

## Reward Verification

- Passed: `{reward_verification.get("passed", False)}`
- Checked rows: `{reward_verification.get("checked_rows", 0)}`
- Minimum terminal reward: `{reward_verification.get("min_terminal_reward", 0.0)}`
- Reward summary: `{json.dumps(reward_verification.get("reward_summary", {}), sort_keys=True)}`

Rows are rejected if the episode fails hidden security/regression/public-route
checks, triggers anti-cheat flags, lacks a positive patch-quality reward, or
falls below the configured terminal reward threshold.

## Schema

Each JSONL row has:

```json
{{
  "messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "{{\\"tool_name\\":\\"...\\",\\"arguments\\":{{...}}}}"}}
  ],
  "metadata": {{
    "target_model": "...",
    "teacher_model": "...",
    "seed": 0,
    "split": "train",
    "difficulty": 0,
    "step": 1,
    "tool_name": "inspect_policy_graph",
    "final_success": true,
    "terminal_total": 12.5,
    "anti_cheat_flags": []
  }}
}}
```
"""
    card_path.write_text(card, encoding="utf-8")
    return card_path


def push_dataset_to_hub(out_dir: Path, *, repo_id: str, private: bool) -> dict[str, Any]:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for --push-to-hub")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required for --push-to-hub") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out_dir),
        path_in_repo=".",
        commit_message="Upload verified CyberSecurity_OWASP SFT dataset",
        delete_patterns=[
            "README.md",
            "manifest.json",
            "train.jsonl",
            "validation.jsonl",
            "hidden_eval.jsonl",
            "trajectories/**",
        ],
    )
    return {
        "repo_id": repo_id,
        "private": bool(private),
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "commit_url": getattr(commit_info, "commit_url", ""),
    }


def push_existing_dataset(
    out_dir: Path,
    *,
    repo_id: str,
    private: bool,
    min_terminal_reward: float,
    required_difficulties: tuple[int, ...],
) -> dict[str, Any]:
    verification = verify_sft_dataset_rewards(
        out_dir,
        min_terminal_reward=min_terminal_reward,
        require_train_rows=True,
        required_difficulties=required_difficulties,
    )
    if not verification["passed"]:
        raise RuntimeError(f"Reward verification failed; refusing Hub push: {verification}")
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "teacher_model": DEFAULT_TEACHER_MODEL,
            "target_model": DEFAULT_TARGET_MODEL,
            "difficulty_levels": [int(level) for level in required_difficulties],
            "rows_by_split": verification.get("rows_by_split", {}),
        }
    manifest["reward_verification"] = verification
    manifest["hub"] = {
        "repo_id": repo_id,
        "private": bool(private),
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }
    write_dataset_card(out_dir, manifest, repo_id)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    hub_result = push_dataset_to_hub(out_dir, repo_id=repo_id, private=private)
    manifest["hub"].update(hub_result)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return {"reward_verification": verification, "hub": manifest["hub"]}


def _write_trajectory(out_dir: Path, trajectory: dict[str, Any]) -> Path:
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{trajectory.get('split', 'train')}_seed{trajectory.get('seed', 0)}_"
        f"{str(trajectory.get('episode_id', 'rejected'))[:12]}.json"
    )
    path = traj_dir / name
    path.write_text(json.dumps(trajectory, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def _reward_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0}
    sorted_values = sorted(values)
    return {
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": float(sorted_values[len(sorted_values) // 2]),
    }


def _parse_int_csv(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    levels = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        levels.append(int(stripped))
    return tuple(dict.fromkeys(levels))


def _difficulty_levels(config: DatasetConfig) -> tuple[int, ...]:
    if config.difficulty_levels:
        return tuple(int(level) for level in config.difficulty_levels)
    return (int(config.difficulty),)


def _configure_difficulty_buckets(config: DatasetConfig, levels: tuple[int, ...]) -> int:
    requested = max(levels) + 1 if levels else int(config.difficulty) + 1
    configured = max(int(config.difficulty_buckets or 0), requested, 1)
    existing = os.getenv("CYBERSECURITY_OWASP_DIFFICULTY_BUCKETS")
    if existing:
        configured = max(configured, int(existing))
    os.environ["CYBERSECURITY_OWASP_DIFFICULTY_BUCKETS"] = str(configured)
    return configured


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(item)
    return rows


def _verify_sft_row_reward(
    row: dict[str, Any],
    *,
    min_terminal_reward: float,
    path: Path,
    line_number: int,
) -> tuple[bool, str, float]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return False, f"{path}:{line_number}: messages must include system/user/assistant", 0.0
    if messages[-1].get("role") != "assistant":
        return False, f"{path}:{line_number}: final message must be assistant", 0.0
    try:
        CyberSecurityOWASPAction(**json.loads(str(messages[-1].get("content", ""))))
    except Exception as exc:
        return False, f"{path}:{line_number}: assistant content is not a valid action: {exc}", 0.0
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return False, f"{path}:{line_number}: missing metadata object", 0.0
    if metadata.get("final_success") is not True:
        return False, f"{path}:{line_number}: final_success is not true", 0.0
    flags = metadata.get("anti_cheat_flags") or []
    if flags:
        return False, f"{path}:{line_number}: anti-cheat flags present: {flags}", 0.0
    reward = float(metadata.get("terminal_total", 0.0) or 0.0)
    if reward < min_terminal_reward:
        return (
            False,
            f"{path}:{line_number}: terminal_total {reward:.3f} below required {min_terminal_reward:.3f}",
            reward,
        )
    breakdown = metadata.get("final_reward_breakdown") or {}
    if not isinstance(breakdown, dict):
        return False, f"{path}:{line_number}: missing final_reward_breakdown", reward
    required_positive = ("security", "regression", "public_routes", "patch_quality", "visible_tests")
    missing = [key for key in required_positive if float(breakdown.get(key, 0.0) or 0.0) <= 0.0]
    if missing:
        return False, f"{path}:{line_number}: non-positive reward components: {', '.join(missing)}", reward
    return True, "", reward


def verify_sft_dataset_rewards(
    out_dir: Path,
    *,
    min_terminal_reward: float = 12.0,
    require_train_rows: bool = True,
    required_difficulties: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Verify generated SFT rows carry successful verifier-backed rewards."""

    checked_rows = 0
    failed_rows: list[str] = []
    rewards: list[float] = []
    rows_by_split: dict[str, int] = {}
    rows_by_difficulty: dict[str, int] = {}
    for split_name in ("train", "validation", "hidden_eval"):
        path = out_dir / f"{split_name}.jsonl"
        rows = _read_jsonl(path)
        if not rows and split_name != "train":
            continue
        rows_by_split[split_name] = len(rows)
        for index, row in enumerate(rows, start=1):
            ok, error, reward = _verify_sft_row_reward(
                row,
                min_terminal_reward=min_terminal_reward,
                path=path,
                line_number=index,
            )
            checked_rows += 1
            if reward:
                rewards.append(reward)
            if not ok:
                failed_rows.append(error)
            metadata = row.get("metadata") if isinstance(row, dict) else {}
            if isinstance(metadata, dict) and "difficulty" in metadata:
                difficulty_key = str(int(metadata.get("difficulty", 0)))
                rows_by_difficulty[difficulty_key] = rows_by_difficulty.get(difficulty_key, 0) + 1
    passed = not failed_rows and (checked_rows > 0 or not require_train_rows)
    if require_train_rows and rows_by_split.get("train", 0) <= 0:
        passed = False
        failed_rows.append(f"{out_dir / 'train.jsonl'}: no train rows found")
    missing_difficulties = [
        int(level)
        for level in required_difficulties
        if rows_by_difficulty.get(str(int(level)), 0) <= 0
    ]
    if missing_difficulties:
        passed = False
        failed_rows.append(f"missing required curriculum difficulty rows: {missing_difficulties}")
    return {
        "passed": passed,
        "checked_rows": checked_rows,
        "failed_rows": failed_rows[:50],
        "failure_count": len(failed_rows),
        "rows_by_split": rows_by_split,
        "rows_by_difficulty": rows_by_difficulty,
        "required_difficulties": [int(level) for level in required_difficulties],
        "missing_difficulties": missing_difficulties,
        "min_terminal_reward": float(min_terminal_reward),
        "reward_summary": _reward_summary(rewards),
    }


def _resolved_worker_count(config: DatasetConfig, job_count: int) -> int:
    if job_count <= 1:
        return 1
    if int(config.workers) > 0:
        return max(1, min(int(config.workers), job_count))
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count, job_count))


def generate_dataset(config: DatasetConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    teacher_local = threading.local()
    teacher_token = None
    if not config.dry_run_oracle:
        teacher_token = os.getenv("HF_TOKEN")
        if not teacher_token:
            raise RuntimeError("HF_TOKEN is required unless --dry-run-oracle is set")

    def teacher_for_thread() -> HuggingFaceTeacher | None:
        if config.dry_run_oracle:
            return None
        teacher = getattr(teacher_local, "teacher", None)
        if teacher is None:
            teacher = HuggingFaceTeacher(
                model=config.teacher_model,
                token=str(teacher_token),
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            teacher_local.teacher = teacher
        return teacher

    difficulty_levels = _difficulty_levels(config)
    difficulty_bucket_count = _configure_difficulty_buckets(config, difficulty_levels)
    validation_seed_start = config.seed_start + int(config.episodes) * len(difficulty_levels)
    split_jobs = [(config.split, config.episodes, config.seed_start)]
    if config.validation_episodes:
        split_jobs.append(("validation", config.validation_episodes, validation_seed_start))
    episode_jobs = [
        {
            "order": job_order,
            "split": split,
            "difficulty": int(difficulty),
            "seed": int(seed_start) + difficulty_index * int(episodes) + offset,
        }
        for job_order, (split, episodes, seed_start) in enumerate(split_jobs)
        for difficulty_index, difficulty in enumerate(difficulty_levels)
        for offset in range(int(episodes))
    ]

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    attempts: list[dict[str, Any]] = []
    rewards: list[float] = []
    accepted = 0
    attempted = len(episode_jobs)
    workers = _resolved_worker_count(config, attempted)

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        seed = int(job["seed"])
        split = str(job["split"])
        difficulty = int(job["difficulty"])
        return {
            "order": int(job["order"]),
            **run_episode(
                seed=seed,
                split=split,
                difficulty=difficulty,
                config=config,
                teacher=teacher_for_thread(),
            ),
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sft-episode") as executor:
        futures = [executor.submit(run_job, job) for job in episode_jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if config.progress:
                print(
                    json.dumps(
                        {
                            "event": "episode_done",
                            "accepted": bool(result.get("accepted")),
                            "split": result.get("split"),
                            "difficulty": result.get("difficulty"),
                            "seed": result.get("seed"),
                            "reason": result.get("reason", ""),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    for result in sorted(
        results,
        key=lambda item: (
            str(item.get("split", "")),
            int(item.get("difficulty", 0)),
            int(item.get("seed", 0)),
        ),
    ):
        seed = int(result["seed"])
        split = str(result["split"])
        difficulty = int(result["difficulty"])
        attempts.append(
            {
                "seed": seed,
                "split": split,
                "difficulty": difficulty,
                "accepted": bool(result["accepted"]),
                "reason": result.get("reason", ""),
                "trajectory_path": str(_write_trajectory(config.out_dir, result["trajectory"])),
            }
        )
        if result["accepted"]:
            accepted += 1
            rows = list(result["rows"])
            rows_by_split.setdefault(split, []).extend(rows)
            rewards.append(float(result["trajectory"].get("terminal_total", 0.0)))

    for split_rows in rows_by_split.values():
        split_rows.sort(
            key=lambda row: (
                int((row.get("metadata") or {}).get("difficulty", 0)),
                int((row.get("metadata") or {}).get("seed", 0)),
                int((row.get("metadata") or {}).get("step", 0)),
            )
        )

    for split_name in ("train", "validation", config.split):
        write_jsonl(config.out_dir / f"{split_name}.jsonl", rows_by_split.get(split_name, []))

    reward_verification = verify_sft_dataset_rewards(
        config.out_dir,
        min_terminal_reward=config.min_terminal_reward,
        require_train_rows=config.split == "train",
        required_difficulties=difficulty_levels if len(difficulty_levels) > 1 else (),
    )

    accepted_by_difficulty: dict[str, int] = {}
    attempted_by_difficulty: dict[str, int] = {}
    reward_by_difficulty: dict[str, list[float]] = {}
    row_count_by_difficulty: dict[str, int] = {}
    for result in results:
        difficulty_key = str(int(result.get("difficulty", 0)))
        attempted_by_difficulty[difficulty_key] = attempted_by_difficulty.get(difficulty_key, 0) + 1
        if result.get("accepted"):
            accepted_by_difficulty[difficulty_key] = accepted_by_difficulty.get(difficulty_key, 0) + 1
            reward_by_difficulty.setdefault(difficulty_key, []).append(
                float((result.get("trajectory") or {}).get("terminal_total", 0.0))
            )
    for split_rows in rows_by_split.values():
        for row in split_rows:
            difficulty_key = str(int((row.get("metadata") or {}).get("difficulty", 0)))
            row_count_by_difficulty[difficulty_key] = row_count_by_difficulty.get(difficulty_key, 0) + 1

    manifest = {
        "teacher_model": config.teacher_model,
        "target_model": config.target_model,
        "split": config.split,
        "difficulty": config.difficulty,
        "difficulty_levels": [int(level) for level in difficulty_levels],
        "difficulty_bucket_count": int(difficulty_bucket_count),
        "episodes_per_difficulty": config.episodes,
        "validation_episodes_per_difficulty": config.validation_episodes,
        "seed_start": config.seed_start,
        "episodes_attempted": attempted,
        "episodes_accepted": accepted,
        "acceptance_rate": accepted / attempted if attempted else 0.0,
        "attempted_by_difficulty": attempted_by_difficulty,
        "accepted_by_difficulty": accepted_by_difficulty,
        "rows_by_difficulty": row_count_by_difficulty,
        "reward_summary_by_difficulty": {
            key: _reward_summary(value) for key, value in sorted(reward_by_difficulty.items())
        },
        "workers": workers,
        "rows_by_split": {key: len(value) for key, value in sorted(rows_by_split.items())},
        "reward_summary": _reward_summary(rewards),
        "reward_verification": reward_verification,
        "git_sha": _git_sha(),
        "verifier_version": "verifier_v1",
        "dry_run_oracle": config.dry_run_oracle,
        "attempts": attempts,
    }
    if config.push_to_hub:
        if not reward_verification["passed"]:
            raise RuntimeError("Reward verification failed; refusing to push dataset to Hub.")
        manifest["hub"] = {
            "repo_id": config.dataset_repo_id,
            "private": bool(config.hub_private),
            "url": f"https://huggingface.co/datasets/{config.dataset_repo_id}",
        }
    write_dataset_card(config.out_dir, manifest, config.dataset_repo_id)
    manifest_path = config.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    if config.push_to_hub:
        hub_result = push_dataset_to_hub(
            config.out_dir,
            repo_id=config.dataset_repo_id,
            private=config.hub_private,
        )
        manifest["hub"].update(hub_result)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--split", default="train", choices=["train", "validation", "hidden_eval"])
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument(
        "--difficulty-levels",
        default="",
        help="Comma-separated curriculum levels to include, for example 0,1,2,3. "
        "When set, --episodes is per difficulty level.",
    )
    parser.add_argument(
        "--difficulty-buckets",
        type=int,
        default=0,
        help=(
            "Number of curriculum difficulty buckets to expose to the environment. "
            "Defaults to max(--difficulty-levels)+1."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sft"))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-teacher-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel episode workers. 0 auto-selects up to 8 workers.",
    )
    parser.add_argument(
        "--min-terminal-reward",
        type=float,
        default=12.0,
        help="Minimum verifier-backed terminal reward required for SFT rows.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing out-dir dataset reward metadata.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the verified dataset folder to a Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one JSON progress event for each completed episode job.",
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Verify and upload an existing out-dir dataset without regenerating rows.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="Humanlearning/CyberSecurity_OWASP-sft-dataset",
        help="Hugging Face dataset repo id used with --push-to-hub.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create/upload the Hugging Face dataset repo as private.",
    )
    parser.add_argument(
        "--dry-run-oracle",
        action="store_true",
        help="Generate deterministic oracle data without calling the HF API.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        teacher_model=args.teacher_model,
        target_model=args.target_model,
        split=args.split,
        difficulty=args.difficulty,
        difficulty_levels=_parse_int_csv(args.difficulty_levels),
        difficulty_buckets=args.difficulty_buckets,
        seed_start=args.seed_start,
        episodes=args.episodes,
        validation_episodes=args.validation_episodes,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        max_teacher_retries=args.max_teacher_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dry_run_oracle=args.dry_run_oracle,
        workers=args.workers,
        min_terminal_reward=args.min_terminal_reward,
        push_to_hub=args.push_to_hub,
        dataset_repo_id=args.dataset_repo_id,
        hub_private=args.hub_private,
        progress=args.progress,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            verification = verify_sft_dataset_rewards(
                args.out_dir,
                min_terminal_reward=args.min_terminal_reward,
                require_train_rows=args.split == "train",
                required_difficulties=_parse_int_csv(args.difficulty_levels),
            )
            print(json.dumps({"reward_verification": verification}, indent=2, sort_keys=True))
            return 0 if verification["passed"] else 2
        if args.push_only:
            result = push_existing_dataset(
                args.out_dir,
                repo_id=args.dataset_repo_id,
                private=args.hub_private,
                min_terminal_reward=args.min_terminal_reward,
                required_difficulties=_parse_int_csv(args.difficulty_levels),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        manifest = generate_dataset(config_from_args(args))
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest.get("reward_verification", {}).get("passed") else 2
    except (RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"error": str(exc), "error_type": exc.__class__.__name__},
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
