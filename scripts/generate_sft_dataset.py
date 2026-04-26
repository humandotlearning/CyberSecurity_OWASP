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


def generate_dataset(config: DatasetConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    teacher = None
    if not config.dry_run_oracle:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required unless --dry-run-oracle is set")
        teacher = HuggingFaceTeacher(
            model=config.teacher_model,
            token=token,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

    split_jobs = [(config.split, config.episodes, config.seed_start)]
    if config.validation_episodes:
        split_jobs.append(("validation", config.validation_episodes, config.seed_start + config.episodes))

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    attempts: list[dict[str, Any]] = []
    rewards: list[float] = []
    accepted = 0
    attempted = 0
    for split, episodes, seed_start in split_jobs:
        for offset in range(int(episodes)):
            seed = int(seed_start) + offset
            attempted += 1
            result = run_episode(
                seed=seed,
                split=split,
                difficulty=config.difficulty,
                config=config,
                teacher=teacher,
            )
            attempts.append(
                {
                    "seed": seed,
                    "split": split,
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

    for split_name in ("train", "validation", config.split):
        write_jsonl(config.out_dir / f"{split_name}.jsonl", rows_by_split.get(split_name, []))

    manifest = {
        "teacher_model": config.teacher_model,
        "target_model": config.target_model,
        "split": config.split,
        "difficulty": config.difficulty,
        "seed_start": config.seed_start,
        "episodes_attempted": attempted,
        "episodes_accepted": accepted,
        "acceptance_rate": accepted / attempted if attempted else 0.0,
        "rows_by_split": {key: len(value) for key, value in sorted(rows_by_split.items())},
        "reward_summary": _reward_summary(rewards),
        "git_sha": _git_sha(),
        "verifier_version": "verifier_v1",
        "dry_run_oracle": config.dry_run_oracle,
        "attempts": attempts,
    }
    manifest_path = config.out_dir / "manifest.json"
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
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest = generate_dataset(config_from_args(args))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
