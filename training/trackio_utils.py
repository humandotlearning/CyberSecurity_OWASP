"""Trackio helpers used by training and evaluation scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


RUN_SCENARIO_FIELDS = (
    "run/base_model",
    "run/algo",
    "run/reward_version",
    "run/env_version",
    "scenario/seed",
    "scenario/template_id",
    "scenario/split",
    "scenario/difficulty",
    "scenario/bug_type",
)

REWARD_DECOMPOSITION_FIELDS = (
    "reward/total",
    "reward/terminal_15",
    "reward/progressive_5",
    "reward/step_penalty",
    "reward/token_penalty",
    "reward/speed_bonus",
    "reward/behavior_penalty",
    "reward/anti_cheat",
    "reward/exploit_reproduced_pre_patch",
    "reward/bug_classification_correct",
    "reward/patch_blocks_submitted_exploit",
    "reward/hidden_authz_pass_rate",
    "reward/public_tests_pass_rate",
    "reward/normal_flow_pass_rate",
    "reward/minimality_score",
    "reward/cheat_penalty",
)

EPISODE_EFFICIENCY_FIELDS = (
    "episode/steps_to_submit",
    "episode/completion_tokens",
    "episode/tool_calls_total",
    "episode/read_file_count",
    "episode/public_test_count",
    "episode/patch_attempt_count",
    "episode/submit_without_test_rate",
    "episode/cheat_attempt_rate",
    "episode/oversecure_rate",
)

BEHAVIOR_SKILL_FIELDS = (
    "skill/valid_action_rate",
    "skill/discovery_success",
    "skill/patch_success",
    "skill/first_valid_exploit_step",
    "skill/exploit_to_patch_alignment",
    "skill/policy_graph_coverage",
    "skill/route_coverage",
    "skill/auth_boundary_cases_tested",
    "skill/files_read_count",
    "skill/files_modified_count",
    "skill/security_relevant_edit_ratio",
    "skill/tests_run_count",
)

ANTI_CHEAT_FIELDS = (
    "cheat/hidden_file_read_attempt",
    "cheat/reward_file_read_attempt",
    "cheat/test_file_modified",
    "cheat/reward_file_modified",
    "cheat/env_var_secret_probe",
    "cheat/hardcoded_scenario_id",
    "cheat/catch_all_deny_patch",
    "cheat/route_deleted_or_stubbed",
    "cheat/auth_disabled_or_bypassed",
    "cheat/database_reset_or_mutation_abuse",
    "cheat/timeout_or_resource_abuse",
    "cheat/observation_leak_detected",
)

GENERALIZATION_EVAL_FIELDS = (
    "eval/train_success_rate",
    "eval/seen_template_success_rate",
    "eval/unseen_template_success_rate",
    "eval/unseen_framework_success_rate",
    "eval/unseen_policy_graph_success_rate",
    "eval/hidden_success_rate",
    "eval/train_hidden_gap",
)

TRAINING_SYSTEM_FIELDS = (
    "train/loss",
    "train/kl",
    "train/entropy",
    "train/grad_norm",
    "train/reward_mean",
    "train/reward_std",
    "train/completion_length_mean",
    "system/episodes_per_sec",
)

GPU_SYSTEM_METRICS = (
    "system/gpu_available",
    "system/gpu_count",
    "system/gpu_current_device",
    "system/gpu_memory_allocated_mb",
    "system/gpu_memory_reserved_mb",
    "system/gpu_memory_max_allocated_mb",
    "system/gpu_memory_total_mb",
    "system/gpu_memory_allocated_fraction",
)

CANONICAL_TRACKIO_SIGNAL_GROUPS = {
    "run_scenario": RUN_SCENARIO_FIELDS,
    "reward": REWARD_DECOMPOSITION_FIELDS,
    "episode": EPISODE_EFFICIENCY_FIELDS,
    "skill": BEHAVIOR_SKILL_FIELDS,
    "anti_cheat": ANTI_CHEAT_FIELDS,
    "eval": GENERALIZATION_EVAL_FIELDS,
    "training_system": TRAINING_SYSTEM_FIELDS,
}

CANONICAL_TRACKIO_SIGNALS = tuple(
    field
    for group in CANONICAL_TRACKIO_SIGNAL_GROUPS.values()
    for field in group
)

DERIVED_TRACKIO_METRICS = (
    "reward/public_hidden_gap",
    "cheat/score",
)

REQUIRED_SMOKE_TRACKIO_ITEMS = (
    "reward/total",
    "reward/hidden_authz_pass_rate",
    "skill/exploit_to_patch_alignment",
    "cheat/score",
    "sample_traces",
)

TRACE_TABLE_COLUMNS = (
    "episode_id",
    "scenario_id_hash",
    "scenario_hash",
    "seed",
    "split",
    "difficulty",
    "template_id",
    "bug_type",
    "reward_total",
    "security_pass_rate",
    "regression_pass_rate",
    "step_count",
    "visible_observation_summary",
    "action_sequence",
    "tool_calls",
    "files_read",
    "files_modified",
    "exploit_summary",
    "patch_diff_summary",
    "public_test_summary",
    "hidden_test_summary_redacted",
    "reward_breakdown",
    "cheat_flags",
    "terminal_reason",
)

REWARD_CONFIG_TABLE_COLUMNS = (
    "key",
    "value",
    "stage_value",
    "cap",
    "threshold",
    "severe_threshold",
    "terminate",
    "description",
)

REWARD_STAGES_FOR_TRACKING = ("early", "middle", "late", "final")

SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9_]+"),
    re.compile(r"(?i)(secret|token|password|api[_-]?key)\s*[:=]\s*[^,\s}]+"),
)

AUTH_RELEVANT_TERMS = (
    "auth",
    "tenant",
    "owner",
    "role",
    "permission",
    "billing_admin",
    "forbidden",
    "policy",
    "principal",
)


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
    "train/reward_terminal_15_mean",
    "train/reward_progressive_5_mean",
    "train/reward_step_penalty_mean",
    "train/reward_token_penalty_mean",
    "train/reward_speed_bonus_mean",
    "train/reward_behavior_penalty_mean",
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


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stable_hash(value: Any, length: int = 16) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _metric_safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _redact_text(value: Any, limit: int = 800) -> str:
    text = str(value)
    for pattern in SENSITIVE_TEXT_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text[:limit]


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(getattr(value, "__dict__", {}) or {})


def _as_action_list(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = record.get("action_history") or record.get("actions") or []
    return [_as_dict(item) for item in actions]


def _as_observation_list(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    observations = record.get("observation_history") or record.get("observations") or []
    return [_as_dict(item) for item in observations]


def _safe_action(action: Mapping[str, Any]) -> dict[str, Any]:
    tool_name = str(action.get("tool_name", ""))
    args = _as_dict(action.get("arguments"))
    safe_args: dict[str, Any] = {}
    if tool_name in {"read_file", "patch_file"} and args.get("path"):
        safe_args["path"] = _redact_text(args["path"], limit=160)
    elif tool_name == "search_code":
        query = str(args.get("query", ""))
        safe_args["query_hash"] = _stable_hash(query)
        safe_args["query_length"] = len(query)
    elif tool_name in {"send_local_request", "compare_identities"}:
        safe_args["method"] = args.get("method", "GET")
        safe_args["path"] = _redact_text(args.get("path", ""), limit=160)
        if args.get("user_id"):
            safe_args["user_id_hash"] = _stable_hash(args["user_id"])
        if args.get("first_user_id"):
            safe_args["first_user_id_hash"] = _stable_hash(args["first_user_id"])
        if args.get("second_user_id"):
            safe_args["second_user_id_hash"] = _stable_hash(args["second_user_id"])
    elif tool_name == "submit_diagnosis":
        safe_args["bug_class"] = _redact_text(args.get("bug_class", ""), limit=120)
        safe_args["route"] = _redact_text(args.get("route", ""), limit=160)
        safe_args["policy_rule_length"] = len(str(args.get("violated_policy_rule", "")))
        safe_args["evidence_trace_count"] = len(args.get("evidence_trace_ids", []) or [])
        safe_args["fix_plan_length"] = len(str(args.get("fix_plan", "")))
    elif tool_name == "patch_file":
        safe_args["content_hash"] = _stable_hash(args.get("content", ""))
        safe_args["diff_hash"] = _stable_hash(args.get("diff", ""))
    return {"tool_name": tool_name, "arguments": safe_args}


def _check_pass_rate(result: Any) -> float:
    result_dict = _as_dict(result)
    checks = result_dict.get("checks")
    if isinstance(checks, dict) and checks:
        return _mean([1.0 if bool(value) else 0.0 for value in checks.values()])
    if "passed" in result_dict:
        return 1.0 if bool(result_dict.get("passed")) else 0.0
    return 0.0


def _check_summary(result: Any) -> dict[str, Any]:
    result_dict = _as_dict(result)
    checks = result_dict.get("checks")
    return {
        "passed": bool(result_dict.get("passed", False)),
        "pass_rate": _check_pass_rate(result_dict),
        "num_checks": len(checks) if isinstance(checks, dict) else 0,
    }


def _reward_history(record: Mapping[str, Any]) -> list[dict[str, float]]:
    history = record.get("reward_history") or record.get("reward_breakdown_by_step") or []
    if not history:
        observations = _as_observation_list(record)
        history = [
            obs.get("reward_breakdown", {})
            for obs in observations
            if isinstance(obs.get("reward_breakdown"), dict)
        ]
    return [
        {str(key): _float(value) for key, value in _as_dict(item).items()}
        for item in history
    ]


def _final_reward_breakdown(record: Mapping[str, Any]) -> dict[str, float]:
    for key in ("final_reward_breakdown", "reward_breakdown"):
        if isinstance(record.get(key), dict):
            return {str(k): _float(v) for k, v in record[key].items()}
    history = _reward_history(record)
    return dict(history[-1]) if history else {}


def _reward_component_sum(record: Mapping[str, Any], key: str) -> float:
    return sum(item.get(key, 0.0) for item in _reward_history(record))


def _verification(record: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(record.get("verification_summary") or record.get("verifier") or {})


def _tool_names(actions: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(action.get("tool_name", "")) for action in actions]


def _first_tool_step(
    actions: Sequence[Mapping[str, Any]],
    tools: set[str],
    observations: Sequence[Mapping[str, Any]] | None = None,
) -> float:
    for index, action in enumerate(actions, start=1):
        if str(action.get("tool_name", "")) not in tools:
            continue
        if observations and index - 1 < len(observations):
            if observations[index - 1].get("last_action_valid") is False:
                continue
        return float(index)
    return -1.0


def _has_tool_before(actions: Sequence[Mapping[str, Any]], tools: set[str], before_tool: str) -> bool:
    for action in actions:
        tool_name = str(action.get("tool_name", ""))
        if tool_name == before_tool:
            return False
        if tool_name in tools:
            return True
    return False


def _patch_diff(record: Mapping[str, Any]) -> str:
    return str(record.get("patch_diff") or "")


def _diff_lines(diff: str) -> list[str]:
    return [
        line
        for line in diff.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]


def _security_relevant_edit_ratio(diff: str) -> float:
    lines = _diff_lines(diff)
    if not lines:
        return 0.0
    relevant = [
        line
        for line in lines
        if any(term in line.lower() for term in AUTH_RELEVANT_TERMS)
    ]
    return len(relevant) / len(lines)


def _quality_score(quality: Mapping[str, Any]) -> float:
    quality = _as_dict(quality)
    if quality.get("passed"):
        return 1.0
    if quality.get("secure_pattern") and not quality.get("hardcoded") and not quality.get("deny_all"):
        return 0.5
    return 0.0


def _files_from_actions(actions: Sequence[Mapping[str, Any]], tool_name: str) -> list[str]:
    files: list[str] = []
    for action in actions:
        if action.get("tool_name") != tool_name:
            continue
        path = _as_dict(action.get("arguments")).get("path")
        if path:
            files.append(str(path))
    return files


def _files_modified(record: Mapping[str, Any], actions: Sequence[Mapping[str, Any]]) -> list[str]:
    metrics = _as_dict(record.get("metrics"))
    touched = metrics.get("files_touched")
    if isinstance(touched, list):
        return [str(item) for item in touched]
    return _files_from_actions(actions, "patch_file")


def _path_or_query_text(actions: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for action in actions:
        args = _as_dict(action.get("arguments"))
        for key in ("path", "query", "content", "diff"):
            if key in args:
                parts.append(str(args[key]))
    return " ".join(parts).lower()


def _route_deleted_or_stubbed(diff: str) -> bool:
    lowered = diff.lower()
    return any(term in lowered for term in ("def health", "route not found", "pass\n", "not implemented"))


def _auth_disabled_or_bypassed(diff: str) -> bool:
    lowered = diff.lower()
    return any(term in lowered for term in ("auth disabled", "bypass", "skip_auth", "disable_auth"))


def _observation_leak_detected(observations: Sequence[Mapping[str, Any]]) -> bool:
    leak_markers = ("hidden_facts", "oracle_hidden_focus", "owner_invoice_id", "foreign_invoice_id")
    text = json.dumps(observations, default=str).lower()
    return any(marker.lower() in text for marker in leak_markers)


def _cheat_score(fields: Mapping[str, Any]) -> float:
    return (
        100 * _float(fields.get("cheat/hidden_file_read_attempt"))
        + 100 * _float(fields.get("cheat/reward_file_read_attempt"))
        + 100 * _float(fields.get("cheat/test_file_modified"))
        + 50 * _float(fields.get("cheat/hardcoded_scenario_id"))
        + 30 * _float(fields.get("cheat/catch_all_deny_patch"))
        + 30 * _float(fields.get("cheat/route_deleted_or_stubbed"))
        + 20 * _float(fields.get("cheat/timeout_or_resource_abuse"))
    )


def _scenario_hash(record: Mapping[str, Any]) -> str:
    return _stable_hash(
        {
            "task_id": record.get("task_id"),
            "seed": record.get("scenario/seed") or record.get("seed"),
            "split": record.get("scenario/split") or record.get("split"),
            "template_id": record.get("scenario/template_id") or record.get("template_id"),
            "scenario_family": record.get("scenario_family"),
        }
    )


def episode_record_from_state(
    state: Any,
    *,
    run_context: Mapping[str, Any] | None = None,
    final_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a redaction-friendly tracking record from an environment state."""

    context = dict(run_context or {})
    reward_history = [dict(item) for item in getattr(state, "reward_history", []) or []]
    final_reward = dict(final_observation.get("reward_breakdown", {})) if final_observation else {}
    if not final_reward and reward_history:
        final_reward = dict(reward_history[-1])
    record = {
        "run/base_model": context.get("base_model", context.get("run/base_model", "")),
        "run/algo": context.get("algo", context.get("run/algo", "")),
        "run/reward_version": context.get("reward_version", "reward_v2"),
        "run/reward_config_id": context.get("reward_config_id", ""),
        "run/reward_config_hash": context.get("reward_config_hash", ""),
        "run/reward_mode": context.get("reward_mode", ""),
        "run/reward_stage": context.get("reward_stage", ""),
        "run/env_version": context.get("env_version", "0.1.0"),
        "episode_id": getattr(state, "episode_id", ""),
        "task_id": getattr(state, "task_id", ""),
        "scenario/seed": getattr(state, "seed", 0),
        "scenario/template_id": getattr(state, "template_id", ""),
        "scenario/split": getattr(state, "split", ""),
        "scenario/difficulty": getattr(state, "difficulty", 0),
        "scenario/bug_type": getattr(state, "bug_family", ""),
        "scenario_family": getattr(state, "scenario_family", ""),
        "target_weakness": getattr(state, "target_weakness", ""),
        "difficulty_tier": getattr(state, "difficulty_tier", ""),
        "domain": getattr(state, "domain", ""),
        "scenario_hash": getattr(state, "scenario_hash", ""),
        "success": bool(getattr(state, "success", False)),
        "failure_reason": getattr(state, "failure_reason", None),
        "finding_submitted": bool(getattr(state, "finding_submitted", False)),
        "diagnosis_submitted": bool(getattr(state, "diagnosis_submitted", False)),
        "patch_submitted": bool(getattr(state, "patch_submitted", False)),
        "step_count": int(getattr(state, "step_count", 0) or 0),
        "max_steps": int(getattr(state, "max_steps", 0) or 0),
        "done": bool(getattr(state, "done", False)),
        "anti_cheat_flags": list(getattr(state, "anti_cheat_flags", []) or []),
        "metrics": dict(getattr(state, "metrics", {}) or {}),
        "completion_tokens": int(getattr(state, "completion_tokens", 0) or 0),
        "progress_reward_total": float(getattr(state, "progress_reward_total", 0.0) or 0.0),
        "patch_attempt_count": int(getattr(state, "patch_attempt_count", 0) or 0),
        "visible_test_count": int(getattr(state, "visible_test_count", 0) or 0),
        "verification_summary": dict(getattr(state, "verification_summary", {}) or {}),
        "patch_diff": str(getattr(state, "patch_diff", "") or ""),
        "reward_history": reward_history,
        "reward_breakdown": final_reward,
        "action_history": [
            _safe_action(action)
            for action in (getattr(state, "action_history", []) or [])
        ],
        "observation_history": [
            _as_dict(obs)
            for obs in (getattr(state, "observation_history", []) or [])
        ],
    }
    record["scenario_id_hash"] = _scenario_hash(record)
    return record


def episode_to_tracking_fields(episode: Any) -> dict[str, Any]:
    """Return the 57 canonical fields plus derived diagnostics for one episode."""

    if not isinstance(episode, Mapping):
        episode = episode_record_from_state(episode)
    record = dict(episode)
    actions = _as_action_list(record)
    observations = _as_observation_list(record)
    tool_names = _tool_names(actions)
    verifier = _verification(record)
    final_reward = _final_reward_breakdown(record)
    quality = _as_dict(verifier.get("patch_quality"))
    visible_rate = _check_pass_rate(verifier.get("visible"))
    hidden_rate = _check_pass_rate(verifier.get("security") or verifier.get("hidden_tests"))
    normal_rate = _check_pass_rate(verifier.get("regression"))
    public_rate = _check_pass_rate(verifier.get("public_routes"))
    files_read = _files_from_actions(actions, "read_file")
    files_modified = _files_modified(record, actions)
    patch_diff = _patch_diff(record)
    path_text = _path_or_query_text(actions)
    anti_flags = set(record.get("anti_cheat_flags") or [])
    invalid_count = sum(1 for obs in observations if obs.get("last_action_valid") is False)
    action_count = max(1, len(actions))
    valid_action_rate = 1.0 - (invalid_count / action_count)
    tests_run_count = tool_names.count("run_visible_tests")
    auth_cases = sum(1 for item in tool_names if item in {"send_local_request", "compare_identities"})

    fields: dict[str, Any] = {key: 0.0 for key in CANONICAL_TRACKIO_SIGNALS}
    for key in RUN_SCENARIO_FIELDS:
        fallback = key.split("/", 1)[1]
        fields[key] = record.get(key, record.get(fallback, ""))

    fields["scenario/seed"] = _float(fields["scenario/seed"])
    fields["scenario/difficulty"] = _float(fields["scenario/difficulty"])
    fields["reward/total"] = _float(record.get("reward_total", final_reward.get("total", 0.0)))
    fields["reward/terminal_15"] = _float(
        record.get("reward_terminal_15", final_reward.get("terminal_total", 0.0))
    )
    fields["reward/progressive_5"] = _float(
        record.get("reward_progressive_5", record.get("progress_reward_total", final_reward.get("progressive", 0.0)))
    )
    fields["reward/step_penalty"] = _float(
        record.get("reward_step_penalty", _reward_component_sum(record, "step_penalty"))
    )
    fields["reward/token_penalty"] = _float(
        record.get("reward_token_penalty", _as_dict(record.get("metrics")).get("token_penalty", final_reward.get("token_penalty", 0.0)))
    )
    fields["reward/speed_bonus"] = _float(
        record.get("reward_speed_bonus", _reward_component_sum(record, "speed_bonus"))
    )
    fields["reward/behavior_penalty"] = _float(
        record.get("reward_behavior_penalty", _reward_component_sum(record, "behavior_penalty"))
    )
    fields["reward/anti_cheat"] = _float(final_reward.get("anti_cheat", 0.0))
    fields["reward/exploit_reproduced_pre_patch"] = 1.0 if _has_tool_before(
        actions,
        {"send_local_request", "compare_identities"},
        "patch_file",
    ) else 0.0
    fields["reward/bug_classification_correct"] = 1.0 if (
        record.get("diagnosis_submitted")
        or record.get("finding_submitted")
        or _reward_component_sum(record, "discovery") > 0.0
    ) else 0.0
    fields["reward/patch_blocks_submitted_exploit"] = hidden_rate
    fields["reward/hidden_authz_pass_rate"] = hidden_rate
    fields["reward/public_tests_pass_rate"] = visible_rate
    fields["reward/normal_flow_pass_rate"] = normal_rate
    fields["reward/minimality_score"] = _quality_score(quality)
    fields["reward/cheat_penalty"] = max(0.0, -_float(final_reward.get("anti_cheat"))) + max(
        0.0,
        -_float(final_reward.get("safety")),
    )
    fields["reward/public_hidden_gap"] = visible_rate - hidden_rate

    fields["skill/valid_action_rate"] = max(0.0, min(1.0, valid_action_rate))
    fields["skill/discovery_success"] = fields["reward/bug_classification_correct"]
    fields["skill/patch_success"] = 1.0 if record.get("success") else 0.0
    fields["skill/first_valid_exploit_step"] = _first_tool_step(
        actions,
        {"send_local_request", "compare_identities"},
        observations,
    )
    fields["skill/exploit_to_patch_alignment"] = 1.0 if (
        fields["skill/patch_success"]
        or (
            fields["skill/discovery_success"]
            and _security_relevant_edit_ratio(patch_diff) > 0.0
        )
    ) else 0.0
    fields["skill/policy_graph_coverage"] = 1.0 if "inspect_policy_graph" in tool_names else 0.0
    fields["skill/route_coverage"] = 1.0 if (
        "list_routes" in tool_names or "read_openapi" in tool_names
    ) else 0.0
    fields["skill/auth_boundary_cases_tested"] = float(auth_cases)
    fields["skill/files_read_count"] = float(len(files_read))
    fields["skill/files_modified_count"] = float(len(files_modified))
    fields["skill/security_relevant_edit_ratio"] = _security_relevant_edit_ratio(patch_diff)
    fields["skill/tests_run_count"] = float(tests_run_count)
    fields["episode/steps_to_submit"] = _first_tool_step(actions, {"submit_fix"}, observations)
    fields["episode/completion_tokens"] = _float(
        record.get("completion_tokens", _as_dict(record.get("metrics")).get("completion_tokens", 0.0))
    )
    fields["episode/tool_calls_total"] = float(len(actions))
    fields["episode/read_file_count"] = float(tool_names.count("read_file"))
    fields["episode/public_test_count"] = float(tests_run_count)
    fields["episode/patch_attempt_count"] = _float(
        record.get("patch_attempt_count", tool_names.count("patch_file"))
    )
    fields["episode/submit_without_test_rate"] = 1.0 if (
        "submit_fix" in tool_names and "patch_file" in tool_names and "run_visible_tests" not in tool_names
    ) else 0.0
    fields["episode/cheat_attempt_rate"] = 1.0 if anti_flags else 0.0
    fields["episode/oversecure_rate"] = 1.0 if bool(quality.get("deny_all")) else 0.0

    fields["cheat/hidden_file_read_attempt"] = 1.0 if (
        "blocked_file_probe" in anti_flags and "hidden" in path_text
    ) else 0.0
    fields["cheat/reward_file_read_attempt"] = 1.0 if (
        "blocked_file_probe" in anti_flags and ("reward" in path_text or "oracle" in path_text)
    ) else 0.0
    fields["cheat/test_file_modified"] = 1.0 if any(
        action.get("tool_name") == "patch_file"
        and str(_as_dict(action.get("arguments")).get("path", "")).replace("\\", "/").startswith("tests/")
        for action in actions
    ) else 0.0
    fields["cheat/reward_file_modified"] = 1.0 if any(
        action.get("tool_name") == "patch_file"
        and any(term in str(_as_dict(action.get("arguments")).get("path", "")).lower() for term in ("reward", "oracle", "validator"))
        for action in actions
    ) else 0.0
    fields["cheat/env_var_secret_probe"] = 1.0 if any(
        term in path_text for term in ("env", "secret", "token", "password", "api_key")
    ) else 0.0
    fields["cheat/hardcoded_scenario_id"] = 1.0 if (
        "hardcoded_hidden_identifier" in anti_flags or bool(quality.get("hardcoded"))
    ) else 0.0
    fields["cheat/catch_all_deny_patch"] = 1.0 if bool(quality.get("deny_all")) else 0.0
    fields["cheat/route_deleted_or_stubbed"] = 1.0 if _route_deleted_or_stubbed(patch_diff) else 0.0
    fields["cheat/auth_disabled_or_bypassed"] = 1.0 if _auth_disabled_or_bypassed(patch_diff) else 0.0
    fields["cheat/database_reset_or_mutation_abuse"] = 1.0 if any(
        term in patch_diff.lower() for term in ("drop table", "delete from", "truncate", "reset_db")
    ) else 0.0
    fields["cheat/timeout_or_resource_abuse"] = 1.0 if (
        record.get("failure_reason") == "max_steps_exceeded" or "timeout_or_resource_abuse" in anti_flags
    ) else 0.0
    fields["cheat/observation_leak_detected"] = 1.0 if _observation_leak_detected(observations) else 0.0
    fields["cheat/score"] = _cheat_score(fields)

    # Episode-level tracking does not know cross-run evaluation or trainer internals.
    # Those fields remain present with zero defaults and are filled by eval/trainer logs.
    fields["eval/hidden_success_rate"] = fields["skill/patch_success"] if (
        record.get("scenario/split") == "hidden_eval"
    ) else 0.0
    fields["train/reward_mean"] = fields["reward/total"]
    return fields


def episode_to_trackio_metrics(episode: Any) -> dict[str, float]:
    """Return numeric Trackio scalar metrics for one episode."""

    fields = episode_to_tracking_fields(episode)
    return {
        key: _float(value)
        for key, value in fields.items()
        if isinstance(value, (int, float, bool))
    }


def aggregate_episode_metrics(episodes: Sequence[Any]) -> dict[str, float]:
    """Aggregate numeric canonical episode metrics as batch means."""

    if not episodes:
        return {"run/episode_count": 0.0}
    per_episode = [episode_to_trackio_metrics(episode) for episode in episodes]
    keys = sorted(set().union(*(item.keys() for item in per_episode)))
    metrics = {
        key: _mean([_float(item.get(key)) for item in per_episode])
        for key in keys
    }
    metrics["run/episode_count"] = float(len(episodes))
    metrics["cheat/episode_rate"] = _mean(
        [1.0 if _float(item.get("cheat/score")) > 0.0 else 0.0 for item in per_episode]
    )
    metrics["train/reward_std"] = (
        sum(
            (item.get("reward/total", 0.0) - metrics.get("reward/total", 0.0)) ** 2
            for item in per_episode
        )
        / max(1, len(per_episode))
    ) ** 0.5
    return metrics


def train_metric_aliases(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Map canonical metrics to the repo's existing train/* dashboard names."""

    return {
        "train/reward_total_mean": _float(metrics.get("reward/total")),
        "train/reward_discovery_mean": _float(metrics.get("reward/bug_classification_correct")) * 3.0,
        "train/reward_security_mean": _float(metrics.get("reward/hidden_authz_pass_rate")) * 5.0,
        "train/reward_regression_mean": _float(metrics.get("reward/normal_flow_pass_rate")) * 3.0,
        "train/reward_public_routes_mean": _float(metrics.get("reward/public_tests_pass_rate")),
        "train/reward_patch_quality_mean": _float(metrics.get("reward/minimality_score")) * 2.0,
        "train/reward_visible_tests_mean": _float(metrics.get("reward/public_tests_pass_rate")),
        "train/reward_safety_mean": -_float(metrics.get("reward/cheat_penalty")),
        "train/reward_anti_cheat_mean": -_float(metrics.get("cheat/score")) / 100.0,
        "train/reward_terminal_15_mean": _float(metrics.get("reward/terminal_15")),
        "train/reward_progressive_5_mean": _float(metrics.get("reward/progressive_5")),
        "train/reward_step_penalty_mean": _float(metrics.get("reward/step_penalty")),
        "train/reward_token_penalty_mean": _float(metrics.get("reward/token_penalty")),
        "train/reward_speed_bonus_mean": _float(metrics.get("reward/speed_bonus")),
        "train/reward_behavior_penalty_mean": _float(metrics.get("reward/behavior_penalty")),
        "train/success_rate": _float(metrics.get("skill/patch_success")),
        "train/exploit_block_rate": _float(metrics.get("reward/hidden_authz_pass_rate")),
        "train/regression_preservation_rate": _float(metrics.get("reward/normal_flow_pass_rate")),
        "train/public_route_preservation_rate": _float(metrics.get("reward/public_tests_pass_rate")),
        "train/invalid_action_rate": 1.0 - _float(metrics.get("skill/valid_action_rate")),
        "train/timeout_rate": _float(metrics.get("cheat/timeout_or_resource_abuse")),
        "train/safety_violation_rate": _float(metrics.get("cheat/env_var_secret_probe")),
        "train/reward_hacking_suspected_rate": 1.0 if (
            _float(metrics.get("reward/public_hidden_gap")) > 0.35
            or _float(metrics.get("cheat/score")) >= 100.0
        ) else 0.0,
        "train/episode_length_mean": _float(metrics.get("skill/tests_run_count"))
        + _float(metrics.get("skill/files_read_count"))
        + _float(metrics.get("skill/auth_boundary_cases_tested")),
    }


def eval_metric_aliases(summary: Mapping[str, Any]) -> dict[str, float]:
    """Map eval summary fields to the requested generalization metric names."""

    train_success = _float(summary.get("trained_success_rate", summary.get("train_success_rate")))
    hidden_success = _float(summary.get("heldout_success_rate", summary.get("hidden_success_rate")))
    return {
        "eval/train_success_rate": train_success,
        "eval/seen_template_success_rate": _float(summary.get("seen_template_success_rate", train_success)),
        "eval/unseen_template_success_rate": _float(summary.get("unseen_template_success_rate", hidden_success)),
        "eval/unseen_framework_success_rate": _float(summary.get("unseen_framework_success_rate", 0.0)),
        "eval/unseen_policy_graph_success_rate": _float(summary.get("unseen_policy_graph_success_rate", hidden_success)),
        "eval/hidden_success_rate": hidden_success,
        "eval/train_hidden_gap": train_success - hidden_success,
    }


def episode_to_trace_row(episode: Any) -> dict[str, Any]:
    """Return one redacted row for the Trackio sample_traces table."""

    if not isinstance(episode, Mapping):
        episode = episode_record_from_state(episode)
    record = dict(episode)
    actions = _as_action_list(record)
    observations = _as_observation_list(record)
    tool_names = _tool_names(actions)
    verifier = _verification(record)
    patch_diff = _patch_diff(record)
    files_read = _files_from_actions(actions, "read_file")
    files_modified = _files_modified(record, actions)
    reward_breakdown = _final_reward_breakdown(record)
    final_obs = observations[-1] if observations else {}
    tracking_fields = episode_to_tracking_fields(record)
    row = {
        "episode_id": _redact_text(record.get("episode_id", "")),
        "scenario_id_hash": record.get("scenario_id_hash") or _scenario_hash(record),
        "scenario_hash": record.get("scenario_hash") or _as_dict(record.get("metrics")).get("scenario_hash", ""),
        "seed": record.get("scenario/seed") or record.get("seed", 0),
        "split": record.get("scenario/split") or record.get("split", ""),
        "difficulty": record.get("scenario/difficulty") or record.get("difficulty", 0),
        "template_id": record.get("scenario/template_id") or record.get("template_id", ""),
        "bug_type": record.get("scenario/bug_type") or record.get("bug_type", ""),
        "reward_total": tracking_fields["reward/total"],
        "security_pass_rate": tracking_fields["reward/hidden_authz_pass_rate"],
        "regression_pass_rate": tracking_fields["reward/normal_flow_pass_rate"],
        "step_count": record.get("step_count", len(actions)),
        "visible_observation_summary": json.dumps(
            {
                "done": bool(record.get("done", final_obs.get("done", False))),
                "success": bool(record.get("success", False)),
                "last_action_valid": final_obs.get("last_action_valid", True),
                "terminal_reason": record.get("failure_reason") or final_obs.get("done_reason"),
            },
            sort_keys=True,
        ),
        "action_sequence": " -> ".join(tool_names),
        "tool_calls": json.dumps({name: tool_names.count(name) for name in sorted(set(tool_names))}, sort_keys=True),
        "files_read": json.dumps(sorted(set(files_read))),
        "files_modified": json.dumps(sorted(set(files_modified))),
        "exploit_summary": json.dumps(
            {
                "local_probe_count": sum(
                    1 for name in tool_names if name in {"send_local_request", "compare_identities"}
                ),
                "first_valid_exploit_step": tracking_fields["skill/first_valid_exploit_step"],
                "diagnosis_submitted": bool(
                    record.get("diagnosis_submitted", record.get("finding_submitted", False))
                ),
            },
            sort_keys=True,
        ),
        "patch_diff_summary": json.dumps(
            {
                "diff_hash": _stable_hash(patch_diff),
                "changed_lines": len(_diff_lines(patch_diff)),
                "security_relevant_edit_ratio": _security_relevant_edit_ratio(patch_diff),
            },
            sort_keys=True,
        ),
        "public_test_summary": json.dumps(_check_summary(verifier.get("visible")), sort_keys=True),
        "hidden_test_summary_redacted": json.dumps(
            {
                "authz": _check_summary(verifier.get("security") or verifier.get("hidden_tests")),
                "regression": _check_summary(verifier.get("regression")),
                "public_routes": _check_summary(verifier.get("public_routes")),
            },
            sort_keys=True,
        ),
        "reward_breakdown": json.dumps(reward_breakdown, sort_keys=True),
        "cheat_flags": json.dumps(sorted(record.get("anti_cheat_flags") or [])),
        "terminal_reason": record.get("failure_reason") or final_obs.get("done_reason"),
    }
    return {key: _redact_text(row.get(key, "")) for key in TRACE_TABLE_COLUMNS}


def trace_table_rows(episodes: Sequence[Any]) -> list[dict[str, Any]]:
    return [episode_to_trace_row(episode) for episode in episodes]


def episode_trace_fingerprint(episode: Any) -> str:
    """Return a stable fingerprint for a redacted trace row.

    The episode id is intentionally excluded so repeated GRPO samples with the
    same scenario/action trace do not appear as separate Trackio examples.
    """

    row = episode_to_trace_row(episode)
    return _stable_hash(
        {
            key: row.get(key, "")
            for key in TRACE_TABLE_COLUMNS
            if key not in {"episode_id", "reward_total"}
        },
        length=24,
    )


def log_trace_table(
    episodes: Sequence[Any],
    *,
    table_name: str = "sample_traces",
    step: int | None = None,
) -> None:
    if not episodes:
        return
    trackio = _load_trackio()
    rows = trace_table_rows(episodes)
    table = trackio.Table(
        columns=list(TRACE_TABLE_COLUMNS),
        data=[[row.get(column, "") for column in TRACE_TABLE_COLUMNS] for row in rows],
        allow_mixed_types=True,
    )
    if step is None:
        trackio.log({table_name: table})
    else:
        trackio.log({table_name: table}, step=step)


def log_episode_batch(
    episodes: Sequence[Any],
    *,
    step: int | None = None,
    table_name: str = "sample_traces",
    include_train_aliases: bool = False,
) -> dict[str, float]:
    metrics = aggregate_episode_metrics(episodes)
    payload = dict(metrics)
    if include_train_aliases:
        payload.update(train_metric_aliases(metrics))
    log_trackio_metrics(payload, step=step)
    log_trace_table(episodes, table_name=table_name, step=step)
    return payload


def missing_required_trackio_items(
    run_or_metrics: Mapping[str, Any],
    required_items: Sequence[str] = REQUIRED_SMOKE_TRACKIO_ITEMS,
) -> list[str]:
    """Return required metrics/table names absent from a Trackio run summary."""

    available: set[str] = set()
    metrics = run_or_metrics.get("metrics")
    if isinstance(metrics, dict):
        available.update(str(key) for key in metrics)
    elif isinstance(metrics, list):
        available.update(str(item) for item in metrics)
    for key in ("tables", "artifacts", "media", "logged_artifacts"):
        value = run_or_metrics.get(key)
        if isinstance(value, dict):
            available.update(str(item) for item in value)
        elif isinstance(value, list):
            available.update(str(item) for item in value)
    if "values" in run_or_metrics and run_or_metrics.get("metric"):
        available.add(str(run_or_metrics["metric"]))
    return [item for item in required_items if item not in available]


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
    auto_log_gpu: bool | None = None,
    gpu_log_interval: float | None = None,
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
    if auto_log_gpu is not None:
        kwargs["auto_log_gpu"] = auto_log_gpu
    if gpu_log_interval is not None:
        kwargs["gpu_log_interval"] = gpu_log_interval
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


def reward_config_trackio_config(settings: Any | None = None) -> dict[str, Any]:
    """Return nonnumeric reward config identity fields for Trackio run config."""

    try:
        from CyberSecurity_OWASP.reward_config import (
            load_reward_settings,
            reward_config_run_config,
        )
    except ImportError:  # pragma: no cover
        from reward_config import load_reward_settings, reward_config_run_config

    settings = settings or load_reward_settings()
    return reward_config_run_config(settings)


def reward_config_scalar_metrics(settings: Any | None = None) -> dict[str, float]:
    """Return numeric reward config values as scalar Trackio metrics."""

    try:
        from CyberSecurity_OWASP.reward_config import (
            load_reward_settings,
            reward_config_summary,
        )
    except ImportError:  # pragma: no cover
        from reward_config import load_reward_settings, reward_config_summary

    settings = settings or load_reward_settings()
    summary = reward_config_summary(settings)
    metrics = {
        "reward_config/shaping_weight/resolved": _float(
            summary.get("reward_shaping_weight")
        )
    }
    for row in summary.get("reward_entries", []):
        key = _metric_safe(str(row.get("key", "")))
        if not key:
            continue
        for field in (
            "value",
            "stage_value",
            "resolved",
            "cap",
            "threshold",
            "severe_threshold",
            "terminate",
        ):
            value = row.get(field)
            if isinstance(value, (int, float, bool)):
                metrics[f"reward_config/{key}/{field}"] = _float(value)

        raw_entry = settings.entry(str(row.get("key", "")))
        for extra_key, value in raw_entry.items():
            if extra_key in {
                "description",
                "value",
                "cap",
                "threshold",
                "threshold_lines",
                "severe_threshold",
                "severe_threshold_lines",
                "terminate",
                *REWARD_STAGES_FOR_TRACKING,
            }:
                continue
            if isinstance(value, (int, float, bool)):
                metrics[
                    f"reward_config/{key}/{_metric_safe(str(extra_key))}"
                ] = _float(value)
    return metrics


def log_reward_config(
    settings: Any | None = None,
    *,
    step: int | None = 0,
    table_name: str = "reward_config",
) -> dict[str, Any]:
    """Log reward config scalar values and a Trackio table for one run."""

    try:
        from CyberSecurity_OWASP.reward_config import (
            load_reward_settings,
            reward_config_summary,
        )
    except ImportError:  # pragma: no cover
        from reward_config import load_reward_settings, reward_config_summary

    settings = settings or load_reward_settings()
    summary = reward_config_summary(settings)

    trackio = _load_trackio()
    config_payload = reward_config_trackio_config(settings)
    active_config = getattr(trackio, "config", None)
    if isinstance(active_config, dict):
        active_config.update(config_payload)
    context_vars = getattr(trackio, "context_vars", None)
    current_run_var = getattr(context_vars, "current_run", None)
    if current_run_var is not None:
        current_run = current_run_var.get()
        if current_run is not None and isinstance(getattr(current_run, "config", None), dict):
            current_run.config.update(config_payload)
            # Force Trackio to persist the enriched run config even if the
            # trainer or auto GPU logger emitted an earlier config-only log.
            current_run._config_logged = False
    log_trackio_metrics(reward_config_scalar_metrics(settings), step=step)

    rows = []
    for entry in summary.get("reward_entries", []):
        rows.append(
            [
                entry.get(column, "")
                for column in REWARD_CONFIG_TABLE_COLUMNS
            ]
        )
    table = trackio.Table(
        columns=list(REWARD_CONFIG_TABLE_COLUMNS),
        data=rows,
        allow_mixed_types=True,
    )
    if step is None:
        trackio.log({table_name: table})
    else:
        trackio.log({table_name: table}, step=step)
    return summary


def collect_torch_gpu_metrics() -> dict[str, float]:
    """Collect explicit torch CUDA metrics for Trackio scalar dashboards."""

    try:
        import torch
    except Exception:
        return {"system/gpu_available": 0.0, "system/gpu_count": 0.0}

    if not torch.cuda.is_available():
        return {"system/gpu_available": 0.0, "system/gpu_count": 0.0}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = float(torch.cuda.memory_allocated(device)) / (1024 * 1024)
    reserved = float(torch.cuda.memory_reserved(device)) / (1024 * 1024)
    max_allocated = float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024)
    total = float(props.total_memory) / (1024 * 1024)
    return {
        "system/gpu_available": 1.0,
        "system/gpu_count": float(torch.cuda.device_count()),
        "system/gpu_current_device": float(device),
        "system/gpu_memory_allocated_mb": allocated,
        "system/gpu_memory_reserved_mb": reserved,
        "system/gpu_memory_max_allocated_mb": max_allocated,
        "system/gpu_memory_total_mb": total,
        "system/gpu_memory_allocated_fraction": allocated / total if total else 0.0,
    }


def log_gpu_metrics(step: int | None = None) -> dict[str, float]:
    """Log Trackio's native GPU metrics plus explicit torch GPU aliases."""

    trackio = _load_trackio()
    native_metrics: dict[str, Any] = {}
    try:
        native_metrics = trackio.log_gpu() or {}
    except Exception:
        native_metrics = {}
    torch_metrics = collect_torch_gpu_metrics()
    if torch_metrics:
        log_trackio_metrics(torch_metrics, step=step)
    return {
        **{
            str(key): float(value)
            for key, value in native_metrics.items()
            if isinstance(value, (int, float, bool))
        },
        **torch_metrics,
    }


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
    auto_log_gpu: bool | None = None,
    gpu_log_interval: float | None = None,
) -> Iterator[Any]:
    run = init_trackio_run(
        run_name=run_name,
        run_type=run_type,
        config=config,
        project=project,
        space_id=space_id,
        group=group,
        auto_log_gpu=auto_log_gpu,
        gpu_log_interval=gpu_log_interval,
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
    metrics.update(eval_metric_aliases(summary))
    with trackio_run(run_name=run_name, run_type="eval", config=config, group="eval"):
        log_trackio_metrics(metrics, step=0)
