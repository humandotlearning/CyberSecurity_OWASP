"""Reward computation for CyberSecurity_OWASP."""

from __future__ import annotations

try:
    from .models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from .reward_config import RewardSettings, load_reward_settings
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from reward_config import RewardSettings, load_reward_settings


REWARD_KEYS = (
    "discovery",
    "security",
    "regression",
    "public_routes",
    "patch_quality",
    "visible_tests",
    "safety",
    "anti_cheat",
    "terminal_total",
    "progressive",
    "step_penalty",
    "speed_bonus",
    "token_penalty",
    "behavior_penalty",
    "train_total",
    "total",
)


def empty_reward() -> dict[str, float]:
    return {key: 0.0 for key in REWARD_KEYS}


def compute_reward(
    state: CyberSecurityOWASPState,
    action: CyberSecurityOWASPAction,
    verifier_result: dict,
) -> dict[str, float]:
    settings = load_reward_settings()
    reward = empty_reward()
    if action.tool_name == "submit_diagnosis":
        diagnosis = verifier_result.get("diagnosis", verifier_result.get("finding", {}))
        reward["discovery"] = _diagnosis_score(diagnosis)
    elif action.tool_name == "run_visible_tests":
        visible = verifier_result.get("visible", {})
        reward["visible_tests"] = 1.0 if visible.get("passed") else 0.0
    elif action.tool_name == "submit_fix":
        _add_terminal_submit_fix_reward(state, verifier_result, reward, settings)

    _add_current_anti_cheat_penalties(verifier_result, reward, settings)

    if verifier_result.get("invalid_action"):
        reward["behavior_penalty"] += settings.value("invalid_action", -0.2)
    if verifier_result.get("repeated_action"):
        reward["behavior_penalty"] += (
            settings.value("repeated_invalid_action", -0.3)
            if verifier_result.get("invalid_action")
            else settings.value("repeated_low_value_action", -0.1)
        )

    reward["progressive"] = _compute_progressive_reward(
        state,
        action,
        verifier_result,
        settings,
    )
    reward["step_penalty"] = _compute_step_penalty(state, settings)
    reward["speed_bonus"] = _compute_speed_bonus(state, action, verifier_result, settings)
    reward["behavior_penalty"] += _compute_behavior_penalty(
        state,
        action,
        verifier_result,
        settings,
        reward["progressive"],
    )

    terminal_total = (
        _component_total(reward)
        if action.tool_name == "submit_fix"
        else reward["safety"] + reward["anti_cheat"]
    )
    reward["terminal_total"] = _cap_terminal(terminal_total, settings)
    reward["train_total"] = _cap_train(
        reward["terminal_total"]
        + settings.shaping_weight * reward["progressive"]
        + reward["step_penalty"]
        + reward["speed_bonus"]
        + reward["token_penalty"]
        + reward["behavior_penalty"],
        settings,
        state,
    )
    reward["total"] = reward["train_total"] if settings.dense_train else reward["terminal_total"]
    return reward


def _followed_repair_phase_order(state: CyberSecurityOWASPState) -> bool:
    tools = [item.get("tool_name") for item in state.action_history]
    required = ["submit_diagnosis", "patch_file", "run_visible_tests", "submit_fix"]
    cursor = 0
    for tool in tools:
        if cursor < len(required) and tool == required[cursor]:
            cursor += 1
    return cursor == len(required)


def should_terminate_for_flags(flags: list[str], settings: RewardSettings | None = None) -> bool:
    settings = settings or load_reward_settings()
    mapping = {
        "external_network_attempt": "external_url_attempt",
        "blocked_file_probe": "hidden_file_probe",
        "host_path_escape": "host_path_escape",
        "test_file_edit_attempt": "test_file_edit",
    }
    return any(settings.terminate(mapping[flag]) for flag in flags if flag in mapping)


def _diagnosis_score(diagnosis: dict) -> float:
    return sum(
        0.25
        for key in (
            "correct_route",
            "correct_bug_class",
            "correct_policy_rule",
            "valid_local_evidence",
        )
        if diagnosis.get(key)
    )


def _add_terminal_submit_fix_reward(
    state: CyberSecurityOWASPState,
    verifier_result: dict,
    reward: dict[str, float],
    settings: RewardSettings,
) -> None:
    security = verifier_result.get("security", {})
    oracle = verifier_result.get("oracle_matrix", {})
    regression = verifier_result.get("regression", {})
    public_routes = verifier_result.get("public_routes", {})
    quality = verifier_result.get("patch_quality", {})
    visible = verifier_result.get("visible", {})
    security_passed = bool(security.get("passed") and oracle.get("passed", True))
    reward["security"] = 5.0 if security_passed else -6.0
    reward["regression"] = 3.0 if regression.get("passed") else -5.0
    reward["public_routes"] = 1.0 if public_routes.get("passed") else -3.0
    reward["patch_quality"] = 2.0 if quality.get("passed") else 0.0
    if quality.get("passed") and _followed_repair_phase_order(state):
        reward["patch_quality"] += 0.5
    reward["visible_tests"] = 1.0 if visible.get("passed") else -6.0
    if quality.get("hardcoded"):
        reward["anti_cheat"] += settings.value("hardcoded_identifier", -2.0)
    if quality.get("deny_all"):
        reward["safety"] += settings.value("deny_all_patch", -3.0)


def _add_current_anti_cheat_penalties(
    verifier_result: dict,
    reward: dict[str, float],
    settings: RewardSettings,
) -> None:
    flags = set(verifier_result.get("anti_cheat_flags", []) or [])
    if "external_network_attempt" in flags:
        reward["safety"] += settings.value("external_url_attempt", -5.0)
    if "blocked_file_probe" in flags:
        reward["anti_cheat"] += settings.value("hidden_file_probe", -5.0)
    if "host_path_escape" in flags:
        reward["anti_cheat"] += settings.value("host_path_escape", -4.0)
    if "test_file_edit_attempt" in flags:
        reward["anti_cheat"] += settings.value("test_file_edit", -5.0)
    if "hardcoded_hidden_identifier" in flags:
        reward["anti_cheat"] += settings.value("hardcoded_identifier", -2.0)


def _compute_progressive_reward(
    state: CyberSecurityOWASPState,
    action: CyberSecurityOWASPAction,
    verifier_result: dict,
    settings: RewardSettings,
) -> float:
    if not settings.dense_train:
        return 0.0
    delta = 0.0
    if action.tool_name == "inspect_policy_graph":
        delta += _award_progress_once(state, "policy_seen", "policy_inspected", settings)
    if action.tool_name in {"list_routes", "read_openapi"}:
        delta += _award_progress_once(state, "route_map_seen", "route_map_inspected", settings)
    if action.tool_name in {"read_file", "search_code"} and _is_relevant_code_action(action):
        delta += _award_progress_once(
            state,
            "relevant_file_seen",
            "relevant_file_inspected",
            settings,
        )
    if action.tool_name in {"send_local_request", "compare_identities"} and any(
        trace.get("unauthorized_success") for trace in state.request_trace
    ):
        delta += _award_progress_once(
            state,
            "local_evidence_found",
            "local_evidence_found",
            settings,
        )
    if action.tool_name == "submit_diagnosis":
        diagnosis = verifier_result.get("diagnosis", verifier_result.get("finding", {}))
        if all(
            diagnosis.get(key)
            for key in (
                "correct_route",
                "correct_bug_class",
                "correct_policy_rule",
                "valid_local_evidence",
            )
        ):
            delta += _award_progress_once(
                state,
                "diagnosis_correct",
                "diagnosis_correct",
                settings,
            )
    if action.tool_name == "patch_file" and not verifier_result.get("invalid_action"):
        delta += _award_progress_once(state, "patch_applies", "patch_applies", settings)
    if action.tool_name == "run_visible_tests":
        visible = verifier_result.get("visible", {})
        checks = visible.get("checks", {}) if isinstance(visible, dict) else {}
        if visible.get("passed"):
            delta += _award_progress_once(
                state,
                "app_boots",
                "app_boots_after_patch",
                settings,
            )
            delta += _award_progress_once(
                state,
                "visible_tests_improved",
                "visible_tests_improved",
                settings,
            )
        if checks.get("health_public"):
            delta += _award_progress_once(
                state,
                "public_routes_visible_pass",
                "public_routes_visible_pass",
                settings,
            )
    return delta


def _award_progress_once(
    state: CyberSecurityOWASPState,
    flag_name: str,
    config_name: str,
    settings: RewardSettings,
) -> float:
    if state.progress_flags.get(flag_name):
        return 0.0
    cap = settings.value("progressive_cap", 5.0)
    remaining = max(0.0, cap - float(state.progress_reward_total or 0.0))
    if remaining <= 0.0:
        return 0.0
    state.progress_flags[flag_name] = True
    value = min(settings.value(config_name, 0.0), remaining)
    state.progress_reward_total += value
    return value


def _is_relevant_code_action(action: CyberSecurityOWASPAction) -> bool:
    args = action.arguments or {}
    text = f"{args.get('path', '')} {args.get('query', '')}".lower()
    return any(
        term in text
        for term in ("auth", "tenant", "owner", "role", "invoice", "route", "guard", "policy")
    )


def _compute_step_penalty(
    state: CyberSecurityOWASPState,
    settings: RewardSettings,
) -> float:
    if not settings.dense_train:
        return 0.0
    rate = settings.value("step_penalty", 0.0)
    if rate >= 0.0:
        return 0.0
    current = float(state.metrics.get("step_penalty_total", 0.0))
    cap = settings.cap("step_penalty", -0.6)
    delta = max(rate, float(cap) - current) if cap is not None else rate
    state.metrics["step_penalty_total"] = current + delta
    return delta


def _compute_speed_bonus(
    state: CyberSecurityOWASPState,
    action: CyberSecurityOWASPAction,
    verifier_result: dict,
    settings: RewardSettings,
) -> float:
    if not settings.dense_train or action.tool_name != "submit_fix":
        return 0.0
    success = all(
        bool((verifier_result.get(key) or {}).get("passed", False))
        for key in ("security", "oracle_matrix", "regression", "public_routes", "patch_quality")
    )
    if not success:
        return 0.0
    max_steps = max(1, int(state.max_steps or 1))
    bonus = settings.value("speed_bonus", 1.0) * (1.0 - min(state.step_count, max_steps) / max_steps)
    return max(0.0, bonus)


def _compute_behavior_penalty(
    state: CyberSecurityOWASPState,
    action: CyberSecurityOWASPAction,
    verifier_result: dict,
    settings: RewardSettings,
    progressive_delta: float,
) -> float:
    if not settings.dense_train:
        return 0.0
    penalty = 0.0
    tools = [item.get("tool_name") for item in state.action_history]
    if action.tool_name == "noop":
        penalty += settings.value("noop_action", -0.02)
    if action.tool_name == "read_file":
        path = str((action.arguments or {}).get("path", ""))
        reads = [
            item
            for item in state.action_history
            if item.get("tool_name") == "read_file"
            and str((item.get("arguments") or {}).get("path", "")) == path
        ]
        if len(reads) > 1:
            penalty += settings.value("repeated_file_read", -0.05)
    if action.tool_name == "send_local_request":
        args = action.arguments or {}
        current = (
            str(args.get("method", "GET")).upper(),
            str(args.get("path", "")),
            str(args.get("user_id", "")),
        )
        matches = [
            item
            for item in state.action_history
            if item.get("tool_name") == "send_local_request"
            and (
                str((item.get("arguments") or {}).get("method", "GET")).upper(),
                str((item.get("arguments") or {}).get("path", "")),
                str((item.get("arguments") or {}).get("user_id", "")),
            )
            == current
        ]
        if len(matches) > 1:
            penalty += settings.value("repeated_local_request", -0.05)
    if action.tool_name == "run_visible_tests" and state.visible_test_count > 1:
        penalty += settings.value("repeated_visible_tests", -0.1)
    if action.tool_name == "patch_file" and not state.progress_flags.get("policy_seen"):
        penalty += settings.value("patch_before_policy", -0.3)
    if action.tool_name == "submit_fix":
        if "patch_file" not in tools:
            penalty += settings.value("submit_without_patch", -0.5)
        if state.patch_attempt_count > 0 and state.visible_test_count == 0:
            penalty += settings.value("submit_without_visible_tests", -0.3)
    if action.tool_name == "patch_file" and state.patch_attempt_count > 3:
        penalty += settings.value("excessive_patch_attempt", -0.2)
    files_touched = state.metrics.get("files_touched", [])
    if isinstance(files_touched, list) and len(files_touched) > 5:
        penalty += settings.value("too_many_files_changed", -0.5)
    if action.tool_name == "patch_file":
        penalty += _oversized_patch_penalty(state, settings)
    if (
        progressive_delta <= 0.0
        and not verifier_result.get("invalid_action")
        and action.tool_name
        in {
            "inspect_policy_graph",
            "list_routes",
            "read_openapi",
            "noop",
            "run_visible_tests",
            "send_local_request",
            "compare_identities",
        }
    ):
        penalty += settings.value("no_progress_action", -0.05)
    return penalty


def _oversized_patch_penalty(
    state: CyberSecurityOWASPState,
    settings: RewardSettings,
) -> float:
    diff_lines = [
        line
        for line in str(state.patch_diff or "").splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]
    entry = settings.entry("oversized_patch")
    threshold = int(entry.get("threshold_lines", 80))
    severe_threshold = int(entry.get("severe_threshold_lines", 180))
    if len(diff_lines) >= severe_threshold:
        return float(entry.get("severe_value", -1.0))
    if len(diff_lines) >= threshold:
        return settings.value("oversized_patch", -0.25)
    return 0.0


def _component_total(reward: dict[str, float]) -> float:
    excluded = {
        "total",
        "terminal_total",
        "progressive",
        "step_penalty",
        "speed_bonus",
        "token_penalty",
        "behavior_penalty",
        "train_total",
    }
    return sum(value for key, value in reward.items() if key not in excluded)


def _cap_terminal(total: float, settings: RewardSettings) -> float:
    cap = settings.value("terminal_cap", 15.0)
    return min(cap, total) if total > 0 else total


def _cap_train(
    total: float,
    settings: RewardSettings,
    state: CyberSecurityOWASPState,
) -> float:
    floor = settings.value("penalty_floor", -6.0)
    capped = max(floor, total)
    cap = settings.value("train_cap", 21.0)
    if capped > 0.0:
        remaining = max(0.0, cap - float(state.accumulated_reward or 0.0))
        return min(capped, remaining)
    return capped
