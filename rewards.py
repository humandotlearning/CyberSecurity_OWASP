"""Reward computation for CyberSecurity_OWASP."""

from __future__ import annotations

from .models import CyberSecurityOWASPAction, CyberSecurityOWASPState


REWARD_KEYS = (
    "discovery",
    "security",
    "regression",
    "public_routes",
    "patch_quality",
    "visible_tests",
    "safety",
    "anti_cheat",
    "total",
)


def empty_reward() -> dict[str, float]:
    return {key: 0.0 for key in REWARD_KEYS}


def compute_reward(
    state: CyberSecurityOWASPState,
    action: CyberSecurityOWASPAction,
    verifier_result: dict,
) -> dict[str, float]:
    reward = empty_reward()
    if action.tool_name == "submit_finding":
        finding = verifier_result.get("finding", {})
        reward["discovery"] = (2.0 if finding.get("valid") else 0.0) + (
            1.0 if finding.get("tied_to_policy") else 0.0
        )
    elif action.tool_name == "run_visible_tests":
        visible = verifier_result.get("visible", {})
        reward["visible_tests"] = 1.0 if visible.get("passed") else 0.0
    elif action.tool_name == "submit_fix":
        security = verifier_result.get("security", {})
        regression = verifier_result.get("regression", {})
        public_routes = verifier_result.get("public_routes", {})
        quality = verifier_result.get("patch_quality", {})
        visible = verifier_result.get("visible", {})
        reward["security"] = 5.0 if security.get("passed") else -6.0
        reward["regression"] = 3.0 if regression.get("passed") else -5.0
        reward["public_routes"] = 1.0 if public_routes.get("passed") else -3.0
        reward["patch_quality"] = 2.0 if quality.get("passed") else 0.0
        reward["visible_tests"] = 1.0 if visible.get("passed") else -6.0
        if quality.get("hardcoded"):
            reward["anti_cheat"] -= 4.0
        if quality.get("deny_all"):
            reward["safety"] -= 5.0

    flags = verifier_result.get("anti_cheat_flags", []) or []
    if flags:
        if "external_network_attempt" in flags:
            reward["safety"] -= 8.0
        if "blocked_file_probe" in flags:
            reward["anti_cheat"] -= 8.0
        if "hardcoded_hidden_identifier" in flags:
            reward["anti_cheat"] -= 4.0

    total = sum(value for key, value in reward.items() if key != "total")
    reward["total"] = min(15.0, total) if total > 0 else total
    return reward
