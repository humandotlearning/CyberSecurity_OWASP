"""Server-side verifier aggregation for terminal scoring."""

from __future__ import annotations

try:
    from ..models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from ..rewards import compute_reward
    from ..validators import (
        patch_quality,
        run_hidden_regression_tests,
        run_hidden_security_tests,
        run_public_route_tests,
        run_visible_tests,
        verify_finding,
    )
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from rewards import compute_reward
    from validators import (
        patch_quality,
        run_hidden_regression_tests,
        run_hidden_security_tests,
        run_public_route_tests,
        run_visible_tests,
        verify_finding,
    )


def evaluate_action(
    state: CyberSecurityOWASPState,
    action: CyberSecurityOWASPAction,
    anti_cheat_flags: list[str] | None = None,
) -> tuple[dict, dict[str, float]]:
    verifier_result: dict = {"anti_cheat_flags": anti_cheat_flags or []}
    if action.tool_name == "submit_finding":
        verifier_result["finding"] = verify_finding(state, action.arguments)
    elif action.tool_name == "run_visible_tests":
        verifier_result["visible"] = run_visible_tests(state)
    elif action.tool_name == "submit_fix":
        verifier_result.update(
            {
                "visible": run_visible_tests(state),
                "security": run_hidden_security_tests(state),
                "regression": run_hidden_regression_tests(state),
                "public_routes": run_public_route_tests(state),
                "patch_quality": patch_quality(state),
            }
        )
    return verifier_result, compute_reward(state, action, verifier_result)
