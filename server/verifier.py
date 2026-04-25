"""Multi-layer deterministic verifier for CyberSecurity_OWASP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    from ..models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from ..validators import (
        patch_quality,
        run_hidden_regression_tests,
        run_hidden_security_tests,
        run_public_route_tests,
        run_visible_tests,
        verify_finding,
    )
    from .authz_oracle import AuthzOracle
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from validators import (
        patch_quality,
        run_hidden_regression_tests,
        run_hidden_security_tests,
        run_public_route_tests,
        run_visible_tests,
        verify_finding,
    )
    from server.authz_oracle import AuthzOracle


@dataclass
class MultiLayerVerifier:
    """Aggregates visible, hidden, oracle, regression, and patch-quality checks."""

    oracle: AuthzOracle = AuthzOracle()

    def evaluate_action(
        self,
        state: CyberSecurityOWASPState,
        action: CyberSecurityOWASPAction,
        anti_cheat_flags: list[str] | None = None,
        *,
        invalid_action: bool = False,
    ) -> dict[str, Any]:
        verifier_result: dict[str, Any] = {
            "anti_cheat_flags": anti_cheat_flags or [],
            "invalid_action": invalid_action,
            "repeated_action": self._is_repeated_action(state, action),
        }
        if action.tool_name == "submit_diagnosis":
            verifier_result["diagnosis"] = verify_finding(state, action.arguments)
            verifier_result["finding"] = verifier_result["diagnosis"]
        elif action.tool_name == "run_visible_tests":
            verifier_result["visible"] = run_visible_tests(state)
        elif action.tool_name == "submit_fix":
            verifier_result.update(self.run_terminal_checks(state))
        return verifier_result

    def run_terminal_checks(self, state: CyberSecurityOWASPState) -> dict[str, Any]:
        security = run_hidden_security_tests(state)
        return {
            "visible": run_visible_tests(state),
            "hidden_tests": security,
            "security": security,
            "oracle_matrix": self.oracle.evaluate(state),
            "regression": run_hidden_regression_tests(state),
            "public_routes": run_public_route_tests(state),
            "patch_quality": patch_quality(state),
        }

    def public_summary(self, verifier_result: dict[str, Any]) -> dict[str, Any]:
        """Return verifier fields that are safe for state/debug summaries."""

        return json.loads(json.dumps(verifier_result))

    def _is_repeated_action(
        self, state: CyberSecurityOWASPState, action: CyberSecurityOWASPAction
    ) -> bool:
        current = {"tool_name": action.tool_name, "arguments": action.arguments}
        return sum(1 for item in state.action_history if item == current) > 1
