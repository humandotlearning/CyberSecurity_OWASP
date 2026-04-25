"""Policy-oracle matrix for deterministic authorization verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from ..models import CyberSecurityOWASPState
    from ..validators import simulate_request
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPState
    from validators import simulate_request


@dataclass(frozen=True)
class AuthzOracle:
    """Builds hidden allowed/denied tuples from policy facts."""

    def matrix(self, state: CyberSecurityOWASPState) -> list[dict[str, Any]]:
        hidden = state.hidden_facts
        return [
            {
                "name": "owner_can_read_owned_invoice",
                "method": "GET",
                "path": f"/invoices/{hidden['owner_invoice_id']}",
                "user_id": hidden["owner_user_id"],
                "expected_status": 200,
                "allowed": True,
            },
            {
                "name": "resource_owner_can_read_own_invoice",
                "method": "GET",
                "path": f"/invoices/{hidden['other_invoice_id']}",
                "user_id": hidden["intruder_user_id"],
                "expected_status": 200,
                "allowed": True,
            },
            {
                "name": "billing_admin_can_read_same_tenant_invoice",
                "method": "GET",
                "path": f"/invoices/{hidden['other_invoice_id']}",
                "user_id": hidden["admin_user_id"],
                "expected_status": 200,
                "allowed": True,
            },
            {
                "name": "same_tenant_non_owner_denied",
                "method": "GET",
                "path": f"/invoices/{hidden['other_invoice_id']}",
                "user_id": hidden["owner_user_id"],
                "expected_status": 403,
                "allowed": False,
            },
            {
                "name": "cross_tenant_admin_denied",
                "method": "GET",
                "path": f"/invoices/{hidden['foreign_invoice_id']}",
                "user_id": hidden["admin_user_id"],
                "expected_status": 403,
                "allowed": False,
            },
            {
                "name": "health_remains_public",
                "method": "GET",
                "path": "/health",
                "user_id": None,
                "expected_status": 200,
                "allowed": True,
            },
        ]

    def evaluate(self, state: CyberSecurityOWASPState) -> dict[str, Any]:
        cases = []
        for case in self.matrix(state):
            response = simulate_request(
                state,
                str(case["method"]),
                str(case["path"]),
                case.get("user_id"),
            )
            actual = int(response["status"])
            cases.append(
                {
                    "name": case["name"],
                    "allowed": bool(case["allowed"]),
                    "expected_status": int(case["expected_status"]),
                    "actual_status": actual,
                    "passed": actual == int(case["expected_status"]),
                }
            )
        return {"passed": all(case["passed"] for case in cases), "cases": cases}
