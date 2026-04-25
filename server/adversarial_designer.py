"""Bounded adversarial scenario targeting for synthetic local lab episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .curriculum import WEAKNESS_TARGETS
except ImportError:  # pragma: no cover
    from server.curriculum import WEAKNESS_TARGETS


TARGET_SPECS: dict[str, dict[str, Any]] = {
    "same_role_cross_object": {
        "description": "Same-role actor tries to read another user's object.",
        "hidden_focus": ["same_tenant_other_user_blocked"],
    },
    "cross_tenant_boundary": {
        "description": "Tenant-local admin is denied access to another tenant's resource.",
        "hidden_focus": ["cross_tenant_blocked"],
    },
    "public_route_overlock": {
        "description": "Public health route must remain unauthenticated after patching.",
        "hidden_focus": ["health_public"],
    },
    "alternate_route_same_service": {
        "description": "Alternate route/service access should follow the same policy oracle.",
        "hidden_focus": ["oracle_matrix"],
    },
    "visible_test_edge_case": {
        "description": "Visible tests are insufficient; hidden policy matrix decides success.",
        "hidden_focus": ["visible_test_only_guard"],
    },
}


@dataclass(frozen=True)
class BoundedAdversarialDesigner:
    """Chooses safe local lab variants that target tracked agent weaknesses."""

    def design(self, *, seed: int, split: str, curriculum_profile: dict[str, Any]) -> dict[str, Any]:
        target = str(curriculum_profile.get("target_weakness") or "")
        if target not in TARGET_SPECS:
            target = WEAKNESS_TARGETS[int(seed) % len(WEAKNESS_TARGETS)]
        family = f"invoices.bola_idor.{target}"
        if split == "hidden_eval":
            family = f"heldout.{family}"
        spec = TARGET_SPECS[target]
        return {
            "domain": "invoices",
            "bug_family": "bola_idor",
            "template_id": "fastapi_basic",
            "scenario_family": family,
            "target_weakness": target,
            "hidden_focus": list(spec["hidden_focus"]),
            "description": spec["description"],
            "safe_lab_only": True,
        }
