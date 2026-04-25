"""Closed-loop scenario factory for CyberSecurity_OWASP."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from ..fixture_generator import visible_workspace_summary
    from ..policy_graph import build_invoice_policy
    from ..template_renderer import render_fastapi_basic
    from .adversarial_designer import BoundedAdversarialDesigner
except ImportError:  # pragma: no cover
    from fixture_generator import visible_workspace_summary
    from policy_graph import build_invoice_policy
    from template_renderer import render_fastapi_basic
    from server.adversarial_designer import BoundedAdversarialDesigner


def _make_workspace(prefix: str) -> Path:
    root = Path(os.getenv("CYBERSECURITY_OWASP_WORKSPACE_ROOT", tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        workspace = root / f"{prefix}{uuid4().hex[:12]}"
        try:
            workspace.mkdir()
        except FileExistsError:
            continue
        return workspace
    raise RuntimeError("Unable to create isolated scenario workspace")


def _visible_policy_hint(public_hint: dict[str, Any]) -> dict[str, Any]:
    """Return partial policy observability without hidden oracle/test labels."""

    return {
        "domain": public_hint.get("domain", "invoices"),
        "policy_rules": list(public_hint.get("policy_rules", [])),
        "fixture_aliases": {
            "users": dict(public_hint.get("users", {})),
            "resources": dict(public_hint.get("resources", {})),
        },
        "public_routes": list(public_hint.get("public_routes", [])),
        "observation_contract": {
            "visible": [
                "product policy summary",
                "fixture aliases needed for local requests",
                "route summaries",
                "visible test results",
            ],
            "hidden": [
                "evaluator-only policy tuples",
                "withheld invariant checks",
                "withheld scenario labels",
                "held-out family label",
            ],
        },
    }


class ScenarioFactory:
    """Compiles deterministic local app scenarios from curriculum profiles."""

    def __init__(self, designer: BoundedAdversarialDesigner | None = None):
        self.designer = designer or BoundedAdversarialDesigner()

    def compile_scenario(
        self,
        seed: int,
        *,
        split: str = "train",
        difficulty: int = 0,
        curriculum_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = curriculum_profile or {
            "difficulty": difficulty,
            "difficulty_tier": "warmup",
            "target_weakness": "same_role_cross_object",
        }
        adversarial_spec = self.designer.design(
            seed=seed, split=split, curriculum_profile=profile
        )
        compiled = build_invoice_policy(seed)
        workspace = _make_workspace(prefix=f"cybersecurity_owasp_{split}_{seed}_")
        public_hint = _visible_policy_hint(compiled.public_hint)
        editable_files = render_fastapi_basic(workspace, public_hint, compiled.hidden_facts)
        workspace_summary = visible_workspace_summary(editable_files, public_hint)
        workspace_summary.update(
            {
                "template_id": adversarial_spec["template_id"],
                "target_weakness": adversarial_spec["target_weakness"],
            }
        )

        hidden = dict(compiled.hidden_facts)
        hidden.update(
            {
                "workspace": str(workspace),
                "editable_files": editable_files,
                "initial_file_hashes": {
                    path: (workspace / path).read_text(encoding="utf-8")
                    for path in editable_files
                },
                "adversarial_spec": adversarial_spec,
                "scenario_family": adversarial_spec["scenario_family"],
                "template_id": adversarial_spec["template_id"],
                "target_weakness": adversarial_spec["target_weakness"],
                "oracle_hidden_focus": adversarial_spec["hidden_focus"],
            }
        )

        return {
            "task_id": f"{split}-invoices-bola-{seed}",
            "workspace": workspace,
            "domain": adversarial_spec["domain"],
            "bug_family": adversarial_spec["bug_family"],
            "scenario_family": adversarial_spec["scenario_family"],
            "template_id": adversarial_spec["template_id"],
            "target_weakness": adversarial_spec["target_weakness"],
            "difficulty": int(profile.get("difficulty", difficulty)),
            "difficulty_tier": str(profile.get("difficulty_tier", "warmup")),
            "curriculum_snapshot": profile,
            "task_brief": (
                "Inspect the generated invoices app and policy. Find the broken "
                "authorization behavior, submit a diagnosis with local evidence, patch "
                "the app, preserve intended owner/admin/public behavior, then submit."
            ),
            "public_hint": public_hint,
            "workspace_summary": workspace_summary,
            "hidden_facts": hidden,
        }
