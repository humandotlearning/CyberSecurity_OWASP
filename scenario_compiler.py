"""Deterministic scenario compiler for CyberSecurity_OWASP."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

try:
    from .fixture_generator import visible_workspace_summary
    from .policy_graph import build_invoice_policy
    from .template_renderer import render_fastapi_basic
except ImportError:  # pragma: no cover
    from fixture_generator import visible_workspace_summary
    from policy_graph import build_invoice_policy
    from template_renderer import render_fastapi_basic


def compile_scenario(seed: int, split: str = "train", difficulty: int = 0) -> dict[str, Any]:
    """Compile one isolated MVP authorization-repair scenario."""

    compiled = build_invoice_policy(seed)
    workspace = Path(tempfile.mkdtemp(prefix=f"cybersecurity_owasp_{split}_{seed}_"))
    editable_files = render_fastapi_basic(workspace, compiled.public_hint, compiled.hidden_facts)
    task_id = f"{split}-invoices-bola-{seed}"
    hidden = dict(compiled.hidden_facts)
    hidden.update(
        {
            "workspace": str(workspace),
            "editable_files": editable_files,
            "initial_file_hashes": {
                path: (workspace / path).read_text(encoding="utf-8")
                for path in editable_files
            },
        }
    )
    return {
        "task_id": task_id,
        "workspace": workspace,
        "domain": "invoices",
        "bug_family": "bola_idor",
        "difficulty": difficulty,
        "task_brief": (
            "Inspect the generated invoices app and policy. Find the broken "
            "authorization behavior, submit a finding with local evidence, patch "
            "the app, preserve intended owner/admin/public behavior, then submit."
        ),
        "public_hint": compiled.public_hint,
        "workspace_summary": visible_workspace_summary(editable_files, compiled.public_hint),
        "hidden_facts": hidden,
    }
