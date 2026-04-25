"""Fixture helpers for scenario compilers."""

from __future__ import annotations

from typing import Any


def visible_workspace_summary(files: list[str], public_hint: dict[str, Any]) -> dict[str, Any]:
    return {
        "framework": "fastapi_style_python",
        "editable_files": files,
        "routes": [
            {"method": "GET", "path": "/health", "public": True},
            {"method": "GET", "path": "/invoices/{invoice_id}", "public": False},
        ],
        "domain": public_hint.get("domain", "invoices"),
    }
