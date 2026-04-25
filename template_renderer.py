"""Render small FastAPI-style generated workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_fastapi_basic(workspace: Path, public_hint: dict[str, Any], hidden: dict[str, Any]) -> list[str]:
    """Render the MVP vulnerable invoices app and visible tests."""

    app_dir = workspace / "app"
    routes_dir = app_dir / "routes"
    tests_dir = workspace / "tests"
    routes_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (routes_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "data.py").write_text(
        "USERS = " + repr(hidden["users"]) + "\n\nINVOICES = " + repr(hidden["invoices"]) + "\n",
        encoding="utf-8",
    )
    (app_dir / "auth.py").write_text(
        """from .data import USERS


def get_actor(user_id: str):
    return USERS.get(user_id)


def is_billing_admin(actor: dict) -> bool:
    return "billing_admin" in actor.get("roles", [])
""",
        encoding="utf-8",
    )
    (routes_dir / "invoices.py").write_text(
        """from app.auth import get_actor, is_billing_admin
from app.data import INVOICES


def get_invoice(invoice_id: str, user_id: str):
    actor = get_actor(user_id)
    if actor is None:
        return {"status": 401, "body": {"detail": "unknown user"}}

    invoice = INVOICES.get(invoice_id)
    if invoice is None:
        return {"status": 404, "body": {"detail": "invoice not found"}}

    # BUG: this only checks that the caller is authenticated. It forgets the
    # owner/admin and tenant policy checks required by the policy graph.
    return {"status": 200, "body": invoice}


def health():
    return {"status": 200, "body": {"status": "ok"}}
""",
        encoding="utf-8",
    )
    (app_dir / "main.py").write_text(
        """from app.routes.invoices import get_invoice, health


ROUTES = [
    {"method": "GET", "path": "/health", "handler": health, "public": True},
    {"method": "GET", "path": "/invoices/{invoice_id}", "handler": get_invoice, "public": False},
]
""",
        encoding="utf-8",
    )
    (tests_dir / "test_visible.py").write_text(
        f"""from app.routes.invoices import get_invoice, health


def test_health_is_public():
    assert health()["status"] == 200


def test_owner_can_read_own_invoice():
    result = get_invoice("{hidden['owner_invoice_id']}", "{hidden['owner_user_id']}")
    assert result["status"] == 200


def test_admin_can_read_same_tenant_invoice():
    result = get_invoice("{hidden['other_invoice_id']}", "{hidden['admin_user_id']}")
    assert result["status"] == 200
""",
        encoding="utf-8",
    )
    return [
        "app/main.py",
        "app/auth.py",
        "app/data.py",
        "app/routes/invoices.py",
        "tests/test_visible.py",
    ]
