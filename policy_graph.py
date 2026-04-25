"""Policy graph generation for MVP authorization-repair scenarios."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompiledPolicy:
    public_hint: dict[str, Any]
    hidden_facts: dict[str, Any]


def build_invoice_policy(seed: int) -> CompiledPolicy:
    """Create a deterministic invoices policy graph with randomized IDs."""

    rng = random.Random(seed)
    suffix = rng.randint(1000, 9999)
    tenant_a = f"tenant-{suffix}-a"
    tenant_b = f"tenant-{suffix}-b"
    owner = f"user-{suffix}-owner"
    intruder = f"user-{suffix}-intruder"
    admin = f"user-{suffix}-billing-admin"
    owner_invoice = f"inv-{suffix}-owned"
    other_invoice = f"inv-{suffix}-other"
    foreign_invoice = f"inv-{suffix}-foreign"

    users = {
        owner: {"user_id": owner, "tenant_id": tenant_a, "roles": ["user"]},
        intruder: {"user_id": intruder, "tenant_id": tenant_a, "roles": ["user"]},
        admin: {"user_id": admin, "tenant_id": tenant_a, "roles": ["billing_admin"]},
        f"user-{suffix}-outside": {
            "user_id": f"user-{suffix}-outside",
            "tenant_id": tenant_b,
            "roles": ["user"],
        },
    }
    invoices = {
        owner_invoice: {
            "invoice_id": owner_invoice,
            "tenant_id": tenant_a,
            "owner_user_id": owner,
            "amount": 1200,
            "status": "open",
        },
        other_invoice: {
            "invoice_id": other_invoice,
            "tenant_id": tenant_a,
            "owner_user_id": intruder,
            "amount": 410,
            "status": "open",
        },
        foreign_invoice: {
            "invoice_id": foreign_invoice,
            "tenant_id": tenant_b,
            "owner_user_id": f"user-{suffix}-outside",
            "amount": 990,
            "status": "open",
        },
    }

    public_hint = {
        "domain": "invoices",
        "policy_rules": [
            "Authenticated users may read invoices they own.",
            "Billing admins may read invoices in their own tenant.",
            "Users must not read another user's invoice unless they have a billing_admin role.",
            "Cross-tenant invoice reads are forbidden.",
            "GET /health is intentionally public.",
        ],
        "users": {
            alias: {
                "user_id": value["user_id"],
                "tenant_id": value["tenant_id"],
                "roles": value["roles"],
            }
            for alias, value in {
                "owner": users[owner],
                "same_tenant_other_user": users[intruder],
                "billing_admin": users[admin],
            }.items()
        },
        "resources": {
            "owned_invoice": owner_invoice,
            "same_tenant_other_invoice": other_invoice,
            "foreign_tenant_invoice": foreign_invoice,
        },
        "public_routes": [{"method": "GET", "path": "/health"}],
    }
    hidden_facts = {
        "users": users,
        "invoices": invoices,
        "owner_user_id": owner,
        "intruder_user_id": intruder,
        "admin_user_id": admin,
        "owner_invoice_id": owner_invoice,
        "other_invoice_id": other_invoice,
        "foreign_invoice_id": foreign_invoice,
        "tenant_a": tenant_a,
        "tenant_b": tenant_b,
        "bug_family": "bola_idor",
    }
    return CompiledPolicy(public_hint=public_hint, hidden_facts=hidden_facts)
