"""Bug-family metadata for generated authorization defects."""

BUG_FAMILIES = {
    "bola_idor": {
        "name": "BOLA/IDOR",
        "defect": "Invoice lookup returns any invoice to any authenticated user.",
        "repair": "Require same tenant and either owner or billing_admin.",
    },
    "bfla": {"name": "BFLA", "status": "scaffolded"},
    "tenant_leak": {"name": "Tenant leak", "status": "scaffolded"},
    "jwt_claim_trust": {"name": "JWT claim trust", "status": "scaffolded"},
    "public_route_trap": {"name": "Public route trap", "status": "scaffolded"},
}


def describe_bug_family(name: str) -> dict:
    return BUG_FAMILIES.get(name, {"name": name, "status": "unknown"})
