"""Deterministic scenario registry for SecOps Evidence Gym."""

from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any


DEFAULT_TASK_ID = "secret_exposure_easy"

TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "name": "list_assets",
        "description": "List synthetic services, routes, and artifact collections.",
        "args": {},
    },
    {
        "name": "get_log_events",
        "description": "Return sanitized telemetry evidence ids for a service/query.",
        "args": {"service_id": "str", "query": "str"},
    },
    {
        "name": "check_security_headers",
        "description": "Inspect a service header snapshot and return pass/fail evidence.",
        "args": {"service_id": "str"},
    },
    {
        "name": "search_repo",
        "description": "Search synthetic repo/config snippets for evidence ids.",
        "args": {"query": "str"},
    },
    {
        "name": "scan_dependencies",
        "description": "Inspect a synthetic dependency manifest excerpt.",
        "args": {},
    },
    {
        "name": "create_finding",
        "description": "Store a candidate finding for verifier review.",
        "args": {
            "finding_type": "str",
            "evidence_ids": "list[str]",
            "severity_guess": "str",
            "remediation": "str",
        },
    },
    {
        "name": "validate_finding",
        "description": "Run the deterministic verifier for a candidate finding.",
        "args": {"finding_id": "str"},
    },
    {
        "name": "submit_report",
        "description": "Submit the final structured report and end the episode.",
        "args": {"report_json": "dict"},
    },
]


SCENARIOS: dict[str, dict[str, Any]] = {
    "secret_exposure_easy": {
        "task_id": "secret_exposure_easy",
        "difficulty": "easy",
        "title": "Secret exposure in repo snapshot",
        "alert": (
            "Repository hygiene alert: a synthetic config snapshot may contain "
            "credential-like material. Investigate, cite evidence, validate, and "
            "submit a remediation report."
        ),
        "ground_truth_id": "GT-SECRET-001",
        "finding_type": "secret_exposure",
        "severity": "high",
        "required_evidence": ["EVID-101"],
        "remediation_keywords": ["rotate", "remove"],
        "impact_keywords": ["secret", "credential", "api key", "leak"],
        "assets": {
            "services": ["gateway", "profile-service", "admin-service"],
            "routes": ["/login", "/profile", "/admin/export"],
            "collections": ["repo_snapshot", "telemetry", "headers", "dependencies"],
        },
        "repo": [
            {
                "evidence_id": "EVID-101",
                "source": "repo_snapshot/config/profile-service.env",
                "service_id": "profile-service",
                "tags": ["api key", "secret", "credential", "config"],
                "text": (
                    "PROFILE_EXPORT_API_KEY=CYBERLAB_DO_NOT_USE_API_KEY_7F3A "
                    "# synthetic lab credential"
                ),
            },
            {
                "evidence_id": "EVID-102",
                "source": "repo_snapshot/config/gateway.env",
                "service_id": "gateway",
                "tags": ["config", "benign"],
                "text": "LOG_LEVEL=info; FEATURE_SAFE_HEADERS=true",
            },
        ],
        "logs": [
            {
                "evidence_id": "EVID-103",
                "service_id": "profile-service",
                "tags": ["benign", "startup"],
                "text": "profile-service boot completed for synthetic tenant secops-lab",
            }
        ],
        "headers": {
            "gateway": {
                "evidence_id": "EVID-104",
                "present": ["Strict-Transport-Security", "Content-Security-Policy"],
                "missing": [],
            }
        },
        "dependencies": [
            {
                "evidence_id": "EVID-105",
                "source": "repo_snapshot/requirements.lock",
                "tags": ["dependency", "benign"],
                "text": "fastapi==0.115.0; pydantic==2.8.2",
            }
        ],
    },
    "missing_security_headers_medium": {
        "task_id": "missing_security_headers_medium",
        "difficulty": "medium",
        "title": "Missing security headers",
        "alert": (
            "Gateway response-hardening alert: verify whether required security "
            "headers are missing or weak and submit evidence-backed remediation."
        ),
        "ground_truth_id": "GT-HEADERS-001",
        "finding_type": "missing_security_headers",
        "severity": "medium",
        "required_evidence": ["EVID-201"],
        "remediation_keywords": ["hsts", "csp"],
        "impact_keywords": ["header", "hsts", "csp", "clickjacking"],
        "assets": {
            "services": ["gateway", "profile-service", "admin-service"],
            "routes": ["/login", "/profile", "/admin/export"],
            "collections": ["repo_snapshot", "telemetry", "headers", "dependencies"],
        },
        "repo": [
            {
                "evidence_id": "EVID-202",
                "source": "repo_snapshot/gateway/security_headers.py",
                "service_id": "gateway",
                "tags": ["headers", "config"],
                "text": "X-Frame-Options is set, but HSTS and CSP are not configured.",
            }
        ],
        "logs": [
            {
                "evidence_id": "EVID-203",
                "service_id": "gateway",
                "tags": ["benign", "response"],
                "text": "GET /profile 200 request_id=req-442 synthetic header audit",
            }
        ],
        "headers": {
            "gateway": {
                "evidence_id": "EVID-201",
                "present": ["X-Frame-Options", "X-Content-Type-Options"],
                "missing": ["Strict-Transport-Security", "Content-Security-Policy"],
            }
        },
        "dependencies": [
            {
                "evidence_id": "EVID-204",
                "source": "repo_snapshot/requirements.lock",
                "tags": ["dependency", "benign"],
                "text": "starlette==0.38.2; uvicorn==0.30.1",
            }
        ],
    },
    "authz_boundary_hard": {
        "task_id": "authz_boundary_hard",
        "difficulty": "hard",
        "title": "Authorisation boundary misconfiguration",
        "alert": (
            "Access-control drift alert: investigate a route/role mismatch in the "
            "synthetic admin service and submit a validated remediation report."
        ),
        "ground_truth_id": "GT-AUTHZ-001",
        "finding_type": "authz_boundary_misconfiguration",
        "severity": "critical",
        "required_evidence": ["EVID-301"],
        "supporting_evidence": ["EVID-302"],
        "remediation_keywords": ["least privilege", "policy", "regression"],
        "impact_keywords": ["authorization", "authorisation", "role", "admin"],
        "assets": {
            "services": ["gateway", "profile-service", "admin-service"],
            "routes": ["/login", "/profile", "/admin/export"],
            "collections": ["repo_snapshot", "telemetry", "headers", "dependencies"],
        },
        "repo": [
            {
                "evidence_id": "EVID-301",
                "source": "repo_snapshot/admin-service/policy_matrix.yaml",
                "service_id": "admin-service",
                "tags": ["authorization", "role", "policy", "admin export"],
                "text": (
                    "route=/admin/export allowed_roles=[admin, analyst] "
                    "expected_roles=[admin]"
                ),
            }
        ],
        "logs": [
            {
                "evidence_id": "EVID-302",
                "service_id": "admin-service",
                "tags": ["authorization", "role", "admin export"],
                "text": (
                    "request_id=req-913 route=/admin/export role=analyst "
                    "decision=allow synthetic boundary-check event"
                ),
            },
            {
                "evidence_id": "EVID-303",
                "service_id": "gateway",
                "tags": ["benign", "auth"],
                "text": "request_id=req-912 route=/profile role=user decision=allow",
            },
        ],
        "headers": {
            "admin-service": {
                "evidence_id": "EVID-304",
                "present": ["Strict-Transport-Security", "Content-Security-Policy"],
                "missing": [],
            }
        },
        "dependencies": [
            {
                "evidence_id": "EVID-305",
                "source": "repo_snapshot/requirements.lock",
                "tags": ["dependency", "benign"],
                "text": "pyyaml==6.0.2; fastapi==0.115.0",
            }
        ],
    },
}


def list_task_ids() -> list[str]:
    return list(SCENARIOS)


def build_scenario(task_id: str | None, seed: int | None = None) -> dict[str, Any]:
    """Return a deep-copied scenario with deterministic benign variation."""

    selected_task_id = task_id if task_id in SCENARIOS else DEFAULT_TASK_ID
    scenario = deepcopy(SCENARIOS[selected_task_id])
    scenario["seed"] = seed

    rng = Random(seed if seed is not None else 0)
    service_alias_sets = [
        ["gateway", "profile-service", "admin-service"],
        ["edge-gateway", "user-profile", "admin-service"],
        ["public-gateway", "profile-api", "backoffice-admin"],
    ]
    aliases = service_alias_sets[rng.randrange(len(service_alias_sets))]
    original_services = scenario["assets"]["services"]
    alias_map = dict(zip(original_services, aliases, strict=True))

    scenario["service_aliases"] = alias_map
    scenario["assets"]["services"] = [alias_map.get(s, s) for s in original_services]

    for collection_name in ("repo", "logs"):
        for item in scenario.get(collection_name, []):
            service_id = item.get("service_id")
            if service_id in alias_map:
                item["service_id"] = alias_map[service_id]

    scenario["headers"] = {
        alias_map.get(service_id, service_id): snapshot
        for service_id, snapshot in scenario.get("headers", {}).items()
    }

    for entries_name in ("repo", "logs", "dependencies"):
        rng.shuffle(scenario.get(entries_name, []))

    return scenario
