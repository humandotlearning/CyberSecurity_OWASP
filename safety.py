"""Safety boundaries for local-only defensive AppSec episodes."""

from __future__ import annotations


FORBIDDEN_BEHAVIOR = (
    "external network access",
    "host filesystem reads",
    "hidden test access",
    "oracle access",
    "credential extraction",
    "persistence or evasion",
)


def is_local_route(path: str) -> bool:
    return path.startswith("/") and not path.startswith("//") and "://" not in path
