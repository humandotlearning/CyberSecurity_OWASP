"""CyberSecurity_OWASP OpenEnv client."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    CyberSecurityOWASPAction,
    CyberSecurityOWASPObservation,
    CyberSecurityOWASPState,
)


class CyberSecurityOWASPEnv(
    EnvClient[CyberSecurityOWASPAction, CyberSecurityOWASPObservation, CyberSecurityOWASPState]
):
    """WebSocket client for the CyberSecurity_OWASP environment."""

    def _step_payload(self, action: CyberSecurityOWASPAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CyberSecurityOWASPObservation]:
        obs_data = payload.get("observation", {})
        observation = CyberSecurityOWASPObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> CyberSecurityOWASPState:
        return CyberSecurityOWASPState(**payload)


# Backward-compatible alias from generated scaffold.
CybersecurityOwaspEnv = CyberSecurityOWASPEnv
