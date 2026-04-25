"""Server-side verifier aggregation for terminal scoring."""

from __future__ import annotations

try:
    from ..models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from ..rewards import compute_reward
    from .verifier import MultiLayerVerifier
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPAction, CyberSecurityOWASPState
    from rewards import compute_reward
    from server.verifier import MultiLayerVerifier


def evaluate_action(
    state: CyberSecurityOWASPState,
    action: CyberSecurityOWASPAction,
    anti_cheat_flags: list[str] | None = None,
    *,
    invalid_action: bool = False,
) -> tuple[dict, dict[str, float]]:
    verifier_result = MultiLayerVerifier().evaluate_action(
        state,
        action,
        anti_cheat_flags,
        invalid_action=invalid_action,
    )
    return verifier_result, compute_reward(state, action, verifier_result)
