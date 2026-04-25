"""Typed OpenEnv models for the CyberSecurity_OWASP environment."""

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


CyberSecurityOWASPPhase = Literal["discover", "patch", "done"]
CyberSecurityOWASPSplit = Literal["train", "validation", "hidden_eval"]


class CyberSecurityOWASPAction(Action):
    """One typed action emitted by the single defensive AppSec agent."""

    tool_name: Literal[
        "inspect_policy_graph",
        "list_routes",
        "read_openapi",
        "read_file",
        "search_code",
        "send_local_request",
        "compare_identities",
        "submit_finding",
        "patch_file",
        "run_visible_tests",
        "submit_fix",
        "noop",
    ] = Field(..., description="Tool to execute for this step")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="JSON-serializable tool arguments"
    )


class CyberSecurityOWASPObservation(Observation):
    """Structured observation returned after reset and every action."""

    phase: CyberSecurityOWASPPhase = "discover"
    message: str = ""
    task_brief: str = ""
    visible_policy_hint: dict[str, Any] = Field(default_factory=dict)
    workspace_summary: dict[str, Any] = Field(default_factory=dict)
    available_actions: list[str] = Field(default_factory=list)
    last_tool_result: str = ""
    last_action_valid: bool = True
    last_action_error: str | None = None
    visible_test_result: str | None = None
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    done_reason: str | None = None


class CyberSecurityOWASPState(State):
    """Internal state used for replay, validation, reward, and eval logging."""

    task_id: str = ""
    seed: int = 0
    split: CyberSecurityOWASPSplit = "train"
    difficulty: int = 0
    domain: str = ""
    bug_family: str = ""
    phase: CyberSecurityOWASPPhase = "discover"
    max_steps: int = 40
    done: bool = False
    success: bool = False
    failure_reason: str | None = None
    finding_submitted: bool = False
    patch_submitted: bool = False
    accumulated_reward: float = 0.0
    last_reward: float = 0.0
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    reward_history: list[dict[str, float]] = Field(default_factory=list)
    visible_facts: dict[str, Any] = Field(default_factory=dict)
    hidden_facts: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    anti_cheat_flags: list[str] = Field(default_factory=list)


# Backward-compatible aliases from the OpenEnv scaffold.
CybersecurityOwaspAction = CyberSecurityOWASPAction
CybersecurityOwaspObservation = CyberSecurityOWASPObservation
CybersecurityOwaspState = CyberSecurityOWASPState
