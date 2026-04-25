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
        "submit_diagnosis",
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
    difficulty_tier: str = "warmup"
    domain: str = ""
    bug_family: str = ""
    scenario_family: str = ""
    template_id: str = "fastapi_basic"
    target_weakness: str = "same_role_cross_object"
    cache_key: dict[str, Any] = Field(default_factory=dict)
    scenario_hash: str = ""
    generator_version: str = ""
    verifier_version: str = ""
    cache_hit: bool = False
    reset_latency_ms: float = 0.0
    phase: CyberSecurityOWASPPhase = "discover"
    max_steps: int = 40
    done: bool = False
    success: bool = False
    failure_reason: str | None = None
    finding_submitted: bool = False
    diagnosis_submitted: bool = False
    patch_submitted: bool = False
    accumulated_reward: float = 0.0
    last_reward: float = 0.0
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    reward_history: list[dict[str, float]] = Field(default_factory=list)
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    progress_reward_total: float = 0.0
    diagnosis: dict[str, Any] = Field(default_factory=dict)
    request_trace: list[dict[str, Any]] = Field(default_factory=list)
    patch_attempt_count: int = 0
    visible_test_count: int = 0
    completion_tokens: int = 0
    visible_facts: dict[str, Any] = Field(default_factory=dict)
    hidden_facts: dict[str, Any] = Field(default_factory=dict)
    curriculum_snapshot: dict[str, Any] = Field(default_factory=dict)
    verification_summary: dict[str, Any] = Field(default_factory=dict)
    patch_diff: str = ""
    episode_artifact_path: str | None = None
    observation_history: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    anti_cheat_flags: list[str] = Field(default_factory=list)


# Backward-compatible aliases from the OpenEnv scaffold.
CybersecurityOwaspAction = CyberSecurityOWASPAction
CybersecurityOwaspObservation = CyberSecurityOWASPObservation
CybersecurityOwaspState = CyberSecurityOWASPState
