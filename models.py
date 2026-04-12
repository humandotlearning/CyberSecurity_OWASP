# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the Cyber Analyst OpenEnv environment."""

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class CyberAnalystAction(Action):
    """A bounded simulator tool call."""

    tool_name: str = Field(..., description="Name of the approved simulator tool")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments. The environment ignores unsupported keys.",
    )


class CyberAnalystObservation(Observation):
    """Observation returned after reset or an environment step."""

    task_id: str = Field(default="", description="Current benchmark task id")
    alert: str = Field(default="", description="Initial alert or task prompt")
    phase: str = Field(default="investigate", description="Current episode phase")
    tool_catalog: list[dict[str, Any]] = Field(
        default_factory=list, description="Approved tools and their schemas"
    )
    tool_result: dict[str, Any] = Field(
        default_factory=dict, description="Result returned by the latest tool call"
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="Evidence ids discovered so far"
    )
    verified_findings: list[dict[str, Any]] = Field(
        default_factory=list, description="Verifier-confirmed findings"
    )
    candidate_findings: list[dict[str, Any]] = Field(
        default_factory=list, description="Candidate findings created by the agent"
    )
    step_budget_remaining: int = Field(
        default=0, ge=0, description="Steps remaining before timeout"
    )
    score_breakdown: dict[str, Any] = Field(
        default_factory=dict, description="Deterministic reward/score explanation"
    )
    error: str = Field(default="", description="Non-fatal environment error, if any")


class CyberAnalystState(State):
    """State summary exposed via the OpenEnv state endpoint."""

    task_id: str = Field(default="", description="Current benchmark task id")
    seed: int | None = Field(default=None, description="Current deterministic seed")
    phase: str = Field(default="investigate", description="Current episode phase")
    step_budget_remaining: int = Field(default=0, ge=0)
    recent_evidence_ids: list[str] = Field(default_factory=list)
    verified_finding_ids: list[str] = Field(default_factory=list)
    done: bool = Field(default=False)
