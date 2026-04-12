# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cyber Analyst Environment client."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import CyberAnalystAction, CyberAnalystObservation, CyberAnalystState


class CyberAnalystEnv(
    EnvClient[CyberAnalystAction, CyberAnalystObservation, CyberAnalystState]
):
    """WebSocket client for the Cyber Analyst OpenEnv environment."""

    def _step_payload(self, action: CyberAnalystAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[CyberAnalystObservation]:
        obs_data = dict(payload.get("observation", {}))
        obs_data["done"] = payload.get("done", False)
        obs_data["reward"] = payload.get("reward")
        observation = CyberAnalystObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> CyberAnalystState:
        return CyberAnalystState.model_validate(payload)
