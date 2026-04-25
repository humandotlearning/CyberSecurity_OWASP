"""CyberSecurity_OWASP OpenEnv environment implementation."""

from __future__ import annotations

import json
import shutil
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        CyberSecurityOWASPAction,
        CyberSecurityOWASPObservation,
        CyberSecurityOWASPState,
    )
    from ..validators import detect_cheating
    from .action_tools import ActionTools
    from .curriculum import CurriculumController
    from .episode_logger import EpisodeArtifactLogger
    from .reward_engine import evaluate_action
    from .scenario_factory import ScenarioFactory
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPAction, CyberSecurityOWASPObservation, CyberSecurityOWASPState
    from validators import detect_cheating
    from server.action_tools import ActionTools
    from server.curriculum import CurriculumController
    from server.episode_logger import EpisodeArtifactLogger
    from server.reward_engine import evaluate_action
    from server.scenario_factory import ScenarioFactory


ALLOWED_TOOLS = {
    "discover": {
        "inspect_policy_graph",
        "list_routes",
        "read_openapi",
        "read_file",
        "search_code",
        "send_local_request",
        "compare_identities",
        "submit_finding",
        "noop",
    },
    "patch": {
        "read_file",
        "search_code",
        "patch_file",
        "run_visible_tests",
        "send_local_request",
        "submit_fix",
        "noop",
    },
    "done": set(),
}


class CybersecurityOwaspEnvironment(
    Environment[CyberSecurityOWASPAction, CyberSecurityOWASPObservation, CyberSecurityOWASPState]
):
    """Single-agent defensive authorization-repair environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = CyberSecurityOWASPState(episode_id=str(uuid4()))
        self._task_brief = ""
        self._visible_policy_hint: dict[str, Any] = {}
        self._workspace_summary: dict[str, Any] = {}
        self._last_done_observation: CyberSecurityOWASPObservation | None = None
        self._curriculum = CurriculumController()
        self._scenario_factory = ScenarioFactory()
        self._episode_logger = EpisodeArtifactLogger()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str = "train",
        difficulty: int = 0,
        **_: Any,
    ) -> CyberSecurityOWASPObservation:
        self.close()
        actual_seed = int(seed if seed is not None else 0)
        curriculum_profile = self._curriculum.select_profile(
            seed=actual_seed,
            split=split,
            requested_difficulty=difficulty,
        )
        scenario = self._scenario_factory.compile_scenario(
            actual_seed,
            split=split,
            difficulty=difficulty,
            curriculum_profile=curriculum_profile,
        )
        self._state = CyberSecurityOWASPState(
            episode_id=episode_id or str(uuid4()),
            task_id=scenario["task_id"],
            seed=actual_seed,
            split=split,
            difficulty=scenario["difficulty"],
            difficulty_tier=scenario["difficulty_tier"],
            domain=scenario["domain"],
            bug_family=scenario["bug_family"],
            scenario_family=scenario["scenario_family"],
            template_id=scenario["template_id"],
            target_weakness=scenario["target_weakness"],
            phase="discover",
            step_count=0,
            max_steps=40,
            done=False,
            success=False,
            visible_facts={"workspace_summary": scenario["workspace_summary"]},
            hidden_facts=scenario["hidden_facts"],
            curriculum_snapshot=scenario["curriculum_snapshot"],
            metrics={"reset_count": 1},
        )
        self._task_brief = scenario["task_brief"]
        self._visible_policy_hint = scenario["public_hint"]
        self._workspace_summary = scenario["workspace_summary"]
        self._last_done_observation = None
        return self._observation("Scenario ready. Start in discover phase.", reward=0.0)

    def step(
        self,
        action: CyberSecurityOWASPAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> CyberSecurityOWASPObservation:
        if self._state.done:
            return self._last_done_observation or self._observation(
                "Episode is already done.", reward=0.0, done_reason=self._state.failure_reason
            )

        anti_cheat_flags = detect_cheating(self._state, action)
        for flag in anti_cheat_flags:
            if flag not in self._state.anti_cheat_flags:
                self._state.anti_cheat_flags.append(flag)

        self._state.step_count += 1
        self._state.action_history.append(
            {"tool_name": action.tool_name, "arguments": action.arguments}
        )

        if action.tool_name not in ALLOWED_TOOLS[self._state.phase]:
            verifier, reward = evaluate_action(
                self._state,
                action,
                anti_cheat_flags,
                invalid_action=True,
            )
            return self._finish_step(
                "Action is not allowed in the current phase.",
                reward,
                valid=False,
                error=f"{action.tool_name} is not allowed during {self._state.phase}",
                verifier=verifier,
            )

        try:
            result, verifier, reward, visible_tests = self._execute(action, anti_cheat_flags)
            return self._finish_step(
                result,
                reward,
                valid=True,
                verifier=verifier,
                visible_test_result=visible_tests,
            )
        except Exception as exc:  # keep malformed agent actions from crashing the server
            verifier, reward = evaluate_action(
                self._state,
                action,
                anti_cheat_flags,
                invalid_action=True,
            )
            return self._finish_step(
                "Tool execution failed.",
                reward,
                valid=False,
                error=str(exc),
                verifier=verifier,
            )

    @property
    def state(self) -> CyberSecurityOWASPState:
        return self._state

    def close(self) -> None:
        workspace = self._state.hidden_facts.get("workspace")
        if workspace:
            shutil.rmtree(workspace, ignore_errors=True)

    def _execute(
        self, action: CyberSecurityOWASPAction, anti_cheat_flags: list[str]
    ) -> tuple[str, dict, dict[str, float], str | None]:
        verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)

        if action.tool_name in {
            "noop",
            "inspect_policy_graph",
            "list_routes",
            "read_openapi",
            "read_file",
            "search_code",
            "send_local_request",
            "compare_identities",
            "patch_file",
        }:
            result = ActionTools(
                self._state,
                self._visible_policy_hint,
                self._workspace_summary,
            ).execute(action)
            return result.message, verifier, reward, result.visible_test_result
        if action.tool_name == "submit_finding":
            verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)
            self._state.verification_summary = verifier
            if verifier.get("finding", {}).get("valid"):
                self._state.finding_submitted = True
                self._state.phase = "patch"
                return "Finding accepted. Patch phase unlocked.", verifier, reward, None
            return "Finding was not specific enough to unlock patching.", verifier, reward, None
        if action.tool_name == "run_visible_tests":
            verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)
            self._state.verification_summary = verifier
            visible_tests = json.dumps(verifier.get("visible", {}), indent=2, sort_keys=True)
            return visible_tests, verifier, reward, visible_tests
        if action.tool_name == "submit_fix":
            verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)
            self._state.verification_summary = verifier
            self._state.patch_submitted = True
            security = verifier.get("security", {}).get("passed", False)
            oracle = verifier.get("oracle_matrix", {}).get("passed", False)
            regression = verifier.get("regression", {}).get("passed", False)
            public = verifier.get("public_routes", {}).get("passed", False)
            quality = verifier.get("patch_quality", {}).get("passed", False)
            self._state.success = bool(security and oracle and regression and public and quality)
            self._state.done = True
            self._state.phase = "done"
            self._state.failure_reason = None if self._state.success else "hidden_verifier_failed"
            return json.dumps(verifier, indent=2, sort_keys=True), verifier, reward, None
        raise ValueError(f"Unhandled tool {action.tool_name}")

    def _finish_step(
        self,
        message: str,
        reward: dict[str, float],
        *,
        valid: bool,
        error: str | None = None,
        verifier: dict | None = None,
        visible_test_result: str | None = None,
    ) -> CyberSecurityOWASPObservation:
        self._state.last_reward = float(reward.get("total", 0.0))
        self._state.accumulated_reward += self._state.last_reward
        self._state.reward_history.append(reward)
        if self._state.step_count >= self._state.max_steps and not self._state.done:
            self._state.done = True
            self._state.phase = "done"
            self._state.failure_reason = "max_steps_exceeded"
        obs = self._observation(
            message,
            reward=self._state.last_reward,
            valid=valid,
            error=error,
            reward_breakdown=reward,
            visible_test_result=visible_test_result,
            done_reason=self._state.failure_reason,
        )
        observation_record = obs.model_dump()
        self._state.observation_history.append(observation_record)
        if self._state.done:
            self._finalize_terminal_episode(observation_record)
            self._last_done_observation = obs
        return obs

    def _observation(
        self,
        message: str,
        *,
        reward: float,
        valid: bool = True,
        error: str | None = None,
        reward_breakdown: dict[str, float] | None = None,
        visible_test_result: str | None = None,
        done_reason: str | None = None,
    ) -> CyberSecurityOWASPObservation:
        return CyberSecurityOWASPObservation(
            phase=self._state.phase,
            message=message,
            task_brief=self._task_brief,
            visible_policy_hint=self._visible_policy_hint,
            workspace_summary=self._workspace_summary,
            available_actions=sorted(ALLOWED_TOOLS[self._state.phase]),
            last_tool_result=message,
            last_action_valid=valid,
            last_action_error=error,
            visible_test_result=visible_test_result,
            reward_breakdown=reward_breakdown or {},
            done_reason=done_reason,
            done=self._state.done,
            reward=reward,
            metadata={"episode_id": self._state.episode_id, "step_count": self._state.step_count},
        )

    def _finalize_terminal_episode(self, observation_record: dict[str, Any]) -> None:
        if self._state.episode_artifact_path:
            return
        mastery = self._curriculum.record_episode(self._state)
        self._state.curriculum_snapshot = {
            **self._state.curriculum_snapshot,
            "post_episode_mastery": mastery,
        }
        self._episode_logger.log_episode(
            self._state,
            final_observation=observation_record,
        )
