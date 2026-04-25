"""CyberSecurity_OWASP OpenEnv environment implementation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        CyberSecurityOWASPAction,
        CyberSecurityOWASPObservation,
        CyberSecurityOWASPState,
    )
    from ..scenario_compiler import compile_scenario
    from ..safety import is_local_route
    from ..validators import detect_cheating, is_path_allowed, simulate_request
    from .reward_engine import evaluate_action
except ImportError:  # pragma: no cover
    from models import CyberSecurityOWASPAction, CyberSecurityOWASPObservation, CyberSecurityOWASPState
    from scenario_compiler import compile_scenario
    from safety import is_local_route
    from validators import detect_cheating, is_path_allowed, simulate_request
    from server.reward_engine import evaluate_action


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
        scenario = compile_scenario(actual_seed, split=split, difficulty=difficulty)
        self._state = CyberSecurityOWASPState(
            episode_id=episode_id or str(uuid4()),
            task_id=scenario["task_id"],
            seed=actual_seed,
            split=split,
            difficulty=difficulty,
            domain=scenario["domain"],
            bug_family=scenario["bug_family"],
            phase="discover",
            step_count=0,
            max_steps=40,
            done=False,
            success=False,
            visible_facts={"workspace_summary": scenario["workspace_summary"]},
            hidden_facts=scenario["hidden_facts"],
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
            verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)
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
            verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)
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
        verifier: dict = {"anti_cheat_flags": anti_cheat_flags}
        reward = {key: 0.0 for key in (
            "discovery",
            "security",
            "regression",
            "public_routes",
            "patch_quality",
            "visible_tests",
            "safety",
            "anti_cheat",
            "total",
        )}
        visible_tests = None
        args = action.arguments or {}

        if action.tool_name == "noop":
            return "No operation.", verifier, reward, None
        if action.tool_name == "inspect_policy_graph":
            return json.dumps(self._visible_policy_hint, indent=2, sort_keys=True), verifier, reward, None
        if action.tool_name == "list_routes":
            return json.dumps(self._workspace_summary["routes"], indent=2), verifier, reward, None
        if action.tool_name == "read_openapi":
            return json.dumps(
                {
                    "openapi": "3.1.0",
                    "info": {"title": "Generated invoices app", "version": "0.1.0"},
                    "paths": {
                        "/health": {"get": {"x-public": True}},
                        "/invoices/{invoice_id}": {"get": {"x-public": False}},
                    },
                },
                indent=2,
            ), verifier, reward, None
        if action.tool_name == "read_file":
            path = self._resolve_path(str(args.get("path", "")))
            return path.read_text(encoding="utf-8"), verifier, reward, None
        if action.tool_name == "search_code":
            return self._search_code(str(args.get("query", ""))), verifier, reward, None
        if action.tool_name == "send_local_request":
            if not is_local_route(str(args.get("path", ""))):
                raise ValueError("send_local_request only accepts local route paths")
            response = simulate_request(
                self._state,
                str(args.get("method", "GET")),
                str(args.get("path", "")),
                args.get("user_id"),
            )
            return json.dumps(response, indent=2, sort_keys=True), verifier, reward, None
        if action.tool_name == "compare_identities":
            path = str(args.get("path", ""))
            first = str(args.get("first_user_id", ""))
            second = str(args.get("second_user_id", ""))
            if not is_local_route(path):
                raise ValueError("compare_identities only accepts local route paths")
            response = {
                "first": simulate_request(self._state, str(args.get("method", "GET")), path, first),
                "second": simulate_request(self._state, str(args.get("method", "GET")), path, second),
            }
            return json.dumps(response, indent=2, sort_keys=True), verifier, reward, None
        if action.tool_name == "submit_finding":
            verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)
            if verifier.get("finding", {}).get("valid"):
                self._state.finding_submitted = True
                self._state.phase = "patch"
                return "Finding accepted. Patch phase unlocked.", verifier, reward, None
            return "Finding was not specific enough to unlock patching.", verifier, reward, None
        if action.tool_name == "patch_file":
            path = self._resolve_path(str(args.get("path", "")), write=True)
            if "content" in args:
                path.write_text(str(args["content"]), encoding="utf-8")
            else:
                self._apply_unified_diff(path, str(args.get("diff", "")))
            return f"Patched {args.get('path')}.", verifier, reward, None
        if action.tool_name == "run_visible_tests":
            verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)
            visible_tests = json.dumps(verifier.get("visible", {}), indent=2, sort_keys=True)
            return visible_tests, verifier, reward, visible_tests
        if action.tool_name == "submit_fix":
            verifier, reward = evaluate_action(self._state, action, anti_cheat_flags)
            self._state.patch_submitted = True
            security = verifier.get("security", {}).get("passed", False)
            regression = verifier.get("regression", {}).get("passed", False)
            public = verifier.get("public_routes", {}).get("passed", False)
            quality = verifier.get("patch_quality", {}).get("passed", False)
            self._state.success = bool(security and regression and public and quality)
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
        if self._state.done:
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

    def _resolve_path(self, path: str, *, write: bool = False) -> Path:
        allowed, normalized_or_error = is_path_allowed(self._state, path, write=write)
        if not allowed:
            raise ValueError(normalized_or_error)
        return Path(str(self._state.hidden_facts["workspace"])) / normalized_or_error

    def _search_code(self, query: str) -> str:
        if not query:
            raise ValueError("query is required")
        results: list[str] = []
        workspace = Path(str(self._state.hidden_facts["workspace"]))
        for rel in self._state.hidden_facts.get("editable_files", []):
            path = workspace / rel
            text = path.read_text(encoding="utf-8")
            for idx, line in enumerate(text.splitlines(), start=1):
                if query.lower() in line.lower():
                    results.append(f"{rel}:{idx}: {line}")
        return "\n".join(results) or "No matches."

    def _apply_unified_diff(self, path: Path, diff: str) -> None:
        if not diff.strip():
            raise ValueError("diff or content is required")
        original = path.read_text(encoding="utf-8").splitlines(True)
        output: list[str] = []
        old_index = 0
        lines = diff.splitlines(True)
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.startswith("@@"):
                i += 1
                continue
            old_start = int(line.split()[1].split(",")[0][1:])
            output.extend(original[old_index : old_start - 1])
            old_index = old_start - 1
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                hunk_line = lines[i]
                if hunk_line.startswith(" "):
                    output.append(original[old_index])
                    old_index += 1
                elif hunk_line.startswith("-"):
                    old_index += 1
                elif hunk_line.startswith("+"):
                    output.append(hunk_line[1:])
                elif hunk_line.startswith("\\"):
                    pass
                i += 1
        output.extend(original[old_index:])
        path.write_text("".join(output), encoding="utf-8")
