# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SecOps Evidence Gym environment implementation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        CyberAnalystAction,
        CyberAnalystObservation,
        CyberAnalystState,
    )
    from .graders import safe_reward, score_report
    from .tasks import DEFAULT_TASK_ID, TOOL_CATALOG, build_scenario
except ImportError:  # pragma: no cover - supports direct module execution
    from models import CyberAnalystAction, CyberAnalystObservation, CyberAnalystState
    from server.graders import safe_reward, score_report
    from server.tasks import DEFAULT_TASK_ID, TOOL_CATALOG, build_scenario


class CyberAnalystEnvironment(
    Environment[CyberAnalystAction, CyberAnalystObservation, CyberAnalystState]
):
    """A safe, deterministic evidence-grounded cyber analyst benchmark."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 12
    REPEAT_HARD_STOP = 6

    def __init__(self):
        super().__init__()
        self._scenario: dict[str, Any] = {}
        self._state = CyberAnalystState()
        self._discovered_evidence: set[str] = set()
        self._candidate_findings: dict[str, dict[str, Any]] = {}
        self._verified_findings: list[dict[str, Any]] = []
        self._validated_finding_ids: set[str] = set()
        self._action_counts: Counter[str] = Counter()
        self._last_score_breakdown: dict[str, Any] = {}
        self._trajectory_events: list[dict[str, Any]] = []
        self._initialize_episode(DEFAULT_TASK_ID, seed=None, episode_id=None)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = DEFAULT_TASK_ID,
        **_: Any,
    ) -> CyberAnalystObservation:
        """Reset the selected deterministic task."""

        self._initialize_episode(task_id=task_id, seed=seed, episode_id=episode_id)
        tool_result = {
            "message": "Cyber Analyst environment ready.",
            "allowed_scope": "Synthetic artifacts only. No live targets or shell.",
        }
        obs = self._observation(
            tool_result={
                **tool_result,
                "trajectory_jsonl": self.export_trajectory_jsonl(),
            },
            reward=0.01,
        )
        self._record_trajectory("reset", None, tool_result, obs.reward, obs.done, obs.error)
        return obs

    def step(  # type: ignore[override]
        self,
        action: CyberAnalystAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> CyberAnalystObservation:
        """Execute one bounded simulator tool call."""

        del timeout_s

        if self._state.done:
            tool_result = {"message": "Episode is already complete."}
            obs = self._observation(
                tool_result=tool_result,
                reward=0.01,
                done=True,
                error="episode_already_done",
            )
            self._record_trajectory("step", action, tool_result, obs.reward, obs.done, obs.error)
            return obs

        self._state.step_count += 1
        self._state.step_budget_remaining = max(
            0, self.MAX_STEPS - self._state.step_count
        )

        signature = self._action_signature(action)
        self._action_counts[signature] += 1
        repeat_count = self._action_counts[signature]

        if repeat_count >= self.REPEAT_HARD_STOP:
            self._state.phase = "done"
            self._state.done = True
            self._last_score_breakdown = {
                "score": 0.03,
                "repeat_hard_stop": True,
                "signature": signature,
            }
            tool_result = {"message": "Episode stopped after repeated identical actions."}
            obs = self._observation(
                tool_result=tool_result,
                reward=0.03,
                done=True,
                error="repeat_hard_stop",
            )
            self._record_trajectory("step", action, tool_result, obs.reward, obs.done, obs.error)
            return obs

        handler = getattr(self, f"_tool_{action.tool_name}", None)
        if handler is None:
            tool_result = {
                "ok": False,
                "message": f"Unsupported tool: {action.tool_name}",
                "available_tools": [tool["name"] for tool in TOOL_CATALOG],
            }
            obs = self._step_observation(
                tool_result=tool_result,
                repeat_count=repeat_count,
                error="unsupported_tool",
            )
            self._record_trajectory("step", action, tool_result, obs.reward, obs.done, obs.error)
            return obs

        try:
            result, reward_delta, done = handler(action.args)
            error = ""
        except Exception as exc:  # pragma: no cover - defensive rollout guard
            result = {"ok": False, "message": str(exc)}
            reward_delta = -0.05
            done = False
            error = exc.__class__.__name__

        if self._state.step_budget_remaining <= 0 and not done:
            done = True
            self._state.phase = "done"
            self._state.done = True
            result = {
                **result,
                "timeout": True,
                "message": "Step budget exhausted before report submission.",
            }
            reward_delta -= 0.10

        obs = self._step_observation(
            tool_result=result,
            repeat_count=repeat_count,
            reward_delta=reward_delta,
            done=done,
            error=error,
        )
        self._record_trajectory("step", action, result, obs.reward, obs.done, obs.error)
        return obs

    @property
    def state(self) -> CyberAnalystState:
        """Return the current episode state summary."""

        return self._state

    def _initialize_episode(
        self, task_id: str, seed: int | None, episode_id: str | None
    ) -> None:
        self._scenario = build_scenario(task_id, seed)
        self._discovered_evidence = set()
        self._candidate_findings = {}
        self._verified_findings = []
        self._validated_finding_ids = set()
        self._action_counts = Counter()
        self._last_score_breakdown = {}
        self._trajectory_events = []
        self._state = CyberAnalystState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._scenario["task_id"],
            seed=seed,
            phase="investigate",
            step_budget_remaining=self.MAX_STEPS,
            recent_evidence_ids=[],
            verified_finding_ids=[],
            done=False,
        )

    def export_trajectory_jsonl(self) -> str:
        """Return the current episode trajectory as JSONL for offline analysis."""

        return "\n".join(
            json.dumps(event, sort_keys=True, default=str)
            for event in self._trajectory_events
        )

    def _record_trajectory(
        self,
        event_type: str,
        action: CyberAnalystAction | None,
        tool_result: dict[str, Any],
        reward: float | int | None,
        done: bool,
        error: str,
    ) -> None:
        action_payload = None
        if action is not None:
            action_payload = action.model_dump(exclude_none=True)
        self._trajectory_events.append(
            {
                "episode_id": self._state.episode_id,
                "task_id": self._state.task_id,
                "seed": self._state.seed,
                "event_type": event_type,
                "step": self._state.step_count,
                "phase": self._state.phase,
                "action": action_payload,
                "tool_result": tool_result,
                "evidence_ids": sorted(self._discovered_evidence),
                "verified_finding_ids": list(self._state.verified_finding_ids),
                "reward": reward,
                "done": done,
                "error": error,
            }
        )

    def _observation(
        self,
        tool_result: dict[str, Any] | None = None,
        reward: float = 0.01,
        done: bool | None = None,
        error: str = "",
    ) -> CyberAnalystObservation:
        done_value = self._state.done if done is None else done
        return CyberAnalystObservation(
            task_id=self._scenario.get("task_id", ""),
            alert=self._scenario.get("alert", ""),
            phase=self._state.phase,
            tool_catalog=TOOL_CATALOG,
            tool_result=tool_result or {},
            evidence_ids=sorted(self._discovered_evidence),
            verified_findings=list(self._verified_findings),
            candidate_findings=list(self._candidate_findings.values()),
            step_budget_remaining=self._state.step_budget_remaining,
            score_breakdown=dict(self._last_score_breakdown),
            error=error,
            done=done_value,
            reward=safe_reward(reward),
        )

    def _step_observation(
        self,
        tool_result: dict[str, Any],
        repeat_count: int,
        reward_delta: float = 0.0,
        done: bool = False,
        error: str = "",
    ) -> CyberAnalystObservation:
        reward = 0.04 + reward_delta - 0.01
        if repeat_count > 2:
            reward -= 0.03 * (repeat_count - 2)

        if done:
            self._state.phase = "done"
            self._state.done = True

        self._state.recent_evidence_ids = sorted(self._discovered_evidence)[-5:]
        self._state.verified_finding_ids = [
            finding["finding_id"] for finding in self._verified_findings
        ]

        return self._observation(
            tool_result=tool_result,
            reward=safe_reward(reward),
            done=self._state.done,
            error=error,
        )

    def _action_signature(self, action: CyberAnalystAction) -> str:
        payload = {
            "tool_name": action.tool_name,
            "args": action.args,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def _record_evidence(self, evidence_ids: list[str]) -> int:
        relevant = set(self._scenario.get("required_evidence", [])) | set(
            self._scenario.get("supporting_evidence", [])
        )
        new_relevant = 0
        for evidence_id in evidence_ids:
            if evidence_id not in self._discovered_evidence and evidence_id in relevant:
                new_relevant += 1
            self._discovered_evidence.add(evidence_id)
        return new_relevant

    def _filter_entries(
        self, entries: list[dict[str, Any]], service_id: str = "", query: str = ""
    ) -> list[dict[str, Any]]:
        normalized_service = self._resolve_service_id(service_id).lower()
        normalized_query = query.strip().lower()
        matches: list[dict[str, Any]] = []
        for entry in entries:
            service_matches = (
                not normalized_service
                or str(entry.get("service_id", "")).lower() == normalized_service
            )
            search_blob = " ".join(
                [
                    str(entry.get("text", "")),
                    str(entry.get("source", "")),
                    " ".join(str(tag) for tag in entry.get("tags", [])),
                ]
            ).lower()
            query_matches = not normalized_query or normalized_query in search_blob
            if service_matches and query_matches:
                matches.append(entry)
        return matches

    def _resolve_service_id(self, service_id: str) -> str:
        normalized = service_id.strip()
        aliases = self._scenario.get("service_aliases", {})
        return str(aliases.get(normalized, normalized))

    def _evidence_payload(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        evidence_ids = [entry["evidence_id"] for entry in entries]
        new_relevant = self._record_evidence(evidence_ids)
        return {
            "ok": True,
            "evidence_ids": evidence_ids,
            "new_relevant_evidence": new_relevant,
            "entries": [
                {
                    "evidence_id": entry["evidence_id"],
                    "service_id": entry.get("service_id", ""),
                    "source": entry.get("source", ""),
                    "text": entry.get("text", ""),
                }
                for entry in entries
            ],
        }

    def _tool_list_assets(self, args: dict[str, Any]) -> tuple[dict[str, Any], float, bool]:
        del args
        return {"ok": True, "assets": self._scenario["assets"]}, 0.0, False

    def _tool_get_log_events(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool]:
        entries = self._filter_entries(
            self._scenario.get("logs", []),
            service_id=str(args.get("service_id", "")),
            query=str(args.get("query", "")),
        )
        payload = self._evidence_payload(entries)
        return payload, 0.02 * payload["new_relevant_evidence"], False

    def _tool_check_security_headers(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool]:
        requested_service = self._resolve_service_id(str(args.get("service_id", ""))).lower()
        snapshots = self._scenario.get("headers", {})
        results = []
        evidence_ids = []
        for service_id, snapshot in snapshots.items():
            if requested_service and service_id.lower() != requested_service:
                continue
            evidence_ids.append(snapshot["evidence_id"])
            results.append(
                {
                    "service_id": service_id,
                    "evidence_id": snapshot["evidence_id"],
                    "present": snapshot.get("present", []),
                    "missing": snapshot.get("missing", []),
                    "passed": not snapshot.get("missing"),
                }
            )
        new_relevant = self._record_evidence(evidence_ids)
        return (
            {
                "ok": True,
                "evidence_ids": evidence_ids,
                "new_relevant_evidence": new_relevant,
                "header_results": results,
            },
            0.02 * new_relevant,
            False,
        )

    def _tool_search_repo(self, args: dict[str, Any]) -> tuple[dict[str, Any], float, bool]:
        entries = self._filter_entries(
            self._scenario.get("repo", []), query=str(args.get("query", ""))
        )
        payload = self._evidence_payload(entries)
        return payload, 0.02 * payload["new_relevant_evidence"], False

    def _tool_scan_dependencies(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool]:
        del args
        payload = self._evidence_payload(self._scenario.get("dependencies", []))
        return payload, 0.02 * payload["new_relevant_evidence"], False

    def _tool_create_finding(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool]:
        evidence_ids = args.get("evidence_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        evidence_ids = [str(evidence_id) for evidence_id in evidence_ids]

        finding_id = f"FND-{len(self._candidate_findings) + 1:03d}"
        finding = {
            "finding_id": finding_id,
            "finding_type": str(args.get("finding_type", "")),
            "evidence_ids": evidence_ids,
            "severity_guess": str(args.get("severity_guess", "")),
            "remediation": str(args.get("remediation", "")),
            "validated": False,
            "matching_gt_id": None,
        }
        self._candidate_findings[finding_id] = finding

        well_formed = bool(
            finding["finding_type"] and evidence_ids and finding["remediation"]
        )
        return (
            {"ok": True, "finding_id": finding_id, "finding": finding},
            0.03 if well_formed else 0.0,
            False,
        )

    def _tool_validate_finding(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool]:
        finding_id = str(args.get("finding_id", ""))
        finding = self._candidate_findings.get(finding_id)
        if finding is None:
            return (
                {"ok": False, "message": f"Unknown finding_id: {finding_id}"},
                -0.03,
                False,
            )

        expected_type = self._scenario["finding_type"]
        required_evidence = set(self._scenario.get("required_evidence", []))
        supplied_evidence = set(finding.get("evidence_ids", []))
        verified = (
            finding.get("finding_type") == expected_type
            and bool(required_evidence & supplied_evidence)
        )
        self._validated_finding_ids.add(finding_id)
        finding["validated"] = verified
        finding["matching_gt_id"] = self._scenario["ground_truth_id"] if verified else None

        if verified and not any(
            item["finding_id"] == finding_id for item in self._verified_findings
        ):
            self._verified_findings.append(dict(finding))

        return (
            {
                "ok": True,
                "finding_id": finding_id,
                "verified": verified,
                "matching_gt_id": finding["matching_gt_id"],
            },
            0.08 if verified else -0.02,
            False,
        )

    def _tool_submit_report(
        self, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool]:
        report = args.get("report_json", {})
        score, breakdown = score_report(
            self._scenario["task_id"],
            report,
            verified_findings=self._verified_findings,
            validation_attempted=bool(self._validated_finding_ids),
        )
        self._last_score_breakdown = breakdown
        return (
            {
                "ok": True,
                "submitted": True,
                "score": score,
                "score_breakdown": breakdown,
                "trajectory_jsonl": self.export_trajectory_jsonl(),
            },
            score,
            True,
        )
