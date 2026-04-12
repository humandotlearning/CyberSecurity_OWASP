"""Deterministic graders for the Cyber Analyst OpenEnv tasks."""

from __future__ import annotations

import json
from typing import Any

try:
    from .tasks import SCENARIOS
except ImportError:  # pragma: no cover - supports direct module execution
    from tasks import SCENARIOS


MIN_SCORE = 0.01
MAX_SCORE = 0.99


def safe_reward(score: float | int | None) -> float:
    """Clamp validator-facing scores to the strict open interval (0, 1)."""

    try:
        value = float(score if score is not None else 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(MIN_SCORE, min(MAX_SCORE, value))


def _coerce_report(report: Any) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    if isinstance(report, str):
        try:
            decoded = json.loads(report)
        except json.JSONDecodeError:
            return {"summary": report, "findings": []}
        return decoded if isinstance(decoded, dict) else {"findings": []}
    return {"findings": []}


def _text_contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _report_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings = report.get("findings", [])
    if isinstance(findings, dict):
        findings = [findings]
    return [finding for finding in findings if isinstance(finding, dict)]


def score_report(
    task_id: str,
    report: Any,
    verified_findings: list[dict[str, Any]] | None = None,
    validation_attempted: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Score a submitted report against one task's deterministic ground truth."""

    scenario = SCENARIOS.get(task_id)
    report_dict = _coerce_report(report)
    report_findings = _report_findings(report_dict)
    verified_findings = verified_findings or []

    if scenario is None:
        return MIN_SCORE, {"unknown_task": task_id}

    expected_type = scenario["finding_type"]
    expected_evidence = set(scenario.get("required_evidence", [])) | set(
        scenario.get("supporting_evidence", [])
    )

    matching_verified = [
        finding
        for finding in verified_findings
        if finding.get("finding_type") == expected_type
    ]
    matching_report = [
        finding for finding in report_findings if finding.get("finding_type") == expected_type
    ]

    score = 0.05
    breakdown: dict[str, Any] = {
        "base": 0.05,
        "verified_correct": 0.0,
        "valid_evidence": 0.0,
        "actionable_remediation": 0.0,
        "hallucination_penalty": 0.0,
        "validation_penalty": 0.0,
    }

    if matching_verified and matching_report:
        impact_text = " ".join(
            str(finding.get("impact", "")) + " " + str(finding.get("description", ""))
            for finding in matching_report
        )
        if _text_contains_any(impact_text, scenario.get("impact_keywords", [])):
            score += 0.60
            breakdown["verified_correct"] = 0.60

    report_evidence: set[str] = set()
    for finding in matching_report:
        evidence_ids = finding.get("evidence_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        report_evidence.update(str(evidence_id) for evidence_id in evidence_ids)

    if report_evidence & expected_evidence:
        score += 0.15
        breakdown["valid_evidence"] = 0.15

    remediation_text = " ".join(
        str(finding.get("remediation", "")) for finding in matching_report
    )
    if _text_contains_any(remediation_text, scenario.get("remediation_keywords", [])):
        score += 0.15
        breakdown["actionable_remediation"] = 0.15

    verified_types = {finding.get("finding_type") for finding in verified_findings}
    hallucinated = [
        finding
        for finding in report_findings
        if finding.get("finding_type") not in verified_types
    ]
    if hallucinated:
        penalty = 0.40 * len(hallucinated)
        score -= penalty
        breakdown["hallucination_penalty"] = -penalty

    if not validation_attempted:
        score -= 0.20
        breakdown["validation_penalty"] = -0.20

    final_score = safe_reward(score)
    breakdown["raw_score"] = round(score, 4)
    breakdown["score"] = final_score
    return final_score, breakdown


def _payload_from_args(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if args and isinstance(args[0], dict):
        payload = dict(args[0])
    else:
        payload = {}
    payload.update(kwargs)
    return payload


def grade_task(task_id: str, *args: Any, **kwargs: Any) -> float:
    """Manifest-friendly grader adapter."""

    payload = _payload_from_args(*args, **kwargs)
    report = payload.get("report") or payload.get("report_json") or payload
    verified_findings = payload.get("verified_findings", [])
    validation_attempted = bool(payload.get("validation_attempted", False))
    score, _ = score_report(task_id, report, verified_findings, validation_attempted)
    return score


def grade_secret_exposure_easy(*args: Any, **kwargs: Any) -> float:
    return grade_task("secret_exposure_easy", *args, **kwargs)


def grade_missing_security_headers_medium(*args: Any, **kwargs: Any) -> float:
    return grade_task("missing_security_headers_medium", *args, **kwargs)


def grade_authz_boundary_hard(*args: Any, **kwargs: Any) -> float:
    return grade_task("authz_boundary_hard", *args, **kwargs)
