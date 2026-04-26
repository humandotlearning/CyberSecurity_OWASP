from Cyber_analyst.models import CyberAnalystAction
from Cyber_analyst.server.Cyber_analyst_environment import CyberAnalystEnvironment
from Cyber_analyst.server.graders import (
    grade_authz_boundary_hard,
    grade_missing_security_headers_medium,
    grade_secret_exposure_easy,
    safe_reward,
)


def _run_success_path(task_id, actions):
    env = CyberAnalystEnvironment()
    obs = env.reset(task_id=task_id, seed=7)
    assert obs.task_id == task_id

    for action in actions:
        obs = env.step(action)

    assert obs.done is True
    assert obs.tool_result["score"] > 0.5
    assert 0.01 <= obs.tool_result["score"] <= 0.99
    assert obs.error == ""
    return obs


def test_secret_exposure_success_path():
    report = {
        "findings": [
            {
                "finding_type": "secret_exposure",
                "evidence_ids": ["EVID-101"],
                "impact": "A synthetic API key secret is exposed in config.",
                "remediation": "Remove the key and rotate the credential.",
            }
        ]
    }
    obs = _run_success_path(
        "secret_exposure_easy",
        [
            CyberAnalystAction(tool_name="search_repo", args={"query": "api key"}),
            CyberAnalystAction(
                tool_name="create_finding",
                args={
                    "finding_type": "secret_exposure",
                    "evidence_ids": ["EVID-101"],
                    "severity_guess": "high",
                    "remediation": "Remove and rotate the synthetic credential.",
                },
            ),
            CyberAnalystAction(tool_name="validate_finding", args={"finding_id": "FND-001"}),
            CyberAnalystAction(tool_name="submit_report", args={"report_json": report}),
        ],
    )
    assert obs.verified_findings[0]["matching_gt_id"] == "GT-SECRET-001"
    assert "trajectory_jsonl" in obs.tool_result
    assert "search_repo" in obs.tool_result["trajectory_jsonl"]


def test_missing_security_headers_success_path():
    report = {
        "findings": [
            {
                "finding_type": "missing_security_headers",
                "evidence_ids": ["EVID-201"],
                "impact": "The gateway is missing HSTS and CSP headers.",
                "remediation": "Add HSTS and CSP at the gateway.",
            }
        ]
    }
    obs = _run_success_path(
        "missing_security_headers_medium",
        [
            CyberAnalystAction(
                tool_name="check_security_headers", args={"service_id": "gateway"}
            ),
            CyberAnalystAction(
                tool_name="create_finding",
                args={
                    "finding_type": "missing_security_headers",
                    "evidence_ids": ["EVID-201"],
                    "severity_guess": "medium",
                    "remediation": "Add HSTS and CSP headers.",
                },
            ),
            CyberAnalystAction(tool_name="validate_finding", args={"finding_id": "FND-001"}),
            CyberAnalystAction(tool_name="submit_report", args={"report_json": report}),
        ],
    )
    assert obs.score_breakdown["valid_evidence"] == 0.15


def test_authz_boundary_success_path_with_alias_compatible_service_ids():
    report = {
        "findings": [
            {
                "finding_type": "authz_boundary_misconfiguration",
                "evidence_ids": ["EVID-301", "EVID-302"],
                "impact": "The admin route authorization policy allows an analyst role.",
                "remediation": "Apply least privilege in the policy and add a regression test.",
            }
        ]
    }
    obs = _run_success_path(
        "authz_boundary_hard",
        [
            CyberAnalystAction(tool_name="list_assets", args={}),
            CyberAnalystAction(
                tool_name="get_log_events",
                args={"service_id": "admin-service", "query": "admin export"},
            ),
            CyberAnalystAction(tool_name="search_repo", args={"query": "admin export"}),
            CyberAnalystAction(
                tool_name="create_finding",
                args={
                    "finding_type": "authz_boundary_misconfiguration",
                    "evidence_ids": ["EVID-301", "EVID-302"],
                    "severity_guess": "critical",
                    "remediation": "Apply least privilege and add a regression test.",
                },
            ),
            CyberAnalystAction(tool_name="validate_finding", args={"finding_id": "FND-001"}),
            CyberAnalystAction(tool_name="submit_report", args={"report_json": report}),
        ],
    )
    assert obs.score_breakdown["actionable_remediation"] == 0.15


def test_invalid_tool_returns_observation_error():
    env = CyberAnalystEnvironment()
    env.reset(task_id="secret_exposure_easy", seed=1)
    obs = env.step(CyberAnalystAction(tool_name="shell", args={"cmd": "whoami"}))
    assert obs.done is False
    assert obs.error == "unsupported_tool"
    assert obs.tool_result["ok"] is False


def test_hallucinated_report_scores_low_but_in_range():
    env = CyberAnalystEnvironment()
    env.reset(task_id="secret_exposure_easy", seed=1)
    obs = env.step(
        CyberAnalystAction(
            tool_name="submit_report",
            args={
                "report_json": {
                    "findings": [
                        {
                            "finding_type": "remote_code_execution",
                            "evidence_ids": [],
                            "impact": "Unsupported claim.",
                            "remediation": "Unsupported remediation.",
                        }
                    ]
                }
            },
        )
    )
    assert obs.done is True
    assert obs.tool_result["score"] == 0.01


def test_repeated_action_hard_stops_episode():
    env = CyberAnalystEnvironment()
    env.reset(task_id="secret_exposure_easy", seed=1)
    obs = None
    for _ in range(6):
        obs = env.step(CyberAnalystAction(tool_name="list_assets", args={}))
    assert obs is not None
    assert obs.done is True
    assert obs.error == "repeat_hard_stop"


def test_seed_determinism_for_assets():
    env_one = CyberAnalystEnvironment()
    env_two = CyberAnalystEnvironment()
    env_one.reset(task_id="authz_boundary_hard", seed=22)
    env_two.reset(task_id="authz_boundary_hard", seed=22)
    obs_one = env_one.step(CyberAnalystAction(tool_name="list_assets", args={}))
    obs_two = env_two.step(CyberAnalystAction(tool_name="list_assets", args={}))
    assert obs_one.tool_result == obs_two.tool_result


def test_grader_adapters_and_clamp_are_strictly_in_range():
    assert safe_reward(-1) == 0.01
    assert safe_reward(2) == 0.99
    assert 0.01 <= grade_secret_exposure_easy() <= 0.99
    assert 0.01 <= grade_missing_security_headers_medium() <= 0.99
    assert 0.01 <= grade_authz_boundary_hard() <= 0.99
