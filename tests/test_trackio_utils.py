import json
import sys
import types

from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
from CyberSecurity_OWASP.reward_config import load_reward_settings
from training.trackio_utils import (
    CANONICAL_TRACKIO_SIGNALS,
    DERIVED_TRACKIO_METRICS,
    REWARD_CONFIG_TABLE_COLUMNS,
    aggregate_episode_metrics,
    episode_record_from_state,
    episode_trace_fingerprint,
    episode_to_trace_row,
    episode_to_tracking_fields,
    log_reward_config,
    reward_config_scalar_metrics,
)

from .helpers import apply_secure_patch, make_env, secure_invoice_source, submit_valid_finding


def test_canonical_tracking_fields_exist_and_are_numeric_where_expected():
    assert len(CANONICAL_TRACKIO_SIGNALS) >= 57

    env = make_env(70)
    try:
        submit_valid_finding(env)
        apply_secure_patch(env)
        env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
        final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))

        record = episode_record_from_state(env.state, final_observation=final.model_dump())
        fields = episode_to_tracking_fields(record)

        for key in (*CANONICAL_TRACKIO_SIGNALS, *DERIVED_TRACKIO_METRICS):
            assert key in fields

        assert isinstance(fields["reward/hidden_authz_pass_rate"], float)
        assert isinstance(fields["reward/normal_flow_pass_rate"], float)
        assert isinstance(fields["reward/public_hidden_gap"], float)
        assert isinstance(fields["skill/exploit_to_patch_alignment"], float)

        metrics = aggregate_episode_metrics([record])
        assert metrics["reward/hidden_authz_pass_rate"] == 1.0
        assert metrics["skill/patch_success"] == 1.0
    finally:
        env.close()


def test_tracking_cheat_score_flags_hardcoded_identifier():
    env = make_env(71)
    try:
        submit_valid_finding(env)
        hidden = env.state.hidden_facts
        source = secure_invoice_source(env) + f"\n# hardcoded {hidden['owner_invoice_id']}\n"
        env.step(
            CyberSecurityOWASPAction(
                tool_name="patch_file",
                arguments={"path": "app/routes/invoices.py", "content": source},
            )
        )
        final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))

        record = episode_record_from_state(env.state, final_observation=final.model_dump())
        fields = episode_to_tracking_fields(record)

        assert fields["cheat/hardcoded_scenario_id"] == 1.0
        assert fields["cheat/score"] >= 50.0
    finally:
        env.close()


def test_trace_rows_redact_hidden_values_from_action_arguments():
    env = make_env(72)
    try:
        hidden = dict(env.state.hidden_facts)
        submit_valid_finding(env)
        apply_secure_patch(env)
        env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
        final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))

        record = episode_record_from_state(env.state, final_observation=final.model_dump())
        row = episode_to_trace_row(record)
        row_text = json.dumps(row, sort_keys=True)

        for key in (
            "owner_user_id",
            "intruder_user_id",
            "admin_user_id",
            "owner_invoice_id",
            "other_invoice_id",
            "foreign_invoice_id",
            "tenant_a",
            "tenant_b",
        ):
            value = str(hidden.get(key, ""))
            assert not value or value not in row_text
    finally:
        env.close()


def test_trace_fingerprint_ignores_episode_id_but_tracks_action_changes():
    base_record = {
        "episode_id": "episode-a",
        "task_id": "task-1",
        "scenario/seed": 123,
        "scenario/split": "train",
        "scenario/difficulty": 0,
        "scenario/bug_type": "bola_idor",
        "scenario/template_id": "fastapi_basic",
        "scenario_hash": "scenario-a",
        "action_history": [
            {
                "tool_name": "read_file",
                "arguments": {"path": "app/routes/invoices.py"},
            }
        ],
        "observation_history": [{"last_action_valid": True}],
        "reward_breakdown": {"total": 0.0},
    }
    same_trace = dict(base_record)
    same_trace["episode_id"] = "episode-b"
    token_only_reward_change = dict(base_record)
    token_only_reward_change["reward_total"] = -0.25
    changed_trace = dict(base_record)
    changed_trace["action_history"] = [
        *base_record["action_history"],
        {"tool_name": "submit_fix", "arguments": {}},
    ]
    different_scenario = dict(base_record)
    different_scenario["scenario_hash"] = "scenario-b"

    assert episode_trace_fingerprint(base_record) == episode_trace_fingerprint(same_trace)
    assert episode_trace_fingerprint(base_record) == episode_trace_fingerprint(token_only_reward_change)
    assert episode_trace_fingerprint(base_record) != episode_trace_fingerprint(changed_trace)
    assert episode_trace_fingerprint(base_record) != episode_trace_fingerprint(different_scenario)


def test_log_reward_config_emits_scalar_values_and_table(monkeypatch):
    logged: list[tuple[dict, int | None]] = []

    class FakeTable:
        def __init__(self, *, columns, data=None, rows=None, allow_mixed_types=False):
            self.columns = columns
            self.rows = data if data is not None else rows
            self.data = self.rows
            self.allow_mixed_types = allow_mixed_types

    fake_trackio = types.SimpleNamespace(config={}, Table=FakeTable)

    def fake_log(payload, step=None):
        logged.append((payload, step))

    fake_trackio.log = fake_log
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_STAGE", "early")

    settings = load_reward_settings()
    summary = log_reward_config(settings, step=0)

    assert fake_trackio.config["reward_config_hash"] == summary["reward_config_hash"]
    assert fake_trackio.config["reward_config_values"]["policy_inspected"]["value"] == 0.30
    assert fake_trackio.config["reward_config__policy_inspected__value"] == 0.30
    scalar_payload = next(payload for payload, _step in logged if "reward_config/policy_inspected/value" in payload)
    assert scalar_payload["reward_config/policy_inspected/value"] == 0.30
    assert scalar_payload["reward_config/shaping_weight/resolved"] == 1.0
    assert scalar_payload["reward_config/invalid_action/value"] == -0.20
    assert scalar_payload["reward_config/progressive_cap/value"] == 5.0
    assert scalar_payload["reward_config/oversized_patch/severe_value"] == -1.0

    table = next(payload["reward_config"] for payload, _step in logged if "reward_config" in payload)
    assert table.columns == list(REWARD_CONFIG_TABLE_COLUMNS)
    assert table.allow_mixed_types is True
    rows = {row[0]: row for row in table.rows}
    assert rows["policy_inspected"][1] == 0.30
    assert rows["shaping_weight"][2] == 1.0
    assert rows["hidden_file_probe"][6] is True

    logged_text = json.dumps(
        {
            "summary": summary,
            "scalar_payload": scalar_payload,
            "table_rows": table.rows,
        },
        sort_keys=True,
        default=str,
    )
    assert "owner_invoice_id" not in logged_text
    assert "foreign_invoice_id" not in logged_text


def test_reward_config_scalar_metrics_uses_stage_resolved_values(monkeypatch):
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_STAGE", "late")

    metrics = reward_config_scalar_metrics(load_reward_settings())

    assert metrics["reward_config/shaping_weight/resolved"] == 0.4
    assert metrics["reward_config/shaping_weight/stage_value"] == 0.4
    assert metrics["reward_config/step_penalty/stage_value"] == -0.02
    assert metrics["reward_config/token_penalty/target_tokens"] == 350.0
