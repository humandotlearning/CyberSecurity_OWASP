import json

from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
from training.trackio_utils import (
    CANONICAL_TRACKIO_SIGNALS,
    DERIVED_TRACKIO_METRICS,
    aggregate_episode_metrics,
    episode_record_from_state,
    episode_to_trace_row,
    episode_to_tracking_fields,
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
