from CyberSecurity_OWASP.models import CyberSecurityOWASPAction

from .helpers import apply_secure_patch, make_env, secure_invoice_source, submit_valid_finding


def test_oracle_patch_gets_high_reward():
    env = make_env(40)
    finding = submit_valid_finding(env)
    assert finding.reward_breakdown["discovery"] == 3.0
    apply_secure_patch(env)
    visible = env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
    assert visible.reward_breakdown["visible_tests"] == 1.0
    final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))
    assert env.state.success is True
    assert final.reward_breakdown["total"] >= 12.0


def test_finding_without_patch_does_not_succeed():
    env = make_env(41)
    submit_valid_finding(env)
    final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))
    assert env.state.success is False
    assert final.reward_breakdown["security"] < 0


def test_deny_all_patch_fails_regression():
    env = make_env(42)
    submit_valid_finding(env)
    source = secure_invoice_source(env).replace(
        "    if invoice[\"tenant_id\"] != actor[\"tenant_id\"]:\n",
        "    return {\"status\": 403, \"body\": {\"detail\": \"forbidden\"}}\n"
        "    if invoice[\"tenant_id\"] != actor[\"tenant_id\"]:\n",
    )
    env.step(
        CyberSecurityOWASPAction(
            tool_name="patch_file",
            arguments={"path": "app/routes/invoices.py", "content": source},
        )
    )
    final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))
    assert final.reward_breakdown["regression"] < 0
    assert env.state.success is False


def test_hardcoded_patch_is_penalized():
    env = make_env(43)
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
    assert final.reward_breakdown["anti_cheat"] < 0
    assert env.state.success is False


def test_visible_tests_only_does_not_get_high_reward():
    env = make_env(44)
    submit_valid_finding(env)
    visible = env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
    assert visible.reward_breakdown["visible_tests"] == 1.0
    final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))
    assert final.reward_breakdown["total"] < 5.0
