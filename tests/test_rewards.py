from CyberSecurity_OWASP.models import CyberSecurityOWASPAction

from .helpers import apply_secure_patch, make_env, secure_invoice_source, submit_valid_finding


def test_oracle_patch_gets_high_reward():
    env = make_env(40)
    finding = submit_valid_finding(env)
    assert finding.reward_breakdown["discovery"] == 1.0
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


def test_sparse_mode_does_not_pay_progressive_reward(monkeypatch):
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "sparse_eval")
    env = make_env(45)
    obs = env.step(CyberSecurityOWASPAction(tool_name="inspect_policy_graph"))
    assert obs.reward_breakdown["progressive"] == 0.0
    assert obs.reward_breakdown["total"] == 0.0


def test_dense_mode_pays_capped_progressive_reward(monkeypatch):
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_STAGE", "early")
    env = make_env(46)
    obs = env.step(CyberSecurityOWASPAction(tool_name="inspect_policy_graph"))
    assert obs.reward_breakdown["progressive"] == 0.30
    assert obs.reward_breakdown["step_penalty"] < 0.0
    assert obs.reward_breakdown["total"] > 0.0


def test_terminal_score_unchanged_by_dense_shaping(monkeypatch):
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "sparse_eval")
    sparse_env = make_env(47)
    submit_valid_finding(sparse_env)
    apply_secure_patch(sparse_env)
    sparse_env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
    sparse_final = sparse_env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))

    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    dense_env = make_env(47)
    dense_env.step(CyberSecurityOWASPAction(tool_name="inspect_policy_graph"))
    submit_valid_finding(dense_env)
    apply_secure_patch(dense_env)
    dense_env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
    dense_final = dense_env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))

    assert dense_final.reward_breakdown["terminal_total"] == sparse_final.reward_breakdown["terminal_total"]
    assert dense_final.reward_breakdown["train_total"] != dense_final.reward_breakdown["terminal_total"]


def test_repeated_futile_actions_are_penalized(monkeypatch):
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    env = make_env(48)

    first = env.step(CyberSecurityOWASPAction(tool_name="inspect_policy_graph"))
    second = env.step(CyberSecurityOWASPAction(tool_name="inspect_policy_graph"))

    assert first.reward_breakdown["progressive"] > 0.0
    assert second.reward_breakdown["progressive"] == 0.0
    assert second.reward_breakdown["behavior_penalty"] <= -0.10
    assert second.reward_breakdown["total"] < 0.0


def test_dense_episode_reward_cap_blocks_repeated_positive_farming(monkeypatch):
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    env = make_env(49)
    env.state.accumulated_reward = 20.99

    capped = env.step(CyberSecurityOWASPAction(tool_name="inspect_policy_graph"))

    assert 0.0 <= capped.reward_breakdown["total"] <= 0.011
    assert env.state.accumulated_reward <= 21.001
