from CyberSecurity_OWASP.models import CyberSecurityOWASPAction

from .helpers import make_env, submit_valid_finding


def test_wrong_phase_action_is_structured_error():
    env = make_env(30)
    obs = env.step(
        CyberSecurityOWASPAction(
            tool_name="patch_file",
            arguments={"path": "app/routes/invoices.py", "content": ""},
        )
    )
    assert obs.last_action_valid is False
    assert "not allowed" in (obs.last_action_error or "")


def test_blocked_hidden_path_does_not_crash():
    env = make_env(31)
    obs = env.step(CyberSecurityOWASPAction(tool_name="read_file", arguments={"path": "../hidden.py"}))
    assert obs.last_action_valid is False
    assert "blocked" in (obs.last_action_error or "")
    assert "blocked_file_probe" in env.state.anti_cheat_flags


def test_external_request_is_blocked():
    env = make_env(32)
    obs = env.step(
        CyberSecurityOWASPAction(
            tool_name="send_local_request",
            arguments={"method": "GET", "path": "https://example.com"},
        )
    )
    assert obs.last_action_valid is False
    assert "external_network_attempt" in env.state.anti_cheat_flags


def test_visible_tests_are_not_patchable():
    env = make_env(33)
    submit_valid_finding(env)
    obs = env.step(
        CyberSecurityOWASPAction(
            tool_name="patch_file",
            arguments={"path": "tests/test_visible.py", "content": ""},
        )
    )
    assert obs.last_action_valid is False
    assert "not patchable" in (obs.last_action_error or "")
