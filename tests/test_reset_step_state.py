from CyberSecurity_OWASP.models import CyberSecurityOWASPAction

from .helpers import make_env


def test_reset_initializes_scenario_and_state():
    env = make_env(10)
    state = env.state
    assert state.seed == 10
    assert state.phase == "discover"
    assert state.domain == "invoices"
    assert state.bug_family == "bola_idor"


def test_step_count_and_done_stability():
    env = make_env(11)
    env.step(CyberSecurityOWASPAction(tool_name="noop"))
    assert env.state.step_count == 1
    env.state.done = True
    env.state.phase = "done"
    first = env.step(CyberSecurityOWASPAction(tool_name="noop"))
    second = env.step(CyberSecurityOWASPAction(tool_name="noop"))
    assert first.done is True
    assert second.done is True
    assert env.state.step_count == 1
