from CyberSecurity_OWASP.evals import bad_policy, random_policy
from CyberSecurity_OWASP.models import CyberSecurityOWASPAction

from .helpers import apply_secure_patch, make_env, submit_valid_finding


def test_random_policy_does_not_crash():
    env = make_env(60)
    for action in random_policy():
        obs = env.step(action)
        assert obs is not None


def test_bad_policy_is_penalized_or_flagged():
    env = make_env(61)
    for action in bad_policy():
        obs = env.step(action)
    assert env.state.anti_cheat_flags
    assert obs.reward <= 0


def test_scripted_oracle_solves_episode():
    env = make_env(62)
    submit_valid_finding(env)
    apply_secure_patch(env)
    env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
    final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))
    assert final.done is True
    assert env.state.success is True
