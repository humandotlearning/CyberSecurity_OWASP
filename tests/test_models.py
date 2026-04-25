from CyberSecurity_OWASP import (
    CyberSecurityOWASPAction,
    CyberSecurityOWASPObservation,
    CyberSecurityOWASPState,
)


def test_models_serialize():
    action = CyberSecurityOWASPAction(tool_name="noop")
    assert action.model_dump()["tool_name"] == "noop"
    obs = CyberSecurityOWASPObservation(phase="discover", message="ok")
    assert obs.model_dump()["phase"] == "discover"
    state = CyberSecurityOWASPState(episode_id="e1", seed=1)
    assert state.model_dump()["seed"] == 1
