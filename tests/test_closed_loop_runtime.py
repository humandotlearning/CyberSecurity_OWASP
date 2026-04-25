import json
from pathlib import Path

from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
from CyberSecurity_OWASP.server.adversarial_designer import BoundedAdversarialDesigner
from CyberSecurity_OWASP.server.authz_oracle import AuthzOracle
from CyberSecurity_OWASP.server.curriculum import CurriculumController
from CyberSecurity_OWASP.server.verifier import MultiLayerVerifier

from .helpers import apply_secure_patch, make_env, submit_valid_finding


def test_curriculum_selects_profile_and_tracks_mastery():
    controller = CurriculumController()
    profile = controller.select_profile(seed=3, split="train", requested_difficulty=1)

    assert profile["difficulty_tier"] == "beginner"
    assert profile["target_weakness"]
    assert "target_mastery" in profile["mastery"]

    env = make_env(70)
    controller.record_episode(env.state)
    snapshot = controller.mastery_snapshot()
    assert snapshot["episodes_seen"] == 1


def test_adversarial_designer_marks_hidden_eval_as_heldout_family():
    designer = BoundedAdversarialDesigner()
    spec = designer.design(
        seed=4,
        split="hidden_eval",
        curriculum_profile={"target_weakness": "cross_tenant_boundary"},
    )

    assert spec["safe_lab_only"] is True
    assert spec["scenario_family"].startswith("heldout.")
    assert spec["target_weakness"] == "cross_tenant_boundary"


def test_reset_records_scenario_family_and_partial_observability():
    env = make_env(71)
    obs = env.reset(seed=71, split="hidden_eval", difficulty=1)
    serialized_hint = json.dumps(obs.visible_policy_hint).lower()

    assert env.state.scenario_family.startswith("heldout.")
    assert env.state.difficulty_tier in {"advanced", "expert"}
    assert "oracle_matrix" not in serialized_hint
    assert "hidden_tests" not in serialized_hint
    assert "injected bug" not in serialized_hint


def test_authz_oracle_fails_vulnerable_app_and_passes_secure_patch():
    env = make_env(72)
    oracle = AuthzOracle()

    vulnerable = oracle.evaluate(env.state)
    assert vulnerable["passed"] is False

    submit_valid_finding(env)
    apply_secure_patch(env)
    fixed = oracle.evaluate(env.state)
    assert fixed["passed"] is True


def test_multilayer_verifier_aggregates_terminal_layers():
    env = make_env(73)
    submit_valid_finding(env)
    apply_secure_patch(env)

    verifier = MultiLayerVerifier().run_terminal_checks(env.state)
    assert verifier["visible"]["passed"] is True
    assert verifier["hidden_tests"]["passed"] is True
    assert verifier["oracle_matrix"]["passed"] is True
    assert verifier["regression"]["passed"] is True
    assert verifier["public_routes"]["passed"] is True
    assert verifier["patch_quality"]["passed"] is True


def test_solved_episode_writes_jsonl_artifact_with_verifier_fields():
    env = make_env(74)
    submit_valid_finding(env)
    apply_secure_patch(env)
    env.step(CyberSecurityOWASPAction(tool_name="run_visible_tests"))
    final = env.step(CyberSecurityOWASPAction(tool_name="submit_fix"))

    artifact_path = Path(env.state.episode_artifact_path or "")
    assert final.done is True
    assert artifact_path.exists()
    record = json.loads(artifact_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["episode_id"] == env.state.episode_id
    assert record["final_status"] == "resolved"
    assert record["hidden_test_result"]["passed"] is True
    assert record["oracle_result"]["passed"] is True
    assert record["reward_breakdown"]["total"] >= 12.0
