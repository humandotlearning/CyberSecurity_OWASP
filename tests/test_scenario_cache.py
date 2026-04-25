import json
import shutil
from pathlib import Path

import pytest

from CyberSecurity_OWASP.config import load_scenario_authoring_config
from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
    CybersecurityOwaspEnvironment,
)
from CyberSecurity_OWASP.server.curriculum import CurriculumController
from CyberSecurity_OWASP.server.scenario_cache import (
    SCENARIO_CACHE_REQUIRED_FILES,
    ScenarioCache,
    ScenarioCacheMiss,
    cache_key_for_scenario,
    prepare_scenario_cache,
    validate_bundle,
)
from CyberSecurity_OWASP.server.scenario_factory import ScenarioFactory


def _small_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CYBERSECURITY_OWASP_DIFFICULTY_BUCKETS", "1")
    monkeypatch.setenv("CYBERSECURITY_OWASP_TRAIN_SCENARIOS_PER_BUCKET", "1")
    monkeypatch.setenv("CYBERSECURITY_OWASP_VALIDATION_SCENARIOS_PER_BUCKET", "1")
    monkeypatch.setenv("CYBERSECURITY_OWASP_HELDOUT_SCENARIOS_PER_BUCKET", "1")
    settings = load_scenario_authoring_config()
    result = prepare_scenario_cache(cache_dir=tmp_path, settings=settings, force=True)
    return settings, result


def test_scenario_cache_bundle_contract_and_key_hash(monkeypatch, tmp_path):
    settings, result = _small_cache(monkeypatch, tmp_path)
    assert result["created"] >= 1

    cache = ScenarioCache(tmp_path, settings=settings)
    bundle_path = cache.find_bundle(seed=0, split="train", difficulty=0)
    assert bundle_path is not None
    validate_bundle(bundle_path)

    for name in SCENARIO_CACHE_REQUIRED_FILES:
        assert (bundle_path / name).exists()

    scenario = json.loads((bundle_path / "scenario.json").read_text(encoding="utf-8"))
    key = scenario["cache_key"]
    assert set(key) == {
        "difficulty_level",
        "authz_bug_type",
        "app_family",
        "framework",
        "policy_shape",
        "tenant_model",
        "exploit_depth",
        "patch_scope",
        "regression_risk",
        "generator_version",
        "verifier_version",
        "scenario_hash",
    }
    assert len(key["scenario_hash"]) == 64

    # The helper should produce the same hash for the same stable scenario payload.
    profile = CurriculumController(settings=settings).select_profile(
        seed=0,
        split="train",
        requested_difficulty=0,
    )
    compiled = ScenarioFactory().compile_scenario(
        0,
        split="train",
        difficulty=0,
        curriculum_profile=profile,
    )
    try:
        assert cache_key_for_scenario(compiled, settings=settings).scenario_hash == key["scenario_hash"]
    finally:
        shutil.rmtree(compiled["workspace"], ignore_errors=True)


def test_runtime_reset_uses_required_cache_without_compiling(monkeypatch, tmp_path):
    settings, _ = _small_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE", "require")

    def fail_compile(*args, **kwargs):
        raise AssertionError("reset must not compile scenarios in required cache mode")

    monkeypatch.setattr(ScenarioFactory, "compile_scenario", fail_compile)

    env = CybersecurityOwaspEnvironment()
    obs = env.reset(seed=0, split="train", difficulty=0)

    try:
        assert obs.phase == "discover"
        assert env.state.cache_hit is True
        assert env.state.scenario_hash
        assert env.state.metrics["scenario_cache_hit"] is True
        assert env.state.metrics["scenario_bundle_load_latency_ms"] >= 0.0
        assert env.state.reset_latency_ms >= 0.0
    finally:
        env.close()


def test_required_cache_mode_fails_on_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE", "require")

    env = CybersecurityOwaspEnvironment()
    with pytest.raises(RuntimeError, match="Scenario cache miss"):
        env.reset(seed=999, split="train", difficulty=0)


def test_cached_hidden_files_are_not_editable_or_readable(monkeypatch, tmp_path):
    _small_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE", "require")

    env = CybersecurityOwaspEnvironment()
    env.reset(seed=0, split="train", difficulty=0)
    try:
        editable = set(env.state.hidden_facts["editable_files"])
        assert "hidden_tests.py" not in editable
        assert "oracle_tests.py" not in editable

        obs = env.step(
            CyberSecurityOWASPAction(
                tool_name="read_file",
                arguments={"path": "hidden_tests.py"},
            )
        )
        assert obs.last_action_valid is False
        assert "blocked" in (obs.last_action_error or "")
    finally:
        env.close()


def test_cache_coverage_reports_missing_bucket(monkeypatch, tmp_path):
    settings, _ = _small_cache(monkeypatch, tmp_path)
    cache = ScenarioCache(tmp_path, settings=settings)
    assert cache.assert_coverage(split="train", difficulty=0)["entries"] >= 1

    missing = tmp_path / "manifest.json"
    missing.unlink()
    for metadata_path in tmp_path.glob("**/metadata.json"):
        metadata_path.unlink()
    with pytest.raises(ScenarioCacheMiss):
        cache.assert_coverage(split="train", difficulty=0)
