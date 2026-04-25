import json

import pytest

from CyberSecurity_OWASP.config import load_scenario_authoring_config


def test_default_scenario_authoring_config_uses_deepseek_defaults(monkeypatch):
    for key in list(
        name for name in __import__("os").environ if name.startswith("CYBERSECURITY_OWASP_")
    ):
        monkeypatch.delenv(key, raising=False)

    settings = load_scenario_authoring_config()

    assert settings.scenario_author.model_id == "deepseek-ai/DeepSeek-V4-Pro"
    assert settings.scenario_author.provider == "huggingface"
    assert settings.scenario_author.thinking_mode == "thinking"
    assert settings.scenario_author.reasoning_effort == "high"
    assert settings.scenario_author.temperature == 1.0
    assert settings.scenario_author.top_p == 1.0
    assert settings.curriculum.difficulty_bucket_count == 4
    assert settings.curriculum.train_scenarios_per_bucket == 25
    assert settings.curriculum.heldout_eval_scenarios_per_bucket == 10
    assert settings.curriculum.target_cache_hit_rate == 0.95
    assert settings.curriculum.target_reset_latency_ms == 200
    assert settings.curriculum.scenario_refresh_rate_per_epoch == 0.05
    assert settings.curriculum.difficulty_calibration_strategy == "baseline_agent_pass_rate"


def test_scenario_authoring_config_env_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "scenario_author": {},
                "curriculum": {"difficulty_labels": ["D0", "D1"]},
                "runtime": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_CONFIG", str(config_path))
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_AUTHOR_MODEL", "test/model")
    monkeypatch.setenv("CYBERSECURITY_OWASP_DIFFICULTY_BUCKETS", "2")
    monkeypatch.setenv("CYBERSECURITY_OWASP_TRAIN_SCENARIOS_PER_BUCKET", "3")
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE", "require")

    settings = load_scenario_authoring_config()

    assert settings.scenario_author.model_id == "test/model"
    assert settings.curriculum.difficulty_bucket_count == 2
    assert settings.curriculum.train_scenarios_per_bucket == 3
    assert settings.runtime.cache_mode == "require"


def test_scenario_authoring_config_rejects_bad_values(monkeypatch, tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(
        json.dumps(
            {
                "scenario_author": {"temperature": 0},
                "curriculum": {},
                "runtime": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CYBERSECURITY_OWASP_SCENARIO_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="sampling"):
        load_scenario_authoring_config()
