from pathlib import Path

import pytest

from CyberSecurity_OWASP.reward_config import (
    compute_token_penalty,
    load_reward_settings,
)


def test_default_reward_config_has_descriptions():
    settings = load_reward_settings()

    assert settings.mode == "sparse_eval"
    assert settings.training_mode == "dense_train"
    assert settings.value("terminal_cap") == 15.0
    for key, value in settings.raw.items():
        if isinstance(value, dict):
            assert value.get("description")


def test_reward_config_env_overrides(monkeypatch):
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_STAGE", "late")
    monkeypatch.setenv("CYBERSECURITY_OWASP_SHAPING_WEIGHT", "0.25")

    settings = load_reward_settings()

    assert settings.mode == "dense_train"
    assert settings.stage == "late"
    assert settings.shaping_weight == 0.25
    assert compute_token_penalty(850, settings) == -0.5


def test_reward_config_rejects_missing_descriptions(monkeypatch):
    config_path = Path("outputs/test_reward_config_bad.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "reward:\n  mode: sparse_eval\n  policy_inspected:\n    value: 0.3\n",
        encoding="utf-8",
    )
    try:
        monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_CONFIG", str(config_path))

        with pytest.raises(ValueError, match="description"):
            load_reward_settings()
    finally:
        config_path.unlink(missing_ok=True)
