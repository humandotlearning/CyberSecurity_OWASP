from pathlib import Path

import pytest

from CyberSecurity_OWASP.reward_config import (
    compute_token_penalty,
    flatten_reward_config,
    load_reward_settings,
    reward_config_hash,
    reward_config_run_config,
    reward_config_summary,
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


def test_reward_config_hash_and_flattened_values_are_deterministic(monkeypatch):
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_MODE", "dense_train")
    monkeypatch.setenv("CYBERSECURITY_OWASP_REWARD_STAGE", "middle")

    settings = load_reward_settings()
    first_hash = reward_config_hash(settings)
    second_hash = reward_config_hash(load_reward_settings())
    summary = reward_config_summary(settings)
    run_config = reward_config_run_config(settings)
    rows = {row["key"]: row for row in flatten_reward_config(settings)}

    assert first_hash == second_hash
    assert len(first_hash) == 64
    assert summary["reward_config_hash"] == first_hash
    assert summary["reward_config_id"].endswith(first_hash[:12])
    assert run_config["reward_config_hash"] == first_hash
    assert run_config["reward_mode"] == "dense_train"
    assert run_config["reward_stage"] == "middle"
    assert run_config["reward_config_values"]["policy_inspected"]["value"] == 0.30
    assert run_config["reward_config_values"]["shaping_weight"]["stage_value"] == 0.7
    assert run_config["reward_config__policy_inspected__value"] == 0.30
    assert run_config["reward_config__shaping_weight__stage_value"] == 0.7
    assert "policy_inspected" in run_config["reward_config_values_json"]
    assert rows["policy_inspected"]["value"] == 0.30
    assert rows["shaping_weight"]["stage_value"] == 0.7
    assert rows["shaping_weight"]["resolved"] == 0.7
    assert rows["step_penalty"]["stage_value"] == -0.01
    assert rows["oversized_patch"]["threshold"] == 80
    assert rows["oversized_patch"]["severe_threshold"] == 180
    assert rows["hidden_file_probe"]["terminate"] is True


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
