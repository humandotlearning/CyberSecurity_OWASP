from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_modal_train_uses_persistent_required_scenario_cache():
    source = (ROOT / "scripts" / "modal_train_grpo.py").read_text(encoding="utf-8")

    assert "SCENARIO_CACHE_VOLUME_NAME = \"CyberSecurity_OWASP-scenario-cache\"" in source
    assert "SCENARIO_CACHE_DIR = pathlib.Path(\"/scenario-cache\")" in source
    assert "CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE" in source
    assert "\"require\" if required else \"fallback\"" in source
    assert "mode == \"prepare-cache\"" in source
    assert "def verify_modal_scenario_cache_for_training" in source
    assert "CPU scenario cache preflight passed" in source
    assert "scenario_cache.assert_coverage" in source
    assert "volumes={RUNS_DIR: volume, CACHE_DIR: cache_volume, SCENARIO_CACHE_DIR: scenario_cache_volume}" in source


def test_modal_ephemeral_smoke_uses_required_scenario_cache():
    source = (ROOT / "scripts" / "modal_ephemeral_train.py").read_text(encoding="utf-8")

    assert "SCENARIO_CACHE_VOLUME_NAME = \"CyberSecurity_OWASP-scenario-cache\"" in source
    assert "SCENARIO_CACHE_DIR = Path(\"/scenario-cache\")" in source
    assert "mode == \"prepare-cache\"" in source
    assert "_configure_scenario_cache_env(required=True)" in source
    assert "ScenarioCache(SCENARIO_CACHE_DIR" in source


def test_modal_training_is_pinned_to_gemma4_e2b():
    source = (ROOT / "scripts" / "modal_train_grpo.py").read_text(encoding="utf-8")

    assert "DEFAULT_GEMMA_MODEL = \"unsloth/gemma-4-E2B-it\"" in source
    assert "def _ensure_gemma4_model(model_name: str) -> str:" in source
    assert "model_name = _ensure_gemma4_model(model_name)" in source
    assert "from unsloth import FastVisionModel" in source
    assert "Qwen" not in source
    assert "FastLanguageModel" not in source
