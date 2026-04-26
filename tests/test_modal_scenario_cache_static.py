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


def test_modal_sft_defaults_match_300_episode_fast_handoff_plan():
    source = (ROOT / "scripts" / "modal_train_sft.py").read_text(encoding="utf-8")

    assert 'SFT_GPU_FALLBACK = ["H200", "H100", "A100-80GB", "L40S"]' in source
    assert "gpu=SFT_GPU_FALLBACK" in source
    assert "DEFAULT_TOTAL_TRAIN_EPISODES = 300" in source
    assert "DEFAULT_EPISODES_PER_LEVEL = 75" in source
    assert 'DEFAULT_CURRICULUM_LEVELS = "0,1,2,3"' in source
    assert (
        'DEFAULT_SFT_OUTPUT_REPO_ID = (\n'
        '    "Humanlearning/CyberSecurity_OWASP-unsloth-gemma-4-e2b-it-sft-lora"'
    ) in source
    assert "output_repo_id = output_repo_id or DEFAULT_SFT_OUTPUT_REPO_ID" in source
    assert source.count("max_steps: int = -1") >= 2
    assert source.count("per_device_train_batch_size: int = 4") >= 2
    assert source.count("gradient_accumulation_steps: int = 4") >= 2
    assert '"assistant_only_loss": False' in source
    assert '"packing": False' in source
    assert '"packing_strategy": "bfd"' not in source
    assert '"dataset_num_proc": None' in source
    assert "Dataset.from_list(tokenized_rows)" in source
    assert "tokenizer.apply_chat_template" in source
    assert "class CyberSecurityOWASPSFTTrainer(SFTTrainer)" in source
    assert "Trainer.compute_loss(self, model, inputs" in source
    assert '"bf16": True' in source
    assert '"tf32": True' in source
    assert '"hub_strategy": "every_save"' in source
    assert 'trackio_space_id: str = DEFAULT_TRACKIO_SPACE_ID' in source
    assert 'trackio_project: str = DEFAULT_TRACKIO_PROJECT' in source
    assert 'os.environ["TRACKIO_SPACE_ID"] = trackio_space_id' in source
    assert 'os.environ["TRACKIO_PROJECT"] = trackio_project' in source


def test_modal_grpo_loads_sft_adapter_from_hub_as_trainable_lora():
    source = (ROOT / "scripts" / "modal_train_grpo.py").read_text(encoding="utf-8")

    assert "initial_adapter_repo_id" in source
    assert "Downloading initial SFT adapter" in source
    assert "snapshot_download(" in source
    assert "Attaching Unsloth LoRA before loading SFT weights" in source
    assert "load_safetensors_file(str(adapter_weights_path), device=\"cpu\")" in source
    assert "set_peft_model_state_dict(" in source
    assert "unexpected_adapter_keys" in source
