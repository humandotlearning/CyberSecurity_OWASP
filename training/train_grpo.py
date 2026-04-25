"""Minimal GRPO training entrypoint scaffold.

This file intentionally does not start training on import. It validates that the
required TRL/Trackio configuration can be constructed when optional training
dependencies are installed.
"""

from __future__ import annotations

import os

from training.trackio_utils import build_run_name, get_git_sha


def build_grpo_config():
    from trl import GRPOConfig

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
    difficulty = int(os.getenv("DIFFICULTY", "0"))
    output_dir = os.getenv("OUTPUT_DIR", "CyberSecurity_OWASP-qwen3-1.7b-grpo")
    trackio_space_id = os.getenv("TRACKIO_SPACE_ID", output_dir)
    os.environ.setdefault("TRACKIO_PROJECT", "CyberSecurity_OWASP-grpo")
    run_name = os.getenv(
        "RUN_NAME",
        build_run_name(model_name, "grpo", difficulty, git_sha=get_git_sha()),
    )
    return GRPOConfig(
        output_dir=output_dir,
        report_to="trackio",
        trackio_space_id=trackio_space_id,
        run_name=run_name,
        logging_steps=1,
        save_steps=25,
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_generations=2,
        max_prompt_length=4096,
        max_completion_length=768,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=False,
    )


def main():
    config = build_grpo_config()
    print(config)


if __name__ == "__main__":
    main()
