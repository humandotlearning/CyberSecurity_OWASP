import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

from CyberSecurity_OWASP.models import CyberSecurityOWASPAction
from CyberSecurity_OWASP.server.CyberSecurity_OWASP_environment import (
    CybersecurityOwaspEnvironment,
)


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_sft_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_sft_dataset", MODULE_PATH)
generate_sft_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generate_sft_dataset
SPEC.loader.exec_module(generate_sft_dataset)


def _isolated_out_dir(label: str) -> Path:
    root = Path("outputs") / "sft_dataset_tests" / f"{label}_{uuid.uuid4().hex[:8]}"
    workspace_root = root / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    os.environ["CYBERSECURITY_OWASP_WORKSPACE_ROOT"] = str(workspace_root)
    return root / "sft"


def test_extracts_and_validates_action_json():
    action = generate_sft_dataset.parse_action_text(
        '```json\n{"tool_name":"inspect_policy_graph","arguments":{}}\n```'
    )

    assert isinstance(action, CyberSecurityOWASPAction)
    assert action.tool_name == "inspect_policy_graph"


def test_prompt_uses_visible_observation_only():
    _isolated_out_dir("prompt")
    env = CybersecurityOwaspEnvironment()
    try:
        obs = env.reset(seed=501, split="train", difficulty=0)
        prompt = generate_sft_dataset.build_user_prompt(obs, [])
    finally:
        env.close()

    lowered = prompt.lower()
    assert "hidden_facts" not in lowered
    assert "oracle_hidden_focus" not in lowered
    assert "reward_engine" not in lowered
    assert "validators.py" not in lowered
    assert "tests/hidden" not in lowered
    assert "hidden tests" not in lowered


def test_chat_row_matches_conversational_sft_shape():
    _isolated_out_dir("chat_row")
    env = CybersecurityOwaspEnvironment()
    try:
        obs = env.reset(seed=502, split="train", difficulty=0)
        messages = generate_sft_dataset.build_chat_messages(obs, [])
        action = CyberSecurityOWASPAction(tool_name="inspect_policy_graph", arguments={})
        row = generate_sft_dataset.make_chat_row(
            messages=messages,
            action=action,
            metadata={
                "target_model": generate_sft_dataset.DEFAULT_TARGET_MODEL,
                "teacher_model": generate_sft_dataset.DEFAULT_TEACHER_MODEL,
                "seed": 502,
            },
        )
    finally:
        env.close()

    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert json.loads(row["messages"][-1]["content"]) == action.model_dump()
    assert row["metadata"]["target_model"] == "unsloth/gemma-4-E2B-it"


def test_dry_run_oracle_creates_chat_jsonl_without_network():
    out_dir = _isolated_out_dir("dry_run")
    manifest = generate_sft_dataset.generate_dataset(
        generate_sft_dataset.DatasetConfig(
            episodes=2,
            validation_episodes=1,
            out_dir=out_dir,
            dry_run_oracle=True,
        )
    )

    assert manifest["episodes_attempted"] == 3
    assert manifest["episodes_accepted"] == 3
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "validation.jsonl").exists()
    train_rows = [
        json.loads(line)
        for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_rows = [
        json.loads(line)
        for line in (out_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert train_rows
    assert validation_rows
    assert all(row["messages"][-1]["role"] == "assistant" for row in train_rows)


def test_saved_oracle_trajectory_replays_to_success():
    out_dir = _isolated_out_dir("replay")
    generate_sft_dataset.generate_dataset(
        generate_sft_dataset.DatasetConfig(
            episodes=1,
            out_dir=out_dir,
            dry_run_oracle=True,
        )
    )
    trajectory_path = next((out_dir / "trajectories").glob("train_seed*.json"))
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))

    env = CybersecurityOwaspEnvironment()
    try:
        env.reset(
            seed=int(trajectory["seed"]),
            split=trajectory["split"],
            difficulty=int(trajectory["difficulty"]),
        )
        final = None
        for action_data in trajectory["actions"]:
            final = env.step(CyberSecurityOWASPAction(**action_data))
        assert final is not None
        assert final.done is True
        assert env.state.success is True
        assert not env.state.anti_cheat_flags
    finally:
        env.close()
