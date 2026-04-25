"""Minimal rollout loop for CyberSecurity_OWASP episodes."""

from __future__ import annotations

import json
from typing import Any

from CyberSecurity_OWASP import CyberSecurityOWASPAction


def build_cybersecurity_owasp_prompt(observation, action_trace, observation_trace) -> str:
    return (
        "You are a defensive AppSec repair agent. Output exactly one JSON action.\n"
        f"Phase: {observation.phase}\n"
        f"Task: {observation.task_brief}\n"
        f"Available actions: {observation.available_actions}\n"
        f"Last result: {observation.last_tool_result}\n"
        'Example: {"tool_name":"read_file","arguments":{"path":"app/routes/invoices.py"}}'
    )


def parse_action_json(text: str) -> CyberSecurityOWASPAction:
    data = json.loads(text)
    return CyberSecurityOWASPAction(**data)


def generate_rollout_completions(trainer, prompts: list[str]) -> list[dict[str, Any]]:
    if hasattr(trainer, "generate_rollout_completions"):
        return trainer.generate_rollout_completions(prompts)
    return [
        {
            "text": '{"tool_name":"noop","arguments":{}}',
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
        }
        for _ in prompts
    ]


def rollout_once(trainer, env, tokenizer=None, dataset_prompt: str = "", max_steps: int = 40) -> dict:
    result = env.reset()
    observation = result.observation if hasattr(result, "observation") else result

    prompt_ids = []
    completion_ids = []
    logprobs = []
    reward_trace = []
    action_trace = []
    observation_trace = []

    for _ in range(max_steps):
        if getattr(observation, "done", False):
            break
        prompt = build_cybersecurity_owasp_prompt(observation, action_trace, observation_trace)
        rollout_output = generate_rollout_completions(trainer, [prompt])[0]
        action = parse_action_json(rollout_output["text"])
        result = env.step(action)
        observation = result.observation if hasattr(result, "observation") else result

        prompt_ids.extend(rollout_output["prompt_ids"])
        completion_ids.extend(rollout_output["completion_ids"])
        logprobs.extend(rollout_output["logprobs"])
        reward_trace.append(float(getattr(observation, "reward", 0.0) or 0.0))
        action_trace.append(action.model_dump())
        observation_trace.append(observation.model_dump())

    final_breakdown = getattr(observation, "reward_breakdown", {}) or {}
    state = env.state if not callable(getattr(env, "state", None)) else env.state()
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "reward_total": float(final_breakdown.get("total", sum(reward_trace))),
        "reward_discovery": float(final_breakdown.get("discovery", 0.0)),
        "reward_security": float(final_breakdown.get("security", 0.0)),
        "reward_regression": float(final_breakdown.get("regression", 0.0)),
        "reward_patch_quality": float(final_breakdown.get("patch_quality", 0.0)),
        "reward_anti_cheat": float(final_breakdown.get("anti_cheat", 0.0)),
        "success": bool(getattr(state, "success", False)),
        "episode_length": len(action_trace),
        "actions": action_trace,
        "observations": observation_trace,
    }
