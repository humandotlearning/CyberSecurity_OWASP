#!/usr/bin/env python3
"""Model-backed baseline inference for the Cyber Analyst OpenEnv environment."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from Cyber_analyst import CyberAnalystAction, CyberAnalystEnv, CyberAnalystObservation


ENV_NAME = "Cyber_analyst"
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it:fastest")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
SEED = int(os.getenv("SEED", "7"))
TASK_IDS = [
    "secret_exposure_easy",
    "missing_security_headers_medium",
    "authz_boundary_hard",
]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are running a bounded Cyber Analyst benchmark. You may only choose one
    tool call from the provided tool catalog per turn. All evidence is synthetic;
    do not request shell access, live network access, or external investigation.

    Return exactly one compact JSON object and no surrounding text:
    {"tool_name":"<tool name>","args":{...}}

    To complete an episode, first discover relevant evidence, then create and
    validate a finding, then submit a report_json with findings that include
    finding_type, evidence_ids, impact, and remediation.
    """
).strip()


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model_name: str
    temperature: float
    max_tokens: int


class ModelActionError(RuntimeError):
    """Raised when the model cannot provide a valid benchmark action."""


def build_llm_config() -> LLMConfig:
    return LLMConfig(
        base_url=API_BASE_URL,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


def build_openai_client() -> OpenAI:
    """Return an OpenAI-compatible client for the Hugging Face Router."""

    return OpenAI(base_url=API_BASE_URL, api_key=os.environ["HF_TOKEN"])


def single_line(value: str) -> str:
    return " ".join(str(value).split())


def action_to_log(action: CyberAnalystAction) -> str:
    payload = {"tool_name": action.tool_name, "args": action.args}
    return single_line(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def log_start(task_id: str, llm_config: LLMConfig) -> None:
    print(
        f"[START] task={task_id} env={ENV_NAME} model={llm_config.model_name}",
        flush=True,
    )


def log_step(
    step: int, action: CyberAnalystAction, reward: float | None, done: bool, error: str
) -> None:
    reward_value = 0.0 if reward is None else float(reward)
    error_value = single_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={action_to_log(action)} "
        f"reward={reward_value:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(task_id: str, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] task={task_id} success={str(success).lower()} "
        f"steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def observation_payload(obs: CyberAnalystObservation) -> dict[str, Any]:
    return {
        "task_id": obs.task_id,
        "alert": obs.alert,
        "phase": obs.phase,
        "tool_catalog": obs.tool_catalog,
        "tool_result": obs.tool_result,
        "evidence_ids": obs.evidence_ids,
        "candidate_findings": obs.candidate_findings,
        "verified_findings": obs.verified_findings,
        "step_budget_remaining": obs.step_budget_remaining,
        "score_breakdown": obs.score_breakdown,
        "error": obs.error,
    }


def build_user_prompt(task_id: str, step: int, obs: CyberAnalystObservation) -> str:
    payload = {
        "task_id": task_id,
        "step": step,
        "observation": observation_payload(obs),
    }
    return textwrap.dedent(
        f"""
        Choose the next benchmark tool call.
        Current state JSON:
        {json.dumps(payload, sort_keys=True)}
        """
    ).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ModelActionError(f"model_parse_error: {exc.msg}") from exc

    if not isinstance(decoded, dict):
        raise ModelActionError("model_parse_error: response is not a JSON object")
    return decoded


def parse_model_action(text: str) -> CyberAnalystAction:
    payload = extract_json_object(text)
    tool_name = payload.get("tool_name")
    args = payload.get("args", {})

    if not isinstance(tool_name, str) or not tool_name:
        raise ModelActionError("model_parse_error: missing tool_name")
    if not isinstance(args, dict):
        raise ModelActionError("model_parse_error: args must be an object")

    return CyberAnalystAction(tool_name=tool_name, args=args)


def get_model_action(
    client: OpenAI,
    llm_config: LLMConfig,
    task_id: str,
    step: int,
    obs: CyberAnalystObservation,
) -> CyberAnalystAction:
    try:
        completion = client.chat.completions.create(
            model=llm_config.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task_id, step, obs)},
            ],
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            stream=False,
        )
    except Exception as exc:
        raise ModelActionError(f"model_request_error: {exc}") from exc

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise ModelActionError("model_parse_error: empty response")
    return parse_model_action(text)


def error_action(error: Exception) -> CyberAnalystAction:
    message = single_line(str(error))
    if message.startswith("model_request_error"):
        tool_name = "model_request_error"
    elif message.startswith("model_parse_error"):
        tool_name = "model_parse_error"
    else:
        tool_name = "model_action_error"
    return CyberAnalystAction(
        tool_name=tool_name,
        args={"message": message[:500]},
    )


def run_task(task_id: str, client: OpenAI, llm_config: LLMConfig) -> None:
    log_start(task_id, llm_config)
    rewards: list[float] = []
    steps_taken = 0
    final_score = 0.01
    success = False

    try:
        with CyberAnalystEnv(base_url=ENV_URL).sync() as env:
            reset_result = env.reset(task_id=task_id, seed=SEED)
            obs = reset_result.observation

            for step in range(1, MAX_STEPS + 1):
                if obs.done:
                    break

                model_failed = False
                try:
                    action = get_model_action(client, llm_config, task_id, step, obs)
                except ModelActionError as exc:
                    action = error_action(exc)
                    model_failed = True

                result = env.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                steps_taken = step

                log_step(step, action, result.reward, result.done, obs.error)

                if isinstance(obs.tool_result, dict) and "score" in obs.tool_result:
                    final_score = float(obs.tool_result["score"])

                if result.done or model_failed:
                    success = final_score > 0.5
                    break

    except Exception as exc:
        action = CyberAnalystAction(
            tool_name="runtime_error",
            args={"message": single_line(str(exc))[:500]},
        )
        steps_taken = max(steps_taken, 1)
        rewards.append(0.01)
        log_step(steps_taken, action, 0.01, True, single_line(str(exc)))

    log_end(task_id, success, steps_taken, final_score, rewards)


def selected_task_ids() -> list[str]:
    task_override = os.getenv("TASK_NAME")
    return [task_override] if task_override else TASK_IDS


def main() -> None:
    llm_config = build_llm_config()
    try:
        client = build_openai_client()
    except KeyError:
        print("HF_TOKEN must be set for inference.", file=sys.stderr, flush=True)
        raise SystemExit(2) from None

    for task_id in selected_task_ids():
        run_task(task_id, client, llm_config)


if __name__ == "__main__":
    main()
