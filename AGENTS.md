# AGENTS.md — CyberSecurity_OWASP Builder Instructions

## Purpose

This repository implements **CyberSecurity_OWASP**, an OpenEnv-compliant RL environment for training a **single LLM agent** to perform a defensive application-security workflow:

```text
inspect generated app + policy -> discover authorization bug -> submit safe finding -> patch code -> preserve intended behavior
```

The environment must train the model to do real interactive work, not answer static security questions. The model must act step by step through typed OpenEnv actions, observe consequences, receive deterministic reward, and improve through RL.

The canonical repository and OpenEnv environment name is **`CyberSecurity_OWASP`**. Use this exact name in `openenv.yaml`, `pyproject.toml`, HF Spaces repo naming, Docker image tags, Trackio run names, command examples, and documentation.

The target stack is:

```text
CyberSecurity_OWASP OpenEnv environment
  -> deterministic verifier + hidden tests
  -> rollout loop
  -> HF TRL / Unsloth GRPO
  -> Trackio logging
  -> held-out evaluation
  -> HF Spaces deployment
```

The final project must show measurable improvement in reward, exploit-block rate, regression-preservation rate, and held-out generalization after training.

---

## Product definition

CyberSecurity_OWASP generates a new local application scenario every `reset(seed)`. Each episode contains:

- a policy graph describing users, roles, tenants, resources, ownership, permissions, and public routes;
- a generated FastAPI-style application workspace;
- exactly one injected OWASP A01-style authorization defect;
- visible tests for normal behavior;
- hidden invariant tests for authorization correctness, regression protection, public-route preservation, and anti-cheat checks.

The environment has **one LLM agent**, not separate red-team and blue-team LLMs. The environment itself acts as the scenario generator, tool server, verifier, and judge.

---

## Highest-priority objectives

When making implementation decisions, optimize in this order:

1. **Verifier correctness**: deterministic tests must decide whether the patch actually fixes the authorization defect.
2. **Reward integrity**: reward must be hard to hack and must punish insecure or regressive patches.
3. **Anti-overfitting**: the model must generalize across apps, layouts, policies, domains, names, and bug families.
4. **OpenEnv compliance**: expose typed `Action`, `Observation`, and `State`; implement `reset()`, `step(action)`, and `state` correctly.
5. **Trainability**: baseline model should sometimes get partial reward; curriculum should make early learning possible.
6. **Real-world usefulness**: the workflow should resemble secure code review / AppSec authorization repair.
7. **Demo clarity**: show before/after rollouts, reward curves, and why the trained model improved.
8. **Hackathon competitiveness**: prioritize a novel, interactive, professionally useful environment with a coherent training pipeline.

Do not train before the environment, verifier, anti-cheat tests, and before/after evaluation are stable.

---

## Hackathon alignment requirements

The implementation must satisfy these minimum requirements:

- use the latest OpenEnv release;
- include a minimal HF TRL or Unsloth training script;
- use Trackio as the default tracker for training and evaluation;
- be deployable as an OpenEnv-compliant Hugging Face Space;
- include a README / mini-blog style explanation;
- show baseline-vs-trained improvement.

Optimize for judging:

| Criterion | Weight | CyberSecurity_OWASP evidence |
|---|---:|---|
| Environment innovation | 40% | procedural OWASP authorization-repair environment with generated code, policy, and hidden verifier |
| Storytelling | 30% | single LLM learns discover + patch, before/after security behavior |
| Showing improvement in rewards | 20% | reward curves, exploit-block pass rate, regression-preservation rate |
| Reward/training pipeline | 10% | deterministic reward, GRPO/PPO rollout loop, Trackio metrics |

---

## Non-negotiable environment design

CyberSecurity_OWASP must be a **single-agent** environment:

```text
phase = discover -> patch -> done
```

Do not implement a two-LLM red-team/blue-team setup. The single model must learn both discovery and repair.

The environment must be defensive and local only. It must never target real systems or teach unauthorized exploitation. All probing must be limited to the generated local workspace controlled by the environment.

---

## Required repository structure

Prefer this structure:

```text
.
├── AGENTS.md
├── README.md
├── 00_PROJECT_BRIEF.md
├── 01_ARCHITECTURE.md
├── pyproject.toml
├── openenv.yaml
├── envs/
│   └── CyberSecurity_OWASP/
│       ├── __init__.py
│       ├── models.py
│       ├── client.py
│       ├── README.md
│       ├── rewards.py
│       ├── validators.py
│       ├── safety.py
│       ├── evals.py
│       ├── server/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── environment.py
│       │   ├── scenario_compiler.py
│       │   ├── policy_graph.py
│       │   ├── template_renderer.py
│       │   ├── bug_mutator.py
│       │   ├── fixture_generator.py
│       │   ├── reward_engine.py
│       │   ├── requirements.txt
│       │   └── Dockerfile
│       ├── templates/
│       │   └── fastapi_basic/
│       ├── scenario_cache/
│       │   ├── train/
│       │   ├── validation/
│       │   └── hidden_eval/
│       └── tests/
│           ├── test_models.py
│           ├── test_reset_step_state.py
│           ├── test_rewards.py
│           ├── test_anti_cheat.py
│           ├── test_seed_reproducibility.py
│           ├── test_invalid_actions.py
│           └── test_rollouts.py
├── training/
│   ├── train_grpo.py
│   ├── rollout.py
│   ├── reward_funcs.py
│   ├── eval_before_after.py
│   ├── trackio_utils.py
│   └── configs/
│       └── grpo_small.yaml
├── scripts/
│   ├── run_local.sh
│   ├── docker_build.sh
│   ├── docker_run.sh
│   ├── smoke_test.sh
│   ├── generate_scenarios.sh
│   └── push_space.sh
├── assets/
│   └── anti_overfitting_training_flow_diagram.png
└── outputs/
    ├── logs/
    ├── evals/
    └── rollouts/
```

If `openenv init CyberSecurity_OWASP` creates a different structure, preserve the generated structure and add the missing files around it.

---

## Architecture overview

CyberSecurity_OWASP has 7 main components:

1. **Policy Graph + Domain Sampler** — samples users, roles, tenants, ownership, public routes, and business exceptions.
2. **Template / Framework Randomizer** — renders FastAPI-style apps with randomized layouts and naming.
3. **A01 Bug Mutator** — injects one authorization defect per scenario.
4. **Fixture + Hidden Test Generator** — creates users, resources, visible tests, and hidden invariant tests.
5. **OpenEnv Server** — exposes typed `Action`, `Observation`, and `State` through `reset`, `step`, and `state`.
6. **LLM Agent + LoRA** — one model performs discover + patch.
7. **Deterministic Reward Engine** — hidden tests score exploit blocking, normal-flow preservation, patch quality, and anti-cheat.

An optional LLM reviewer may score rationale quality and ASVS/OWASP mapping only. It must not provide the primary reward.

---

## Scenario compiler requirements

### Policy Graph + Domain Sampler

The policy graph is the source of truth. It must define:

- users;
- tenants;
- roles;
- resources;
- ownership relationships;
- role permissions;
- public routes;
- business exceptions.

Initial domains:

| Domain | Example resources | Example policy rule |
|---|---|---|
| invoices | invoices, payments, accounts | owner or billing admin can read invoice |
| support | tickets, comments, customer records | assigned agent can update ticket |
| projects | projects, documents, milestones | project member can read project docs |
| marketplace | orders, returns, seller records | buyer owns own orders; seller owns own listings |
| HR | employee profiles, reviews, payroll records | HR admin can read employee records |

### Template / Framework Randomizer

First version: FastAPI only. Still randomize structure so the model cannot memorize one app.

Randomize:

- path naming;
- parameter names;
- helper names;
- folder layout;
- route/service/auth split;
- fixture names;
- error messages within valid policy bounds.

Examples:

```text
/routes/invoices.py
/api/billing.py
/controllers/accounts.py
/services/access.py
/authz/guards.py
```

### A01 Bug Mutator

Inject exactly one primary bug per scenario.

Initial bug families:

| Bug family | Defect | Desired repair pattern |
|---|---|---|
| BOLA/IDOR | resource ID lookup lacks owner/tenant check | check server-side owner/tenant relation |
| BFLA | privileged route lacks role/function check | add reusable role or permission guard |
| tenant leak | request/header tenant ID is trusted | derive tenant from authenticated principal or server-side mapping |
| JWT claim trust | mutable claim is treated as authoritative | verify against server-side user/role record |
| public-route trap | route is intentionally public | do not over-secure public allowlisted route |

### Fixture + Hidden Test Generator

Visible tests should check that the app boots and normal happy paths work.

Hidden tests must check:

- exploit request is blocked;
- legitimate owner flow still works;
- legitimate admin/support flow still works;
- public routes remain public;
- cross-tenant access is denied;
- randomized IDs/names defeat hardcoded patches;
- hidden tests, fixtures, oracle, and reward files are not modified.

### Scenario Cache + Seeded Reset

Training should use cached scenarios for speed.

Recommended first cache:

| Split | Seeds | Purpose |
|---|---:|---|
| train | 500–1,000 | RL rollouts |
| validation | 100–200 | checkpoint selection and curriculum signal |
| hidden_eval | 100–200 | final generalization proof |

### Hold-Out Generalization Splitter

Hold out at least 4 dimensions:

1. domains;
2. policy graph shapes;
3. code layouts;
4. bug-family/domain combinations.

Example: train on invoices/support/projects, evaluate on marketplace/HR.

---

## OpenEnv model definitions

Implement these in `envs/CyberSecurity_OWASP/models.py`.

```python
from dataclasses import dataclass, field
from typing import Any, Literal
from openenv.core.env_server import Action, Observation, State

CyberSecurityOWASPPhase = Literal["discover", "patch", "done"]
CyberSecurityOWASPSplit = Literal["train", "validation", "hidden_eval"]

@dataclass
class CyberSecurityOWASPAction(Action):
    tool_name: Literal[
        "inspect_policy_graph",
        "list_routes",
        "read_openapi",
        "read_file",
        "search_code",
        "send_local_request",
        "compare_identities",
        "submit_diagnosis",
        "patch_file",
        "run_visible_tests",
        "submit_fix",
        "noop",
    ]
    arguments: dict[str, Any] = field(default_factory=dict)

@dataclass
class CyberSecurityOWASPObservation(Observation):
    phase: CyberSecurityOWASPPhase
    message: str
    task_brief: str
    visible_policy_hint: dict[str, Any] = field(default_factory=dict)
    workspace_summary: dict[str, Any] = field(default_factory=dict)
    available_actions: list[str] = field(default_factory=list)
    last_tool_result: str = ""
    last_action_valid: bool = True
    last_action_error: str | None = None
    visible_test_result: str | None = None
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    done_reason: str | None = None

@dataclass
class CyberSecurityOWASPState(State):
    episode_id: str = ""
    task_id: str = ""
    seed: int = 0
    split: CyberSecurityOWASPSplit = "train"
    difficulty: int = 0
    domain: str = ""
    bug_family: str = ""
    phase: CyberSecurityOWASPPhase = "discover"
    step_count: int = 0
    max_steps: int = 40
    done: bool = False
    success: bool = False
    failure_reason: str | None = None
    finding_submitted: bool = False
    patch_submitted: bool = False
    accumulated_reward: float = 0.0
    last_reward: float = 0.0
    action_history: list[dict[str, Any]] = field(default_factory=list)
    reward_history: list[dict[str, float]] = field(default_factory=list)
    visible_facts: dict[str, Any] = field(default_factory=dict)
    hidden_facts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    anti_cheat_flags: list[str] = field(default_factory=list)
```

---

## Action design and phase gating

Actions must be explicit, typed, serializable, and constrained. Invalid actions must not crash the server.

### Phase-gated tools

| Phase | Allowed tools |
|---|---|
| discover | `inspect_policy_graph`, `list_routes`, `read_openapi`, `read_file`, `search_code`, `send_local_request`, `compare_identities`, `submit_diagnosis`, `noop` |
| patch | `read_file`, `search_code`, `patch_file`, `run_visible_tests`, `send_local_request`, `submit_fix`, `noop` |
| done | no state-changing tools; return stable done observation |

### Tool contracts

`inspect_policy_graph`
: Returns public policy hints. Must not reveal hidden bug labels or hidden tests.

`list_routes`
: Returns route method/path summaries from the generated app.

`read_openapi`
: Returns generated OpenAPI metadata.

`read_file`
: Reads editable workspace files only. Must block hidden tests, reward files, oracle files, and host files.

`search_code`
: Searches editable workspace files only.

`send_local_request`
: Sends a request to the local generated app only. Must block external URLs and host network access.

`compare_identities`
: Runs the same local request as two generated users and summarizes behavioral differences.

`submit_diagnosis`
: Accepts structured evidence of the suspected authorization bug. Required before patch phase unless curriculum level explicitly allows blind patching.

`patch_file`
: Applies a bounded unified diff to editable app files only.

`run_visible_tests`
: Runs visible tests only. Must not run or reveal hidden tests.

`submit_fix`
: Triggers hidden evaluation.

---

## Observation rules

Observations should provide enough information to act but must not leak the answer.

Include:

- current phase;
- task brief;
- visible policy hints;
- workspace summary;
- available tools;
- previous tool output;
- visible test output;
- public reward breakdown after terminal evaluation.

Do not include:

- hidden bug family if not meant to be visible;
- hidden test contents;
- hidden oracle;
- exact exploit path labels;
- hidden seed split labels that allow memorization;
- reward implementation details that allow proxy hacking.

---

## State rules

State is the source of truth for deterministic replay and debugging.

Required state properties:

- `reset(seed)` must create a fresh independent state;
- same seed + same action sequence should produce same result;
- each WebSocket session must be isolated;
- `step_count` increments once per processed action;
- terminal states return stable done observations;
- hidden facts never appear in observations;
- all actions and reward breakdowns are stored for debugging.

---

## Environment API contract

Implement in `envs/CyberSecurity_OWASP/server/environment.py`.

```python
from openenv.core.env_server import Environment
from ..models import CyberSecurityOWASPAction, CyberSecurityOWASPObservation, CyberSecurityOWASPState

class CyberSecurityOWASPEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = CyberSecurityOWASPState()

    def reset(self) -> CyberSecurityOWASPObservation:
        ...

    def step(self, action: CyberSecurityOWASPAction) -> CyberSecurityOWASPObservation:
        ...

    @property
    def state(self) -> CyberSecurityOWASPState:
        return self._state
```

`step(action)` must follow this order:

1. If done, return stable done observation.
2. Validate action and phase permissions.
3. Increment step count.
4. Execute the tool.
5. Update state/history.
6. Run verifier if `submit_diagnosis`, `run_visible_tests`, or `submit_fix`.
7. Compute reward components.
8. Check terminal conditions.
9. Return observation, reward, and done through OpenEnv step result handling.

---

## FastAPI server and client

`envs/CyberSecurity_OWASP/server/app.py` must use the OpenEnv FastAPI helper.

```python
from openenv.core.env_server import create_fastapi_app
from ..models import CyberSecurityOWASPAction, CyberSecurityOWASPObservation
from .environment import CyberSecurityOWASPEnvironment

env = CyberSecurityOWASPEnvironment()
app = create_fastapi_app(env, CyberSecurityOWASPAction, CyberSecurityOWASPObservation)
```

`envs/CyberSecurity_OWASP/client.py` must parse `StepResult`, observations, rewards, done flags, and state into typed classes.

---

## Deterministic reward engine

The reward engine is the task specification. Build it before training.

Maximum reward: **15.0**.

| Component | Reward |
|---|---:|
| valid finding with reproducible local evidence | +2.0 |
| finding tied to violated policy rule | +1.0 |
| hidden exploit blocked | +5.0 |
| legitimate owner/admin/support flows preserved | +3.0 |
| public routes preserved | +1.0 |
| patch is localized, reusable, and policy-aligned | +2.0 |
| visible tests pass and app boots | +1.0 |
| exploit still works | -6.0 |
| legitimate flow regression | -5.0 |
| public route incorrectly locked | -3.0 |
| hardcoded user/tenant/resource fix | -4.0 |
| modified tests, fixtures, oracle, or hidden files | -8.0 |
| app does not parse or boot | -6.0 |

The reward breakdown must use stable keys:

```python
reward_breakdown = {
    "discovery": discovery_reward,
    "security": security_reward,
    "regression": regression_reward,
    "public_routes": public_route_reward,
    "patch_quality": patch_quality_reward,
    "visible_tests": visible_tests_reward,
    "safety": safety_reward,
    "anti_cheat": anti_cheat_reward,
    "total": total_reward,
}
```

Do not let explanation quality dominate the reward. If using an LLM reviewer, keep it separate from `total` or cap it at a small non-primary auxiliary signal.

---

## Verifier requirements

Create `envs/CyberSecurity_OWASP/validators.py` and `envs/CyberSecurity_OWASP/server/reward_engine.py`.

Required verifier functions:

```python
def verify_finding(state: CyberSecurityOWASPState, finding: dict) -> dict:
    ...


def run_hidden_security_tests(state: CyberSecurityOWASPState) -> dict:
    ...


def run_hidden_regression_tests(state: CyberSecurityOWASPState) -> dict:
    ...


def detect_cheating(state: CyberSecurityOWASPState, action: CyberSecurityOWASPAction) -> list[str]:
    ...


def compute_reward(state: CyberSecurityOWASPState, action: CyberSecurityOWASPAction, verifier_result: dict) -> dict[str, float]:
    ...
```

Verifier tests must prove:

- correct patch receives high reward;
- exploit-only finding without patch does not complete the episode;
- deny-all patch fails regression tests;
- hardcoded patch fails randomized hidden tests;
- modified hidden files produce anti-cheat penalty;
- visible-test-only patch does not guarantee high reward;
- repeated intermediate actions cannot inflate reward indefinitely.

---

## Anti-overfitting requirements

CyberSecurity_OWASP must prevent overfitting to one app or scenario.

Use all of these defenses:

| Risk | Required defense |
|---|---|
| memorizes one app | many domains and templates |
| memorizes route names | randomized path, resource, parameter, helper names |
| memorizes bug location | vary route/service/auth layer placement |
| learns deny-all patch | hidden positive-flow and public-route tests |
| learns hardcoded patch | randomized users, tenants, resource IDs, role names |
| overfits visible tests | hidden invariant tests and held-out eval |
| overfits one bug family | curriculum-sampled bug mix |
| overfits one code layout | hold out entire layouts and domains |
| optimizes explanation only | deterministic reward is primary |

Acceptance target: at least **20%** of domain/layout/bug combinations must be held out from training.

---

## Safety and cybersecurity boundaries

This is a defensive AppSec training environment.

Allowed:

- local generated app probing;
- authorization reasoning;
- secure patching;
- visible and hidden test execution;
- policy-to-code mapping;
- defensive vulnerability validation in sandbox.

Forbidden:

- real-world exploitation;
- credential theft;
- persistence/evasion/malware behavior;
- scanning external targets;
- bypassing real services;
- writing exploit instructions for systems outside the local generated lab.

`send_local_request` must only target the generated local app.

---

## Curriculum controller

RL needs partial successes. Implement at least 3 difficulty levels.

```text
level_0: BOLA/IDOR, small app, direct route, obvious policy hint
level_1: BFLA or tenant bug, moderate app, realistic distractors
level_2: JWT trust or nested tenant/resource route, multiple files, false-positive traps
level_3: held-out domain/layout/bug combo, harder naming, fewer hints
```

Curriculum signal:

```text
if exploit_block_rate < 60%:
    increase level_0 and level_1 tasks
elif regression_rate > 20%:
    increase positive-flow and public-route traps
elif public_route_false_positive_rate > 10%:
    increase intentionally public route examples
elif validation_reward plateaus:
    increase unseen layouts and nested resources
else:
    increase difficulty by 1
```

---

## Training requirements

Create a runnable minimal training script using HF TRL or Unsloth.

Required files:

```text
training/train_grpo.py
training/rollout.py
training/reward_funcs.py
training/eval_before_after.py
training/trackio_utils.py
training/configs/grpo_small.yaml
```

Recommended first model:

```text
Qwen/Qwen3-1.7B
```

Acceptable alternatives:

```text
Qwen2.5-Coder-1.5B-Instruct
Qwen2.5-Coder-3B-Instruct
```

Use LoRA / QLoRA. Do not full-finetune unless explicitly required.

---

## Rollout function requirements

`training/rollout.py` must run a full OpenEnv episode.

```python
def rollout_once(trainer, env, tokenizer, dataset_prompt: str, max_steps: int = 40) -> dict:
    result = env.reset()
    observation = result.observation

    prompt_ids = []
    completion_ids = []
    logprobs = []
    reward_trace = []
    action_trace = []
    observation_trace = []

    for _ in range(max_steps):
        if result.done:
            break

        prompt = build_cybersecurity_owasp_prompt(observation, action_trace, observation_trace)
        rollout_output = generate_rollout_completions(trainer, [prompt])[0]
        action = parse_action_json(rollout_output["text"])

        result = env.step(action)
        observation = result.observation

        prompt_ids.extend(rollout_output["prompt_ids"])
        completion_ids.extend(rollout_output["completion_ids"])
        logprobs.extend(rollout_output["logprobs"])
        reward_trace.append(float(result.reward or 0.0))
        action_trace.append(action)
        observation_trace.append(observation)

    final_breakdown = getattr(observation, "reward_breakdown", {}) or {}
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
        "success": bool(getattr(env.state(), "success", False)),
        "episode_length": len(action_trace),
    }
```

The prompt must require the model to output exactly one JSON action at a time.

Example action format:

```json
{"tool_name":"read_file","arguments":{"path":"app/routes/invoices.py"}}
```

---

## Reward functions for TRL

`training/reward_funcs.py` must expose separate reward functions for GRPO/PPO logging.

```python
def reward_total(completions, **kwargs):
    return [float(x) for x in kwargs.get("reward_total", [0.0] * len(completions))]


def reward_security(completions, **kwargs):
    return [float(x) for x in kwargs.get("reward_security", [0.0] * len(completions))]


def reward_regression(completions, **kwargs):
    return [float(x) for x in kwargs.get("reward_regression", [0.0] * len(completions))]


def reward_patch_quality(completions, **kwargs):
    return [float(x) for x in kwargs.get("reward_patch_quality", [0.0] * len(completions))]


def reward_anti_cheat(completions, **kwargs):
    return [float(x) for x in kwargs.get("reward_anti_cheat", [0.0] * len(completions))]
```

---

## GRPO training config

Use Trackio in `GRPOConfig`.

```python
import os
from trl import GRPOConfig

output_dir = os.getenv("OUTPUT_DIR", "CyberSecurity_OWASP-qwen3-1.7b-grpo")
trackio_space_id = os.getenv("TRACKIO_SPACE_ID", "Humanlearning/CyberSecurity_OWASP-trackio")

grpo_config = GRPOConfig(
    output_dir=output_dir,
    report_to="trackio",
    trackio_space_id=trackio_space_id,
    logging_steps=1,
    save_steps=25,
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_generations=6,
    max_prompt_length=4096,
    max_completion_length=768,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    push_to_hub=False,
)
```

Start with small debug runs before scaling.

---

## Trackio logging requirements

Trackio is mandatory for training and evaluation visibility.

Canonical Trackio Space:

```text
https://huggingface.co/spaces/Humanlearning/CyberSecurity_OWASP-trackio
```

Use `TRACKIO_SPACE_ID=Humanlearning/CyberSecurity_OWASP-trackio` for training,
evaluation, and smoke runs. This is separate from the OpenEnv HF Space
`Humanlearning/CyberSecurity_OWASP`; do not send Trackio runs to the
environment Space.

Run naming convention:

```text
CyberSecurity_OWASP-<model>-<algo>-level<difficulty>-<YYYYMMDD-HHMM>-<git_sha>
```

Log these training metrics:

```text
train/reward_total_mean
train/reward_discovery_mean
train/reward_security_mean
train/reward_regression_mean
train/reward_public_routes_mean
train/reward_patch_quality_mean
train/reward_visible_tests_mean
train/reward_safety_mean
train/reward_anti_cheat_mean
train/success_rate
train/exploit_block_rate
train/regression_preservation_rate
train/public_route_preservation_rate
train/invalid_action_rate
train/timeout_rate
train/safety_violation_rate
train/reward_hacking_suspected_rate
train/episode_length_mean
train/episode_length_p95
train/rollouts_per_second
train/tokens_per_second
train/loss
train/learning_rate
train/kl
train/grad_norm
```

Log these evaluation metrics:

```text
eval/baseline_success_rate
eval/trained_success_rate
eval/absolute_success_improvement
eval/baseline_mean_reward
eval/trained_mean_reward
eval/absolute_reward_improvement
eval/heldout_success_rate
eval/heldout_mean_reward
eval/exploit_block_rate
eval/regression_preservation_rate
eval/public_route_preservation_rate
eval/anti_cheat_pass_rate
eval/invalid_action_rate
eval/timeout_rate
eval/safety_violation_rate
eval/mean_episode_length
```

Log these environment metrics:

```text
env/reset_latency_ms
env/step_latency_ms
env/verifier_latency_ms
env/reward_latency_ms
env/scenario_compile_latency_ms
env/error_rate
env/task_difficulty
env/task_seed
```

---

## Rollout artifact requirements

Save sampled rollouts under `outputs/rollouts/`.

Each rollout JSON must include:

```json
{
  "run_name": "...",
  "episode_id": "...",
  "task_id": "...",
  "seed": 123,
  "split": "validation",
  "difficulty": 1,
  "domain": "invoices",
  "bug_family": "bola_idor",
  "actions": [],
  "observations": [],
  "reward_breakdown_by_step": [],
  "final_reward_breakdown": {},
  "total_reward": 0.0,
  "success": false,
  "failure_reason": null,
  "safety_violations": [],
  "anti_cheat_flags": []
}
```

Minimum artifacts:

- 10 baseline rollouts;
- 10 mid-training rollouts;
- 10 trained rollouts;
- 10 held-out evaluation rollouts.

---

## Evaluation requirements

Create `training/eval_before_after.py`.

It must evaluate:

| Metric | Required |
|---|---:|
| baseline success rate | yes |
| trained success rate | yes |
| absolute success improvement | yes |
| baseline mean reward | yes |
| trained mean reward | yes |
| absolute reward improvement | yes |
| held-out success rate | yes |
| exploit-block rate | yes |
| regression-preservation rate | yes |
| public-route preservation rate | yes |
| invalid action rate | yes |
| anti-cheat pass rate | yes |

Save output:

```text
outputs/evals/<run_name>_eval_summary.json
```

Minimum hackathon target:

```text
>= 50 evaluation episodes
>= 3 independently logged reward components
>= 1 held-out split
>= 1 baseline-vs-trained comparison
>= 1 anti-cheat evaluation
```

Preferred demo target:

```text
mean reward improvement >= 30%
hidden exploit-block pass rate >= 70%
regression-preservation pass rate >= 80%
public-route preservation pass rate >= 90%
anti-cheat pass rate >= 95%
```

---

## Testing requirements

Before training, all tests must pass.

Required tests:

```text
test_models.py
test_reset_step_state.py
test_rewards.py
test_anti_cheat.py
test_seed_reproducibility.py
test_invalid_actions.py
test_rollouts.py
```

Implement at least 3 scripted policies:

```text
random_policy: explores action space; should usually fail but not crash
bad_policy: tries invalid/cheating actions; should be penalized
oracle_policy: uses internal test-only access to solve; should get high reward
```

The oracle policy is only for tests and must never be exposed to the model during training.

---

## Deployment requirements

The environment must run in these modes:

1. local Python / Uvicorn;
2. Docker container;
3. Hugging Face Space;
4. OpenEnv client over WebSocket.

Required commands:

```bash
# initialize if not already scaffolded
openenv init CyberSecurity_OWASP

# local development
uv sync
uv run server
curl http://localhost:8000/health

# Docker
openenv build -t CyberSecurity_OWASP:latest
# or:
docker build -t CyberSecurity_OWASP:latest -f envs/CyberSecurity_OWASP/server/Dockerfile .
docker run -p 8000:8000 CyberSecurity_OWASP:latest

# HF Spaces
openenv push --repo-id <username>/CyberSecurity_OWASP

# client install from Space
pip install git+https://huggingface.co/spaces/<username>/CyberSecurity_OWASP
```

Use WebSocket mode for training rollouts. HTTP endpoints are acceptable for debugging only.

---

## Scaling rules

Before scaling training, confirm:

1. one manual episode works;
2. scripted oracle can solve easy seeds;
3. random policy does not crash;
4. 10 validation rollouts complete;
5. reward distributions make sense;
6. Trackio receives metrics;
7. rollout artifacts are saved.

Then scale gradually:

```text
1 episode -> 10 episodes -> 50 episodes -> 100+ rollouts -> training run
```

For high-volume rollouts, prefer local Docker or Uvicorn over remote HF Spaces because local WebSocket sessions reduce latency and avoid Space limits.

### Parallel Modal training runs

Parallel Modal GRPO runs are allowed only when they do not overwrite each
other's evidence, checkpoints, scenario assignments, or Hub outputs.

Before launching another run:

1. Check active Modal apps:

```bash
uv run --extra modal modal app list
```

2. If a `CyberSecurity_OWASP` app is active, inspect it before launching:

```bash
uv run --extra modal modal app logs <app-id>
```

3. Use Modal CLI-level detach and the launcher detach flag together, otherwise
the spawned GPU function may stop when the local entrypoint exits:

```bash
uv run --extra modal modal run --detach scripts/modal_train_grpo.py \
  --max-steps 300 \
  --dataset-size 64 \
  --num-generations 8 \
  --max-completion-length 768 \
  --difficulty 0 \
  --trace-log-every 10 \
  --seed-start 10000 \
  --detach
```

When running jobs in parallel:

- Give every run a distinct `--seed-start` range, spaced by at least 10,000
  seeds unless a smaller controlled comparison is intentional.
- Keep `CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE=require`; do not compile
  scenarios in the training hot path.
- Do not run `prepare-cache --cache-force` while any training job is active.
  Scenario-cache writes can invalidate or race training resets.
- Leave `--push-to-hub` off for parallel experiments unless each run has a
  unique `--output-repo-id`.
- Keep run names unique. The launcher timestamp normally handles this; set an
  explicit `RUN_NAME` only when it is globally unique.
- Use different Trackio run names but the same Trackio Space so reward,
  throughput, GPU utilization, invalid-action rate, and success metrics remain
  comparable.
- Treat the shared Modal volumes as shared infrastructure: model cache and
  scenario cache should be read-only during parallel training; run/checkpoint
  outputs must live under each run's unique output directory.
- If the goal is a clean reward comparison, keep model, difficulty,
  `dataset-size`, `num-generations`, `max-completion-length`, and reward config
  fixed, changing only `seed-start` or the one hyperparameter being tested.

---

## README requirements

The README must explain:

- what CyberSecurity_OWASP models;
- why authorization repair is useful for LLM RL;
- action space;
- observation space;
- state fields;
- scenario generation;
- reward components;
- hidden tests;
- anti-overfitting safeguards;
- anti-cheat safeguards;
- curriculum;
- local/Docker/HF Spaces commands;
- training with TRL/Unsloth;
- before/after evaluation.

Include a demo narrative:

```text
1. Baseline model attempts a generated A01 authorization repair episode.
2. Verifier shows whether it discovered the bug and whether the patch regressed normal flows.
3. RL training improves reward and pass rates.
4. Trained model handles held-out domain/layout seeds.
5. Anti-cheat tests prove it is not using deny-all, hardcoding, or fixture tampering.
```

---

## Implementation workflow for Codex

When implementing this repo, follow this exact order:

1. Inspect existing structure and tests.
2. Create/update `00_PROJECT_BRIEF.md` and `01_ARCHITECTURE.md` if missing.
3. Define `CyberSecurityOWASPAction`, `CyberSecurityOWASPObservation`, and `CyberSecurityOWASPState`.
4. Implement a dummy OpenEnv server and client.
5. Implement scenario compiler with 1 domain and 1 BOLA/IDOR mutator.
6. Implement editable workspace generation.
7. Implement local request tool.
8. Implement visible tests.
9. Implement hidden verifier and reward engine.
10. Add anti-cheat checks.
11. Add tests for normal, failing, and cheating rollouts.
12. Add oracle, random, and bad scripted policies.
13. Add scenario cache and seeded splits.
14. Add 3 domains and 3 bug families.
15. Add GRPO training script.
16. Add Trackio logging.
17. Add before/after evaluation script.
18. Add HF Spaces deployment config.
19. Run tests and smoke tests.
20. Produce demo artifacts and README results.

Do not jump to training code before environment and verifier are correct.

---

## Definition of done

CyberSecurity_OWASP is done only when all are true:

- `reset()`, `step(action)`, and `state` work;
- actions, observations, and state are typed dataclasses;
- the environment runs locally;
- the environment runs in Docker;
- the environment is deployable to HF Spaces;
- there are at least 5 meaningful reward components;
- reward components are logged separately;
- hidden tests exist;
- anti-cheat tests exist and pass;
- scenario cache has train/validation/hidden-eval splits;
- at least 3 bug families exist;
- at least 3 domains exist;
- at least 3 scripted policies exist;
- Trackio is configured for training and evaluation;
- before/after evaluation exists;
- held-out evaluation exists;
- at least 40 rollout artifacts are saved;
- README explains environment, reward, training, and demo story;
- demo shows baseline behavior, trained behavior, reward improvement, and safeguards.

---

## Final PR checklist

Every PR summary must answer:

1. What real-world workflow does this implement?
2. What does the agent observe?
3. What actions can the agent take?
4. What hidden state exists and why is it hidden?
5. What terminates an episode?
6. What exact checks prove success?
7. What are the reward components and ranges?
8. How could the model hack the reward?
9. What anti-cheat checks prevent that?
10. What tests prove the reward cannot be trivially hacked?
11. What baseline success rate did we observe?
12. What trained success rate did we observe?
13. What held-out success rate did we observe?
14. What Trackio run contains the evidence?
15. Does behavior improve, or only the reward proxy?
16. Is the environment ready for HF Spaces deployment?

---

## Source grounding and credibility

| Source | Why used | Credibility |
|---|---|---:|
| OWASP Top 10 A01 Broken Access Control | Authorization bug taxonomy and prevention framing | 8.5/10 |
| OWASP ASVS | Access-control verification grounding | 9/10 |
| NIST SP 800-218 SSDF | Secure software development lifecycle grounding | 9.5/10 |
| Smith et al., ESEC/FSE 2015, “Is the Cure Worse Than the Disease?” | Peer-reviewed basis for hidden tests and repair-overfitting risk | 9/10 |
| OpenEnv build/deploy/training docs | Typed model, server, client, deployment, and training mechanics | 8/10 |
| Meta OpenEnv Hackathon criteria | Judging alignment and minimum requirements | 8/10 |

---

## Non-negotiable rule

A reward that can be hacked is worse than no reward. Build the verifier, hidden tests, anti-cheat tests, and held-out evaluation before scaling training.
