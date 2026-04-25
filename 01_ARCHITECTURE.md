# 01_ARCHITECTURE.md

# CyberSecurity_OWASP — Architecture

## 1. System goal

`CyberSecurity_OWASP` is an OpenEnv environment for training a **single LLM policy** to perform a complete defensive authorization-repair workflow:

```text
Understand policy → discover local evidence → patch code → validate → submit
```

The environment is intentionally not a two-agent red-team/blue-team setup. The agent is one model with one trajectory. It must learn both sides of the defensive workflow: finding the policy violation and fixing it safely.

## 2. Final architecture diagram

Rendered asset:

![CyberSecurity_OWASP architecture](assets/architecture_diagram.svg)

Editable source: `assets/architecture_diagram.mmd`

```mermaid
flowchart TB
    subgraph A[Async Scenario Authoring + Curriculum Factory]
        A1[Config-guided LLM Scenario Author\nDeepSeek-V4-Pro default]
        A2[ScenarioSpec JSON\npolicy, app family, bug target]
        A3[Template + A01 Mutator\nFastAPI code variants]
        A4[Deterministic Compiler\nexecutable bundle]
        A5[Static + Dynamic Verifier\nsolvable, safe, hidden/visible tests]
        A6[Difficulty Calibrator\nbaseline pass-rate buckets]
        A7[Versioned Scenario Cache\nsplit, difficulty, family, hash]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7
    end

    subgraph B[CyberSecurity_OWASP OpenEnv Runtime]
        B1[reset\(seed, difficulty, family_budget\)\ncache lookup only]
        B2[Curriculum Sampler\nvalidated cache slice]
        B3[Episode State Store\nphase, history, cache metadata, patch diff]
        B4[Typed Action Tools\ninspect, request, patch, visible tests]
        B5[Ephemeral App Sandbox\ncloned cached workspace + fixtures]
        B6[Multi-layer Verifier\nvisible, hidden, oracle, regression]
        B7[Deterministic Reward Engine\nstable components + penalties]
        B8[Episode Artifact Logger\nJSONL transcript + verifier + diff]
        B1 --> B2 --> B3 --> B4
        B4 <--> B5
        B5 --> B6 --> B7 --> B3
        B3 --> B8
    end

    subgraph C[Single LLM Agent]
        C1[Observation Parser]
        C2[AuthZ + Code Reasoning]
        C3[Discover → Diagnose → Patch → Test\none JSON action]
        C1 --> C2 --> C3
    end

    subgraph D[Training + Evaluation + Demo]
        D1[Parallel Rollouts\nfast cached reset]
        D2[TRL GRPO + LoRA]
        D3[Trackio Curves\nreward, pass rates, cache metrics]
        D4[Held-out Family Eval\nbase vs trained model]
        D5[Demo Artifacts\nbefore/after traces + JSONL]
        D1 --> D2 --> D3 --> D4 --> D5
    end

    subgraph E[Feedback / Adaptation Loop]
        E1[Episode logs + failures]
        E2[Mastery Model\nweakness and plateau tracking]
        E3[Cache Sampling Weights\nnew generation queue]
        E1 --> E2 --> E3
    end

    A7 --> B1
    C3 -->|typed action| B4
    B4 -->|observation + reward + done| C1
    B7 --> D1
    D2 --> C1
    B8 --> E1
    E3 --> A1
```

## 3. Component responsibilities

### 3.1 Async Scenario Authoring Plane

Scenario generation is offline, asynchronous, validated, and cached. Runtime `reset()` must not call an LLM and must not compile a fresh app during Modal smoke, training, or evaluation runs.

The scenario authoring plane outputs complete executable bundles:

- `scenario.json`;
- `app_source/`;
- `policy_graph.json`;
- `visible_tests.py`;
- `hidden_tests.py`;
- `oracle_tests.py`;
- `expected_exploit_trace.json`;
- `reward_config.json`;
- `metadata.json`.

The default scenario/curriculum author is configured in `configs/scenario_authoring.small.json`:

```yaml
provider: huggingface
model_id: deepseek-ai/DeepSeek-V4-Pro
thinking_mode: thinking
reasoning_effort: high
temperature: 1.0
top_p: 1.0
```

DeepSeek-V4-Pro is only used for offline scenario/curriculum authoring. It is not the RL policy model unless explicitly selected for training.

The compiler remains the main anti-overfitting mechanism. It should vary:

- route names;
- schema names;
- ORM query structure;
- framework template;
- role names;
- tenant IDs;
- object ownership patterns;
- file layout;
- visible test coverage;
- hidden invariant seeds.

The runtime treats curriculum and cache sampling as first-class scenario inputs:

- `CurriculumController` tracks target weakness mastery, recent reward trend, failure counts, and difficulty tier.
- Offline cache prep uses the configured LLM author, deterministic compiler, verifier, and baseline-agent difficulty calibrator.
- `ScenarioCache` stores validated bundles by split, difficulty, family, generator version, verifier version, and scenario hash.
- Hidden-eval episodes hold out scenario families, not only seeds, by marking evaluation-only scenario-family metadata in state rather than observations.

Cache keys include:

```text
difficulty_level
authz_bug_type
app_family
framework
policy_shape
tenant_model
exploit_depth
patch_scope
regression_risk
generator_version
verifier_version
scenario_hash
```

### 3.2 Policy Graph Generator

The policy graph is the ground truth for intended behavior.

Example internal representation:

```yaml
resources:
  invoice:
    owner_field: owner_user_id
    tenant_field: tenant_id
roles:
  user:
    can:
      - read:invoice where owner_user_id == actor.user_id
      - update:invoice where owner_user_id == actor.user_id and status != locked
  support:
    can:
      - read:invoice where tenant_id == actor.tenant_id
  admin:
    can:
      - read:any_invoice where tenant_id == actor.tenant_id
      - update:any_invoice where tenant_id == actor.tenant_id
public_routes:
  - GET /health
  - GET /pricing
forbidden:
  - cross_tenant_read
  - cross_tenant_update
  - user_reads_other_user_invoice
```

The policy graph prevents false rewards for over-securing intentionally public or intentionally allowed routes.

### 3.3 Bug Injector

The bug injector creates controlled, defensive lab scenarios. It should only generate bugs inside local synthetic apps.

MVP bug classes:

| Bug class | Example failure mode | Expected fix type |
|---|---|---|
| Missing route guard | Protected endpoint lacks authorization middleware | Add policy check/middleware |
| IDOR / ownership bug | User can access another user’s object by changing ID | Add owner check in query/policy |
| Tenant leak | Tenant A can list Tenant B records | Add tenant filter |
| Role confusion | Support/editor/admin boundary is wrong | Correct role-to-permission mapping |
| Client-side-only auth | Server trusts UI to hide forbidden action | Enforce server-side authorization |
| Query omission | List/export/search endpoint lacks auth filter | Filter query by actor permissions |
| Over-broad mutation | User can update/delete forbidden object | Add mutation permission check |
| Public route decoy | Agent may wrongly lock down intended public endpoint | Preserve intended public behavior |

### 3.4 OpenEnv Server

The OpenEnv server should implement the standard lifecycle:

- `reset()` — initialize a fresh episode from a cached scenario bundle.
- `step(action)` — execute one typed action and return observation, reward, and done.
- `state()` — expose episode metadata for debugging and evaluation.

Recommended package/class names:

```text
Repo name:      CyberSecurity_OWASP
Python package: cybersecurity_owasp
Client class:   CyberSecurityOWASPEnv
Action class:   CyberSecurityOWASPAction
Observation:    CyberSecurityOWASPObservation
State:          CyberSecurityOWASPState
```

### 3.5 Tool API

The agent should interact through typed actions. Keep the interface small enough for RL but expressive enough for realistic repair.

```python
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
    arguments: dict
```

Recommended actions:

| Action | Purpose | Safety boundary |
|---|---|---|
| `inspect_policy_graph` | Read intended authorization rules. | Only synthetic policy. |
| `list_routes` | See local app route map. | No internet target. |
| `read_file` | Inspect selected source file. | Sandbox allowlist only. |
| `send_local_request` | Validate behavior against local app. | Local generated app only. |
| `submit_diagnosis` | Record bug class, route, policy rule, evidence trace IDs, and fix plan. | Does not reveal hidden tests. |
| `run_visible_tests` | Run visible tests. | No hidden test disclosure. |
| `patch_file` | Modify source through unified diff or full content. | Patch size and file allowlist limits. |
| `submit_fix` | End episode and trigger hidden eval. | Final hidden score only, no leaked test details. |

### 3.6 Observation schema

Observations should be compact and structured.

```python
@dataclass
class CyberSecurityOWASPObservation(Observation):
    phase: Literal["discover", "patch", "done"]
    message: str
    task_brief: str
    visible_policy_hint: dict
    workspace_summary: dict
    available_actions: list[str]
    last_tool_result: str
    visible_test_result: str | None = None
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    done_reason: str | None = None
```

The policy hint is deliberately partial. It may include product rules, fixture aliases, route summaries, and public-route intent, but it must not expose the hidden oracle matrix, hidden test bodies, injected bug labels, or held-out family labels.

### 3.7 State schema

State should support debugging and training analytics.

```python
@dataclass
class CyberSecurityOWASPState(State):
    episode_id: str
    task_id: str
    split: Literal["train", "validation", "hidden_eval"]
    step_count: int = 0
    max_steps: int = 40
    difficulty_tier: str = "warmup"
    scenario_family: str = ""
    template_id: str = "fastapi_basic"
    target_weakness: str = ""
    curriculum_snapshot: dict = field(default_factory=dict)
    verification_summary: dict = field(default_factory=dict)
    patch_diff: str = ""
    episode_artifact_path: str | None = None
    accumulated_reward: float = 0.0
```

## 4. Episode lifecycle

```text
1. reset()
   - curriculum selects difficulty tier and target weakness
   - runtime samples or directly loads a validated cached bundle
   - clone cached `app_source/` into an isolated ephemeral workspace
   - initialize fixture state, cache metadata, and sandbox handles
   - return initial observation

2. agent loop
   - inspect policy/routes/files
   - send local requests only inside sandbox
   - run public tests
   - apply one or more patches
   - rerun public tests

3. submit_fix
   - freeze patch
   - run public tests
   - run hidden authorization invariants
   - run policy-oracle matrix
   - run regression and public-route preservation tests
   - compute deterministic reward
   - return final observation, reward, done=True

4. logging
   - append JSONL artifact with scenario metadata, action trace, observations, patch diff, verifier result, and reward components
   - feed terminal success/failure back into curriculum mastery tracking
   - send metrics to Trackio during training/eval
```

`CYBERSECURITY_OWASP_SCENARIO_CACHE_MODE=require` is mandatory for Modal smoke, training, and evaluation. In that mode a missing cache bundle is a hard failure. Local development may use `fallback`, which compiles deterministically on a miss, but that path is not allowed for meaningful training.

## 5. Reward design

The reward should be deterministic, decomposed, and resistant to reward hacking. The maximum terminal reward remains **15.0** and high reward requires deterministic verifier success, not explanation quality.

Stable reward keys:

```python
{
    "discovery": 0.0,
    "security": 0.0,
    "regression": 0.0,
    "public_routes": 0.0,
    "patch_quality": 0.0,
    "visible_tests": 0.0,
    "safety": 0.0,
    "anti_cheat": 0.0,
    "terminal_total": 0.0,
    "progressive": 0.0,
    "step_penalty": 0.0,
    "speed_bonus": 0.0,
    "token_penalty": 0.0,
    "behavior_penalty": 0.0,
    "train_total": 0.0,
    "total": 0.0,
}
```

Sparse evaluation uses `terminal_total` as `total`. Dense training uses
`terminal_total + shaping_weight * progressive + efficiency - penalties` as `total`,
with all reward values and short descriptions configured in
`training/configs/grpo_small.yaml`.

### Reward components

| Component | Purpose |
|---|---|
| `discovery` | Valid local evidence and correct violated policy rule. |
| `security` | Hidden exploit blocking plus policy-oracle matrix pass. |
| `regression` | Legitimate owner/admin/support flows still work. |
| `public_routes` | Intentionally public routes remain public. |
| `patch_quality` | Localized policy-aligned patch and efficient phase order. |
| `visible_tests` | Visible tests pass and app still boots. |
| `safety` | Penalizes invalid action patterns, unsafe targets, timeouts, and deny-all behavior. |
| `anti_cheat` | Penalizes hidden-file probing, hardcoded fixture IDs, and test/oracle tampering. |

### Penalties

| Penalty | Trigger |
|---|---|
| public route penalty | Breaks a route intentionally marked public. |
| anti-cheat penalty | Deletes or probes tests, hidden files, reward code, oracle data, or host paths. |
| hardcoding penalty | Hardcodes seed-specific IDs, users, tenants, or hidden assumptions. |
| safety penalty | Over-broad denial, malformed/invalid actions, repeated failed actions, or external target attempts. |

The LLM judge, if used at all, should only annotate trace quality for analysis. It must not decide security-critical reward.

## 6. Hidden tests and anti-overfitting

Hidden tests are necessary because visible tests can be gamed or memorized. They should test policy invariants rather than exact implementation details.

Use **4 anti-overfitting layers**:

1. **Seed diversity** — route names, user IDs, tenant IDs, object names, and schemas change every episode.
2. **Template diversity** — same policy bug appears in different frameworks and file layouts.
3. **Hidden invariant tests** — final reward uses unseen authorization cases.
4. **Held-out eval split** — at least 20% of scenario families/seeds are never used in training.

Recommended split:

```text
Train:      70%
Validation: 10%
Held-out:   20%
```

## 7. Evaluation plan

Run before/after evaluation on the same held-out suite.

### Metrics

| Metric | Meaning |
|---|---|
| `episode_success_rate` | Public + hidden + regression tests pass. |
| `hidden_authz_pass_rate` | Security-critical hidden checks pass. |
| `regression_pass_rate` | Normal valid behavior remains intact. |
| `oversecure_rate` | Agent blocks intended legitimate/public behavior. |
| `patch_compile_rate` | Patch applies and app still runs. |
| `median_steps_to_submit` | Efficiency of the repair workflow. |
| `median_files_changed` | Patch focus/minimality. |
| `reward_hacking_rate` | Attempts to delete tests, hardcode fixtures, or bypass eval. |

### Eval table template

| Model | Split | Success | Hidden authz | Regression | Oversecure | Median steps | Median files changed |
|---|---|---:|---:|---:|---:|---:|---:|
| Base model | heldout | TBD | TBD | TBD | TBD | TBD | TBD |
| RL-trained model | heldout | TBD | TBD | TBD | TBD | TBD | TBD |

## 8. Training flow

Rendered asset:

![CyberSecurity_OWASP RL training flow](assets/env_rl_training_flow_diagram.svg)

Editable source: `assets/env_rl_training_flow_diagram.mmd`

```text
1. Build CyberSecurity_OWASP OpenEnv server.
2. Prepare validated scenario cache once per generator/verifier version.
3. Run baseline eval with cached validation/held-out bundles.
4. Train with GRPO/TRL or Unsloth using cached rollout episodes.
5. Log reward components, pass rates, reset latency, and cache hit metrics to Trackio.
6. Run held-out eval every N training steps.
7. Inspect failure clusters and cache sampling weights.
8. Refresh only 5-10% of scenarios per epoch when new weak spots are found.
9. Produce final demo: before/after trace + reward curve + held-out eval table.
```

Recommended initial training setup (Modal-first):

```text
Model: unsloth/gemma-4-E2B-it
Algorithm: GRPO via TRL or Unsloth-compatible loop
Dataset prompt: repeated task instruction with randomized scenario IDs
Max steps per episode: 30
Rollouts per prompt: 2-4
Logging: Trackio
Primary eval: held-out deterministic test pass rate
Scenario cache mode: require
Scenario cache volume: CyberSecurity_OWASP-scenario-cache

Training execution is expected to run on Modal (persistent or ephemeral) rather than locally.
```

## 9. Deployment architecture

The environment should be runnable in 3 modes:

| Mode | Purpose |
|---|---|
| Local Uvicorn | Fast engineer iteration. |
| Docker | Reproducible local training/eval. |
| Hugging Face Spaces | Public hackathon demo and OpenEnv-compliant hosting. |

Expected endpoints:

```text
/ws       OpenEnv client session
/health   health check
/reset    debug reset
/step     debug step
/state    debug state
/docs     FastAPI docs
/web      optional web UI
```

## 10. Implementation milestones

### Milestone 1 — Skeleton environment

- `models.py`
- `client.py`
- `server/environment.py`
- `server/app.py`
- `server/Dockerfile`
- `openenv.yaml`
- health check
- one hand-written scenario

### Milestone 2 — Scenario compiler

- policy graph format
- app template renderer
- bug injector
- DB fixture generator
- public and hidden test generator

### Milestone 3 — Reward engine

- public test score
- hidden invariant score
- regression score
- patch minimality score
- safety/reward-hacking penalties
- reward component logging

### Milestone 4 — Training script

- rollout loop
- GRPO/TRL or Unsloth training script
- Trackio logging
- checkpoint save/push
- baseline and post-training eval

### Milestone 5 — Hackathon demo

- HF Spaces deployment
- mini-blog
- 2-minute video
- before/after traces
- reward curve
- held-out eval table

## 11. Engineering notes

- Keep scenario apps small: ideally 5-15 files each.
- Prefer deterministic tests over LLM judging.
- Hide final hidden test details from observations.
- Log enough trace data to debug failures but never leak hidden tests to the agent.
- Include intentionally public routes and allowed cross-role cases so the model does not learn “add auth everywhere.”
- The best demo is not just “agent finds bug,” but “agent learns not to break valid business behavior.”

## 12. Source notes and credibility

| Source | How it informs this architecture | Credibility |
|---|---|---:|
| OWASP Top 10 2025 / A01 Broken Access Control | Confirms why access control is the right security focus. | 10/10 |
| OWASP ASVS access-control guidance | Informs policy invariants and server-side authorization checks. | 9.5/10 |
| OpenEnv environment-building docs | Defines required models, reset/step/state, FastAPI server, Docker, and client. | 8.5/10 |
| OpenEnv quickstart/architecture docs | Informs WebSocket client/server design, typed EnvClient, and container isolation. | 8.5/10 |
| OpenEnv deployment docs | Informs HF Spaces deployment, endpoints, Docker workflow, and installable client package. | 8.5/10 |
| Hackathon judging criteria | Informs demo priorities: innovation, storytelling, reward improvement, and training pipeline. | 9/10 |
| TRL/OpenEnv training example | Informs rollout function, decomposed reward functions, and Trackio logging pattern. | 8/10 |
| Kube SRE Gym README | Informs the closed-loop pattern: adversarial scenario design, curriculum mastery tracking, real tool interaction, verification, and artifact-driven storytelling. | 8/10 |
| DeepSeek-V4-Pro Hugging Face model card and encoding notes | Informs the default offline scenario-author config and the note that prompt handling should not assume a Jinja chat template. | 8/10 |
