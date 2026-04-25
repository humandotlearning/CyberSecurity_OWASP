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
    %% =========================
    %% Offline Build Layer
    %% =========================
    subgraph A[Offline Scenario Factory]
        A1[Policy Graph Generator\nroles, users, tenants, ownership, route intent]
        A2[App Template Library\nFastAPI, Express, Django MVP templates]
        A3[Bug Injector\nmissing guard, IDOR, tenant leak, role confusion, query omission]
        A4[Scenario Compiler\nmaterializes app + DB + public tests + hidden invariants]
        A5[Split Manager\ntrain seeds, validation seeds, hidden held-out seeds]
        A1 --> A4
        A2 --> A4
        A3 --> A4
        A5 --> A4
    end

    %% =========================
    %% OpenEnv Runtime
    %% =========================
    subgraph B[CyberSecurity_OWASP OpenEnv Server]
        B1[reset\(\)\nselect scenario + start sandbox]
        B2[Sandbox App Runtime\nlocal app, DB fixture, logs, route map]
        B3[Tool API exposed through step\(action\)\nReadFile, ListRoutes, SendLocalRequest, RunTests, ApplyPatch, SubmitFix]
        B4[State Store\nepisode_id, step_count, scenario_id, patch diff, test history]
        B5[Deterministic Reward Engine\npolicy tests + hidden tests + regression tests + penalties]
        B6[state\(\)\nstructured metadata for debugging/eval]
        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B4 --> B6
    end

    %% =========================
    %% Agent + Training
    %% =========================
    subgraph C[Single LLM Agent]
        C1[Observation Parser]
        C2[Planner\npolicy reasoning + patch strategy]
        C3[Action Generator\nchooses next OpenEnv action]
        C1 --> C2 --> C3
    end

    subgraph D[Training + Evaluation]
        D1[Rollout Loop\nreset → step* → final reward]
        D2[GRPO / TRL / Unsloth Training]
        D3[Trackio Metrics\nreward curves, pass rates, patch size, steps]
        D4[Held-out Eval Suite\nunseen templates, seeds, names, route structures]
        D5[Demo Artifacts\nbefore/after traces, mini-blog, 2-minute video]
        D1 --> D2 --> D3
        D3 --> D4 --> D5
    end

    A4 --> B1
    C3 -->|typed action| B3
    B3 -->|observation + reward + done| C1
    B5 --> D1
    D2 --> C1
    B5 --> D4
```

## 3. Component responsibilities

### 3.1 Scenario Factory

The scenario factory generates many small but realistic web apps from a structured authorization policy.

It should output:

- application code;
- route map;
- database fixture;
- user/session/token fixtures;
- policy graph;
- intentionally injected access-control bug;
- public tests visible to the agent;
- hidden tests invisible to the agent;
- metadata for eval and debugging.

The scenario compiler is the main anti-overfitting mechanism. It should vary:

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

- `reset()` — initialize a fresh scenario instance.
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
    action_type: Literal[
        "read_file",
        "list_files",
        "list_routes",
        "inspect_policy",
        "send_local_request",
        "run_public_tests",
        "apply_patch",
        "submit_fix",
    ]
    arguments: dict
```

Recommended actions:

| Action | Purpose | Safety boundary |
|---|---|---|
| `inspect_policy` | Read intended authorization rules. | Only synthetic policy. |
| `list_routes` | See local app route map. | No internet target. |
| `read_file` | Inspect selected source file. | Sandbox allowlist only. |
| `send_local_request` | Validate behavior against local app. | Local generated app only. |
| `run_public_tests` | Run visible tests. | No hidden test disclosure. |
| `apply_patch` | Modify source through unified diff. | Patch size and file allowlist limits. |
| `submit_fix` | End episode and trigger hidden eval. | Final hidden score only, no leaked test details. |

### 3.6 Observation schema

Observations should be compact and structured.

```python
@dataclass
class CyberSecurityOWASPObservation(Observation):
    message: str
    visible_policy_summary: str
    route_summary: list[dict]
    last_action_result: dict
    public_test_summary: dict
    patch_summary: dict
    done_reason: str | None = None
```

Do not expose hidden test bodies, hidden expected outputs, or seed-specific solution hints.

### 3.7 State schema

State should support debugging and training analytics.

```python
@dataclass
class CyberSecurityOWASPState(State):
    episode_id: str
    scenario_id: str
    split: Literal["train", "validation", "heldout"]
    step_count: int = 0
    max_steps: int = 30
    scenario_family: str = ""
    app_template: str = ""
    files_touched: list[str] = field(default_factory=list)
    public_tests_passed: int = 0
    public_tests_total: int = 0
    hidden_tests_passed: int = 0
    hidden_tests_total: int = 0
    accumulated_reward: float = 0.0
```

## 4. Episode lifecycle

```text
1. reset()
   - sample train/validation scenario seed
   - compile app from policy graph + template + injected bug
   - start local sandbox app and DB fixture
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
   - run regression tests
   - compute deterministic reward
   - return final observation, reward, done=True

4. logging
   - record scenario_id, action trace, patch diff, reward components
   - send metrics to Trackio during training/eval
```

## 5. Reward design

The reward should be deterministic, decomposed, and resistant to reward hacking.

Recommended reward formula:

```text
R = 0.35 * public_policy_tests
  + 0.30 * hidden_authz_invariants
  + 0.15 * regression_preservation
  + 0.10 * evidence_quality
  + 0.05 * patch_minimality
  + 0.05 * efficiency
  - penalties
```

### Reward components

| Component | Weight | What it rewards |
|---|---:|---|
| Public policy tests | 0.35 | Agent fixes known failing behavior. |
| Hidden authz invariants | 0.30 | Patch generalizes beyond visible tests. |
| Regression preservation | 0.15 | Valid user flows and intended public routes still work. |
| Evidence quality | 0.10 | Agent gathered relevant policy/test/file evidence before patching. |
| Patch minimality | 0.05 | Small focused patches instead of broad rewrites. |
| Efficiency | 0.05 | Fewer wasted steps and repeated actions. |

### Penalties

| Penalty | Trigger |
|---|---|
| `-0.25` | Breaks public route intentionally marked public. |
| `-0.25` | Deletes tests, policy file, or route instead of fixing authorization. |
| `-0.20` | Hardcodes seed-specific IDs, users, tenants, or hidden assumptions. |
| `-0.15` | Over-broad denial that blocks legitimate authorized users. |
| `-0.10` | Patch exceeds file or diff-size budget. |
| `-1.00` | Attempts external network access, credential extraction, persistence, or unsafe behavior. |

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
2. Generate 600 MVP scenarios.
3. Run baseline eval with the base model.
4. Train with GRPO/TRL or Unsloth using rollout episodes.
5. Log reward components to Trackio.
6. Run held-out eval every N training steps.
7. Inspect failure clusters.
8. Add scenario mutations only if failures reveal overfitting.
9. Produce final demo: before/after trace + reward curve + held-out eval table.
```

Recommended initial training setup:

```text
Model: Qwen/Qwen3-1.7B or similar small instruct model
Algorithm: GRPO via TRL or Unsloth-compatible loop
Dataset prompt: repeated task instruction with randomized scenario IDs
Max steps per episode: 30
Rollouts per prompt: 2-4
Logging: Trackio
Primary eval: held-out deterministic test pass rate
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
