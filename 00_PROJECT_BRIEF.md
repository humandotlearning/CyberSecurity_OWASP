# 00_PROJECT_BRIEF.md

# CyberSecurity_OWASP — Project Brief

## 1. One-line summary

`CyberSecurity_OWASP` is an OpenEnv reinforcement-learning environment where a **single LLM agent learns the full defensive workflow for OWASP access-control bugs**: understand the intended authorization policy, discover a broken access-control path in a local synthetic app, patch the code, and prove that the fix blocks unauthorized access without breaking valid user flows.

## 2. Problem

Broken access control remains one of the most important web-application security risks because the correct behavior is usually **application-specific**. Generic scanners can find some missing checks, but they often lack enough context to answer the real engineering question:

> “Given this app’s policy, users, roles, tenants, routes, and data model, is this behavior intended or a security bug?”

Modern LLMs can read code, reason about tests, and propose patches, but they still struggle with:

- distinguishing intended public/feature behavior from accidental over-permission;
- following authorization logic across routes, middleware, ORM queries, tenants, roles, and ownership checks;
- validating that a patch fixes the bug without introducing regressions;
- avoiding reward hacking when tests are visible or too narrow;
- generalizing across app templates instead of memorizing one codebase.

`CyberSecurity_OWASP` turns this into a trainable environment.

## 3. What the environment trains

The environment trains **one agent**, not a separate red-team and blue-team pair. The same model must perform the entire secure-repair loop:

1. **Understand policy** — read the policy graph, user roles, route intent, tenant rules, and allowed operations.
2. **Discover evidence** — use safe local requests, logs, route metadata, and visible tests to identify the likely access-control failure.
3. **Patch** — edit application code, middleware, route guards, query filters, or policy mappings.
4. **Validate** — run public tests, policy checks, and regression tests.
5. **Submit** — final answer is judged by deterministic hidden tests and reward logic.

## 4. Scope for MVP

The MVP should focus on **OWASP A01: Broken Access Control** with ASVS-inspired access-control requirements.

Initial scenario families:

1. Missing route-level authorization check.
2. Insecure direct object reference / object ownership bug.
3. Cross-tenant data leakage.
4. Role confusion: user/admin/support/editor boundary error.
5. Client-side-only authorization assumption.
6. Query filter omission in list/search/export endpoint.
7. Over-broad update/delete permission.
8. Feature route intentionally public, so the agent must not over-secure it.

Recommended MVP size: **8 scenario families × 3 app templates × 25 seeds = 600 trainable scenarios**, with separate held-out families and hidden seeds for evaluation.

## 5. Why this is useful

This environment is useful because it targets a real gap between today’s scanners and useful defensive agents:

- **Scanners detect patterns.** This environment trains policy-aware reasoning.
- **Unit tests check known cases.** This environment includes hidden authorization invariants.
- **Static repair can overfit.** This environment forces the model to preserve valid business behavior.
- **One-app benchmarks are easy to memorize.** This environment compiles many equivalent-but-different apps from policy graphs, templates, route shapes, schema names, and hidden test seeds.

The outcome is a model that becomes better at a practical DevSecOps workflow: safely reviewing and repairing authorization logic in small-to-medium web apps.

## 6. What success looks like

A successful submission should show **measurable reward improvement** and better held-out security behavior after RL training.

### Minimum success criteria

- Environment runs through OpenEnv `reset`, `step`, and `state` APIs.
- Hosted on Hugging Face Spaces.
- Provides a minimal GRPO/TRL or Unsloth training script.
- Tracks training/eval metrics with Trackio or equivalent.
- Shows reward curves and before/after agent behavior.
- Uses deterministic reward as the primary reward source.
- Keeps hidden tests hidden from the agent.

### Target metrics

| Metric | MVP target |
|---|---:|
| Valid episode completion rate | ≥ 85% |
| Hidden authorization test pass rate | ≥ 65% after initial RL run |
| Regression preservation rate | ≥ 80% |
| Held-out scenario success lift vs base model | ≥ +15 percentage points |
| Reward-hacking incidents found in eval | 0 critical |
| Median patch size | ≤ 3 files changed |

## 7. Core design principle

The environment should reward **correct defensive repair**, not exploit creativity. The discovery stage exists only to help the agent gather enough local evidence to make a safe patch. The reward engine must never reward real-world misuse, data exfiltration, persistence, credential theft, or evasion behavior.

## 8. Deliverables for engineers

Initial implementation should produce:

```text
CyberSecurity_OWASP/
├── 00_PROJECT_BRIEF.md
├── 01_ARCHITECTURE.md
├── README.md
├── pyproject.toml
├── openenv.yaml
├── cybersecurity_owasp/
│   ├── __init__.py
│   ├── models.py
│   ├── client.py
│   ├── rewards.py
│   ├── scenarios/
│   │   ├── compiler.py
│   │   ├── policy_graph.py
│   │   ├── templates/
│   │   └── seeds/
│   ├── apps/
│   │   ├── fastapi_basic/
│   │   ├── express_basic/
│   │   └── django_basic/
│   ├── evals/
│   │   ├── public_tests.py
│   │   ├── hidden_invariants.py
│   │   └── heldout_eval.py
│   └── server/
│       ├── environment.py
│       ├── app.py
│       ├── requirements.txt
│       └── Dockerfile
├── training/
│   ├── train_grpo.py
│   ├── rollout.py
│   └── eval_before_after.py
└── outputs/
    ├── logs/
    ├── evals/
    └── reward_curves/
```

## 9. Source notes and credibility

| Source | How it informs this project | Credibility |
|---|---|---:|
| OWASP Top 10 2025 / A01 Broken Access Control | Confirms current relevance of Broken Access Control as a top web-app risk. | 10/10 |
| OWASP ASVS | Provides security-control requirements that can be translated into policy invariants and hidden tests. | 9.5/10 |
| OpenEnv build/deploy docs | Defines the required OpenEnv structure: models, server, client, Docker, HF Spaces deployment. | 8.5/10 |
| Hackathon judging criteria | Aligns deliverables with scoring: innovation, storytelling, reward improvement, and training pipeline. | 9/10 |
| TRL/OpenEnv GRPO example | Shows a practical pattern for environment rollouts, reward functions, and Trackio logging. | 8/10 |

