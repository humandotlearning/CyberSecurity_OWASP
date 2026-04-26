# From Mythos to Mobile-Sized Defenders: Training Small Models to Repair OWASP 2025 top vulnerability

## Motivation

Anthropic's Project Glasswing was the moment this project clicked for me.[1]

Glasswing is aimed at securing critical software with Claude Mythos Preview, a frontier model Anthropic describes as capable of finding high-severity vulnerabilities in major operating systems and browsers.[2] That is important defensive work, but it raises an uncomfortable question:

**What about everyone else?**

Large operating systems, browsers, banks, and cloud providers may get access to frontier cybersecurity models and expensive scanning pipelines. Smaller teams, solo developers, open-source maintainers, indie hackers, and "vibe coders" are also shipping real software. Their code handles invoices, accounts, uploads, profiles, subscriptions, internal dashboards, and customer data. They face the same class of vulnerabilities, but they do not have the same budget, security staff, or model access.

So I built **CyberSecurity_OWASP** around a different idea:

> If frontier models can scale vulnerability discovery, small RL-trained defenders should scale vulnerability prevention.

The goal is not another benchmark where an LLM answers security trivia. The goal is an OpenEnv environment where a small open model can learn an actual defensive workflow: inspect an application, understand the intended authorization policy, discover a broken access control bug, patch the code, and preserve legitimate behavior.

## Why OWASP A01?

The first target is **OWASP A01:2025 - Broken Access Control**.

OWASP ranks Broken Access Control as the number one web application security risk in the 2025 Top 10. Access control failures can let users act outside their intended permissions, including unauthorized disclosure, modification, or destruction of data.[3]

This is exactly the kind of bug that small teams often ship accidentally. A route works. The visible test passes. The developer checks authentication. But one missing ownership check lets Bob read Alice's invoice by changing an ID in the URL.

OWASP lists insecure direct object references, parameter tampering, missing API access controls, privilege escalation, and force browsing as common broken access control patterns. It also recommends enforcing access control in trusted server-side code, denying by default, and checking record ownership instead of allowing arbitrary object access.

That makes A01 a strong starting point for reinforcement learning:

- common enough to matter
- subtle enough that shallow unit tests miss it
- concrete enough for deterministic verification
- realistic enough to map to developer workflows
- expandable into many scenario families

## What CyberSecurity_OWASP Does

**CyberSecurity_OWASP** is an OpenEnv-compliant reinforcement-learning environment for a single LLM agent performing a defensive authorization-repair task.

The episode loop is:

```text
inspect generated app + policy
-> discover authorization bug
-> submit diagnosis
-> patch code
-> preserve intended behavior
```

The current MVP focuses on generated FastAPI-style invoice applications with injected OWASP A01 BOLA/IDOR defects. The agent must inspect the app, compare identities, use safe local requests, diagnose the bug, patch the vulnerable route or service code, run visible checks, and submit a final fix.

This is not a static multiple-choice benchmark. It is an interactive environment with tools, state, hidden checks, and reward feedback.

The agent can use tools such as:

```text
inspect_policy_graph
list_routes
read_openapi
read_file
search_code
send_local_request
compare_identities
submit_diagnosis
patch_file
run_visible_tests
submit_fix
```

The tools are phase-gated. During discovery, the agent can inspect policy, routes, files, OpenAPI summaries, and safe local request behavior. During patching, it can edit allowed app files and run visible tests. After completion, it receives a stable terminal observation.

## The Task the Model Learns

A typical episode looks like this:

1. The environment generates an invoices app with users, tenants, invoices, routes, and a policy graph.
2. One authorization defect is injected.
3. The model sees partial information: product rules, route summaries, fixture aliases, visible test results, and tool outputs.
4. The model must infer the intended policy.
5. It must find a route where one user can access another user's resource.
6. It must submit a diagnosis before patching.
7. It must patch the application without breaking valid owner, admin, or public-route behavior.
8. It must pass visible tests and hidden verifier checks.

For example, the bug may be:

```text
GET /invoices/{invoice_id}
```

The route authenticates the user, loads the invoice by ID, and returns it. But it forgets to verify that the invoice belongs to the current user's tenant or that the current user is an admin. A shallow test may confirm that Alice can fetch Alice's invoice. The hidden exploit checks whether Bob can fetch Alice's invoice.

A useful model must not simply block everything. It has to preserve intended behavior:

```text
owner can read own invoice       -> allowed
admin can read tenant invoice    -> allowed
other user can read invoice      -> denied
public status route still works  -> allowed
```

That is the core reason this is useful for RL. The model is rewarded not for sounding secure, but for making the application secure while preserving product behavior.

## Reward Design

The reward is decomposed so the model cannot win by shortcutting:

```python
{
    "discovery": ...,
    "security": ...,
    "regression": ...,
    "public_routes": ...,
    "patch_quality": ...,
    "visible_tests": ...,
    "safety": ...,
    "anti_cheat": ...,
    "terminal_total": ...,
}
```

Training can also enable dense shaping signals:

```python
{
    "progressive": ...,
    "step_penalty": ...,
    "speed_bonus": ...,
    "token_penalty": ...,
    "behavior_penalty": ...,
    "train_total": ...,
}
```

The verifier checks multiple layers:

```text
visible tests
+ hidden authorization exploit tests
+ policy-oracle matrix
+ regression checks
+ public-route preservation
+ patch-quality checks
+ anti-cheat rules
```

The model is penalized for patterns like:

```text
deny-all fixes
hardcoded user IDs
hardcoded invoice IDs
test or fixture tampering
probing hidden files
external URL attempts
invalid or repeated action loops
breaking public routes
```

This matters because security environments are especially vulnerable to reward hacking. A model that "fixes" IDOR by returning 403 for every endpoint is not a security agent; it is a product outage generator. CyberSecurity_OWASP rewards the harder behavior: block the exploit while preserving the intended application contract.

## Why This Is an Environment

CyberSecurity_OWASP is an environment because the model must interact with a partially observable world. It does not receive the answer. It has to gather evidence, run safe local probes, understand policy, make edits, and submit a final fix.

It is long-horizon because success requires a sequence of correct steps. Reading the wrong file, skipping diagnosis, patching the wrong layer, or breaking public routes can reduce reward.

It is self-improving because the curriculum controller can select difficulty tiers and target weak spots. Scenario generation is cache-backed, configurable, and extensible. The environment can create more authorization tasks as the model improves.

It is measurable because every episode ends with deterministic verification. The model either blocked the exploit and preserved behavior, or it did not.

## Training Approach

The target policy model is **Gemma 4 E2B Instruct** through Unsloth.

The model choice is intentional: small, instruction-tuned, code-capable models are closer to the cost and latency profile needed for local developer workflows. Google describes Gemma 4 as an open model family designed for reasoning, agentic workflows, function calling, structured JSON output, code generation, and efficient deployment across hardware sizes.[4] Unsloth supports fine-tuning Gemma 4 E2B, including text and RL workflows.[5]

The training scaffold has two stages.

### 1. Synthetic SFT warm start

A teacher model executes real environment trajectories. Only trajectories that pass the deterministic verifier are kept. This creates supervised data where each row teaches the model a valid step in a successful security-repair workflow.

Example actions include:

```json
{"tool_name": "inspect_policy_graph"}
{"tool_name": "read_file", "arguments": {"path": "app/routes/invoices.py"}}
{"tool_name": "send_local_request", "arguments": {"method": "GET", "path": "/invoices/inv_alice_001"}}
{"tool_name": "submit_diagnosis", "arguments": {"bug_type": "broken_access_control"}}
{"tool_name": "patch_file", "arguments": {"path": "app/routes/invoices.py", "diff": "..."}}
{"tool_name": "submit_fix"}
```

### 2. GRPO reinforcement learning

After SFT, GRPO trains the model against the live OpenEnv environment. The model receives reward from the verifier, not from a preference label. This lets it optimize for real task success: discovering the bug, repairing it, and preserving behavior.

Runs are logged through Trackio. Modal launchers support cache preparation, smoke tests, SFT, and GRPO. The environment keeps scenario generation separate from training so GPU jobs do not waste time compiling scenarios during rollout.

## Evaluation Focus

The most important metric is not whether the model can say "this is IDOR." The important metric is whether it can produce a patch that passes hidden authorization checks while keeping legitimate application behavior intact.

The evaluation suite tracks:

- average terminal reward
- exploit-block rate
- regression-preservation rate
- public-route preservation rate
- invalid-action rate
- anti-cheat pass rate
- full success rate

The blog should be paired with the latest Trackio dashboard and `outputs/evals/` summaries for concrete before-and-after numbers. This write-up intentionally avoids claiming training results that are not present in the repository.

## What Makes This Useful

Most developer security tooling is reactive. A scanner runs after code is written. A bug bounty report arrives after deployment. A security review happens late, if it happens at all.

The long-term direction here is proactive protection.

Imagine a small local model that runs inside a developer workflow:

```text
before commit
before deploy
inside CI
inside an IDE
inside a mobile or offline coding assistant
```

It reads the policy, inspects changed routes, tries safe local probes, identifies authorization gaps, and proposes patches. Because the model is small, it can be cheap enough to run repeatedly. Because it is RL-trained in an environment with hidden policy-oracle checks, it can learn behavior beyond static pattern matching.

The point is not to replace professional security review. The point is to make a baseline level of defensive reasoning available to teams that currently have almost none.

## Responsible Scope

CyberSecurity_OWASP is defensive by construction.

The environment uses synthetic generated applications and safe local requests. Hidden tests, exploit labels, oracle tuples, and scenario-family labels are not exposed to the agent. External URL attempts are penalized. The model is trained to diagnose and patch authorization bugs in a sandbox, not to attack real systems.

That boundary is important. The same AI progress that makes automated vulnerability discovery powerful also makes safe training environments more urgent.

## Future Work

The current MVP focuses on OWASP A01 Broken Access Control, especially BOLA/IDOR-style defects. The same framework can expand to additional OWASP risk families and richer application shapes:

- more app domains
- more policy shapes
- multi-route authorization chains
- schema drift
- stronger curriculum adaptation
- realistic CI-style patch review
- larger held-out scenario families

The bigger vision is a family of small, specialized security agents that can run close to where software is written.

Project Glasswing shows what frontier models may do for the world's most critical software. CyberSecurity_OWASP asks a complementary question:

**Can we train small open models to protect the long tail of everyday software?**

This submission is a first step toward that answer.

[1]: https://www.anthropic.com/glasswing "Project Glasswing: Securing critical software for the AI era | Anthropic"
[2]: https://red.anthropic.com/2026/mythos-preview/ "Claude Mythos Preview | red.anthropic.com"
[3]: https://owasp.org/Top10/2025/A01_2025-Broken_Access_Control/ "A01 Broken Access Control - OWASP Top 10:2025"
[4]: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ "Gemma 4: Our most capable open models to date"
[5]: https://unsloth.ai/docs/models/gemma-4/train "Gemma 4 Fine-tuning Guide | Unsloth Documentation"
