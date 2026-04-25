# Hackathon Training Notes

Sources:

- `D:/delete_later/[External] Meta OpenEnv Hackathon Participant Help Guide.docx`
- `D:/delete_later/Hackathon FAQs (participants).docx`
- `D:/delete_later/OpenEnv_Hackathon_Resources.docx`

## Core Build Order

- Build the environment before the trainer. Define observation, action space, episode end conditions, reward, abuse limits, and deterministic replay first.
- Treat the verifier and reward engine as the task specification. Prefer executable checks over subjective judgments.
- Use OpenEnv for `reset`, `step`, `state`, typed action/observation models, server deployment, and trainer integration.
- Use TRL for GRPO or related post-training, and use Unsloth when memory or rollout speed is the bottleneck.
- Deploy or run the environment early through local Python, Docker, or Hugging Face Spaces so packaging and client-server issues surface before training.

## Training Readiness

Do not start meaningful training until:

- `reset`, `step`, rewards, timeouts, and logs work locally.
- The verifier has been adversarially tested.
- At least a few easy tasks produce non-zero reward.
- Random, bad, and oracle policies provide expected behavior.
- Sample rollouts can be inspected and do not reveal reward-hacking shortcuts.

Use this scale-up order:

1. One manual episode.
2. Scripted policies.
3. Ten validation rollouts.
4. Tiny frozen-model or debug GRPO run.
5. Larger rollout count.
6. Full training run.

## Reward Engineering

- Use multiple independent reward components instead of one scalar only.
- Reward true outcomes first: exploit blocked, legitimate flows preserved, public routes preserved, app boots, and visible tests pass.
- Penalize shortcuts: protected-file edits, hidden-test access, hardcoded identities/resources, deny-all patches, external network attempts, and environment abuse.
- Keep explanation quality auxiliary only; do not let an LLM judge dominate the primary reward.
- Watch for rising reward without better behavior. That usually means the reward was hacked or the verifier is too weak.

## Curriculum

- Start with short-horizon, easy, high-signal tasks where success probability is above zero.
- Increase difficulty only after the model gets reliable partial reward.
- If exploit blocking is poor, add easier security tasks.
- If regressions increase, add positive-flow and public-route traps.
- If validation reward plateaus, add unseen layouts, domains, and harder held-out combinations.

## Monitoring And Demo Evidence

Track overall reward, component rewards, success indicators, timeouts, invalid actions, and sampled generated strategies. Inspect actual rollouts throughout training.

A strong hackathon demo shows:

- Baseline attempt and verifier output.
- Trained attempt and measurable improvement.
- Held-out domain/layout/bug results.
- Reward curves and component metrics.
- Anti-cheat evidence showing the model did not learn deny-all, hardcoding, or fixture tampering.

## Common Mistakes

- Training before the environment and verifier are stable.
- Choosing a task with near-zero chance of reward.
- Using only one reward function.
- Monitoring average reward but not sampled behavior.
- Forgetting timeouts, sandboxing, or protected-file checks.
- Saving LoRA/QLoRA models through an unsafe merge path.
