"""Reward functions exposed for TRL/GRPO logging."""


def _values(name: str, completions, kwargs):
    return [float(x) for x in kwargs.get(name, [0.0] * len(completions))]


def reward_total(completions, **kwargs):
    return _values("reward_total", completions, kwargs)


def reward_security(completions, **kwargs):
    return _values("reward_security", completions, kwargs)


def reward_regression(completions, **kwargs):
    return _values("reward_regression", completions, kwargs)


def reward_patch_quality(completions, **kwargs):
    return _values("reward_patch_quality", completions, kwargs)


def reward_anti_cheat(completions, **kwargs):
    return _values("reward_anti_cheat", completions, kwargs)
