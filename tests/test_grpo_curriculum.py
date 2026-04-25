from training.grpo_curriculum import (
    AdaptiveDifficultyCurriculum,
    ScenarioGroupRegistry,
    build_scenario_group_rows,
)


def _entries():
    return [
        {
            "seed": 10,
            "split": "train",
            "difficulty": 0,
            "template_id": "fastapi_basic",
            "bug_family": "bola_idor",
            "scenario_hash": "hash-a",
            "validated": True,
        },
        {
            "seed": 20,
            "split": "train",
            "difficulty": 1,
            "template_id": "fastapi_basic",
            "bug_family": "bfla",
            "scenario_hash": "hash-b",
            "validated": True,
        },
        {
            "seed": 30,
            "split": "train",
            "difficulty": 1,
            "template_id": "fastapi_basic",
            "bug_family": "tenant_leak",
            "scenario_hash": "hash-c",
            "validated": True,
        },
    ]


def test_scenario_group_reuses_assignment_for_all_generations():
    registry = ScenarioGroupRegistry(
        _entries(),
        split="train",
        initial_difficulty=0,
        rng_seed=1,
        max_level=1,
    )

    first = registry.assignment_for(scenario_group_id=101, difficulty_policy="adaptive")
    second = registry.assignment_for(scenario_group_id=101, difficulty_policy="adaptive")

    assert first == second


def test_different_scenario_groups_use_different_cached_scenarios_when_available():
    registry = ScenarioGroupRegistry(
        _entries(),
        split="train",
        initial_difficulty=1,
        rng_seed=3,
        max_level=1,
    )

    first = registry.assignment_for(
        scenario_group_id=201,
        requested_seed=20,
        requested_difficulty=1,
        split="train",
        difficulty_policy="fixed",
    )
    second = registry.assignment_for(
        scenario_group_id=202,
        requested_seed=30,
        requested_difficulty=1,
        split="train",
        difficulty_policy="fixed",
    )

    assert first["scenario_hash"] != second["scenario_hash"]


def test_fixed_assignment_uses_dataset_seed_and_difficulty():
    registry = ScenarioGroupRegistry(
        _entries(),
        split="train",
        initial_difficulty=0,
        rng_seed=1,
        max_level=1,
    )

    assignment = registry.assignment_for(
        scenario_group_id=301,
        requested_seed=20,
        requested_difficulty=1,
        split="train",
        difficulty_policy="fixed",
    )

    assert assignment["seed"] == 20
    assert assignment["difficulty"] == 1
    assert assignment["scenario_hash"] == "hash-b"


def test_adaptive_curriculum_promotes_and_demotes_at_thresholds():
    promote = AdaptiveDifficultyCurriculum(
        min_level=0,
        max_level=2,
        current_level=0,
        promote_after=50,
    )
    for _ in range(50):
        promote.update(0, True)
    assert promote.current_level == 1

    demote = AdaptiveDifficultyCurriculum(
        min_level=0,
        max_level=2,
        current_level=1,
        promote_after=50,
    )
    for _ in range(50):
        demote.update(1, False)
    assert demote.current_level == 0


def test_build_scenario_group_rows_include_grpo_group_columns():
    rows = build_scenario_group_rows(
        dataset_size=2,
        training_prompt="repair local app",
        seed_start=7,
        split="train",
        difficulty=1,
    )

    assert rows[0]["scenario_group_id"] == 7
    assert rows[1]["scenario_group_id"] == 8
    assert rows[0]["difficulty_policy"] == "adaptive"
    assert rows[0]["prompt"][0]["content"] == "repair local app"
