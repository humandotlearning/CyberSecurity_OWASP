"""Versioned executable scenario cache for fast deterministic reset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

try:
    from ..config import ScenarioAuthoringSettings, load_scenario_authoring_config
    from .curriculum import CurriculumController
    from .scenario_factory import ScenarioFactory
except ImportError:  # pragma: no cover
    from config import ScenarioAuthoringSettings, load_scenario_authoring_config
    from server.curriculum import CurriculumController
    from server.scenario_factory import ScenarioFactory


SCENARIO_CACHE_REQUIRED_FILES = (
    "scenario.json",
    "app_source",
    "policy_graph.json",
    "visible_tests.py",
    "hidden_tests.py",
    "oracle_tests.py",
    "expected_exploit_trace.json",
    "reward_config.json",
    "metadata.json",
)
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class ScenarioCacheKey:
    difficulty_level: int
    authz_bug_type: str
    app_family: str
    framework: str
    policy_shape: str
    tenant_model: str
    exploit_depth: str
    patch_scope: str
    regression_risk: str
    generator_version: str
    verifier_version: str
    scenario_hash: str

    def stable_id(self) -> str:
        return _stable_hash(asdict(self))[:16]

    def path_slug(self) -> str:
        return (
            f"d{self.difficulty_level}-{self.authz_bug_type}-"
            f"{self.app_family}-{self.framework}-{self.stable_id()}"
        ).replace("/", "-").replace("_style_python", "")


@dataclass(frozen=True)
class ScenarioCacheLoad:
    scenario: dict[str, Any]
    bundle_path: Path
    load_latency_ms: float


class ScenarioCacheMiss(RuntimeError):
    """Raised when runtime cache mode requires a bundle that is not present."""


class ScenarioCache:
    """Reads and writes complete executable scenario bundles."""

    def __init__(
        self,
        root: str | Path,
        *,
        settings: ScenarioAuthoringSettings | None = None,
    ):
        self.root = Path(root)
        self.settings = settings or load_scenario_authoring_config()

    def write_bundle(self, scenario: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
        key = cache_key_for_scenario(scenario, settings=self.settings)
        bundle_path = self._bundle_path(
            split=str(scenario["split"] if "split" in scenario else scenario["curriculum_snapshot"].get("split", "train")),
            difficulty=int(scenario["difficulty"]),
            key=key,
        )
        if bundle_path.exists() and not force:
            metadata = self._read_json(bundle_path / "metadata.json")
            return {"created": False, "bundle_path": str(bundle_path), **metadata}

        workspace = Path(scenario["workspace"])
        if bundle_path.exists():
            shutil.rmtree(bundle_path)
        bundle_path.mkdir(parents=True, exist_ok=True)
        app_source = bundle_path / "app_source"
        app_source.mkdir(parents=True, exist_ok=True)

        editable_files = list(scenario["hidden_facts"].get("editable_files", []))
        for rel in editable_files:
            source = workspace / rel
            target = app_source / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        hidden_facts = _cacheable_hidden_facts(scenario["hidden_facts"])
        scenario_record = {
            "schema_version": 1,
            "task_id": scenario["task_id"],
            "seed": _seed_from_task_id(scenario["task_id"]),
            "split": scenario["curriculum_snapshot"].get("split", "train"),
            "difficulty": int(scenario["difficulty"]),
            "difficulty_tier": scenario["difficulty_tier"],
            "domain": scenario["domain"],
            "bug_family": scenario["bug_family"],
            "scenario_family": scenario["scenario_family"],
            "template_id": scenario["template_id"],
            "target_weakness": scenario["target_weakness"],
            "task_brief": scenario["task_brief"],
            "public_hint": scenario["public_hint"],
            "workspace_summary": scenario["workspace_summary"],
            "hidden_facts": hidden_facts,
            "editable_files": editable_files,
            "curriculum_snapshot": scenario.get("curriculum_snapshot", {}),
            "cache_key": asdict(key),
        }
        metadata = {
            "cache_key": asdict(key),
            "scenario_hash": key.scenario_hash,
            "generator_version": self.settings.runtime.generator_version,
            "verifier_version": self.settings.runtime.verifier_version,
            "scenario_author_model": self.settings.scenario_author.model_id,
            "scenario_author_provider": self.settings.scenario_author.provider,
            "difficulty_calibration_strategy": (
                self.settings.curriculum.difficulty_calibration_strategy
            ),
            "validated": True,
            "bundle_files": list(SCENARIO_CACHE_REQUIRED_FILES),
        }

        _write_json(bundle_path / "scenario.json", scenario_record)
        _write_json(bundle_path / "policy_graph.json", scenario["public_hint"])
        _write_json(bundle_path / "expected_exploit_trace.json", _expected_exploit_trace(hidden_facts))
        _write_json(bundle_path / "reward_config.json", _reward_config())
        _write_json(bundle_path / "metadata.json", metadata)
        (bundle_path / "visible_tests.py").write_text(
            (workspace / "tests/test_visible.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (bundle_path / "hidden_tests.py").write_text(
            _hidden_tests_contract(),
            encoding="utf-8",
        )
        (bundle_path / "oracle_tests.py").write_text(
            _oracle_tests_contract(),
            encoding="utf-8",
        )
        self._update_manifest(bundle_path, scenario_record, metadata)
        return {"created": True, "bundle_path": str(bundle_path), **metadata}

    def load_bundle(
        self,
        *,
        seed: int,
        split: str,
        difficulty: int,
        family_budget: dict[str, Any] | None = None,
    ) -> ScenarioCacheLoad:
        del family_budget  # reserved for weighted family sampling once multiple families exist
        started = time.perf_counter()
        bundle_path = self.find_bundle(seed=seed, split=split, difficulty=difficulty)
        if bundle_path is None:
            raise ScenarioCacheMiss(
                f"No cached scenario bundle for split={split!r}, difficulty={difficulty}, seed={seed}."
            )
        validate_bundle(bundle_path)
        scenario_record = self._read_json(bundle_path / "scenario.json")
        metadata = self._read_json(bundle_path / "metadata.json")
        workspace = _make_workspace(prefix=f"cybersecurity_owasp_cached_{split}_{seed}_")
        shutil.copytree(bundle_path / "app_source", workspace, dirs_exist_ok=True)

        editable_files = list(scenario_record["editable_files"])
        hidden_facts = dict(scenario_record["hidden_facts"])
        hidden_facts.update(
            {
                "workspace": str(workspace),
                "editable_files": editable_files,
                "initial_file_hashes": {
                    rel: (workspace / rel).read_text(encoding="utf-8")
                    for rel in editable_files
                },
                "scenario_cache": {
                    "bundle_path": str(bundle_path),
                    "cache_key": metadata["cache_key"],
                    "scenario_hash": metadata["scenario_hash"],
                    "generator_version": metadata["generator_version"],
                    "verifier_version": metadata["verifier_version"],
                },
            }
        )
        scenario = {
            "task_id": scenario_record["task_id"],
            "workspace": workspace,
            "domain": scenario_record["domain"],
            "bug_family": scenario_record["bug_family"],
            "scenario_family": scenario_record["scenario_family"],
            "template_id": scenario_record["template_id"],
            "target_weakness": scenario_record["target_weakness"],
            "difficulty": int(scenario_record["difficulty"]),
            "difficulty_tier": scenario_record["difficulty_tier"],
            "curriculum_snapshot": {
                **scenario_record.get("curriculum_snapshot", {}),
                "split": split,
                "cache_key": metadata["cache_key"],
                "scenario_hash": metadata["scenario_hash"],
            },
            "task_brief": scenario_record["task_brief"],
            "public_hint": scenario_record["public_hint"],
            "workspace_summary": scenario_record["workspace_summary"],
            "hidden_facts": hidden_facts,
            "cache": {
                "hit": True,
                "bundle_path": str(bundle_path),
                "cache_key": metadata["cache_key"],
                "scenario_hash": metadata["scenario_hash"],
                "load_latency_ms": (time.perf_counter() - started) * 1000,
            },
        }
        return ScenarioCacheLoad(
            scenario=scenario,
            bundle_path=bundle_path,
            load_latency_ms=float(scenario["cache"]["load_latency_ms"]),
        )

    def find_bundle(self, *, seed: int, split: str, difficulty: int) -> Path | None:
        entries = [
            entry
            for entry in self._manifest_entries()
            if entry.get("seed") == int(seed)
            and entry.get("split") == split
            and entry.get("difficulty") == int(difficulty)
            and entry.get("validated") is True
        ]
        if not entries:
            return None
        selected = sorted(entries, key=lambda item: str(item.get("scenario_hash", "")))[0]
        path = self.root / str(selected["bundle_path"])
        return path if path.exists() else None

    def coverage(self) -> dict[str, Any]:
        counts: dict[str, dict[str, int]] = {}
        for entry in self._manifest_entries():
            if not entry.get("validated"):
                continue
            split = str(entry.get("split", "train"))
            difficulty = str(entry.get("difficulty", 0))
            counts.setdefault(split, {})
            counts[split][difficulty] = counts[split].get(difficulty, 0) + 1
        return {"root": str(self.root), "counts": counts, "entries": len(self._manifest_entries())}

    def assert_coverage(self, *, split: str, difficulty: int | None = None) -> dict[str, Any]:
        coverage = self.coverage()
        required = self.settings.curriculum.minimum_for_split(split)
        difficulties: Iterable[int]
        if difficulty is None:
            difficulties = range(self.settings.curriculum.difficulty_bucket_count)
        else:
            difficulties = [difficulty]
        missing: list[dict[str, int]] = []
        split_counts = coverage["counts"].get(split, {})
        for item in difficulties:
            actual = int(split_counts.get(str(item), 0))
            if actual < required:
                missing.append({"difficulty": int(item), "actual": actual, "required": required})
        if missing:
            raise ScenarioCacheMiss(
                f"Scenario cache coverage is below minimum for split={split!r}: {missing}"
            )
        return coverage

    def _bundle_path(self, *, split: str, difficulty: int, key: ScenarioCacheKey) -> Path:
        return self.root / split / f"difficulty_{difficulty}" / key.path_slug()

    def _manifest_entries(self) -> list[dict[str, Any]]:
        manifest_path = self.root / MANIFEST_FILE
        if manifest_path.exists():
            return list(self._read_json(manifest_path).get("entries", []))
        return self._scan_entries()

    def _scan_entries(self) -> list[dict[str, Any]]:
        entries = []
        for metadata_path in self.root.glob("**/metadata.json"):
            bundle_path = metadata_path.parent
            try:
                validate_bundle(bundle_path)
                scenario = self._read_json(bundle_path / "scenario.json")
                metadata = self._read_json(metadata_path)
            except Exception:
                continue
            entries.append(_manifest_entry(self.root, bundle_path, scenario, metadata))
        return entries

    def _update_manifest(
        self,
        bundle_path: Path,
        scenario_record: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / MANIFEST_FILE
        entries = self._manifest_entries()
        entry = _manifest_entry(self.root, bundle_path, scenario_record, metadata)
        entries = [
            item for item in entries if item.get("bundle_path") != entry["bundle_path"]
        ]
        entries.append(entry)
        _write_json(manifest_path, {"schema_version": 1, "entries": sorted(entries, key=lambda item: item["bundle_path"])})

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def cache_key_for_scenario(
    scenario: dict[str, Any],
    *,
    settings: ScenarioAuthoringSettings | None = None,
) -> ScenarioCacheKey:
    settings = settings or load_scenario_authoring_config()
    workspace_summary = scenario.get("workspace_summary", {})
    hidden = scenario.get("hidden_facts", {})
    stable_payload = {
        "task_id": scenario.get("task_id"),
        "difficulty": scenario.get("difficulty"),
        "domain": scenario.get("domain"),
        "bug_family": scenario.get("bug_family"),
        "scenario_family": scenario.get("scenario_family"),
        "template_id": scenario.get("template_id"),
        "target_weakness": scenario.get("target_weakness"),
        "public_hint": scenario.get("public_hint"),
        "users": hidden.get("users"),
        "invoices": hidden.get("invoices"),
    }
    return ScenarioCacheKey(
        difficulty_level=int(scenario.get("difficulty", 0)),
        authz_bug_type=str(scenario.get("bug_family", "unknown")),
        app_family=str(scenario.get("domain", "unknown")),
        framework=str(workspace_summary.get("framework", "unknown")),
        policy_shape="owner_admin_tenant_policy",
        tenant_model="same_tenant_with_foreign_tenant",
        exploit_depth=str(scenario.get("target_weakness", "direct_object_reference")),
        patch_scope="route_guard",
        regression_risk="owner_admin_public_routes",
        generator_version=settings.runtime.generator_version,
        verifier_version=settings.runtime.verifier_version,
        scenario_hash=_stable_hash(stable_payload),
    )


def validate_bundle(bundle_path: str | Path) -> None:
    path = Path(bundle_path)
    missing = [name for name in SCENARIO_CACHE_REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise ScenarioCacheMiss(f"Scenario bundle is incomplete at {path}: missing {missing}")
    scenario = json.loads((path / "scenario.json").read_text(encoding="utf-8"))
    editable = set(scenario.get("editable_files", []))
    protected = {"hidden_tests.py", "oracle_tests.py", "reward_config.json", "metadata.json"}
    if editable.intersection(protected):
        raise ScenarioCacheMiss(f"Scenario bundle exposes protected files as editable: {protected}")


def prepare_scenario_cache(
    *,
    cache_dir: str | Path | None = None,
    settings: ScenarioAuthoringSettings | None = None,
    seed_start: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    settings = settings or load_scenario_authoring_config()
    cache_root = Path(cache_dir or settings.runtime.cache_dir)
    cache = ScenarioCache(cache_root, settings=settings)
    factory = ScenarioFactory()
    curriculum = CurriculumController()
    created: list[dict[str, Any]] = []
    split_counts = {
        "train": settings.curriculum.train_scenarios_per_bucket,
        "validation": settings.curriculum.validation_scenarios_per_bucket,
        "hidden_eval": settings.curriculum.heldout_eval_scenarios_per_bucket,
    }
    for split, per_bucket in split_counts.items():
        for requested_difficulty in range(settings.curriculum.difficulty_bucket_count):
            for index in range(per_bucket):
                seed = int(seed_start) + requested_difficulty * per_bucket + index
                profile = curriculum.select_profile(
                    seed=seed,
                    split=split,
                    requested_difficulty=requested_difficulty,
                )
                scenario = factory.compile_scenario(
                    seed,
                    split=split,
                    difficulty=requested_difficulty,
                    curriculum_profile=profile,
                )
                try:
                    created.append(cache.write_bundle(scenario, force=force))
                finally:
                    workspace = scenario.get("workspace")
                    if workspace:
                        shutil.rmtree(workspace, ignore_errors=True)
    return {
        "cache_dir": str(cache_root),
        "created": sum(1 for item in created if item.get("created")),
        "seen": len(created),
        "coverage": cache.coverage(),
        "config": {
            "difficulty_bucket_count": settings.curriculum.difficulty_bucket_count,
            "train_scenarios_per_bucket": settings.curriculum.train_scenarios_per_bucket,
            "validation_scenarios_per_bucket": settings.curriculum.validation_scenarios_per_bucket,
            "heldout_eval_scenarios_per_bucket": settings.curriculum.heldout_eval_scenarios_per_bucket,
            "scenario_author_model": settings.scenario_author.model_id,
        },
    }


def _cacheable_hidden_facts(hidden: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in hidden.items()
        if key not in {"workspace", "editable_files", "initial_file_hashes"}
    }


def _expected_exploit_trace(hidden: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "GET",
        "path": f"/invoices/{hidden['other_invoice_id']}",
        "actor": hidden["owner_user_id"],
        "vulnerable_status": 200,
        "fixed_status": 403,
    }


def _reward_config() -> dict[str, Any]:
    return {
        "max_reward": 15.0,
        "components": [
            "discovery",
            "security",
            "regression",
            "public_routes",
            "patch_quality",
            "visible_tests",
            "safety",
            "anti_cheat",
        ],
    }


def _hidden_tests_contract() -> str:
    return (
        "# Hidden invariant contract for the deterministic verifier.\n"
        "# Runtime observations must never expose this file.\n"
        "CHECKS = ['same_tenant_other_user_blocked', 'cross_tenant_blocked']\n"
    )


def _oracle_tests_contract() -> str:
    return (
        "# Oracle matrix contract for allowed/denied authorization tuples.\n"
        "# Runtime observations must never expose this file.\n"
        "CHECKS = ['owner_allowed', 'admin_allowed', 'public_allowed', 'cross_tenant_denied']\n"
    )


def _manifest_entry(
    root: Path,
    bundle_path: Path,
    scenario_record: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bundle_path": str(bundle_path.relative_to(root)).replace("\\", "/"),
        "seed": int(scenario_record.get("seed", 0)),
        "split": str(scenario_record.get("split", "train")),
        "difficulty": int(scenario_record.get("difficulty", 0)),
        "scenario_hash": str(metadata.get("scenario_hash", "")),
        "cache_key": metadata.get("cache_key", {}),
        "validated": bool(metadata.get("validated", False)),
    }


def _make_workspace(prefix: str) -> Path:
    root = Path(os.getenv("CYBERSECURITY_OWASP_WORKSPACE_ROOT", tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        workspace = root / f"{prefix}{uuid4().hex[:12]}"
        try:
            workspace.mkdir()
        except FileExistsError:
            continue
        return workspace
    raise RuntimeError("Unable to create isolated cached scenario workspace")


def _seed_from_task_id(task_id: str) -> int:
    try:
        return int(task_id.rsplit("-", 1)[-1])
    except ValueError:
        return 0


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
