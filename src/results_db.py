"""
Results Database for Story Theory Benchmark.

Provides persistent storage for benchmark results with support for:
- Incremental model addition (run new model on all tasks)
- Incremental task addition (run new task on all models)
- Deduplication (don't re-run existing combinations)
- Consolidated leaderboard generation
- Atomic saves with file locking (concurrent-safe)
"""

import fcntl
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from measurement_contract import PROTOCOL, digest, identity, project_report, cost_summary

from utils import get_project_root, load_all_tasks


@dataclass
class GenerationRecord:
    """A single generation result."""

    task_id: str
    task_type: str
    theory: str
    model: str
    sample: int
    output: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: (
        int  # Reasoning/thinking tokens (for CoT models like o3, DeepSeek-R1)
    )
    generation_cost: float
    timestamp: str
    success: bool
    error: str | None = None


@dataclass
class EvaluationRecord:
    """A single evaluation result with composite scoring."""

    task_id: str
    task_type: str
    model: str
    sample: int
    evaluator_model: str
    evaluation_cost: float
    timestamp: str
    success: bool
    final_score: float
    score_breakdown: dict[str, Any]
    llm_results: dict[str, Any]
    error: str | None = None


class ResultsDatabase:
    """
    JSON-based results database for the benchmark.

    Structure:
    {
        "benchmark_version": "1.0.0",
        "last_updated": "ISO timestamp",
        "task_version": "hash or count of tasks",
        "generations": [...],
        "evaluations": [...],
        "metadata": {
            "total_generation_cost": 0.0,
            "total_evaluation_cost": 0.0,
            "models_evaluated": [...],
            "tasks_evaluated": [...]
        }
    }
    """

    def __init__(self, db_path: str | Path | None = None, *, assignment_manifest: dict | None = None):
        self.assignment_manifest = assignment_manifest
        if db_path is None:
            db_path = get_project_root() / "results" / "benchmark_results.json"
        self.db_path = Path(db_path)
        self._data: dict[str, Any] = self._load_or_create()

    def _load_or_create(self) -> dict[str, Any]:
        """Load existing database or create new one."""
        if self.db_path.exists():
            with open(self.db_path, "r") as f:
                return json.load(f)
        return self._empty_db()

    def _empty_db(self) -> dict[str, Any]:
        """Create an empty database structure."""
        return {
            "protocol": PROTOCOL if self.assignment_manifest else None,
            "assignment_hash": digest(self.assignment_manifest) if self.assignment_manifest else None,
            "benchmark_version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "generations": [],
            "evaluations": [],
            "metadata": {
                "total_generation_cost": 0.0,
                "total_evaluation_cost": 0.0,
                "models_evaluated": [],
                "tasks_evaluated": [],
            },
        }

    def _reload(self) -> dict[str, Any]:
        """Reload database from disk (for merging concurrent changes)."""
        if self.db_path.exists():
            with open(self.db_path, "r") as f:
                return json.load(f)
        return self._empty_db()

    def _save(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def _update_metadata(self):
        """Metadata preserves unknown spend; report computes its own ledger."""
        self._data["metadata"] = {
            "generation_count": len(self._data["generations"]),
            "evaluation_count": len(self._data["evaluations"]),
            "models_evaluated": sorted({r["model"] for r in self._data["generations"]}),
            "tasks_evaluated": sorted({r["task_id"] for r in self._data["generations"]}),
            "cost": cost_summary([a for kind in ("generations", "evaluations") for r in self._data[kind] for a in r.get("attempts", [])])
        }

    # =========== Query Methods ===========

    def get_existing_generations(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def get_existing_evaluations(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def get_missing_generations(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def get_missing_evaluations(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def get_generation(self, task_id, model, sample=0, *, condition, assignment_manifest):
        # A lookup is a cache hit only against the complete current manifest.
        self.get_results_summary(assignment_manifest)
        return next((r for r in self._data["generations"] if identity(r) == (model, task_id, sample, condition)), None)

    def get_models(self) -> list[str]:
        """Get list of all models that have been evaluated."""
        return self._data["metadata"].get("models_evaluated", [])

    def get_tasks(self) -> list[str]:
        """Get list of all tasks that have been evaluated."""
        return self._data["metadata"].get("tasks_evaluated", [])

    # =========== Write Methods ===========

    def add_generation(self, record):
        self.add_record("generations", asdict(record) if not isinstance(record, dict) else record)

    def add_evaluation(self, record):
        self.add_record("evaluations", asdict(record) if not isinstance(record, dict) else record)

    def add_record(self, kind, record):
        """Append-only, content-checked records under existing atomic merge lock."""
        if kind not in {"generations", "evaluations"}:
            raise ValueError("Unknown record kind")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path.with_suffix(".lock"), "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._data = self._reload()
                self.get_results_summary()
                existing = [r for k in ("generations", "evaluations") for r in self._data[k] if r["record_id"] == record.get("record_id")]
                if existing:
                    if len(existing) == 1 and digest(existing[0]) == digest(record) and existing[0] in self._data[kind]:
                        return
                    raise ValueError("Conflicting record ID")
                candidate = {**self._data, kind: [*self._data[kind], record]}
                project_report(candidate, self.assignment_manifest)
                self._data = candidate
                self._save_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save_unlocked(self) -> None:
        """Save database to disk (assumes lock is already held)."""
        self._data["last_updated"] = datetime.now().isoformat()
        self._update_metadata()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first (atomic write pattern)
        fd, temp_path = tempfile.mkstemp(
            dir=self.db_path.parent, prefix=".benchmark_results_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            # Atomic rename (works on POSIX systems)
            os.rename(temp_path, self.db_path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def remove_failed_generations(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def remove_failed_evaluations(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def add_generations_batch(self, records):
        for record in records:
            self.add_record("generations", asdict(record) if not isinstance(record, dict) else record)

    def add_evaluations_batch(self, records):
        for record in records:
            self.add_record("evaluations", asdict(record) if not isinstance(record, dict) else record)

    # =========== Analysis Methods ===========

    def get_aggregated_score(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def get_score_breakdown(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def get_results_summary(self, assignment_manifest=None):
        """Fixed-assignment offline projection; legacy success-only ranks disabled."""
        return project_report(self._data, assignment_manifest or self.assignment_manifest)

    def generate_leaderboard_md(self, assignment_manifest=None):
        report = self.get_results_summary(assignment_manifest)
        return "# Measurement report (unvalidated diagnostics)\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    def get_task_analysis(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def generate_task_analysis_md(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")

    def print_status(self):
        print(json.dumps({"protocol": self._data.get("protocol", "legacy-unqualified"),
                          "generation_rows": len(self._data["generations"]),
                          "evaluation_rows": len(self._data["evaluations"]),
                          "measurement": "Explicit repaired assignment manifest required"}, indent=2))

    def rebuild_from_yaml(self, *args, **kwargs):
        raise ValueError("Legacy mutation/cache/ranking disabled; use explicit assignment manifest and append-only records")


# =========== Helper Functions ===========


def get_missing_work(
    gen_models: list[str],
    eval_models: list[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """
    Get missing generations and evaluations for given models.

    Returns:
        (missing_generations, missing_evaluations)
        - generations: list of (task_id, model) tuples
        - evaluations: list of (task_id, gen_model, eval_model) tuples
    """
    db = ResultsDatabase()
    tasks = load_all_tasks()
    task_ids = [t["task_id"] for t in tasks]

    missing_gens = db.get_missing_generations(task_ids, gen_models)
    missing_evals = db.get_missing_evaluations(gen_models, eval_models)

    return missing_gens, missing_evals


if __name__ == "__main__":
    # Quick test
    db = ResultsDatabase()
    db.print_status()
    print("\n" + db.generate_leaderboard_md())
