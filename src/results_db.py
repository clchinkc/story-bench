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

    def __init__(self, db_path: str | Path | None = None):
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

    def _save(self) -> None:
        """
        Save database to disk with file locking and atomic write.

        Uses:
        1. File locking (fcntl) to prevent concurrent writes
        2. Atomic write (temp file + rename) to prevent corruption
        """
        self._data["last_updated"] = datetime.now().isoformat()
        self._update_metadata()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create lock file path
        lock_path = self.db_path.with_suffix(".lock")

        # Acquire exclusive lock
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
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
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _update_metadata(self) -> None:
        """Update metadata based on current data."""
        models = set()
        tasks = set()
        gen_cost = 0.0
        eval_cost = 0.0

        for g in self._data["generations"]:
            models.add(g["model"])
            tasks.add(g["task_id"])
            gen_cost += g.get("generation_cost", 0)

        for e in self._data["evaluations"]:
            eval_cost += e.get("evaluation_cost", 0)

        self._data["metadata"] = {
            "total_generation_cost": gen_cost,
            "total_evaluation_cost": eval_cost,
            "total_cost": gen_cost + eval_cost,
            "models_evaluated": sorted(models),
            "tasks_evaluated": sorted(tasks),
            "generation_count": len(self._data["generations"]),
            "evaluation_count": len(self._data["evaluations"]),
        }

    # =========== Query Methods ===========

    def get_existing_generations(self) -> set[tuple[str, str]]:
        """Get set of (task_id, model) tuples that have been generated."""
        return {
            (g["task_id"], g["model"])
            for g in self._data["generations"]
            if g.get("success", True)
        }

    def get_existing_evaluations(self) -> set[tuple[str, str, str]]:
        """Get set of (task_id, gen_model, eval_model) tuples that have been evaluated."""
        return {
            (e["task_id"], e["model"], e["evaluator_model"])
            for e in self._data["evaluations"]
            if e.get("success", True)
        }

    def get_missing_generations(
        self,
        task_ids: list[str],
        models: list[str],
    ) -> list[tuple[str, str]]:
        """Get list of (task_id, model) tuples that need generation."""
        existing = self.get_existing_generations()
        missing = []
        for task_id in task_ids:
            for model in models:
                if (task_id, model) not in existing:
                    missing.append((task_id, model))
        return missing

    def get_missing_evaluations(
        self,
        gen_models: list[str],
        eval_models: list[str],
    ) -> list[tuple[str, str, str]]:
        """Get (task_id, gen_model, eval_model) tuples that need evaluation."""
        existing_evals = self.get_existing_evaluations()
        missing = []

        for g in self._data["generations"]:
            if not g.get("success", True):
                continue
            if g["model"] not in gen_models:
                continue
            # Check each evaluator model
            for eval_model in eval_models:
                key = (g["task_id"], g["model"], eval_model)
                if key not in existing_evals:
                    missing.append(key)

        return missing

    def get_generation(self, task_id: str, model: str, sample: int = 0) -> dict | None:
        """Get a specific generation record."""
        for g in self._data["generations"]:
            if (
                g["task_id"] == task_id
                and g["model"] == model
                and g["sample"] == sample
            ):
                return g
        return None

    def get_models(self) -> list[str]:
        """Get list of all models that have been evaluated."""
        return self._data["metadata"].get("models_evaluated", [])

    def get_tasks(self) -> list[str]:
        """Get list of all tasks that have been evaluated."""
        return self._data["metadata"].get("tasks_evaluated", [])

    # =========== Write Methods ===========

    def add_generation(self, record: GenerationRecord) -> None:
        """
        Add a generation record with concurrent-safe merge.

        Reloads data from disk before merging to prevent data loss
        from concurrent processes.
        """
        # Create lock file path
        lock_path = self.db_path.with_suffix(".lock")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Reload latest data from disk
                self._data = self._reload()

                # Remove existing record for same combo if exists (for re-runs)
                self._data["generations"] = [
                    g
                    for g in self._data["generations"]
                    if not (
                        g["task_id"] == record.task_id
                        and g["model"] == record.model
                        and g["sample"] == record.sample
                    )
                ]
                self._data["generations"].append(asdict(record))

                # Save without re-acquiring lock (we already hold it)
                self._save_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def add_evaluation(self, record: EvaluationRecord) -> None:
        """
        Add an evaluation record with concurrent-safe merge.

        Reloads data from disk before merging to prevent data loss
        from concurrent processes.
        """
        lock_path = self.db_path.with_suffix(".lock")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Reload latest data from disk
                self._data = self._reload()

                # Remove existing record for same combo if exists
                self._data["evaluations"] = [
                    e
                    for e in self._data["evaluations"]
                    if not (
                        e["task_id"] == record.task_id
                        and e["model"] == record.model
                        and e["evaluator_model"] == record.evaluator_model
                    )
                ]
                self._data["evaluations"].append(asdict(record))

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

    def remove_failed_generations(self, model: str | None = None) -> int:
        """
        Remove failed generations (to allow retry).

        Args:
            model: If provided, only remove failed generations for this model.
                   If None, remove all failed generations.

        Returns: Number of failed generations removed.
        """
        lock_path = self.db_path.with_suffix(".lock")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._data = self._reload()
                before_count = len(self._data["generations"])

                # Remove failed generations
                if model:
                    self._data["generations"] = [
                        g
                        for g in self._data["generations"]
                        if not (g["model"] == model and not g.get("success", True))
                    ]
                else:
                    self._data["generations"] = [
                        g for g in self._data["generations"] if g.get("success", True)
                    ]

                removed = before_count - len(self._data["generations"])
                if removed > 0:
                    self._save_unlocked()
                return removed
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def remove_failed_evaluations(self, model: str | None = None) -> int:
        """
        Remove failed evaluations (to allow retry).

        Args:
            model: If provided, only remove failed evaluations for this gen model.
                   If None, remove all failed evaluations.

        Returns: Number of failed evaluations removed.
        """
        lock_path = self.db_path.with_suffix(".lock")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._data = self._reload()
                before_count = len(self._data["evaluations"])

                # Remove failed evaluations
                if model:
                    self._data["evaluations"] = [
                        e
                        for e in self._data["evaluations"]
                        if not (e["model"] == model and not e.get("success", True))
                    ]
                else:
                    self._data["evaluations"] = [
                        e for e in self._data["evaluations"] if e.get("success", True)
                    ]

                removed = before_count - len(self._data["evaluations"])
                if removed > 0:
                    self._save_unlocked()
                return removed
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def add_generations_batch(self, records: list[GenerationRecord]) -> None:
        """Add multiple generation records efficiently with locking."""
        lock_path = self.db_path.with_suffix(".lock")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._data = self._reload()
                for record in records:
                    self._data["generations"] = [
                        g
                        for g in self._data["generations"]
                        if not (
                            g["task_id"] == record.task_id
                            and g["model"] == record.model
                            and g["sample"] == record.sample
                        )
                    ]
                    self._data["generations"].append(asdict(record))
                self._save_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def add_evaluations_batch(self, records: list[EvaluationRecord]) -> None:
        """Add multiple evaluation records efficiently with locking."""
        lock_path = self.db_path.with_suffix(".lock")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._data = self._reload()
                for record in records:
                    self._data["evaluations"] = [
                        e
                        for e in self._data["evaluations"]
                        if not (
                            e["task_id"] == record.task_id
                            and e["model"] == record.model
                            and e["sample"] == record.sample
                        )
                    ]
                    self._data["evaluations"].append(asdict(record))
                self._save_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # =========== Analysis Methods ===========

    def get_aggregated_score(self, task_id: str, gen_model: str) -> float | None:
        """
        Get aggregated final score for a task/model combo using median.
        Returns None if no evaluations with scores exist.
        """
        import statistics

        evals = [
            e
            for e in self._data["evaluations"]
            if e["task_id"] == task_id
            and e["model"] == gen_model
            and e.get("success", True)
            and e.get("final_score") is not None
        ]
        if not evals:
            return None

        scores = [e["final_score"] for e in evals]
        return statistics.median(scores)

    def get_score_breakdown(
        self, task_id: str, gen_model: str
    ) -> dict[str, float] | None:
        """
        Get aggregated score breakdown for a task/model combo.
        Returns median of each score component.

        For standard tasks: word_count, programmatic, llm_judge
        For agentic tasks: process, output (averaged sub-scores)
        """
        import statistics

        evals = [
            e
            for e in self._data["evaluations"]
            if e["task_id"] == task_id
            and e["model"] == gen_model
            and e.get("success", True)
            and e.get("score_breakdown") is not None
        ]
        if not evals:
            return None

        # Check if these are agentic evaluations
        agentic_evals = [
            e for e in evals if e.get("score_breakdown", {}).get("agentic", False)
        ]

        if agentic_evals:
            # Agentic evaluations: aggregate process_scores and output_scores
            process_scores = []
            output_scores = []

            for e in agentic_evals:
                breakdown = e.get("score_breakdown", {})
                proc = breakdown.get("process_scores", {})
                out = breakdown.get("output_scores", {})

                # Average all process scores
                if proc:
                    proc_vals = [
                        v for v in proc.values() if isinstance(v, (int, float))
                    ]
                    if proc_vals:
                        process_scores.append(statistics.mean(proc_vals))

                # Average all output scores
                if out:
                    out_vals = [v for v in out.values() if isinstance(v, (int, float))]
                    if out_vals:
                        output_scores.append(statistics.mean(out_vals))

            return {
                "process": statistics.median(process_scores)
                if process_scores
                else None,
                "output": statistics.median(output_scores) if output_scores else None,
                # For compatibility with leaderboard display
                "llm_judge": statistics.median(output_scores)
                if output_scores
                else None,
            }

        # Standard evaluations: word_count, programmatic, llm_judge
        standard_evals = [
            e
            for e in evals
            if e.get("score_breakdown", {}).get("components") is not None
        ]
        if not standard_evals:
            return None

        # Aggregate each component from unified format
        # Format: components.programmatic.breakdown.{word_count_score, repetition_score, slop_score}
        wc_scores = []
        prog_scores = []
        llm_scores = []

        for e in standard_evals:
            components = e["score_breakdown"]["components"]

            # Extract word count score from programmatic breakdown
            wc_score = components["programmatic"]["breakdown"].get(
                "word_count_score", 0.0
            )
            wc_scores.append(wc_score)

            # Extract programmatic score
            prog_scores.append(components["programmatic"].get("score", 0.0))

            # Extract LLM judge score
            llm_scores.append(components["llm_judge"].get("score", 0.0))

        return {
            "word_count": statistics.median(wc_scores) if wc_scores else 0.0,
            "programmatic": statistics.median(prog_scores) if prog_scores else 0.0,
            "llm_judge": statistics.median(llm_scores) if llm_scores else 0.0,
        }

    def get_results_summary(self) -> dict[str, Any]:
        """Get summary statistics for leaderboard with scores."""
        import statistics

        # Get unique evaluators
        evaluators = set(
            e["evaluator_model"]
            for e in self._data["evaluations"]
            if e.get("success", True)
        )

        # Calculate per-model generation costs and token statistics
        model_gen_costs: dict[str, float] = {}
        model_gen_counts: dict[str, int] = {}
        model_token_stats: dict[str, dict[str, int]] = {}
        for g in self._data["generations"]:
            if not g.get("success", True):
                continue
            model = g["model"]
            cost = g.get("generation_cost", 0)
            model_gen_costs[model] = model_gen_costs.get(model, 0) + cost
            model_gen_counts[model] = model_gen_counts.get(model, 0) + 1

            # Track token statistics
            if model not in model_token_stats:
                model_token_stats[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "output_tokens": 0,
                }
            model_token_stats[model]["prompt_tokens"] += g.get("prompt_tokens", 0)
            completion = g.get("completion_tokens", 0)
            reasoning = g.get("reasoning_tokens", 0)
            model_token_stats[model]["completion_tokens"] += completion
            model_token_stats[model]["reasoning_tokens"] += reasoning
            model_token_stats[model]["output_tokens"] += completion - reasoning

        # Group evaluations by (gen_model, task_id)
        task_types_map: dict[tuple[str, str], str] = {}
        for e in self._data["evaluations"]:
            if not e.get("success", True):
                continue
            key = (e["model"], e["task_id"])
            task_types_map[key] = e.get("task_type", "unknown")

        # Calculate aggregated scores and breakdowns for each (model, task)
        aggregated_score: dict[tuple[str, str], float | None] = {}
        aggregated_breakdown: dict[tuple[str, str], dict | None] = {}
        for key in task_types_map.keys():
            gen_model, task_id = key
            aggregated_score[key] = self.get_aggregated_score(task_id, gen_model)
            aggregated_breakdown[key] = self.get_score_breakdown(task_id, gen_model)

        # Group by model
        by_model: dict[str, dict[str, Any]] = {}
        for gen_model, task_id in task_types_map.keys():
            if gen_model not in by_model:
                by_model[gen_model] = {"tasks": {}}
            task_type = task_types_map[(gen_model, task_id)]
            by_model[gen_model]["tasks"][task_id] = {
                "score": aggregated_score[(gen_model, task_id)],
                "breakdown": aggregated_breakdown[(gen_model, task_id)],
                "task_type": task_type,
            }

        # Calculate stats per model
        model_stats = {}
        for model, data in by_model.items():
            tasks = data["tasks"]
            total = len(tasks)

            # Calculate scores
            task_scores = [t["score"] for t in tasks.values() if t["score"] is not None]
            avg_score = statistics.mean(task_scores) if task_scores else None

            # Calculate component averages (handle both standard and agentic breakdowns)
            wc_scores = []
            prog_scores = []
            llm_scores = []
            for t in tasks.values():
                if t["breakdown"]:
                    # Standard tasks have word_count, programmatic, llm_judge
                    # Agentic tasks have process, output, llm_judge
                    if "word_count" in t["breakdown"]:
                        wc_scores.append(t["breakdown"]["word_count"])
                    if "programmatic" in t["breakdown"]:
                        prog_scores.append(t["breakdown"]["programmatic"])
                    if (
                        "llm_judge" in t["breakdown"]
                        and t["breakdown"]["llm_judge"] is not None
                    ):
                        llm_scores.append(t["breakdown"]["llm_judge"])

            avg_components = {
                "word_count": statistics.mean(wc_scores) if wc_scores else None,
                "programmatic": statistics.mean(prog_scores) if prog_scores else None,
                "llm_judge": statistics.mean(llm_scores) if llm_scores else None,
            }

            # By task type (with scores)
            by_type: dict[str, dict] = {}
            for task_id, task_data in tasks.items():
                tt = task_data["task_type"]
                if tt not in by_type:
                    by_type[tt] = {"total": 0, "scores": [], "breakdowns": []}
                by_type[tt]["total"] += 1
                if task_data["score"] is not None:
                    by_type[tt]["scores"].append(task_data["score"])
                if task_data["breakdown"] is not None:
                    by_type[tt]["breakdowns"].append(task_data["breakdown"])

            # Cost and token metrics
            gen_cost = model_gen_costs.get(model, 0)
            gen_count = model_gen_counts.get(model, 0)
            avg_cost = gen_cost / gen_count if gen_count > 0 else 0
            token_stats = model_token_stats.get(
                model,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "output_tokens": 0,
                },
            )

            # Build by_task_type with safe component extraction
            by_task_type_stats = {}
            for tt, d in by_type.items():
                # Safely extract component scores (handle missing keys)
                wc_vals = [
                    b["word_count"]
                    for b in d["breakdowns"]
                    if "word_count" in b and b["word_count"] is not None
                ]
                prog_vals = [
                    b["programmatic"]
                    for b in d["breakdowns"]
                    if "programmatic" in b and b["programmatic"] is not None
                ]
                llm_vals = [
                    b["llm_judge"]
                    for b in d["breakdowns"]
                    if "llm_judge" in b and b["llm_judge"] is not None
                ]

                by_task_type_stats[tt] = {
                    "total": d["total"],
                    "avg_score": statistics.mean(d["scores"]) if d["scores"] else None,
                    "avg_components": {
                        "word_count": statistics.mean(wc_vals) if wc_vals else None,
                        "programmatic": statistics.mean(prog_vals)
                        if prog_vals
                        else None,
                        "llm_judge": statistics.mean(llm_vals) if llm_vals else None,
                    }
                    if d["breakdowns"]
                    else None,
                }

            model_stats[model] = {
                "total": total,
                "avg_score": avg_score,
                "avg_components": avg_components,
                "generation_cost": gen_cost,
                "avg_cost_per_task": avg_cost,
                "token_stats": token_stats,
                "by_task_type": by_task_type_stats,
            }

        # Get all task types
        all_task_types = set()
        for stats in model_stats.values():
            all_task_types.update(stats["by_task_type"].keys())

        return {
            "models": model_stats,
            "task_types": sorted(all_task_types),
            "evaluators": sorted(evaluators),
            "metadata": self._data["metadata"],
            "last_updated": self._data["last_updated"],
        }

    def generate_leaderboard_md(self) -> str:
        """Generate markdown leaderboard with scores, component breakdown, and quality-to-cost metrics."""
        summary = self.get_results_summary()

        lines = ["# Story Theory Benchmark Leaderboard\n"]
        lines.append(f"*Last updated: {summary['last_updated']}*\n")

        # Metadata
        meta = summary["metadata"]
        evaluators = summary.get("evaluators", [])
        lines.append("## Overview\n")
        lines.append(f"- **Models evaluated**: {len(meta.get('models_evaluated', []))}")
        lines.append(f"- **Tasks**: {len(meta.get('tasks_evaluated', []))}")
        lines.append(
            f"- **Evaluator models**: {', '.join(e.split('/')[-1] for e in evaluators) if evaluators else 'None'}"
        )
        lines.append("- **Aggregation**: Median across evaluators")
        lines.append("- **Scoring**: Programmatic (50%) + LLM Judge (50%)")
        lines.append(f"- **Total generations**: {meta.get('generation_count', 0)}")
        lines.append(f"- **Total evaluations**: {meta.get('evaluation_count', 0)}")
        lines.append(f"- **Total cost**: ${meta.get('total_cost', 0):.4f}\n")

        # Model rankings
        if summary["models"]:
            lines.append("## Model Rankings\n")
            lines.append(
                "| Rank | Model | Company | Score | Gen Cost | Value | LLM Judge |"
            )
            lines.append(
                "|------|-------|---------|-------|----------|---------|-----------|"
            )

            # Sort by score
            sorted_models = sorted(
                summary["models"].items(),
                key=lambda x: x[1].get("avg_score") or 0,
                reverse=True,
            )

            # Company mapping
            company_names = {
                "anthropic": "Anthropic",
                "openai": "OpenAI",
                "google": "Google",
                "deepseek": "DeepSeek",
                "meta-llama": "Meta",
                "x-ai": "xAI",
                "qwen": "Alibaba",
                "mistralai": "Mistral",
                "minimax": "MiniMax",
            }

            for rank, (model, stats) in enumerate(sorted_models, 1):
                model_short = model.split("/")[-1] if "/" in model else model
                company = model.split("/")[0] if "/" in model else "unknown"
                company_name = company_names.get(company, company.title())
                gen_cost = stats.get("generation_cost", 0)
                avg_score = stats.get("avg_score")
                score_str = f"{avg_score:.1%}" if avg_score is not None else "-"

                # Quality-to-cost ratio: Score²/Cost (rewards quality quadratically)
                if avg_score is not None and gen_cost > 0:
                    score_per_dollar = (avg_score**2 * 100) / gen_cost
                    ratio_str = f"{score_per_dollar:.1f}"
                else:
                    ratio_str = "-"

                # Component scores
                comps = stats.get("avg_components", {})
                llm_str = (
                    f"{comps.get('llm_judge'):.1%}"
                    if comps.get("llm_judge") is not None
                    else "-"
                )

                lines.append(
                    f"| {rank} | {model_short} | {company_name} | {score_str} | ${gen_cost:.4f} | {ratio_str} | {llm_str} |"
                )

            # Best Value Rankings (sorted by Score²/$)
            lines.append("\n## Best Value (Score²/Cost)\n")
            lines.append(
                "*Higher = better value. Formula: Score² / Cost (rewards quality quadratically)*\n"
            )
            lines.append("| Rank | Model | Company | Score | Gen Cost | Value |")
            lines.append("|------|-------|---------|-------|----------|-------|")

            # Calculate Score²/Cost for each model and sort by it
            value_rankings = []
            for model, stats in sorted_models:
                avg_score = stats.get("avg_score")
                gen_cost = stats.get("generation_cost", 0)
                if avg_score is not None and gen_cost > 0:
                    score_per_dollar = (avg_score**2 * 100) / gen_cost
                    value_rankings.append((model, stats, score_per_dollar))

            value_rankings.sort(key=lambda x: x[2], reverse=True)

            for rank, (model, stats, score_per_dollar) in enumerate(value_rankings, 1):
                model_short = model.split("/")[-1] if "/" in model else model
                company = model.split("/")[0] if "/" in model else "unknown"
                company_name = company_names.get(company, company.title())
                gen_cost = stats.get("generation_cost", 0)
                avg_score = stats.get("avg_score")
                score_str = f"{avg_score:.1%}" if avg_score is not None else "-"

                lines.append(
                    f"| {rank} | {model_short} | {company_name} | {score_str} | ${gen_cost:.4f} | {score_per_dollar:.1f} |"
                )

            # Scores by Task Type
            lines.append("\n## Scores by Task Type\n")
            task_types = summary["task_types"]

            header = "| Model | " + " | ".join(task_types) + " |"
            sep = "|-------|" + "|".join(["-------"] * len(task_types)) + "|"
            lines.append(header)
            lines.append(sep)

            for model, stats in sorted_models:
                model_short = model.split("/")[-1] if "/" in model else model
                row = f"| {model_short} |"
                for tt in task_types:
                    if tt in stats["by_task_type"]:
                        avg_score = stats["by_task_type"][tt].get("avg_score")
                        if avg_score is not None:
                            row += f" {avg_score:.1%} |"
                        else:
                            row += " - |"
                    else:
                        row += " - |"
                lines.append(row)

            # Component breakdown by task type
            lines.append("\n## Component Breakdown by Task Type\n")
            for model, stats in sorted_models:
                model_short = model.split("/")[-1] if "/" in model else model
                lines.append(f"\n### {model_short}\n")
                lines.append("| Task Type | Score | Programmatic | LLM Judge |")
                lines.append("|-----------|-------|--------------|-----------|")

                for tt in task_types:
                    if tt in stats["by_task_type"]:
                        tt_stats = stats["by_task_type"][tt]
                        score = tt_stats.get("avg_score")
                        score_str = f"{score:.1%}" if score is not None else "-"

                        comps = tt_stats.get("avg_components", {}) or {}
                        prog_str = (
                            f"{comps.get('programmatic'):.1%}"
                            if comps.get("programmatic") is not None
                            else "-"
                        )
                        llm_str = (
                            f"{comps.get('llm_judge'):.1%}"
                            if comps.get("llm_judge") is not None
                            else "-"
                        )

                        lines.append(f"| {tt} | {score_str} | {prog_str} | {llm_str} |")

            # Cost Efficiency section (integrates token usage with cost)
            lines.append("\n## Cost Efficiency\n")
            lines.append(
                "*Note: Reasoning tokens (for CoT models) are billed but don't produce output, affecting cost efficiency.*\n"
            )
            lines.append(
                "| Model | Gen Cost | Output Tokens | Reasoning % | $/1K Output |"
            )
            lines.append(
                "|-------|----------|---------------|-------------|-------------|"
            )

            for model, stats in sorted_models:
                model_short = model.split("/")[-1] if "/" in model else model
                token_stats = stats.get("token_stats", {})
                gen_cost = stats.get("generation_cost", 0)
                output = token_stats.get("output_tokens", 0)
                reasoning = token_stats.get("reasoning_tokens", 0)
                completion = token_stats.get("completion_tokens", 0)
                reasoning_pct = (reasoning / completion * 100) if completion > 0 else 0
                cost_per_1k_output = (gen_cost / output * 1000) if output > 0 else 0

                lines.append(
                    f"| {model_short} | ${gen_cost:.4f} | {output:,} | {reasoning_pct:.1f}% | ${cost_per_1k_output:.4f} |"
                )

        else:
            lines.append("\n*No evaluations yet. Run the benchmark to see results.*\n")

        return "\n".join(lines)

    def get_task_analysis(self) -> dict[str, Any]:
        """Get detailed per-task analysis for TASK_ANALYSIS.md."""
        import statistics
        from utils import load_all_tasks

        tasks = load_all_tasks()
        task_info = {t["task_id"]: t for t in tasks}

        # Group evaluations by task_id
        task_evals: dict[str, list[dict]] = {}
        for e in self._data["evaluations"]:
            if not e.get("success", True) or e.get("final_score") is None:
                continue
            task_id = e["task_id"]
            if task_id not in task_evals:
                task_evals[task_id] = []
            task_evals[task_id].append(e)

        # Group generations by task_id for cost analysis
        task_gens: dict[str, list[dict]] = {}
        for g in self._data["generations"]:
            if not g.get("success", True):
                continue
            task_id = g["task_id"]
            if task_id not in task_gens:
                task_gens[task_id] = []
            task_gens[task_id].append(g)

        # Calculate per-task stats
        task_stats: dict[str, dict] = {}
        for task_id, evals in task_evals.items():
            # Get per-model median scores (aggregate across evaluators)
            model_scores: dict[str, list[float]] = {}
            for e in evals:
                model = e["model"]
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(e["final_score"])

            # Compute median per model
            model_medians = {
                m: statistics.median(scores) for m, scores in model_scores.items()
            }

            all_scores = list(model_medians.values())
            if not all_scores:
                continue

            avg_score = statistics.mean(all_scores)
            min_score = min(all_scores)
            max_score = max(all_scores)
            spread = max_score - min_score
            std_dev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0

            # Best and worst models
            sorted_models = sorted(
                model_medians.items(), key=lambda x: x[1], reverse=True
            )
            best_model = sorted_models[0] if sorted_models else (None, None)
            worst_model = sorted_models[-1] if sorted_models else (None, None)

            # Cost analysis
            gens = task_gens.get(task_id, [])
            total_cost = sum(g.get("generation_cost", 0) for g in gens)
            avg_cost = total_cost / len(gens) if gens else 0

            # Get task metadata
            info = task_info.get(task_id, {})

            task_stats[task_id] = {
                "task_type": info.get(
                    "task_type", evals[0].get("task_type", "unknown")
                ),
                "theory": info.get("theory", "Unknown"),
                "models_evaluated": len(model_medians),
                "evaluations_count": len(evals),
                "avg_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "spread": spread,
                "std_dev": std_dev,
                "best_model": best_model[0],
                "best_score": best_model[1],
                "worst_model": worst_model[0],
                "worst_score": worst_model[1],
                "total_cost": total_cost,
                "avg_cost": avg_cost,
                "model_scores": model_medians,
            }

        return task_stats

    def generate_task_analysis_md(self) -> str:
        """Generate TASK_ANALYSIS.md with detailed per-task metrics."""
        import statistics

        task_stats = self.get_task_analysis()
        if not task_stats:
            return "# Task Analysis\n\n*No data available.*"

        lines = ["# Story Theory Benchmark - Task Analysis\n"]
        lines.append(f"*Last updated: {self._data.get('last_updated', 'unknown')}*\n")

        # Overview
        lines.append("## Overview\n")
        lines.append(
            "This document provides detailed analysis of each benchmark task, including:"
        )
        lines.append(
            "- **Score Spread**: Difference between best and worst model (higher = more discriminative)"
        )
        lines.append("- **Average Score**: Mean score across all models")
        lines.append(
            "- **Best/Worst Models**: Which models excel or struggle on each task"
        )
        lines.append("- **Cost Analysis**: Generation costs per task\n")

        # Task Type Summary
        lines.append("## Task Type Summary\n")
        lines.append(
            "| Task Type | Tasks | Avg Score | Avg Spread | Best Spread | Discriminative Power |"
        )
        lines.append(
            "|-----------|-------|-----------|------------|-------------|---------------------|"
        )

        # Group by task type
        by_type: dict[str, list[dict]] = {}
        for task_id, stats in task_stats.items():
            tt = stats["task_type"]
            if tt not in by_type:
                by_type[tt] = []
            by_type[tt].append({"task_id": task_id, **stats})

        type_summary = []
        for tt, tasks in sorted(by_type.items()):
            avg_score = statistics.mean([t["avg_score"] for t in tasks])
            avg_spread = statistics.mean([t["spread"] for t in tasks])
            best_spread = max(t["spread"] for t in tasks)

            # Discriminative power rating
            if avg_spread >= 0.20:
                power = "Excellent"
            elif avg_spread >= 0.15:
                power = "Good"
            elif avg_spread >= 0.10:
                power = "Moderate"
            else:
                power = "Low"

            type_summary.append(
                (tt, len(tasks), avg_score, avg_spread, best_spread, power)
            )

        # Sort by discriminative power (avg_spread)
        type_summary.sort(key=lambda x: x[3], reverse=True)

        for tt, count, avg_score, avg_spread, best_spread, power in type_summary:
            lines.append(
                f"| {tt} | {count} | {avg_score:.1%} | {avg_spread:.1%} | {best_spread:.1%} | {power} |"
            )

        # Most Discriminative Tasks (Best for benchmarking)
        lines.append("\n## Most Discriminative Tasks\n")
        lines.append(
            "*Tasks with highest score spread - best for distinguishing model capabilities*\n"
        )
        lines.append(
            "| Rank | Task ID | Type | Spread | Avg Score | Best Model | Worst Model |"
        )
        lines.append(
            "|------|---------|------|--------|-----------|------------|-------------|"
        )

        sorted_by_spread = sorted(
            task_stats.items(), key=lambda x: x[1]["spread"], reverse=True
        )
        for rank, (task_id, stats) in enumerate(sorted_by_spread[:10], 1):
            best = stats["best_model"].split("/")[-1] if stats["best_model"] else "-"
            worst = stats["worst_model"].split("/")[-1] if stats["worst_model"] else "-"
            lines.append(
                f"| {rank} | {task_id} | {stats['task_type']} | {stats['spread']:.1%} | "
                f"{stats['avg_score']:.1%} | {best} | {worst} |"
            )

        # Hardest Tasks (Lowest average score)
        lines.append("\n## Hardest Tasks\n")
        lines.append("*Tasks with lowest average scores across models*\n")
        lines.append(
            "| Rank | Task ID | Type | Avg Score | Spread | Best Score | Best Model |"
        )
        lines.append(
            "|------|---------|------|-----------|--------|------------|------------|"
        )

        sorted_by_difficulty = sorted(
            task_stats.items(), key=lambda x: x[1]["avg_score"]
        )
        for rank, (task_id, stats) in enumerate(sorted_by_difficulty[:10], 1):
            best = stats["best_model"].split("/")[-1] if stats["best_model"] else "-"
            lines.append(
                f"| {rank} | {task_id} | {stats['task_type']} | {stats['avg_score']:.1%} | "
                f"{stats['spread']:.1%} | {stats['best_score']:.1%} | {best} |"
            )

        # Easiest Tasks (Highest average score)
        lines.append("\n## Easiest Tasks\n")
        lines.append(
            "*Tasks with highest average scores across models (potential ceiling effects)*\n"
        )
        lines.append(
            "| Rank | Task ID | Type | Avg Score | Spread | Worst Score | Worst Model |"
        )
        lines.append(
            "|------|---------|------|-----------|--------|-------------|-------------|"
        )

        sorted_by_ease = sorted(
            task_stats.items(), key=lambda x: x[1]["avg_score"], reverse=True
        )
        for rank, (task_id, stats) in enumerate(sorted_by_ease[:10], 1):
            worst = stats["worst_model"].split("/")[-1] if stats["worst_model"] else "-"
            lines.append(
                f"| {rank} | {task_id} | {stats['task_type']} | {stats['avg_score']:.1%} | "
                f"{stats['spread']:.1%} | {stats['worst_score']:.1%} | {worst} |"
            )

        # Detailed Per-Task Analysis
        lines.append("\n## Detailed Task Breakdown\n")

        for tt in sorted(by_type.keys()):
            lines.append(f"\n### {tt}\n")

            tasks = by_type[tt]
            tasks.sort(key=lambda x: x["spread"], reverse=True)

            lines.append(
                "| Task ID | Theory | Avg | Min | Max | Spread | Std Dev | Best Model | Worst Model |"
            )
            lines.append(
                "|---------|--------|-----|-----|-----|--------|---------|------------|-------------|"
            )

            for t in tasks:
                best = t["best_model"].split("/")[-1] if t["best_model"] else "-"
                worst = t["worst_model"].split("/")[-1] if t["worst_model"] else "-"
                lines.append(
                    f"| {t['task_id']} | {t['theory']} | {t['avg_score']:.1%} | "
                    f"{t['min_score']:.1%} | {t['max_score']:.1%} | {t['spread']:.1%} | "
                    f"{t['std_dev']:.2f} | {best} | {worst} |"
                )

        # Cost Analysis
        lines.append("\n## Cost Analysis\n")
        lines.append("*Generation cost per task (evaluation costs not included)*\n")
        lines.append("| Task Type | Tasks | Total Cost | Avg Cost/Task |")
        lines.append("|-----------|-------|------------|---------------|")

        for tt, tasks in sorted(by_type.items()):
            total = sum(t["total_cost"] for t in tasks)
            avg = statistics.mean([t["avg_cost"] for t in tasks])
            lines.append(f"| {tt} | {len(tasks)} | ${total:.4f} | ${avg:.4f} |")

        # Per-Task Model Scores Matrix
        lines.append("\n## Per-Task Model Scores\n")
        lines.append(
            "*Detailed scores for each model on each task (median across evaluators)*\n"
        )

        # Get all models
        all_models = set()
        for stats in task_stats.values():
            all_models.update(stats["model_scores"].keys())
        models_sorted = sorted(all_models)
        model_shorts = [m.split("/")[-1] for m in models_sorted]

        for tt in sorted(by_type.keys()):
            lines.append(f"\n### {tt}\n")

            header = "| Task ID | " + " | ".join(model_shorts) + " |"
            sep = "|---------|" + "|".join(["-----"] * len(models_sorted)) + "|"
            lines.append(header)
            lines.append(sep)

            tasks = sorted(by_type[tt], key=lambda x: x["task_id"])
            for t in tasks:
                row = f"| {t['task_id']} |"
                for m in models_sorted:
                    score = t["model_scores"].get(m)
                    if score is not None:
                        row += f" {score:.1%} |"
                    else:
                        row += " - |"
                lines.append(row)

        return "\n".join(lines)

    def print_status(self) -> None:
        """Print current benchmark status."""
        meta = self._data["metadata"]
        evaluators = set(
            e["evaluator_model"]
            for e in self._data["evaluations"]
            if e.get("success", True)
        )
        print("\n=== Story Theory Benchmark Status ===\n")
        print(f"Generation models: {len(meta.get('models_evaluated', []))}")
        for m in meta.get("models_evaluated", []):
            print(f"  - {m}")
        print(f"\nEvaluator models: {len(evaluators)}")
        for e in sorted(evaluators):
            print(f"  - {e}")
        print(f"\nTasks: {len(meta.get('tasks_evaluated', []))}")
        print(f"Generations: {meta.get('generation_count', 0)}")
        print(f"Evaluations: {meta.get('evaluation_count', 0)}")
        print(f"Total cost: ${meta.get('total_cost', 0):.4f}")
        print(f"\nLast updated: {self._data.get('last_updated', 'never')}")

    def rebuild_from_yaml(self) -> tuple[int, int]:
        """
        Rebuild database from YAML files in results/generations, results/evaluations,
        and results/agentic (for agentic task results).

        Returns: (generations_loaded, evaluations_loaded)
        """
        import yaml

        project_root = get_project_root()
        gen_dir = project_root / "results" / "generations"
        eval_dir = project_root / "results" / "evaluations"
        agentic_dir = project_root / "results" / "agentic"

        # Reset data
        self._data = self._empty_db()

        gen_count = 0
        eval_count = 0

        # Load standard generations
        if gen_dir.exists():
            for yaml_file in gen_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                    if data and data.get("task_id"):
                        # Convert to generation record format
                        gen_record = {
                            "task_id": data["task_id"],
                            "task_type": data.get("task_type", "unknown"),
                            "theory": data.get("theory", "Unknown"),
                            "model": data["model"],
                            "sample": data.get("sample_index", 0),
                            "output": data.get("output", ""),
                            "prompt_tokens": data.get("metadata", {}).get(
                                "prompt_tokens", 0
                            ),
                            "completion_tokens": data.get("metadata", {}).get(
                                "completion_tokens", 0
                            ),
                            "reasoning_tokens": data.get("metadata", {}).get(
                                "reasoning_tokens", 0
                            ),
                            "generation_cost": data.get("metadata", {}).get("cost", 0),
                            "timestamp": data.get("metadata", {}).get("timestamp", ""),
                            "success": data.get("metadata", {}).get("success", True),
                            "error": data.get("metadata", {}).get("error"),
                        }
                        self._data["generations"].append(gen_record)
                        gen_count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {yaml_file}: {e}")

        # Load agentic generations (from results/agentic/gen_*.yaml)
        if agentic_dir.exists():
            for yaml_file in agentic_dir.glob("gen_*.yaml"):
                try:
                    with open(yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                    if data and data.get("task_id"):
                        # Agentic generations have different structure
                        token_usage = data.get("token_usage", {})
                        gen_record = {
                            "task_id": data["task_id"],
                            "task_type": data.get("task_type", "unknown"),
                            "theory": data.get("theory", "Unknown"),
                            "model": data["model"],
                            "sample": data.get("sample_index", 0),
                            "output": data.get("output", ""),
                            "prompt_tokens": token_usage.get("total_prompt_tokens", 0),
                            "completion_tokens": token_usage.get(
                                "total_completion_tokens", 0
                            ),
                            "reasoning_tokens": token_usage.get(
                                "total_reasoning_tokens", 0
                            ),
                            "generation_cost": token_usage.get("total_cost", 0),
                            "timestamp": data.get("timestamp", ""),
                            "success": data.get("success", True),
                            "error": data.get("error"),
                            "agentic_type": data.get("agentic_type"),
                            "metrics": data.get("metrics", {}),
                        }
                        self._data["generations"].append(gen_record)
                        gen_count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {yaml_file}: {e}")

        # Load standard evaluations
        if eval_dir.exists():
            for yaml_file in eval_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                    if data and data.get("task_id"):
                        eval_record = {
                            "task_id": data["task_id"],
                            "task_type": data.get("task_type", "unknown"),
                            "model": data["model"],
                            "sample": data.get("sample", 0),
                            "evaluator_model": data.get("evaluator_model", "unknown"),
                            "evaluation_cost": data.get("evaluator_cost", 0),
                            "timestamp": data.get("timestamp", ""),
                            "success": data.get("success", True),
                            "final_score": data.get("final_score"),
                            "score_breakdown": data.get("score_breakdown", {}),
                            "llm_results": data.get("llm_results", {}),
                            "error": data.get("error"),
                        }
                        self._data["evaluations"].append(eval_record)
                        eval_count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {yaml_file}: {e}")

        # Load agentic evaluations (from results/agentic/eval_*.yaml)
        if agentic_dir.exists():
            for yaml_file in agentic_dir.glob("eval_*.yaml"):
                try:
                    with open(yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                    if data and data.get("task_id"):
                        # Agentic evaluations have different structure (process_scores, output_scores)
                        # We normalize to score_breakdown for consistency
                        process_scores = data.get("process_scores", {})
                        output_scores = data.get("output_scores", {})

                        # Build normalized score_breakdown
                        score_breakdown = {
                            "final_score": data.get("final_score"),
                            "agentic": True,
                            "process_scores": process_scores,
                            "output_scores": output_scores,
                        }

                        # Infer task_type from task_id if not present
                        task_id = data["task_id"]
                        task_type = data.get("task_type")
                        if not task_type:
                            # Parse from task_id like "agentic_constraint_discovery_001"
                            parts = task_id.rsplit("_", 1)
                            task_type = parts[0] if len(parts) > 1 else task_id

                        eval_record = {
                            "task_id": task_id,
                            "task_type": task_type,
                            "model": data["model"],
                            "sample": data.get("sample", 0),
                            "evaluator_model": data.get("evaluator_model", "unknown"),
                            "evaluation_cost": data.get("evaluator_cost", 0),
                            "timestamp": data.get("timestamp", ""),
                            "success": data.get("success", True),
                            "final_score": data.get("final_score"),
                            "score_breakdown": score_breakdown,
                            "llm_results": data.get("llm_results", {}),
                            "error": data.get("error"),
                            "agentic_type": data.get("agentic_type"),
                        }
                        self._data["evaluations"].append(eval_record)
                        eval_count += 1
                except Exception as e:
                    print(f"Warning: Failed to load {yaml_file}: {e}")

        # Save rebuilt database
        self._save()

        return gen_count, eval_count


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
