"""
Benchmark Comparison Module.

Enables head-to-head A/B comparison of two benchmark result files,
answering: "Did Model B improve over Model A, and on what specific tasks?"

Supports full JSON database files from results_db.py.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelResult:
    """Aggregated results for a single model on a single task."""

    model: str
    task_type: str
    theory: str
    task_id: str
    sample: int
    final_score: float
    programmatic_score: float
    llm_judge_score: float
    word_count_score: float | None = None
    repetition_score: float | None = None
    slop_score: float | None = None
    element_count_score: float | None = None


@dataclass
class ComparisonDelta:
    """Delta between two models on a single task."""

    task_id: str
    task_type: str
    theory: str
    left_score: float
    right_score: float
    delta: float
    delta_pct: float
    programmatic_delta: float | None = None
    llm_judge_delta: float | None = None
    element_count_delta: float | None = None


@dataclass
class ComparisonSummary:
    """Summary of a full model comparison."""

    left_model: str
    right_model: str
    total_tasks: int
    right_wins: int
    left_wins: int
    ties: int
    avg_delta: float
    avg_pct_improvement: float
    task_type_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    component_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_model": self.left_model,
            "right_model": self.right_model,
            "total_tasks": self.total_tasks,
            "right_wins": self.right_wins,
            "left_wins": self.left_wins,
            "ties": self.ties,
            "avg_delta": round(self.avg_delta, 4),
            "avg_pct_improvement": round(self.avg_pct_improvement, 2),
            "task_type_breakdown": self.task_type_breakdown,
            "component_breakdown": self.component_breakdown,
        }


def _extract_evaluations(data: dict[str, Any]) -> list[ModelResult]:
    """Extract flat list of ModelResult from a results database dict."""
    results = []
    for eval_record in data.get("evaluations", []):
        if not eval_record.get("success", True):
            continue
        breakdown = eval_record.get("score_breakdown", {})
        prog = breakdown.get("programmatic", {})
        results.append(
            ModelResult(
                model=eval_record["model"],
                task_type=eval_record["task_type"],
                theory=eval_record.get("theory", "Unknown"),
                task_id=eval_record["task_id"],
                sample=eval_record["sample"],
                final_score=eval_record.get("final_score", 0.0),
                programmatic_score=prog.get("score", 0.0),
                llm_judge_score=breakdown.get("llm_judge", {}).get("score", 0.0),
                word_count_score=prog.get("breakdown", {}).get("word_count_score"),
                repetition_score=prog.get("breakdown", {}).get("repetition_score"),
                slop_score=prog.get("breakdown", {}).get("slop_score"),
                element_count_score=prog.get("breakdown", {}).get("element_count_score"),
            )
        )
    return results


def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


class StoryBenchmarkComparison:
    """
    Compare two benchmark result files head-to-head.

    Accepts full JSON database files from results_db.py.

    Usage:
        comp = StoryBenchmarkComparison("results_A.json", "results_B.json")
        summary = comp.compare()
        print(comp.print_summary())
    """

    def __init__(self, left_path: str | Path, right_path: str | Path):
        self.left_path = Path(left_path)
        self.right_path = Path(right_path)
        self._left_data: dict[str, Any] | None = None
        self._right_data: dict[str, Any] | None = None

    def _load(self, path: Path) -> dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    @property
    def left_data(self) -> dict[str, Any]:
        if self._left_data is None:
            self._left_data = self._load(self.left_path)
        return self._left_data

    @property
    def right_data(self) -> dict[str, Any]:
        if self._right_data is None:
            self._right_data = self._load(self.right_path)
        return self._right_data

    def compare(self) -> ComparisonSummary:
        """
        Run the full comparison and return a structured summary.
        """
        left_evals = _extract_evaluations(self.left_data)
        right_evals = _extract_evaluations(self.right_data)

        # Build lookup: (task_id, model, sample) -> result
        left_by_key: dict[tuple[str, str, int], ModelResult] = {
            (r.task_id, r.model, r.sample): r for r in left_evals
        }
        right_by_key: dict[tuple[str, str, int], ModelResult] = {
            (r.task_id, r.model, r.sample): r for r in right_evals
        }

        left_model = left_evals[0].model if left_evals else "unknown"
        right_model = right_evals[0].model if right_evals else "unknown"

        deltas: list[ComparisonDelta] = []

        # Per-task-type accumulation: {task_type: {"left": [...], "right": [...], "deltas": [...]}}
        tt_agg: dict[str, dict[str, list[float]]] = {}

        # Component accumulation
        comp_agg: dict[str, dict[str, list[float]]] = {
            "programmatic": {"left": [], "right": []},
            "llm_judge": {"left": [], "right": []},
            "element_count": {"left": [], "right": []},
        }

        for r_key, right_res in right_by_key.items():
            left_res = left_by_key.get(r_key)
            if left_res is None:
                continue

            delta = right_res.final_score - left_res.final_score
            delta_pct = (
                (delta / left_res.final_score) * 100
                if left_res.final_score > 0
                else (100.0 if right_res.final_score > 0 else 0.0)
            )
            prog_delta = right_res.programmatic_score - left_res.programmatic_score
            llm_delta = right_res.llm_judge_score - left_res.llm_judge_score
            elem_delta = (
                (right_res.element_count_score - left_res.element_count_score)
                if right_res.element_count_score is not None
                and left_res.element_count_score is not None
                else None
            )

            deltas.append(
                ComparisonDelta(
                    task_id=right_res.task_id,
                    task_type=right_res.task_type,
                    theory=right_res.theory,
                    left_score=left_res.final_score,
                    right_score=right_res.final_score,
                    delta=delta,
                    delta_pct=delta_pct,
                    programmatic_delta=prog_delta,
                    llm_judge_delta=llm_delta,
                    element_count_delta=elem_delta,
                )
            )

            # Task type aggregation
            tt_agg.setdefault(right_res.task_type, {"left": [], "right": [], "deltas": []})
            tt_agg[right_res.task_type]["left"].append(left_res.final_score)
            tt_agg[right_res.task_type]["right"].append(right_res.final_score)
            tt_agg[right_res.task_type]["deltas"].append(delta)

            # Component aggregation
            comp_agg["programmatic"]["left"].append(left_res.programmatic_score)
            comp_agg["programmatic"]["right"].append(right_res.programmatic_score)
            comp_agg["llm_judge"]["left"].append(left_res.llm_judge_score)
            comp_agg["llm_judge"]["right"].append(right_res.llm_judge_score)
            if elem_delta is not None:
                comp_agg["element_count"]["left"].append(
                    left_res.element_count_score or 0.0
                )
                comp_agg["element_count"]["right"].append(
                    right_res.element_count_score or 0.0
                )

        total = len(deltas)
        right_wins = sum(1 for d in deltas if d.delta > 0)
        left_wins = sum(1 for d in deltas if d.delta < 0)
        ties = sum(1 for d in deltas if d.delta == 0)
        avg_delta = _mean([d.delta for d in deltas]) if deltas else 0.0
        avg_pct = _mean([d.delta_pct for d in deltas]) if deltas else 0.0

        # Task type breakdown
        task_type_breakdown: dict[str, dict[str, float]] = {}
        for tt, agg in tt_agg.items():
            task_type_breakdown[tt] = {
                "count": len(agg["left"]),
                "mean_left": _mean(agg["left"]),
                "mean_right": _mean(agg["right"]),
                "mean_delta": _mean(agg["deltas"]),
            }

        # Component breakdown
        component_breakdown: dict[str, dict[str, float]] = {}
        for comp, sides in comp_agg.items():
            if sides["left"] and sides["right"]:
                component_breakdown[comp] = {
                    "mean_left": _mean(sides["left"]),
                    "mean_right": _mean(sides["right"]),
                    "mean_delta": _mean(sides["right"]) - _mean(sides["left"]),
                }

        return ComparisonSummary(
            left_model=left_model,
            right_model=right_model,
            total_tasks=total,
            right_wins=right_wins,
            left_wins=left_wins,
            ties=ties,
            avg_delta=avg_delta,
            avg_pct_improvement=avg_pct,
            task_type_breakdown=task_type_breakdown,
            component_breakdown=component_breakdown,
        )

    def print_summary(self) -> str:
        """Generate a human-readable Markdown comparison table."""
        summary = self.compare()
        n = max(summary.total_tasks, 1)

        lines = [
            f"# Benchmark Comparison: {summary.left_model} vs {summary.right_model}",
            "",
            f"**Total tasks evaluated**: {summary.total_tasks}",
            f"| Metric | Value |",
            f"|---|---|",
            f"| **{summary.right_model} wins** | {summary.right_wins} ({summary.right_wins / n * 100:.1f}%) |",
            f"| **{summary.left_model} wins** | {summary.left_wins} ({summary.left_wins / n * 100:.1f}%) |",
            f"| **Ties** | {summary.ties} ({summary.ties / n * 100:.1f}%) |",
            f"| **Avg Delta** | {summary.avg_delta:+.4f} |",
            f"| **Avg % Improvement** | {summary.avg_pct_improvement:+.2f}% |",
            "",
            "## Task Type Breakdown",
            "",
            f"| Task Type | Count | {summary.left_model} | {summary.right_model} | Delta |",
            f"|---|---|---|---|---|",
        ]

        for tt, agg in sorted(summary.task_type_breakdown.items()):
            lines.append(
                f"| {tt} | {agg['count']} | {agg['mean_left']:.4f} | {agg['mean_right']:.4f} | {agg['mean_delta']:+.4f} |"
            )

        if summary.component_breakdown:
            lines += [
                "",
                "## Component Breakdown",
                "",
                f"| Component | {summary.left_model} | {summary.right_model} | Delta |",
                f"|---|---|---|---|",
            ]
            for comp, agg in sorted(summary.component_breakdown.items()):
                lines.append(
                    f"| {comp} | {agg['mean_left']:.4f} | {agg['mean_right']:.4f} | {agg['mean_delta']:+.4f} |"
                )

        return "\n".join(lines)
