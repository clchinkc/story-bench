"""
Analysis module for the Story Theory Benchmark.

This module provides statistical analysis, leaderboard generation,
and visualization capabilities for benchmark results.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils import (
    get_analysis_path,
    get_project_root,
    get_timestamp,
    load_yaml,
    save_json,
    save_yaml,
)


@dataclass
class ModelStats:
    """Statistics for a single model."""

    model: str
    total_samples: int
    passed: int
    failed: int
    pass_rate: float
    variance: float
    std_dev: float
    by_task_type: dict[str, dict[str, Any]]
    by_theory: dict[str, dict[str, Any]]


@dataclass
class Comparison:
    """Statistical comparison between two models."""

    model_a: str
    model_b: str
    t_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_interpretation: str


class BenchmarkAnalyzer:
    """Analyzer for benchmark results."""

    def __init__(self, evaluations_dir: str | Path | None = None):
        if evaluations_dir is None:
            evaluations_dir = get_project_root() / "results" / "evaluations"
        self.evaluations_dir = Path(evaluations_dir)
        self.evaluations: list[dict[str, Any]] = []
        self._load_evaluations()

    def _load_evaluations(self) -> None:
        """Load all evaluation files."""
        if not self.evaluations_dir.exists():
            return

        for eval_file in self.evaluations_dir.glob("*.yaml"):
            try:
                evaluation = load_yaml(eval_file)
                self.evaluations.append(evaluation)
            except Exception as e:
                print(f"Warning: Could not load {eval_file}: {e}")

    def aggregate_scores(self) -> dict[str, Any]:
        """Aggregate results by model, task type, and theory."""
        if not self.evaluations:
            return {"error": "No evaluations loaded"}

        # Group by model
        by_model: dict[str, list[bool]] = defaultdict(list)
        # Group by task type
        by_task_type: dict[str, list[bool]] = defaultdict(list)
        # Group by theory (need to look up from task)
        defaultdict(list)
        # Cross-tabulation: model × task_type
        model_task_type: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for e in self.evaluations:
            model = e.get("model", "unknown")
            task_type = e.get("task_type", "unknown")
            passed = e.get("overall_pass", False)

            by_model[model].append(passed)
            by_task_type[task_type].append(passed)
            model_task_type[model][task_type].append(passed)

        # Calculate statistics
        results = {
            "by_model": {},
            "by_task_type": {},
            "model_x_task_type": {},
            "overall": {
                "total_evaluations": len(self.evaluations),
                "passed": sum(
                    1 for e in self.evaluations if e.get("overall_pass", False)
                ),
            },
        }

        results["overall"]["pass_rate"] = (
            results["overall"]["passed"] / results["overall"]["total_evaluations"]
            if results["overall"]["total_evaluations"] > 0
            else 0
        )

        for model, passes in by_model.items():
            pass_rate = sum(passes) / len(passes) if passes else 0
            variance = np.var(passes) if len(passes) > 1 else 0
            results["by_model"][model] = {
                "total_samples": len(passes),
                "passed": sum(passes),
                "pass_rate": pass_rate,
                "variance": float(variance),
                "std_dev": float(np.std(passes)) if len(passes) > 1 else 0,
            }

        for task_type, passes in by_task_type.items():
            pass_rate = sum(passes) / len(passes) if passes else 0
            results["by_task_type"][task_type] = {
                "total_samples": len(passes),
                "passed": sum(passes),
                "pass_rate": pass_rate,
            }

        for model, task_data in model_task_type.items():
            results["model_x_task_type"][model] = {}
            for task_type, passes in task_data.items():
                pass_rate = sum(passes) / len(passes) if passes else 0
                results["model_x_task_type"][model][task_type] = {
                    "total": len(passes),
                    "passed": sum(passes),
                    "pass_rate": pass_rate,
                }

        return results

    def statistical_analysis(self) -> list[Comparison]:
        """Perform statistical analysis comparing models."""
        if not self.evaluations:
            return []

        # Group scores by model
        model_scores: dict[str, list[int]] = defaultdict(list)
        for e in self.evaluations:
            model = e.get("model", "unknown")
            passed = 1 if e.get("overall_pass", False) else 0
            model_scores[model].append(passed)

        models = list(model_scores.keys())
        comparisons = []

        # Pairwise comparisons
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                scores_a = model_scores[model_a]
                scores_b = model_scores[model_b]

                if len(scores_a) < 2 or len(scores_b) < 2:
                    continue

                # T-test
                t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

                # Cohen's d effect size
                effect_size = self._cohens_d(scores_a, scores_b)

                # Interpret effect size
                if abs(effect_size) < 0.2:
                    interpretation = "negligible"
                elif abs(effect_size) < 0.5:
                    interpretation = "small"
                elif abs(effect_size) < 0.8:
                    interpretation = "medium"
                else:
                    interpretation = "large"

                comparisons.append(
                    Comparison(
                        model_a=model_a,
                        model_b=model_b,
                        t_statistic=float(t_stat),
                        p_value=float(p_value),
                        significant=p_value < 0.05,
                        effect_size=float(effect_size),
                        effect_interpretation=interpretation,
                    )
                )

        return comparisons

    def _cohens_d(self, group1: list[int], group2: list[int]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def generate_leaderboard(self) -> str:
        """Generate a markdown leaderboard."""
        aggregated = self.aggregate_scores()

        if "error" in aggregated:
            return "No data available for leaderboard."

        lines = ["# Story Theory Benchmark Leaderboard\n"]
        lines.append(f"*Generated: {get_timestamp()}*\n")

        # Overall summary
        overall = aggregated["overall"]
        lines.append("## Overall Summary\n")
        lines.append(f"- Total Evaluations: {overall['total_evaluations']}")
        lines.append(f"- Passed: {overall['passed']}")
        lines.append(f"- Overall Pass Rate: {overall['pass_rate']:.1%}\n")

        # Model rankings
        lines.append("## Model Rankings\n")
        lines.append("| Rank | Model | Pass Rate | Passed/Total | Std Dev |")
        lines.append("|------|-------|-----------|--------------|---------|")

        sorted_models = sorted(
            aggregated["by_model"].items(),
            key=lambda x: x[1]["pass_rate"],
            reverse=True,
        )

        for rank, (model, data) in enumerate(sorted_models, 1):
            model_short = model.split("/")[-1] if "/" in model else model
            lines.append(
                f"| {rank} | {model_short} | {data['pass_rate']:.1%} | "
                f"{data['passed']}/{data['total_samples']} | {data['std_dev']:.3f} |"
            )

        # Task type breakdown
        lines.append("\n## Performance by Task Type\n")
        lines.append("| Task Type | Pass Rate | Passed/Total |")
        lines.append("|-----------|-----------|--------------|")

        for task_type, data in sorted(
            aggregated["by_task_type"].items(),
            key=lambda x: x[1]["pass_rate"],
            reverse=True,
        ):
            lines.append(
                f"| {task_type} | {data['pass_rate']:.1%} | "
                f"{data['passed']}/{data['total_samples']} |"
            )

        # Model × Task Type matrix
        lines.append("\n## Model × Task Type Matrix\n")

        task_types = list(aggregated["by_task_type"].keys())
        header = "| Model | " + " | ".join(task_types) + " |"
        separator = "|-------|" + "|".join(["-------"] * len(task_types)) + "|"

        lines.append(header)
        lines.append(separator)

        for model, task_data in aggregated["model_x_task_type"].items():
            model_short = model.split("/")[-1] if "/" in model else model
            row = f"| {model_short} |"
            for tt in task_types:
                if tt in task_data:
                    rate = task_data[tt]["pass_rate"]
                    row += f" {rate:.0%} |"
                else:
                    row += " - |"
            lines.append(row)

        # Statistical comparisons
        comparisons = self.statistical_analysis()
        if comparisons:
            lines.append("\n## Statistical Comparisons\n")
            lines.append(
                "| Comparison | t-stat | p-value | Significant | Effect Size |"
            )
            lines.append(
                "|------------|--------|---------|-------------|-------------|"
            )

            for c in comparisons:
                model_a_short = c.model_a.split("/")[-1]
                model_b_short = c.model_b.split("/")[-1]
                sig = "Yes" if c.significant else "No"
                lines.append(
                    f"| {model_a_short} vs {model_b_short} | {c.t_statistic:.3f} | "
                    f"{c.p_value:.4f} | {sig} | {c.effect_size:.3f} ({c.effect_interpretation}) |"
                )

        return "\n".join(lines)

    def generate_visualizations(
        self, output_dir: str | Path | None = None
    ) -> list[str]:
        """Generate visualization plots."""
        if output_dir is None:
            output_dir = get_project_root() / "results" / "analysis"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []
        aggregated = self.aggregate_scores()

        if "error" in aggregated:
            return generated_files

        # 1. Model comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(aggregated["by_model"].keys())
        pass_rates = [aggregated["by_model"][m]["pass_rate"] * 100 for m in models]
        model_labels = [m.split("/")[-1] for m in models]

        bars = ax.bar(model_labels, pass_rates, color=["#4CAF50", "#2196F3"])
        ax.set_ylabel("Pass Rate (%)")
        ax.set_title("Model Performance Comparison")
        ax.set_ylim(0, 100)

        for bar, rate in zip(bars, pass_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        path = output_dir / "model_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated_files.append(str(path))

        # 2. Task type performance chart
        fig, ax = plt.subplots(figsize=(12, 6))
        task_types = list(aggregated["by_task_type"].keys())
        task_rates = [
            aggregated["by_task_type"][t]["pass_rate"] * 100 for t in task_types
        ]

        bars = ax.barh(task_types, task_rates, color="#9C27B0")
        ax.set_xlabel("Pass Rate (%)")
        ax.set_title("Performance by Task Type")
        ax.set_xlim(0, 100)

        for bar, rate in zip(bars, task_rates):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%",
                ha="left",
                va="center",
            )

        plt.tight_layout()
        path = output_dir / "task_type_performance.png"
        plt.savefig(path, dpi=150)
        plt.close()
        generated_files.append(str(path))

        # 3. Radar chart for model × task type (if enough data)
        if len(models) >= 2 and len(task_types) >= 3:
            fig, ax = plt.subplots(
                figsize=(10, 10), subplot_kw=dict(projection="polar")
            )

            angles = np.linspace(0, 2 * np.pi, len(task_types), endpoint=False).tolist()
            angles += angles[:1]  # Complete the loop

            colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]

            for idx, model in enumerate(models[:4]):  # Max 4 models
                model_data = aggregated["model_x_task_type"].get(model, {})
                values = [
                    model_data.get(tt, {}).get("pass_rate", 0) * 100
                    for tt in task_types
                ]
                values += values[:1]  # Complete the loop

                ax.plot(
                    angles,
                    values,
                    "o-",
                    linewidth=2,
                    label=model.split("/")[-1],
                    color=colors[idx % len(colors)],
                )
                ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(task_types)
            ax.set_ylim(0, 100)
            ax.set_title("Model Performance Radar")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

            plt.tight_layout()
            path = output_dir / "radar_chart.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            generated_files.append(str(path))

        return generated_files

    def identify_insights(self) -> dict[str, Any]:
        """Identify key insights from the benchmark results."""
        aggregated = self.aggregate_scores()

        if "error" in aggregated:
            return {"error": "No data available"}

        insights = {
            "hardest_task_types": [],
            "easiest_task_types": [],
            "biggest_model_gaps": [],
            "notable_patterns": [],
        }

        # Sort task types by difficulty
        task_type_sorted = sorted(
            aggregated["by_task_type"].items(),
            key=lambda x: x[1]["pass_rate"],
        )

        if task_type_sorted:
            insights["hardest_task_types"] = [
                {"task_type": tt, "pass_rate": data["pass_rate"]}
                for tt, data in task_type_sorted[:2]
            ]
            insights["easiest_task_types"] = [
                {"task_type": tt, "pass_rate": data["pass_rate"]}
                for tt, data in task_type_sorted[-2:]
            ]

        # Find biggest performance gaps between models per task type
        for task_type in aggregated["by_task_type"]:
            rates = []
            for model, task_data in aggregated["model_x_task_type"].items():
                if task_type in task_data:
                    rates.append((model, task_data[task_type]["pass_rate"]))

            if len(rates) >= 2:
                rates.sort(key=lambda x: x[1], reverse=True)
                gap = rates[0][1] - rates[-1][1]
                if gap > 0.15:  # Significant gap
                    insights["biggest_model_gaps"].append(
                        {
                            "task_type": task_type,
                            "best_model": rates[0][0],
                            "worst_model": rates[-1][0],
                            "gap": gap,
                        }
                    )

        return insights

    def save_full_report(self) -> dict[str, str]:
        """Generate and save all analysis outputs."""
        paths = {}

        # Save aggregated scores
        aggregated = self.aggregate_scores()
        path = get_analysis_path("aggregated_scores")
        save_yaml(aggregated, path)
        paths["aggregated_scores"] = str(path)

        # Save leaderboard
        leaderboard = self.generate_leaderboard()
        path = get_project_root() / "results" / "analysis" / "LEADERBOARD.md"
        with open(path, "w") as f:
            f.write(leaderboard)
        paths["leaderboard"] = str(path)

        # Save statistical comparisons
        comparisons = self.statistical_analysis()
        comparison_data = [
            {
                "model_a": c.model_a,
                "model_b": c.model_b,
                "t_statistic": c.t_statistic,
                "p_value": c.p_value,
                "significant": c.significant,
                "effect_size": c.effect_size,
                "effect_interpretation": c.effect_interpretation,
            }
            for c in comparisons
        ]
        path = get_analysis_path("statistical_comparisons")
        save_json(comparison_data, path)
        paths["statistical_comparisons"] = str(path)

        # Save insights
        insights = self.identify_insights()
        path = get_analysis_path("insights")
        save_yaml(insights, path)
        paths["insights"] = str(path)

        # Generate visualizations
        viz_paths = self.generate_visualizations()
        paths["visualizations"] = viz_paths

        return paths


if __name__ == "__main__":
    print("Analyzer module loaded successfully.")
    print("\nTo run analysis:")
    print("  analyzer = BenchmarkAnalyzer()")
    print("  paths = analyzer.save_full_report()")
    print("  print(analyzer.generate_leaderboard())")
