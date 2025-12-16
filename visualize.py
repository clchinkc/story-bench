#!/usr/bin/env python3
"""
Visualization script for Story Theory Benchmark results.

Creates focused visualizations:
1. Cost vs Score Scatter (primary - shows cost-effectiveness)
2. Model Rankings Bar Chart
3. Task Type Performance Heatmap
4. Component Score Breakdown
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from results_db import ResultsDatabase


# Company classification and colors
COMPANY_INFO = {
    "anthropic": {"name": "Anthropic", "color": "#D4A574"},  # Warm tan/clay
    "openai": {"name": "OpenAI", "color": "#10A37F"},  # OpenAI green
    "google": {"name": "Google", "color": "#4285F4"},  # Google blue
    "deepseek": {"name": "DeepSeek", "color": "#6366F1"},  # Indigo
    "meta-llama": {"name": "Meta", "color": "#0668E1"},  # Meta blue
    "x-ai": {"name": "xAI", "color": "#1DA1F2"},  # X/Twitter blue
    "qwen": {"name": "Alibaba", "color": "#FF6A00"},  # Alibaba orange
    "mistralai": {"name": "Mistral", "color": "#FF7000"},  # Mistral orange
    "minimax": {"name": "MiniMax", "color": "#9333EA"},  # Purple
}


def get_company(model: str) -> str:
    """Extract company/provider from model path."""
    if "/" in model:
        return model.split("/")[0]
    return "unknown"


def get_company_color(model: str) -> str:
    """Get color for a model based on its company."""
    company = get_company(model)
    return COMPANY_INFO.get(company, {}).get("color", "#808080")


def get_company_name(model: str) -> str:
    """Get company display name for a model."""
    company = get_company(model)
    return COMPANY_INFO.get(company, {}).get("name", company.title())


def get_short_model_name(model: str) -> str:
    """Extract short model name from full path."""
    name = model.split("/")[-1] if "/" in model else model
    # Further shorten common prefixes for display
    if name.startswith("claude-"):
        name = name.replace("claude-", "Claude ")
    elif name.startswith("gpt-"):
        name = name.replace("gpt-", "GPT-")
    elif name.startswith("gemini-"):
        name = name.replace("gemini-", "Gemini ")
    elif name.startswith("deepseek-"):
        name = name.replace("deepseek-", "DeepSeek ")
    elif name.startswith("grok-"):
        name = name.replace("grok-", "Grok ")
    return name


def load_benchmark_data() -> dict:
    """Load benchmark results from database."""
    db = ResultsDatabase()
    return db.get_results_summary()


def plot_cost_vs_score(data: dict, ax: plt.Axes) -> None:
    """Create scatter plot of cost vs performance with company colors.

    Shows total token usage (including reasoning) via marker size and
    actual monetary cost on x-axis. Colors represent different companies.
    """
    models = data["models"]

    model_keys = []
    names = []
    scores = []
    costs = []
    total_tokens = []
    colors = []

    for model, stats in models.items():
        score = stats.get("avg_score")
        cost = stats.get("generation_cost", 0)
        token_stats = stats.get("token_stats", {})
        # Total tokens = prompt + completion (completion includes reasoning + output)
        tokens = token_stats.get("prompt_tokens", 0) + token_stats.get("completion_tokens", 0)
        if score is not None:
            model_keys.append(model)
            names.append(get_short_model_name(model))
            scores.append(score * 100)
            costs.append(cost)
            total_tokens.append(tokens)
            colors.append(get_company_color(model))

    # Normalize token counts to marker sizes (100-400 range)
    if total_tokens and max(total_tokens) > 0:
        min_tokens = min(total_tokens)
        max_tokens_val = max(total_tokens)
        if max_tokens_val > min_tokens:
            sizes = [100 + 300 * (t - min_tokens) / (max_tokens_val - min_tokens) for t in total_tokens]
        else:
            sizes = [200] * len(total_tokens)
    else:
        sizes = [200] * len(costs)

    # Scatter with company colors, size based on total tokens
    for i in range(len(names)):
        ax.scatter(
            costs[i], scores[i],
            c=colors[i],
            s=sizes[i], edgecolors="black", linewidths=1.5,
            zorder=5
        )

    # Add model labels with smart positioning
    for i, name in enumerate(names):
        # Offset labels to avoid overlap with points
        offset_x = 0.008
        offset_y = 1.5
        ha = "left"

        # Adjust for edge cases
        if costs[i] > 0.3:
            offset_x = -0.008
            ha = "right"

        ax.annotate(
            name, (costs[i], scores[i]),
            xytext=(offset_x, offset_y), textcoords="offset points",
            fontsize=8, fontweight="bold", alpha=0.9,
            ha=ha, va="bottom"
        )

    # Add Pareto efficiency frontier line (connecting best cost-score tradeoffs)
    sorted_by_cost = sorted(zip(costs, scores, names), key=lambda x: x[0])
    frontier_costs = [sorted_by_cost[0][0]]
    frontier_scores = [sorted_by_cost[0][1]]
    max_score = frontier_scores[0]

    for c, s, n in sorted_by_cost[1:]:
        if s > max_score:
            frontier_costs.append(c)
            frontier_scores.append(s)
            max_score = s

    if len(frontier_costs) > 1:
        ax.plot(frontier_costs, frontier_scores, '--', color='#E74C3C', alpha=0.7, linewidth=2.5, zorder=1, label='Pareto Frontier')
        ax.legend(loc='lower right', fontsize=9)

    # Add company legend
    legend_handles = []
    companies_shown = set()
    for model in model_keys:
        company = get_company(model)
        if company not in companies_shown:
            companies_shown.add(company)
            legend_handles.append(plt.scatter([], [], c=get_company_color(model), s=100,
                                             edgecolors='black', linewidths=1, label=get_company_name(model)))

    ax.legend(handles=legend_handles, loc='lower right', fontsize=8, title='Company', title_fontsize=9)

    ax.set_xlabel("Generation Cost ($)", fontsize=11)
    ax.set_ylabel("Benchmark Score (%)", fontsize=11)
    ax.set_title("Cost vs Performance by Company\n(Higher & Left = Better, Larger = More Tokens)", fontsize=14, fontweight="bold")

    # Set axis limits with padding
    ax.set_xlim(0, max(costs) * 1.15)
    ax.set_ylim(min(60, min(scores) - 5), 100)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)


def plot_model_scores_stacked(data: dict, ax: plt.Axes) -> None:
    """Create stacked horizontal bar chart showing weighted component contributions to total score."""
    models = data["models"]

    # Sort by score
    sorted_items = sorted(
        models.items(),
        key=lambda x: x[1].get("avg_score") or 0,
        reverse=True
    )

    names = [get_short_model_name(m) for m, _ in sorted_items]
    total_scores = [(s.get("avg_score") or 0) * 100 for _, s in sorted_items]

    # Extract weighted component contributions (actual contribution to final score)
    # Weights: word_count=0.25, programmatic=0.25, llm_judge=0.50
    wc_contrib = []
    prog_contrib = []
    llm_contrib = []
    for _, stats in sorted_items:
        comps = stats.get("avg_components", {})
        wc_contrib.append((comps.get("word_count") or 0) * 0.25 * 100)
        prog_contrib.append((comps.get("programmatic") or 0) * 0.25 * 100)
        llm_contrib.append((comps.get("llm_judge") or 0) * 0.50 * 100)

    y = np.arange(len(names))
    height = 0.6

    # Stacked horizontal bars showing actual contribution to total
    bars_wc = ax.barh(y, wc_contrib, height, label="Word Count (25%)", color="#3498db", edgecolor='white', linewidth=0.5)
    bars_prog = ax.barh(y, prog_contrib, height, left=wc_contrib, label="Programmatic (25%)", color="#e74c3c", edgecolor='white', linewidth=0.5)
    left_for_llm = [w + p for w, p in zip(wc_contrib, prog_contrib)]
    bars_llm = ax.barh(y, llm_contrib, height, left=left_for_llm, label="LLM Judge (50%)", color="#2ecc71", edgecolor='white', linewidth=0.5)

    # Add total score label on right
    for i, total in enumerate(total_scores):
        ax.annotate(f"{total:.1f}%", xy=(total + 1, i), fontsize=10, fontweight="bold",
                   va="center", ha="left", color="#2c3e50")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Weighted Score Contribution (%)", fontsize=11)
    ax.set_title("Model Rankings (Stacked Component Contributions)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(total_scores) + 12)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.3)


def plot_task_type_heatmap(data: dict, ax: plt.Axes) -> None:
    """Create heatmap of model performance by task type."""
    models = data["models"]
    task_types = sorted(data["task_types"])

    # Sort models by overall score
    sorted_items = sorted(
        models.items(),
        key=lambda x: x[1].get("avg_score") or 0,
        reverse=True
    )

    names = [get_short_model_name(m) for m, _ in sorted_items]

    # Build score matrix
    matrix = []
    for _, stats in sorted_items:
        row = []
        for tt in task_types:
            if tt in stats.get("by_task_type", {}):
                score = stats["by_task_type"][tt].get("avg_score")
                row.append((score or 0) * 100)
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)

    # Create heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    # Labels
    task_labels = [tt.replace("_", " ").title() for tt in task_types]
    ax.set_xticks(np.arange(len(task_types)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(task_labels, fontsize=9)
    ax.set_yticklabels(names, fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(names)):
        for j in range(len(task_types)):
            score = matrix[i, j]
            color = "white" if score < 50 else "black"
            ax.text(j, i, f"{score:.0f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_title("Performance by Task Type (%)", fontsize=14, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Score (%)", fontsize=10)


def plot_component_scores_raw(data: dict, ax: plt.Axes) -> None:
    """Create grouped bar chart showing raw (unweighted) component scores for comparison."""
    models = data["models"]

    # Sort by overall score
    sorted_items = sorted(
        models.items(),
        key=lambda x: x[1].get("avg_score") or 0,
        reverse=True
    )

    names = [get_short_model_name(m) for m, _ in sorted_items]

    # Extract raw component scores (not weighted)
    word_count = []
    programmatic = []
    llm_judge = []

    for _, stats in sorted_items:
        comps = stats.get("avg_components", {})
        word_count.append((comps.get("word_count") or 0) * 100)
        programmatic.append((comps.get("programmatic") or 0) * 100)
        llm_judge.append((comps.get("llm_judge") or 0) * 100)

    x = np.arange(len(names))
    width = 0.25

    bars_wc = ax.bar(x - width, word_count, width, label="Word Count", color="#3498db", edgecolor='black', linewidth=0.5)
    bars_prog = ax.bar(x, programmatic, width, label="Programmatic", color="#e74c3c", edgecolor='black', linewidth=0.5)
    bars_llm = ax.bar(x + width, llm_judge, width, label="LLM Judge", color="#2ecc71", edgecolor='black', linewidth=0.5)

    # Add value labels on top of bars
    for bar, score in zip(bars_wc, word_count):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{score:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, score in zip(bars_prog, programmatic):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{score:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, score in zip(bars_llm, llm_judge):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{score:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Raw Score (%)", fontsize=11)
    ax.set_title("Raw Component Scores by Model (Unweighted)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)


def create_summary_dashboard(output_path: str = "results/visualizations/benchmark_visualization.png") -> None:
    """Create dashboard with cost-effectiveness, rankings, and task performance."""
    print("Loading benchmark data...")
    data = load_benchmark_data()

    if not data["models"]:
        print("No benchmark data available. Run the benchmark first.")
        return

    print(f"Found {len(data['models'])} models, {len(data['task_types'])} task types")

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, width_ratios=[1, 1.2])

    # Plot 1: Cost vs Score (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_cost_vs_score(data, ax1)

    # Plot 2: Model Rankings with Stacked Components (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_model_scores_stacked(data, ax2)

    # Plot 3: Task Type Heatmap (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_task_type_heatmap(data, ax3)

    # Plot 4: Raw Component Scores (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_component_scores_raw(data, ax4)

    # Title with actual cost from metadata
    evaluators = ", ".join(e.split("/")[-1] for e in data.get("evaluators", []))
    total_cost = data.get("metadata", {}).get("total_cost", 0)
    fig.suptitle(
        f"Story Theory Benchmark Results\n"
        f"{len(data['models'])} Models | {len(data['task_types'])} Task Types | "
        f"Evaluators: {evaluators} | Total Cost: ${total_cost:.2f}",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nVisualization saved to: {output_path}")

    save_individual_plots(data, output_path.parent)
    plt.close()


def save_individual_plots(data: dict, output_dir: Path) -> None:
    """Save individual plots as separate files."""
    plots = [
        ("cost_vs_score", plot_cost_vs_score, (10, 8)),
        ("rankings_stacked", plot_model_scores_stacked, (12, 8)),
        ("heatmap", plot_task_type_heatmap, (12, 8)),
        ("components_raw", plot_component_scores_raw, (14, 6)),
    ]

    for name, plot_func, figsize in plots:
        fig, ax = plt.subplots(figsize=figsize)
        plot_func(data, ax)
        plt.tight_layout()
        plt.savefig(output_dir / f"benchmark_{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    print(f"Individual plots saved to: {output_dir}")


def print_summary(data: dict) -> None:
    """Print text summary of benchmark results."""
    print("\n" + "=" * 60)
    print("STORY THEORY BENCHMARK SUMMARY")
    print("=" * 60)

    # Sort models by score
    sorted_models = sorted(
        data["models"].items(),
        key=lambda x: x[1].get("avg_score") or 0,
        reverse=True
    )

    print(f"\nModels Evaluated: {len(sorted_models)}")
    print(f"Task Types: {', '.join(data['task_types'])}")
    print(f"Evaluators: {', '.join(e.split('/')[-1] for e in data.get('evaluators', []))}")

    print("\n--- RANKINGS ---")
    for rank, (model, stats) in enumerate(sorted_models, 1):
        name = get_short_model_name(model)
        score = stats.get("avg_score")
        score_str = f"{score:.1%}" if score else "N/A"
        cost = stats.get("generation_cost", 0)
        print(f"  {rank}. {name:25s} {score_str:>8s}  (${cost:.4f})")

    print("\n--- TOP PERFORMERS BY TASK TYPE ---")
    for tt in sorted(data["task_types"]):
        best_model = None
        best_score = 0
        for model, stats in sorted_models:
            if tt in stats.get("by_task_type", {}):
                score = stats["by_task_type"][tt].get("avg_score") or 0
                if score > best_score:
                    best_score = score
                    best_model = model
        if best_model:
            print(f"  {tt:30s}: {get_short_model_name(best_model)} ({best_score:.1%})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Story Theory Benchmark results")
    parser.add_argument(
        "--output", "-o",
        default="results/visualizations/benchmark_visualization.png",
        help="Output path for visualization"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Print text summary only"
    )

    args = parser.parse_args()

    data = load_benchmark_data()

    if args.summary:
        print_summary(data)
    else:
        print_summary(data)
        create_summary_dashboard(args.output)
