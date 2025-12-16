#!/usr/bin/env python3
"""
Story Theory Benchmark - CLI Runner

Simple command-line interface for running the benchmark incrementally.

Usage:
    python run.py --gen-model "anthropic/claude-sonnet-4"     # Run one model
    python run.py --gen-model "model1" "model2" --eval-model "evaluator1" "evaluator2"
    python run.py --gen-model "model" --task-type "agentic_constraint_discovery"  # Specific task type
    python run.py --status                                     # Show current status
    python run.py --leaderboard                                # Generate leaderboard
    python run.py --list-missing "anthropic/claude-sonnet-4"   # Show what would be run
    python run.py --clean-failed                               # Remove all failed records
    python run.py --clean-failed "anthropic/claude-sonnet-4"   # Remove failed for model
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

# Default models
DEFAULT_EVAL_MODELS = ["anthropic/claude-haiku-4.5", "openai/gpt-5-mini", "google/gemini-2.5-flash"]


def check_api_key() -> bool:
    """Check if API key is configured."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key or key == "your-api-key-here":
        print("ERROR: OPENROUTER_API_KEY not configured in .env file")
        return False
    return True


def cmd_status(args):
    """Show current benchmark status."""
    from results_db import ResultsDatabase

    db = ResultsDatabase()
    db.print_status()


def cmd_leaderboard(args):
    """Generate and display leaderboard."""
    from results_db import ResultsDatabase

    db = ResultsDatabase()
    leaderboard = db.generate_leaderboard_md()
    print(leaderboard)

    # Also save to file
    output_path = Path("results/LEADERBOARD.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(leaderboard)
    print(f"\nSaved to: {output_path}")


def cmd_rebuild_db(args):
    """Rebuild JSON database from YAML files."""
    from results_db import ResultsDatabase

    print("Rebuilding database from YAML files...")
    db = ResultsDatabase()
    gen_count, eval_count = db.rebuild_from_yaml()
    print(f"Loaded {gen_count} generations and {eval_count} evaluations")
    db.print_status()


def cmd_task_analysis(args):
    """Generate and display task analysis."""
    from results_db import ResultsDatabase

    db = ResultsDatabase()
    analysis = db.generate_task_analysis_md()
    print(analysis)

    # Also save to file
    output_path = Path("results/TASK_ANALYSIS.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(analysis)
    print(f"\nSaved to: {output_path}")


def cmd_clean_failed(args):
    """Remove failed generations and evaluations to allow retry."""
    from results_db import ResultsDatabase

    db = ResultsDatabase()
    model = args.clean_model if hasattr(args, 'clean_model') and args.clean_model else None

    if model:
        print(f"Cleaning failed records for model: {model}")
    else:
        print("Cleaning all failed records")

    gen_removed = db.remove_failed_generations(model)
    eval_removed = db.remove_failed_evaluations(model)

    print(f"Removed {gen_removed} failed generation(s)")
    print(f"Removed {eval_removed} failed evaluation(s)")

    if gen_removed > 0 or eval_removed > 0:
        print("\nUpdated status:")
        db.print_status()


def cmd_list_missing(args):
    """List missing generations and evaluations for given models."""
    from results_db import ResultsDatabase
    from utils import load_all_tasks

    db = ResultsDatabase()
    tasks = load_all_tasks()
    task_ids = [t["task_id"] for t in tasks]

    gen_models = args.gen_models
    eval_models = args.eval_models or DEFAULT_EVAL_MODELS

    print(f"\nGeneration models: {gen_models}")
    print(f"Evaluator models: {eval_models}")
    print(f"Tasks in dataset: {len(task_ids)}")

    missing_gens = db.get_missing_generations(task_ids, gen_models)
    missing_evals = db.get_missing_evaluations(gen_models, eval_models)

    print(f"\nMissing generations: {len(missing_gens)}")
    if missing_gens and args.verbose:
        for task_id, model in missing_gens[:10]:
            print(f"  - {task_id} / {model}")
        if len(missing_gens) > 10:
            print(f"  ... and {len(missing_gens) - 10} more")

    print(f"\nMissing evaluations: {len(missing_evals)}")
    if missing_evals and args.verbose:
        for task_id, gen_model, eval_model in missing_evals[:10]:
            print(f"  - {task_id} / {gen_model} (eval: {eval_model})")
        if len(missing_evals) > 10:
            print(f"  ... and {len(missing_evals) - 10} more")

    # Cost estimate
    est_gen_cost = len(missing_gens) * 0.001  # ~$0.001 per generation avg
    est_eval_cost = len(missing_evals) * 0.0007  # ~$0.0007 per evaluation
    print(f"\nEstimated cost: ${est_gen_cost + est_eval_cost:.4f}")


def cmd_run(args):
    """Run benchmark for specified models."""
    if not check_api_key():
        return

    from results_db import ResultsDatabase, GenerationRecord, EvaluationRecord
    from generator import BenchmarkGenerator
    from evaluator import BenchmarkEvaluator
    from utils import load_all_tasks, count_words, save_yaml, get_generation_path, get_evaluation_path, is_agentic_task

    db = ResultsDatabase()
    tasks = load_all_tasks()

    # Filter by task type if specified
    if args.task_type:
        tasks = [t for t in tasks if t["task_type"] == args.task_type]
        print(f"Filtered to task type: {args.task_type}")

    # Limit number of tasks if specified
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
        print(f"Limited to first {args.max_tasks} task(s)")

    # Build task lookup and get all task IDs (both standard and agentic)
    task_lookup = {t["task_id"]: t for t in tasks}
    task_ids = [t["task_id"] for t in tasks]  # ALL tasks

    # Count for display
    standard_count = sum(1 for t in tasks if not is_agentic_task(t))
    agentic_count = sum(1 for t in tasks if is_agentic_task(t))

    gen_models = args.gen_models
    eval_models = args.eval_models or DEFAULT_EVAL_MODELS

    print(f"\n=== Story Theory Benchmark Runner ===")
    print(f"Generation models: {gen_models}")
    print(f"Evaluator models: {eval_models}")
    print(f"Standard tasks: {standard_count}")
    if agentic_count:
        print(f"Agentic tasks: {agentic_count}")

    # Auto-cleanup: Remove failed generations for each model before retry
    # This ensures failed generations can be retried without manual cleanup
    for model in gen_models:
        removed = db.remove_failed_generations(model)
        if removed > 0:
            print(f"  Cleaned up {removed} failed generation(s) for {model}")

    # Find missing generations (ALL tasks - both standard and agentic)
    missing_gens = db.get_missing_generations(task_ids, gen_models) if task_ids else []

    if not missing_gens and not args.force:
        print("\nAll generations complete for specified models.")
    else:
        if args.force:
            # Force re-run all
            missing_gens = [
                (task_id, model)
                for task_id in task_ids
                for model in gen_models
            ]
            print(f"\nForce mode: re-running all {len(missing_gens)} generations")
        elif missing_gens:
            print(f"\nMissing generations: {len(missing_gens)}")

        if args.dry_run:
            print("(Dry run - not executing)")
        elif missing_gens:
            # Import agentic tools for agentic tasks
            from agentic_generator import AgenticGenerator, AgenticConfig
            from agentic_evaluator import (
                AgenticEvaluator,
                create_constraint_discovery_oracle,
                create_feedback_generator,
            )
            from generator import GeneratorConfig

            # Setup generators
            config = GeneratorConfig(max_reasoning_tokens=args.max_reasoning_tokens)
            generator = BenchmarkGenerator(config=config)
            agentic_config = AgenticConfig()
            agentic_gen = AgenticGenerator(config=agentic_config)

            results_dir = Path("results/agentic")
            results_dir.mkdir(parents=True, exist_ok=True)

            # Unified generation loop - handles both standard and agentic tasks
            print("\n--- Running Generations ---")
            from tqdm import tqdm
            for task_id, model in tqdm(missing_gens, desc="Generating"):
                task = task_lookup[task_id]

                if is_agentic_task(task):
                    # Agentic task generation
                    agentic_type = task.get("agentic_type")

                    try:
                        if agentic_type == "constraint_discovery":
                            oracle = create_constraint_discovery_oracle(task)
                            result = agentic_gen.run_constraint_discovery(task, model, 0, oracle)
                        elif agentic_type == "planning_execution":
                            result = agentic_gen.run_planning_execution(task, model, 0)
                        elif agentic_type == "iterative_revision":
                            feedback_gen = create_feedback_generator(task)
                            result = agentic_gen.run_iterative_revision(task, model, 0, feedback_gen)
                        elif agentic_type == "critique_improvement":
                            result = agentic_gen.run_critique_improvement(task, model, 0)
                        else:
                            print(f"  Unknown agentic type: {agentic_type}")
                            continue

                        # Save agentic generation result
                        gen_path = results_dir / f"gen_{task_id}_{model.replace('/', '_')}.yaml"
                        save_yaml(result.to_dict(), gen_path)

                        # Store in DB for tracking
                        record = GenerationRecord(
                            task_id=task_id,
                            task_type=task["task_type"],
                            theory=task.get("theory", "Unknown"),
                            model=model,
                            sample=0,
                            output=result.final_output,
                            prompt_tokens=result.total_prompt_tokens,
                            completion_tokens=result.total_completion_tokens,
                            reasoning_tokens=result.total_reasoning_tokens,
                            generation_cost=result.total_cost,
                            timestamp=result.timestamp,
                            success=result.success,
                            error=result.error,
                        )
                        db.add_generation(record)

                    except Exception as e:
                        print(f"  Error on {task_id}/{model}: {e}")
                else:
                    # Standard task generation
                    result = generator.generate_sample(task, model, 0)

                    record = GenerationRecord(
                        task_id=task_id,
                        task_type=task["task_type"],
                        theory=task.get("theory", "Unknown"),
                        model=model,
                        sample=0,
                        output=result.output,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        reasoning_tokens=result.reasoning_tokens,
                        generation_cost=result.cost,
                        timestamp=result.timestamp,
                        success=result.success,
                        error=result.error,
                    )
                    db.add_generation(record)

                    # Save individual YAML file
                    yaml_path = get_generation_path(task_id, model, 0)
                    save_yaml(result.to_dict(), yaml_path)

    # Find missing evaluations (ALL tasks - both standard and agentic)
    missing_evals = db.get_missing_evaluations(gen_models, eval_models)

    if not missing_evals and not args.force:
        print("\nAll evaluations complete for specified models.")
    else:
        if args.force:
            # Force re-evaluate all
            missing_evals = [
                (g["task_id"], g["model"], eval_model)
                for g in db._data["generations"]
                if g["model"] in gen_models and g.get("success", True)
                for eval_model in eval_models
            ]
            print(f"\nForce mode: re-running all {len(missing_evals)} evaluations")
        else:
            print(f"\nMissing evaluations: {len(missing_evals)}")

        if args.dry_run:
            print("(Dry run - not executing)")
        elif missing_evals:
            # Import agentic evaluator for agentic tasks
            from agentic_evaluator import AgenticEvaluator
            import yaml

            results_dir = Path("results/agentic")

            # Unified evaluation loop - handles both standard and agentic tasks
            print("\n--- Running Evaluations ---")

            from tqdm import tqdm
            for task_id, gen_model, eval_model in tqdm(missing_evals, desc="Evaluating"):
                task = task_lookup.get(task_id)
                if not task:
                    continue

                generation = db.get_generation(task_id, gen_model, 0)
                if not generation:
                    continue

                if is_agentic_task(task):
                    # Agentic task evaluation
                    gen_path = results_dir / f"gen_{task_id}_{gen_model.replace('/', '_')}.yaml"

                    if not gen_path.exists():
                        continue

                    with open(gen_path) as f:
                        gen_result = yaml.safe_load(f)

                    agentic_eval = AgenticEvaluator(evaluator_model=eval_model)
                    result = agentic_eval.evaluate_agentic_result(task, gen_result)

                    # Save evaluation
                    eval_path = results_dir / f"eval_{task_id}_{gen_model.replace('/', '_')}_{eval_model.replace('/', '_')}.yaml"
                    save_yaml(result.to_dict(), eval_path)

                    # Store in DB
                    record = EvaluationRecord(
                        task_id=task_id,
                        task_type=task["task_type"],
                        model=gen_model,
                        sample=0,
                        evaluator_model=result.evaluator_model,
                        evaluation_cost=result.evaluator_cost,
                        timestamp=result.timestamp,
                        success=result.success,
                        final_score=result.final_score,
                        score_breakdown={"process": result.process_scores, "output": result.output_scores},
                        llm_results=result.llm_results,
                        error=result.error,
                    )
                    db.add_evaluation(record)
                else:
                    # Standard task evaluation
                    word_count = count_words(generation["output"])

                    evaluator = BenchmarkEvaluator(evaluator_model=eval_model)
                    result = evaluator.evaluate_generation(task, {
                        "generation_id": f"{task_id}_{gen_model}_0",
                        "model": gen_model,
                        "sample_index": 0,
                        "output": generation["output"],
                        "word_count": word_count,
                    })

                    record = EvaluationRecord(
                        task_id=task_id,
                        task_type=task["task_type"],
                        model=gen_model,
                        sample=0,
                        evaluator_model=result.evaluator_model,
                        evaluation_cost=result.evaluator_cost,
                        timestamp=result.timestamp,
                        success=result.success,
                        final_score=result.final_score,
                        score_breakdown=result.score_breakdown,
                        llm_results=result.llm_results,
                        error=result.error,
                    )
                    db.add_evaluation(record)

                    # Save individual YAML file
                    eval_id = f"{task_id}_{gen_model}_0_{eval_model}".replace("/", "_")
                    yaml_path = get_evaluation_path(eval_id)
                    save_yaml(result.to_dict(), yaml_path)

    # Show summary
    print("\n--- Summary ---")
    db.print_status()

    # Generate leaderboard and task analysis
    if not args.dry_run:
        leaderboard = db.generate_leaderboard_md()
        leaderboard_path = Path("results/LEADERBOARD.md")
        with open(leaderboard_path, "w") as f:
            f.write(leaderboard)
        print(f"\nLeaderboard saved to: {leaderboard_path}")

        analysis = db.generate_task_analysis_md()
        analysis_path = Path("results/TASK_ANALYSIS.md")
        with open(analysis_path, "w") as f:
            f.write(analysis)
        print(f"Task analysis saved to: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Story Theory Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --status
  python run.py --gen-model "anthropic/claude-sonnet-4"
  python run.py --gen-model "model1" "model2" --eval-model "evaluator1" "evaluator2"
  python run.py --gen-model "model" --task-type "agentic_constraint_discovery"
  python run.py --list-missing "anthropic/claude-sonnet-4" -v
  python run.py --leaderboard
  python run.py --clean-failed
  python run.py --clean-failed "anthropic/claude-sonnet-4"
        """
    )

    # Commands
    parser.add_argument(
        "--status", action="store_true",
        help="Show current benchmark status"
    )
    parser.add_argument(
        "--leaderboard", action="store_true",
        help="Generate and display leaderboard"
    )
    parser.add_argument(
        "--rebuild-db", action="store_true",
        help="Rebuild JSON database from YAML files"
    )
    parser.add_argument(
        "--task-analysis", action="store_true",
        help="Generate and display task analysis"
    )
    parser.add_argument(
        "--clean-failed", nargs="?", const=True, default=None, dest="clean_failed", metavar="MODEL",
        help="Remove failed generations/evaluations (optionally filter by model)"
    )
    parser.add_argument(
        "--list-missing", nargs="+", dest="list_missing_models", metavar="MODEL",
        help="List missing generations/evaluations for models"
    )
    parser.add_argument(
        "--gen-model", "-g", nargs="+", dest="gen_models", metavar="MODEL",
        help="Model(s) to generate stories with"
    )
    parser.add_argument(
        "--eval-model", "-e", nargs="+", dest="eval_models", metavar="MODEL",
        help=f"Model(s) to evaluate with (default: {DEFAULT_EVAL_MODELS})"
    )

    # Options
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-run even if results exist"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--max-reasoning-tokens", type=int, default=1000,
        help="Max reasoning/thinking tokens for models that support it (default: 1000)"
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Limit to first N tasks (for testing)"
    )
    parser.add_argument(
        "--task-type", type=str, default=None,
        help="Filter to specific task type (e.g., 'beat_interpolation', 'agentic_constraint_discovery')"
    )

    args = parser.parse_args()

    # Route to appropriate command
    if args.status:
        cmd_status(args)
    elif args.leaderboard:
        cmd_leaderboard(args)
    elif args.rebuild_db:
        cmd_rebuild_db(args)
    elif args.task_analysis:
        cmd_task_analysis(args)
    elif args.clean_failed is not None:
        # Handle both --clean-failed (True) and --clean-failed MODEL (string)
        args.clean_model = args.clean_failed if isinstance(args.clean_failed, str) else None
        cmd_clean_failed(args)
    elif args.list_missing_models:
        args.gen_models = args.list_missing_models
        cmd_list_missing(args)
    elif args.gen_models:
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
