"""
Utility functions for the Story Theory Benchmark.

This module provides shared functionality for loading tasks, configuration,
validating schemas, and other common operations.
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jsonschema import ValidationError, validate


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], file_path: str | Path) -> None:
    """Save data to a YAML file."""
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_json(file_path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], file_path: str | Path, indent: int = 2) -> None:
    """Save data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_config() -> dict[str, Any]:
    """Load all configuration files."""
    root = get_project_root()
    config = {}
    config["models"] = load_yaml(root / "config" / "models.yaml")
    config["settings"] = load_yaml(root / "config" / "settings.yaml")
    return config


def load_task_schema() -> dict[str, Any]:
    """Load the task validation schema."""
    root = get_project_root()
    return load_json(root / "dataset" / "schemas" / "task_schema.json")


def load_evaluation_schema() -> dict[str, Any]:
    """Load the evaluation validation schema."""
    root = get_project_root()
    return load_json(root / "dataset" / "schemas" / "evaluation_schema.json")


def validate_task(task: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate a task against the task schema.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        schema = load_task_schema()
        validate(instance=task, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e.message)


def load_all_tasks() -> list[dict[str, Any]]:
    """Load all task files from the dataset directory."""
    root = get_project_root()
    tasks_dir = root / "dataset" / "tasks"
    tasks = []

    # All task types (standard + agentic)
    task_types = [
        "beat_interpolation",
        "beat_revision",
        "constrained_continuation",
        "theory_conversion",
        "multi_beat_synthesis",
        "agentic_constraint_discovery",
        "agentic_planning_execution",
        "agentic_iterative_revision",
        "critique_improvement",
    ]

    for task_type in task_types:
        type_dir = tasks_dir / task_type
        if type_dir.exists():
            for task_file in sorted(type_dir.glob("*.yaml")):
                task = load_yaml(task_file)
                task["_file_path"] = str(task_file)
                tasks.append(task)

    return tasks


def is_agentic_task(task: dict[str, Any]) -> bool:
    """Check if a task is an agentic (multi-turn) task."""
    task_type = task.get("task_type", "")
    return task_type.startswith("agentic_") or task_type == "critique_improvement"


def load_tasks_by_type(task_type: str) -> list[dict[str, Any]]:
    """Load all tasks of a specific type."""
    all_tasks = load_all_tasks()
    return [t for t in all_tasks if t.get("task_type") == task_type]


def generate_id() -> str:
    """Generate a unique ID for generations and evaluations."""
    return str(uuid.uuid4())[:8]


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def count_words(text: str) -> int:
    """Count words in a text string."""
    # Remove extra whitespace and split
    words = text.split()
    return len(words)


def validate_word_count(text: str, min_words: int, max_words: int) -> bool:
    """Check if text word count is within range."""
    count = count_words(text)
    return min_words <= count <= max_words


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """
    Extract JSON from a response that might contain additional text.
    Handles cases where the model includes explanation before/after JSON.
    """
    # Try to parse the whole response first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Look for JSON block in markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Look for JSON object pattern
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Calculate the cost of an API call."""
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (completion_tokens / 1_000_000) * output_price_per_million
    return input_cost + output_cost


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_generation_path(task_id: str, model: str, sample_index: int) -> Path:
    """Get the path for a generation result file."""
    root = get_project_root()
    gen_dir = ensure_directory(root / "results" / "generations")
    filename = f"{task_id}_{model.replace('/', '_')}_{sample_index}.yaml"
    return gen_dir / filename


def get_evaluation_path(generation_id: str) -> Path:
    """Get the path for an evaluation result file."""
    root = get_project_root()
    eval_dir = ensure_directory(root / "results" / "evaluations")
    return eval_dir / f"eval_{generation_id}.yaml"


def get_analysis_path(analysis_type: str) -> Path:
    """Get the path for an analysis result file."""
    root = get_project_root()
    analysis_dir = ensure_directory(root / "results" / "analysis")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return analysis_dir / f"{analysis_type}_{timestamp}.yaml"


class TaskTypeInfo:
    """Information about task type requirements and scoring."""

    TASK_TYPES = {
        "beat_interpolation": {
            "pass_threshold": 1.0,
            "criteria": [
                "beat_elements_present",
                "word_count_valid",
                "character_consistent",
                "logical_bridge",
                "setting_continuity",
            ],
        },
        "beat_revision": {
            "required_criteria": ["flaw_corrected", "beat_definition_satisfied"],
            "additional_pass_count": 3,
            "criteria": [
                "flaw_corrected",
                "beat_definition_satisfied",
                "preserved_elements",
                "word_count_valid",
                "no_new_errors",
            ],
        },
        "constrained_continuation": {
            "pass_threshold": 0.80,
        },
        "theory_conversion": {
            "required_criteria": ["all_target_beats_present", "core_preserved"],
            "min_structural_accuracy": 0.70,
        },
        "multi_beat_synthesis": {
            "cross_beat_pass_threshold": 1.0,
            "beat_specific_pass_threshold": 0.70,
        },
    }

    @classmethod
    def get_info(cls, task_type: str) -> dict[str, Any]:
        """Get scoring information for a task type."""
        return cls.TASK_TYPES.get(task_type, {})

    @classmethod
    def calculate_pass(cls, task_type: str, evaluation_results: dict[str, Any]) -> bool:
        """Calculate if an evaluation passes based on task type rules."""
        info = cls.get_info(task_type)

        if task_type == "beat_interpolation":
            # All 5 criteria must pass
            criteria = info["criteria"]
            return all(evaluation_results.get(c, False) for c in criteria)

        elif task_type == "beat_revision":
            # Required criteria must pass + 3 others
            required = info["required_criteria"]
            if not all(evaluation_results.get(c, False) for c in required):
                return False
            other_criteria = [c for c in info["criteria"] if c not in required]
            passed_others = sum(1 for c in other_criteria if evaluation_results.get(c, False))
            return passed_others >= info["additional_pass_count"]

        elif task_type == "constrained_continuation":
            # 80% of constraints must be satisfied
            constraints = evaluation_results.get("constraints_checked", {})
            if not constraints:
                return False
            passed = sum(1 for v in constraints.values() if v)
            return (passed / len(constraints)) >= info["pass_threshold"]

        elif task_type == "theory_conversion":
            # All target beats + core preserved + structural accuracy >= 0.7
            required = info["required_criteria"]
            if not all(evaluation_results.get(c, False) for c in required):
                return False
            accuracy = evaluation_results.get("structural_accuracy_score", 0)
            return accuracy >= info["min_structural_accuracy"]

        elif task_type == "multi_beat_synthesis":
            # All cross-beat constraints + 70% beat-specific
            cross_beat = evaluation_results.get("cross_beat_constraints", {})
            if cross_beat and not all(cross_beat.values()):
                return False

            beat_reqs = evaluation_results.get("beat_requirements", {})
            if beat_reqs:
                total_reqs = 0
                passed_reqs = 0
                for beat_data in beat_reqs.values():
                    for req_passed in beat_data.values():
                        total_reqs += 1
                        if req_passed:
                            passed_reqs += 1
                if total_reqs > 0:
                    beat_rate = passed_reqs / total_reqs
                    return beat_rate >= info["beat_specific_pass_threshold"]

            return evaluation_results.get("overall_pass", False)

        return False


if __name__ == "__main__":
    # Quick test
    print("Loading configuration...")
    config = load_config()
    print(f"Model tiers: {list(config['models']['model_tiers'].keys())}")

    print("\nLoading tasks...")
    tasks = load_all_tasks()
    print(f"Total tasks loaded: {len(tasks)}")

    for task_type in [
        "beat_interpolation",
        "beat_revision",
        "constrained_continuation",
        "theory_conversion",
        "multi_beat_synthesis",
    ]:
        type_tasks = [t for t in tasks if t.get("task_type") == task_type]
        print(f"  {task_type}: {len(type_tasks)}")
