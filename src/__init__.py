"""
Story Theory Benchmark - LLM Narrative Generation Evaluation

This package provides tools for evaluating LLM narrative generation capabilities
across 5 task types based on established story theory frameworks.
"""

from .analyzer import BenchmarkAnalyzer
from .evaluator import BenchmarkEvaluator, EvaluationResult
from .generator import BenchmarkGenerator, GenerationResult, PromptBuilder
from .results_db import ResultsDatabase, GenerationRecord, EvaluationRecord
from .utils import (
    TaskTypeInfo,
    count_words,
    generate_id,
    get_project_root,
    load_all_tasks,
    load_config,
    load_tasks_by_type,
    validate_task,
    validate_word_count,
)

__version__ = "1.0.0"
__all__ = [
    # Core classes
    "BenchmarkGenerator",
    "BenchmarkEvaluator",
    "BenchmarkAnalyzer",
    "ResultsDatabase",
    # Data classes
    "GenerationResult",
    "EvaluationResult",
    "GenerationRecord",
    "EvaluationRecord",
    # Utilities
    "PromptBuilder",
    "TaskTypeInfo",
    "load_all_tasks",
    "load_tasks_by_type",
    "load_config",
    "validate_task",
    "count_words",
    "validate_word_count",
    "generate_id",
    "get_project_root",
]
