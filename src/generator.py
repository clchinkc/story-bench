"""
Story generation module for the Story Theory Benchmark.

This module handles generating story outputs from LLMs via OpenRouter API.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

# Configure logging for generation errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

from llm_client import LLMClient, TASK_TOKEN_CONFIG, get_llm_client
from utils import (
    generate_id,
    get_generation_path,
    get_timestamp,
    load_all_tasks,
    load_config,
    save_yaml,
)


@dataclass
class GenerationResult:
    """Result of a single generation."""

    generation_id: str
    task_id: str
    task_type: str
    theory: str
    model: str
    sample_index: int
    output: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost: float
    timestamp: str
    success: bool
    error: str | None = None
    finish_reason: str | None = None  # "stop", "length", "content_filter", etc.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generation_id": self.generation_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "theory": self.theory,
            "model": self.model,
            "sample_index": self.sample_index,
            "output": self.output,
            "metadata": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "output_tokens": self.completion_tokens - self.reasoning_tokens,
                "cost": self.cost,
                "timestamp": self.timestamp,
                "success": self.success,
                "error": self.error,
                "finish_reason": self.finish_reason,
            },
        }


@dataclass
class GeneratorConfig:
    """Configuration for the generator.

    Note: max_tokens must be greater than max_reasoning_tokens to leave room
    for the actual output after reasoning.
    """

    temperature: float = 0.7
    max_tokens: int = 3000  # Default, overridden by TASK_TOKEN_CONFIG
    max_reasoning_tokens: int | None = 1000  # Limit reasoning/thinking tokens
    samples_per_task: int = 3
    retry_attempts: int = 3
    retry_delay: int = 5


# Task token configuration is imported from llm_client module


class PromptBuilder:
    """Builds prompts for different task types."""

    SYSTEM_PROMPT_TEMPLATE = """You are a professional storyteller who writes CONCISE, focused narratives.

=== CRITICAL CONSTRAINTS ===

**TOKEN BUDGET (for reasoning models):**
- Reasoning: {max_reasoning_tokens} tokens MAX for thinking
- Output: {output_tokens} tokens MAX for story
- If you use all tokens on reasoning without output, you FAIL

**WORD COUNT IS A HARD REQUIREMENT:**
- The task specifies a word count range (e.g., 400-600 words)
- EXCEEDING THE MAXIMUM IS AN AUTOMATIC FAILURE
- Count your words as you write. Stop when you reach the limit.
- Better to be slightly under than over

**OUTPUT FORMAT:**
- Output ONLY story content - NO headers, labels, beat markers, or explanations
- Write continuous narrative prose only
- Maintain character voice, setting, and tone consistency

IMPORTANT: Plan your word budget BEFORE writing. Keep reasoning brief."""

    @classmethod
    def build_system_prompt(cls, task_type: str) -> str:
        """Build the system prompt for a task type with token budget info."""
        config = TASK_TOKEN_CONFIG[task_type]
        max_tokens = config["max_tokens"]
        max_reasoning = config["max_reasoning_tokens"]
        output_tokens = max_tokens - max_reasoning

        return cls.SYSTEM_PROMPT_TEMPLATE.format(
            max_tokens=max_tokens,
            max_reasoning_tokens=max_reasoning,
            output_tokens=output_tokens,
        )

    @classmethod
    def build_user_prompt(cls, task: dict[str, Any]) -> str:
        """Build the user prompt for a specific task."""
        task_type = task.get("task_type")

        if task_type == "beat_interpolation":
            return cls._build_beat_interpolation_prompt(task)
        elif task_type == "beat_revision":
            return cls._build_beat_revision_prompt(task)
        elif task_type == "constrained_continuation":
            return cls._build_constrained_continuation_prompt(task)
        elif task_type == "theory_conversion":
            return cls._build_theory_conversion_prompt(task)
        elif task_type == "multi_beat_synthesis":
            return cls._build_multi_beat_synthesis_prompt(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @classmethod
    def _build_beat_interpolation_prompt(cls, task: dict[str, Any]) -> str:
        """Build prompt for beat interpolation task."""
        beat_before = task["beat_before"]
        beat_after = task["beat_after"]
        missing_beat = task["missing_beat"]
        requirements = task["requirements"]

        # Build must_not section if present
        must_not_section = ""
        if requirements.get("must_not_include"):
            must_not_items = "\n".join(f"  - {item}" for item in requirements["must_not_include"])
            must_not_section = f"\nMUST NOT include (these would be easy but wrong):\n{must_not_items}\n"

        # Get beat definition if available
        beat_def = missing_beat.get("definition", "")
        beat_def_section = f"\nBEAT DEFINITION ({missing_beat['name']}): {beat_def}\n" if beat_def else ""

        prompt = f"""Given these two story beats from the {task['theory']}:

BEAT BEFORE ({beat_before['name']}):
{beat_before['content'].strip()}

BEAT AFTER ({beat_after['name']}):
{beat_after['content'].strip()}

Write the missing beat: {missing_beat['name']}
{beat_def_section}
Requirements:
- Word count: {requirements['word_count'][0]}-{requirements['word_count'][1]} words
- Must include ALL of:
  - {chr(10).join('  - ' + item for item in requirements['must_include'])}
- Maintain character voice: {requirements.get('character_voice', 'Consistent with the given beats')}
- Setting continuity: No contradictions with established details
- Logical bridge: Your beat must connect naturally FROM the beat before TO the beat after—ensure cause-and-effect makes sense
- Beat execution: Your beat must fulfill the narrative function of "{missing_beat['name']}" in {task['theory']}—not just connect the scenes
{must_not_section}
{task.get('generator_prompt_additions', '')}

Output ONLY the beat text."""
        return prompt

    @classmethod
    def _build_beat_revision_prompt(cls, task: dict[str, Any]) -> str:
        """Build prompt for beat revision task.

        HARDENED: Model is NOT told there is definitely a flaw. They must:
        1. Analyze the segment against the beat definition
        2. Diagnose IF there are problems (not told in advance)
        3. Fix problems if they exist

        This tests true diagnostic ability, not just following instructions.
        CONSTRAINED REVISION: Model must preserve specific sentences/phrases.
        """
        requirements = task["requirements"]

        # Build preservation constraint section if present
        preservation_section = ""
        if "preservation_requirements" in requirements:
            pres_req = requirements["preservation_requirements"]
            preserved_items = "\n".join(f'  - "{item}"' for item in pres_req.get("required_preserved", []))
            forbidden_items = "\n".join(f"  - {item}" for item in pres_req.get("forbidden_changes", []))

            # Add conflicting constraints if present (makes task harder)
            conflicts_section = ""
            if "conflicting_constraints" in requirements:
                conflicts = requirements["conflicting_constraints"]
                conflict_items = "\n".join(f"  - {c['conflict']}" for c in conflicts)
                conflicts_section = f"""
TENSION POINTS (these requirements may conflict - resolve skillfully):
{conflict_items}
"""

            preservation_section = f"""
=== CONSTRAINED REVISION REQUIREMENTS ===
{pres_req.get('min_preservation_description', 'Preserve as much of the original as possible while fixing any issues.')}

MUST PRESERVE (keep these exact or nearly exact):
{preserved_items}

FORBIDDEN CHANGES:
{forbidden_items}
{conflicts_section}
Your goal is MINIMAL MODIFICATION - address any beat execution issues while keeping as much
of the original text intact as possible. Do not rewrite from scratch.
"""

        prompt = f"""You are a professional storyteller and editor with expertise in {task['theory']}.

This story segment attempts to execute the "{task['beat_name']}" beat. Your task is to:
1. Analyze how well this segment executes the beat (compare to the beat definition)
2. Identify any execution problems (there may or may not be issues)
3. If problems exist, rewrite the segment to properly execute the beat
{preservation_section}
SEGMENT TO ANALYZE:
{task['flawed_segment']['content'].strip()}

BEAT DEFINITION ({task['beat_name']}):
{task['beat_definition'].strip()}

Requirements:
- Word count: {requirements['word_count'][0]}-{requirements['word_count'][1]} words
- Preserve these elements: {', '.join(requirements['preserve'])}
- If you identify flaws, your revision must address them
- Quality check: Do NOT introduce new narrative errors, contradictions, or problems

Output ONLY the revised segment (no explanation of what you changed). If the segment already properly executes the beat, output "NO REVISION NEEDED" followed by a brief explanation."""
        return prompt

    @classmethod
    def _build_constrained_continuation_prompt(cls, task: dict[str, Any]) -> str:
        """Build prompt for constrained continuation task."""
        opening = task["story_opening"]
        constraints = task["continuation_constraints"]

        must_include = "\n".join(f"  [_] {item}" for item in constraints["must_include"])
        must_not_include = "\n".join(f"  [X] {item}" for item in constraints["must_not_include"])

        min_wc, max_wc = constraints['word_count']
        prompt = f"""=== WORD LIMIT: {min_wc}-{max_wc} WORDS (DO NOT EXCEED {max_wc}) ===

Continue this story according to the {task['theory']} framework.

STORY OPENING:
{opening['content'].strip()}

Current beat: {opening['current_beat']}
Continue through these beats: {', '.join(constraints['next_beats'])}

=== CHECKLIST (satisfy ALL requirements) ===

MUST INCLUDE (ensure each appears in your continuation):
{must_include}

MUST NOT INCLUDE (avoid these completely):
{must_not_include}

[_] Tone: {constraints['tone']}
[_] Ending: {constraints['ending_requirement']}

⚠️ WORD COUNT: {min_wc}-{max_wc} words. Count as you write. Stop at {max_wc}.

Output ONLY the story continuation as continuous prose."""
        return prompt

    @classmethod
    def _build_theory_conversion_prompt(cls, task: dict[str, Any]) -> str:
        """Build prompt for theory conversion task."""
        original = task["original_segment"]
        target = task["target_requirements"]

        min_wc, max_wc = target['word_count']
        prompt = f"""=== WORD LIMIT: {min_wc}-{max_wc} WORDS (DO NOT EXCEED {max_wc}) ===

Rewrite this story from {task['from_theory']} structure to {task['to_theory']} structure.

ORIGINAL ({task['from_theory']}):
{original['content'].strip()}

ORIGINAL BEATS:
{', '.join(original['beats'])}

TARGET STRUCTURE ({task['to_theory']}):
Required beats: {', '.join(target['beats'])}

REQUIREMENTS:
- Preserve these elements: {', '.join(target['preserve'])}
- Tone: {target.get('tone', 'Maintain original tone')}

⚠️ WORD COUNT: {min_wc}-{max_wc} words. Exceeding {max_wc} words is FAILURE.

Output ONLY the rewritten story."""
        return prompt

    @classmethod
    def _build_multi_beat_synthesis_prompt(cls, task: dict[str, Any]) -> str:
        """Build prompt for multi-beat synthesis task."""
        context = task["story_context"]
        beats = task["beats_to_generate"]
        cross_constraints = task["cross_beat_constraints"]

        # Calculate suggested word allocation per beat
        min_words = task['word_count'][0]
        max_words = task['word_count'][1]
        num_beats = len(beats)
        words_per_beat_min = min_words // num_beats
        words_per_beat_max = max_words // num_beats

        beats_text = ""
        for i, beat in enumerate(beats, 1):
            reqs = "\n".join(f"    - {r}" for r in beat["requirements"])
            beats_text += f"""
  Beat {i}: {beat['name']} (~{words_per_beat_min}-{words_per_beat_max} words)
  Requirements:
{reqs}
"""

        constraints_text = "\n".join(
            f"  - {c['type']}: {c['requirement'].strip()}" for c in cross_constraints
        )

        prompt = f"""=== HARD WORD LIMIT: {min_words}-{max_words} WORDS (DO NOT EXCEED {max_words}) ===

Write these connected beats following the {task['theory']}:

STORY CONTEXT:
- Protagonist: {context['protagonist'].strip()}
- Setting: {context['setting'].strip()}
- Central Conflict: {context['central_conflict'].strip()}
- Tone: {context['tone']}

=== WORD BUDGET ===
Total: {min_words}-{max_words} words MAXIMUM
Per beat: ~{words_per_beat_min}-{words_per_beat_max} words each
⚠️ COUNT YOUR WORDS. Stop at {max_words}. Going over = AUTOMATIC FAILURE.

=== BEATS TO GENERATE ===
{beats_text}

=== CROSS-BEAT CONSTRAINTS (must satisfy ALL) ===
{constraints_text}

=== OUTPUT FORMAT ===
Write as CONTINUOUS NARRATIVE PROSE.
Do NOT use labels like "Beat 1:" or headers.
Transition naturally between beats.

Output ONLY the story content (no explanations)."""
        return prompt


class BenchmarkGenerator:
    """Main generator class for the benchmark."""

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.config = config or GeneratorConfig()
        self.model_config = load_config()["models"]

    def generate_sample(
        self,
        task: dict[str, Any],
        model: str,
        sample_index: int,
    ) -> GenerationResult:
        """Generate a single sample for a task."""
        task_id = task["task_id"]
        task_type = task["task_type"]
        theory = task.get("theory", "Unknown")

        # Get task-specific token configuration (same for all models - fairness)
        task_config = TASK_TOKEN_CONFIG.get(task_type, {
            "max_tokens": self.config.max_tokens,
            "max_reasoning_tokens": self.config.max_reasoning_tokens,
        })
        max_tokens = task_config["max_tokens"]
        max_reasoning_tokens = task_config["max_reasoning_tokens"]

        system_prompt = PromptBuilder.build_system_prompt(task_type)
        user_prompt = PromptBuilder.build_user_prompt(task)

        generation_id = generate_id()
        timestamp = get_timestamp()

        # Use shared LLM client with built-in retry logic
        response = self.llm_client.call(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=max_tokens,
            max_reasoning_tokens=max_reasoning_tokens,
            retry_attempts=self.config.retry_attempts,
            retry_delay=self.config.retry_delay,
        )

        if not response.success:
            logger.error(f"[{task_id}] [{model}] API error: {response.error}")
            return GenerationResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                theory=theory,
                model=model,
                sample_index=sample_index,
                output="",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                reasoning_tokens=response.reasoning_tokens,
                cost=response.cost,
                timestamp=timestamp,
                success=False,
                error=response.error,
                finish_reason=response.finish_reason,
            )

        output = response.content
        output_tokens = response.completion_tokens - response.reasoning_tokens

        # Detect empty output: tokens generated but no actual content
        if response.completion_tokens > 0 and not output.strip():
            # Distinguish between content filtering and reasoning-only
            if response.reasoning_tokens > 0 and response.reasoning_tokens >= response.completion_tokens * 0.9:
                error_msg = f"Model produced only reasoning tokens (reasoning={response.reasoning_tokens}, output={output_tokens})"
            else:
                error_msg = f"Content filtered (tokens={response.completion_tokens})"

            logger.error(f"[{task_id}] [{model}] {error_msg}")
            return GenerationResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                theory=theory,
                model=model,
                sample_index=sample_index,
                output="",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                reasoning_tokens=response.reasoning_tokens,
                cost=response.cost,
                timestamp=timestamp,
                success=False,
                error=error_msg,
                finish_reason=response.finish_reason,
            )

        # Check for non-stop finish reasons (length, content_filter, etc.)
        if response.finish_reason and response.finish_reason != "stop":
            error_msg = f"Generation failed: finish_reason={response.finish_reason}"
            logger.error(
                f"[{task_id}] [{model}] {error_msg} "
                f"(completion={response.completion_tokens}, reasoning={response.reasoning_tokens}, "
                f"output={output_tokens})"
            )
            return GenerationResult(
                generation_id=generation_id,
                task_id=task_id,
                task_type=task_type,
                theory=theory,
                model=model,
                sample_index=sample_index,
                output=output,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                reasoning_tokens=response.reasoning_tokens,
                cost=response.cost,
                timestamp=timestamp,
                success=False,
                error=error_msg,
                finish_reason=response.finish_reason,
            )

        return GenerationResult(
            generation_id=generation_id,
            task_id=task_id,
            task_type=task_type,
            theory=theory,
            model=model,
            sample_index=sample_index,
            output=output,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            reasoning_tokens=response.reasoning_tokens,
            cost=response.cost,
            timestamp=timestamp,
            success=True,
            finish_reason=response.finish_reason,
        )

    def run_benchmark(
        self,
        tasks: list[dict[str, Any]] | None = None,
        models: list[str] | None = None,
        samples_per_task: int | None = None,
    ) -> list[GenerationResult]:
        """Run the full benchmark generation."""
        if tasks is None:
            tasks = load_all_tasks()

        if models is None:
            # Default models - one from each tier
            models = [
                "anthropic/claude-sonnet-4",  # Strong tier
                "anthropic/claude-3-5-haiku",  # Cheap tier
            ]

        if samples_per_task is None:
            samples_per_task = self.config.samples_per_task

        results = []
        total_iterations = len(tasks) * len(models) * samples_per_task

        with tqdm(total=total_iterations, desc="Generating") as pbar:
            for task in tasks:
                for model in models:
                    for sample_idx in range(samples_per_task):
                        result = self.generate_sample(task, model, sample_idx)
                        results.append(result)

                        # Save incrementally
                        file_path = get_generation_path(
                            result.task_id, result.model, result.sample_index
                        )
                        save_yaml(result.to_dict(), file_path)

                        pbar.update(1)
                        pbar.set_postfix(
                            task=task["task_id"][:20],
                            model=model.split("/")[-1][:10],
                            cost=f"${sum(r.cost for r in results):.4f}",
                        )

        return results

    def generate_cost_report(self, results: list[GenerationResult]) -> dict[str, Any]:
        """Generate a cost report from generation results."""
        total_cost = sum(r.cost for r in results)
        total_prompt_tokens = sum(r.prompt_tokens for r in results)
        total_completion_tokens = sum(r.completion_tokens for r in results)

        by_model: dict[str, dict[str, Any]] = {}
        for r in results:
            if r.model not in by_model:
                by_model[r.model] = {
                    "count": 0,
                    "cost": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "successes": 0,
                    "failures": 0,
                }
            by_model[r.model]["count"] += 1
            by_model[r.model]["cost"] += r.cost
            by_model[r.model]["prompt_tokens"] += r.prompt_tokens
            by_model[r.model]["completion_tokens"] += r.completion_tokens
            if r.success:
                by_model[r.model]["successes"] += 1
            else:
                by_model[r.model]["failures"] += 1

        by_task_type: dict[str, dict[str, Any]] = {}
        for r in results:
            if r.task_type not in by_task_type:
                by_task_type[r.task_type] = {"count": 0, "cost": 0}
            by_task_type[r.task_type]["count"] += 1
            by_task_type[r.task_type]["cost"] += r.cost

        return {
            "total_generations": len(results),
            "total_cost": total_cost,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
            "by_model": by_model,
            "by_task_type": by_task_type,
        }


if __name__ == "__main__":
    # Test prompt building
    print("Testing prompt builder...")

    tasks = load_all_tasks()
    if tasks:
        for task_type in [
            "beat_interpolation",
            "beat_revision",
            "constrained_continuation",
            "theory_conversion",
            "multi_beat_synthesis",
        ]:
            type_tasks = [t for t in tasks if t.get("task_type") == task_type]
            if type_tasks:
                prompt = PromptBuilder.build_user_prompt(type_tasks[0])
                print(f"\n{task_type} prompt preview (first 500 chars):")
                print(prompt[:500] + "...")

    print("\n\nTo run the full benchmark, use:")
    print("  generator = BenchmarkGenerator()")
    print("  results = generator.run_benchmark()")
