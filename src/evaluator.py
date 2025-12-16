"""
Evaluation module for the Story Theory Benchmark.

This module handles LLM-as-judge evaluation of generated story outputs.
Now includes multi-component scoring (word count + LLM + programmatic).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from scoring import (
    ScoringWeights,
    ScoreBreakdown,
    calculate_final_score,
    count_words,
)
from utils import (
    extract_json_from_response,
    generate_id,
    get_evaluation_path,
    get_project_root,
    get_timestamp,
    load_all_tasks,
    load_config,
    load_yaml,
    save_yaml,
)

load_dotenv()


@dataclass
class EvaluationResult:
    """Result of evaluating a single generation with composite scoring."""

    evaluation_id: str
    task_id: str
    generation_id: str
    task_type: str
    model: str
    sample_index: int
    evaluator_model: str
    evaluator_cost: float
    timestamp: str
    success: bool
    # Scoring fields
    final_score: float
    score_breakdown: dict[str, Any]
    llm_results: dict[str, Any]  # Raw LLM evaluation results
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "task_id": self.task_id,
            "generation_id": self.generation_id,
            "task_type": self.task_type,
            "model": self.model,
            "sample_index": self.sample_index,
            "evaluator_model": self.evaluator_model,
            "evaluator_cost": self.evaluator_cost,
            "timestamp": self.timestamp,
            "success": self.success,
            "final_score": self.final_score,
            "score_breakdown": self.score_breakdown,
            "llm_results": self.llm_results,
            "error": self.error,
        }


class EvalPromptBuilder:
    """Builds evaluation prompts for different task types."""

    SYSTEM_PROMPT = """Evaluate story outputs against criteria. Be STRICT and CALIBRATED using this scale:

SCORING ANCHORS (use these to calibrate your scores):
- 1.0 = Exceptional: Clearly exceeds requirements, professional quality
- 0.8 = Good: Fully satisfies requirements with only minor issues
- 0.6 = Adequate: Meets basic requirements but has noticeable gaps or weaknesses
- 0.4 = Weak: Partially meets requirements with significant issues
- 0.2 = Poor: Barely attempts requirements, major problems
- 0.0 = Missing: Completely fails or doesn't attempt the requirement

Respond with valid JSON only."""

    # Story theory reference for evaluators
    THEORY_DEFINITIONS = {
        "Hero's Journey": """Joseph Campbell's 12-stage monomyth: Ordinary World, Call to Adventure, Refusal of Call, Meeting the Mentor, Crossing the Threshold, Tests/Allies/Enemies, Approach to Inmost Cave, Ordeal, Reward, The Road Back, Resurrection, Return with Elixir.""",
        "Save the Cat": """Blake Snyder's 15 beats: Opening Image, Theme Stated, Set-Up, Catalyst, Debate, Break into Two, B Story, Fun and Games, Midpoint, Bad Guys Close In, All Is Lost, Dark Night of the Soul, Break into Three, Finale, Final Image.""",
        "Story Circle": """Dan Harmon's 8 steps: You (comfort zone), Need (want something), Go (unfamiliar situation), Search (adapt), Find (get what wanted), Take (pay price), Return (go back), Change (capable of change).""",
        "Freytag's Pyramid": """Gustav Freytag's 5 stages: Exposition (setup), Rising Action (complications), Climax (turning point), Falling Action (consequences), Resolution (denouement).""",
        "Three-Act Structure": """Aristotle/Syd Field's 3 acts: Act 1 Setup (introduce world/characters/conflict), Act 2 Confrontation (complications/obstacles), Act 3 Resolution (climax/resolution).""",
    }

    @classmethod
    def build_eval_prompt(cls, task: dict[str, Any], generated_output: str, word_count: int = 0) -> str:
        """Build the evaluation prompt for a specific task and output."""
        task_type = task.get("task_type")

        if task_type == "beat_interpolation":
            return cls._build_beat_interpolation_eval(task, generated_output, word_count)
        elif task_type == "beat_revision":
            return cls._build_beat_revision_eval(task, generated_output, word_count)
        elif task_type == "constrained_continuation":
            return cls._build_constrained_continuation_eval(task, generated_output, word_count)
        elif task_type == "theory_conversion":
            return cls._build_theory_conversion_eval(task, generated_output, word_count)
        elif task_type == "multi_beat_synthesis":
            return cls._build_multi_beat_synthesis_eval(task, generated_output, word_count)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @classmethod
    def _build_beat_interpolation_eval(cls, task: dict[str, Any], output: str, word_count: int) -> str:
        """Build evaluation prompt for beat interpolation with partial credit scoring."""
        beat_before = task["beat_before"]
        beat_after = task["beat_after"]
        missing_beat = task["missing_beat"]
        requirements = task["requirements"]
        theory = task.get("theory", "Unknown")
        theory_def = cls.THEORY_DEFINITIONS.get(theory, "")

        must_include_list = requirements["must_include"]
        must_include_str = ", ".join(must_include_list)
        num_elements = len(must_include_list)
        word_count_min, word_count_max = requirements['word_count']
        word_count_valid = word_count_min <= word_count <= word_count_max

        # Build must_not section if present
        must_not_list = requirements.get("must_not_include", [])
        must_not_str = ", ".join(must_not_list) if must_not_list else "None specified"
        num_violations = len(must_not_list)

        # Get beat definition if available
        beat_def = missing_beat.get("definition", f"The '{missing_beat['name']}' beat in {theory}")

        return f"""Theory: {theory} - {theory_def}

BEFORE ({beat_before['name']}): {beat_before['content'].strip()[:800]}

GENERATED ({missing_beat['name']}): {output}

AFTER ({beat_after['name']}): {beat_after['content'].strip()[:800]}

BEAT DEFINITION: {beat_def}

Check: word_count={word_count} (valid={word_count_valid}, range={word_count_min}-{word_count_max})
Required elements ({num_elements} total): {must_include_str}
Must NOT include ({num_violations} items): {must_not_str}

PARTIAL CREDIT SCORING - Rate each criterion 0.0-1.0:
- beat_elements_score: What fraction of {num_elements} required elements are present? (0.0-1.0)
- beat_execution_score: Does the output actually execute the "{missing_beat['name']}" beat's narrative function? (0.0=wrong beat/generic content, 0.5=partially, 1.0=clearly executes the beat)
- must_not_score: Are the must-not items avoided? (1.0=all avoided, deduct proportionally for violations)
- character_score: How consistent is character voice/behavior? (0.0=inconsistent, 1.0=perfect)
- bridge_score: How well does it bridge the two beats? (0.0=no connection, 1.0=seamless)
- continuity_score: How well is setting continuity maintained? (0.0=contradictions, 1.0=perfect)

JSON response:
{{"beat_elements_score":float,"elements_found":int,"elements_total":{num_elements},"beat_execution_score":float,"must_not_score":float,"violations_found":int,"word_count_valid":{str(word_count_valid).lower()},"character_score":float,"bridge_score":float,"continuity_score":float,"evidence":"brief explanation of any deductions"}}"""

    @classmethod
    def _build_beat_revision_eval(cls, task: dict[str, Any], output: str, word_count: int) -> str:
        """Build evaluation prompt for beat revision.

        Note: The generator was NOT told the flaw or fix - they had to identify it themselves.
        This evaluator checks if they correctly identified and fixed the right problem.

        CONSTRAINED REVISION: Also evaluates minimal modification (how much was preserved).
        NO-FLAW TASKS: Some tasks have no flaw - tests if model correctly identifies "NO REVISION NEEDED".
        """
        requirements = task["requirements"]
        theory = task.get("theory", "Unknown")
        theory_def = cls.THEORY_DEFINITIONS.get(theory, "")
        preserve_list = requirements["preserve"]
        preserve_str = ", ".join(preserve_list)
        num_preserve = len(preserve_list)
        word_count_min, word_count_max = requirements['word_count']
        word_count_valid = word_count_min <= word_count <= word_count_max

        # Check if this is a "no flaw" task (correctly executed beat)
        ground_truth = task.get("ground_truth", {})
        has_flaw = ground_truth.get("has_flaw", True)  # Default to True for backward compat

        if not has_flaw:
            # Special evaluation for "no flaw" tasks
            return cls._build_no_flaw_revision_eval(task, output, theory, theory_def)

        # Build preservation requirements section if present
        preservation_eval_section = ""
        required_preserved_str = ""
        num_required_preserved = 0
        if "preservation_requirements" in requirements:
            pres_req = requirements["preservation_requirements"]
            required_preserved = pres_req.get("required_preserved", [])
            num_required_preserved = len(required_preserved)
            required_preserved_str = "\n".join(f'  - "{item}"' for item in required_preserved)
            forbidden_changes = pres_req.get("forbidden_changes", [])
            forbidden_str = "\n".join(f"  - {item}" for item in forbidden_changes)

            preservation_eval_section = f"""
=== CONSTRAINED REVISION EVALUATION ===
The model was instructed to make MINIMAL modifications while fixing the flaw.

Required preserved phrases ({num_required_preserved} items - check if present in revision):
{required_preserved_str}

Forbidden changes:
{forbidden_str}

- required_preserved_score: How many of {num_required_preserved} required phrases are present (exact or near-exact)? (0.0-1.0)
- minimal_change_score: How much of the original text structure was preserved? (0.0=complete rewrite, 0.5=moderate changes, 1.0=minimal surgical edits)
"""

        return f"""Theory: {theory} - {theory_def}

ORIGINAL FLAWED SEGMENT: {task['flawed_segment']['content'].strip()[:800]}

THE ACTUAL FLAW (model was NOT told this): {task['flawed_segment']['flaw_description'].strip()}
THE REQUIRED FIX (model was NOT told this): {requirements['fix']}

MODEL'S REVISION: {output}

Beat "{task['beat_name']}": {task['beat_definition'].strip()[:600]}

Check: word_count={word_count} (valid={word_count_valid}, range={word_count_min}-{word_count_max})
Preserve ({num_preserve} elements): {preserve_str}
{preservation_eval_section}
SCORING - Did the model correctly identify and fix the flaw without being told?
- diagnosis_score: Did the revision address the ACTUAL flaw listed above? (0.0=wrong problem, 0.5=partially, 1.0=correct diagnosis)
- flaw_correction_score: How completely is the actual flaw fixed? (0.0=not fixed, 0.5=partially, 1.0=fully fixed)
- beat_satisfaction_score: How well does it execute the beat definition? (0.0-1.0)
- preservation_score: What fraction of {num_preserve} elements are preserved? (0.0-1.0)
- required_preserved_score: What fraction of {num_required_preserved} required phrases are present? (0.0-1.0, use 1.0 if no requirements)
- minimal_change_score: How minimal were the changes? (0.0=complete rewrite, 1.0=surgical minimal edits)
- quality_score: Overall writing quality, no new errors introduced (0.0-1.0)

JSON response:
{{"diagnosis_score":float,"flaw_correction_score":float,"beat_satisfaction_score":float,"preservation_score":float,"preserved_count":int,"preserved_total":{num_preserve},"required_preserved_score":float,"required_preserved_count":int,"required_preserved_total":{num_required_preserved},"minimal_change_score":float,"word_count_valid":{str(word_count_valid).lower()},"quality_score":float,"evidence":"what flaw did the model appear to address? was it correct? how minimal were the changes?"}}"""

    @classmethod
    def _build_no_flaw_revision_eval(cls, task: dict[str, Any], output: str, theory: str, theory_def: str) -> str:
        """Build evaluation prompt for beat revision tasks where the segment is CORRECTLY executed.

        These tasks test whether models can recognize when NO revision is needed.
        Over-diagnosis (revising a well-executed beat) should be penalized.
        """
        beat_name = task.get("beat_name", "Unknown Beat")
        beat_definition = task.get("beat_definition", "")
        original_segment = task["flawed_segment"]["content"].strip()
        ground_truth = task.get("ground_truth", {})
        expected_reasoning = ground_truth.get("reasoning", "")

        return f"""Theory: {theory} - {theory_def}

ORIGINAL SEGMENT (this is CORRECTLY EXECUTED - no flaw exists):
{original_segment[:1000]}

Beat "{beat_name}" definition:
{beat_definition[:600]}

MODEL'S RESPONSE:
{output}

=== NO-FLAW TASK EVALUATION ===
This segment was CORRECTLY executed and needed NO revision. The model should have recognized this.

Expected correct response: "NO REVISION NEEDED" (or equivalent)

Ground truth reasoning why no revision is needed:
{expected_reasoning[:500]}

SCORING - Did the model correctly recognize this as a well-executed beat?

- correct_diagnosis_score: Did the model correctly identify that NO revision is needed?
  (1.0=clearly stated no revision needed, 0.5=ambiguous/hedged, 0.0=attempted unnecessary revision)

- beat_understanding_score: Did the model demonstrate understanding of the beat requirements?
  (1.0=accurately explained why the beat works, 0.5=partial understanding, 0.0=misunderstood beat)

- false_positive_avoided_score: Did the model AVOID making unnecessary changes?
  (1.0=no changes made/suggested, 0.5=minor suggestions, 0.0=rewrote or heavily modified)

- reasoning_quality_score: If model said "no revision needed", how good was the explanation?
  (1.0=comprehensive and accurate reasoning, 0.5=partial reasoning, 0.0=no reasoning or wrong reasoning)

JSON response:
{{"correct_diagnosis_score":float,"beat_understanding_score":float,"false_positive_avoided_score":float,"reasoning_quality_score":float,"model_said_no_revision":bool,"evidence":"Did model recognize segment was fine? What did model do/say?"}}"""

    @classmethod
    def _build_constrained_continuation_eval(cls, task: dict[str, Any], output: str, word_count: int) -> str:
        """Build evaluation prompt for constrained continuation with partial credit scoring."""
        opening = task["story_opening"]
        constraints = task["continuation_constraints"]
        theory = task.get("theory", "Unknown")
        theory_def = cls.THEORY_DEFINITIONS.get(theory, "")

        beats_list = constraints["next_beats"]
        num_beats = len(beats_list)
        beats_str = ", ".join(beats_list)
        must_include_list = constraints["must_include"]
        num_must_include = len(must_include_list)
        must_include_str = "; ".join(must_include_list)
        must_not_list = constraints["must_not_include"]
        num_must_not = len(must_not_list)
        must_not_str = "; ".join(must_not_list)
        word_count_min, word_count_max = constraints['word_count']
        word_count_valid = word_count_min <= word_count <= word_count_max

        return f"""Theory: {theory} - {theory_def}

OPENING: {opening['content'].strip()[:800]}

CONTINUATION: {output}

Constraints to check:
- {num_beats} Beats required: {beats_str}
- word_count={word_count} (valid={word_count_valid}, range={word_count_min}-{word_count_max})
- {num_must_include} MUST INCLUDE: {must_include_str}
- {num_must_not} MUST NOT INCLUDE: {must_not_str}
- Tone: {constraints['tone']}
- Ending: {constraints['ending_requirement']}

PARTIAL CREDIT SCORING - Count what's present/absent:
- beats_score: How many of {num_beats} required beats are present? (count/{num_beats})
- must_include_score: How many of {num_must_include} required elements are included? (count/{num_must_include})
- must_not_score: How many of {num_must_not} forbidden elements are successfully avoided? (count/{num_must_not})
- tone_score: How well is the required tone maintained? (0.0-1.0)
- ending_score: How well is the ending requirement met? (0.0-1.0)

JSON response:
{{"beats_present":int,"beats_total":{num_beats},"beats_score":float,"must_include_present":int,"must_include_total":{num_must_include},"must_include_score":float,"must_not_avoided":int,"must_not_total":{num_must_not},"must_not_score":float,"word_count_valid":{str(word_count_valid).lower()},"tone_score":float,"ending_score":float,"failed_constraints":["list specific failures"]}}"""

    @classmethod
    def _build_theory_conversion_eval(cls, task: dict[str, Any], output: str, word_count: int) -> str:
        """Build evaluation prompt for theory conversion with partial credit scoring."""
        original = task["original_segment"]
        target = task["target_requirements"]
        from_theory = task["from_theory"]
        to_theory = task["to_theory"]
        from_theory_def = cls.THEORY_DEFINITIONS.get(from_theory, "")
        to_theory_def = cls.THEORY_DEFINITIONS.get(to_theory, "")

        beats_list = target["beats"]
        num_beats = len(beats_list)
        beats_str = ", ".join(beats_list)
        preserve_list = target["preserve"]
        num_preserve = len(preserve_list)
        preserve_str = ", ".join(preserve_list)
        word_count_min, word_count_max = target['word_count']
        word_count_valid = word_count_min <= word_count <= word_count_max

        return f"""Convert {from_theory} -> {to_theory}
Source: {from_theory_def}
Target: {to_theory_def}

ORIGINAL: {original['content'].strip()[:1000]}

CONVERTED: {output}

Criteria:
- {num_beats} Target beats: {beats_str}
- {num_preserve} Elements to preserve: {preserve_str}
- word_count={word_count} (valid={word_count_valid}, range={word_count_min}-{word_count_max})
- Tone: {target.get('tone', 'Maintain original')}

PARTIAL CREDIT SCORING:
- beats_score: How many of {num_beats} target beats are present? (count/{num_beats})
- preservation_score: How many of {num_preserve} core elements are preserved? (count/{num_preserve})
- structural_accuracy_score: How well does it follow the target theory structure? (0.0-1.0)
- tone_score: How well is the tone maintained? (0.0-1.0)

JSON response:
{{"beats_present":int,"beats_total":{num_beats},"beats_score":float,"preserved_count":int,"preserved_total":{num_preserve},"preservation_score":float,"word_count_valid":{str(word_count_valid).lower()},"structural_accuracy_score":float,"tone_score":float,"evidence":"brief explanation of any deductions"}}"""

    @classmethod
    def _build_multi_beat_synthesis_eval(cls, task: dict[str, Any], output: str, word_count: int) -> str:
        """Build evaluation prompt for multi-beat synthesis with partial credit scoring."""
        beats = task["beats_to_generate"]
        cross_constraints = task["cross_beat_constraints"]
        context = task["story_context"]
        theory = task.get("theory", "Unknown")
        theory_def = cls.THEORY_DEFINITIONS.get(theory, "")

        num_beats = len(beats)
        total_reqs = 0
        beats_reqs = ""
        for i, beat in enumerate(beats, 1):
            reqs = beat["requirements"]
            total_reqs += len(reqs)
            reqs_str = "; ".join(reqs)
            beats_reqs += f"Beat {i} ({beat['name']}, {len(reqs)} reqs): {reqs_str}\n"

        num_cross = len(cross_constraints)
        cross_str = "; ".join(f"{c['type']}: {c['requirement']}" for c in cross_constraints)
        word_count_min, word_count_max = task['word_count']
        word_count_valid = word_count_min <= word_count <= word_count_max

        return f"""Theory: {theory} - {theory_def}

Context: {context['protagonist'][:300]} | {context['setting'][:200]} | {context['central_conflict'][:200]} | Tone: {context['tone']}

GENERATED: {output}

word_count={word_count} (valid={word_count_valid}, range={word_count_min}-{word_count_max})

{num_beats} Beats with {total_reqs} total requirements:
{beats_reqs}
{num_cross} Cross-beat constraints: {cross_str}

PARTIAL CREDIT SCORING:
- beat_requirements_score: What fraction of {total_reqs} beat requirements are satisfied? (count/{total_reqs})
- cross_beat_score: What fraction of {num_cross} cross-beat constraints are satisfied? (count/{num_cross})
- context_score: How well is the story context maintained (protagonist, setting, tone)? (0.0-1.0)
- coherence_score: How well do the beats flow together as a unified narrative? (0.0-1.0)

JSON response:
{{"beat_reqs_satisfied":int,"beat_reqs_total":{total_reqs},"beat_requirements_score":float,"cross_beat_satisfied":int,"cross_beat_total":{num_cross},"cross_beat_score":float,"word_count_valid":{str(word_count_valid).lower()},"context_score":float,"coherence_score":float,"failed_items":["list specific failures"]}}"""


def get_word_count_range(task: dict[str, Any]) -> tuple[int, int]:
    """Extract word count range from a task based on task type."""
    task_type = task.get("task_type")

    if task_type == "beat_interpolation":
        return tuple(task.get("requirements", {}).get("word_count", [300, 500]))
    elif task_type == "beat_revision":
        return tuple(task.get("requirements", {}).get("word_count", [300, 500]))
    elif task_type == "constrained_continuation":
        return tuple(task.get("continuation_constraints", {}).get("word_count", [400, 600]))
    elif task_type == "theory_conversion":
        return tuple(task.get("target_requirements", {}).get("word_count", [500, 800]))
    elif task_type == "multi_beat_synthesis":
        return tuple(task.get("word_count", [600, 900]))
    else:
        return (300, 600)  # Default


class BenchmarkEvaluator:
    """Main evaluator class for the benchmark."""

    def __init__(
        self,
        api_key: str | None = None,
        evaluator_model: str = "anthropic/claude-haiku-4.5",
        scoring_weights: ScoringWeights | None = None,
        word_count_method: str = "gaussian",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.evaluator_model = evaluator_model
        self.model_config = load_config()["models"]
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.word_count_method = word_count_method

    def get_evaluator_pricing(self) -> tuple[float, float]:
        """Get pricing for the evaluator model."""
        pricing = {
            "anthropic/claude-haiku-4.5": (5.0, 25.0),
            "openai/gpt-4o-mini": (0.15, 0.60),
            "google/gemini-flash-1.5": (0.075, 0.30),
            "google/gemini-2.5-flash": (0.30, 2.50),
        }
        return pricing.get(self.evaluator_model, (0.25, 1.0))

    def evaluate_generation(
        self,
        task: dict[str, Any],
        generation: dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate a single generation and compute composite score."""
        task_id = task["task_id"]
        task_type = task["task_type"]
        generation_id = generation.get("generation_id", "unknown")
        model = generation.get("model", "unknown")
        sample_index = generation.get("sample_index", 0)
        generated_output = generation.get("output", "")
        word_count = generation.get("word_count", 0)

        eval_id = generate_id()
        timestamp = get_timestamp()

        if not generated_output:
            return EvaluationResult(
                evaluation_id=eval_id,
                task_id=task_id,
                generation_id=generation_id,
                task_type=task_type,
                model=model,
                sample_index=sample_index,
                evaluator_model=self.evaluator_model,
                evaluator_cost=0,
                timestamp=timestamp,
                success=False,
                final_score=0.0,
                score_breakdown={},
                llm_results={},
                error="Empty generation output",
            )

        eval_prompt = EvalPromptBuilder.build_eval_prompt(task, generated_output, word_count)

        try:
            response = self.client.chat.completions.create(
                model=self.evaluator_model,
                messages=[
                    {"role": "system", "content": EvalPromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0.1,
                max_tokens=2500,  # Prevent truncation of evidence field
            )

            response_text = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            input_price, output_price = self.get_evaluator_pricing()
            cost = (prompt_tokens / 1_000_000) * input_price + (
                completion_tokens / 1_000_000
            ) * output_price

            # Parse JSON response
            llm_results = extract_json_from_response(response_text)

            if llm_results is None:
                return EvaluationResult(
                    evaluation_id=eval_id,
                    task_id=task_id,
                    generation_id=generation_id,
                    task_type=task_type,
                    model=model,
                    sample_index=sample_index,
                    evaluator_model=self.evaluator_model,
                    evaluator_cost=cost,
                    timestamp=timestamp,
                    success=False,
                    final_score=0.0,
                    score_breakdown={},
                    llm_results={"raw_response": response_text},
                    error="Failed to parse JSON from evaluator response",
                )

            # Calculate composite score
            word_range = get_word_count_range(task)
            score_breakdown = calculate_final_score(
                text=generated_output,
                word_count=word_count,
                target_word_range=word_range,
                llm_results=llm_results,
                task_type=task_type,
                weights=self.scoring_weights,
                word_count_method=self.word_count_method,
            )

            return EvaluationResult(
                evaluation_id=eval_id,
                task_id=task_id,
                generation_id=generation_id,
                task_type=task_type,
                model=model,
                sample_index=sample_index,
                evaluator_model=self.evaluator_model,
                evaluator_cost=cost,
                timestamp=timestamp,
                success=True,
                final_score=score_breakdown.final_score,
                score_breakdown=score_breakdown.to_dict(),
                llm_results=llm_results,
            )

        except Exception as e:
            return EvaluationResult(
                evaluation_id=eval_id,
                task_id=task_id,
                generation_id=generation_id,
                task_type=task_type,
                model=model,
                sample_index=sample_index,
                evaluator_model=self.evaluator_model,
                evaluator_cost=0,
                timestamp=timestamp,
                success=False,
                final_score=0.0,
                score_breakdown={},
                llm_results={},
                error=str(e),
            )

    def run_evaluation(
        self,
        generations_dir: str | Path | None = None,
    ) -> list[EvaluationResult]:
        """Run evaluation on all generations in a directory."""
        if generations_dir is None:
            generations_dir = get_project_root() / "results" / "generations"
        else:
            generations_dir = Path(generations_dir)

        # Load all tasks for lookup
        tasks = load_all_tasks()
        task_lookup = {t["task_id"]: t for t in tasks}

        # Find all generation files
        gen_files = list(generations_dir.glob("*.yaml"))

        results = []
        with tqdm(total=len(gen_files), desc="Evaluating") as pbar:
            for gen_file in gen_files:
                generation = load_yaml(gen_file)
                task_id = generation.get("task_id")

                if task_id not in task_lookup:
                    print(f"Warning: Task {task_id} not found, skipping")
                    pbar.update(1)
                    continue

                task = task_lookup[task_id]
                result = self.evaluate_generation(task, generation)
                results.append(result)

                # Save incrementally
                file_path = get_evaluation_path(result.generation_id)
                save_yaml(result.to_dict(), file_path)

                pbar.update(1)
                pbar.set_postfix(
                    task=task_id[:15],
                    score=f"{result.final_score:.2f}",
                )

        return results

    def generate_evaluation_report(
        self, results: list[EvaluationResult]
    ) -> dict[str, Any]:
        """Generate a summary report from evaluation results."""
        import statistics

        total = len(results)
        successful = [r for r in results if r.success]
        total_cost = sum(r.evaluator_cost for r in results)
        errors = sum(1 for r in results if not r.success)

        # Calculate average scores
        scores = [r.final_score for r in successful]
        avg_score = statistics.mean(scores) if scores else 0.0

        # By model
        by_model: dict[str, dict[str, Any]] = {}
        for r in successful:
            if r.model not in by_model:
                by_model[r.model] = {"total": 0, "scores": []}
            by_model[r.model]["total"] += 1
            by_model[r.model]["scores"].append(r.final_score)

        # By task type
        by_task_type: dict[str, dict[str, Any]] = {}
        for r in successful:
            if r.task_type not in by_task_type:
                by_task_type[r.task_type] = {"total": 0, "scores": []}
            by_task_type[r.task_type]["total"] += 1
            by_task_type[r.task_type]["scores"].append(r.final_score)

        return {
            "summary": {
                "total_evaluations": total,
                "successful": len(successful),
                "errors": errors,
                "avg_score": avg_score,
                "total_cost": total_cost,
            },
            "by_model": {
                model: {
                    "total": data["total"],
                    "avg_score": statistics.mean(data["scores"]) if data["scores"] else 0.0,
                }
                for model, data in by_model.items()
            },
            "by_task_type": {
                tt: {
                    "total": data["total"],
                    "avg_score": statistics.mean(data["scores"]) if data["scores"] else 0.0,
                }
                for tt, data in by_task_type.items()
            },
        }


if __name__ == "__main__":
    print("Evaluator module loaded successfully.")
    print("\nTo run evaluation:")
    print("  evaluator = BenchmarkEvaluator()")
    print("  results = evaluator.run_evaluation()")
    print("  report = evaluator.generate_evaluation_report(results)")
