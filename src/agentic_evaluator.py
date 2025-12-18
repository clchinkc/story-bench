"""
Agentic evaluation module for the Story Theory Benchmark.

This module handles evaluation of multi-turn agentic task outputs.
Evaluates both the process (questions asked, plan quality, revision trajectory)
and the final output quality.
"""

from dataclasses import dataclass
from typing import Any

from agentic_generator import AgenticPromptBuilder
from llm_client import LLMClient, get_llm_client
from utils import (
    extract_json_from_response,
    generate_id,
    get_timestamp,
    load_config,
)


@dataclass
class AgenticEvaluationResult:
    """Result of evaluating an agentic generation."""

    evaluation_id: str
    task_id: str
    generation_id: str
    agentic_type: str
    model: str
    evaluator_model: str
    evaluator_cost: float
    timestamp: str
    success: bool

    # Process scores (specific to agentic type)
    process_scores: dict[str, float]

    # Output scores (shared)
    output_scores: dict[str, float]

    # Final composite score
    final_score: float

    # Raw LLM evaluation results
    llm_results: dict[str, Any]

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "task_id": self.task_id,
            "generation_id": self.generation_id,
            "agentic_type": self.agentic_type,
            "model": self.model,
            "evaluator_model": self.evaluator_model,
            "evaluator_cost": self.evaluator_cost,
            "timestamp": self.timestamp,
            "success": self.success,
            "process_scores": self.process_scores,
            "output_scores": self.output_scores,
            "final_score": self.final_score,
            "llm_results": self.llm_results,
            "error": self.error,
        }


class AgenticEvalPromptBuilder:
    """Builds evaluation prompts for agentic task types."""

    SYSTEM_PROMPT = """Evaluate agentic task outputs. Be STRICT and CALIBRATED.

SCORING ANCHORS:
- 1.0 = Exceptional: Clearly exceeds requirements
- 0.8 = Good: Fully satisfies requirements with minor issues
- 0.6 = Adequate: Meets basic requirements with gaps
- 0.4 = Weak: Partially meets requirements
- 0.2 = Poor: Major problems
- 0.0 = Missing: Complete failure

Respond with valid JSON only."""

    @classmethod
    def build_constraint_discovery_eval(
        cls,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        """Build evaluation prompt for constraint discovery."""
        turns = result.get("turns", [])
        final_output = result.get("final_output", "") or result.get("output", "")
        questions_asked = result.get("metrics", {}).get("questions_asked", 0)
        constraints_discovered = result.get("metrics", {}).get("constraints_discovered", 0)

        # Extract questions and answers
        qa_pairs = []
        for i, turn in enumerate(turns):
            if turn.get("turn_type") == "question":
                answer_turn = turns[i + 1] if i + 1 < len(turns) else None
                answer = answer_turn.get("content", "N/A") if answer_turn else "N/A"
                qa_pairs.append(f"Q: {turn.get('content', '')}\nA: {answer}")

        qa_text = "\n\n".join(qa_pairs) if qa_pairs else "No questions asked"

        hidden_constraints = task.get("hidden_constraints", [])
        constraints_list = "\n".join(
            f"- {c['constraint']}" for c in hidden_constraints
        )
        total_constraints = len(hidden_constraints)

        return f"""CONSTRAINT DISCOVERY TASK EVALUATION

Hidden constraints (model did NOT see these):
{constraints_list}

Questions asked ({questions_asked} total):
{qa_text}

Final output:
{final_output}

PROCESS EVALUATION:
- discovery_efficiency: How many of {total_constraints} constraints were discoverable through the questions? (0.0-1.0)
- question_quality: Were questions strategic, well-formed, and targeted? (0.0-1.0)
- question_coverage: Did questions cover different constraint areas? (0.0-1.0)

OUTPUT EVALUATION:
- constraint_satisfaction: How many discovered constraints are properly incorporated? (0.0-1.0)
- beat_execution: Are required beats properly executed? (0.0-1.0)
- narrative_quality: Overall story quality, coherence, engagement (0.0-1.0)

JSON response:
{{"discovery_efficiency":float,"constraints_discoverable":int,"constraints_total":{total_constraints},"question_quality":float,"question_coverage":float,"constraint_satisfaction":float,"beat_execution":float,"narrative_quality":float,"evidence":"brief analysis"}}"""

    @classmethod
    def build_planning_execution_eval(
        cls,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        """Build evaluation prompt for planning-execution."""
        turns = result.get("turns", [])
        final_output = result.get("final_output", "") or result.get("output", "")

        # Extract plan
        plan = ""
        for turn in turns:
            if turn.get("turn_type") == "plan":
                plan = turn.get("content", "")
                break

        constraints = task.get("constraints", [])
        constraints_str = "\n".join(f"- {c}" for c in constraints)
        num_constraints = len(constraints)

        required_beats = task.get("required_beats", [])
        beats_str = ", ".join(required_beats)
        num_beats = len(required_beats)

        return f"""PLANNING-EXECUTION TASK EVALUATION

Constraints:
{constraints_str}

Required beats: {beats_str}

Model's plan:
{plan[:2000]}{"..." if len(plan) > 2000 else ""}

Final output:
{final_output}

PROCESS EVALUATION:
- plan_completeness: Does plan address all constraints and beats? (0.0-1.0)
- plan_specificity: Is plan concrete with specific events/details? (0.0-1.0)
- plan_adherence: Does final output follow the plan? (0.0-1.0)

OUTPUT EVALUATION:
- constraint_satisfaction: How many of {num_constraints} constraints are satisfied? (0.0-1.0)
- beat_execution: How many of {num_beats} beats are properly executed? (0.0-1.0)
- narrative_quality: Overall story quality, pacing, tension (0.0-1.0)

JSON response:
{{"plan_completeness":float,"plan_specificity":float,"plan_adherence":float,"constraint_satisfaction":float,"constraints_satisfied":int,"constraints_total":{num_constraints},"beat_execution":float,"beats_satisfied":int,"beats_total":{num_beats},"narrative_quality":float,"evidence":"brief analysis"}}"""

    @classmethod
    def build_iterative_revision_eval(
        cls,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        """Build evaluation prompt for iterative revision."""
        turns = result.get("turns", [])
        final_output = result.get("final_output", "") or result.get("output", "")
        revision_count = result.get("metrics", {}).get("revision_count", 0)

        # Extract versions
        versions = []
        for turn in turns:
            if turn.get("turn_type") == "generation":
                rev_num = turn.get("metadata", {}).get("revision", 0)
                versions.append((rev_num, turn.get("content", "")[:500]))

        versions_text = "\n\n".join(
            f"VERSION {v[0]}:\n{v[1]}..." for v in versions
        )

        constraints = task.get("constraints", [])
        constraints_str = "\n".join(f"- {c}" for c in constraints)
        num_constraints = len(constraints)

        return f"""ITERATIVE REVISION TASK EVALUATION

Constraints:
{constraints_str}

Revision trajectory ({revision_count} revisions):
{versions_text}

Final output:
{final_output}

PROCESS EVALUATION:
- improvement_trajectory: Did quality improve across revisions? (0.0-1.0)
- feedback_responsiveness: Were feedback points addressed? (0.0-1.0)
- preservation: Were good elements preserved across revisions? (0.0-1.0)

OUTPUT EVALUATION:
- constraint_satisfaction: How many of {num_constraints} constraints are satisfied? (0.0-1.0)
- beat_execution: Are required beats properly executed? (0.0-1.0)
- narrative_quality: Final story quality (0.0-1.0)

JSON response:
{{"improvement_trajectory":float,"feedback_responsiveness":float,"preservation":float,"constraint_satisfaction":float,"constraints_satisfied":int,"constraints_total":{num_constraints},"beat_execution":float,"narrative_quality":float,"evidence":"how did quality change across versions?"}}"""

    @classmethod
    def build_critique_improvement_eval(
        cls,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        """Build evaluation prompt for critique improvement."""
        turns = result.get("turns", [])
        final_output = result.get("final_output", "") or result.get("output", "")
        revision_count = result.get("metrics", {}).get("revision_count", 0)

        # Extract versions and critiques
        versions = []
        critiques = []
        for turn in turns:
            if turn.get("turn_type") == "generation":
                rev_num = turn.get("metadata", {}).get("revision", 0)
                versions.append((rev_num, turn.get("content", "")[:400]))
            elif turn.get("turn_type") == "critique":
                round_num = turn.get("metadata", {}).get("round", 0)
                critiques.append((round_num, turn.get("content", "")[:300]))

        versions_text = "\n\n".join(
            f"VERSION {v[0]}:\n{v[1]}..." for v in versions[:4]  # Limit to avoid token overflow
        )
        critiques_text = "\n\n".join(
            f"CRITIQUE {c[0]}:\n{c[1]}..." for c in critiques[:3]
        )

        constraints = task.get("constraints", [])
        constraints_str = "\n".join(f"- {c}" for c in constraints)
        num_constraints = len(constraints)

        criteria = task.get("evaluation_criteria", [])
        criteria_str = "\n".join(f"- {c['criterion']}: {c.get('description', '')}" for c in criteria)

        return f"""CRITIQUE IMPROVEMENT TASK EVALUATION

Task constraints:
{constraints_str}

Evaluation criteria:
{criteria_str}

Critiques received:
{critiques_text}

Revision trajectory ({revision_count} revisions):
{versions_text}

Final output:
{final_output}

PROCESS EVALUATION:
- critique_responsiveness: How well were critique points addressed in revisions? (0.0-1.0)
- improvement_trajectory: Did quality demonstrably improve across revisions? (0.0-1.0)
- preservation: Were strengths preserved while fixing weaknesses? (0.0-1.0)

OUTPUT EVALUATION:
- constraint_satisfaction: How many of {num_constraints} constraints are satisfied? (0.0-1.0)
- beat_execution: Are required beats properly executed? (0.0-1.0)
- narrative_quality: Final story quality (0.0-1.0)

JSON response:
{{"critique_responsiveness":float,"improvement_trajectory":float,"preservation":float,"constraint_satisfaction":float,"constraints_satisfied":int,"constraints_total":{num_constraints},"beat_execution":float,"narrative_quality":float,"evidence":"how did quality change across critique rounds?"}}"""


class AgenticEvaluator:
    """Evaluator for multi-turn agentic tasks."""

    def __init__(
        self,
        evaluator_model: str = "anthropic/claude-haiku-4.5",
        llm_client: LLMClient | None = None,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.evaluator_model = evaluator_model
        self.model_config = load_config()["models"]

    def get_evaluator_pricing(self) -> tuple[float, float]:
        """Get pricing for the evaluator model."""
        pricing = {
            "anthropic/claude-haiku-4.5": (5.0, 25.0),
            "openai/gpt-4o-mini": (0.15, 0.60),
            "google/gemini-2.5-flash": (0.30, 2.50),
        }
        return pricing.get(self.evaluator_model, (0.25, 1.0))

    def evaluate_agentic_result(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> AgenticEvaluationResult:
        """Evaluate an agentic generation result."""
        task_id = task["task_id"]
        agentic_type = task.get("agentic_type", "unknown")
        generation_id = result.get("generation_id", "unknown")
        model = result.get("model", "unknown")
        final_output = result.get("final_output", "") or result.get("output", "")

        eval_id = generate_id()
        timestamp = get_timestamp()

        if not final_output:
            return AgenticEvaluationResult(
                evaluation_id=eval_id,
                task_id=task_id,
                generation_id=generation_id,
                agentic_type=agentic_type,
                model=model,
                evaluator_model=self.evaluator_model,
                evaluator_cost=0,
                timestamp=timestamp,
                success=False,
                process_scores={},
                output_scores={},
                final_score=0.0,
                llm_results={},
                error="Empty final output",
            )

        # Build evaluation prompt based on agentic type
        if agentic_type == "constraint_discovery":
            eval_prompt = AgenticEvalPromptBuilder.build_constraint_discovery_eval(task, result)
        elif agentic_type == "planning_execution":
            eval_prompt = AgenticEvalPromptBuilder.build_planning_execution_eval(task, result)
        elif agentic_type == "iterative_revision":
            eval_prompt = AgenticEvalPromptBuilder.build_iterative_revision_eval(task, result)
        elif agentic_type == "critique_improvement":
            eval_prompt = AgenticEvalPromptBuilder.build_critique_improvement_eval(task, result)
        else:
            return AgenticEvaluationResult(
                evaluation_id=eval_id,
                task_id=task_id,
                generation_id=generation_id,
                agentic_type=agentic_type,
                model=model,
                evaluator_model=self.evaluator_model,
                evaluator_cost=0,
                timestamp=timestamp,
                success=False,
                process_scores={},
                output_scores={},
                final_score=0.0,
                llm_results={},
                error=f"Unknown agentic type: {agentic_type}",
            )

        try:
            response = self.llm_client.call(
                model=self.evaluator_model,
                messages=[
                    {"role": "system", "content": AgenticEvalPromptBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0.1,
                max_tokens=2500,  # Increased to prevent truncation of evidence field
            )

            if not response.success:
                return AgenticEvaluationResult(
                    evaluation_id=eval_id,
                    task_id=task_id,
                    generation_id=generation_id,
                    agentic_type=agentic_type,
                    model=model,
                    evaluator_model=self.evaluator_model,
                    evaluator_cost=0,
                    timestamp=timestamp,
                    success=False,
                    process_scores={},
                    output_scores={},
                    final_score=0.0,
                    llm_results={},
                    error=response.error,
                )

            response_text = response.content
            cost = response.cost

            llm_results = extract_json_from_response(response_text)

            if llm_results is None:
                return AgenticEvaluationResult(
                    evaluation_id=eval_id,
                    task_id=task_id,
                    generation_id=generation_id,
                    agentic_type=agentic_type,
                    model=model,
                    evaluator_model=self.evaluator_model,
                    evaluator_cost=cost,
                    timestamp=timestamp,
                    success=False,
                    process_scores={},
                    output_scores={},
                    final_score=0.0,
                    llm_results={"raw_response": response_text},
                    error="Failed to parse JSON from evaluator response",
                )

            # Extract process and output scores based on agentic type
            process_scores, output_scores, final_score = self._compute_scores(
                agentic_type, llm_results, final_output, task
            )

            return AgenticEvaluationResult(
                evaluation_id=eval_id,
                task_id=task_id,
                generation_id=generation_id,
                agentic_type=agentic_type,
                model=model,
                evaluator_model=self.evaluator_model,
                evaluator_cost=cost,
                timestamp=timestamp,
                success=True,
                process_scores=process_scores,
                output_scores=output_scores,
                final_score=final_score,
                llm_results=llm_results,
            )

        except Exception as e:
            return AgenticEvaluationResult(
                evaluation_id=eval_id,
                task_id=task_id,
                generation_id=generation_id,
                agentic_type=agentic_type,
                model=model,
                evaluator_model=self.evaluator_model,
                evaluator_cost=0,
                timestamp=timestamp,
                success=False,
                process_scores={},
                output_scores={},
                final_score=0.0,
                llm_results={},
                error=str(e),
            )

    def _safe_float(self, value: Any, default: float = 0.5) -> float:
        """Safely convert a value to float."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _compute_scores(
        self,
        agentic_type: str,
        llm_results: dict[str, Any],
        final_output: str,
        task: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float], float]:
        """Compute process scores, output scores, and final score."""

        if agentic_type == "constraint_discovery":
            process_scores = {
                "discovery_efficiency": self._safe_float(llm_results.get("discovery_efficiency")),
                "question_quality": self._safe_float(llm_results.get("question_quality")),
                "question_coverage": self._safe_float(llm_results.get("question_coverage")),
            }
            output_scores = {
                "constraint_satisfaction": self._safe_float(llm_results.get("constraint_satisfaction")),
                "beat_execution": self._safe_float(llm_results.get("beat_execution")),
                "narrative_quality": self._safe_float(llm_results.get("narrative_quality")),
            }
            # Weights: process (40%), output (60%)
            process_avg = sum(process_scores.values()) / len(process_scores)
            output_avg = sum(output_scores.values()) / len(output_scores)
            final_score = 0.40 * process_avg + 0.60 * output_avg

        elif agentic_type == "planning_execution":
            process_scores = {
                "plan_completeness": self._safe_float(llm_results.get("plan_completeness")),
                "plan_specificity": self._safe_float(llm_results.get("plan_specificity")),
                "plan_adherence": self._safe_float(llm_results.get("plan_adherence")),
            }
            output_scores = {
                "constraint_satisfaction": self._safe_float(llm_results.get("constraint_satisfaction")),
                "beat_execution": self._safe_float(llm_results.get("beat_execution")),
                "narrative_quality": self._safe_float(llm_results.get("narrative_quality")),
            }
            # Weights: process (35%), output (65%)
            process_avg = sum(process_scores.values()) / len(process_scores)
            output_avg = sum(output_scores.values()) / len(output_scores)
            final_score = 0.35 * process_avg + 0.65 * output_avg

        elif agentic_type == "iterative_revision":
            process_scores = {
                "improvement_trajectory": self._safe_float(llm_results.get("improvement_trajectory")),
                "feedback_responsiveness": self._safe_float(llm_results.get("feedback_responsiveness")),
                "preservation": self._safe_float(llm_results.get("preservation")),
            }
            output_scores = {
                "constraint_satisfaction": self._safe_float(llm_results.get("constraint_satisfaction")),
                "beat_execution": self._safe_float(llm_results.get("beat_execution")),
                "narrative_quality": self._safe_float(llm_results.get("narrative_quality")),
            }
            # Weights: process (30%), output (70%) - final output matters most
            process_avg = sum(process_scores.values()) / len(process_scores)
            output_avg = sum(output_scores.values()) / len(output_scores)
            final_score = 0.30 * process_avg + 0.70 * output_avg

        elif agentic_type == "critique_improvement":
            process_scores = {
                "critique_responsiveness": self._safe_float(llm_results.get("critique_responsiveness")),
                "improvement_trajectory": self._safe_float(llm_results.get("improvement_trajectory")),
                "preservation": self._safe_float(llm_results.get("preservation")),
            }
            output_scores = {
                "constraint_satisfaction": self._safe_float(llm_results.get("constraint_satisfaction")),
                "beat_execution": self._safe_float(llm_results.get("beat_execution")),
                "narrative_quality": self._safe_float(llm_results.get("narrative_quality")),
            }
            # Weights: process (35%), output (65%) - critique responsiveness matters
            process_avg = sum(process_scores.values()) / len(process_scores)
            output_avg = sum(output_scores.values()) / len(output_scores)
            final_score = 0.35 * process_avg + 0.65 * output_avg

        else:
            process_scores = {}
            output_scores = {}
            final_score = 0.0

        return process_scores, output_scores, final_score


def create_constraint_discovery_oracle(
    task: dict[str, Any],
    oracle_model: str = "anthropic/claude-haiku-4.5",
    llm_client: LLMClient | None = None,
):
    """Create an LLM-based answer oracle for constraint discovery tasks.

    Uses semantic matching via LLM to determine if a question is asking about
    any of the hidden constraints. This is more robust than keyword matching.

    Args:
        task: Task definition with hidden_constraints
        oracle_model: LLM model to use for semantic matching (cheap model recommended)
        llm_client: Optional shared LLMClient instance (uses singleton if not provided)

    Returns a function that takes a question and returns YES/NO based on
    whether the question semantically matches any hidden constraint.
    """
    hidden_constraints = task.get("hidden_constraints", [])
    story_context = task.get("story_context", {})

    # Build constraint descriptions for the LLM
    constraint_info = []
    for i, c in enumerate(hidden_constraints):
        constraint_info.append({
            "id": c.get("id", f"constraint_{i}"),
            "description": c.get("constraint", ""),
            "answer": c.get("answer", "NO"),
        })

    # Use shared LLM client
    client = llm_client or get_llm_client()

    def oracle(question: str) -> str:
        """Use LLM to semantically match question to constraints."""
        # Use AgenticPromptBuilder for consistent prompt building
        prompt = AgenticPromptBuilder.build_oracle_prompt(
            question=question,
            story_context=story_context,
            constraint_info=constraint_info,
        )

        try:
            response = client.call(
                model=oracle_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic
                max_tokens=10,
            )

            if not response.success:
                # Fallback to keyword matching if LLM fails
                return _keyword_fallback(question, hidden_constraints)

            answer = response.content.strip().upper()

            # Parse the response
            if answer == "NONE" or answer.startswith("NONE"):
                return "NO"

            # Try to extract constraint number
            try:
                # Handle responses like "1" or "1." or "Constraint 1"
                num_str = "".join(c for c in answer if c.isdigit())
                if num_str:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(constraint_info):
                        return constraint_info[idx]["answer"]
            except (ValueError, IndexError):
                pass

            # If parsing fails but response looks affirmative, return YES
            if "YES" in answer or answer.isdigit():
                return "YES"
            return "NO"

        except Exception:
            # Fallback to keyword matching if LLM fails
            return _keyword_fallback(question, hidden_constraints)

    return oracle


def _keyword_fallback(question: str, hidden_constraints: list[dict[str, Any]]) -> str:
    """Fallback to keyword matching if LLM fails."""
    question_lower = question.lower()
    for constraint in hidden_constraints:
        patterns = constraint.get("question_patterns", [])
        for pattern in patterns:
            if pattern.lower() in question_lower:
                return constraint.get("answer", "NO")
    return "NO"


def create_feedback_generator(task: dict[str, Any]):
    """Create a feedback generator for iterative revision tasks.

    Returns a function that takes (output, task) and returns feedback list.
    Uses the feedback_rules in the task to generate targeted feedback.
    """
    feedback_rules = task.get("feedback_rules", {})
    constraints = task.get("constraints", [])
    revision_count = [0]  # Mutable to track state

    def feedback_generator(output: str, task_dict: dict[str, Any]) -> list[str]:
        revision_count[0] += 1
        round_key = f"round_{revision_count[0]}"

        if round_key not in feedback_rules:
            return []  # No more feedback rounds

        round_rules = feedback_rules[round_key]
        check_items = round_rules.get("check_items", [])

        # Generate feedback based on focus area
        # In a real implementation, this would use an LLM to generate feedback
        # For now, return the check items as feedback prompts
        return check_items

    return feedback_generator


if __name__ == "__main__":
    print("Agentic evaluator module loaded successfully.")
    print("\nSupported agentic evaluation types:")
    print("  - constraint_discovery: Evaluates question quality + output")
    print("  - planning_execution: Evaluates plan quality + adherence + output")
    print("  - iterative_revision: Evaluates improvement trajectory + output")
    print("  - critique_improvement: Evaluates critique responsiveness + improvement")
