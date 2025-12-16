"""
Scoring module for the Story Theory Benchmark.

Implements the two-component scoring system:
- Programmatic Score (50%): Word count + repetition penalty + slop detection
- LLM-as-Judge Score (50%): Normalized criteria evaluation

All scores are 0.0 - 1.0 where 1.0 is best.
"""

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ScoringWeights:
    """
    Configurable weights for score components.
    Programmatic (50%): word count + repetition + slop
    LLM Judge (50%): task-specific criteria
    """
    programmatic: float = 0.50
    llm_judge: float = 0.50

    def __post_init__(self):
        total = self.programmatic + self.llm_judge
        if not math.isclose(total, 1.0, rel_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class ProgrammaticScores:
    """Breakdown of programmatic metric scores.

    Includes:
    - word_count_score: Gaussian penalty for deviation from target range
    - repetition_score: Penalizes overused words/phrases (1.0 = no repetition)
    - slop_score: Penalizes "GPT-isms" like "tapestry", "delve" (1.0 = no slop)
    """
    word_count_score: float      # Word count penalty (1.0 = in target range)
    repetition_score: float      # Word/phrase repetition penalty (1.0 = no repetition)
    slop_score: float            # "GPT-isms" detection penalty (1.0 = no slop)
    overall: float               # Weighted combination

    def to_dict(self) -> dict[str, float]:
        return {
            "word_count_score": round(self.word_count_score, 4),
            "repetition_score": round(self.repetition_score, 4),
            "slop_score": round(self.slop_score, 4),
            "overall": round(self.overall, 4),
        }


@dataclass
class ScoreBreakdown:
    """
    Complete score breakdown for a generation.
    Two components: Programmatic (50%) + LLM Judge (50%)
    """
    programmatic_scores: ProgrammaticScores
    llm_judge_score: float
    final_score: float
    weights: ScoringWeights

    # Metadata
    word_count: int
    target_range: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": round(self.final_score, 4),
            "components": {
                "programmatic": {
                    "score": round(self.programmatic_scores.overall, 4),
                    "weight": self.weights.programmatic,
                    "breakdown": self.programmatic_scores.to_dict(),
                    "word_count": {
                        "actual": self.word_count,
                        "target_range": list(self.target_range),
                    },
                },
                "llm_judge": {
                    "score": round(self.llm_judge_score, 4),
                    "weight": self.weights.llm_judge,
                },
            },
        }


# ============ Word Count Scoring ============

def word_count_score_gaussian(
    word_count: int,
    target_min: int,
    target_max: int,
    sigma_factor: float = 0.8,
) -> float:
    """
    Calculate word count score using a Gaussian penalty function.

    Returns 1.0 if word count is within target range.
    Drops off smoothly using Gaussian curve for values outside range.

    Args:
        word_count: Actual word count
        target_min: Minimum target word count
        target_max: Maximum target word count
        sigma_factor: Controls how quickly score drops (fraction of range width)

    Returns:
        Score from 0.0 to 1.0
    """
    if target_min <= word_count <= target_max:
        return 1.0

    # Calculate sigma based on range width
    range_width = target_max - target_min
    sigma = max(range_width * sigma_factor, 50)  # Minimum sigma of 50 words

    # Calculate deviation from nearest boundary
    if word_count < target_min:
        deviation = target_min - word_count
    else:
        deviation = word_count - target_max

    # Gaussian penalty: e^(-(deviation^2)/(2*sigma^2))
    score = math.exp(-(deviation ** 2) / (2 * sigma ** 2))

    return max(0.0, min(1.0, score))


def word_count_score_tanh(
    word_count: int,
    target_min: int,
    target_max: int,
    steepness: float = 0.02,
) -> float:
    """
    Calculate word count score using a tanh-based penalty function.

    Returns 1.0 if word count is within target range.
    Smoothly transitions to 0 for values far outside range.

    Args:
        word_count: Actual word count
        target_min: Minimum target word count
        target_max: Maximum target word count
        steepness: Controls transition sharpness (higher = sharper)

    Returns:
        Score from 0.0 to 1.0
    """
    if target_min <= word_count <= target_max:
        return 1.0

    # Calculate deviation from nearest boundary
    if word_count < target_min:
        deviation = target_min - word_count
    else:
        deviation = word_count - target_max

    # tanh-based scoring: 1 - tanh(steepness * deviation)
    score = 1.0 - math.tanh(steepness * deviation)

    return max(0.0, min(1.0, score))


def word_count_score_sigmoid(
    word_count: int,
    target_min: int,
    target_max: int,
    k: float = 0.03,
) -> float:
    """
    Calculate word count score using a sigmoid-based penalty function.

    Returns 1.0 if word count is within target range.
    Uses sigmoid to smoothly penalize deviations.

    Args:
        word_count: Actual word count
        target_min: Minimum target word count
        target_max: Maximum target word count
        k: Steepness parameter (higher = sharper transition)

    Returns:
        Score from 0.0 to 1.0
    """
    if target_min <= word_count <= target_max:
        return 1.0

    # Calculate deviation from nearest boundary
    if word_count < target_min:
        deviation = target_min - word_count
    else:
        deviation = word_count - target_max

    # Sigmoid-based penalty: 2 / (1 + e^(k * deviation))
    # This gives 1.0 at deviation=0 and approaches 0 as deviation increases
    score = 2.0 / (1.0 + math.exp(k * deviation))

    return max(0.0, min(1.0, score))


# Default word count scorer
def word_count_score(
    word_count: int,
    target_min: int,
    target_max: int,
    method: str = "gaussian",
) -> float:
    """
    Calculate word count score using specified method.

    Args:
        word_count: Actual word count
        target_min: Minimum target word count
        target_max: Maximum target word count
        method: "gaussian", "tanh", or "sigmoid"

    Returns:
        Score from 0.0 to 1.0
    """
    if method == "gaussian":
        return word_count_score_gaussian(word_count, target_min, target_max)
    elif method == "tanh":
        return word_count_score_tanh(word_count, target_min, target_max)
    elif method == "sigmoid":
        return word_count_score_sigmoid(word_count, target_min, target_max)
    else:
        raise ValueError(f"Unknown scoring method: {method}")


# ============ Programmatic Metrics ============

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def tokenize_words(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    # Remove punctuation and split
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def get_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Filter empty and strip
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# Common words to exclude from repetition counting
COMMON_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "same", "so", "than", "too", "very", "just", "also",
    "now", "here", "there", "then", "once", "again", "into", "out",
    "up", "down", "about", "after", "before", "over", "under", "through",
}

# "Slop" words - overused LLM phrases that indicate poor creative writing
# Based on EQ-Bench research and common "GPT-isms"
SLOP_WORDS = {
    # High-frequency slop (2 points each)
    "tapestry": 2, "testament": 2, "delve": 2, "realm": 2, "crucial": 2,
    "multifaceted": 2, "intricacies": 2, "nuanced": 2, "embark": 2,
    "captivate": 2, "captivating": 2, "resonate": 2, "resonated": 2,
    "resonating": 2, "landscape": 2, "navigating": 2, "navigate": 2,
    "elevate": 2, "elevating": 2, "evolving": 2, "pivotal": 2,
    "unravel": 2, "unraveling": 2, "unveil": 2, "unveiling": 2,
    "profound": 2, "profoundly": 2, "intricate": 2, "intricately": 2,
    "overarching": 2, "underscore": 2, "underscores": 2, "underscoring": 2,
    "superb": 2, "meticulously": 2, "meticulous": 2, "seamlessly": 2,
    "seamless": 2, "intrinsic": 2, "intrinsically": 2, "noteworthy": 2,

    # Medium-frequency slop (1 point each)
    "foray": 1, "myriad": 1, "plethora": 1, "albeit": 1, "whilst": 1,
    "amidst": 1, "burgeoning": 1, "endeavor": 1, "endeavors": 1,
    "paramount": 1, "quintessential": 1, "tantalizing": 1, "tantalize": 1,
    "whimsical": 1, "whimsy": 1, "heartwarming": 1, "heartfelt": 1,
    "poignant": 1, "evocative": 1, "visceral": 1, "palpable": 1,
    "cacophony": 1, "symphony": 1, "kaleidoscope": 1, "mosaic": 1,
    "beacon": 1, "harbinger": 1, "testament": 1, "pinnacle": 1,
    "zenith": 1, "nadir": 1, "cusp": 1, "brink": 1, "precipice": 1,
    "labyrinth": 1, "labyrinthine": 1, "enigmatic": 1, "enigma": 1,
    "conundrum": 1, "dichotomy": 1, "juxtaposition": 1, "paradox": 1,
}


def repetition_score(text: str) -> float:
    """
    Calculate repetition score based on word frequency analysis.

    Measures how often non-common words are repeated excessively.
    Good creative writing varies word choice; repetitive text scores lower.

    Returns:
        Score from 0.0 to 1.0 where 1.0 = no excessive repetition
    """
    words = tokenize_words(text)
    if len(words) < 20:
        return 0.7  # Not enough words to evaluate

    # Filter out common words
    content_words = [w for w in words if w not in COMMON_WORDS and len(w) > 2]
    if len(content_words) < 10:
        return 0.7

    # Count word frequencies
    from collections import Counter
    word_counts = Counter(content_words)

    # Calculate expected frequency (uniform distribution)
    total_content = len(content_words)
    unique_content = len(word_counts)

    # Calculate excess repetition penalty
    # Words appearing more than 3x their expected frequency are penalized
    expected_freq = total_content / unique_content if unique_content > 0 else 1
    threshold = max(3, expected_freq * 2)  # At least 3 uses before penalty

    penalty = 0
    for word, count in word_counts.items():
        if count > threshold:
            # Penalty increases with excess repetition
            excess = count - threshold
            penalty += excess * 0.02  # 2% per excess repetition

    # Also check for repeated phrases (2-3 word combinations)
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    for phrase, count in bigram_counts.items():
        # Exclude common phrases
        if count > 3:
            penalty += (count - 3) * 0.03  # 3% per excess phrase repetition

    # Convert penalty to score (capped at 0.5 max penalty)
    score = max(0.0, 1.0 - min(0.5, penalty))
    return score


def slop_score(text: str) -> float:
    """
    Calculate slop score based on presence of overused LLM phrases.

    "Slop" refers to words/phrases that LLMs overuse, making text sound
    artificial. Based on EQ-Bench research into "GPT-isms".

    Returns:
        Score from 0.0 to 1.0 where 1.0 = no slop detected
    """
    words = tokenize_words(text)
    if len(words) < 20:
        return 0.8  # Not enough words to evaluate

    total_words = len(words)

    # Count slop words and their weighted penalty
    slop_penalty = 0
    slop_count = 0

    for word in words:
        if word in SLOP_WORDS:
            slop_count += 1
            slop_penalty += SLOP_WORDS[word]

    # Also check for slop phrases
    text_lower = text.lower()
    slop_phrases = [
        ("a testament to", 2),
        ("delve into", 2),
        ("tapestry of", 2),
        ("rich tapestry", 2),
        ("it's important to note", 2),
        ("it's worth noting", 2),
        ("stands as a testament", 2),
        ("navigating the", 1),
        ("embarked on a", 1),
        ("in the realm of", 1),
        ("pivotal moment", 1),
        ("profound impact", 1),
        ("at the heart of", 1),
        ("a beacon of", 1),
    ]

    for phrase, weight in slop_phrases:
        count = text_lower.count(phrase)
        if count > 0:
            slop_count += count
            slop_penalty += count * weight

    # Normalize by text length (penalty per 100 words)
    normalized_penalty = (slop_penalty / total_words) * 100

    # Convert to score: 0 slop = 1.0, heavy slop = lower score
    # Each point of normalized penalty reduces score by 5%
    score = max(0.0, 1.0 - normalized_penalty * 0.05)
    return score


def calculate_programmatic_scores(
    text: str,
    word_count: int,
    target_min: int,
    target_max: int,
) -> ProgrammaticScores:
    """
    Calculate all programmatic metrics for a text.

    Three components:
    - Word count score (40%): Gaussian penalty for deviation from target
    - Repetition score (35%): Penalizes excessive word/phrase repetition
    - Slop score (25%): Penalizes overused "GPT-isms"

    Returns:
        ProgrammaticScores with individual and overall scores
    """
    wc_score = word_count_score(word_count, target_min, target_max)
    rep_score = repetition_score(text)
    slop = slop_score(text)

    # Weighted combination: word_count (40%) + repetition (35%) + slop (25%)
    overall = 0.40 * wc_score + 0.35 * rep_score + 0.25 * slop

    return ProgrammaticScores(
        word_count_score=wc_score,
        repetition_score=rep_score,
        slop_score=slop,
        overall=overall,
    )


# ============ LLM Judge Score Normalization ============

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default if not possible."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_llm_results_to_score(
    results: dict[str, Any],
    task_type: str,
) -> float:
    """
    Convert LLM evaluation results to a normalized score (0.0-1.0).

    Supports both old binary format and new partial credit format.
    Different task types have different criteria structures and weights.

    Args:
        results: Raw LLM evaluation results dict
        task_type: Type of task for scoring rules

    Returns:
        Normalized score from 0.0 to 1.0
    """
    if not results:
        return 0.0

    if task_type == "beat_interpolation":
        # Weighted scoring: elements (25%), beat_execution (25%), must_not (15%), character (10%), bridge (15%), continuity (10%)
        elements = _safe_float(results.get("beat_elements_score"), 0.5)
        beat_exec = _safe_float(results.get("beat_execution_score"), 0.5)
        must_not = _safe_float(results.get("must_not_score"), 1.0)  # Default 1.0 if no violations specified
        character = _safe_float(results.get("character_score"), 0.5)
        bridge = _safe_float(results.get("bridge_score"), 0.5)
        continuity = _safe_float(results.get("continuity_score"), 0.5)
        return 0.25 * elements + 0.25 * beat_exec + 0.15 * must_not + 0.10 * character + 0.15 * bridge + 0.10 * continuity

    elif task_type == "beat_revision":
        # Check if this is a "no flaw" task (different scoring fields)
        if "correct_diagnosis_score" in results:
            # NO-FLAW task: Tests if model recognizes segment needs no revision
            # Weighted: correct_diagnosis (40%), false_positive_avoided (30%),
            #           beat_understanding (15%), reasoning_quality (15%)
            correct_diagnosis = _safe_float(results.get("correct_diagnosis_score"), 0.0)
            false_positive = _safe_float(results.get("false_positive_avoided_score"), 0.0)
            beat_understanding = _safe_float(results.get("beat_understanding_score"), 0.5)
            reasoning = _safe_float(results.get("reasoning_quality_score"), 0.5)
            return (0.40 * correct_diagnosis + 0.30 * false_positive +
                    0.15 * beat_understanding + 0.15 * reasoning)

        # Standard FLAWED task: Model must identify flaw themselves (not told what's wrong)
        # CONSTRAINED REVISION: Also evaluates minimal modification
        # Weighted: diagnosis (20%), flaw fix (20%), beat satisfaction (20%),
        #           preservation (10%), required_preserved (10%), minimal_change (10%), quality (10%)
        diagnosis = _safe_float(results.get("diagnosis_score"), 0.5)
        flaw = _safe_float(results.get("flaw_correction_score"), 0.5)
        beat = _safe_float(results.get("beat_satisfaction_score"), 0.5)
        preserve = _safe_float(results.get("preservation_score"), 0.5)
        required_preserved = _safe_float(results.get("required_preserved_score"), 1.0)  # Default 1.0 if no requirements
        minimal_change = _safe_float(results.get("minimal_change_score"), 0.5)
        quality = _safe_float(results.get("quality_score"), 0.5)
        return (0.20 * diagnosis + 0.20 * flaw + 0.20 * beat +
                0.10 * preserve + 0.10 * required_preserved + 0.10 * minimal_change + 0.10 * quality)

    elif task_type == "constrained_continuation":
        # Weighted: beats (20%), must_include (30%), must_not (25%), tone (15%), ending (10%)
        beats = _safe_float(results.get("beats_score"), 0.5)
        must_include = _safe_float(results.get("must_include_score"), 0.5)
        must_not = _safe_float(results.get("must_not_score"), 0.5)
        tone = _safe_float(results.get("tone_score"), 0.5)
        ending = _safe_float(results.get("ending_score"), 0.5)
        return 0.20 * beats + 0.30 * must_include + 0.25 * must_not + 0.15 * tone + 0.10 * ending

    elif task_type == "theory_conversion":
        # Weighted: beats (35%), preservation (30%), structural (20%), tone (15%)
        beats = _safe_float(results.get("beats_score"), 0.5)
        preserve = _safe_float(results.get("preservation_score"), 0.5)
        structural = _safe_float(results.get("structural_accuracy_score"), 0.5)
        tone = _safe_float(results.get("tone_score"), 0.5)
        return 0.35 * beats + 0.30 * preserve + 0.20 * structural + 0.15 * tone

    elif task_type == "multi_beat_synthesis":
        # Weighted: beat reqs (40%), cross-beat (35%), context (15%), coherence (10%)
        beat_reqs = _safe_float(results.get("beat_requirements_score"), 0.5)
        cross_beat = _safe_float(results.get("cross_beat_score"), 0.5)
        context = _safe_float(results.get("context_score"), 0.5)
        coherence = _safe_float(results.get("coherence_score"), 0.5)
        return 0.40 * beat_reqs + 0.35 * cross_beat + 0.15 * context + 0.10 * coherence

    else:
        raise ValueError(f"Unknown task type: {task_type}")


# ============ Final Score Calculation ============

def calculate_final_score(
    text: str,
    word_count: int,
    target_word_range: tuple[int, int],
    llm_results: dict[str, Any],
    task_type: str,
    weights: ScoringWeights | None = None,
) -> ScoreBreakdown:
    """
    Calculate the final composite score for a generation.

    Two components (50/50 split):
    1. Programmatic (50%): Word count + repetition + slop detection
    2. LLM Judge (50%): Normalized score from LLM criteria evaluation

    Args:
        text: The generated text
        word_count: Pre-computed word count
        target_word_range: (min, max) target word count
        llm_results: Results dict from LLM evaluation
        task_type: Type of task
        weights: Scoring weights (default 50/50)

    Returns:
        ScoreBreakdown with all component scores and final score
    """
    if weights is None:
        weights = ScoringWeights()

    # 1. Programmatic scores (includes word count)
    prog_scores = calculate_programmatic_scores(
        text,
        word_count,
        target_word_range[0],
        target_word_range[1],
    )

    # 2. LLM judge score
    llm_score = normalize_llm_results_to_score(llm_results, task_type)

    # Weighted final score (50/50)
    final = (
        weights.programmatic * prog_scores.overall +
        weights.llm_judge * llm_score
    )

    return ScoreBreakdown(
        programmatic_scores=prog_scores,
        llm_judge_score=llm_score,
        final_score=final,
        weights=weights,
        word_count=word_count,
        target_range=target_word_range,
    )


# ============ Testing ============

if __name__ == "__main__":
    # Test word count scoring
    print("=== Word Count Scoring Tests ===")
    test_cases = [
        (500, 400, 600),   # In range
        (350, 400, 600),   # Slightly under
        (200, 400, 600),   # Way under
        (650, 400, 600),   # Slightly over
        (900, 400, 600),   # Way over
    ]

    for wc, min_wc, max_wc in test_cases:
        g = word_count_score_gaussian(wc, min_wc, max_wc)
        t = word_count_score_tanh(wc, min_wc, max_wc)
        s = word_count_score_sigmoid(wc, min_wc, max_wc)
        print(f"WC={wc} (target {min_wc}-{max_wc}): gaussian={g:.3f}, tanh={t:.3f}, sigmoid={s:.3f}")

    print("\n=== Programmatic Scoring Tests ===")
    # Good text - no repetition, no slop
    good_text = """
    The hero stood at the edge of the cliff, gazing down at the vast ocean below.
    Waves crashed against the rocks. The wind howled through the canyon.
    She knew what she had to do. There was no turning back now.
    With a deep breath, she stepped forward into the unknown.
    """

    prog = calculate_programmatic_scores(good_text, len(good_text.split()), 50, 100)
    print(f"Good text - WC: {prog.word_count_score:.3f}, Rep: {prog.repetition_score:.3f}, Slop: {prog.slop_score:.3f}, Overall: {prog.overall:.3f}")

    # Sloppy text - contains GPT-isms
    sloppy_text = """
    The tapestry of her journey was a testament to her resilience. She delved into
    the intricacies of the realm, navigating the multifaceted landscape with
    profound determination. Each pivotal moment unveiled new insights, resonating
    deeply with her evolving understanding. The seamless integration of her
    experiences was truly noteworthy, a beacon of hope in an enigmatic world.
    """

    prog = calculate_programmatic_scores(sloppy_text, len(sloppy_text.split()), 50, 100)
    print(f"Sloppy text - WC: {prog.word_count_score:.3f}, Rep: {prog.repetition_score:.3f}, Slop: {prog.slop_score:.3f}, Overall: {prog.overall:.3f}")

    # Repetitive text
    repetitive_text = """
    The hero walked. The hero looked. The hero thought. The hero decided.
    The hero moved forward. The hero saw the enemy. The hero raised her sword.
    The hero attacked. The hero dodged. The hero struck. The hero won.
    """

    prog = calculate_programmatic_scores(repetitive_text, len(repetitive_text.split()), 50, 100)
    print(f"Repetitive text - WC: {prog.word_count_score:.3f}, Rep: {prog.repetition_score:.3f}, Slop: {prog.slop_score:.3f}, Overall: {prog.overall:.3f}")

    print("\n=== Full Score Test ===")
    # New partial credit format
    mock_llm_results = {
        "beat_elements_score": 0.8,
        "elements_found": 4,
        "elements_total": 5,
        "character_score": 0.9,
        "bridge_score": 0.85,
        "continuity_score": 0.95,
    }

    breakdown = calculate_final_score(
        text=good_text,
        word_count=len(good_text.split()),
        target_word_range=(50, 100),
        llm_results=mock_llm_results,
        task_type="beat_interpolation",
    )

    print(f"Final score: {breakdown.final_score:.3f}")
    print(f"  1. Programmatic ({breakdown.weights.programmatic}): {breakdown.programmatic_scores.overall:.3f}")
    print(f"     - Word Count: {breakdown.programmatic_scores.word_count_score:.3f}")
    print(f"     - Repetition: {breakdown.programmatic_scores.repetition_score:.3f}")
    print(f"     - Slop: {breakdown.programmatic_scores.slop_score:.3f}")
    print(f"  2. LLM judge ({breakdown.weights.llm_judge}): {breakdown.llm_judge_score:.3f}")
