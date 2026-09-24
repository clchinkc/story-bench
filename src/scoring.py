"""
Scoring module for the Story Theory Benchmark.

Unvalidated mechanical and judge diagnostics (not literary quality):
- Mechanical diagnostics: Word count + repetition penalty + slop detection + optional element count
- Judge diagnostics: Normalized criteria evaluation

Named diagnostics lie in [0, 1]; calibration and literary preference remain unqualified.
"""

import math
import re
from dataclasses import dataclass
from typing import Any

from measurement_contract import diagnostic_score, number


@dataclass
class ScoringWeights:
    """
    Optional explicit diagnostic weights; never calibrated literary quality.
    """

    programmatic: float = 0.50
    llm_judge: float = 0.50

    def __post_init__(self):
        number(self.programmatic, "programmatic weight")
        number(self.llm_judge, "judge weight")
        total = self.programmatic + self.llm_judge
        if not math.isclose(total, 1.0, rel_tol=1e-12):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class ProgrammaticScores:
    """Breakdown of programmatic metric scores.

    Includes:
    - word_count_score: Gaussian penalty for deviation from target range
    - repetition_score: Penalizes overused words/phrases (1.0 = no repetition)
    - slop_score: Penalizes "GPT-isms" like "tapestry", "delve" (1.0 = no slop)
    - element_count_score: Count-Correct score for structural element adherence (1.0 = perfect match)
    """

    word_count_score: float  # Word count penalty (1.0 = in target range)
    repetition_score: float  # Word/phrase repetition penalty (1.0 = no repetition)
    slop_score: float  # "GPT-isms" detection penalty (1.0 = no slop)
    element_count_score: float | None = None  # Count-Correct for structural elements (1.0 = perfect match)
    overall: float = 0.0  # Weighted combination

    def to_dict(self) -> dict[str, float]:
        result = {
            "word_count_score": round(self.word_count_score, 4),
            "repetition_score": round(self.repetition_score, 4),
            "slop_score": round(self.slop_score, 4),
            "overall": round(self.overall, 4),
        }
        if self.element_count_score is not None:
            result["element_count_score"] = round(self.element_count_score, 4)
        return result


@dataclass
class ScoreBreakdown:
    """
    Unvalidated diagnostics; the composite is absent unless weights are explicit.
    """

    programmatic_scores: ProgrammaticScores
    llm_judge_score: float | None
    final_score: float | None
    weights: ScoringWeights | None

    # Metadata
    word_count: int
    target_range: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostic_status": "unvalidated; no literary quality claim",
            "final_score": None if self.final_score is None else round(self.final_score, 4),
            "components": {
                "programmatic": {
                    "score": round(self.programmatic_scores.overall, 4),
                    "weight": None if self.weights is None else self.weights.programmatic,
                    "breakdown": self.programmatic_scores.to_dict(),
                    "word_count": {
                        "actual": self.word_count,
                        "target_range": list(self.target_range),
                    },
                },
                "llm_judge": {
                    "score": None if self.llm_judge_score is None else round(self.llm_judge_score, 4),
                    "weight": None if self.weights is None else self.weights.llm_judge,
                },
            },
        }


# ============ Element Count Scoring ============


def element_count_score(gt_count: int, pd_count: int) -> float:
    """
    Count-Correct score for structural element adherence.

    Measures how precisely the generated text meets the target count of
    structural elements (dialogue beats, required inclusions, etc.).

    Formula: score = 1 - |gt_count - pd_count| / (gt_count + pd_count)

    Args:
        gt_count: Ground truth / target element count from prompt constraints
        pd_count: Predicted / actual element count in generated text

    Returns:
        Score from 0.0 to 1.0 where 1.0 = perfect match
    """
    if type(gt_count) is not int or type(pd_count) is not int or min(gt_count, pd_count) < 0:
        raise ValueError("Element counts must be nonnegative integers")
    if gt_count == 0 and pd_count == 0:
        return 1.0  # Both zero: target and output agree there are no elements
    denominator = gt_count + pd_count
    if denominator == 0:
        return 1.0
    return 1.0 - abs(gt_count - pd_count) / denominator


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
    score = math.exp(-(deviation**2) / (2 * sigma**2))

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
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return words


def get_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)
    # Filter empty and strip
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# Common words to exclude from repetition counting
COMMON_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "our",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
    "now",
    "here",
    "there",
    "then",
    "once",
    "again",
    "into",
    "out",
    "up",
    "down",
    "about",
    "after",
    "before",
    "over",
    "under",
    "through",
}

# Historical English word-list diagnostic; not evidence of quality or AI authorship
# Based on EQ-Bench research and common "GPT-isms"
SLOP_WORDS = {
    # High-frequency slop (2 points each)
    "tapestry": 2,
    "testament": 2,
    "delve": 2,
    "realm": 2,
    "crucial": 2,
    "multifaceted": 2,
    "intricacies": 2,
    "nuanced": 2,
    "embark": 2,
    "captivate": 2,
    "captivating": 2,
    "resonate": 2,
    "resonated": 2,
    "resonating": 2,
    "landscape": 2,
    "navigating": 2,
    "navigate": 2,
    "elevate": 2,
    "elevating": 2,
    "evolving": 2,
    "pivotal": 2,
    "unravel": 2,
    "unraveling": 2,
    "unveil": 2,
    "unveiling": 2,
    "profound": 2,
    "profoundly": 2,
    "intricate": 2,
    "intricately": 2,
    "overarching": 2,
    "underscore": 2,
    "underscores": 2,
    "underscoring": 2,
    "superb": 2,
    "meticulously": 2,
    "meticulous": 2,
    "seamlessly": 2,
    "seamless": 2,
    "intrinsic": 2,
    "intrinsically": 2,
    "noteworthy": 2,
    # Medium-frequency slop (1 point each)
    "foray": 1,
    "myriad": 1,
    "plethora": 1,
    "albeit": 1,
    "whilst": 1,
    "amidst": 1,
    "burgeoning": 1,
    "endeavor": 1,
    "endeavors": 1,
    "paramount": 1,
    "quintessential": 1,
    "tantalizing": 1,
    "tantalize": 1,
    "whimsical": 1,
    "whimsy": 1,
    "heartwarming": 1,
    "heartfelt": 1,
    "poignant": 1,
    "evocative": 1,
    "visceral": 1,
    "palpable": 1,
    "cacophony": 1,
    "symphony": 1,
    "kaleidoscope": 1,
    "mosaic": 1,
    "beacon": 1,
    "harbinger": 1,
    "pinnacle": 1,
    "zenith": 1,
    "nadir": 1,
    "cusp": 1,
    "brink": 1,
    "precipice": 1,
    "labyrinth": 1,
    "labyrinthine": 1,
    "enigmatic": 1,
    "enigma": 1,
    "conundrum": 1,
    "dichotomy": 1,
    "juxtaposition": 1,
    "paradox": 1,
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
    # Words with count above max(3, 2x expected frequency) are penalized
    expected_freq = total_content / unique_content if unique_content > 0 else 1
    threshold = max(3, expected_freq * 2)  # At least 3 uses before penalty

    penalty = 0
    for word, count in word_counts.items():
        if count > threshold:
            # Penalty increases with excess repetition
            excess = count - threshold
            penalty += excess * 0.02  # 2% per excess repetition

    # Also check for repeated bigrams (2-word phrases)
    bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
    bigram_counts = Counter(bigrams)
    for phrase, count in bigram_counts.items():
        # Penalize only phrases repeated more than 3 times
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
    word_count_method: str = "gaussian",
    gt_element_count: int | None = None,
    pd_element_count: int | None = None,
) -> ProgrammaticScores:
    """
    Calculate all programmatic metrics for a text.

    Three core components:
    - Word count score (40%): Penalty for deviation from target (method configurable)
    - Repetition score (35%): Penalizes excessive word/phrase repetition
    - Slop score (25%): Penalizes overused "GPT-isms"

    Optional fourth component for tasks with strict structural counts:
    - Element count score (Count-Correct): penalizes deviation from required element counts
      Fixed relative weights 27/27/27/10, normalized by their sum (0.91).

    Returns:
        ProgrammaticScores with individual and overall scores
    """
    if (gt_element_count is None) != (pd_element_count is None):
        raise ValueError("Both element counts are required")
    wc_score = word_count_score(
        word_count, target_min, target_max, method=word_count_method
    )
    rep_score = repetition_score(text)
    slop = slop_score(text)

    # Element count score is optional — included only when both counts are provided
    elem_score: float | None = None
    if gt_element_count is not None and pd_element_count is not None:
        elem_score = element_count_score(gt_element_count, pd_element_count)
        # With element count: wc (27%) + rep (27%) + slop (27%) + elem (10%)
        overall = (0.27 * wc_score + 0.27 * rep_score + 0.27 * slop + 0.10 * elem_score) / 0.91
    else:
        # Standard: word_count (40%) + repetition (35%) + slop (25%)
        overall = 0.40 * wc_score + 0.35 * rep_score + 0.25 * slop

    return ProgrammaticScores(
        word_count_score=wc_score,
        repetition_score=rep_score,
        slop_score=slop,
        element_count_score=elem_score,
        overall=overall,
    )


# ============ LLM Judge Score Normalization ============


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Reject invalid values; never substitute credit."""
    return number(value, "criterion")


def normalize_llm_results_to_score(results: dict[str, Any], task_type: str, subtype: str | None = None) -> float | None:
    """Strict selected-schema diagnostic, not a literary quality outcome."""
    return diagnostic_score(results, task_type, subtype)


# ============ Final Score Calculation ============


def calculate_final_score(
    text: str,
    word_count: int,
    target_word_range: tuple[int, int],
    llm_results: dict[str, Any],
    task_type: str,
    weights: ScoringWeights | None = None,
    word_count_method: str = "gaussian",
    gt_element_count: int | None = None,
    pd_element_count: int | None = None,
    subtype: str | None = None,
) -> ScoreBreakdown:
    """
    Calculate the final composite score for a generation.

    Default: separate unvalidated diagnostics and no composite.
    Explicit weights may request an unvalidated diagnostic composite.

    Args:
        text: The generated text
        word_count: Pre-computed word count
        target_word_range: (min, max) target word count
        llm_results: Results dict from LLM evaluation
        task_type: Type of task
        weights: Optional explicit diagnostic weights (default no composite)
        word_count_method: Method for word count scoring ("gaussian", "tanh", or "sigmoid")
        gt_element_count: Ground truth / target element count (optional)
        pd_element_count: Predicted / actual element count (optional)
        subtype: Task subtype selecting the judge score schema (optional)

    Returns:
        ScoreBreakdown with all component scores (final_score is None unless weights are explicit)
    """

    # 1. Programmatic scores (includes word count, optionally element count)
    prog_scores = calculate_programmatic_scores(
        text,
        word_count,
        target_word_range[0],
        target_word_range[1],
        word_count_method=word_count_method,
        gt_element_count=gt_element_count,
        pd_element_count=pd_element_count,
    )

    # 2. LLM judge score
    llm_score = normalize_llm_results_to_score(llm_results, task_type, subtype)

    # Optional diagnostic composite; never a headline quality/value rank
    final = None if weights is None or llm_score is None else weights.programmatic * prog_scores.overall + weights.llm_judge * llm_score

    return ScoreBreakdown(
        programmatic_scores=prog_scores,
        llm_judge_score=llm_score,
        final_score=final,
        weights=weights,
        word_count=word_count,
        target_range=target_word_range,
    )


