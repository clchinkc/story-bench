"""The frozen assessor: one model call, one score, pinned so it cannot drift under the run.

This is the reward's signal (D001 — lint is only a veto). Because it carries the whole signal, it is treated
as an instrument rather than a prompt: the model id, the prompt version and the sampling parameters are
constants here, and every assessment records which version produced it.

Two properties are load-bearing and were both violated by the naive version:

    frozen       temperature 0, fixed seed, fixed prompt text, fixed model file. A judge that samples is a
                 different judge on every rollout, and GRPO would be chasing its own noise.

The 2B was tried first and rejected on measurement, not taste: it scored vivid prose and hollow prose both at
95, i.e. it could not discriminate, and a constant reward trains nothing. The 4B base separates the same
three controls 72 / 45 / 25. It is the policy's own base weights, so self-preference is a live risk — which is
exactly why the verdict is judged by a different, stronger model, and why the reward judge is the frozen base
rather than any adapter.
    blind        the judge is never told whether the text came from the base model, the trained model or a
                 published book. It sees the opening and the continuation, nothing else.

The parse is strict on purpose: a judge that cannot answer in the required JSON is a broken instrument, and
failing loudly beats silently scoring it 0 — which would look like a bad model rather than a bad judge.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

# Two backends: the local server (free, but measured invalid — D002) and OpenRouter (paid, validated). The
# backend is an environment choice so a run's provenance is explicit rather than implied by a constant.
BACKEND = os.environ.get("PROSE_JUDGE_BACKEND", "local")
OPENROUTER_MODEL = os.environ.get("PROSE_JUDGE_MODEL", "anthropic/claude-sonnet-4.5")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

JUDGE_URL = os.environ.get("PROSE_JUDGE_URL", "http://127.0.0.1:8768")
JUDGE_MODEL_ID = "Qwen3.5-4B-Q4_K_M.gguf"
JUDGE_PROMPT_VERSION = "craft-v1"
JUDGE_TEMPERATURE = 0.0
JUDGE_SEED = 0
JUDGE_MAX_TOKENS = 96

# Bands, not points: the protocol is band-first, and a 2B judge is more reliable at "which band" than at
# "which number inside the band".
BANDS = (
    ("publishable", 90, 100, "could stand in a published book unchanged"),
    ("competent", 70, 89, "clear and readable, but flat or generic in places"),
    ("serviceable", 50, 69, "conveys events without voice, tension or interior life"),
    ("hollow", 30, 49, "grammatical but empty: no stakes, no specificity, nothing observed"),
    ("broken", 0, 29, "incoherent, contradicts the opening, or is not prose"),
)

PROMPT = """You are scoring one continuation of a fiction scene against a craft rubric.

The scene opening (given to the writer):
---
{opening}
---

The continuation to score:
---
{continuation}
---

Score the CONTINUATION only, on one axis: is this prose a reader would keep reading?

Publishable (90-100): specific and observed; the sentences carry a voice; there is tension or interior
life in the detail; nothing is explained that could be shown.
Competent (70-89): clear and readable, correctly in the scene, but flat or generic in places.
Serviceable (50-69): conveys events without voice, tension or interior life.
Hollow (30-49): grammatical but empty — no stakes, no specificity, nothing actually observed.
Broken (0-29): incoherent, contradicts the opening, or is not prose.

Anchor on the evidence, not on a habitual middle. If the continuation is a summary of the scene rather
than prose inside it, it is not competent.

Reply with JSON only, no prose around it:
{{"score": <integer 0-100>, "band": "<publishable|competent|serviceable|hollow|broken>", "note": "<at most 12 words>"}}
"""


class JudgeError(RuntimeError):
    """The judge did not answer in the required shape — an instrument failure, not a low score."""


@dataclass(frozen=True)
class Assessment:
    score: int
    band: str
    note: str
    version: str = JUDGE_PROMPT_VERSION
    model: str = JUDGE_MODEL_ID

    def normalised(self) -> float:
        return self.score / 100.0


def active_judge() -> tuple[str, str]:
    """(backend, model) actually in force — recorded with every score so a run's provenance is auditable."""
    return (BACKEND, OPENROUTER_MODEL) if BACKEND == "openrouter" else ("local", JUDGE_MODEL_ID)


def band_of(score: int) -> str:
    for name, low, high, _ in BANDS:
        if low <= score <= high:
            return name
    raise JudgeError(f"{score} is outside every band")


def build_prompt(opening: str, continuation: str) -> str:
    return PROMPT.format(opening=opening.strip(), continuation=continuation.strip())


def _post(body: dict) -> dict:
    request = urllib.request.Request(
        f"{JUDGE_URL.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JudgeError(f"judge unreachable at {JUDGE_URL}: {exc}") from exc


def parse(text: str) -> Assessment:
    if not isinstance(text, str) or not text.strip():
        raise JudgeError(f"judge returned no content: {text!r}")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise JudgeError(f"judge reply carries no JSON object: {text[:160]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge JSON did not parse: {match.group(0)[:160]!r}") from exc
    score = payload.get("score")
    if not isinstance(score, int) or not 0 <= score <= 100:
        raise JudgeError(f"judge score is not an int in 0..100: {score!r}")
    declared = payload.get("band")
    expected = band_of(score)
    if declared is not None and declared != expected:
        raise JudgeError(f"judge band {declared!r} disagrees with its score {score} ({expected})")
    return Assessment(score=score, band=expected, note=str(payload.get("note", ""))[:120])


def _post_openrouter(prompt: str) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise JudgeError("PROSE_JUDGE_BACKEND=openrouter but OPENROUTER_API_KEY is unset")
    body = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": JUDGE_TEMPERATURE, "max_tokens": 200}
    request = urllib.request.Request(
        OPENROUTER_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JudgeError(f"openrouter judge unreachable: {exc}") from exc


def assess(opening: str, continuation: str) -> Assessment:
    """One frozen call. Raises rather than inventing a score."""
    if not continuation.strip():
        raise JudgeError("empty continuation cannot be assessed")
    body = {
        "model": JUDGE_MODEL_ID,
        "messages": [{"role": "user", "content": build_prompt(opening, continuation)}],
        "temperature": JUDGE_TEMPERATURE,
        "top_p": 1.0,
        "seed": JUDGE_SEED,
        "n_predict": JUDGE_MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if BACKEND == "openrouter":
        reply = _post_openrouter(build_prompt(opening, continuation))
    else:
        reply = _post(body)
    try:
        text = reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise JudgeError(f"judge reply has no message content: {str(reply)[:160]}") from exc
    scored = parse(text)
    backend, model = active_judge()
    return Assessment(score=scored.score, band=scored.band, note=scored.note, version=JUDGE_PROMPT_VERSION,
                      model=f"{backend}:{model}")
