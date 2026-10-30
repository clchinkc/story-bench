# Diagnostic instrument contract

**Status:** proposed 2026-10-30, not yet enforced by code.
**Scope:** every *diagnostic* measurement in this program. A diagnostic is anything that describes a
text (slop, repetition, style range, homogenization, punctuation habit) and is **never** an input to
the terminal reward.

**Why this file exists.** `docs/experiment-contract.md` § 6 already states that slop/repetition/style
range are *"diagnostics, not AI detectors or a global quality axis"*, and § 7 fixes the terminal reward
as binary resolved success under the authority and critical-continuity gates. This file does not
change either. It specifies what a diagnostic must satisfy **to be reportable at all** — because a
diagnostic that is quietly wrong is worse than no diagnostic: it looks like evidence.

Provenance for the requirements: `📚-KNOWLEDGE/01-AI-Technology/Sources/ai-slop-tells-evidence-map-2026-10-30.md`
(three sources, A/B scored; the disagreements are recorded there, not averaged).

---

## D-0. A diagnostic never enters the reward

No diagnostic may gate, weight, scale or filter a unit's terminal reward. A diagnostic may appear in
a report **only** in a section marked `diagnostic`, and never in the same table as a reward, so that
no reader can mistake one for the other. This is a restatement of § 6/§ 7, not an addition to them.

*Verified by:* grepping the analysis code for diagnostic names and confirming none appears in the
reward path. *Failure mode it prevents:* a style score silently becoming a quality claim.

---

## D-1. Every style diagnostic is a RATIO against a human base rate

**Required.** A diagnostic value is `marker_rate(text) / marker_rate(human_baseline)` for the same
task and register. The manifest records the numerator rate and the denominator rate, each with the
corpus it came from.

**Forbidden.** A fixed lexicon count reported as a score. The 2026 source explicitly reports that the
"entire classic slop lexicon … is BUSTED in 2026 models", so a wordlist measures the list, not the text.

**Positive control.** A known default LLM output must score >3× the human baseline.
**Negative control.** The author's own unaided text must score ≤1× the human baseline.
**Both must hold.** One control passing is a failed diagnostic — the two are the only thing separating
"we measured slop" from "we measured our own opinion".

*Failure mode it prevents:* a diagnostic that fires on human writing, which would silently penalise
the very thing the benchmark rewards.

---

## D-2. Normalisation uses a fixed external anchor; the roster is never the scale

**Required.** Any reported score is `f(text, fixed_human_baseline)`. It must not depend on the current
roster's maximum or minimum, nor on which other models happen to be in the run.

**Invariance test — the acceptance condition.** The same text and the same baseline must produce a
**byte-identical** score when the roster is 17 models and when it is 3 models. If it moves, the score
is roster-relative and must be withdrawn.

**Scope declaration.** If a baseline exists only for some locales or registers, the diagnostic carries
`scope: subset` and names them, and must not be reported as a general axis.

*Failure mode it prevents:* a score that changes when the roster changes, which makes cross-run
comparison meaningless while looking precise. This is the same defect as a checker that copies a rule
instead of calling it: the copy drifts, and nobody notices because the output still looks like a number.

---

## D-3. Homogenization is measured within a prompt, across samples — never across models

**Required.** Opener reuse and skeleton repetition are computed between N samples of the **same prompt
from the same model**, at a declared N. Cross-model comparison of this quantity is meaningless and must
not be reported.

**Positive control.** N samples from one model on one prompt show measurable reuse.
**Negative control.** Texts by N *different* human authors show a significantly lower reuse rate.
**If the two controls do not separate, delete the diagnostic** — do not report it with a caveat.

*Failure mode it prevents:* reporting "model X is more templated than model Y" when the measure never
established that templating is measurable at all.

---

## D-4. Punctuation diagnostics are locale-checked before use, and `——` is not `—`

**Required.** In the en / zh-Hant / zh-Hans design, any punctuation-based diagnostic is validated for
**locale invariance** on parallel translations before it is reported. The full-width Chinese dash
`——` (U+2014 ×2) is a different code point and a different keystroke path from `—`; they must not share
a regex, and per-locale punctuation rates must never be compared across locales.

**Acceptance condition.** The same content in three locales produces values inside a pre-declared band
(for example ±10% relative). Outside it, the diagnostic is marked `locale-confounded` and withdrawn
for that locale.

**Also required.** Distinguish the *glyph* (a typographic fingerprint, which measures an era and a
keyboard) from the *rhetorical move* (the dash used as a connective). The 2026 sources establish that
counting only U+2014 measures "the writer's ERA AND KEYBOARD, not their slop", and separately that
2026-era models have largely stopped over-using it — together those mean a plain em-dash count is
neither a slop measure nor a current one.

*Failure mode it prevents:* a locale confound reported as a quality finding, in a benchmark whose entire
design is locale-stratified.

---

## D-5. Every diagnostic expires, and its retirement is recorded

**Required.** Each diagnostic binds:
- a `probe_set` — 20 frozen texts: 10 human, 5 known-slop, 5 borderline;
- a `separation` figure — mean(positive control) − mean(negative control) on that probe set;
- an `expires_on` trigger — a roster version change, or a declared date.

On trigger, the probe set is re-run and separation re-measured. **If separation falls below 80% of its
prior value, or if positive and negative controls overlap by more than a pre-declared threshold, the
diagnostic is retired** — removed from reports, not annotated.

**Retirement is itself recorded**, with the date and the measured separation that caused it. A diagnostic
quietly left in place past its evidence is the failure this requirement exists to prevent.

*Provenance:* the 2026 sources report that 55–72% of tells are unique to a single model version and 65%
to a single family, and that tells "shift between model versions, but they do not go away". A tell list
is therefore a perishable instrument, and an instrument with no expiry is an assumption.

---

## What this file is not

It is **not** an AI detector. It never scores whether a text was written by a model, and D-4's one-shot
saturation check (if run at all) records an environment fact in the episode manifest — it does not enter
a gate, a score, or a report table.

It is **not** a change to the frozen contract or to `studies/revision-v1.yaml`. Requirements D-1, D-2,
D-4 and D-5 are diagnostic-layer only. Adding a diagnostic to the **preregistered endpoint set** would
change `revision-v1.yaml` and is an owner decision, not an agent one.

## Removal condition

This file is deleted if, and only if, the program reports no diagnostics at all. It is a contract for
instruments we choose to report; if we never report one, the contract has nothing to govern and should
not be carried.
