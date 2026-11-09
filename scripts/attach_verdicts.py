#!/usr/bin/env python3
"""Join judge verdicts onto collected units, producing the JSONL `analyze_study` reads.

    uv run python scripts/attach_verdicts.py --units results/revision-v1/units.jsonl \
        --judgements <evidence>/judgements --out results/revision-v1/units-scored.jsonl

WHY THIS IS A SEPARATE STEP FROM `collect_units`. Collection must not invent a score. It emits
`resolved: null` for every arm because at collection time no judge has read the generation, and
writing `false` there would enter every arm as a task failure. This step is the one that turns a null
into a measured value, and it may only do so from a verdict that actually parsed.

THE JOIN IS BY ATTEMPT ID, NOT BY NAME, and that is the whole point. A unit record and a verdict
record describe the same two generations in two vocabularies: the unit carries `model`
(`anthropic/claude-opus-5.5`) and `task_family` (`beat_interpolation`), while the verdict carries
`subject_model_snapshot` (`anthropic/claude-opus-5.5-20260921`) and `task_id`
(`beat_interpolation_003`). Matching those by prefix or by "close enough" is how a run quietly
compares unlike things -- the interval still prints and nothing looks wrong. The verdict already
records `control_attempt_id` and `treatment_attempt_id`, and the unit already records the attempt id
of each arm, so the join is an exact string equality on a value nobody had to normalise.

WHICH ARM IS WHICH. `control_attempt_id` is condition A and `treatment_attempt_id` is condition D.
That is the same convention `judge_units` blinds into slots and back out again; if it ever disagreed,
the D-minus-A lift would flip sign, so the script also refuses a verdict whose two attempt ids are
equal or empty.

A VERDICT THAT FAILED SCORES NOTHING. `verdict.json` records a `failure` when the reply did not
parse, contradicted itself twice, or never completed. Such a verdict contributes no `resolved`; the
unit keeps `resolved: null` plus `invalid_kind: "grader"`, which `study_analysis` excludes from the
point estimate. An arm nobody scored is not a task the writer failed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ARMS = ("A", "D")
CONTROL, TREATMENT = "control_attempt_id", "treatment_attempt_id"
# Only `verdict.json` is the verdict of record. `verdict.rejected.json` is the reading the corrective
# re-ask replaced; attaching it would score the answer the judge was told to redo.
VERDICT_FILE = "verdict.json"


def _load_units(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _claims(document, path):
    """The two attempt ids a verdict claims, or a reason it cannot be used."""
    judgement = document.get("judgement_id", path.parent.name)
    control, treatment = document.get(CONTROL), document.get(TREATMENT)
    if not control or not treatment or control == treatment:
        return None, dict(judgement_id=judgement,
                          reason=f"control/treatment attempt ids unusable: "
                                 f"{control!r} / {treatment!r}")
    return (control, treatment), None


def _usable(document, path):
    """(index entries, failure note, refusal note) for one verdict document."""
    judgement = document.get("judgement_id", path.parent.name)
    if document.get("failure") is not None:
        return None, dict(judgement_id=judgement, failure=str(document["failure"])[:200]), None
    claims, refusal = _claims(document, path)
    if refusal is not None:
        return None, None, refusal
    analysis = document.get("analysis") or {}
    if set(analysis) != set(ARMS):
        return None, None, dict(judgement_id=judgement,
                                reason=f"analysis names arms {sorted(analysis)}, expected {ARMS}")
    entries = []
    for arm, attempt_id in zip(ARMS, claims):
        entry = analysis[arm]
        entries.append((attempt_id, dict(
            judgement_id=judgement, arm=arm, resolved=entry.get("resolved"),
            quality=entry.get("quality"), invalid_kind=entry.get("invalid_kind"),
            corrective_reask=document.get("corrective_reask"))))
    return entries, None, None


def _verdict_index(directory):
    """attempt id -> verdict reading, plus what was skipped and why.

    Refuses a duplicate claim on one attempt id rather than letting the last file win: two verdicts
    for one generation means the ledger was replayed against a different rubric, and silently picking
    one would hide that.
    """
    index, failed, refused = {}, [], []
    for path in sorted(pathlib.Path(directory).glob(f"*/{VERDICT_FILE}")):
        entries, failure, refusal = _usable(json.loads(path.read_text(encoding="utf-8")), path)
        if failure is not None:
            failed.append(failure)
            continue
        if refusal is not None:
            refused.append(refusal)
            continue
        for attempt_id, entry in entries:
            if attempt_id in index:
                refused.append(dict(
                    judgement_id=entry["judgement_id"],
                    reason=f"attempt {attempt_id} already scored by "
                           f"{index[attempt_id]['judgement_id']}"))
                continue
            index[attempt_id] = entry
    return index, failed, refused


def _apply_arm(payload, verdict):
    """Copy one verdict's reading onto one arm payload. 'none' when there was no verdict.

    A verdict that reaches here has already parsed: `_verdict_index` drops every document that
    recorded a `failure`. So `resolved: None` here is a MEASUREMENT -- the judge scored the criteria
    and found the submission not fully resolved -- and it is carried through verbatim. Rewriting it
    to `invalid_kind: "grader"` would exclude the pair from the point estimate and turn "did not
    resolve" into "was never measured", which biases the estimate upward. `grader` belongs only to an
    arm with no usable verdict at all.
    """
    if verdict is None:
        payload["resolved"] = None
        payload["invalid_kind"] = "grader"
        return "none"
    payload["resolved"] = verdict["resolved"]
    payload["invalid_kind"] = verdict["invalid_kind"]
    if verdict["quality"] is not None:
        payload["quality"] = verdict["quality"]
    if verdict["corrective_reask"]:
        payload["judge_reasked"] = True
    return "scored"


def _tally(stats, scored, total):
    if scored == total:
        stats["unit_fully_scored"] += 1
    elif scored:
        stats["unit_partially_scored"] += 1
    else:
        stats["unit_unscored"] += 1


def _attach(units, index):
    stats = Counter()
    unmatched = set(index)
    for unit in units:
        arms = unit.get("arms") or {}
        outcomes = Counter()
        for arm in sorted(arms):
            payload = arms[arm]
            attempt_id = payload.get("attempt_id")
            outcome = _apply_arm(payload, index.get(attempt_id))
            if outcome != "none":
                unmatched.discard(attempt_id)
            outcomes[outcome] += 1
            stats[f"arm_{outcome if outcome != 'none' else 'no_verdict'}"] += 1
        _tally(stats, outcomes["scored"], len(arms))
    return stats, unmatched


def _report(units, stats, index, failed, refused, unmatched, out):
    print(f"units                 : {len(units)}")
    print(f"  fully scored        : {stats['unit_fully_scored']}")
    print(f"  partially scored    : {stats['unit_partially_scored']}")
    print(f"  unscored            : {stats['unit_unscored']}")
    print(f"arms scored           : {stats['arm_scored']}")
    print(f"arms grader-invalid   : {stats['arm_invalid']}")
    print(f"arms with no verdict  : {stats['arm_no_verdict']}")
    print(f"verdicts usable       : {len({entry['judgement_id'] for entry in index.values()})}")
    print(f"verdicts failed       : {len(failed)}")
    for item in failed[:5]:
        print(f"  failed: {item['judgement_id']}: {item['failure'][:100]}")
    print(f"verdicts refused      : {len(refused)}")
    for item in refused[:5]:
        print(f"  refused: {item['judgement_id']}: {item['reason'][:120]}")
    print(f"attempt ids matching nothing: {len(unmatched)}")
    for attempt_id in sorted(unmatched)[:5]:
        print(f"  unmatched: {attempt_id}")
    print(f"wrote                 : {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", type=pathlib.Path, required=True,
                        help="the UNSCORED JSONL from collect_units")
    parser.add_argument("--judgements", type=pathlib.Path, required=True,
                        help="the evidence directory's `judgements/` folder")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--allow-unmatched", action="store_true",
                        help="exit 0 even when a verdict matched no unit (a partly-judged run)")
    args = parser.parse_args(argv)

    units = _load_units(args.units)
    index, failed, refused = _verdict_index(args.judgements)
    stats, unmatched = _attach(units, index)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit, sort_keys=True) + "\n")

    _report(units, stats, index, failed, refused, unmatched, args.out)

    if refused:
        print("REFUSED: a verdict claimed an attempt id twice, or its arms were unusable",
              file=sys.stderr)
        return 1
    if unmatched and not args.allow_unmatched:
        print("REFUSED: verdicts matched no unit; pass --allow-unmatched to keep an "
              "unfinished run's partial join", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
