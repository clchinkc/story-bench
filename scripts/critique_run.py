"""Build the seven requests of one critique_improvement episode, and prove they all construct.

THE PROBLEM THIS SOLVES. `pilot_run` builds one request per row from `PromptBuilder`, and
`PromptBuilder` has no `critique_improvement` builder — that is the "Unknown task type" that kept
those 72 rows out of every run. The episode's prompts come from `AgenticPromptBuilder` instead, and
its turns alternate between TWO routes: the row's model drafts and revises, the study's critic route
critiques. A request per row cannot express either fact.

WHAT IS AND IS NOT HERE. `build_specs` constructs every turn's `AttemptSpec` — route, artifacts,
settings, attempt id — without touching a ledger or a socket. That is the risky half: a wrong route,
a reused attempt id, or a settings dict that drifts from the route would each be a paid mistake.
The sending loop is then `run_episode(task, send, ...)` with `send` bound to `execute_attempt`, and
it is deliberately not built here.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import provider_attempts as pa  # noqa: E402
import provider_prices as pp  # noqa: E402

import critique_episode as ce  # noqa: E402
import dispatch_runner as dr  # noqa: E402
import rehearse_attempt as reh  # noqa: E402
from measurement_contract import digest  # noqa: E402

# The critic's calls are grading work, so they draw the `data_judge` envelope with the research cost
# role; the subject's drafts and revisions are generation and draw `pilot`. The validator pairs the
# two, and pairing them wrongly is refused as "invalid envelope/activity/cost role".
SUBJECT_SPEND = dict(envelope="pilot", cost_role="deployment", activity_role="generation")
CRITIC_SPEND = dict(envelope="data_judge", cost_role="research",
                    activity_role="research_evaluation")


def critic_route(study: dict) -> str:
    route = (study.get("critique_episode") or {}).get("critic_route")
    if not isinstance(route, str) or not route:
        raise SystemExit("studies/revision-v1.yaml declares no critique_episode.critic_route; the "
                         "critic is a second model by design and choosing one here would be a guess")
    return route


def turn_fields(turn, limits) -> pa.RequestFields:
    """The request for one turn, carrying the ONE reasoning setting the route's rule selects.

    THE CRITIC GETS THE SAME BUDGET AS THE SUBJECT, and that is not laziness. A separate, smaller
    critic cap was tried at 1,500 tokens; gemini at `effort: high` spent 1,497 of them thinking and
    returned zero characters -- `outcome: truncated`, $0.0177 for an empty critique. Under `effort`
    the thinking shares `max_tokens` with the answer, so a cap sized for "just a critique" is a cap
    sized for nothing. The prompt still asks for feedback rather than a rewrite; the budget is the
    route's, not the role's.
    """
    cap = limits.max_tokens
    messages = tuple(turn.messages)
    if limits.encoding == "max_tokens":
        return pa.RequestFields(messages=messages, max_tokens=cap,
                                reasoning_max_tokens=limits.reasoning_max_tokens)
    return pa.RequestFields(messages=messages, max_tokens=cap, reasoning_effort=limits.effort)


def build_specs(base, raw: bytes, index: int, unit: str, *, study: dict, rounds: int, builder):
    """Every turn's (spec, fields, snapshot), in order, with a fake response for prompt chaining.

    The fake responses are `DRY-<kind>-<n>`. They are only used to satisfy the next turn's prompt,
    which is why this can be run without a model: the SPEC is real, the text is a placeholder.
    """
    manifest = json.loads(raw)
    row = manifest["assignments"][index]
    assignment_hash = digest(manifest)
    critic = critic_route(study)
    specs, seen = [], set()

    def send(turn, model):
        record, endpoints, profile, artifact = reh.load_route(base, model)
        identity = record["route_identity"]
        snapshot = pp.validate_snapshot(
            ((base / "openrouter-models.json").read_bytes(), endpoints, profile, artifact),
            reh._route(identity), identity["observed_at"], identity["expires_at"])
        limits = reh.study_limits(row["task"]["task_type"], model)
        fields = turn_fields(turn, limits)
        # BOTH the model id and its snapshot must follow the turn's route. Overriding only `model`
        # left `model_snapshot` pointing at the subject, so the critic's spec claimed opus while its
        # route was gemini -- exactly the drift `verify_plan` refuses as "assigned model differs
        # from route". Caught by the dry build, before a dollar was spent.
        turn_row = {**row, "model": model, "model_snapshot": identity["canonical_model_slug"]}
        artifacts = reh.artifacts_for(turn_row, fields, raw, base)
        attempt_id = dr.attempt_id(model, f"{unit}-r{turn.index}", "A")
        if attempt_id in seen:
            raise SystemExit(f"two turns claim attempt id {attempt_id!r}; each turn is one ledger "
                             f"slot, forever")
        seen.add(attempt_id)
        spec = reh.spec_for(turn_row,
                            dict(attempt_id=attempt_id, unit_id=f"{unit}-r{turn.index}"),
                            fields.reasoning_effort or f"tokens:{fields.reasoning_max_tokens}",
                            artifacts, assignment_hash,
                            **(CRITIC_SPEND if turn.role == ce.CRITIC else SUBJECT_SPEND))
        specs.append(dict(turn=turn, spec=spec, fields=fields, snapshot=snapshot,
                          artifacts=artifacts))
        return f"DRY-{turn.kind}-{turn.index}"

    ce.run_episode(row["task"], send, rounds=rounds, builder=builder,
                   subject_model=row["model"], critic_model=critic)
    return specs


def run_row(*, base, assignment, index, unit, evidence, ledger, authorise, study, builder,
            rounds, announce=True):
    """Run one critique episode and return a summary the pilot runner can record as ONE row.

    Returns `dict(turns, states, cost_usd, attempt_ids)`. The caller decides what a partial episode
    means; this function refuses to return at all if a turn fails, so a 3-call accident can never be
    filed as a 7-call episode.
    """
    import asyncio

    import paid_attempt as paid

    paid._token(authorise)
    raw = pathlib.Path(assignment).read_bytes()
    import json as _json
    row = _json.loads(raw)["assignments"][index]
    critic = critic_route(study)
    evidence = pathlib.Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    sent, ids = [], []

    def send(turn, model):
        limits = reh.study_limits(row["task"]["task_type"], model)
        fields = turn_fields(turn, limits)
        spend = CRITIC_SPEND if turn.role == ce.CRITIC else SUBJECT_SPEND
        attempt_id = dr.attempt_id(model, f"{unit}-r{turn.index}", row["condition"])
        _, plan, admission, _ = paid.prepared(
            pathlib.Path(base), raw, index, f"{unit}-r{turn.index}", evidence,
            fields=fields, model=model, attempt_id=attempt_id, spend=spend)
        result = asyncio.run(pa.execute_attempt(plan, ledger, admission))
        sent.append(dict(turn=turn.index, kind=turn.kind, role=turn.role, model=model,
                         attempt_id=attempt_id, state=result.ledger_state))
        ids.append(attempt_id)
        if announce:
            print(f"    {turn.index} {turn.kind:9s} {model[:30]:30s} {result.ledger_state}", flush=True)
        return _response_text(plan, attempt_id)

    ce.run_episode(row["task"], send, rounds=rounds, builder=builder,
                   subject_model=row["model"], critic_model=critic)
    total = sum((ledger.attempt(i)["actual_micros"] or 0) for i in ids if i in set(ledger.status()["attempt_ids"])) / 1e6
    return dict(turns=len(sent), states=[s["state"] for s in sent], cost_usd=total,
                attempt_ids=tuple(ids))


def execute_episode(args, row, index, rounds, study, *, builder):
    """The CLI path: run one episode and report it."""
    import os

    for required, flag in ((args.ledger, "--ledger"), (args.evidence, "--evidence"),
                           (args.authorise, "--authorise")):
        if required is None:
            raise SystemExit(f"--execute requires {flag}")
    if not os.environ.get(pa.CREDENTIAL_ENV):
        raise SystemExit(f"{pa.CREDENTIAL_ENV} absent; refusing an episode that cannot send")
    ledger = __import__("spend_ledger").Ledger.open(args.ledger, program_id=reh.PROGRAM_ID,
                                                    store_id=reh.STORE_ID)
    summary = run_row(base=args.base, assignment=args.assignment, index=index, unit=args.unit,
                      evidence=args.evidence, ledger=ledger, authorise=args.authorise,
                      study=study, builder=builder, rounds=rounds)
    print(f"\n  episode complete: {summary['turns']} turns, ${summary['cost_usd']:.4f}")
    for i, state in enumerate(summary["states"]):
        print(f"    slot {i} {state}")
    return 0

def _response_text(plan, attempt_id):
    """The text a turn produced, from the evidence the turn left behind.

    Read back from disk rather than from memory: the evidence IS the record of what happened, and a
    critique that is fed forward from a variable could differ from the one that was paid for.
    """
    directory = pathlib.Path(plan.evidence_root) / "attempts" / attempt_id
    body = directory / "response.bin"
    if not body.is_file():
        raise SystemExit(f"turn {attempt_id}: no response.bin; a turn with no evidence is not a turn")
    payload = json.loads(body.read_bytes())
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not text:
        raise SystemExit(f"turn {attempt_id}: the provider returned no text "
                         f"(outcome={((payload.get('usage') or {}).get('cost'))})")
    return text


def main(argv=None):
    import argparse

    import yaml

    from agentic_generator import AgenticPromptBuilder  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--assignment", type=pathlib.Path, required=True)
    ap.add_argument("--index", type=int, default=None,
                    help="row index in the manifest. Omitted: the FIRST critique_improvement row is "
                         "used, because making the caller compute an index is how a good command "
                         "turns into a wrong one -- --index 0 is beat_interpolation")
    ap.add_argument("--unit", default="critique-dry")
    ap.add_argument("--study", type=pathlib.Path, default=None)
    ap.add_argument("--execute", action="store_true",
                    help="actually send the seven calls. Requires --ledger, --evidence, --authorise "
                         "and the credential; without it this only builds and verifies the requests")
    ap.add_argument("--ledger", type=pathlib.Path, default=None)
    ap.add_argument("--evidence", type=pathlib.Path, default=None)
    ap.add_argument("--authorise", default=None)
    ap.add_argument("--stop-after-failures", type=int, default=1)
    ap.add_argument("--verify", action="store_true",
                    help="run each spec through prepare_attempt in a throwaway root: this is the "
                         "drift/admission check that would otherwise run after reserving money")
    args = ap.parse_args(argv)

    study_path = args.study or (reh.REPO / "studies" / "revision-v1.yaml")
    study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    episode = study.get("critique_episode") or {}
    rounds = int(episode.get("rounds", 3))
    raw = args.assignment.read_bytes()
    rows = json.loads(raw)["assignments"]
    index = args.index
    if index is None:
        index = next((i for i, r in enumerate(rows) if r["task_family"] == "critique_improvement"), None)
        if index is None:
            raise SystemExit("this manifest has no critique_improvement row")
        print(f"  row {index} chosen (first critique_improvement row): {rows[index]['task_id']} "
              f"{rows[index]['model']}")
    elif rows[index]["task_family"] != "critique_improvement":
        raise SystemExit(f"row {index} is {rows[index]['task_family']}, not critique_improvement; "
                         f"omit --index to take the first one")
    args.index = index
    row = rows[index]

    specs = build_specs(args.base, raw, args.index, args.unit, study=study, rounds=rounds,
                        builder=AgenticPromptBuilder)
    print(f"  {len(specs)} turns built for {row['task_id']} (subject {row['model']})")
    for item in specs:
        turn, spec = item["turn"], item["spec"]
        print(f"    {turn.index} {turn.kind:9s} {turn.role:8s} "
              f"{spec.assigned_model_snapshot[:36]:36s} max_tokens={item['fields'].max_tokens:6d}")
    print(f"  distinct attempt ids: {len({i['spec'].attempt_id for i in specs})} of {len(specs)}")

    if args.execute:
        return execute_episode(args, row, index, rounds, study, builder=AgenticPromptBuilder)

    if args.verify:
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            # `.resolve()` because macOS /var is a symlink and the evidence root refuses one.
            root = pathlib.Path(root).resolve()
            for item in specs:
                plan = pa.prepare_attempt(item["spec"], item["snapshot"],
                                          tuple(sorted(item["artifacts"].items())),
                                          item["fields"], root)
                print(f"    verified turn {item['turn'].index} -> {plan.spec.attempt_id[-24:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
