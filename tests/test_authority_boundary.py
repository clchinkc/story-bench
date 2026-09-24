"""W2c authority-boundary tests (risk rows F1-F8 and B5).

B5 drives the matched task through the BASIC protected MCP only (the stdio
server process), with the arm-B package loaded and the specialized namespace
absent, to a persisted accepted transition, and separately rejects an
unauthorized proposal with accepted bytes preserved. A no-op or
always-rejecting propose channel fails the task.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import environment_contract as ec
import native_workspace as nw
from authority_boundary import (
    PROPOSE_PHASES,
    AuthorityBoundary,
    Entitlement,
    OwnerWaveAuthority,
)
from protected_mcp import McpConfig, ProtectedMcpServer
from conftest import CooperativeNativeEngine

WRITES = [{"path": "stories/scene-01.md", "content": "Scene one text.\n"}]
GENESIS_SOURCE = {"id": "story-x", "title": "X", "planSource": "# Plan source\n"}


def _run_matched(propose, waves):
    """The S13.1 matched task: GROUND -> FRAME -> OWNER WAVE -> REALIZE -> PROVE
    -> OWNER WAVE -> RECONCILE -> ADVANCE, driven only through propose."""
    g1 = waves.issue(request_id="req-genesis", decision="apply", revision=0)
    genesis = propose(
        {
            "phase": "genesis",
            "move": "plan-change",
            "source": GENESIS_SOURCE,
            "owner_wave_ref": g1,
            "expected_revision": 0,
        }
    )
    frame = propose({"phase": "frame", "move": "scene-draft", "writes": WRITES, "rationale": "bounded change"})
    if frame.get("status") == "refused":
        return {"status": frame.get("status"), "genesis": genesis, "frame": frame}
    request = frame["pending_owner_wave"]
    realize_wave = waves.issue(request_id=request, decision="apply", revision=frame["revision"])
    realize = propose(
        {
            "phase": "realize",
            "move": "scene-draft",
            "writes": WRITES,
            "owner_wave_ref": realize_wave,
            "expected_revision": frame["revision"],
        }
    )
    candidate = realize["candidate_bundle"]
    payload_sha256 = ec.digest({entry["path"]: entry["content"] for entry in WRITES})
    prove = propose(
        {
            "phase": "prove",
            "candidate_bundle": candidate,
            "evidence_kind": "proof-draft",
            "evidence": {"candidate_bundle": candidate, "payload_sha256": payload_sha256},
        }
    )
    evaluation = prove["evaluation_id"]
    resume = propose({"phase": "resume", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": evaluation})
    result_request = resume["pending_owner_wave"]
    reconcile_wave = waves.issue(
        request_id=result_request, decision="apply", candidate_bundle=candidate, revision=resume["revision"]
    )
    reconcile = propose(
        {
            "phase": "reconcile",
            "move": "scene-draft",
            "candidate_bundle": candidate,
            "proof_evaluation": evaluation,
            "owner_wave_ref": reconcile_wave,
            "expected_revision": resume["revision"],
        }
    )
    return {
        "status": reconcile.get("status"),
        "genesis": genesis,
        "frame": frame,
        "realize": realize,
        "prove": prove,
        "resume": resume,
        "reconcile": reconcile,
        "candidate_bundle": candidate,
        "evaluation_id": evaluation,
    }


# --------------------------------------------------------------------------
# F1
# --------------------------------------------------------------------------
def test_unchanged_files_preserved_end_to_end(w2c_harness):
    instance, server, boundary, engine, waves = w2c_harness("B")
    before = nw.census_files(instance.destination)
    result = _run_matched(boundary.propose, waves)
    assert result["status"] == "accepted"
    after = nw.census_files(instance.destination)
    changed = sorted(set(after.paths()) - set(before.paths()))
    assert "stories/scene-01.md" in changed
    # files outside the authorized transition keep their exact bytes
    for rel in ("plan/source.md", "sources/brief.md"):
        assert after.entries[rel] == before.entries[rel], rel
    # red control: a reset that silently rewrites an unrelated file is detected
    (instance.destination / "sources" / "brief.md").write_text("tampered\n", encoding="utf-8")
    assert nw.census_files(instance.destination).entries["sources/brief.md"] != before.entries["sources/brief.md"]


# --------------------------------------------------------------------------
# F2
# --------------------------------------------------------------------------
def test_changed_files_bound_to_native_transition(w2c_harness):
    instance, server, boundary, engine, waves = w2c_harness("B")
    result = _run_matched(boundary.propose, waves)
    assert result["status"] == "accepted"
    candidate = json.loads((instance.destination / ".narrative" / "candidates" / (result["candidate_bundle"] + ".json")).read_text(encoding="utf-8"))
    evaluation = json.loads((instance.destination / ".narrative" / "evaluations" / (result["evaluation_id"] + ".json")).read_text(encoding="utf-8"))
    reconciles = sorted((instance.destination / ".narrative" / "reconciles").glob("*.json"))
    assert reconciles, "no reconcile record persisted"
    reconcile = json.loads(reconciles[-1].read_text(encoding="utf-8"))
    assert evaluation["candidate_bundle"] == result["candidate_bundle"]
    assert reconcile["candidate_bundle"] == result["candidate_bundle"]
    assert reconcile["proof_evaluation"] == result["evaluation_id"]
    payload = candidate["payloads"]["stories/scene-01.md"]
    assert (instance.destination / "stories" / "scene-01.md").read_text(encoding="utf-8") == payload

    # red control: a changed file with no bound native transition is detected
    stray = instance.destination / "stories" / "stray.md"
    stray.write_text("unbound\n", encoding="utf-8")
    bound = {entry for record in reconciles for entry in json.loads(record.read_text(encoding="utf-8"))["changed_paths"]}
    assert "stories/stray.md" not in bound


# --------------------------------------------------------------------------
# F3
# --------------------------------------------------------------------------
def test_rejection_preserves_accepted_bytes(w2c_harness):
    instance, server, boundary, engine, waves = w2c_harness("B")
    _run_matched(boundary.propose, waves)
    fingerprint = boundary.accepted_fingerprint()
    before = nw.census_files(instance.destination).digest

    denials = [
        {"phase": "reconcile", "move": "scene-draft", "candidate_bundle": "cb-missing", "proof_evaluation": "eval-missing", "owner_wave_ref": "wave-forged"},
        {"phase": "realize", "move": "scene-draft", "writes": WRITES, "owner_wave_ref": "wave-forged"},
        {"phase": "advance"},
        {"phase": "frame", "move": "scene-draft", "writes": [{"path": "../escape.md", "content": "x"}]},
        {"phase": "frame", "move": "scene-draft", "writes": WRITES, "expected_revision": 0},
    ]
    for proposal in denials:
        result = boundary.propose(proposal)
        assert result["status"] == "refused", proposal
        assert boundary.accepted_fingerprint() == fingerprint, proposal
    assert nw.census_files(instance.destination).digest == before

    # red control: a partial mutation hidden by a later rejection is detected
    (instance.destination / "stories" / "scene-01.md").write_text("partial\n", encoding="utf-8")
    assert nw.census_files(instance.destination).digest != before


# --------------------------------------------------------------------------
# F4
# --------------------------------------------------------------------------
def test_scope_escalation_traversal_symlink_refused(w2c_harness, tmp_path):
    instance, server, boundary, engine, waves = w2c_harness("B")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret\n", encoding="utf-8")
    (instance.destination / "stories").mkdir(exist_ok=True)
    try:
        (instance.destination / "stories" / "link.md").symlink_to(outside / "secret.md")
        symlinked = True
    except OSError:
        symlinked = False
    (instance.destination / "stories" / "evil").symlink_to(outside, target_is_directory=True)

    cases = [
        {"phase": "realize", "writes": [{"path": "../escape.md", "content": "x"}], "move": "scene-draft"},
        {"phase": "realize", "writes": [{"path": "/etc/hosts", "content": "x"}], "move": "scene-draft"},
        {"phase": "realize", "writes": [{"path": "results/leak.md", "content": "x"}], "move": "scene-draft"},
        {"phase": "realize", "writes": [{"path": "stories/evil/leak.md", "content": "x"}], "move": "scene-draft"},
    ]
    if symlinked:
        cases.append({"phase": "realize", "writes": [{"path": "stories/link.md", "content": "x"}], "move": "scene-draft"})
    fingerprint = boundary.accepted_fingerprint()
    for proposal in cases:
        result = boundary.propose(proposal)
        assert result["status"] == "refused", proposal
        assert result["code"] in {"scope-refused", "phase-refused", "owner-wave-refused", "stale-refused"}
    assert boundary.accepted_fingerprint() == fingerprint

    # an allowed move alone cannot authorize another path
    result = boundary.propose({"phase": "realize", "move": "scene-draft", "writes": [{"path": "results/leak.md", "content": "x"}]})
    assert result["status"] == "refused"
    assert result["code"] == "scope-refused"

    # red control: the same path passes once it is inside the declared scope
    assert boundary.entitlement.permits_write("stories/scene-01.md")
    assert not boundary.entitlement.permits_write("results/leak.md")


# --------------------------------------------------------------------------
# F5
# --------------------------------------------------------------------------
def test_stale_and_replayed_approval_refused(w2c_harness):
    instance, server, boundary, engine, waves = w2c_harness("B")
    frame = boundary.propose({"phase": "frame", "move": "scene-draft", "writes": WRITES})
    request = frame["pending_owner_wave"]
    wave = waves.issue(request_id=request, decision="apply", revision=frame["revision"])
    realize = boundary.propose(
        {"phase": "realize", "move": "scene-draft", "writes": WRITES, "owner_wave_ref": wave, "expected_revision": frame["revision"]}
    )
    assert realize["status"] == "ok"

    # red control: replaying the same owner wave is refused
    replayed = boundary.propose(
        {"phase": "realize", "move": "scene-draft", "writes": WRITES, "owner_wave_ref": wave, "expected_revision": frame["revision"]}
    )
    assert replayed["status"] == "refused"
    assert replayed["code"] == "stale-refused"

    # red control: a wave bound to one candidate cannot authorize another
    candidate = realize["candidate_bundle"]
    other = waves.issue(request_id="req-other", decision="apply", candidate_bundle="cb-other", revision=1)
    mismatch = boundary.propose(
        {"phase": "reconcile", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": "eval-x", "owner_wave_ref": other, "expected_revision": 1}
    )
    assert mismatch["status"] == "refused"

    # red control: a stale expected revision is refused
    stale = boundary.propose({"phase": "frame", "move": "scene-draft", "writes": WRITES, "expected_revision": 99})
    assert stale["status"] == "refused"
    assert stale["code"] == "stale-refused"


# --------------------------------------------------------------------------
# F6
# --------------------------------------------------------------------------
def test_conflicting_concurrent_proposals_exactly_one_accepted(w2c_harness):
    instance, server, boundary, engine, waves = w2c_harness("B")
    _run_matched(boundary.propose, waves)
    frame = boundary.propose({"phase": "frame", "move": "scene-draft", "writes": [{"path": "stories/scene-02.md", "content": "Two.\n"}]})
    request = frame["pending_owner_wave"]
    wave = waves.issue(request_id=request, decision="apply", revision=frame["revision"])
    realize = boundary.propose(
        {"phase": "realize", "move": "scene-draft", "writes": [{"path": "stories/scene-02.md", "content": "Two.\n"}], "owner_wave_ref": wave, "expected_revision": frame["revision"]}
    )
    candidate = realize["candidate_bundle"]
    payload_sha256 = ec.digest({"stories/scene-02.md": "Two.\n"})
    evaluation = boundary.propose(
        {"phase": "prove", "candidate_bundle": candidate, "evidence_kind": "proof-draft", "evidence": {"candidate_bundle": candidate, "payload_sha256": payload_sha256}}
    )["evaluation_id"]
    resume = boundary.propose({"phase": "resume", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": evaluation})
    reconcile_wave = waves.issue(request_id=resume["pending_owner_wave"], decision="apply", candidate_bundle=candidate, revision=resume["revision"])
    proposal = {
        "phase": "reconcile", "move": "scene-draft", "candidate_bundle": candidate,
        "proof_evaluation": evaluation, "owner_wave_ref": reconcile_wave, "expected_revision": resume["revision"],
    }
    results = []
    lock = threading.Lock()

    def worker():
        outcome = boundary.propose(dict(proposal))
        with lock:
            results.append(outcome["status"])

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("accepted") == 1, results
    assert results.count("refused") == 5, results

    # red control: without the currentness check, distinct waves for the same
    # revision would let a second conflicting write through
    loose = AuthorityBoundary(
        engine=engine,
        workspace_root=instance.destination,
        entitlement=boundary.entitlement,
        owner_waves=OwnerWaveAuthority(),
        arm="B",
    )
    loose_results = []
    for _ in range(3):
        w = loose.owner_waves.issue(request_id="req-loose", decision="apply", candidate_bundle=candidate, revision=None)
        loose_results.append(
            loose.propose({"phase": "reconcile", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": evaluation, "owner_wave_ref": w})["status"]
        )
    assert "accepted" in loose_results


# --------------------------------------------------------------------------
# F7
# --------------------------------------------------------------------------
def test_outcome_comes_from_native_evaluation_records(w2c_harness):
    instance, server, boundary, engine, waves = w2c_harness("B")
    result = _run_matched(boundary.propose, waves)
    assert result["status"] == "accepted"
    evaluation_path = instance.destination / ".narrative" / "evaluations" / (result["evaluation_id"] + ".json")
    assert evaluation_path.is_file()
    observation = boundary.observations()[-1]
    assert observation["status"] == "accepted"
    assert boundary.authorities()["evaluations"], "no native evaluation record"
    # no parallel outcome store is written
    assert not (instance.destination / "outcome.json").exists()
    assert not (instance.destination / "results").exists()

    # red control: a fabricated outcome with no native record is detected
    native_ids = set(boundary.authorities()["evaluations"])
    assert "eval-fabricated" not in native_ids
    assert result["evaluation_id"] in native_ids


# --------------------------------------------------------------------------
# F8
# --------------------------------------------------------------------------
def test_owner_wave_ref_is_host_issued_and_unforgeable(w2c_harness):
    instance, server, boundary, engine, waves = w2c_harness("B")
    tool_names = " ".join(server.tool_names()).lower()
    for forbidden in ("decision", "answer", "wave", "owner"):
        assert forbidden not in tool_names

    # red controls: a participant-minted wave is refused
    forged = boundary.propose({"phase": "genesis", "move": "plan-change", "source": GENESIS_SOURCE, "owner_wave_ref": "wave-minted-by-participant"})
    assert forged["status"] == "refused"
    assert forged["code"] == "owner-wave-refused"

    frame = boundary.propose({"phase": "frame", "move": "scene-draft", "writes": WRITES})
    request = frame["pending_owner_wave"]
    realize_wave = waves.issue(request_id=request, decision="apply", revision=frame["revision"])
    realize = boundary.propose({"phase": "realize", "move": "scene-draft", "writes": WRITES, "owner_wave_ref": realize_wave, "expected_revision": frame["revision"]})
    candidate = realize["candidate_bundle"]
    payload_sha256 = ec.digest({"stories/scene-01.md": "Scene one text.\n"})
    evaluation = boundary.propose({"phase": "prove", "candidate_bundle": candidate, "evidence_kind": "proof-draft", "evidence": {"candidate_bundle": candidate, "payload_sha256": payload_sha256}})["evaluation_id"]
    resume = boundary.propose({"phase": "resume", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": evaluation})

    # an evidence pass cannot substitute for an owner answer
    no_wave = boundary.propose({"phase": "reconcile", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": evaluation, "owner_wave_ref": "wave-forged"})
    assert no_wave["status"] == "refused"

    # an owner answer cannot substitute for missing proof
    reconcile_wave = waves.issue(request_id=resume["pending_owner_wave"], decision="apply", candidate_bundle=candidate, revision=resume["revision"])
    no_proof = boundary.propose({"phase": "reconcile", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": "eval-missing", "owner_wave_ref": reconcile_wave, "expected_revision": resume["revision"]})
    assert no_proof["status"] == "refused"

    # an unknown or self-answered decision is refused
    with pytest.raises(Exception):
        waves.issue(request_id=request, decision="approve-forever")


# --------------------------------------------------------------------------
# B5
# --------------------------------------------------------------------------
def test_arm_b_completes_matched_task_through_basic_mcp(w2c_reset, w2c_stdio_client, w2c_harness, tmp_path):
    instance = w2c_reset("B")
    wave_store = tmp_path / "waves"
    host_waves = OwnerWaveAuthority(store_dir=wave_store)
    config = {
        "workspace": str(instance.destination),
        "entitlement": {"read": ["**"], "write_scope": ["stories", "characters", "plan", "project.json", "story"], "denied": []},
        "arm": "B",
        "specialized": False,
        "run_dir": str(instance.run_dir),
        "strict_mcp_config": True,
        "owner_wave_store": str(wave_store),
        "native_engine": {
            "module": "conftest",
            "attr": "CooperativeNativeEngine",
            "kwargs": {"workspace_root": str(instance.destination)},
        },
    }
    client = w2c_stdio_client(config)
    initialized = client.request("initialize", {"protocolVersion": "2024-11-05"})
    assert initialized["result"]["serverInfo"]["name"] == "story-bench-protected-mcp"
    tools = {tool["name"] for tool in client.request("tools/list")["result"]["tools"]}
    assert "workspace.propose" in tools
    assert not any(name.startswith("nc.") for name in tools), "specialized namespace must be absent for B"

    # red control (c): an always-rejecting propose channel must NOT reach accepted
    noop_instance, noop_server, noop_boundary, noop_engine, noop_waves = w2c_harness("B")

    class AlwaysRefusingBoundary:
        def __init__(self, real):
            self._real = real

        def propose(self, proposal):
            return {"status": "refused", "code": "phase-refused", "message": "always-rejecting mutant build"}

    before_noop = noop_boundary.accepted_fingerprint()
    red = _run_matched(AlwaysRefusingBoundary(noop_boundary).propose, noop_waves)
    assert red["status"] != "accepted"
    assert noop_boundary.accepted_fingerprint() == before_noop
    assert not (noop_instance.destination / "stories" / "scene-01.md").exists()

    # (a) authorized matched task through the basic MCP only
    genesis_wave = host_waves.issue(request_id="req-genesis", decision="apply", revision=0)
    genesis = client.call_tool(
        "workspace.propose",
        {"phase": "genesis", "move": "plan-change", "source": GENESIS_SOURCE, "owner_wave_ref": genesis_wave, "expected_revision": 0},
    )
    assert genesis["status"] == "accepted", genesis
    assert genesis["revision"] == 1

    frame = client.call_tool("workspace.propose", {"phase": "frame", "move": "scene-draft", "writes": WRITES, "rationale": "bounded"})
    realize_wave = host_waves.issue(request_id=frame["pending_owner_wave"], decision="apply", revision=frame["revision"])
    realize = client.call_tool(
        "workspace.propose",
        {"phase": "realize", "move": "scene-draft", "writes": WRITES, "owner_wave_ref": realize_wave, "expected_revision": frame["revision"]},
    )
    candidate = realize["candidate_bundle"]
    payload_sha256 = ec.digest({"stories/scene-01.md": "Scene one text.\n"})
    prove = client.call_tool(
        "workspace.propose",
        {"phase": "prove", "candidate_bundle": candidate, "evidence_kind": "proof-draft", "evidence": {"candidate_bundle": candidate, "payload_sha256": payload_sha256}},
    )
    evaluation = prove["evaluation_id"]
    resume = client.call_tool("workspace.propose", {"phase": "resume", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": evaluation})
    reconcile_wave = host_waves.issue(request_id=resume["pending_owner_wave"], decision="apply", candidate_bundle=candidate, revision=resume["revision"])
    reconcile = client.call_tool(
        "workspace.propose",
        {"phase": "reconcile", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": evaluation, "owner_wave_ref": reconcile_wave, "expected_revision": resume["revision"]},
    )
    assert reconcile["status"] == "accepted", reconcile
    assert reconcile["revision"] == 2

    observed = client.call_tool("workspace.observe", {"revision": reconcile["revision"]})
    assert observed["revision"] == 2
    assert observed["accepted_state_fingerprint"] == reconcile["observation"]["accepted_state_fingerprint"]
    assert (instance.destination / "stories" / "scene-01.md").read_text(encoding="utf-8") == "Scene one text.\n"

    # (a) the accepted transition is persisted as a native record chain
    native = instance.destination / ".narrative"
    assert list((native / "requests").glob("*.json")), "no Decision/request record"
    assert (native / "candidates" / (candidate + ".json")).is_file(), "no candidate record"
    assert (native / "evaluations" / (evaluation + ".json")).is_file(), "no Evaluation record"
    assert list((native / "reconciles").glob("*.json")), "no reconcile record"
    assert list((native / "accepted").rglob("*.md")), "no accepted snapshot"

    # (b) separate unauthorized proposal rejected with accepted bytes preserved
    fingerprint = observed["accepted_state_fingerprint"]
    accepted_bytes = nw.census_files(instance.destination).digest
    unauthorized = client.call_tool(
        "workspace.propose",
        {"phase": "reconcile", "move": "scene-draft", "candidate_bundle": candidate, "proof_evaluation": evaluation, "owner_wave_ref": "wave-forged", "expected_revision": 2},
    )
    assert unauthorized["status"] == "refused", unauthorized
    assert unauthorized["code"] == "owner-wave-refused"
    out_of_scope = client.call_tool(
        "workspace.propose",
        {"phase": "frame", "move": "scene-draft", "writes": [{"path": "results/leak.md", "content": "x"}]},
    )
    assert out_of_scope["status"] == "refused"
    after = client.call_tool("workspace.observe", {"revision": 2})
    assert after["accepted_state_fingerprint"] == fingerprint
    assert nw.census_files(instance.destination).digest == accepted_bytes


# --------------------------------------------------------------------------
# F-05 (conditions repair 1, AC-4a): resolve_inside escape check, reached with a
# genuine host owner wave so the ONLY possible refusal is scope-refused.
# --------------------------------------------------------------------------
def test_resolve_inside_escape_refused_with_a_genuine_owner_wave(w2c_harness, tmp_path):
    """R08 kill: the shipped symlink cases never assert the exact escape refusal,
    because without a wave the mutant still returns refused (with
    owner-wave-refused). Here a genuine host owner wave answers the pending frame
    request, so on the unmodified code only the resolve_inside workspace-root/
    symlink escape check can refuse the proposal (scope-refused); on the mutant the
    proposal is accepted instead.
    """
    instance, server, boundary, engine, waves = w2c_harness("B")
    outside = tmp_path / "outside-target"
    outside.mkdir()
    (outside / "secret.md").write_text("secret\n", encoding="utf-8")
    (instance.destination / "stories").mkdir(exist_ok=True)
    (instance.destination / "stories" / "evil").symlink_to(outside, target_is_directory=True)
    try:
        (instance.destination / "stories" / "link.md").symlink_to(outside / "secret.md")
        symlinked_file = True
    except OSError:
        symlinked_file = False

    genesis_wave = waves.issue(request_id="req-genesis", decision="apply", revision=0)
    genesis = boundary.propose(
        {
            "phase": "genesis",
            "move": "plan-change",
            "source": GENESIS_SOURCE,
            "owner_wave_ref": genesis_wave,
            "expected_revision": 0,
        }
    )
    assert genesis["status"] == "accepted", genesis

    frame = boundary.propose({"phase": "frame", "move": "scene-draft", "writes": WRITES})
    assert frame["status"] == "ok", frame
    request = frame["pending_owner_wave"]
    fingerprint = boundary.accepted_fingerprint()

    escape_paths = ["stories/evil/leak.md"]
    if symlinked_file:
        escape_paths.append("stories/link.md")
    for rel in escape_paths:
        wave = waves.issue(request_id=request, decision="apply", revision=frame["revision"])
        result = boundary.propose(
            {
                "phase": "realize",
                "move": "scene-draft",
                "writes": [{"path": rel, "content": "escaped\n"}],
                "owner_wave_ref": wave,
                "expected_revision": frame["revision"],
            }
        )
        assert result["status"] == "refused", (rel, result)
        assert result["code"] == "scope-refused", (rel, result)
    assert boundary.accepted_fingerprint() == fingerprint

    # red control: the SAME wave-authorized realize succeeds for an in-scope path,
    # so the refusal above is the escape check and not a missing wave or revision.
    ok_wave = waves.issue(request_id=request, decision="apply", revision=frame["revision"])
    ok = boundary.propose(
        {
            "phase": "realize",
            "move": "scene-draft",
            "writes": WRITES,
            "owner_wave_ref": ok_wave,
            "expected_revision": frame["revision"],
        }
    )
    assert ok["status"] == "ok", ok
