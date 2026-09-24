"""W2c environment-contract tests (risk rows E1-E6, PS1, N1, R1-R5, P3).

Every behavioral test carries its own deliberately-wrong build (red control)
and asserts that the wrong build is refused or detected, then asserts the
correct build succeeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import environment_contract as ec
import native_binding as nb
import native_workspace as nw
from native_workspace import ResetRefused, SnapshotRefused

ARMS = ("A", "B", "C", "D")


def _changed(member, value):
    if member == "arm":
        return "C" if value != "C" else "A"
    if value is None:
        return "changed"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    return str(value) + "-changed"


# --------------------------------------------------------------------------
# E1
# --------------------------------------------------------------------------
def test_reset_creates_fresh_disposable_workspace(w2c_reset, w2c_source_snapshot):
    before = nw.census_files(w2c_source_snapshot)
    instance = w2c_reset("B")
    assert instance.destination.exists()
    assert instance.destination.resolve() != w2c_source_snapshot.resolve()
    assert w2c_source_snapshot.resolve() not in instance.destination.resolve().parents
    after = nw.census_files(w2c_source_snapshot)
    assert after.digest == before.digest
    assert instance.destination_census_digest == before.digest

    # red control: a reset that adopts an occupied destination is refused
    with pytest.raises(ResetRefused):
        w2c_reset("B", destination=instance.destination)
    # red control: a destination inside the source snapshot is refused
    with pytest.raises(ResetRefused):
        w2c_reset("B", destination=w2c_source_snapshot / "nested-destination")
    assert nw.census_files(w2c_source_snapshot).digest == before.digest


# --------------------------------------------------------------------------
# E2
# --------------------------------------------------------------------------
def test_reset_clears_conversation_memory_cache(w2c_reset):
    instance = w2c_reset("B")
    state = instance.empty_state()
    assert state["conversation_length"] == 0
    assert state["memory_entries"] == []
    assert state["cache_entries"] == []

    # red control: a reset that inherited a transcript, memory or cache is detected
    instance.conversation_path.write_text(ec.canonical_json({"messages": [{"role": "user"}]}), encoding="utf-8")
    (instance.memory_dir / "episode.json").write_text("{}", encoding="utf-8")
    (instance.cache_dir / "prompt.json").write_text("{}", encoding="utf-8")
    dirty = instance.empty_state()
    assert dirty["conversation_length"] == 1
    assert dirty["memory_entries"] == ["episode.json"]
    assert dirty["cache_entries"] == ["prompt.json"]
    # restore so teardown and other assertions see the clean contract
    instance.conversation_path.write_text(ec.canonical_json({"messages": []}), encoding="utf-8")
    (instance.memory_dir / "episode.json").unlink()
    (instance.cache_dir / "prompt.json").unlink()
    assert instance.empty_state()["conversation_length"] == 0


# --------------------------------------------------------------------------
# E3
# --------------------------------------------------------------------------
def test_reset_issues_unique_episode_credentials(w2c_reset):
    first = w2c_reset("B")
    second = w2c_reset("C")
    assert first.credentials.credential_id != second.credentials.credential_id
    assert first.credentials.episode_id == first.episode_id
    credentials = json.loads((first.run_dir / "credentials.json").read_text(encoding="utf-8"))
    assert credentials["episode_id"] == first.episode_id
    assert credentials["credential_id"] == first.credentials.credential_id
    assert credentials["revoked"] is False

    # red control: a reused/global credential is detected by inequality
    reused = first.credentials.credential_id
    assert reused != second.credentials.credential_id

    # teardown revokes and removes the credential
    nw.teardown_workspace(first)
    assert not first.destination.exists()
    assert not first.run_dir.exists()
    assert not (first.run_dir / "credentials.json").exists()


# --------------------------------------------------------------------------
# E4
# --------------------------------------------------------------------------
def test_environment_id_deterministic_complete_and_instance_fresh(w2c_reset):
    instance = w2c_reset("B")
    recorded = json.loads(instance.environment_path.read_text(encoding="utf-8"))
    assert set(recorded["record"]) == set(ec.ENVIRONMENT_ID_MEMBERS)
    assert len(recorded["record"]) == 16
    assert ec.verify_environment_id(recorded["record"], recorded["environment_id"]) == instance.environment_id

    for member in ec.ENVIRONMENT_ID_MEMBERS:
        mutated = dict(recorded["record"])
        mutated[member] = _changed(member, mutated[member])
        assert ec.compute_environment_id(mutated) != instance.environment_id, member

    # red controls: 15-member and 17-member records are refused
    fifteen = dict(recorded["record"])
    fifteen.pop("native_binding_digest")
    with pytest.raises(ec.EnvironmentIdError):
        ec.build_environment_id_record(fifteen)
    seventeen = dict(recorded["record"])
    seventeen["trajectory_digest"] = "not-a-member"
    with pytest.raises(ec.EnvironmentIdError):
        ec.build_environment_id_record(seventeen)
    with pytest.raises(ec.EnvironmentIdError):
        ec.build_environment_id_record({**recorded["record"], "task_id": None})

    # freshness: two successive resets of the same task differ
    other = w2c_reset("B")
    assert other.environment_id != instance.environment_id
    assert other.workspace_instance_id != instance.workspace_instance_id


# --------------------------------------------------------------------------
# E5
# --------------------------------------------------------------------------
def test_reset_census_matches_source_snapshot_for_every_arm(w2c_reset, w2c_source_snapshot):
    source_census = nw.census_files(w2c_source_snapshot)
    for arm in ARMS:
        instance = w2c_reset(arm)
        destination_census = nw.census_files(instance.destination)
        assert destination_census.paths() == source_census.paths(), arm
        for rel in source_census.paths():
            assert destination_census.entries[rel] == source_census.entries[rel], (arm, rel)
        # the treatment packet and harness argv live outside the censused root
        assert ".narrative" not in destination_census.paths()
        assert not (instance.destination / "arm_b_package").exists()

    # red control: an unauthorized overlay inside the destination is detected
    instance = w2c_reset("A")
    overlay = instance.destination / "treatment-packet.txt"
    overlay.write_text("packet", encoding="utf-8")
    assert nw.census_files(instance.destination).paths() != source_census.paths()
    overlay.unlink()
    assert nw.census_files(instance.destination).paths() == source_census.paths()


# --------------------------------------------------------------------------
# E6
# --------------------------------------------------------------------------
def test_teardown_removes_only_own_destination(w2c_reset, w2c_source_snapshot):
    first = w2c_reset("A")
    second = w2c_reset("B")
    source_before = nw.census_files(w2c_source_snapshot).digest
    nw.teardown_workspace(first)
    assert not first.destination.exists()
    assert second.destination.exists()
    assert nw.census_files(second.destination).paths()
    assert nw.census_files(w2c_source_snapshot).digest == source_before


# --------------------------------------------------------------------------
# PS1
# --------------------------------------------------------------------------
def test_project_setting_source_census_is_empty_and_refused(w2c_reset, w2c_source_snapshot, tmp_path):
    for arm in ARMS:
        instance = w2c_reset(arm)
        assert nw.project_setting_source_census(instance.destination) == []

    forbidden = [
        ".claude",
        ".claude/settings.json",
        ".claude/settings.local.json",
        ".claude/hooks/hook.py",
        ".claude/skills/narrative-craft/SKILL.md",
        ".claude/commands/deploy.md",
        ".claude/README",
        ".mcp.json",
    ]
    for rel in forbidden:
        source = tmp_path / ("bad-snapshot-" + rel.replace("/", "_"))
        source.mkdir()
        (source / "plan").mkdir()
        (source / "plan" / "source.md").write_text("plan", encoding="utf-8")
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel == ".claude":
            target = source / ".claude"
            target.mkdir(exist_ok=True)
        else:
            target.write_text("{}", encoding="utf-8")
        with pytest.raises(SnapshotRefused):
            nw.assert_snapshot_admissible(source)
        with pytest.raises(SnapshotRefused):
            w2c_reset("B", source=source, destination=tmp_path / ("bad-dest-" + rel.replace("/", "_")))

    # participant cwd: fresh, empty and outside source/destination
    instance = w2c_reset("B")
    cwd = ec.participant_cwd()
    try:
        assert cwd.is_dir()
        assert list(cwd.iterdir()) == []
        ec.assert_participant_cwd(cwd, source=instance.source, destination=instance.destination)
        # red control: a destination-internal cwd carrying inherited memory is refused
        leak = instance.destination / "leaky-cwd"
        leak.mkdir()
        (leak / "CLAUDE.md").write_text("# memory", encoding="utf-8")
        with pytest.raises(ec.ParticipantEnvironmentError):
            ec.assert_participant_cwd(leak, source=instance.source, destination=instance.destination)
    finally:
        ec.cleanup_participant_cwd(cwd)

    source_copy = tmp_path / "source-with-mcp"
    source_copy.mkdir()
    (source_copy / "plan").mkdir()
    (source_copy / "plan" / "source.md").write_text("plan", encoding="utf-8")
    (source_copy / ".mcp.json").write_text('{"mcpServers":{}}', encoding="utf-8")
    with pytest.raises(SnapshotRefused):
        nw.assert_snapshot_admissible(source_copy)


# --------------------------------------------------------------------------
# N1
# --------------------------------------------------------------------------
def test_native_binding_pinned_and_implicit_discovery_refused(tmp_path):
    binding = nb.resolve_native_binding(environ={"PATH": ""})
    assert binding.commit == nb.PINNED_COMMIT
    assert binding.wheel_sha256 == nb.PINNED_WHEEL_SHA256
    assert binding.sdist_sha256 == nb.PINNED_SDIST_SHA256
    assert binding.digest == nb.native_binding_digest()
    assert nb.native_binding_digest() == ec.digest(nb.PINNED_BINDING_RECORD)

    record = {
        "schema_version": 1,
        "task_id": "t",
        "task_family_id": "f",
        "episode_id": "e",
        "arm": "B",
        "workspace_instance_id": "w",
        "initial_state_fingerprint": "i",
        "source_snapshot_digest": "s",
        "skill_package_digest": None,
        "tool_manifest_digest": "tm",
        "native_binding_digest": binding.digest,
        "model_snapshot": "m",
        "harness_version": "h",
        "protocol_version": "p",
        "information_entitlement_digest": "ie",
        "memory_init_digest": "mi",
    }
    assert ec.build_environment_id_record(record)["native_binding_digest"] == binding.digest

    # red control: PATH discovery of an unpinned native executable is refused
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / "narrative-craft").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "narrative-craft").chmod(0o755)
    with pytest.raises(nb.NativeBindingRefused):
        nb.resolve_native_binding(environ={"PATH": str(fake_bin)})

    # red control: PYTHONPATH would decide the native module
    fake_pysrc = tmp_path / "pysrc"
    (fake_pysrc / "narrative_craft").mkdir(parents=True)
    (fake_pysrc / "narrative_craft" / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(nb.NativeBindingRefused):
        nb.resolve_native_binding(environ={"PATH": "", "PYTHONPATH": str(fake_pysrc)})

    # red control: a mismatched artifact digest is refused
    with pytest.raises(nb.NativeBindingRefused):
        nb.resolve_native_binding(environ={"PATH": ""}, artifacts={"wheel_sha256": "0" * 64})


# --------------------------------------------------------------------------
# R1
# --------------------------------------------------------------------------
def _episode_payload(record):
    observations = [
        {"sequence": 1, "phase": "frame", "status": "ok", "revision": 1},
        {"sequence": 2, "phase": "reconcile", "status": "accepted", "revision": 2, "stop_reason": "completed"},
    ]
    return observations


def test_environment_replay_reproduces_observations(w2c_env_inputs):
    record = ec.build_environment_id_record(
        {**w2c_env_inputs, "schema_version": 1, "episode_id": "ep-1", "arm": "B",
         "workspace_instance_id": "w-1", "skill_package_digest": None}
    )
    environment_id = ec.compute_environment_id(record)
    registry = ec.EpisodeRegistry()
    observations = _episode_payload(record)
    registered = registry.register(
        episode_id="ep-1", environment_id=environment_id, environment_record=record, observations=observations
    )
    replayed = ec.replay_observations(registered, environment_record=record, observations=observations)
    assert list(replayed) == observations

    # red control: a changed member refuses replay
    stale = dict(record)
    stale["model_snapshot"] = "different"
    with pytest.raises(ec.StaleIdentityError):
        ec.replay_observations(registered, environment_record=stale)
    # red control: a changed trajectory refuses replay rather than rerunning
    with pytest.raises(ec.ReplayMismatch):
        ec.replay_observations(registered, observations=[{"sequence": 1, "phase": "frame", "status": "ok", "revision": 1}])


# --------------------------------------------------------------------------
# R2
# --------------------------------------------------------------------------
def test_duplicate_episode_identity_is_idempotent(w2c_env_inputs):
    record = ec.build_environment_id_record(
        {**w2c_env_inputs, "schema_version": 1, "episode_id": "ep-2", "arm": "C",
         "workspace_instance_id": "w-2", "skill_package_digest": None}
    )
    environment_id = ec.compute_environment_id(record)
    registry = ec.EpisodeRegistry()
    observations = _episode_payload(record)
    first = registry.register(episode_id="ep-2", environment_id=environment_id, environment_record=record, observations=observations)
    second = registry.register(episode_id="ep-2", environment_id=environment_id, environment_record=record, observations=observations)
    assert len(registry) == 1
    assert first.content_digest == second.content_digest

    # red control: different content under the same episode id is refused and
    # the existing record and accepted bytes are preserved
    with pytest.raises(ec.ReplayMismatch):
        registry.register(
            episode_id="ep-2",
            environment_id=environment_id,
            environment_record=record,
            observations=[*observations, {"sequence": 3, "phase": "author-source", "status": "ok", "revision": 2}],
        )
    assert len(registry) == 1
    assert registry.get("ep-2").content_digest == first.content_digest


# --------------------------------------------------------------------------
# R3
# --------------------------------------------------------------------------
def test_replayed_episode_with_changed_content_refused(w2c_env_inputs):
    record = ec.build_environment_id_record(
        {**w2c_env_inputs, "schema_version": 1, "episode_id": "ep-3", "arm": "D",
         "workspace_instance_id": "w-3", "skill_package_digest": "pkg"}
    )
    environment_id = ec.compute_environment_id(record)
    registry = ec.EpisodeRegistry()
    observations = _episode_payload(record)
    registered = registry.register(episode_id="ep-3", environment_id=environment_id, environment_record=record, observations=observations)
    with pytest.raises(ec.ReplayMismatch):
        registry.register(
            episode_id="ep-3", environment_id=environment_id, environment_record=record,
            observations=[{"sequence": 1, "phase": "frame", "status": "refused", "revision": 1}],
        )
    assert registry.get("ep-3").content_digest == registered.content_digest


# --------------------------------------------------------------------------
# R4
# --------------------------------------------------------------------------
def test_stale_task_protocol_snapshot_or_binding_refused(w2c_env_inputs):
    record = ec.build_environment_id_record(
        {**w2c_env_inputs, "schema_version": 1, "episode_id": "ep-4", "arm": "A",
         "workspace_instance_id": "w-4", "skill_package_digest": None}
    )
    frozen = ec.EpisodeRegistry().register(
        episode_id="ep-4", environment_id=ec.compute_environment_id(record), environment_record=record,
        observations=_episode_payload(record),
    )
    for member in ("task_id", "protocol_version", "source_snapshot_digest", "arm", "model_snapshot",
                   "native_binding_digest", "episode_id", "workspace_instance_id"):
        changed = dict(record)
        changed[member] = _changed(member, changed[member])
        with pytest.raises(ec.StaleIdentityError):
            ec.replay_observations(frozen, environment_record=changed)
    # an unchanged record replays
    assert ec.replay_observations(frozen, environment_record=record)


# --------------------------------------------------------------------------
# R5
# --------------------------------------------------------------------------
def test_truncation_and_stop_reason_preserved(w2c_env_inputs):
    record = ec.build_environment_id_record(
        {**w2c_env_inputs, "schema_version": 1, "episode_id": "ep-5", "arm": "B",
         "workspace_instance_id": "w-5", "skill_package_digest": None}
    )
    truncated_observations = [
        {"sequence": 1, "phase": "realize", "status": "ok", "revision": 1},
        {"sequence": 2, "phase": "advance", "status": "ok", "revision": 1, "truncated": True, "text": "The scene ends mid-"},
    ]
    registry = ec.EpisodeRegistry()
    frozen = registry.register(
        episode_id="ep-5", environment_id=ec.compute_environment_id(record), environment_record=record,
        observations=truncated_observations, stop_reason="truncated",
    )
    replayed = ec.replay_observations(frozen)
    assert ec.truncation_stop_reason(replayed) == "truncated"
    assert frozen.stop_reason == "truncated"

    # red control: dropping the truncation fact would make it look complete
    promoted = [{key: value for key, value in obs.items() if key != "truncated"} for obs in truncated_observations]
    assert ec.truncation_stop_reason(promoted) == "completed"
    assert ec.truncation_stop_reason(replayed) == "truncated"


# --------------------------------------------------------------------------
# P3
# --------------------------------------------------------------------------
def _honest_contract(text):
    required = [
        "Protected by W2c",
        "Constructed or asserted",
        "Planned and not tested",
        "Assumed about the host",
        "NOT a kernel sandbox",
        "UNRESOLVED tier-M release gate G1",
        "do not authenticate a caller",
    ]
    missing = [phrase for phrase in required if phrase not in text]
    overclaims = ["kernel sandbox enforced", "MCP survival verified", "universal safety guaranteed"]
    claimed = [phrase for phrase in overclaims if phrase in text]
    return not missing and not claimed


def test_contract_records_host_assumptions():
    doc = (Path(__file__).resolve().parent.parent / "docs" / "environment-contract.md").read_text(encoding="utf-8")
    assert _honest_contract(doc)
    # red controls: an OS-isolation over-claim and a behaviorally-verified claim fail
    assert not _honest_contract(doc + "\nThe environment is a kernel sandbox enforced for every process.\n")
    assert not _honest_contract(doc + "\nMCP survival verified under tools-empty.\n")
    assert not _honest_contract(doc.replace("Assumed about the host", "Facts"))


# --------------------------------------------------------------------------
# F-02 (conditions repair 1, AC-2a..d): the closed participant credential policy.
# --------------------------------------------------------------------------
F02_CREDENTIAL_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "NC_EVAL_ROOT",
    "NC_EVAL_CASE",
    "nc_eval_root",
    "nc_Eval_Mixed",
    "aws_access_key_id",
    "Github_Token",
)


def test_participant_child_env_removes_the_closed_credential_policy(tmp_path):
    environ = {name: "leaked-" + name for name in F02_CREDENTIAL_NAMES}
    environ["HOME"] = "/Users/someone"
    environ["SAFE_UNRELATED"] = "kept"
    cwd = tmp_path / "participant-cwd"
    cwd.mkdir()
    env = ec.participant_child_env(cwd, environ=environ, pinned_bin_dir=tmp_path / "empty-bin")
    for name in F02_CREDENTIAL_NAMES:
        assert name not in env, name
    assert env["HOME"] == "/Users/someone"
    assert env["PWD"] == str(cwd)
    assert env["SAFE_UNRELATED"] == "kept"

    # the policy is a documented closed LAW, not an ad-hoc partial list
    for prefix in ("AWS_", "AZURE_OPENAI_", "GITHUB_", "GH_", "NC_EVAL_"):
        assert prefix in ec.CREDENTIAL_KEY_PREFIXES
    for name in F02_CREDENTIAL_NAMES:
        assert ec.is_credential_key(name), name
    assert ec.is_credential_key("NC_EVAL_ANYTHING_SUFFIXED")
    assert not ec.is_credential_key("SAFE_UNRELATED")


# --------------------------------------------------------------------------
# F-04 (conditions repair 1, AC-3a): the destination census equality guard.
# --------------------------------------------------------------------------
def test_reset_refuses_a_destination_census_with_an_extra_entry(w2c_reset, tmp_path, monkeypatch):
    """R07 kill: the per-entry loop only walks source paths, so only the
    destination_census.paths() == source_census.paths() guard can see an EXTRA
    destination entry. A copytree that appends one unauthorized overlay file must
    be refused instead of silently producing a non-byte-equal destination.
    """
    real_copytree = nw.shutil.copytree

    def copytree_with_overlay(source, destination, *args, **kwargs):
        result = real_copytree(source, destination, *args, **kwargs)
        if not args:  # the top-level reset call; recursive calls pass positionals
            (Path(destination) / "unauthorized-overlay.txt").write_text("tampered\n", encoding="utf-8")
        return result

    monkeypatch.setattr(nw.shutil, "copytree", copytree_with_overlay)
    destination = tmp_path / "dest-with-overlay"
    with pytest.raises(ResetRefused) as refused:
        w2c_reset("B", destination=destination)
    assert "census" in str(refused.value)
    assert not destination.exists()

    # red control: restoring the real copytree makes the same reset succeed
    monkeypatch.setattr(nw.shutil, "copytree", real_copytree)
    clean = w2c_reset("B", destination=tmp_path / "dest-clean")
    assert clean.destination.exists()
    assert nw.census_files(clean.destination).paths() == nw.census_files(clean.source).paths()


# --------------------------------------------------------------------------
# F-06 (conditions repair 1, AC-5a..b): teardown refuses a foreign destination.
# --------------------------------------------------------------------------
def test_teardown_refuses_a_destination_that_is_not_this_instance(w2c_reset, tmp_path):
    instance = w2c_reset("B")
    original = instance.destination
    bystander = tmp_path / "innocent-bystander"
    bystander.mkdir()
    (bystander / "keep.md").write_text("keep\n", encoding="utf-8")
    instance.destination = bystander
    try:
        with pytest.raises(nw.TeardownRefused):
            nw.teardown_workspace(instance)
        assert (bystander / "keep.md").is_file()
        assert original.exists()
    finally:
        instance.destination = original

    # the normal path still removes exactly the recorded destination
    nw.teardown_workspace(instance)
    assert not original.exists()
    assert not instance.run_dir.exists()
    assert (bystander / "keep.md").is_file()
