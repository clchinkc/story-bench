"""W2c treatment-isolation tests (risk rows T1-T9, L1, B1-B4, B6, P1, P2; conditions-repair rows F-01, F-02)."""
from __future__ import annotations

import hashlib
import importlib
import importlib.resources
import json
import sys
from pathlib import Path

import pytest

import arm_b_manifest as abm
import arm_manifest as am
import capability_probe as cap
import environment_contract as ec
import native_binding as nb
import protected_mcp as pm
from environment_contract import ContractViolation

ARMS = ("A", "B", "C", "D")
REPO_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_MARKERS = ("KNOWLEDGE", "Ghost Reader", "ghost-reader", "Firstory", "GROUND-TRUTH", "Obsidian", "personal memory")


# --------------------------------------------------------------------------
# T1
# --------------------------------------------------------------------------
def test_basic_mcp_inventory_equal_across_arms():
    baseline = pm.basic_inventory(specialized=False)
    for arm in ARMS:
        manifest = am.resolve_arm(arm)
        inventory = pm.basic_inventory(specialized=manifest.specialized_tools)
        assert inventory["server_id"] == baseline["server_id"]
        assert inventory["server_version"] == baseline["server_version"]
        assert inventory["transport"] == baseline["transport"]
        assert inventory["latency_budget_ms"] == baseline["latency_budget_ms"]
        assert inventory["input_schemas"] == baseline["input_schemas"]
        if arm in ("A", "B"):
            assert inventory["tool_names"] == baseline["tool_names"]
            assert not any(name.startswith("nc.") for name in inventory["tool_names"])
        else:
            assert any(name.startswith("nc.") for name in inventory["tool_names"])
            assert inventory["tool_names"][: len(baseline["tool_names"])] == baseline["tool_names"]

    # red control: a differing basic tool set for one arm is detected
    mutated = pm.basic_inventory(specialized=False)
    mutated["tool_names"] = [*mutated["tool_names"], "workspace.export"]
    assert mutated != baseline


# --------------------------------------------------------------------------
# T2
# --------------------------------------------------------------------------
def test_participant_cannot_observe_arm_or_other_arm_state(w2c_harness, tmp_path):
    instance, server, boundary, engine, waves = w2c_harness("B")
    sibling = tmp_path / "dest-C"
    sibling.mkdir()
    (sibling / "secret.md").write_text("other arm state\n", encoding="utf-8")

    responses = [
        server.call_tool("workspace.list", {}),
        server.call_tool("workspace.frontier", {}),
        server.call_tool("workspace.authorities", {}),
        server.call_tool("workspace.observe", {"revision": 0}),
    ]
    for response in responses:
        text = ec.canonical_json(response)
        assert "\"arm\"" not in text
        assert str(sibling) not in text
        assert "holdout" not in text

    cross = server.call_tool("workspace.read", {"path": "../dest-C/secret.md"})
    assert cross["status"] == "refused"
    assert "other arm state" not in cross.get("text", "")

    # red control: leaking the arm into a response is detected
    leaked = dict(responses[1])
    leaked["arm"] = "B"
    assert "\"arm\"" in ec.canonical_json(leaked)


# --------------------------------------------------------------------------
# T3
# --------------------------------------------------------------------------
def test_cross_arm_reads_and_holdout_refused(w2c_harness, tmp_path):
    instance, server, boundary, engine, waves = w2c_harness("A")
    sibling = tmp_path / "dest-D"
    sibling.mkdir()
    (sibling / "state.md").write_text("sibling arm\n", encoding="utf-8")
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "labels.json").write_text('{"label":"gold"}\n', encoding="utf-8")
    grader = tmp_path / "grader"
    grader.mkdir()
    (grader / "criteria.md").write_text("private criteria\n", encoding="utf-8")

    for path in ("../dest-D/state.md", "../holdout/labels.json", "../grader/criteria.md"):
        response = server.call_tool("workspace.read", {"path": path})
        assert response["status"] == "refused", path
        assert "text" not in response

    # red control: a shared parent root is exactly the mutant that makes another
    # arm readable, while the destination root refuses it
    from authority_boundary import ScopeRefused, resolve_inside

    assert resolve_inside(tmp_path, "dest-D/state.md").is_file()
    with pytest.raises(ScopeRefused):
        resolve_inside(instance.destination, "../dest-D/state.md")


# --------------------------------------------------------------------------
# T4
# --------------------------------------------------------------------------
def test_arm_b_has_no_cli_source_shell_or_inherited_skill(w2c_harness, driver_probe_env, tmp_path):
    instance, server, boundary, engine, waves = w2c_harness("B")
    cwd = ec.participant_cwd()
    try:
        assert list(cwd.iterdir()) == []
        ec.assert_participant_cwd(cwd, source=instance.source, destination=instance.destination)
        env = ec.participant_child_env(cwd, environ={
            "PATH": "/usr/bin",
            "PYTHONPATH": "/native/src",
            "PYTHONHOME": "/native",
            "VIRTUAL_ENV": "/native/.venv",
            "NC_EVAL_ROOT": "/runs",
            "NC_EVAL_CASE": "one",
            "ANTHROPIC_API_KEY": "secret",
            "OPENAI_API_KEY": "secret",
            "HOME": "/Users/someone",
        }, pinned_bin_dir=tmp_path / "empty-bin")
        assert "PYTHONPATH" not in env and "PYTHONHOME" not in env and "VIRTUAL_ENV" not in env
        assert "NC_EVAL_ROOT" not in env and "NC_EVAL_CASE" not in env
        assert "ANTHROPIC_API_KEY" not in env and "OPENAI_API_KEY" not in env
        assert env["HOME"] == "/Users/someone"
        assert env["PWD"] == str(cwd)
        assert "narrative-craft" not in env["PATH"]
        assert not (cwd / "CLAUDE.md").exists() and not (cwd / "AGENTS.md").exists()

        report, spec = driver_probe_env(permissive=False)
        cap.assert_report_complete(report)
        assert cap.denied_channels(report) == list(cap.PROBE_CHANNELS)
    finally:
        ec.cleanup_participant_cwd(cwd)

    # red controls
    leaky = instance.destination / "leaky"
    leaky.mkdir()
    (leaky / "CLAUDE.md").write_text("# memory", encoding="utf-8")
    with pytest.raises(ec.ParticipantEnvironmentError):
        ec.assert_participant_cwd(leaky, source=instance.source, destination=instance.destination)

    native_bin = tmp_path / "native-bin"
    native_bin.mkdir()
    (native_bin / "narrative-craft").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (native_bin / "narrative-craft").chmod(0o755)
    with pytest.raises(nb.NativeBindingRefused):
        nb.assert_pinned_name_is_absent_from_path(str(native_bin))


# --------------------------------------------------------------------------
# T5
# --------------------------------------------------------------------------
def test_ambient_mcp_config_not_loaded():
    assert pm.mcp_server_inventory({"strict_mcp_config": True}) == [pm.SERVER_ID]
    assert pm.assert_only_protected_inventory({"strict_mcp_config": True}) == [pm.SERVER_ID]

    ambient = REPO_ROOT / ".mcp.json"
    if ambient.exists():
        declared = json.loads(ambient.read_text(encoding="utf-8")).get("mcpServers", {})
        assert "linear-server" in declared

    # red control: dropping strict mode loads the ambient server and is refused
    lax = pm.mcp_server_inventory({"strict_mcp_config": False})
    assert "linear-server" in lax
    with pytest.raises(ContractViolation):
        pm.assert_only_protected_inventory({"strict_mcp_config": False})


# --------------------------------------------------------------------------
# T6
# --------------------------------------------------------------------------
def test_harness_flags_match_frozen_spec(tmp_path):
    config_path = tmp_path / "protected-mcp.json"
    expected = [
        "claude", "-p", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
        "--mcp-config", str(config_path), "--no-session-persistence", "--model", am.model_id(),
    ]
    for arm in ARMS:
        assert am.harness_argv(arm, mcp_config_path=config_path, model_id=am.model_id()) == expected, arm
    argv = am.harness_argv("B", mcp_config_path=config_path, model_id=am.model_id())
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert "--no-session-persistence" in argv

    # red control: a dropped isolation flag is refused, not defaulted
    document = am.load_arm_document()
    mutant = json.loads(json.dumps(document))
    mutant["harness"]["argv_template"] = [part for part in mutant["harness"]["argv_template"] if part != "--strict-mcp-config"]
    with pytest.raises(am.ArmManifestError):
        am.validate_arm_document(mutant)


# --------------------------------------------------------------------------
# T7
# --------------------------------------------------------------------------
def test_equalisation_every_arm_same_facts_intent_memory_budget(w2c_env_inputs):
    inputs = {
        "source_facts_digest": "sf-1",
        "task_brief": "Author brief.",
        "memory_init_digest": w2c_env_inputs["memory_init_digest"],
        "information_entitlement_digest": w2c_env_inputs["information_entitlement_digest"],
        "native_binding_digest": w2c_env_inputs["native_binding_digest"],
    }
    records = {arm: am.equalisation_record(episode_inputs=inputs) for arm in ARMS}
    am.assert_equalisation(records)
    assert "arm" not in records["A"]

    # red control: a larger budget for one arm is refused
    mutant = {arm: dict(record) for arm, record in records.items()}
    mutant["B"]["max_turns"] = 400
    with pytest.raises(am.ArmManifestError):
        am.assert_equalisation(mutant)


# --------------------------------------------------------------------------
# T8
# --------------------------------------------------------------------------
def _contains_private_marker(text):
    return sorted(marker for marker in PRIVATE_MARKERS if marker in text)


def test_no_private_knowledge_in_package_or_environment(w2c_harness):
    package = abm.load_package()
    assert _contains_private_marker(package.instruction_text) == []
    excluded_kinds = {entry["kind"] for entry in package.manifest["excluded"]}
    assert "vault-overlay-skill" in excluded_kinds
    overlay = [entry for entry in package.manifest["excluded"] if entry["kind"] == "vault-overlay-skill"][0]
    assert overlay["reference"].startswith(".claude/skills/narrative-craft")

    instance, server, boundary, engine, waves = w2c_harness("B")
    mounted = []
    for path in instance.destination.rglob("*"):
        if path.is_file():
            mounted.append(path.read_text(encoding="utf-8", errors="replace"))
    assert _contains_private_marker("\n".join(mounted)) == []

    # red control: a private route bundled into the treatment is detected
    assert _contains_private_marker(package.instruction_text + "\nsee KNOWLEDGE for details\n") == ["KNOWLEDGE"]


# --------------------------------------------------------------------------
# T9
# --------------------------------------------------------------------------
def test_specialized_tools_cannot_mutate_or_leak(w2c_harness):
    instance, server, boundary, engine, waves = w2c_harness("C", specialized=True)
    authorities = server.call_tool("workspace.authorities", {})
    assert server.call_tool("nc.proof_families", {})["proof_families"] == authorities["proof_families"]
    assert server.call_tool("nc.ground", {})["models"] == authorities["models"]
    assert server.call_tool("nc.plan_candidate", {})["plan"] == authorities["models"]["plan"]

    fingerprint = boundary.accepted_fingerprint()
    assert server.call_tool("nc.proof_families", {})["proof_families"] == authorities["proof_families"]
    assert boundary.accepted_fingerprint() == fingerprint

    # a specialized call cannot mutate accepted state except through workspace.propose
    refused = server.call_tool("nc.critique_ingest", {"candidate_bundle": "cb-missing", "evidence": {}})
    assert refused["status"] == "refused"
    assert boundary.accepted_fingerprint() == fingerprint

    # red control: the accepted-state detector is sensitive to a real accepted
    # transition through the single mutation channel
    genesis_wave = waves.issue(request_id="req-genesis", decision="apply", revision=0)
    genesis = boundary.propose({"phase": "genesis", "move": "plan-change", "source": {"id": "s", "title": "T", "planSource": "p"}, "owner_wave_ref": genesis_wave, "expected_revision": 0})
    assert genesis["status"] == "accepted"
    assert boundary.accepted_fingerprint() != fingerprint


# --------------------------------------------------------------------------
# L1
# --------------------------------------------------------------------------
def test_per_arm_logging_complete_and_symmetric(w2c_harness):
    channel_sets = {}
    for arm in ARMS:
        instance, server, boundary, engine, waves = w2c_harness(arm)
        log = pm.EpisodeLog(arm)
        for tool in server.tool_names():
            log.record_tool_call(tool=tool, disclosure="full", availability="available", content_reference="ref", cost_usd=0.0)
        assert log.missing_channels() == []
        log.assert_complete()
        channel_sets[arm] = set(log.channels)

    baseline = channel_sets["A"]
    for arm in ARMS:
        assert channel_sets[arm] == baseline

    # red control: dropping one channel for one arm is detected
    log = pm.EpisodeLog("B")
    log.record_tool_call(tool="workspace.read", disclosure="full", availability="available", content_reference="ref", cost_usd=0.0)
    del log.channels["cost"]
    with pytest.raises(ContractViolation):
        log.assert_complete()


# --------------------------------------------------------------------------
# B1
# --------------------------------------------------------------------------
def test_arm_b_package_versioned_and_digest_bound():
    package = abm.load_package()
    manifest = package.manifest
    assert manifest["packageVersion"] == "arm-b-v1"
    assert manifest["deliveryChannel"] == "host-treatment-packet"
    assert manifest["instructionSha256"] == hashlib.sha256(package.instruction_text.encode("utf-8")).hexdigest()
    assert manifest["sourceSkill"]["sha256"] == nb.PINNED_INSTALLED_SKILL_SHA256
    assert manifest["sourceSkill"]["sha256"] == "3f4ddcf4245f232400485ccedf5931f59916793e62b0a15a3c16601ac7b42830"

    # red controls: unversioned and digest-mismatched packages refuse to load
    mutant = json.loads(json.dumps(manifest))
    mutant["packageVersion"] = ""
    with pytest.raises(abm.PackageRefused):
        abm.validate_manifest(mutant, instruction_text=package.instruction_text)
    mutant = json.loads(json.dumps(manifest))
    mutant["instructionSha256"] = "0" * 64
    with pytest.raises(abm.PackageRefused):
        abm.validate_manifest(mutant, instruction_text=package.instruction_text)
    mutant = json.loads(json.dumps(manifest))
    mutant["sourceSkill"]["sha256"] = "1" * 64
    with pytest.raises(abm.PackageRefused):
        abm.validate_manifest(mutant, instruction_text=package.instruction_text)


# --------------------------------------------------------------------------
# B2
# --------------------------------------------------------------------------
def test_arm_b_derivation_log_covers_every_edit():
    package = abm.load_package()
    manifest = package.manifest
    assert manifest["derivation"], "no derivation log"
    for entry in manifest["derivation"]:
        assert entry["kind"] and entry["basicOperation"] and entry["mechanismPreserved"]
        abm.validate_capability_delta(entry["capabilityDelta"], entry_id=entry["id"])
    host_services = {
        entry["capabilityDelta"]["reference"]
        for entry in manifest["derivation"]
        if entry["capabilityDelta"]["kind"] == "host-service"
    }
    assert host_services <= abm.HOST_SERVICES
    assert package.separability_verdict == abm.SEPARABLE_BY_CONTRACT
    assert package.arm_b_skill_only_claim is True

    base = package.instruction_text

    def _with_delta(delta):
        mutant = json.loads(json.dumps(manifest))
        mutant["derivation"][0]["capabilityDelta"] = delta
        return mutant

    with pytest.raises(abm.PackageRefused):
        abm.validate_manifest(_with_delta("documented"), instruction_text=base)
    with pytest.raises(abm.PackageRefused):
        abm.validate_manifest(
            _with_delta({"kind": "participant-loss", "reference": None, "reason": "lost a capability",
                         "hostServiceEqualInAllArms": True, "removesMatchedTaskStep": False, "parentAcceptanceRef": None}),
            instruction_text=base,
        )
    with pytest.raises(abm.PackageRefused) as blocked:
        abm.validate_manifest(
            _with_delta({"kind": "none", "reference": None, "reason": "removes a step",
                         "hostServiceEqualInAllArms": True, "removesMatchedTaskStep": True, "parentAcceptanceRef": None}),
            instruction_text=base,
        )
    assert abm.BLOCKED in str(blocked.value)
    with pytest.raises(abm.PackageRefused):
        abm.validate_manifest(
            _with_delta({"kind": "host-service", "reference": None, "reason": "x",
                         "hostServiceEqualInAllArms": True, "removesMatchedTaskStep": False, "parentAcceptanceRef": None}),
            instruction_text=base,
        )


# --------------------------------------------------------------------------
# B3
# --------------------------------------------------------------------------
_FENCE = chr(96) * 3
FORBIDDEN_PACKAGE_TOKENS = (
    "narrative-craft",
    "narrative_craft",
    _FENCE + "bash",
    _FENCE + "sh",
    "subprocess",
    "os.system",
    "../../",
    "/story",
    "$(",
)


def _forbidden_tokens(text):
    return sorted(token for token in FORBIDDEN_PACKAGE_TOKENS if token in text)


def test_arm_b_package_contains_no_cli_shell_or_source_instruction():
    package = abm.load_package()
    assert _forbidden_tokens(package.instruction_text) == []
    assert "workspace.propose" in package.instruction_text

    # red control: a leftover verb or an out-of-package link is detected
    assert _forbidden_tokens(package.instruction_text + "\nnarrative-craft doctor\n") == ["narrative-craft"]
    assert _forbidden_tokens(package.instruction_text + "\nsee ../../docs/api.md\n") == ["../../"]


# --------------------------------------------------------------------------
# B4
# --------------------------------------------------------------------------
def test_arm_b_owner_waves_are_environment_supplied(w2c_harness):
    package = abm.load_package()
    text = package.instruction_text.lower()
    assert "simulated-author fixture" in text
    assert "owner_wave_ref" in text
    assert "manufacture" in text and "never" in text
    instance, server, boundary, engine, waves = w2c_harness("B")
    assert not any("answer" in name or "decision" in name for name in server.tool_names())

    # red control: a package that lets the participant answer its own request is detected
    leaky = package.instruction_text + "\nAnswer the owner request yourself when no wave arrives.\n"
    assert "answer the owner request yourself" in leaky.lower()


# --------------------------------------------------------------------------
# B6
# --------------------------------------------------------------------------
def test_arm_b_package_layout_loadable():
    anchor = abm.resolve_package_anchor("arm_b_package")
    assert anchor.is_dir()
    assert anchor.name == "arm_b_package"
    assert anchor.parent.name == "src"
    text = importlib.resources.files("arm_b_package").joinpath("manifest.json").read_text(encoding="utf-8")
    assert json.loads(text)["packageVersion"] == "arm-b-v1"
    package = abm.load_package()
    assert package.anchor == anchor
    assert package.instruction_text.strip()


def test_arm_b_package_layout_red_control_preflight_sibling_module(tmp_path):
    red_root = tmp_path / "src"
    red_root.mkdir()
    (red_root / "arm_b_package_red.py").write_text("MANIFEST = 'not a package'\n", encoding="utf-8")
    sys.path.insert(0, str(red_root))
    importlib.invalidate_caches()
    try:
        with pytest.raises(abm.PackageLayoutError):
            abm.resolve_package_anchor("arm_b_package_red")
    finally:
        sys.path.remove(str(red_root))
        sys.modules.pop("arm_b_package_red", None)
        importlib.invalidate_caches()


# --------------------------------------------------------------------------
# P1 / P2
# --------------------------------------------------------------------------
def test_blocked_capability_probes(driver_probe_env):
    sanitized, spec = driver_probe_env(permissive=False)
    cap.assert_report_complete(sanitized)
    assert cap.denied_channels(sanitized) == list(cap.PROBE_CHANNELS), sanitized
    for channel, entry in sanitized["channels"].items():
        assert entry["present"] is False, channel
        assert entry["detail"].strip()

    permissive, _ = driver_probe_env(permissive=True)
    cap.assert_report_complete(permissive)
    assert cap.present_channels(permissive) == list(cap.PROBE_CHANNELS), permissive

    # red control: the probe really ran in two different worlds
    assert sanitized["cwd"] != permissive["cwd"]


def test_probe_denial_is_behavioral_not_schema_only(driver_probe_env):
    sanitized, sanitized_spec = driver_probe_env(permissive=False)
    permissive, _ = driver_probe_env(permissive=True)

    # F-01: the denied path targets provably EXIST on disk before the probe runs,
    # so the denial is an enforcement refusal, not a missing-file absence.
    denied_workspace = Path(sanitized_spec["workspace_root"])
    for channel in ("filesystem_escape", "source_inspection", "inherited_settings"):
        target = denied_workspace / sanitized_spec["targets"][channel]
        assert target.is_file(), channel
        entry = sanitized["channels"][channel]
        assert entry["outcome"] == "denied", channel
        assert entry["present"] is False, channel
        assert entry["detail"].startswith("ScopeRefused"), (channel, entry["detail"])
        assert "entitlement" in entry["detail"], (channel, entry["detail"])
        assert "FileNotFoundError" not in entry["detail"], (channel, entry["detail"])
        # the honest raw-OS observation: the bytes were physically readable
        assert entry["raw_os"] == "readable", channel
        assert permissive["channels"][channel]["outcome"] == "granted", channel
        assert permissive["channels"][channel]["detail"] == "operation succeeded", channel

    # the third world: a permitted target that does not exist is an explicit
    # "absent" outcome, carrying its FileNotFoundError and NOT counted as a
    # denial by denied_channels().
    absent, absent_spec = driver_probe_env(world="absent")
    absent_workspace = Path(absent_spec["workspace_root"])
    for channel in ("filesystem_escape", "source_inspection", "inherited_settings"):
        target = absent_workspace / absent_spec["targets"][channel]
        assert not target.exists(), channel
        entry = absent["channels"][channel]
        assert entry["outcome"] == "absent", channel
        assert entry["present"] is False, channel
        assert entry["detail"].startswith("FileNotFoundError"), (channel, entry["detail"])
        assert channel not in cap.denied_channels(absent), channel

    # real child operations: a failed open/spawn is an exception, not a config read
    assert sanitized["channels"]["cli_execution"]["detail"].startswith("FileNotFoundError")
    assert permissive["channels"]["cli_execution"]["detail"] == "operation succeeded"
    assert sanitized["channels"]["environment_secrets"]["present"] is False
    assert permissive["channels"]["environment_secrets"]["present"] is True
    assert sanitized["channels"]["inherited_settings"]["present"] is False
    assert permissive["channels"]["inherited_settings"]["present"] is True
    assert sanitized["channels"]["tool_discovery"]["present"] is False
    assert permissive["channels"]["tool_discovery"]["present"] is True


# --------------------------------------------------------------------------
# F-01 (conditions repair 1, AC-1a..d): the three-world behavioural property.
# --------------------------------------------------------------------------
PATH_PROBE_CHANNELS = ("filesystem_escape", "source_inspection", "inherited_settings")


def _three_world_property_holds(grant, deny, absent):
    """AC-1a: grant True, deny False with an enforcement refusal, absent explicit."""
    for channel in PATH_PROBE_CHANNELS:
        if grant["channels"][channel]["outcome"] != "granted":
            return False
        if deny["channels"][channel]["outcome"] != "denied":
            return False
        if "ScopeRefused" not in deny["channels"][channel]["detail"]:
            return False
        if "FileNotFoundError" in deny["channels"][channel]["detail"]:
            return False
        if absent["channels"][channel]["outcome"] != "absent":
            return False
        if channel in cap.denied_channels(absent):
            return False
    return True


def test_probe_path_denial_three_world_property(driver_probe_env):
    grant, grant_spec = driver_probe_env(world="grant")
    deny, deny_spec = driver_probe_env(world="deny")
    absent, _ = driver_probe_env(world="absent")

    # AC-1b: the W_DENY targets exist on disk before the probe runs
    work = Path(deny_spec["workspace_root"])
    for channel in PATH_PROBE_CHANNELS:
        assert (work / deny_spec["targets"][channel]).is_file(), channel
    # AC-1c: W_DENY and W_GRANT differ ONLY in the entitlement
    core = ("workspace_root", "targets", "seam", "seam_paths")
    assert {key: grant_spec[key] for key in core} == {key: deny_spec[key] for key in core}
    assert grant_spec["entitlement"] != deny_spec["entitlement"]
    assert grant_spec["entitlement"] == cap.WORLD_ENTITLEMENTS["grant"]
    assert deny_spec["entitlement"] == cap.WORLD_ENTITLEMENTS["deny"]

    assert _three_world_property_holds(grant, deny, absent) is True


def test_probe_path_denial_is_falsifiable_with_stub_seams(driver_probe_env, tmp_path):
    """AC-1d: an always-deny stub and an accept-all stub EACH fail AC-1a."""
    stub_dir = tmp_path / "stub-seams"
    stub_dir.mkdir()
    (stub_dir / "always_deny_seam.py").write_text(
        "class SeamRefused(Exception):\n"
        "    pass\n"
        "\n"
        "\n"
        "def guard_read(workspace_root, rel, entitlement):\n"
        "    raise SeamRefused('stub refuses every path')\n",
        encoding="utf-8",
    )
    (stub_dir / "accept_all_seam.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def guard_read(workspace_root, rel, entitlement):\n"
        "    return Path(workspace_root) / rel\n",
        encoding="utf-8",
    )
    paths = [str(stub_dir)]

    grant_real, _ = driver_probe_env(world="grant")
    deny_real, _ = driver_probe_env(world="deny")
    absent, _ = driver_probe_env(world="absent")
    grant_deny_stub, _ = driver_probe_env(world="grant", seam="always_deny_seam:guard_read", seam_paths=paths)
    deny_accept_stub, _ = driver_probe_env(world="deny", seam="accept_all_seam:guard_read", seam_paths=paths)

    assert _three_world_property_holds(grant_real, deny_real, absent) is True
    # an always-deny seam cannot observe the W_GRANT grant
    assert _three_world_property_holds(grant_deny_stub, deny_real, absent) is False
    # an accept-all seam cannot observe the W_DENY entitlement refusal
    assert _three_world_property_holds(grant_real, deny_accept_stub, absent) is False


# --------------------------------------------------------------------------
# F-02 (conditions repair 1, AC-2c): the probe secret set is independent.
# --------------------------------------------------------------------------
def test_probe_secret_set_is_independently_derived(driver_probe_env):
    legacy = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"}
    f02 = {
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_PROFILE", "AWS_DEFAULT_REGION", "GITHUB_TOKEN", "GH_TOKEN",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "NC_EVAL_ROOT",
        "nc_eval_root",
    }
    assert isinstance(cap.DEFAULT_SECRET_KEYS, tuple)
    assert legacy < set(cap.DEFAULT_SECRET_KEYS)  # a strict superset of the old set
    assert {name.upper() for name in f02} <= {key.upper() for key in cap.DEFAULT_SECRET_KEYS}
    # The probe is NOT derived from the strip policy: it names a credential the
    # strip policy does not remove, so a strip-list gap stays observable instead
    # of the probe confirming the strip list with its own key set.
    assert not ec.is_credential_key("W2C_PROBE_SENTINEL_CREDENTIAL")
    assert "W2C_PROBE_SENTINEL_CREDENTIAL" in cap.DEFAULT_SECRET_KEYS

    # A world that leaks an AWS/GitHub credential - exactly the pre-repair F-02
    # leak the pre-repair strip list did not remove - is reported present=True.
    _, spec = driver_probe_env(permissive=False)
    leaky = {
        "PATH": "",
        "AWS_ACCESS_KEY_ID": "AKIA-leaked",
        "AWS_SECRET_ACCESS_KEY": "leaked",
        "GITHUB_TOKEN": "ghp_leaked",
        "nc_eval_root": "/runs/leaked",
        "W2C_PROBE_SENTINEL_CREDENTIAL": "sentinel-leaked",
    }
    cwd = ec.participant_cwd()
    try:
        leaked = cap.run_probe(spec=spec, cwd=cwd, env=leaky)
    finally:
        ec.cleanup_participant_cwd(cwd)
    assert leaked["channels"]["environment_secrets"]["present"] is True
    detail = leaked["channels"]["environment_secrets"]["detail"]
    for name in ("AWS_ACCESS_KEY_ID", "GITHUB_TOKEN", "nc_eval_root", "W2C_PROBE_SENTINEL_CREDENTIAL"):
        assert name in detail
