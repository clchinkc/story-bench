"""Offline-only suite: never read credentials or contact a provider."""
import socket
import dotenv
import json
import pytest
from measurement_contract import PROTOCOL, digest, context_packet

dotenv.load_dotenv = lambda *args, **kwargs: False


def pytest_sessionstart(session):
    def deny_network(*args, **kwargs):
        raise AssertionError("Network is forbidden in offline regression tests")
    socket.socket.connect = deny_network


@pytest.fixture
def fixture_records():
    """Named independent outcomes; no network, native outcomes or human labels."""
    def make(models=("A", "B"), tasks=("easy", "hard"), samples=(0,)):
        assignments, generations, evaluations = [], [], []
        for model in models:
            for task_id in tasks:
                for sample in samples:
                    task = {"task_id": task_id, "task_type": "theory_conversion", "subtype": "default", "source": "EARLY CAUSE\nEND ‘quote’"}
                    assigned = dict(model=model, task_id=task_id, sample=sample, condition="plain", story_id="one-story", task=task, task_hash=digest(task), model_snapshot=model + "@frozen", judge_snapshot="fake-judge@1", prompt_version="offline-fixture-v1")
                    assignments.append(assigned)
                    ident = f"{model}-{task_id}-{sample}"
                    parts = {"task": json.dumps(task, sort_keys=True, ensure_ascii=False)}
                    prompt, context = context_packet(parts, max_bytes=10000)
                    generation = dict(**{k: assigned[k] for k in ("model", "task_id", "sample", "condition", "task_hash", "model_snapshot")}, protocol=PROTOCOL, record_id="gen-" + ident, status="completed", timestamp="2026-09-21T00:00:00Z", output="Complete end.", output_hash=digest("Complete end."), prompt=prompt, prompt_hash=digest(prompt), context_parts=parts, context=context, finish_reason="stop", attempts=[dict(attempt_id="g-" + ident, role="generation", status="completed", cost_usd=2., usage={"prompt_tokens": 20, "completion_tokens": 5, "reasoning_tokens": 2})])
                    generations.append(generation)
                    generation["prompt_version"] = assigned["prompt_version"]
                    parts = {**parts, "output": generation["output"]}
                    prompt, context = context_packet(parts, max_bytes=10000)
                    # Independently set A .75/.25 and B .5/.5 => both means .5.
                    score = (.75 if task_id == "easy" else .25) if model == "A" else .5
                    evaluation = dict(**{k: assigned[k] for k in ("model", "task_id", "sample", "condition", "task_hash", "model_snapshot", "judge_snapshot")}, protocol=PROTOCOL, record_id="eval-" + ident, status="completed", timestamp="2026-09-21T00:00:00Z", generation_hash=digest(generation), prompt=prompt, prompt_hash=digest(prompt), context_parts=parts, context=context, finish_reason="stop", llm_results={k: score for k in ["beats_score", "preservation_score", "structural_accuracy_score", "tone_score"]}, attempts=[dict(attempt_id="e-" + ident, role="research_evaluation", status="completed", cost_usd=.25, usage=None)])
                    evaluations.append(evaluation)
                    evaluation["prompt_version"] = assigned["prompt_version"]
        manifest = {"protocol": PROTOCOL, "assignments": assignments}
        data = {"protocol": PROTOCOL, "assignment_hash": digest(manifest), "generations": generations, "evaluations": evaluations}
        return manifest, data
    return make

# ==========================================================================
# W2c additive fixtures (unit W2c). The existing offline denial and
# fixture_records above are preserved unchanged.
# ==========================================================================
import hashlib  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

import capability_probe as cap  # noqa: E402
import environment_contract as ec  # noqa: E402
import native_binding as nb  # noqa: E402
import native_workspace as nw  # noqa: E402
from authority_boundary import NativeStepRefused  # noqa: E402
from authority_boundary import Entitlement, OwnerWaveAuthority  # noqa: E402
from protected_mcp import EpisodeLog, McpConfig, ProtectedMcpServer  # noqa: E402

W2C_ROOT = Path(__file__).resolve().parent.parent
W2C_SRC = W2C_ROOT / "src"
W2C_TESTS = Path(__file__).resolve().parent
ARM_B_INSTRUCTION = W2C_SRC / "arm_b_package" / "v1" / "SKILL.md"

DEFAULT_ENTITLEMENT = {
    "read": ["**"],
    "write_scope": ["stories", "characters", "plan", "project.json", "story"],
    "denied": [],
}


class CooperativeNativeEngine:
    """Deterministic cooperative local harness for offline environment tests.

    It is a test double for the pinned native public surface: it persists
    request, candidate, evaluation and reconcile records under .narrative/ and
    advances an accepted revision only on an authorized reconcile. It never
    calls a model or the network.
    """

    PROOF_FAMILIES = ("structure", "voice", "continuity", "semantic", "format")

    def __init__(self, workspace_root):
        self.root = Path(workspace_root)
        self.control = self.root / ".narrative"
        for name in ("requests", "candidates", "evaluations", "reconciles", "packets", "accepted"):
            (self.control / name).mkdir(parents=True, exist_ok=True)
        self._state_path = self.control / "state.json"
        self._lock = threading.RLock()
        if not self._state_path.exists():
            self._save(self._initial_state())

    def _initial_state(self):
        return {
            "revision": 0,
            "accepted_files": {},
            "pending": {},
            "proof_status": "none",
            "last_rejection": None,
            "changed_paths": [],
            "candidate_bundle": None,
            "phase": "ground",
            "decisions": [],
            "evaluations": [],
        }

    def _load(self):
        return json.loads(self._state_path.read_text(encoding="utf-8"))

    def _save(self, state=None):
        if state is not None:
            self._state = state
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(ec.canonical_json(self._state), encoding="utf-8")
        temporary.replace(self._state_path)

    def _journal(self, record):
        with (self.control / "journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(ec.canonical_json(record) + "\n")

    def _fingerprint(self, state):
        return ec.digest({"revision": state["revision"], "accepted_files": state["accepted_files"]})

    def revision(self):
        return self._load()["revision"]

    def accepted_fingerprint(self):
        return self._fingerprint(self._load())

    def _write_record(self, kind, record_id, payload):
        path = self.control / kind / (record_id + ".json")
        path.write_text(ec.canonical_json(payload), encoding="utf-8")
        return path

    def _read_record(self, kind, record_id):
        path = self.control / kind / (record_id + ".json")
        if not path.exists():
            raise NativeStepRefused(f"unknown native {kind} record {record_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _snapshot(self, state, rel, text):
        target = self.control / "accepted" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        state["accepted_files"][rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # -- native steps --------------------------------------------------
    def genesis(self, *, source, owner_wave_ref, wave=None, actor="participant"):
        if wave is None:
            raise NativeStepRefused("genesis requires a host-issued owner wave")
        state = self._load()
        if state["revision"] != 0:
            raise NativeStepRefused("genesis has already accepted an initial state")
        (self.root / "project.json").write_text(
            ec.canonical_json({"storyId": source["id"], "title": source["title"], "planSource": source["planSource"]}),
            encoding="utf-8",
        )
        (self.root / "plan").mkdir(exist_ok=True)
        (self.root / "plan" / "model.json").write_text(
            ec.canonical_json({"kind": "plan", "revision": 1, "source": source["planSource"]}), encoding="utf-8"
        )
        (self.root / "story").mkdir(exist_ok=True)
        (self.root / "story" / "model.json").write_text(
            ec.canonical_json({"kind": "story", "revision": 1, "scenes": []}), encoding="utf-8"
        )
        for rel in ("project.json", "plan/model.json", "story/model.json"):
            self._snapshot(state, rel, (self.root / rel).read_text(encoding="utf-8"))
        state["revision"] = 1
        state["phase"] = "advance"
        state["decisions"].append({"kind": "genesis", "owner_wave_ref": owner_wave_ref, "actor": actor})
        self._save(state)
        self._journal({"step": "genesis", "revision": state["revision"], "owner_wave_ref": owner_wave_ref})
        return {"status": "accepted", "revision": state["revision"], "accepted_state_fingerprint": self._fingerprint(state)}

    def author_source(self, *, subject, source, actor=None):
        state = self._load()
        path = self.root / "characters" / f"{subject}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(source), encoding="utf-8")
        self._save(state)
        self._journal({"step": "author-source", "subject": subject, "revision": state["revision"]})
        return {"status": "ok", "revision": state["revision"], "path": path.relative_to(self.root).as_posix()}

    def setup(self, *, packet_kind, subject=None):
        state = self._load()
        packet_id = "packet-" + ec.digest({"kind": packet_kind, "subject": subject, "revision": state["revision"]})[:12]
        self._write_record("packets", packet_id, {"packet_kind": packet_kind, "subject": subject})
        return {"status": "ok", "packet_id": packet_id, "packet_path": f".narrative/packets/{packet_id}.json"}

    def frame(self, *, move, subject=None, writes=None, rationale=""):
        state = self._load()
        request_id = "req-" + ec.digest({"move": move, "subject": subject, "writes": writes, "revision": state["revision"]})[:12]
        self._write_record("requests", request_id, {"move": move, "subject": subject, "writes": writes, "rationale": rationale})
        state["pending"][request_id] = {"kind": "frame", "move": move}
        state["phase"] = "owner-wave"
        self._save(state)
        return {"status": "ok", "revision": state["revision"], "next_action": "realize", "pending_owner_wave": request_id}

    def realize(self, *, move, subject=None, writes=None, owner_wave_ref=None, wave=None, rationale=""):
        state = self._load()
        if wave is None or wave.request_id not in state["pending"]:
            raise NativeStepRefused("realize requires the host wave answering the frame request")
        if not writes:
            raise NativeStepRefused("realize requires at least one declared write")
        payloads = {entry["path"]: entry.get("content", "") for entry in writes}
        payload_sha256 = ec.digest(payloads)
        candidate_bundle = "cb-" + payload_sha256[:12]
        self._write_record(
            "candidates",
            candidate_bundle,
            {"move": move, "subject": subject, "request_id": wave.request_id, "payloads": payloads, "payload_sha256": payload_sha256},
        )
        state["candidate_bundle"] = candidate_bundle
        state["phase"] = "prove"
        self._save(state)
        return {"status": "ok", "candidate_bundle": candidate_bundle, "revision": state["revision"]}

    def prove(self, *, candidate_bundle, evidence_kind, evidence):
        state = self._load()
        candidate = self._read_record("candidates", candidate_bundle)
        if not isinstance(evidence, dict):
            raise NativeStepRefused("evidence must be a typed object")
        if evidence.get("candidate_bundle") != candidate_bundle:
            raise NativeStepRefused("evidence does not bind the candidate bundle")
        if evidence.get("payload_sha256") != candidate["payload_sha256"]:
            raise NativeStepRefused("evidence does not bind the candidate bytes")
        evaluation_id = "eval-" + ec.digest({"candidate_bundle": candidate_bundle, "evidence_kind": evidence_kind})[:12]
        self._write_record(
            "evaluations",
            evaluation_id,
            {"candidate_bundle": candidate_bundle, "evidence_kind": evidence_kind, "evidence": evidence, "status": "pass"},
        )
        state["proof_status"] = "pass"
        state["evaluations"].append(evaluation_id)
        state["phase"] = "resume"
        self._save(state)
        return {"status": "ok", "evaluation_id": evaluation_id, "revision": state["revision"]}

    def resume(self, *, move=None, subject=None, candidate_bundle, proof_evaluation):
        state = self._load()
        candidate = self._read_record("candidates", candidate_bundle)
        evaluation = self._read_record("evaluations", proof_evaluation)
        if evaluation["candidate_bundle"] != candidate_bundle:
            raise NativeStepRefused("proof evaluation does not bind the candidate bundle")
        request_id = "req-result-" + candidate_bundle[3:11]
        self._write_record("requests", request_id, {"kind": "result", "candidate_bundle": candidate_bundle, "move": candidate["move"]})
        state["pending"][request_id] = {"kind": "result", "candidate_bundle": candidate_bundle}
        state["phase"] = "reconcile"
        self._save(state)
        return {"status": "ok", "revision": state["revision"], "next_action": "reconcile", "pending_owner_wave": request_id}

    def reconcile(self, *, move=None, subject=None, candidate_bundle, proof_evaluation, owner_wave_ref=None, wave=None):
        state = self._load()
        candidate = self._read_record("candidates", candidate_bundle)
        evaluation = self._read_record("evaluations", proof_evaluation)
        if evaluation["candidate_bundle"] != candidate_bundle:
            raise NativeStepRefused("proof evaluation does not bind the candidate bundle")
        if wave is None or wave.candidate_bundle != candidate_bundle:
            raise NativeStepRefused("reconcile requires the host wave binding this candidate")
        changed = []
        for rel, text in candidate["payloads"].items():
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            self._snapshot(state, rel, text)
            changed.append(rel)
        state["revision"] += 1
        state["changed_paths"] = changed
        state["phase"] = "advance"
        self._save(state)
        reconcile_id = "rec-" + candidate_bundle[3:11]
        self._write_record(
            "reconciles",
            reconcile_id,
            {
                "candidate_bundle": candidate_bundle,
                "proof_evaluation": proof_evaluation,
                "owner_wave_ref": owner_wave_ref,
                "revision": state["revision"],
                "changed_paths": changed,
                "accepted_state_fingerprint": self._fingerprint(state),
            },
        )
        self._journal({"step": "reconcile", "revision": state["revision"], "candidate_bundle": candidate_bundle})
        return {"status": "accepted", "revision": state["revision"], "changed_paths": changed}

    def observe(self, *, revision=None):
        state = self._load()
        return {
            "accepted_state_fingerprint": self._fingerprint(state),
            "changed_paths": list(state["changed_paths"]),
            "phase": state["phase"],
            "proof_status": state["proof_status"],
            "candidate_bundle": state["candidate_bundle"],
            "revision": state["revision"],
        }

    def frontier(self):
        state = self._load()
        return {
            "story": _maybe_json(self.root / "story" / "model.json"),
            "plan": _maybe_json(self.root / "plan" / "model.json"),
            "character": [],
            "revision": state["revision"],
            "accepted_state_fingerprint": self._fingerprint(state),
            "pending_owner_waves": sorted(state["pending"]),
            "package_status": "ok",
            "checks": [{"name": "format", "status": "pass"}],
        }

    def authorities(self, *, kind=None):
        state = self._load()
        models = {
            "plan": _maybe_json(self.root / "plan" / "model.json"),
            "story": _maybe_json(self.root / "story" / "model.json"),
            "character": None,
        }
        return {
            "models": models,
            "source_status": [{"path": "plan/source.md", "status": "current"}],
            "decisions": list(state["decisions"]),
            "evaluations": list(state["evaluations"]),
            "proof_families": list(self.PROOF_FAMILIES),
        }


def _maybe_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

from authority_boundary import AuthorityBoundary  # noqa: E402
from native_binding import resolve_native_binding  # noqa: E402


def arm_b_instruction_sha256():
    return hashlib.sha256(ARM_B_INSTRUCTION.read_bytes()).hexdigest()


@pytest.fixture
def w2c_source_snapshot(tmp_path):
    source = tmp_path / "source-snapshot"
    (source / "plan").mkdir(parents=True)
    (source / "story").mkdir()
    (source / "sources").mkdir()
    (source / "plan" / "source.md").write_text("# Plan source\n", encoding="utf-8")
    (source / "story" / "model.json").write_text(
        ec.canonical_json({"kind": "story", "revision": 0, "scenes": []}), encoding="utf-8"
    )
    (source / "sources" / "brief.md").write_text("Author brief.\n", encoding="utf-8")
    return source


@pytest.fixture
def w2c_env_inputs():
    return {
        "task_id": "task-scene-draft",
        "task_family_id": "family-scene-draft",
        "initial_state_fingerprint": "isf-0001",
        "source_snapshot_digest": "ssd-0001",
        "tool_manifest_digest": "tmd-0001",
        "native_binding_digest": nb.native_binding_digest(),
        "model_snapshot": "claude-sonnet-4-5@frozen-2026-09",
        "information_entitlement_digest": "ied-0001",
        "memory_init_digest": "mid-0001",
        "harness_version": ec.HARNESS_VERSION,
        "protocol_version": ec.PROTOCOL_VERSION,
    }


@pytest.fixture
def w2c_reset(tmp_path, w2c_source_snapshot, w2c_env_inputs):
    created = []

    def _reset(arm="B", *, episode_id=None, destination=None, source=None, upstream=None, skill_package_digest="auto", **extra):
        source = Path(source) if source is not None else w2c_source_snapshot
        destination = Path(destination) if destination is not None else (tmp_path / f"dest-{arm}-{len(created)}")
        inputs = dict(w2c_env_inputs)
        inputs.update(upstream or {})
        if skill_package_digest == "auto":
            skill_package_digest = arm_b_instruction_sha256() if arm in ("B", "D") else None
        instance = nw.reset_workspace(
            source=source,
            destination=destination,
            episode_id=episode_id or ec.new_episode_id(),
            arm=arm,
            skill_package_digest=skill_package_digest,
            **inputs,
            **extra,
        )
        created.append(instance)
        return instance

    yield _reset
    for instance in created:
        nw.teardown_workspace(instance)


@pytest.fixture
def w2c_engine_factory():
    def _make(workspace_root):
        return CooperativeNativeEngine(workspace_root)

    return _make


@pytest.fixture
def w2c_harness(w2c_reset, tmp_path):
    def _make(arm="B", *, specialized=False, entitlement=None, wave_store=None, engine=None, logger=None):
        instance = w2c_reset(arm)
        engine = engine if engine is not None else CooperativeNativeEngine(instance.destination)
        ent = entitlement or Entitlement(
            read_allow=("**",),
            write_scope=("stories", "characters", "plan", "project.json", "story"),
            denied=(),
        )
        waves = OwnerWaveAuthority(store_dir=wave_store)
        log = logger if logger is not None else EpisodeLog(arm)
        boundary = AuthorityBoundary(
            engine=engine,
            workspace_root=instance.destination,
            entitlement=ent,
            owner_waves=waves,
            arm=arm,
            logger=log,
        )
        config = McpConfig(
            workspace_root=instance.destination,
            read_allow=ent.read_allow,
            write_scope=ent.write_scope,
            denied=ent.denied,
            arm=arm,
            specialized=specialized,
            run_dir=instance.run_dir,
        )
        server = ProtectedMcpServer(config, engine=engine, owner_waves=waves, logger=log)
        return instance, server, boundary, engine, waves

    return _make


class StdioMcpClient:
    """A minimal stdio JSON-RPC client for the protected MCP server process."""

    def __init__(self, process, config_path):
        self.process = process
        self.config_path = config_path
        self._counter = 0

    def _next_id(self):
        self._counter += 1
        return self._counter

    def request(self, method, params=None, timeout=120):
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params or {}}
        assert self.process.stdin is not None
        self.process.stdin.write(ec.canonical_json(payload) + chr(10))
        self.process.stdin.flush()
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError(f"protected MCP server exited: {stderr.strip()[:5000]}")
        return json.loads(line)

    def call_tool(self, name, arguments=None):
        response = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        if "error" in response:
            raise RuntimeError(f"server error: {response['error']}")
        return json.loads(response["result"]["content"][0]["text"])

    def close(self):
        try:
            self.process.terminate()
            self.process.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.process.kill()


@pytest.fixture
def w2c_stdio_client(tmp_path):
    clients = []

    def _make(config_data, cwd=None):
        config_path = tmp_path / f"mcp-config-{len(clients)}.json"
        config_path.write_text(ec.canonical_json(config_data), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(W2C_SRC), str(W2C_TESTS)])
        env["W2C_MCP_CONFIG"] = str(config_path)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", "protected_mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd or tmp_path),
            env=env,
        )
        client = StdioMcpClient(process, config_path)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()


@pytest.fixture
def driver_probe_env(tmp_path):
    """Drive the probe in one of the three F-01 worlds.

    The workspace disk layout is shared across calls (tmp_path/probe-world), so a
    grant spec and a deny spec for the same test differ ONLY in the entitlement.
    ``permissive=True`` is the red control: the grant entitlement plus the
    permissive CLI/import/secret surface.
    """

    def _run(*, permissive=False, world=None, pinned_bin_dir=None, seam=None, seam_paths=None):
        effective = "grant" if permissive else (world or "deny")
        probe_root = tmp_path / "probe-world"
        probe_root.mkdir(parents=True, exist_ok=True)
        workspace = cap.build_probe_world(probe_root, open_world=permissive)
        spec = cap.build_probe_spec(
            probe_root=probe_root,
            workspace_root=workspace,
            entitlement=cap.WORLD_ENTITLEMENTS[effective],
            targets=cap.world_targets(effective),
            world=effective,
            seam=seam,
            seam_paths=seam_paths,
        )
        cwd = ec.participant_cwd()
        if permissive:
            base = {key: value for key, value in os.environ.items() if not ec.is_credential_key(key)}
            env = cap.permissive_env(base, probe_root)
        else:
            empty_bin = tmp_path / "empty-bin"
            empty_bin.mkdir(exist_ok=True)
            env = ec.participant_child_env(cwd, pinned_bin_dir=pinned_bin_dir or empty_bin)
        try:
            report = cap.run_probe(spec=spec, cwd=cwd, env=env)
        finally:
            ec.cleanup_participant_cwd(cwd)
        return report, spec

    return _run


