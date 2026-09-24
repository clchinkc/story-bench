"""Proposal-to-observation adapter over the native public surface.

One single mutation channel with a closed phase enum (shared-interface.md S2.1).
The adapter re-applies the same native authorization, currentness, scope,
proof-admission, concurrency and accepted-byte-preservation checks to every
phase. It owns no narrative state: every native step is delegated to an injected
NativeEngine (the pinned native CLI in production; the cooperative local
harness in offline tests).

Owner waves are host-issued only. The participant surface can never mint,
answer, replay or alter a wave.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from environment_contract import ContractViolation, canonical_json, digest, parse_canonical_json

PROPOSE_PHASES: tuple[str, ...] = (
    "genesis",
    "author-source",
    "setup",
    "frame",
    "realize",
    "prove",
    "resume",
    "reconcile",
)
ADVANCE_IS_A_READ = True
ACCEPTING_PHASES: frozenset[str] = frozenset({"genesis", "reconcile"})

SETUP_PACKET_KINDS: frozenset[str] = frozenset({"semantic", "critique", "character"})
EVIDENCE_KINDS: frozenset[str] = frozenset(
    {"proof-draft", "critique-findings", "character-evidence", "schema-validation"}
)
DECLARED_MOVES: frozenset[str] = frozenset(
    {"plan-change", "character-change", "scene-draft", "scene-revision", "story-model-correction", "publish"}
)


class AuthorityRefused(ContractViolation):
    code = "authority-refused"


class PhaseRefused(AuthorityRefused):
    code = "phase-refused"


class ScopeRefused(AuthorityRefused):
    code = "scope-refused"


class OwnerWaveRefused(AuthorityRefused):
    code = "owner-wave-refused"


class StaleRefused(AuthorityRefused):
    code = "stale-refused"


class ProofRefused(AuthorityRefused):
    code = "proof-refused"


class NativeStepRefused(AuthorityRefused):
    code = "native-step-refused"


class NativeStepError(ContractViolation):
    code = "native-step-error"


class NativeEngine(Protocol):
    def revision(self) -> int: ...
    def accepted_fingerprint(self) -> str: ...
    def genesis(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def author_source(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def setup(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def frame(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def realize(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def prove(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def resume(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def reconcile(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def observe(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def frontier(self) -> Mapping[str, Any]: ...
    def authorities(self, **kwargs: Any) -> Mapping[str, Any]: ...


# --------------------------------------------------------------------------
# Path and scope safety
# --------------------------------------------------------------------------
def normalize_rel_path(rel: str) -> str:
    if not isinstance(rel, str) or not rel:
        raise ScopeRefused("a workspace path must be a non-empty relative POSIX string")
    if rel.startswith("/") or rel.startswith("\\"):
        raise ScopeRefused(f"absolute paths are refused: {rel!r}")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or pure.parts[:1] == ("..",) or ".." in pure.parts:
        raise ScopeRefused(f"directory traversal is refused: {rel!r}")
    if not pure.parts or any(part in ("", ".") for part in pure.parts):
        raise ScopeRefused(f"non-canonical path is refused: {rel!r}")
    if ":" in pure.parts[0]:
        raise ScopeRefused(f"drive-qualified paths are refused: {rel!r}")
    return pure.as_posix()


def resolve_inside(root: str | Path, rel: str) -> Path:
    rel = normalize_rel_path(rel)
    root = Path(root)
    candidate = root / rel
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ScopeRefused(f"path escapes the workspace root: {rel!r}") from exc
    if candidate.is_symlink():
        try:
            candidate.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ScopeRefused(f"symlink escapes the workspace root: {rel!r}") from exc
    return candidate


@dataclass(frozen=True)
class Entitlement:
    read_allow: tuple[str, ...]
    write_scope: tuple[str, ...]
    denied: tuple[str, ...] = ()

    @staticmethod
    def _matches(rel: str, allowed: Sequence[str]) -> bool:
        for entry in allowed:
            if rel == entry or rel.startswith(entry.rstrip("/") + "/") or entry in ("**", "*"):
                return True
        return False

    def permits_read(self, rel: str) -> bool:
        rel = normalize_rel_path(rel)
        if self._matches(rel, self.denied):
            return False
        return self._matches(rel, self.read_allow)

    def permits_write(self, rel: str) -> bool:
        rel = normalize_rel_path(rel)
        if self._matches(rel, self.denied):
            return False
        return self._matches(rel, self.write_scope)

    def digest(self) -> str:
        return digest(
            {
                "read_allow": sorted(self.read_allow),
                "write_scope": sorted(self.write_scope),
                "denied": sorted(self.denied),
            }
        )


# --------------------------------------------------------------------------
# Host-issued owner waves
# --------------------------------------------------------------------------
@dataclass
class OwnerWave:
    wave_id: str
    request_id: str
    decision: str
    candidate_bundle: str | None
    revision: int | None
    issued_by: str = "host:labeled-simulated-author-fixture"
    consumed: bool = False


class OwnerWaveAuthority:
    """The host-side fixture authority. Only the host holds this object.

    In-memory by default. When store_dir is supplied the authority is
    file-backed so a trusted host component (the protected MCP server process)
    can validate waves the host issued in another process. The store lives
    outside the censused destination and outside the participant entitlement.
    Single use is enforced by an exclusive consumed marker.
    """

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self._waves: dict[str, OwnerWave] = {}
        self._counter = 0
        self._store = Path(store_dir) if store_dir is not None else None
        if self._store is not None:
            self._store.mkdir(parents=True, exist_ok=True)

    def issue(
        self,
        *,
        request_id: str,
        decision: str,
        candidate_bundle: str | None = None,
        revision: int | None = None,
    ) -> str:
        if decision not in {"apply", "revise", "reject"}:
            raise OwnerWaveRefused(f"unknown owner decision {decision!r}")
        self._counter += 1
        wave_id = f"wave-{self._counter:04d}-" + os.urandom(8).hex()
        wave = OwnerWave(
            wave_id=wave_id,
            request_id=request_id,
            decision=decision,
            candidate_bundle=candidate_bundle,
            revision=revision,
        )
        self._waves[wave_id] = wave
        if self._store is not None:
            (self._store / f"{wave_id}.json").write_text(
                canonical_json(
                    {
                        "wave_id": wave_id,
                        "request_id": request_id,
                        "decision": decision,
                        "candidate_bundle": candidate_bundle,
                        "revision": revision,
                        "issued_by": wave.issued_by,
                    }
                ),
                encoding="utf-8",
            )
        return wave_id

    def get(self, wave_id: str) -> OwnerWave:
        wave = self._waves.get(wave_id)
        if wave is None and self._store is not None:
            path = self._store / f"{wave_id}.json"
            if path.exists():
                data = parse_canonical_json(path.read_text(encoding="utf-8"))
                wave = OwnerWave(
                    wave_id=data["wave_id"],
                    request_id=data["request_id"],
                    decision=data["decision"],
                    candidate_bundle=data.get("candidate_bundle"),
                    revision=data.get("revision"),
                    issued_by=data.get("issued_by", "host:labeled-simulated-author-fixture"),
                )
                self._waves[wave_id] = wave
        if wave is None:
            raise OwnerWaveRefused(f"owner wave {wave_id!r} was not issued by the host authority")
        return wave

    def validate_and_consume(
        self,
        wave_id: str,
        *,
        request_id: str | None = None,
        candidate_bundle: str | None = None,
        revision: int | None = None,
    ) -> OwnerWave:
        wave = self.get(wave_id)
        if self._store is not None:
            marker = self._store / f"{wave_id}.consumed"
            try:
                descriptor = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
            except FileExistsError as exc:
                raise StaleRefused(f"owner wave {wave_id!r} was already consumed; replay is refused") from exc
        elif wave.consumed:
            raise StaleRefused(f"owner wave {wave_id!r} was already consumed; replay is refused")
        if request_id is not None and wave.request_id != request_id:
            raise OwnerWaveRefused(f"owner wave {wave_id!r} does not answer request {request_id!r}")
        if candidate_bundle is not None and wave.candidate_bundle != candidate_bundle:
            raise OwnerWaveRefused(f"owner wave {wave_id!r} does not bind candidate {candidate_bundle!r}")
        if revision is not None and wave.revision is not None and wave.revision != revision:
            raise StaleRefused(f"owner wave {wave_id!r} binds revision {wave.revision}, not {revision}")
        wave.consumed = True
        return wave


# --------------------------------------------------------------------------
# The single mutation channel
# --------------------------------------------------------------------------
REQUIRED_FIELDS: Mapping[str, tuple[str, ...]] = {
    "genesis": ("source", "owner_wave_ref"),
    "author-source": ("source", "subject"),
    "setup": ("packet_kind",),
    "frame": ("move",),
    "realize": ("move",),
    "prove": ("candidate_bundle", "evidence_kind", "evidence"),
    "resume": ("candidate_bundle", "proof_evaluation"),
    "reconcile": ("candidate_bundle", "proof_evaluation", "owner_wave_ref"),
}
WAVE_PHASES: frozenset[str] = frozenset({"genesis", "realize", "reconcile"})


@dataclass
class Observation:
    sequence: int
    phase: str
    status: str
    revision: int
    accepted_state_fingerprint: str
    changed_paths: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)
    rejection: Mapping[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        record = {
            "sequence": self.sequence,
            "phase": self.phase,
            "status": self.status,
            "revision": self.revision,
            "accepted_state_fingerprint": self.accepted_state_fingerprint,
            "changed_paths": list(self.changed_paths),
            "detail": dict(self.detail),
        }
        if self.rejection is not None:
            record["rejection"] = dict(self.rejection)
        return record


class AuthorityBoundary:
    """The one phase-scoped mutation channel, with host-issued owner waves."""

    def __init__(
        self,
        *,
        engine: NativeEngine,
        workspace_root: str | Path,
        entitlement: Entitlement,
        owner_waves: OwnerWaveAuthority,
        arm: str = "B",
        logger: Any | None = None,
    ) -> None:
        self.engine = engine
        self.workspace_root = Path(workspace_root)
        self.entitlement = entitlement
        self.owner_waves = owner_waves
        self.arm = arm
        self.logger = logger
        self._lock = threading.Lock()
        self._observations: list[Observation] = []
        self._last_rejection: dict[str, Any] | None = None

    # -- accepted-byte preservation -------------------------------------
    def _fingerprint(self) -> str:
        return self.engine.accepted_fingerprint()

    def accepted_fingerprint(self) -> str:
        return self._fingerprint()

    # -- scope ----------------------------------------------------------
    def _validate_write_paths(self, proposal: Mapping[str, Any]) -> list[str]:
        writes = proposal.get("writes") or []
        if not isinstance(writes, list):
            raise ScopeRefused("writes must be a list of {path, base} entries")
        paths: list[str] = []
        for entry in writes:
            if not isinstance(entry, Mapping) or "path" not in entry:
                raise ScopeRefused("each declared write must be an object with a path")
            rel = normalize_rel_path(str(entry["path"]))
            resolve_inside(self.workspace_root, rel)
            if not self.entitlement.permits_write(rel):
                raise ScopeRefused(f"write outside the declared move scope is refused: {rel}")
            paths.append(rel)
        return paths

    def _validate_phase_fields(self, proposal: Mapping[str, Any]) -> str:
        phase = proposal.get("phase")
        if not isinstance(phase, str) or phase not in PROPOSE_PHASES:
            raise PhaseRefused(
                f"unknown phase {phase!r} is refused; the closed enum is {list(PROPOSE_PHASES)}; "
                "ADVANCE is a read (workspace.observe/workspace.frontier), not a phase"
            )
        for required in REQUIRED_FIELDS[phase]:
            if proposal.get(required) in (None, ""):
                raise PhaseRefused(f"phase {phase!r} requires {required!r}")
        move = proposal.get("move")
        if phase in {"frame", "realize", "resume", "reconcile"}:
            if not isinstance(move, str) or move not in DECLARED_MOVES:
                raise PhaseRefused(f"phase {phase!r} requires a declared move; got {move!r}")
        if phase == "setup" and proposal.get("packet_kind") not in SETUP_PACKET_KINDS:
            raise PhaseRefused(f"setup requires packet_kind in {sorted(SETUP_PACKET_KINDS)}")
        if phase == "prove" and proposal.get("evidence_kind") not in EVIDENCE_KINDS:
            raise PhaseRefused(f"prove requires evidence_kind in {sorted(EVIDENCE_KINDS)}")
        return phase

    def _check_owner_wave(self, phase: str, proposal: Mapping[str, Any]) -> OwnerWave | None:
        if phase not in WAVE_PHASES:
            return None
        wave_id = proposal.get("owner_wave_ref")
        candidate = proposal.get("candidate_bundle")
        return self.owner_waves.validate_and_consume(
            str(wave_id),
            request_id=proposal.get("request_id"),
            candidate_bundle=candidate,
            revision=proposal.get("expected_revision"),
        )

    # -- propose --------------------------------------------------------
    def propose(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(proposal, Mapping):
            return self._refuse("phase-refused", "proposal must be an object", phase=None)
        phase_declared = proposal.get("phase")
        with self._lock:
            before = self._fingerprint()
            try:
                phase = self._validate_phase_fields(proposal)
                if phase in {"frame", "realize", "prove", "resume", "reconcile", "setup", "author-source"}:
                    self._validate_write_paths(proposal)
                expected = proposal.get("expected_revision")
                if expected is not None and int(expected) != self.engine.revision():
                    raise StaleRefused(
                        f"stale proposal: expected revision {expected} but current is {self.engine.revision()}"
                    )
                wave = self._check_owner_wave(phase, proposal)
                result = self._dispatch(phase, proposal, wave)
            except AuthorityRefused as refusal:
                return self._refuse(refusal.code, str(refusal), phase=phase_declared, before=before)
            except ContractViolation as exc:
                return self._refuse(exc.code, str(exc), phase=phase_declared, before=before)
            after = self._fingerprint()
            accepted = phase in ACCEPTING_PHASES and result.get("status") == "accepted"
            if not accepted and phase in ACCEPTING_PHASES:
                # An accepted-only phase that did not accept must not have moved bytes.
                if after != before:
                    return self._refuse(
                        "accepted-byte-preservation",
                        f"phase {phase!r} reported {result.get('status')!r} but changed accepted bytes",
                        phase=phase,
                        before=before,
                        after=after,
                    )
            observation = Observation(
                sequence=len(self._observations) + 1,
                phase=phase,
                status=str(result.get("status", "ok")),
                revision=self.engine.revision(),
                accepted_state_fingerprint=after,
                changed_paths=tuple(result.get("changed_paths", ())),
                detail={k: v for k, v in result.items() if k not in {"changed_paths", "status"}},
            )
            self._observations.append(observation)
            if self.logger is not None:
                self.logger.record(
                    arm=self.arm,
                    channel="workspace.propose",
                    detail={"phase": phase, "status": observation.status, "revision": observation.revision},
                )
            return {**dict(result), "observation": observation.as_record()}

    def _refuse(
        self,
        code: str,
        message: str,
        *,
        phase: str | None,
        before: str | None = None,
        after: str | None = None,
    ) -> dict[str, Any]:
        if before is not None and after is not None and before != after:
            raise NativeStepError("accepted bytes changed on a refused proposal")
        safe = self._fingerprint()
        rejection = {
            "status": "refused",
            "code": code,
            "message": message,
            "phase": phase,
            "revision": self.engine.revision(),
            "accepted_state_fingerprint": safe,
        }
        self._last_rejection = rejection
        if self.logger is not None:
            self.logger.record(
                arm=self.arm,
                channel="workspace.propose",
                detail={"phase": phase, "status": "refused", "code": code},
            )
        return rejection

    def _dispatch(self, phase: str, proposal: Mapping[str, Any], wave: OwnerWave | None) -> Mapping[str, Any]:
        engine = self.engine
        if phase == "genesis":
            return engine.genesis(
                source=proposal["source"],
                owner_wave_ref=proposal["owner_wave_ref"],
                wave=wave,
                actor=proposal.get("actor", "participant"),
            )
        if phase == "author-source":
            return engine.author_source(subject=proposal["subject"], source=proposal["source"], actor=proposal.get("actor"))
        if phase == "setup":
            return engine.setup(packet_kind=proposal["packet_kind"], subject=proposal.get("subject"))
        if phase == "frame":
            return engine.frame(
                move=proposal["move"],
                subject=proposal.get("subject"),
                writes=proposal.get("writes") or [],
                rationale=proposal.get("rationale", ""),
            )
        if phase == "realize":
            return engine.realize(
                move=proposal["move"],
                subject=proposal.get("subject"),
                writes=proposal.get("writes") or [],
                owner_wave_ref=proposal["owner_wave_ref"],
                wave=wave,
                rationale=proposal.get("rationale", ""),
            )
        if phase == "prove":
            return engine.prove(
                candidate_bundle=proposal["candidate_bundle"],
                evidence_kind=proposal["evidence_kind"],
                evidence=proposal["evidence"],
            )
        if phase == "resume":
            return engine.resume(
                move=proposal.get("move"),
                subject=proposal.get("subject"),
                candidate_bundle=proposal["candidate_bundle"],
                proof_evaluation=proposal["proof_evaluation"],
            )
        if phase == "reconcile":
            return engine.reconcile(
                move=proposal.get("move"),
                subject=proposal.get("subject"),
                candidate_bundle=proposal["candidate_bundle"],
                proof_evaluation=proposal["proof_evaluation"],
                owner_wave_ref=proposal["owner_wave_ref"],
                wave=wave,
            )
        raise PhaseRefused(f"unreachable phase {phase!r}")

    # -- reads ----------------------------------------------------------
    def observe(self, revision: int | None = None) -> dict[str, Any]:
        record = dict(self.engine.observe(revision=revision))
        record["last_rejection"] = self._last_rejection
        return record

    def frontier(self) -> dict[str, Any]:
        return dict(self.engine.frontier())

    def authorities(self, kind: str | None = None) -> dict[str, Any]:
        return dict(self.engine.authorities(kind=kind))

    def observations(self) -> list[dict[str, Any]]:
        return [obs.as_record() for obs in self._observations]

    def trajectory_digest(self) -> str:
        return digest(self.observations())
