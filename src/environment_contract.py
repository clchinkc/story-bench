"""W2c environment contract: canonical 16-member environment identity, the
instance-scoped reset/replay commitment, stale/duplicate/replay rejection and the
two deterministic participant-construction points.

This module owns no narrative state and performs no I/O beyond explicit host
inputs. It never imports narrative_craft and never writes a shadow store.

Contract basis: bindings/execution/w2c/contract-repair-2/shared-interface.md S4
rule 6, S5, S7, S8 and S9, and risk-test-map.md E1-E6, PS1, N1, R1-R5, T4, P1.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
PROTOCOL_VERSION = "story-bench-episode-protocol-1"
HARNESS_VERSION = "story-bench-harness-0.2c"

# S5.1 — the closed canonical member list. Exactly 16 members. Order is the
# canonical serialization order and is also the set used for exact-key refusal.
ENVIRONMENT_ID_MEMBERS: tuple[str, ...] = (
    "schema_version",
    "task_id",
    "task_family_id",
    "episode_id",
    "arm",
    "workspace_instance_id",
    "initial_state_fingerprint",
    "source_snapshot_digest",
    "skill_package_digest",
    "tool_manifest_digest",
    "native_binding_digest",
    "model_snapshot",
    "harness_version",
    "protocol_version",
    "information_entitlement_digest",
    "memory_init_digest",
)
ENVIRONMENT_ID_MEMBER_COUNT = len(ENVIRONMENT_ID_MEMBERS)
assert ENVIRONMENT_ID_MEMBER_COUNT == 16, "the environment-ID member list is closed at 16"

# Replay-refusal inputs that are deliberately NOT members of the 16-member
# record (N-02): the frozen trajectory digest is independent, and a record that
# adds it as a 17th member is refused by build_environment_id_record().
TRAJECTORY_DIGEST_IS_MEMBER = False

STOP_REASONS: frozenset[str] = frozenset(
    {
        "completed",
        "truncated",
        "budget",
        "policy_rejection",
        "provider_failure",
        "grader_failure",
        "not_run",
        "owner_wait",
    }
)

# The one member that may legitimately be null (S5.1 #9: null when the arm's
# instruction treatment is off).
NULLABLE_MEMBERS: frozenset[str] = frozenset({"skill_package_digest"})


class ContractViolation(Exception):
    """Base class for a typed W2c contract refusal."""

    code = "contract-violation"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code or type(self).code


class CanonicalJsonError(ContractViolation):
    code = "canonical-json"


class EnvironmentIdError(ContractViolation):
    code = "environment-id"


class StaleIdentityError(ContractViolation):
    code = "stale-identity"


class ReplayMismatch(ContractViolation):
    code = "replay-mismatch"


class ParticipantEnvironmentError(ContractViolation):
    code = "participant-environment"


# --------------------------------------------------------------------------
# Canonical JSON law (S5.3): sorted keys, compact separators, unescaped Unicode,
# no NaN/Infinity, duplicate keys rejected.
# --------------------------------------------------------------------------
def _reject_constant(name: str) -> Any:
    raise CanonicalJsonError(f"non-finite number is not canonical JSON: {name}")


def _no_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise CanonicalJsonError(f"duplicate key is not canonical JSON: {key!r}")
        seen[key] = value
    return seen


def parse_canonical_json(text: str) -> Any:
    """Parse JSON under the canonical law, refusing duplicate keys and non-finite numbers."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except CanonicalJsonError:
        raise
    except (TypeError, ValueError) as exc:
        raise CanonicalJsonError(f"unparseable JSON: {exc}") from exc


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalJsonError(f"value is not canonical-JSON serializable: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


# --------------------------------------------------------------------------
# S5.1-S5.3 — the closed 16-member environment-ID record.
# --------------------------------------------------------------------------
def build_environment_id_record(members: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and freeze one closed 16-member record.

    Missing, extra or 15/17-member input is refused; there is no default and no
    recomputation from partial data.
    """
    if not isinstance(members, Mapping):
        raise EnvironmentIdError("environment-ID record must be a mapping")
    given = set(members)
    expected = set(ENVIRONMENT_ID_MEMBERS)
    missing = sorted(expected - given)
    extra = sorted(given - expected)
    if missing:
        raise EnvironmentIdError(f"environment-ID record is missing required members: {missing}")
    if extra:
        raise EnvironmentIdError(f"environment-ID record has non-member fields: {extra}")
    if len(members) != ENVIRONMENT_ID_MEMBER_COUNT:
        raise EnvironmentIdError(
            f"environment-ID record must have exactly {ENVIRONMENT_ID_MEMBER_COUNT} members, got {len(members)}"
        )
    for name in ENVIRONMENT_ID_MEMBERS:
        value = members[name]
        if value is None and name not in NULLABLE_MEMBERS:
            raise EnvironmentIdError(f"environment-ID member {name!r} is required and may not be null")
        if name == "arm" and value not in {"A", "B", "C", "D"}:
            raise EnvironmentIdError(f"environment-ID arm must be one of A/B/C/D, got {value!r}")
    return {name: members[name] for name in ENVIRONMENT_ID_MEMBERS}


def compute_environment_id(members: Mapping[str, Any]) -> str:
    return digest(build_environment_id_record(members))


def verify_environment_id(members: Mapping[str, Any], environment_id: str) -> str:
    computed = compute_environment_id(members)
    if computed != environment_id:
        raise EnvironmentIdError(
            f"environment-ID mismatch: recorded {environment_id!r} != recomputed {computed!r}"
        )
    return computed


def new_workspace_instance_id() -> str:
    """Fresh on every reset, never reused (S5.2)."""
    return "wsi-" + os.urandom(16).hex()


def new_episode_credential_id() -> str:
    return "epc-" + os.urandom(16).hex()


def new_episode_id() -> str:
    return "epi-" + os.urandom(16).hex()


# --------------------------------------------------------------------------
# S4 rule 6 (N-04) — participant cwd and child environment.
# --------------------------------------------------------------------------
_INHERITED_INTERPRETER_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")

# F-02 / AC-2a..b: the CLOSED participant credential policy. It is a denylist
# LAW, not an ad-hoc partial list: a key is removed when its upper-cased name
# is an exact member, starts with a declared credential namespace prefix, or
# ends with a declared credential suffix. The namespace prefixes are matched
# case-insensitively, so NC_EVAL_* (including NC_EVAL_root) and every
# case-variant AWS_/GITHUB_/GH_/AZURE_OPENAI_ spelling is covered. This closes
# the F-02 leak of AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN/
# AWS_PROFILE/AWS_DEFAULT_REGION/GITHUB_TOKEN/GH_TOKEN/AZURE_OPENAI_ENDPOINT and
# the case-variant nc_eval_root.
CREDENTIAL_EXACT_KEYS: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_ORG_ID",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "GROQ_API_KEY",
        "COHERE_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
    }
)
CREDENTIAL_KEY_PREFIXES: tuple[str, ...] = (
    "AWS_",
    "AZURE_OPENAI_",
    "GITHUB_",
    "GH_",
    "NC_EVAL_",
)
CREDENTIAL_KEY_SUFFIXES: tuple[str, ...] = ("_API_KEY", "_API_TOKEN", "_API_SECRET", "_SECRET_KEY", "_ACCESS_TOKEN")


def is_credential_key(name: str) -> bool:
    """True when the closed credential policy removes this environment key."""
    upper = name.upper()
    return (
        upper in CREDENTIAL_EXACT_KEYS
        or upper.startswith(CREDENTIAL_KEY_PREFIXES)
        or upper.endswith(CREDENTIAL_KEY_SUFFIXES)
    )


def is_provider_key(name: str) -> bool:
    """Backward-compatible alias for is_credential_key; delegates to the closed
    credential policy above."""
    return is_credential_key(name)


def default_pinned_bin_dir() -> Path | None:
    """The directory of the pinned claude executable, or None when not resolvable.

    The W2c supersession of the host-runner precedent's untouched PATH: the
    participant PATH is reduced to exactly this directory, so no
    narrative-craft/native CLI and no ambient tool is reachable by name.
    """
    exe = shutil.which("claude")
    return Path(exe).resolve().parent if exe else None


def participant_cwd() -> Path:
    """A freshly created, empty temporary directory outside source and destination.

    Adopted exactly from host_runner.py:117 (Path(tempfile.mkdtemp())): the
    default identity-free prefix names nothing about a run, case or path.
    """
    return Path(tempfile.mkdtemp())


def cleanup_participant_cwd(cwd: Path) -> None:
    shutil.rmtree(cwd, ignore_errors=True)


def participant_child_env(
    cwd: Path,
    *,
    environ: Mapping[str, str] | None = None,
    pinned_bin_dir: Path | str | None = None,
) -> dict[str, str]:
    """The participant's exact sanitized environment.

    A strip-list, deliberately not an allowlist (host_runner.py:146-179):
    os.environ minus PYTHONPATH/PYTHONHOME/VIRTUAL_ENV and minus every key the
    CLOSED credential policy (is_credential_key) names - an exact set plus the
    AWS_/AZURE_OPENAI_/GITHUB_/GH_/NC_EVAL_ namespace prefixes, matched
    case-insensitively, plus the credential suffixes - with PWD set to the temp
    cwd. HOME and the keychain surface survive. W2c supersedes the precedent's
    untouched PATH and provider-key variables only: PATH becomes the pinned
    executable directory and the credential policy is applied.
    """
    base = dict(os.environ if environ is None else environ)
    env = {
        key: value
        for key, value in base.items()
        if key not in _INHERITED_INTERPRETER_VARS and not is_credential_key(key)
    }
    if pinned_bin_dir is None:
        pinned_bin_dir = default_pinned_bin_dir()
    env["PATH"] = str(pinned_bin_dir) if pinned_bin_dir is not None else ""
    env["PWD"] = str(cwd)
    return env


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_participant_cwd(cwd: Path, *, source: Path, destination: Path) -> Path:
    """Refuse a participant cwd that could hand over inherited project memory.

    The cwd must be a freshly created empty directory outside both the source
    snapshot and the byte-equal destination, so no CLAUDE.md/AGENTS.md or
    .claude/ tree is reachable from it.
    """
    cwd = Path(cwd)
    if not cwd.is_dir():
        raise ParticipantEnvironmentError(f"participant cwd is not a directory: {cwd}")
    entries = sorted(p.name for p in cwd.iterdir())
    if entries:
        raise ParticipantEnvironmentError(
            f"participant cwd must be empty; found {entries!r}"
        )
    if _is_within(cwd, source) or _is_within(source, cwd):
        raise ParticipantEnvironmentError("participant cwd overlaps the source snapshot")
    if _is_within(cwd, destination) or _is_within(destination, cwd):
        raise ParticipantEnvironmentError("participant cwd overlaps the destination workspace")
    for marker in ("CLAUDE.md", "AGENTS.md", ".claude", ".mcp.json"):
        if (cwd / marker).exists():
            raise ParticipantEnvironmentError(f"participant cwd exposes inherited memory: {marker}")
    return cwd


# --------------------------------------------------------------------------
# S7/S8 — replay commitment, duplicate identity and stale/replayed refusal.
# --------------------------------------------------------------------------
def trajectory_digest(observations: Iterable[Mapping[str, Any]]) -> str:
    return digest([dict(obs) for obs in observations])


@dataclass(frozen=True)
class EpisodeRecord:
    """One frozen episode: identity, ordered observations and artifacts."""

    episode_id: str
    environment_id: str
    environment_record: Mapping[str, Any]
    trajectory_digest: str
    observations: tuple[Mapping[str, Any], ...]
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    stop_reason: str = "completed"
    content_digest: str = ""


class EpisodeRegistry:
    """In-memory registry used to prove idempotent duplicate identity.

    No file is written here; the host owns persistence. Registering the same
    episode_id with identical content returns the existing record and creates
    no second entry; different content is refused with the accepted bytes
    preserved (the existing record is untouched).
    """

    def __init__(self) -> None:
        self._records: dict[str, EpisodeRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def get(self, episode_id: str) -> EpisodeRecord:
        try:
            return self._records[episode_id]
        except KeyError as exc:
            raise ContractViolation(f"unknown episode_id {episode_id!r}", code="unknown-episode") from exc

    def register(
        self,
        *,
        episode_id: str,
        environment_id: str,
        environment_record: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
        artifacts: Mapping[str, Any] | None = None,
        stop_reason: str = "completed",
    ) -> EpisodeRecord:
        if stop_reason not in STOP_REASONS:
            raise ContractViolation(f"unknown stop reason {stop_reason!r}", code="stop-reason")
        frozen_obs = tuple(dict(obs) for obs in observations)
        traj = trajectory_digest(frozen_obs)
        artifacts = dict(artifacts or {})
        content_digest = digest(
            {
                "episode_id": episode_id,
                "environment_id": environment_id,
                "environment_record": dict(environment_record),
                "trajectory_digest": traj,
                "observations": [dict(o) for o in frozen_obs],
                "artifacts": artifacts,
                "stop_reason": stop_reason,
            }
        )
        existing = self._records.get(episode_id)
        if existing is not None:
            if existing.content_digest == content_digest:
                return existing
            raise ReplayMismatch(
                f"episode_id {episode_id!r} already exists with different content; "
                "the existing record and its accepted bytes are preserved"
            )
        record = EpisodeRecord(
            episode_id=episode_id,
            environment_id=environment_id,
            environment_record=dict(environment_record),
            trajectory_digest=traj,
            observations=frozen_obs,
            artifacts=artifacts,
            stop_reason=stop_reason,
            content_digest=content_digest,
        )
        self._records[episode_id] = record
        return record


def assert_replayable(
    frozen: EpisodeRecord,
    *,
    environment_record: Mapping[str, Any] | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Refuse replay when any of the 16 members or the trajectory digest changed.

    A replay reproduces the recorded observations from frozen artifacts and
    never silently substitutes a fresh result.
    """
    if environment_record is not None:
        record = build_environment_id_record(environment_record)
        changed = {
            name: (frozen.environment_record.get(name), record[name])
            for name in ENVIRONMENT_ID_MEMBERS
            if frozen.environment_record.get(name) != record[name]
        }
        if changed:
            raise StaleIdentityError(f"replay refused: environment members changed: {sorted(changed)}")
    if observations is not None:
        if trajectory_digest(observations) != frozen.trajectory_digest:
            raise ReplayMismatch("replay refused: frozen trajectory digest changed")


def replay_observations(
    frozen: EpisodeRecord,
    *,
    environment_record: Mapping[str, Any] | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    assert_replayable(frozen, environment_record=environment_record, observations=observations)
    return frozen.observations


def truncation_stop_reason(observations: Sequence[Mapping[str, Any]]) -> str:
    """A mid-sentence truncation stays a truncation and cannot be promoted."""
    for obs in observations:
        if obs.get("truncated") is True or obs.get("stop_reason") == "truncated":
            return "truncated"
    return "completed"
