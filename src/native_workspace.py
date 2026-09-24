"""Disposable native workspace lifecycle for W2c.

Reset copies a quiescent, permissioned source snapshot into a fresh unoccupied
destination; the source is never modified. The destination census equals the
authorized source-snapshot census exactly, and reset refuses any snapshot that
carries a .claude/ root at all or an ambient .mcp.json (shared-interface.md S4
rule 2 / S6, closed by N-03), so the zero-entry project-setting-source
postcondition is entailed by the precondition.

Conversation, memory, caches and per-episode credentials live in a host run
directory OUTSIDE the censused destination (S4 rule 4 / S6 census scope).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from environment_contract import (
    HARNESS_VERSION,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ContractViolation,
    build_environment_id_record,
    canonical_json,
    compute_environment_id,
    digest,
    new_episode_credential_id,
    new_workspace_instance_id,
    parse_canonical_json,
    verify_environment_id,
)

FORBIDDEN_SETTING_COMPONENT = ".claude"
AMBIENT_MCP_NAME = ".mcp.json"


class SnapshotRefused(ContractViolation):
    code = "snapshot-refused"


class ResetRefused(ContractViolation):
    code = "reset-refused"


class TeardownRefused(ContractViolation):
    code = "teardown-refused"


@dataclass(frozen=True)
class SnapshotCensus:
    entries: Mapping[str, Mapping[str, Any]]
    digest: str
    root: str

    def __len__(self) -> int:
        return len(self.entries)

    def paths(self) -> list[str]:
        return sorted(self.entries)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def census_tree(root: str | Path) -> SnapshotCensus:
    root = Path(root)
    entries: dict[str, Mapping[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries[rel] = {"kind": "symlink", "target": str(path.readlink())}
        elif path.is_dir():
            entries[rel] = {"kind": "dir"}
        elif path.is_file():
            entries[rel] = {"kind": "file", "size": path.stat().st_size, "sha256": _hash_file(path)}
        else:
            raise SnapshotRefused(f"unsupported filesystem entry kind: {rel}")
    return SnapshotCensus(entries=entries, digest=digest(entries), root=str(root))


def census_files(root: str | Path) -> SnapshotCensus:
    """The file-only census used for byte equality across arms."""
    full = census_tree(root)
    files = {k: v for k, v in full.entries.items() if v.get("kind") == "file"}
    return SnapshotCensus(entries=files, digest=digest(files), root=str(root))


def assert_snapshot_admissible(source: str | Path) -> None:
    """Refuse any .claude/ root at all and any ambient .mcp.json (closed, N-03)."""
    source = Path(source)
    if not source.is_dir():
        raise SnapshotRefused(f"source snapshot is not a directory: {source}")
    if source.name == FORBIDDEN_SETTING_COMPONENT:
        raise SnapshotRefused("the source snapshot root itself is a .claude/ directory")
    for path in [source, *sorted(source.rglob("*"))]:
        rel_parts = path.relative_to(source).parts if path != source else ()
        if FORBIDDEN_SETTING_COMPONENT in rel_parts or path.name == FORBIDDEN_SETTING_COMPONENT:
            raise SnapshotRefused(
                f"source snapshot carries a .claude/ entry: {path.relative_to(source) if path != source else '.'}"
            )
        if path.name == AMBIENT_MCP_NAME:
            raise SnapshotRefused(
                f"source snapshot carries an ambient .mcp.json: {path.relative_to(source) if path != source else '.'}"
            )


def project_setting_source_census(destination: str | Path) -> list[str]:
    """The destination project-setting-source census. Zero entries by construction."""
    destination = Path(destination)
    root = destination / FORBIDDEN_SETTING_COMPONENT
    if not root.exists():
        return []
    return sorted(p.relative_to(destination).as_posix() for p in root.rglob("*"))


def _assert_outside(path: Path, other: Path, *, what: str) -> None:
    try:
        path.resolve().relative_to(other.resolve())
        inside = True
    except ValueError:
        inside = False
    try:
        other.resolve().relative_to(path.resolve())
        ancestor = True
    except ValueError:
        ancestor = False
    if inside or ancestor or path.resolve() == other.resolve():
        raise ResetRefused(f"{what} overlaps {other}")


@dataclass(frozen=True)
class Credentials:
    episode_id: str
    credential_id: str
    revoked: bool = False


@dataclass
class WorkspaceInstance:
    workspace_instance_id: str
    episode_id: str
    task_id: str
    arm: str
    source: Path
    destination: Path
    run_dir: Path
    environment_record: Mapping[str, Any]
    environment_id: str
    source_census_digest: str
    destination_census_digest: str
    credentials: Credentials
    _source_census: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, repr=False)
    _recorded_destination: str = field(default="", repr=False)

    @property
    def conversation_path(self) -> Path:
        return self.run_dir / "conversation.json"

    @property
    def memory_dir(self) -> Path:
        return self.run_dir / "memory"

    @property
    def cache_dir(self) -> Path:
        return self.run_dir / "cache"

    @property
    def environment_path(self) -> Path:
        return self.run_dir / "environment.json"

    def conversation(self) -> Mapping[str, Any]:
        return parse_canonical_json(self.conversation_path.read_text(encoding="utf-8"))

    def empty_state(self) -> dict[str, Any]:
        conversation = self.conversation()
        return {
            "conversation_length": len(conversation.get("messages", [])),
            "memory_entries": sorted(p.name for p in self.memory_dir.rglob("*")) if self.memory_dir.is_dir() else [],
            "cache_entries": sorted(p.name for p in self.cache_dir.rglob("*")) if self.cache_dir.is_dir() else [],
            "credentials_revoked": self.credentials.revoked,
        }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="utf-8")


def reset_workspace(
    *,
    source: str | Path,
    destination: str | Path,
    episode_id: str,
    task_id: str,
    task_family_id: str,
    arm: str,
    initial_state_fingerprint: str,
    source_snapshot_digest: str,
    skill_package_digest: str | None,
    tool_manifest_digest: str,
    native_binding_digest: str,
    model_snapshot: str,
    information_entitlement_digest: str,
    memory_init_digest: str,
    schema_version: int = SCHEMA_VERSION,
    harness_version: str = HARNESS_VERSION,
    protocol_version: str = PROTOCOL_VERSION,
) -> WorkspaceInstance:
    source = Path(source)
    destination = Path(destination)
    assert_snapshot_admissible(source)
    if destination.exists():
        raise ResetRefused(f"destination already exists; reset never adopts a destination: {destination}")
    _assert_outside(destination, source, what="destination")
    run_dir = Path(tempfile.mkdtemp(prefix="w2c-run-"))
    _assert_outside(run_dir, source, what="run directory")
    _assert_outside(run_dir, destination, what="run directory")

    source_census = census_files(source)
    try:
        shutil.copytree(source, destination, symlinks=True)
        _write_json(run_dir / "conversation.json", {"messages": []})
        (run_dir / "memory").mkdir()
        (run_dir / "cache").mkdir()
        credentials = Credentials(episode_id=episode_id, credential_id=new_episode_credential_id())
        _write_json(
            run_dir / "credentials.json",
            {"episode_id": episode_id, "credential_id": credentials.credential_id, "revoked": False},
        )

        workspace_instance_id = new_workspace_instance_id()
        record = build_environment_id_record(
            {
                "schema_version": schema_version,
                "task_id": task_id,
                "task_family_id": task_family_id,
                "episode_id": episode_id,
                "arm": arm,
                "workspace_instance_id": workspace_instance_id,
                "initial_state_fingerprint": initial_state_fingerprint,
                "source_snapshot_digest": source_snapshot_digest,
                "skill_package_digest": skill_package_digest,
                "tool_manifest_digest": tool_manifest_digest,
                "native_binding_digest": native_binding_digest,
                "model_snapshot": model_snapshot,
                "harness_version": harness_version,
                "protocol_version": protocol_version,
                "information_entitlement_digest": information_entitlement_digest,
                "memory_init_digest": memory_init_digest,
            }
        )
        environment_id = compute_environment_id(record)
        _write_json(run_dir / "environment.json", {"record": record, "environment_id": environment_id})
        _write_json(run_dir / "destination.json", {"destination": str(destination)})

        destination_census = census_files(destination)
        if destination_census.paths() != source_census.paths():
            raise ResetRefused("destination census differs from the authorized source snapshot")
        for rel in source_census.paths():
            if destination_census.entries[rel] != source_census.entries[rel]:
                raise ResetRefused(f"destination entry differs from the authorized source snapshot: {rel}")
        if project_setting_source_census(destination):
            raise ResetRefused("destination project-setting-source census is not empty")
        recomputed = parse_canonical_json((run_dir / "environment.json").read_text(encoding="utf-8"))
        verify_environment_id(recomputed["record"], recomputed["environment_id"])
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    return WorkspaceInstance(
        workspace_instance_id=workspace_instance_id,
        episode_id=episode_id,
        task_id=task_id,
        arm=arm,
        source=source,
        destination=destination,
        run_dir=run_dir,
        environment_record=record,
        environment_id=environment_id,
        source_census_digest=source_census.digest,
        destination_census_digest=destination_census.digest,
        credentials=credentials,
        _source_census=source_census.entries,
        _recorded_destination=str(destination),
    )


def teardown_workspace(instance: WorkspaceInstance) -> None:
    """Discard only THIS instance's recorded destination and revoke its credentials.

    Teardown refuses to remove a directory that is not the destination recorded at
    reset time (F-06). The per-instance record is the in-memory
    `_recorded_destination` written by reset_workspace and, defensively, the
    run-directory `destination.json`. A caller that mutates `instance.destination`
    to another path is refused before any removal happens.
    """
    recorded = str(getattr(instance, "_recorded_destination", "") or "")
    destination = Path(instance.destination)
    record_path = instance.run_dir / "destination.json"
    if record_path.exists():
        raw = parse_canonical_json(record_path.read_text(encoding="utf-8"))
        recorded_on_disk = str(raw.get("destination", ""))
        if not recorded or recorded_on_disk != recorded:
            raise TeardownRefused(
                f"teardown refused: run-directory destination record {recorded_on_disk!r} is not {recorded!r}"
            )
        if Path(recorded_on_disk) != destination:
            raise TeardownRefused(
                f"teardown refused: {destination} is not this instance's recorded destination {recorded_on_disk}"
            )
    elif not recorded or Path(recorded) != destination:
        raise TeardownRefused(
            f"teardown refused: {destination} is not this instance's recorded destination {recorded!r}"
        )
    shutil.rmtree(destination, ignore_errors=True)
    if instance.run_dir.exists():
        creds_path = instance.run_dir / "credentials.json"
        if creds_path.exists():
            creds = parse_canonical_json(creds_path.read_text(encoding="utf-8"))
            creds["revoked"] = True
            _write_json(creds_path, creds)
        shutil.rmtree(instance.run_dir, ignore_errors=True)


def verify_source_unchanged(instance: WorkspaceInstance) -> bool:
    current = census_files(instance.source)
    if current.digest != instance.source_census_digest:
        raise ResetRefused("source snapshot changed during the episode")
    return True


def destination_matches_source(instance: WorkspaceInstance) -> bool:
    destination = census_files(instance.destination)
    if destination.paths() != sorted(instance._source_census):
        return False
    return all(destination.entries[rel] == instance._source_census[rel] for rel in destination.paths())
