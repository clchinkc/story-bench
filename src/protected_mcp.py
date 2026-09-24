"""The one basic protected MCP server (all arms) plus the C/D specialized namespace.

Standard-library-only stdio JSON-RPC 2.0 server. The basic surface is identical
in A/B/C/D: entitlement-scoped list/read/search, read-only frontier/authorities/
observe and the single closed-phase proposal channel. Shell, arbitrary path
access, source inspection, environment variables, grading and cross-arm reads
are absent.

Contract basis: shared-interface.md S2, S2.1, S3, S4 rule 5 and risk-test-map.md
T1, T2, T3, T5, T9, L1.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from authority_boundary import (
    AuthorityBoundary,
    Entitlement,
    NativeEngine,
    OwnerWaveAuthority,
    ScopeRefused,
    resolve_inside,
)
from environment_contract import ContractViolation, canonical_json, digest, parse_canonical_json

SERVER_ID = "story-bench-protected-mcp"
SERVER_VERSION = "1.0.0"
TRANSPORT = "stdio"
MCP_PROTOCOL_VERSION = "2024-11-05"
LATENCY_BUDGET_MS = 250

BASIC_TOOL_NAMES: tuple[str, ...] = (
    "workspace.list",
    "workspace.read",
    "workspace.search",
    "workspace.frontier",
    "workspace.authorities",
    "workspace.observe",
    "workspace.propose",
)
SPECIALIZED_TOOL_NAMES: tuple[str, ...] = (
    "nc.ground",
    "nc.proof_families",
    "nc.plan_candidate",
    "nc.critique_packet",
    "nc.critique_ingest",
)
BASIC_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "workspace.list": {"type": "object", "properties": {"scope": {"type": "string"}}, "additionalProperties": False},
    "workspace.read": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "workspace.search": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "scope": {"type": "string"}, "max_hits": {"type": "integer"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "workspace.frontier": {"type": "object", "properties": {}, "additionalProperties": False},
    "workspace.authorities": {"type": "object", "properties": {"kind": {"type": "string"}}, "additionalProperties": False},
    "workspace.observe": {
        "type": "object",
        "properties": {"revision": {"type": "integer"}},
        "required": ["revision"],
        "additionalProperties": False,
    },
    "workspace.propose": {"type": "object", "properties": {"phase": {"type": "string"}}, "required": ["phase"], "additionalProperties": True},
}


def basic_inventory(*, specialized: bool = False) -> dict[str, Any]:
    """The equal basic inventory tuple: identity, version, transport, tools,
    schemas and the equal numeric latency budget. Never arm-specific."""
    names = list(BASIC_TOOL_NAMES) + (list(SPECIALIZED_TOOL_NAMES) if specialized else [])
    # input_schemas is always the BASIC set: the specialized namespace is
    # additive and may never change the equal basic inventory.
    schemas = {name: BASIC_INPUT_SCHEMAS[name] for name in BASIC_TOOL_NAMES}
    return {
        "server_id": SERVER_ID,
        "server_version": SERVER_VERSION,
        "transport": TRANSPORT,
        "protocol_version": MCP_PROTOCOL_VERSION,
        "tool_names": names,
        "input_schemas": schemas,
        "specialized_tool_names": list(SPECIALIZED_TOOL_NAMES) if specialized else [],
        "latency_budget_ms": LATENCY_BUDGET_MS,
    }


def basic_equals_across_arms() -> bool:
    baseline = basic_inventory(specialized=False)
    return all(
        basic_inventory(specialized=False) == baseline for _ in ("A", "B", "C", "D")
    )


def mcp_server_inventory(config: Mapping[str, Any]) -> list[str]:
    """The resolved MCP server inventory for one arm.

    --strict-mcp-config with the explicit protected config yields exactly the
    protected server. Dropping the strict flag is what lets the ambient
    story-bench/.mcp.json linear-server entry load.
    """
    servers = [SERVER_ID]
    if not config.get("strict_mcp_config", True):
        servers.append("linear-server")
    return servers


def assert_only_protected_inventory(config: Mapping[str, Any]) -> list[str]:
    servers = mcp_server_inventory(config)
    if servers != [SERVER_ID]:
        raise ContractViolation(
            f"resolved MCP inventory is {servers!r}; exactly the protected server is required",
            code="ambient-mcp-loaded",
        )
    return servers


# --------------------------------------------------------------------------
# Per-arm logging (risk row L1)
# --------------------------------------------------------------------------
REQUIRED_LOG_CHANNELS: tuple[str, ...] = (
    "output_disclosure",
    "availability",
    "call_count",
    "content_reference",
    "cost",
)


class EpisodeLog:
    """Per-arm episode log. No arm may omit a channel another arm records."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.channels: dict[str, list[dict[str, Any]]] = {name: [] for name in REQUIRED_LOG_CHANNELS}
        self.tool_calls: dict[str, int] = {}

    def record(self, *, arm: str, channel: str, detail: Mapping[str, Any]) -> None:
        if arm != self.arm:
            raise ContractViolation(f"log record for arm {arm!r} in an {self.arm!r} episode log")
        self.channels.setdefault(channel, []).append(dict(detail))
        self.tool_calls[channel] = self.tool_calls.get(channel, 0) + 1

    def record_tool_call(
        self,
        *,
        tool: str,
        disclosure: str,
        availability: str,
        content_reference: str,
        cost_usd: float,
    ) -> None:
        base = {"tool": tool}
        self.record(arm=self.arm, channel="output_disclosure", detail={**base, "disclosure": disclosure})
        self.record(arm=self.arm, channel="availability", detail={**base, "availability": availability})
        self.record(arm=self.arm, channel="call_count", detail={**base, "count": 1})
        self.record(arm=self.arm, channel="content_reference", detail={**base, "reference": content_reference})
        self.record(arm=self.arm, channel="cost", detail={**base, "cost_usd": cost_usd})

    def missing_channels(self) -> list[str]:
        return sorted(name for name in REQUIRED_LOG_CHANNELS if not self.channels.get(name))

    def assert_complete(self) -> None:
        missing = self.missing_channels()
        if missing:
            raise ContractViolation(f"episode log for arm {self.arm} omits channels {missing}", code="log-incomplete")

    def as_record(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "channels": {name: list(records) for name, records in self.channels.items()},
            "tool_calls": dict(self.tool_calls),
        }


# --------------------------------------------------------------------------
# Config and engine construction
# --------------------------------------------------------------------------
@dataclass
class McpConfig:
    workspace_root: Path
    read_allow: tuple[str, ...]
    write_scope: tuple[str, ...]
    denied: tuple[str, ...]
    arm: str
    specialized: bool
    run_dir: Path
    strict_mcp_config: bool = True
    ambient_mcp_path: str | None = None
    owner_wave_store: str | None = None
    engine_spec: Mapping[str, Any] | None = None

    @property
    def entitlement(self) -> Entitlement:
        return Entitlement(read_allow=self.read_allow, write_scope=self.write_scope, denied=self.denied)

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "McpConfig":
        entitlement = data.get("entitlement", {})
        return McpConfig(
            workspace_root=Path(data["workspace"]),
            read_allow=tuple(entitlement.get("read", ["**"])),
            write_scope=tuple(entitlement.get("write_scope", [])),
            denied=tuple(entitlement.get("denied", [])),
            arm=str(data["arm"]),
            specialized=bool(data.get("specialized", False)),
            run_dir=Path(data.get("run_dir", data["workspace"])),
            strict_mcp_config=bool(data.get("strict_mcp_config", True)),
            ambient_mcp_path=data.get("ambient_mcp_path"),
            owner_wave_store=data.get("owner_wave_store"),
            engine_spec=data.get("native_engine"),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace_root),
            "entitlement": {
                "read": list(self.read_allow),
                "write_scope": list(self.write_scope),
                "denied": list(self.denied),
            },
            "arm": self.arm,
            "specialized": self.specialized,
            "run_dir": str(self.run_dir),
            "strict_mcp_config": self.strict_mcp_config,
            "ambient_mcp_path": self.ambient_mcp_path,
            "owner_wave_store": self.owner_wave_store,
            "native_engine": dict(self.engine_spec or {}),
        }


def load_engine(spec: Mapping[str, Any]) -> NativeEngine:
    if "path" in spec:
        path = Path(spec["path"])
        module_name = spec.get("module", path.stem)
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ContractViolation(f"cannot load native engine module from {path}", code="engine-load")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec["module"])
    factory = getattr(module, spec["attr"])
    kwargs = dict(spec.get("kwargs", {}))
    return factory(**kwargs)


# --------------------------------------------------------------------------
# The server
# --------------------------------------------------------------------------
class ProtectedMcpServer:
    def __init__(
        self,
        config: McpConfig,
        *,
        engine: NativeEngine | None = None,
        owner_waves: OwnerWaveAuthority | None = None,
        logger: EpisodeLog | None = None,
    ) -> None:
        self.config = config
        self.owner_waves = owner_waves or OwnerWaveAuthority()
        self.logger = logger or EpisodeLog(config.arm)
        if engine is None:
            if config.engine_spec is None:
                raise ContractViolation("no native engine was supplied", code="engine-missing")
            engine = load_engine(config.engine_spec)
        self.engine = engine
        self.boundary = AuthorityBoundary(
            engine=engine,
            workspace_root=config.workspace_root,
            entitlement=config.entitlement,
            owner_waves=self.owner_waves,
            arm=config.arm,
            logger=self.logger,
        )

    # -- inventory ------------------------------------------------------
    def tool_names(self) -> list[str]:
        return basic_inventory(specialized=self.config.specialized)["tool_names"]

    def inventory(self) -> dict[str, Any]:
        return basic_inventory(specialized=self.config.specialized)

    # -- entitlement-scoped reads ---------------------------------------
    def _permitted_files(self) -> list[str]:
        allowed: list[str] = []
        for path in sorted(self.config.workspace_root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(self.config.workspace_root).as_posix()
            if self.config.entitlement.permits_read(rel):
                allowed.append(rel)
        return allowed

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def tool_list(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        scope = arguments.get("scope")
        entries = []
        for rel in self._permitted_files():
            if scope and not (rel == scope or rel.startswith(str(scope).rstrip("/") + "/")):
                continue
            path = self.config.workspace_root / rel
            entries.append({"path": rel, "kind": "file", "size": path.stat().st_size, "sha256": self._sha256(path)})
        return {"entries": entries, "count": len(entries)}

    def tool_read(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        rel = str(arguments.get("path", ""))
        try:
            if not self.config.entitlement.permits_read(rel):
                raise ScopeRefused(f"read outside the episode entitlement is refused: {rel!r}")
            path = resolve_inside(self.config.workspace_root, rel)
        except ScopeRefused as refusal:
            return {"status": "refused", "code": refusal.code, "message": str(refusal)}
        if not path.is_file():
            return {"status": "refused", "code": "not-found", "message": f"no such permitted file: {rel!r}"}
        raw = path.read_bytes()
        max_bytes = int(arguments.get("max_bytes", 1 << 20))
        truncated = len(raw) > max_bytes
        text = raw[:max_bytes].decode("utf-8", errors="replace")
        return {
            "status": "ok",
            "path": rel,
            "sha256": self._sha256(path),
            "text": text,
            "truncated": truncated,
            "bytes": len(raw),
        }

    def tool_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", ""))
        if not query:
            return {"status": "refused", "code": "search-refused", "message": "query is required"}
        scope = arguments.get("scope")
        max_hits = int(arguments.get("max_hits", 50))
        hits: list[dict[str, Any]] = []
        for rel in self._permitted_files():
            if scope and not (rel == scope or rel.startswith(str(scope).rstrip("/") + "/")):
                continue
            path = self.config.workspace_root / rel
            sha = self._sha256(path)
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if query in line:
                    hits.append({"path": rel, "line": number, "text": line, "sha256": sha})
                    if len(hits) >= max_hits:
                        return {"hits": hits, "count": len(hits), "truncated": True}
        return {"hits": hits, "count": len(hits), "truncated": False}

    # -- specialized namespace (C/D only) -------------------------------
    def _specialized(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not self.config.specialized:
            return {"status": "refused", "code": "namespace-absent", "message": "the specialized namespace is absent for this arm"}
        if name == "nc.proof_families":
            return {"status": "ok", "proof_families": self.boundary.authorities().get("proof_families", [])}
        if name == "nc.ground":
            authorities = self.boundary.authorities()
            return {
                "status": "ok",
                "models": authorities.get("models", {}),
                "decisions": authorities.get("decisions", []),
            }
        if name == "nc.plan_candidate":
            authorities = self.boundary.authorities("plan")
            return {"status": "ok", "plan": authorities.get("models", {}).get("plan")}
        if name == "nc.critique_packet":
            proposal = {
                "phase": "setup",
                "packet_kind": "critique",
                "subject": arguments.get("subject"),
                "move": arguments.get("move", "scene-draft"),
            }
            return self.boundary.propose(proposal)
        if name == "nc.critique_ingest":
            proposal = {
                "phase": "prove",
                "candidate_bundle": arguments["candidate_bundle"],
                "evidence_kind": "critique-findings",
                "evidence": arguments["evidence"],
            }
            return self.boundary.propose(proposal)
        return {"status": "refused", "code": "unknown-specialized-tool", "message": name}

    # -- dispatch -------------------------------------------------------
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        arguments = dict(arguments or {})
        if name in SPECIALIZED_TOOL_NAMES:
            return self._specialized(name, arguments)
        if name not in BASIC_TOOL_NAMES:
            return {"status": "refused", "code": "unknown-tool", "message": name}
        if name == "workspace.list":
            return self.tool_list(arguments)
        if name == "workspace.read":
            return self.tool_read(arguments)
        if name == "workspace.search":
            return self.tool_search(arguments)
        if name == "workspace.frontier":
            return {"status": "ok", **self.boundary.frontier()}
        if name == "workspace.authorities":
            return {"status": "ok", **self.boundary.authorities(arguments.get("kind"))}
        if name == "workspace.observe":
            return {"status": "ok", **self.boundary.observe(arguments.get("revision"))}
        if name == "workspace.propose":
            return self.boundary.propose(arguments)
        return {"status": "refused", "code": "unknown-tool", "message": name}

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        if not isinstance(request, Mapping):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
        request_id = request.get("id")
        method = request.get("method")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "serverInfo": {"name": SERVER_ID, "version": SERVER_VERSION},
                    "capabilities": {"tools": {}},
                },
            }
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            tools = [
                {"name": tool, "inputSchema": BASIC_INPUT_SCHEMAS.get(tool, {"type": "object"})}
                for tool in self.tool_names()
            ]
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}
        if method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name", "")
            payload = self.call_tool(str(name), params.get("arguments") or {})
            is_error = payload.get("status") in {"refused"}
            self.logger.record_tool_call(
                tool=str(name),
                disclosure="full" if not is_error else "typed-refusal",
                availability="available",
                content_reference=digest(payload),
                cost_usd=0.0,
            )
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": canonical_json(payload)}],
                    "isError": bool(is_error),
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }

    def serve_stdio(self, stdin: Any = None, stdout: Any = None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = parse_canonical_json(line)
            except ContractViolation:
                stdout.write(canonical_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
                stdout.flush()
                continue
            response = self.handle(request)
            if response is not None:
                stdout.write(canonical_json(response) + "\n")
                stdout.flush()


def build_server(config_data: Mapping[str, Any]) -> ProtectedMcpServer:
    config = McpConfig.from_mapping(config_data)
    assert_only_protected_inventory({"strict_mcp_config": config.strict_mcp_config})
    waves = OwnerWaveAuthority(store_dir=config.owner_wave_store)
    return ProtectedMcpServer(config, owner_waves=waves)


def main(argv: Sequence[str] | None = None) -> int:
    import os

    config_path = os.environ.get("W2C_MCP_CONFIG")
    if not config_path:
        print("W2C_MCP_CONFIG is required", file=sys.stderr)
        return 2
    data = parse_canonical_json(Path(config_path).read_text(encoding="utf-8"))
    server = build_server(data)
    server.serve_stdio()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
