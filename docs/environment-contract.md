# W2c environment contract (Story-bench)

Status: implementation of the verified contract at
bindings/execution/w2c/contract-repair-2/. This document is the operational
W2c contract and the honest trust boundary. It does not edit the
authorization-bound docs/experiment-contract.md.

## Arms

| Arm | Instruction treatment | Specialized tools | Basic protected MCP |
|---|---|---|---|
| A | off | off | on |
| B | on (portable package v1) | off | on |
| C | off | on | on |
| D | on (portable package v1) | on | on |

The instruction treatment is delivered by the declared host-side
treatment-packet channel, never by settings, hooks, permissions or inherited
skills. The basic protected MCP server identity, version, tool-name set, input
schemas, transport and equal numeric latency budget are identical in A/B/C/D.

## Harness and isolation flags

    claude -p --tools "" --setting-sources "" \
      --strict-mcp-config --mcp-config <protected-mcp-config.json> \
      --no-session-persistence --model <model-id>

The frozen flag set follows the accepted ADR-023 D015 zero-tool precedent
exactly. `--setting-sources ""` loads no user, project or local settings, no
hooks, no permissions and no skills. `--strict-mcp-config` with the explicit
protected config forbids every MCP server except the basic protected one; the
ambient untracked story-bench/.mcp.json (a network server) is never loaded.

## Project-setting-source census (PS1)

`reset` refuses any authorized source snapshot that carries a `.claude/` root at
all - any `.claude/` directory or entry beneath one, including settings.json,
settings.local.json, hooks/, skills/, commands/, README, or any other `.claude/`
entry - and independently refuses an ambient `.mcp.json`. It never strips and
continues. The refusal set is exactly the census root plus the ambient MCP file,
so the precondition entails the zero-entry postcondition.

## Participant cwd and child environment

`participant_cwd()` is a freshly created, empty temporary directory outside both
the source snapshot and the destination workspace, removed after the child
exits. `participant_child_env(cwd)` is a strip-list, deliberately not an
allowlist: `os.environ` minus `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV` and
every `NC_EVAL_*` key, with `PWD` set to the temp cwd and `HOME` preserved
(keychain auth survives). W2c supersedes the accepted host-runner precedent's
untouched `PATH` and provider-key variables only: `PATH` is reduced to the
directory of the pinned executable, and provider-key variables are removed.

The destination-internal-cwd leak is closed by construction: `--setting-sources
""` excludes settings files, but project memory (`CLAUDE.md`/`AGENTS.md`) is
read from the working directory rather than a settings source. A cwd inside the
byte-equal destination could therefore hand over inherited memory through a
settings-free channel the `.claude/` census does not enumerate. The fresh empty
temp cwd removes that channel independent of target-build behavior. Whether the
target build loads such memory under `--setting-sources ""` cannot be settled
read-only in W2c.

## Environment ID

`environment_id = SHA256(canonical_json(ENVIRONMENT_ID_RECORD))` and the record
has exactly 16 members: schema_version, task_id, task_family_id, episode_id,
arm, workspace_instance_id, initial_state_fingerprint, source_snapshot_digest,
skill_package_digest, tool_manifest_digest, native_binding_digest,
model_snapshot, harness_version, protocol_version,
information_entitlement_digest, memory_init_digest. Every member is required.
`skill_package_digest` may be null when the arm's instruction treatment is off.
The workspace instance id, and therefore the environment id, is fresh on every
reset: the postcondition is recompute-equals-recorded within one reset, never
across resets. The trajectory digest is an independent replay-refusal input and
is not a member of the 16-member record.

## Reset, replay and rejection

Reset produces a fresh disposable copy of the source snapshot (byte-equal file
census, no exclusion of any kind), fixed package/artifact bindings, an empty
conversation, empty memory, empty caches and a unique per-episode credential
that is revoked at teardown. Conversation, memory, caches and credentials live
in a host run directory outside the censused destination, so the destination
census equals the authorized source snapshot for every arm. Teardown discards
only its own destination and never modifies the source. Replay reproduces the
recorded observations from frozen artifacts with zero model calls and refuses
when any of the 16 members or the frozen trajectory digest changed. A duplicate
episode identity with identical content is idempotent; different content is
refused with accepted bytes preserved. The single mutation channel is
`workspace.propose` with the closed phase enum genesis, author-source, setup,
frame, realize, prove, resume, reconcile. ADVANCE is a read of
`workspace.observe`/`workspace.frontier`, not a phase.

## Protected by W2c (tier E, tested)

- disposable, destination-only workspace lifecycle; source bytes preserved;
  treatment overlay outside the censused snapshot;
- empty conversation/memory/caches and unique per-episode credentials;
- no `.claude/` root (absent-or-empty project setting source) for every arm;
- entitlement-scoped reads and the single phase-scoped proposal channel;
- native protection/authority checks with accepted-byte preservation on
  rejection;
- arm-flag gating and identical basic tool inventories;
- the constructed isolation flags: `--tools ""`, `--setting-sources ""`,
  `--strict-mcp-config` with an explicit config, `--no-session-persistence`, no
  provider key and no native executable on the participant PATH;
- driver-side blocked-capability probes across filesystem, source inspection,
  tool execution, environment variables, inherited settings and tool discovery,
  each with its red control;
- native binding pinning and refusal of implicit PATH/PYTHONPATH discovery.

## Constructed or asserted, not behaviorally observed in W2c

- the resolved argv/config byte set and the resolved MCP server inventory.

## Planned and not tested - UNRESOLVED tier-M release gate G1

Whether `--tools ""` with a NON-EMPTY `--mcp-config` leaves the protected MCP
reachable while removing built-ins is NOT established. ADR-023 used an empty MCP
config, so its precedent does not settle the non-empty case. No model call is
permitted in W2c, so no behavioral arm-capability claim is made. The release
condition is: a bounded harness probe, run outside W2c's no-model restriction,
resolves either (i) the protected MCP is reachable under `--tools ""` - the
treatment channel is valid; or (ii) it is not - every arm loses its only channel
and the design must be amended. Until then no model-backed study may start and
no separability may be published.

## Assumed about the host (not proven here)

- the host is trusted and quiescent: it mounts the intended workspace, hands
  over no trusted fixture context or bootstrap capability, keeps
  holdout/labels/other-arm directories away from the participant, and places no
  secret in the participant environment;
- W2c is runtime discipline plus explicit denial, NOT a kernel sandbox. A
  participant that escapes its process, or a malicious host that can rewrite
  code and trusted files, defeats this boundary. Zero observed breaches is a
  tested release requirement, not proof of universal impossibility, and
  dataclasses, hashes or config files do not authenticate a caller;
- no claim is made about model behavior, literary calibration, human
  annotation, provider transport or empirical benefit.

## Arm-B package and separability

The versioned portable package at src/arm_b_package/ (arm-b-v1) is loaded through
importlib.resources from the anchor src/arm_b_package; manifest.json is closed,
digest-bound and carries a per-edit derivation log whose capabilityDelta objects
pass the closed acceptance rule. The verdict is SEPARABLE_BY_CONTRACT: the
contract-level obstacle is removed by the phase model and the complete
derivation mapping. It is NOT demonstrated: separability becomes an observed
result only when the matched task reaches a native accepted transition through
the basic MCP with the specialized namespace absent. No separability claim is
published here.
