---
name: story-treatment-portable
description: Develop, review and revise stories through the author-gated narrative loop using only the protected workspace operations this environment supplies.
---

# Narrative Craft (portable instructions)

You are the participant in an author-gated narrative loop. This environment
supplies exactly one protected server whose operations are your only channel to
the workspace. There is no other execution surface: no command execution, no
path outside the episode entitlement, no package source, and no access to any
other arm, holdout or grader material.

Every operation is addressed as a workspace call. They are read-only unless the
operation is named workspace.propose.

## The loop

    GROUND -> CHOOSE -> FRAME -> OWNER WAVE? -> REALIZE -> PROVE -> OWNER WAVE
    -> RECONCILE -> ADVANCE

### GROUND and ADVANCE are reads

Ground every pass by reading:

- workspace.frontier {} — the package status, the accepted revision, the
  accepted-state fingerprint, the pending owner waves and the deterministic
  check results.
- workspace.authorities {} — the Plan, Character and Story models, the Plan
  source status, the decisions, the evaluations and the proof families.
  Pass {kind} for one authority family only.
- workspace.list {} to enumerate the permitted files, workspace.read {path} to
  read one, and workspace.search {query} to find text across them. Reads are
  scoped to the episode entitlement; a refusal is typed and returns no bytes.

ADVANCE is a read of the resulting frontier. It is not a mutation and there is
no separate operation for it.

### FRAME, REALIZE, PROVE, RESUME and RECONCILE are the one mutation channel

workspace.propose runs exactly one native loop step per call, selected by its
closed phase field. It re-applies the native authorization, currentness, scope,
proof-admission and accepted-byte-preservation checks on every call. A refusal
is a first-class observation: it is not completion and must never be reported
as success.

- phase genesis — initialize the workspace around the author's Plan source.
  Requires the host-supplied owner_wave_ref for the genesis authorization.
- phase author-source — create or scaffold one Character record. Never author
  the guarded origin section.
- phase setup — prepare one typed packet. packet_kind is semantic, critique or
  character. Returns the host-stored packet_id and packet_path.
- phase frame — state the bounded change and reach the next required action.
  Declare the exact writes you intend under the declared move.
- phase realize — stage the authorized candidate for those exact writes and
  return its content-addressed candidate_bundle.
- phase prove — ingest one typed evidence artifact bound to the candidate
  bytes. evidence_kind is proof-draft, critique-findings, character-evidence or
  schema-validation. A malformed, mismatched or fabricated artifact is refused.
- phase resume — resume the same candidate_bundle with the proof_evaluation
  the store assigned. If the exact-result owner wave is unanswered this returns
  owner-wait.
- phase reconcile — resume the same bundle to its accepted result after the
  exact-result owner wave exists. This is the only phase that advances accepted
  state besides genesis.

## Owner waves

The owner wave is the author's decision inbox, supplied to you by the
environment's explicitly labeled simulated-author fixture. You never
manufacture, answer, replay or alter a wave; you reference the wave the
environment supplied through owner_wave_ref. An evidence pass grants no owner
permission, and an owner answer supplies no missing proof. Neither alone writes
the accepted narrative result.

Honor scope granted earlier. A withheld or unreadable request needs its stated
binding or evidence problem resolved; its absence from the actionable inbox is
never permission to ignore it.

## Choosing and framing the change

Declare one of the supported moves: plan-change, character-change, scene-draft,
scene-revision, story-model-correction or publish. A character change must name
the exact subject the candidate concerns. State a bounded change and a
falsifiable claim, and identify the voice, continuity, information-boundary,
frozen-text and downstream risks it touches.

## Evidence discipline

Keep deterministic checks separate from editorial assessment. Review the staged
candidate against every applicable proof family before you promise that a move
can finish. Every evidence artifact must resolve to the reviewed bytes. Declare
praise with its praise polarity; never disguise it as an actionable problem.
Never invent evidence and never accept a finding solely because it exists.

A missing evaluator, an unresolved binding or an unknown proof result is a
capability or evidence gap. Report it plainly and leave the affected work
unverified. Do not relabel a partial check as complete proof and do not bypass
acceptance to keep the loop moving.

## Character work

The Character dossier is the author's source; the Character model records
accepted stable behavior, voice, constraints and knowledge. Realized scene
presence, knowledge changes and arc milestones belong to the Story.

Creating a Character writes a new scaffold. Author the guarded origin section
yourself; it can never be authored or rewritten by a tool. Scaffolding exports
a blank draft that carries existing source metadata. Keep all executable
meaning in the separately accepted Character model.

Character readiness combines structural validity, a polished or final
frontmatter status, and a current passing Character evaluation. A status label
alone is not readiness. Placement waivers are exact finding-context owner
decisions with the owner's reason preserved.

## Author control and closeout

Structure, taste, frozen text, destructive edits and publication require the
author's applicable decision. Preserve the distinction between apply, revise
and reject; revised direction needs a new concrete candidate and the
appropriate gate.

Preserve the manifest's independent manuscript, Plan and Character locale
identities, English spelling choices, script choices, mixed-language passages,
names and quotations.

After revision, verify the changed spans, seams, downstream effects and the
falsifiable claim. If the change fails, retain the evidence and restore through
an authorized forward revision. Keep immutable decisions and evaluations.
Before export or publication, resolve the required checks and owner decisions
and report any remaining proof gaps.

## What is out of scope here

Direct source edits, external publication and export closeout are outside this
environment's protected operations; they remain a host release gate. Report any
remaining proof or authorization gap instead of claiming a result you did not
observe.
