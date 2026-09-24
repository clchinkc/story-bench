# Story-bench experiment contract

Approved protocol, 2026-09-21 Query 2; implementation proceeds through the capsule's dependency-ordered waves. Approval identity and original content hashes are preserved in capsule `2026-09-21-story-evaluation-plan-consolidation/bindings/query2-authorization.json`. This file specifies required behavior, not capabilities already present. The current legacy runner is not qualified for these claims. Strategy and locked author choices: [canonical plan](https://linear.app/firstory/issue/DAILY-105). Operational root: [DAILY-105](https://linear.app/firstory/issue/DAILY-105).

## 1. Baseline and authority boundary

Freeze Narrative Craft 38428703b976bf0a23c750b14a356d4303230ff6 and Story-bench 2722769c79c7aa79a8bb7a90fa9da724974ee5e8 from setup; record any subsequent repair commit/artifact as a new explicit treatment version. Installed package qualification is separate from source tests, public-release authority and consumer parity. P0a consumer blockers are documented in [repair register](repair-register.md). Do not roll back silently or track latest during paired execution.

Native project/plan/Character/Story accepted models, Decision rulings, Narrative Loop and Evaluation records are authoritative. No parallel state engine or shadow outcome store. Use manifest-designated paths and supported public API/CLI, not internal imports or assumptions about obsolete Markdown/machine JSON. Native shared schema changes belong upstream; task, experiment, analysis and adapter code stays here. A benchmark episode manifest records execution/cost provenance and references native evaluations by ID/hash.

## 2. Proposed records and data flow

These are schema requirements to implement and test, not a claim that the current native schemas accept them.

**Task manifest**: task_id, task_family_id, story_id, world_id, source_id and rights record; locale; revision/generation/critic/long-form type; source/artifact hashes; request/brief; relevant accepted context; explicit allowed and protected authority scope; initial state fingerprint; allowed observations/actions; information entitlement; stop/budget limits; split/cluster IDs; protocol version. Private criteria and fixtures live in a grader-only namespace.

**Episode manifest**: globally unique run/episode IDs; task/protocol hash; arm and assigned model snapshot; skill/tool/harness/provider versions; reasoning/temperature/seed where supported; isolated environment ID; initial state fingerprint; chronological action/observation references; tool invocations and rejected actions; proposal/acceptance events and author/simulated-author identities; final artifact hashes; stop reason; token, call, latency, attempt and monetary ledger IDs. Include cache provenance and memory/reset evidence. IDs and deduplication must reject stale or duplicate results with different content.

**Evidence**: criterion_id and version; source/final/trajectory binding; exact span or structured field reference plus content hash; extraction method/version; retrieval query/window and coverage including ending; rationale; uncertainty; contradictions/counterevidence; grader identity/version. A quote must match its bound artifact. Generated evidence cannot silently amend accepted state or private criteria.

**Criterion**: applicability predicate; author/task obligation or quality question; evidence requirements; severity and dependency; satisfied/violated/unknown/not-applicable verdict; rationale, assessor and confidence; human override provenance. Missing fields, malformed types, nonfinite/out-of-range numbers, stale hashes and unknown subtypes are invalid, not clipped, defaulted, coerced to success or called N/A. N/A requires justified inapplicability; unknown indicates insufficient evidence.

**Outcome**: native Evaluation record(s) with registered protocol, typed codecs and content addressing. If current protocols cannot faithfully represent these requirements, change the native contract explicitly in P1 with consumers, codecs, examples and distribution tests. Report projection contains per-criterion diagnostics, terminal resolved status, coverage, authority outcomes, critic metrics and quality pair decisions; it must be recomputable from immutable native outcomes, not independently scored.

Pipeline: task → protected environment → trajectory/final accepted state → evidence → criterion verdicts → native outcome → report/reward. One registered criterion/scoring implementation serves operational and final modes. Final graders run outside the agent sandbox and do not reveal private tests, labels, judge prompts or rewards during held-out evaluation.

## 3. Runtime environment contract

Reset must produce a fresh disposable copy of the task's native workspace, fixed package/artifact bindings, empty agent conversation/memory/caches and clean per-episode credentials. Test that one arm cannot read another's state, holdout, prior labels or capability configuration. Never operate on live manuscripts.

Expose the same basic document MCP in every arm: list/read/search permitted raw files and submit scoped edit proposals through trusted native commit/protection checks. Editing arbitrary accepted files directly is forbidden. Build this adapter over current public CLI/API; installed Narrative Craft does not currently advertise a native MCP server. Basic transport/latency/tools are matched across arms. Specialized tools add named Native Craft reading, proof, planning or critique capabilities with explicit schemas. Their output content, calls and costs are logged.

Action → observation includes success/rejection, scope, immutable state revision and safe error. All accepted mutations pass native authorization and revision checks. Reject missing/wrong/stale/replayed approval, path traversal/symlink escape, scope escalation, partial acceptance, direct shell/filesystem bypass and conflicting concurrent proposal. Rejected actions preserve accepted bytes. A simulated author is an explicitly labeled benchmark fixture/service; it cannot impersonate the real author or expand task scope.

Completion requires an explicit final submission AND valid final accepted state. Stop reasons distinguish completion, agent budget exhaustion, policy rejection, provider/infrastructure failure, grader failure and not-run. Maximum turns, retries, wall time, output tokens and dollars are enforced, not advisory. Truncation is a stop reason and coverage fact. An open ending can be artistically complete; an output cut mid-sentence cannot be promoted for brevity.

Record/replay must reproduce environment state transitions and scoring from frozen artifacts. A fresh model call is a fresh stochastic run, even at temperature zero. Do not promise deterministic model replay.

## 4. Treatment packaging and two tracks

| Arm | Skill instructions | Specialized tools | Basic protected MCP |
|---|---|---|---|
| A | off | off | on |
| B | on | off | on |
| C | off | on | on |
| D | on | on | on |

The currently installed skill tells agents to use CLI. P1 must produce a versioned portable instruction package that expresses the treatment without granting B specialized CLI/source/shell capabilities. Preserve the intended instructional mechanism, document all edits from the installed package, and demonstrate B can finish matched tasks through basic MCP. If that cannot be done faithfully, treatment separability is unresolved and P3 is blocked pending an explicit design amendment; do not call a confounded arm “skill-only.”

Equalize raw source facts, author intent, memory initialization, model/reasoning budget, harness and base tool transport. Tools may organize/derive information as their treatment mechanism, but cannot bundle extra private knowledge. Record tool-output disclosure and availability. Private Obsidian routes, project memories, Ghost Reader feedback and personalized knowledge are excluded from the portable package unless separately named interventions. Probe blocked capabilities through filesystem, source inspection, CLI, environment variables, inherited skills and tool discovery.

**Minimal model track**: fixed supplied input/output protocol and model budget, no hidden agent adaptation. Artifact and critic comparisons isolate model/evaluator behavior.
**Agent-system track**: native proposals, author gates, tool trajectories, feedback/revision and accepted outcomes. Aggregate assigned-arm outcomes (intention-to-treat), not only successful invocations. Report invocation/adherence as diagnostics. Primary D−A; B−A, C−A and D−B−C+A are secondary. One harness and at least two model families initially; freeze an affordable specific roster before paid pilot. P4 adds another harness with locked adapters and held-out tasks.

## 5. Dataset specification and rights

Construct approximately equal-sized en, zh-Hant and zh-Hans strata with equal locale weight; include native Chinese source writing. Record author language/script, translation/conversion lineage, genre, length, difficulty and continuity mode. A translated/script-converted variant is correlated with its source, not a new independent story. Include diverse genres and legitimate style/form exceptions instead of forcing one template.

Use locale-appropriate segmentation and length conventions, recorded with each metric. Inspect genre balance without an exhaustive genre-by-capability matrix; finite cases support tested breadth, not all genres. AI may propose cases, defects, repairs and development labels; the user validates legitimacy, feasibility, intent and human references. Never rewrite the author's unaided source sections to pass a gate.

Split by connected story/world/task-family group BEFORE generating variants, flaws or model outputs. Use train/dev for prompt and reward iteration, calibration for judge tuning/selection and held-out judge checks, and untouched final effect evaluation. Seal identifiers and content hashes; final effect holdout is not browsable by agents/developers tuning the system. Record exposure events and quarantine contamination. Public benchmarks/legacy tasks are development or external diagnostics.

Each source record contains original URL/owner, observed version/date, text hash, license or permission evidence, allowed copying/transformation/training/evaluation/redistribution, privacy/provider restrictions and expiration if any. License of benchmark code does not license included literary texts. Use user-owned or explicitly permissioned work first; uncertain rights block that item/use. Do not bulk-scrape AO3 or reuse private manuscripts with unapproved providers. No assumed human annotations.

Revision item inputs: original text, author request, relevant surrounding context, accepted plan/Character/Story facts, authorization scope, protected material and budget. Private rubric expands only entailed obligations. Record user feasibility judgment and multiple valid repair classes; do not compare prose to one exact answer string except exact-byte invariants. Include:
- targeted correction with minimal collateral change;
- cross-scene/name/relationship/knowledge/causal changes that require propagation;
- already-satisfied requests, clean/no-flaw text, ambiguous and under-specified requests;
- justified abstention/clarification and safe no-op versus a missed necessary change;
- adversarial requests that attempt to alter fixed decisions or evade protected scope.

Continuity fixtures label chronology, location, identity, relationships, knowledge versus belief, causes, world rules and setup/payoff. Evidence distinguishes narration/dialogue claims from world facts, flashbacks/dreams, unreliable narration and explicit authorized change. Cases contain both real errors and attractive false positives.

Initial human annotation proposal: small precomputed calibration batches (e.g. 24–36 task clusters spanning three locales, adjusted to author capacity), with 10–15% delayed blinded repeats. These are workload proposals, not power claims or collected labels. Show full relevant context, randomize pair order, hide arm/model names and AI verdicts; record initial verdict, evidence, confidence, tie/cannot-assess and time. Store optional AI-assisted reconsideration separately after locking the initial judgment. Only the user rates; estimate intra-rater consistency, never inter-rater agreement. Dev disagreement can select later calibration examples; no held-out cherry-picking.

## 6. Artifact, critic and generation assessment

Revision primary resolved predicate is requested changes fulfilled, protected decisions/content preserved, dependencies propagated, no new critical continuity defect and valid final submission. Report each component and over-editing. Quality pairwise judgments include all assessable outputs, including unresolved tasks; otherwise the success filter would bias results. A failed empty/absent artifact has its own status and conservative outcome treatment, not a favorable omission.

Use the original as an additional reference when meaningful. Edit distance is a diagnostic, not a universal minimization target. A persuasive self-explanation, declared plan adherence or many tool calls is not task success; an always-rejecting editor fails valid authorized requests. Include recovery time and cost.

Critic protocol fixes source context, evidence budget, diagnostic schema and severity rubric. Match identified findings to adjudicated defects using spans and semantic identity, avoid double counting, and measure precision/recall, false positive rate on clean cases, severity/order calibration, evidence accuracy, abstention and context coverage. Compare agreement with genuine author judgments by locale and defect type. A fixed reviser receives blinded, budget-matched feedback variants (no critique, candidate critique, reference critique where authored) and is evaluated on final repair/preservation/new errors. This separates plausible diagnosis from causal revision usefulness. Do not inject final grader feedback into held-out episodes.

Generation has two conditions: minimal author brief and explicit author-approved plan. A self-generated outline is evaluated as an intermediate artifact and cannot certify final success. Track legitimate deviation authorized by the task. Evaluate finished stories, author intent, coherence and craft with blinded pairwise ties, reversed order and calibrated judges. Separate completion, length and critical errors from literary preference. Use length-stratified/sensitivity analyses, not destructive truncation to equal size. Slop/repetition/style range are diagnostics, not AI detectors or a global quality axis. No mandatory sensory density, sentiment curve, beat structure or resolution style unless requested.

Review applicable prose/voice, character/dialogue, causal development, emotional effectiveness, originality and ending against intent. Meaningful stylistic variation is different from random incoherence; test homogenization and preservation without rewarding diversity automatically.

Long form: scene, neighboring scenes and whole-story requirements; audit full-context or retrieval coverage including ending and early setup. Include changed decision propagation, temporal/causal links, voice drift, unreliable knowledge, delayed payoff and summary/compaction corruption. Query an evidence graph with chapter/time validity; retrieval recall and memory Q&A diagnose access, while separate full artifact evaluation judges story quality. Normalize error statistics using fixed task obligations and report raw counts/coverage so padding cannot dilute failure.

## 7. Reward and training compatibility

Initial terminal reward is binary resolved success under hard authority and critical-continuity gates. Maintain distinct invalid-infrastructure/grader-invalid statuses; do not train them as task-negative labels. Zero new critical errors must be established with required coverage, not by empty critique.

Later quality scalar requires named anchors, human calibration and protection gates. Pairwise population ranking is not an episode reward. Reuse the same criterion/outcome implementation in reporting and training; adapter maps observations/actions/terminal outcome without changing semantics. Train/dev feedback can expose permitted partial diagnostics; held-out labels, judge prompts and tests remain hidden. Store the feedback visibility profile in every manifest.

No step rewards before adversarial tests against repeatedly breaking/repairing the same fact, issuing unnecessary edits, padding criterion lists/text, deleting constraints, manipulating evidence IDs, claiming actions without state change, exploiting unknown/N/A, skipping difficult tasks, truncating endings and gaming judge preferences. Reserve independent fixed holdout and a held-out grader/attack set. Training proceeds only via supported versioned environment adapter with isolated reset, stop semantics, replay, budget ledger and remaining money. Harbor/Verifiers/Agent Lightning are mechanism references, not commitments to install several frameworks or rewrite the engine.

P5 includes a narrow actual training demonstration when approved and affordable, with held-out improvement beyond increasing training reward and no unacceptable quality, preservation or style-range regression. Adapter tests alone do not complete that empirical gate. Closed inference APIs support evaluation, demonstrations and prompt/harness optimization but are not automatically weight-trainable. Select only a model/framework whose training interface is supported. If funds cannot support demonstration, report P5 blocked rather than call compatibility a successful training result.

## 8. Statistical analysis plan to freeze after pilot

Primary estimand: weighted mean within-cluster paired D−A resolved difference over the preregistered model roster, task distribution and equal locale weights. Preserve arm pairing within model/task; cluster at connected story/world/task-family level with correlated variants/repeats together. Hierarchical paired cluster bootstrap with a fixed seed and preregistered resamples is the proposed interval method; pilot simulations must check behavior at small cluster counts and choose/lock a valid alternative before main data if needed. Model families are fixed strata; generalization beyond them is limited.

Benefit working policy: observed difference ≥0.10 and lower paired 95% bound >0. Quality safeguard: Q=(wins+0.5 ties)/valid assessed pairs, observed Q≥0.50 and lower 95% bound >0.45. Label the first as observed improvement plus evidence of positive effect, not proof of true ≥10-point gain. Label the second as a preference-probability noninferiority margin, not 5% literary degradation. Success requires both, valid authority gate and adequate coverage. Failure or inconclusive outcomes remain publishable internal evidence.

Predeclare pair weighting, missingness, retry limits, valid-assessment definition, coverage threshold, stop criteria and family-wise/secondary inference policy. Proposed handling: agent-caused absent/invalid submission counts unresolved; provider failures remain reported, use fixed retries on all arms, then paired missingness sensitivity; grader-invalid can be regraded from frozen outputs without changing generation. Primary results include full assigned denominator and worst/best-case bounds for unassessable outcomes. Do not report a benefit verdict if missingness sensitivity crosses the gate. One-sided missing quality does not become a tie or get silently dropped.

Report resolved count/rate, quality win/tie/loss/cannot-assess, authority breaches/rejections, new-error severity, coverage and missingness, per-model/locale effects, latency and all-attempt costs. Correct or clearly label multiplicity for secondary B/C/interaction and subgroup exploration; no selecting the most favorable judge or model after results.

Pilot measures variance, failure rates, cluster dependence, assessment burden and actual prices/cost. Simulate power/precision at feasible sample sizes for both primary and quality gates. Freeze sample size only after this evidence; if budget cannot answer the question, narrow scope prospectively with author approval or report infeasibility. Do not relabel pilot as holdout, stop at significance, relax thresholds after inspection or count judges/repeats as independent stories. pass@k means at least one success among k attempts with selection cost; reliability means success across repeated independent runs under a stated model. Report separately.

## 9. Spend and failure accounting

One atomic program ledger, TOTAL incremental cap USD 1,000. Initial allocations: repair 100, data/judge 200, pilot 150, main 400, reserve 150. All later work draws from unspent/reallocated funds within 1,000. No per-phase cap reset.

Human time, existing equipment and already-paid fixed subscriptions are outside the incremental cash cap. Unused allocations are not an obligation to spend. Any reserve use or envelope shift must be explicit before dispatch.

Before dispatch reserve a conservative upper bound for input/output/reasoning tokens, tool/product critic/retrieval calls, graders, retries, compute/infra/training and observable agent usage using provider/model price snapshots with timestamps/currency. Lock reservation transactionally across parallel workers. Refuse a call if reservation would exceed either envelope or total remaining allowance. Unknown prices or unbounded usage block paid dispatch; reserve a conservative bound for unobservable session charges rather than assuming zero. Reconcile invoice/usage on response; unresolved charges remain reserved. Failed/refused/timed-out calls may cost money and remain ledger entries.

Distinguish task failure, absent output, provider failure, infrastructure failure, grader failure and not-run. Report total deployment cost (agent+operational support+retries+selection) separately from research grading cost. Use paired common task sets and fixed coverage for cost/quality comparisons; also report full assigned costs. Replace Score²/Cost ranking with outcome-versus-cost/latency Pareto views and cost per resolved task with uncertainty and zero-success handling. Cheap incomplete output never wins by denominator omission.

## 10. Verification and historical disposition

See [repair register](repair-register.md) for frozen source evidence and meaningful planned regressions. Each future test must fail on a deliberately wrong behavior, not only mirror implementation or check schema presence. P0 requires repaired or visibly quarantined active paths, not a blanket validation claim. Calibration remains an additional P2 gate.

Raw historic results are immutable evidence. Version corrected reports: reaggregate valid records; rejudge when judge input/protocol was invalid or truncated; regenerate affected generation/oracle episodes. Preserve original protocol, costs, dates and artifacts. Old 50/50/median/lexical scores and coverage-biased rankings are not literary ground truth or comparable to this protocol.

W1's enabled offline measurement path is implemented and independently accepted: 114 exact regression IDs pass, including all 101 initial cases and 13 repair cases. Parent rerun agrees; capsule `bindings/execution/w1/acceptance.json` binds evidence and limits. Fixed assigned coverage, supplied-attempt known/unknown costs, strict diagnostic/context/identity validation and historical disposition are qualified locally. Whole legacy live provider/evaluator execution and semantic oracle remain disabled/unqualified; later environment, spend, dataset, judge and empirical test names pass only under their own later gates. No benchmark sweep, paid API call, human annotation or empirical benefit is established by this documentation contract.

W3a offline spend accounting is now independently accepted (`bindings/execution/w3a/acceptance.json`; [API and trust boundary](spend-ledger.md)). The full 223-case suite retains all 114 W1 IDs; fresh repair verification adds 21 independent probes and closes the extreme-timestamp LedgerError defect. Prior code/security concurrency, crash, arithmetic and four mutation controls are preserved. W3b provider/price/attempt integration and host protection remain required: synthetic receipts are not invoices, the caller must obey fresh-marker-before-send, and no actual hosted charge has been reconciled. Direct paid spend/reservations USD 0; hosted charges unknown; no paid allowance released. This does not complete P1 or qualify an experiment.

W3b1 local offline provider attempts are accepted at source checkpoint `ac6747af1f81ca77b33d63bbe366f1c85643c25b` (`bindings/execution/w3b/acceptance.json`; [supported API and limits](provider-attempts.md)). Producer, different fresh repair verifier and parent each pass 397 exact test IDs; all 77 original independent probes pass after nine preserved r1 failures, 52 distinct fresh probes pass, and 17 unsafe variants are detected. Five implementation findings are closed: atomic retry slots with verified ancestry, complete usage unknowns, typed malformed-response failures, all-component file confinement and truthful postcommit accounting. The adapter executes only finite cooperative local fixtures. Current genuine route price/limit/finality/funding proof and real transport admission remain W3b2 requirements; actual rights, hosted billing and later phase/human gates remain unsatisfied. No old live route, real program store, paid allowance or native resolved outcome is enabled. W3 and P1 remain incomplete.
