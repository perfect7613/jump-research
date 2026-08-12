# JUMP — Revised Research Plan

Revision of `1786528309242-JUMP_COMPLETE_RESEARCH_PLAN.md`, incorporating the red-team findings in `JUMP_RISK_REVIEW.md` (R1–R23, A1–A12) and the validation findings in `JUMP_VALIDATION_REVIEW.md` (flaws 1–5, sharper plan items 1–8). Date: 2026-08-13.

## Changelog

Material changes from the original plan, with review provenance:

1. **Split into two committed tracks.** The single document that fused a 28–50 day research program with a 2-day hackathon (Risk R1, Validation flaw 1) is now explicitly two tracks: **Track H** (hackathon, ≤48h, ≤$90) and **Track R** (research, 4–7 weeks post-event). No Track R claim may appear in Track H materials.
2. **Descoped the product surface ~70%** (Validation "Scope"): the domain router, adapter registry, uploaded-tabular adapter, statistical-hypothesis adapter, file/image uploads (R9), and most of the clarification policy are cut from Track H. Track H ships one synthetic-2D adapter, three canned conversations, one live path, the swap toggle, and provenance labels.
3. **Added baseline C′ (simulator-as-tool, text results)** as a preregistered principal comparison alongside E vs G and E vs I (Validation flaw 3). If C′ ≈ E, the contribution is repositioned as engineering + benchmark, not mechanism.
4. **Pinned the adapter training objective in Phase 0** and added a training-signal control (dynamics model fit to worlds without hidden-variable structure, or permuted episode–latent pairs) to de-confound treatment E (Validation flaw 2).
5. **Made the `replacement_law` answer field categorical/DSL-enumerated** (`{same: attract|repel, different: attract|repel, exponent: n}`) so no LLM judge or fragile parser is needed (Validation flaw 5).
6. **Inverted demo priority:** matched A/B latent-swap demo is now the must-have; Hidden Law Discovery is the stretch goal (R19, R16). The video segment formerly titled "causal proof" is renamed "swap demonstration."
7. **Cached-first serving.** Precomputed sealed conversations are the default UX; live GPU runs sit behind a rate limiter and global spend cap (R5, R20). The latent-hash provenance invariant is enforced in CI (R18).
8. **Budget restructured:** Track H operating ceiling $90; serving-line conflict resolved by funding one warm container for judging day (~$23) from the reserve via written decision (R20); 80%-of-ceiling forecast rule triggers scope reduction before launch (Validation "Budget").
9. **Compliance additions:** Gemma license pass-through check before any checkpoint release (R12), transient-only conversation handling with a privacy notice (R11), citation URL verification script before README ship (R14), zero pre-window code (R15).
10. **Checkpoint change:** the chat surface uses `gemma-4-12B-it`; hook points must be verified to survive quantization before serving budget is committed (Validation item 6).
11. **Statistical power note added:** paired design, ≥200 episodes/condition pilot detects ~10–12pp at 95% CI; full runs sized at ≥500/condition; chance/shortcut baselines reported next to every accuracy number (Validation items 5 and "chance levels").

## Executive Summary

JUMP remains what the original plan defined: a controlled model organism for mechanistically studying theory revision — testing whether an executable counterfactual world representation causes a multimodal language model to reject an inadequate law and promote a hidden-variable explanation — wrapped in a conversational scientific thought-experiment product.

Both independent reviews reached the same verdict: the scientific core (six-object hidden-type world, four task conditions, A–J baseline matrix, exact scoring without an LLM judge, phase gates, budget ledger, truthfulness rules) is unusually strong; the structure around it is broken in one dominant way. The original plan binds a multi-week gated research program to a 2-day, $188 hackathon and frames a single toy 2D adapter as a general "ask any scientific question" assistant. Nearly every other identified risk — budget overrun, demo fragility, overclaiming, denial-of-wallet, truthfulness commitments collapsing under deadline pressure — is downstream of that compression.

This revision accepts the reviews' joint prescription. The hackathon ships a **benchmark and an instrument**: the model organism, the vertical latent-injection slice, the swap demonstration, and an honest, scoped, cached-first chat Space. The **mechanism study** — localization, causal surgery, mediation — moves entirely post-event onto its own timeline, with strengthened controls (C′ tool-use baseline, adapter training-signal control, required cross-checkpoint replication for any mediation claim). The overall posture is the risk review's CONDITIONAL GO: go for the vertical slice with honest labeling; no-go for any public causal or mechanistic claim inside the hackathon window.

## MVP Definition

### Track H MVP (hackathon deliverable, submission floor)

The Space must be fully demonstrable against recorded result objects with zero live backend (R21). The submission floor, in priority order:

1. **Six-object 2D simulator + deterministic renderer** with hidden binary type, sign-flip force rule, randomized nuisance factors, and unit-tested ground truth.
2. **Frozen-Gemma behavioral baselines A/B/C/C′** on a pilot set, run before any adapter code (Validation item 2) — the cheapest experiment, and the one that determines whether the architecture is worth building.
3. **One trained latent dynamics module + one adapter injection path** (Gemma frozen, adapter-only training with a pinned objective).
4. **The matched World A/World B latent-swap demo** with the swap toggle — the load-bearing demo (R19).
5. **HF Space chat UI** scoped by a visible banner ("JUMP currently executes thought experiments only in synthetic 2D dynamics"), three flagship canned conversations served instantly from sealed cache artifacts, one rate-limited live path, provenance badges backed by latent-tensor hashes, and honest cached-run labels.
6. **Released artifacts:** benchmark data, evaluation scripts, reproducibility notebook, demo video scripted with the plan's own non-claims language.

Explicitly *not* in the MVP: Hidden Law Discovery live (stretch only), latent-only readout (stretch), domain router/registry, uploads, multi-turn investigation memory beyond session scope, any mediation/localization result, any Gemma-derived checkpoint release before license verification.

### Track R MVP (minimum viable research result, post-event)

Unchanged in substance from original §12, with the new controls: clean four-condition benchmark; learned predictive latent containing hidden interaction structure; Gemma+latent beating scrambled/random, same-parameter (I), *and* no-hidden-structure training-signal controls; wrong-world latent shifting the inferred partition in the predicted direction; and **E vs C′ reported prominently even if unflattering** (R17).

## Architecture

The core pipeline is retained from original §5, with revisions marked:

```text
rendered observations
      |
      v
multimodal Gemma (gemma-4-12B-it for chat; base for research runs)
      |                                  \
      |                                   ordinary reasoning baseline (A/B/C)
      v                                   simulator-as-tool text baseline (C′)  [NEW]
bottleneck -> latent dynamics module (world model)
      |
      v
imagined world latent Z --(trainable gated adapter, pinned objective)--> residual-stream injection (E)
      |
      v
diagnostic image decoder (same Z; hash-stamped)   ->  provenance badge = hash match, not assertion
```

Key architectural decisions this revision fixes or adds:

- **Adapter objective pinned in Phase 0** (the largest unspecified degree of freedom): next-token loss on neutral description text, or task loss with explicit disclosure in all write-ups. A companion control trains the identical adapter on latents from a dynamics model fit to worlds *without* hidden-variable structure.
- **Provenance invariant as code, not prose:** hash the latent tensor; stamp the hash into the decoder call and the answer manifest; cache entries are sealed artifacts (latent + image + answer + hash) regression-tested in CI (R18).
- **Compiler demoted from critical path:** the three flagship conversations use template matching with slot filling; the general `ScientificQueryPlan` compiler is best-effort behind them, with every compiled plan logged and surfaced in the "experiment JUMP constructed" card (R3).
- **Executor-side safety:** DSL parsed with a real grammar (never eval/exec); hard caps on objects, dims, coefficient magnitude, exponent range, timesteps, wall time, and memory enforced Modal-side; function-level timeout+kill as backstop; compiler→executor path fuzzed before going public (R7).
- **Serving:** cached-first; anonymous visitors get precomputed conversations with a "run live" button behind per-IP/session rate limits plus a global Modal-side GPU-second cap (R5). Session-scoped in-memory conversation state only; no server persistence (R11).

## Hypotheses

Preregistered, in order of evidentiary ambition (Track H tests only H1–H2; H3–H5 are Track R):

- **H1 (behavioral regime):** A difficulty regime exists in which frozen Gemma succeeds on 20–70% of abductive-condition episodes. *Risk:* R19/Validation flaw 4 — the regime may be narrow or absent; the pivot table (ontology-supplied condition first) applies, and chance level (31 partitions, 4 sign-rule variants) is reported beside every number.
- **H2 (injection transfer):** Injecting the world latent (E) shifts answers relative to matched scrambled/random latents (G), and a World A↔World B swap shifts the inferred partition in the predicted direction. This is Track H's headline, framed as a *swap demonstration*, not causal proof.
- **H3 (beyond cheap alternatives):** E outperforms C′ (simulator counterfactuals pasted as text, matched token budget) and the same-parameter control I. If E ≈ C′, the honest conclusion is that latent injection is an engineering convenience, and the paper repositions around the benchmark.
- **H4 (inadequacy state):** A separable model-inadequacy representation appears after prediction mismatch and before verbalization; ablating it increases persistence with the falsified law; injecting it promotes revision without specifying the replacement.
- **H5 (mediation):** The chain counterfactual representation → inadequacy trigger → promotion → new hypothesis survives mediation analysis with specificity controls, **replicated on a second checkpoint as a requirement, not an option** (Validation "mediation").

## Experiments & Metrics

### Condition matrix

Original A–J retained, plus: **C′** = frozen Gemma + simulator counterfactual outputs as text (matched token budget); **training-signal control** = adapter trained on no-hidden-structure latents or permuted episode–latent pairs. Preregistered principal comparisons: **E vs G, E vs C′, E vs I**, with B vs E reported prominently.

### Behavioral metrics (exact-scored, no LLM judge)

- Partition accuracy (up to label swap) against 31-partition chance.
- Law identification accuracy on the categorical DSL schema `{same: attract|repel, different: attract|repel, exponent: n}`.
- Held-out force prediction error; adequacy-flag correctness; false-abduction penalty (inventing types in the control condition); confidence calibration.
- Power: pilot 200 episodes/condition (paired design detects ~10–12pp at 95% CI); full preregistered runs ≥500/condition.

### World-model metrics

Held-out multi-step prediction error; decodability of hidden pair relations from the latent beyond static appearance (probe with held-out splits).

### Product metrics (Track H, reduced from ~13 to 4 per Validation item 7)

Valid compilation rate, tool execution success rate, counterfactual answer accuracy on the flagship paths, and provenance-labeling correctness (hash-match rate = 100% required). All other §6 metrics move to a post-hackathon appendix marked aspirational.

### Mechanistic metrics (Track R only)

Probe AUC for mismatch/inadequacy/partition subspaces at T0–T4; intervention effect sizes with matched-norm, orthogonal, and generic-error controls; mediation proportion with specificity checks (surprise/uncertainty/negation) and mandatory second-checkpoint replication.

## Risk Mitigations

Consolidated register of accepted mitigations (IDs from the risk review):

| Risk | Mitigation adopted |
|---|---|
| R1 timeline contradiction | Two-track split; hackathon = Phase 0 + thin slice of 1–3 + swap demo; Phases 4–7 publicly declared post-event |
| R2 promise/capability gap | Visible scope banner; refusal path for out-of-domain questions is the most-tested path; prompt chips limited to the three flagship demos |
| R3 compiler on critical path | Templated flagship paths; general compiler best-effort; all plans logged and surfaced |
| R4 rerun underbudgeting | 1%-scale smoke test on T4/L4 before any A100 launch; hard allowlist of extraction layers/timepoints; ledger-enforcing launch wrapper; localization/causal phases removed from paid window |
| R5 denial-of-wallet | Cached-first default; per-IP/session rate limit; global Modal-side spend cap; demo video independent of live infra |
| R6 credit terms unverified | $1 smoke test verifying balance, expiry, GPU-class eligibility, budget-cap behavior in preflight |
| R7 NL→DSL injection | Grammar parser, executor-side hard caps, Modal-level timeout/kill, adversarial fuzzing pre-launch |
| R8 secrets | gitleaks/trufflehog in CI on every push; minimum-scope per-purpose tokens; traceback-suppressing error wrapper; token rotation post-event |
| R9 uploads | Uploads disabled entirely for Track H |
| R10 supply chain | Lockfile from Day 1; sync action pinned by SHA and restricted to protected branch |
| R11 privacy | No server persistence; transient processing notice in About panel; logging opt-in actually implemented |
| R12 Gemma license | License read in preflight; pass-through terms in every model card; no derivative release before verification |
| R13 liability/dual-use | "Research demo; not scientific, medical, or safety advice" disclaimer; refusal templates; chem/bio adapters removed from public roadmap |
| R14 citations | Scripted URL/arXiv resolution check before README ships; unverifiable references dropped or caveated (R-lens removed from deliverables until sourced) |
| R15 rules compliance | Zero pre-window code; prep artifacts are documents only; rules confirmed in announcements channel; honest commit history |
| R16 overclaim drift | "Causal proof" language banned from Track H; demo script uses non-claims language; single truthfulness owner with authority to block the recording |
| R17/flaw 2 confounds | C′ baseline, training-signal control, OOD law family elevated to required for any causal statement |
| R18 provenance | Latent-hash stamping, sealed cache artifacts, CI regression test |
| R20 cold start | Warm container funded for judging day by written reserve decision; instant cached responses with honest "warming live backend…" status |
| R21 integration risk | Space fully demo-able against recorded result objects; Day 1 known-good tag = submission floor |
| R22 deadline SPOFs | Named submission owner; dry-run rehearsal; main frozen at T-3h; direct-push-to-Space backup tested |
| R23 team assumptions | Roles collapsed (evaluation→model, submission→product); AO rehearsal in preflight; pre-agreed cut list for reduced headcount |

## Phases & Timeline

### Track H — hackathon (48 hours)

**Preflight (non-build, documents only):** verify credits ($1 smoke test), Gemma license and gated-access mechanics, hackathon rules, AO rehearsal, pin lockfile plan, write demo script with non-claims language.

**Day 1 — floor first:** simulator + renderer + unit tests (CPU); pilot dataset; frozen baselines A/B/C/C′ *before any adapter code*; Space deployed against mocked result objects; Modal endpoint auth path proven; latency measured; end of day = tagged known-good submission floor.

**Day 2 — one rung at a time:** dynamics module + adapter (pinned objective); swap demonstration (must-have); seal flagship cache artifacts with hashes; record demo video; stretch only if green: Hidden Law Discovery, latent-only readout. T-3h: freeze main, submission owner executes rehearsed checklist.

### Track R — research program (post-event, 4–7 weeks)

Phases 0–7 as originally gated, with revisions: Phase 0 additionally pins the adapter objective and categorical law schema; Phase 1 runs on its own budget with the 20–70% regime gate; Phase 3 adds C′ and the training-signal control to the matrix; Phase 5 retains the "serious research claim" gate (preregistered intervention surviving matched controls); Phase 6 makes second-checkpoint replication mandatory for any mediation claim; Phase 7 write-up includes null results, B vs E, and E vs C′ regardless of outcome, plus one paragraph positioning latent injection against tool-use and trace-fine-tuning alternatives (Validation "opportunity cost").

## Budget Assumptions

- **Total credits: $188** on Modal, assumed valid, non-expiring through the event, all GPU classes eligible — *verified by preflight smoke test, not assumed* (R6/A3).
- **Track H operating ceiling: $90** (Validation estimate $60–90), roughly: simulator/data CPU $5; container builds/cache $10; world module + decoder $20; adapter training + sweep $20; baselines A/B/C/C′ + evaluation $12; serving $23 (one warm L40S ≈ $1.95/hr × 12h judging day, funded from reserve by written decision, resolving the original $15 serving-line conflict).
- **Debug allowance treated as elastic, not 10%:** every job smoke-tested at 1% scale on T4/L4 first; any phase forecast exceeding 80% of its ceiling triggers scope reduction *before* launch.
- **Track R spends the remainder post-event** under the original per-phase gates; activation extraction restricted to preregistered layers/timepoints via a code-level allowlist because full 48-layer × T0–T4 extraction on a 12B model would strain the $25 line (Validation "Budget").
- Enforcement retained from the original: workspace hard budget $175, environment budget ~$165, ordinary experimentation stops at $160 internal ledger, apps tagged `project=jump` with phase tags, billing treated as delayed.
- GPU policy retained: CPU for data/scoring; L4/A10 for small modules; L40S default for Gemma (gemma-4-12B ≈ 12B bf16 fits 48GB with activation headroom per validation); A100-80GB only on measured memory need; H100 only for wall-clock emergencies.

## Go/No-Go Gates

**Overall posture: CONDITIONAL GO** (per risk review §9).

Track H gates:

- **G-H1 (preflight):** credits verified, license read, rules confirmed, checkpoint accessible from Modal. Fail → re-plan before any build hour is spent.
- **G-H2 (Day 1 noon):** baselines A/B/C/C′ produce scoreable pilot results. If C/C′ saturate, pivot the claim to "how injected state is used" and re-weight demos now, not on Day 2.
- **G-H3 (Day 1 end):** Space demo-able on cached result objects; known-good tag exists. Fail → Day 2 becomes UI/cache polish only; no live path ships.
- **G-H4 (Day 2 midday):** swap demonstration reproduces on ≥2 seeds with hash-verified provenance. Fail → ship benchmark + baselines + honest "injection did not yet transfer" narrative; the truthfulness owner may block the demo recording.
- **Hard NO-GO items (do not ship in the window):** any public causal/mechanistic claim; live GPU chat without rate limiting and a global spend cap; general-assistant framing without the scope banner; Gemma-derivative checkpoint release before license confirmation; server-side persistence of user conversations.

Track R gates: the original Phase 1–6 gates stand unchanged, with two hardenings — no "the latent causes X" statement without the OOD law-family evaluation passing, and no mediation claim without second-checkpoint replication.

**Standing decision rule** (retained from original §15): each gate is pass/pivot/kill; the reserve is spent only by written decision; and language in every public artifact is bounded by the weakest gate actually passed.
