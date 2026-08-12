# JUMP Mechanistic Theory-Revision Research — PRD

- **Status:** Proposed
- **Owner:** JUMP PRD owner
- **Posture:** Conditional go
- **Scope:** One Track H benchmark/instrument and one post-event Track R mechanism study

This is the primary implementation PRD. The accompanying JSON Schema is the only separate planning artifact because the experiment contract must be machine-executable. The revised research plan and its risk/validation reviews remain supporting source material.

## Problem Statement

JUMP tests whether an executable counterfactual world representation causes a frozen multimodal language model to reject an inadequate law and promote a hidden-variable replacement. Behavioral improvement or a decoded latent is insufficient evidence: gains may come from task-specific adapter training, parameter count, textual simulator output, generic error/surprise, or distribution-specific fitting. The project needs a reproducible, affordable path from benchmark behavior to localized representations, controlled causal interventions, mediation, OOD transfer, and replication.

The 48-hour product must remain a narrow synthetic-2D benchmark and instrument. It may demonstrate latent swaps but may not claim a mechanism. Mechanistic claims belong exclusively to the gated 4–7 week Track R program.

## Solution

1. **Track H — benchmark/instrument:** deterministic six-object simulator, exact A/B/C/C′ baselines, one frozen-model latent adapter, matched World A/B swap demonstration, and a cached-first scoped UI. It ships no causal claim.
2. **Track R — mechanism study:** an immutable manifest drives sequential, resumable localization, probing, matched swaps, ablation/injection, controls, mediation, OOD evaluation, and second-checkpoint replication.
3. **Claim policy:** public language is bounded by the weakest gate passed. A mechanism claim requires behavioral specificity, controlled intervention, mediation, OOD transfer, and replication.

## User Stories

1. As a research lead, I want immutable hypotheses, splits, layers, timepoints, metrics, and gates, so confirmatory choices cannot drift.
2. As an operator, I want cost forecasts, hard caps, sequential GPU stages, and deterministic resume, so failures do not cause uncontrolled spend or duplicate data.
3. As a model researcher, I want mismatch, inadequacy, and promotion localized at event-aligned activations, so temporal claims are testable.
4. As a causal researcher, I want matched swaps, ablation, and activation injection, so necessity and sufficiency are tested directly.
5. As a reviewer, I want matched-norm, orthogonal, generic-error, and sham controls, so arbitrary perturbation cannot explain an effect.
6. As a reviewer, I want C′ tool use, same-parameter, no-hidden-training, and permuted-latent controls, so cheap alternatives are not ignored.
7. As a statistician, I want exact scores, paired samples, frozen confidence intervals, and chance baselines, so results are auditable without an LLM judge.
8. As a mechanistic researcher, I want mediation estimated with activation clamping/patching, so probe decodability is not mistaken for a causal chain.
9. As a reviewer, I want an entire law family held out and a second checkpoint, so a one-family, one-model result cannot support a general claim.
10. As a reproducer, I want content-addressed manifests, activations, outputs, controls, metrics, and cost records, so every result can be traced.
11. As a demo visitor, I want honest cache/live, scope, and provenance labels, so the product does not overstate its capability.
12. As a release owner, I want license, privacy, security, and truthfulness gates, so unsafe or noncompliant artifacts cannot ship.

## Research Requirements

### Model organism and answer contract

Each episode contains six rendered objects with a hidden binary type. Pairwise force signs depend on whether types match; the distance exponent comes from a manifest-enumerated set. Appearance, initial state, camera, and identifiers are nuisance-randomized independently of type. Control worlds contain no hidden-variable structure.

Answers are exact structured values:

- `partition`: six binary assignments, canonicalized against their complement (31 non-trivial partitions);
- `replacement_law`: `{same: attract|repel, different: attract|repel, exponent: n}`;
- `adequacy`: Boolean;
- `force_prediction`: numeric vectors at declared horizons;
- `confidence`: `[0,1]`.

Parse failures are incorrect and separately reported. No confirmatory score uses an LLM judge.

### Conditions and comparisons

The locked manifest defines A–J and additionally requires:

- **C′:** simulator counterfactual output as text, token-budget matched;
- **E:** true learned world latent through the gated adapter, with the base model frozen;
- **G:** scrambled/random latent with E's architecture;
- **I:** same trainable parameter count and optimization exposure without dynamics information;
- **training controls:** identical adapter objective using no-hidden-structure latents and permuted episode–latent pairs.

Principal paired comparisons are E−G, E−C′, E−I, E−no-hidden-training, and E−permuted-training. B−E is always reported. The adapter objective is locked before training; neutral-description next-token loss is preferred, while task loss requires explicit disclosure and both training controls.

### Representation localization and activation extraction

For the 48-layer primary checkpoint, confirmatory residual-stream layers are zero-indexed **7, 15, 23, 31, 39, 47**. The secondary checkpoint resolves the nearest layers at normalized depths 1/6, 1/3, 1/2, 2/3, 5/6, and final before primary results are unblinded.

Activations are captured only at:

- **T0:** final token of the prior-law statement;
- **T1:** final token of the committed prediction, before new evidence;
- **T2:** final token of confirming or contradictory evidence;
- **T3:** token immediately before the adequacy Boolean;
- **T4:** token immediately before the replacement-law tuple.

Stable sentinels define these positions; missing or duplicate boundaries fail closed. Artifacts include checkpoint/tokenizer revisions, hook names and shapes, prompts, masks, precision, quantization, episode IDs, layers, timepoints, and content hashes. Exploratory full-layer scans are separate and cannot select confirmatory sites.

### Probes

All confirmatory probes are cross-fitted L2-regularized linear models with nested cross-validation on training data only. Splits group by world seed/episode; OOD families are excluded from fitting, normalization, and threshold selection.

- **Mismatch probe:** whether evidence exceeds the locked prior-law error threshold; expected rise T1→T2.
- **Inadequacy probe:** whether the prior law is structurally inadequate in error-magnitude-matched adequate/inadequate episodes; expected at T2/T3 beyond generic surprise.
- **Promotion probe:** which categorical replacement law is favored; expected to strengthen at T4 and be separable from inadequacy.

Report held-out ROC AUC (macro one-vs-rest for promotion), balanced accuracy, 95% episode-cluster intervals, coefficients, counts, and all 6×5 sites. Benjamini–Hochberg FDR is 0.05 within each probe family. Probe success alone is noncausal.

### Matched latent swaps

World A/B pairs share observed prefix, nuisance variables, candidate laws, and tokenized prompt length, but differ in hidden partition and correct consequence. Evaluate each recipient with its own, donor, and control latent in both directions. Primary outcome is donor-directional success; secondary outcomes are donor partition/law accuracy and force displacement toward the donor target. Coincident targets are excluded by construction. Injected, decoded, answer, and cached latent hashes must match.

### Causal ablation, injection, and controls

The intervention site is selected automatically as the earliest preregistered site passing localization, with ties broken by shallower layer then lexical site ID.

- **Ablation:** remove the inadequacy/promotion subspace projection. Inadequacy ablation should increase persistence with the falsified law; promotion ablation should reduce the correct replacement without materially increasing parse failures.
- **Inadequacy injection:** add the cross-fitted inadequacy direction at the median native donor projection norm. It should increase rejection/revision without encoding a specific replacement law.
- **Promotion injection:** add a candidate-specific promotion direction after inadequacy specificity passes; it should shift the structured answer toward that candidate.

Every intervention uses identical decoding and is paired with equal-count controls at the same site and norm:

1. matched-norm random directions;
2. directions orthogonal to inadequacy/promotion subspaces (absolute cosine ≤`1e-5`);
3. generic-error directions for matched surprise, uncertainty, negation, and ordinary factual error;
4. zero-scale sham hooks.

The primary effect must exceed every control distribution, not only zero.

### Mediation

Treatment `T` is true-world latent versus scrambled/wrong-world latent; outcome `Y` is exact correct revision; ordered mediators are inadequacy activation at T2/T3 and promotion activation at T4. Using the locked activation-clamping or patching estimator, report total effect (TE), direct effect (NDE), indirect effect (`NIE = TE − NDE`), and `NIE/TE` only when TE is positive with CI excluding zero. Use 10,000 episode/world-seed bootstrap resamples and repeat matched-norm, orthogonal, generic-error, and sham sensitivity analyses.

The chain requires temporal order, positive controlled indirect effects for both stages, and reduced promotion when inadequacy is ablated. The estimator and thresholds are unchanged on the second checkpoint.

### OOD and replication

At least one entire functional force-law family is `ood_confirmatory` and excluded from every fit, normalization, site selection, and threshold. The manifest freezes its generator, law DSL, prompts, and sample count. OOD repeats behavior, swaps, localization, interventions, and mediation.

A second meaningfully different checkpoint (different scale or independent training, not a quantized copy) is mandatory for mediation/general claims. Its immutable model/tokenizer revisions, license, hook map, precision, and resolved layers are locked before primary unblinding. It repeats the same data generator, timepoints, site-selection rule, controls, metrics, thresholds, and gates.

## Exact Metrics

Per-episode metrics use paired IDs. Unless specified otherwise, uncertainty is a two-sided 95% percentile bootstrap interval with 10,000 episode/world-seed clustered resamples. Equivalence uses two one-sided tests at alpha 0.05 and requires the paired 90% interval inside `[-3,+3]` percentage points.

| Metric | Definition |
|---|---|
| Partition accuracy | Exact canonical partition; chance `1/31 = 3.2258%`. |
| Sign/full law accuracy | Both signs exact (chance 25%); full tuple exact (chance `1/(4×|exponents|)`). |
| Joint theory accuracy | Partition and full law exact; chance `1/(31×4×|exponents|)`. |
| Force/rollout NRMSE | RMSE divided by ground-truth force RMS with manifest epsilon, by horizon and horizon-AUC. |
| Adequacy / false abduction | Balanced accuracy; fraction of control worlds asserting hidden types. |
| Calibration | Brier score and ECE using ten fixed equal-width bins. |
| Hidden relation / probe AUC | Held-out ROC AUC; macro one-vs-rest for promotion, with FDR q-values. |
| Swap success | Fraction of valid directions moved toward donor ground truth. |
| Intervention effect | Paired percentage-point outcome change and standardized target-minus-alternative logit change. |
| Control superiority | Primary effect minus each control mean; paired 95% CI must exclude zero positively. |
| Mediation | TE, NDE, NIE, and mediated proportion under the locked estimator. |
| OOD retention | OOD intervention effect divided by positive ID effect. |
| Provenance | Hash equality across injected latent, decoder input, answer, and cache; required 100%. |
| Cost/completion | GPU-seconds and dollars per stage/attempt; valid completed samples over planned samples. |

Pilot: ≥200 paired episodes/condition for regime finding. Confirmatory behavior: ≥500 paired episodes/condition, with final power frozen before execution. Pilot data are not pooled unless declared before the pilot.

## Pass, Pivot, and Kill Gates

| Gate | Pass | Pivot / kill boundary |
|---|---|---|
| G0 Preflight | License/access, hooks, scorer, schema, resume, cheaper 1% smoke, cost forecast, and hard caps pass. H100 additionally requires measured memory/runtime, comparative cost, and dual approval. | Replan checkpoint/hardware/scope; no launch with unresolved cap, evidence, or approval. |
| G1 Regime | Frozen joint accuracy is 20–70%, parse success ≥95%. | Saturation → study state use; floor → one ontology-supplied pivot; otherwise kill discovery claim. |
| G2 World model | Across 3 seeds, rollout NRMSE improves ≥20% over persistence and relation AUC ≥0.75 with lower CI >0.50. | One horizon/bottleneck pivot; then stop adapter/mechanism spend if both fail. |
| G3 Injection | E beats G, I, and both training controls by ≥5 points with lower CI >0; beats C′ by ≥3 with lower CI >0; swap success ≥65% with lower CI >50%; provenance 100%. | E≈C′ by equivalence rule → benchmark/engineering only; failure against latent/training controls kills latent-specific claim. |
| G4 Localization | Inadequacy AUC ≥0.75 at T2/T3, lower CI >0.50, rise from T1 ≥0.10, FDR passes, and AUC exceeds generic-error transfer; promotion meets threshold at T4. | One probe only → report correlate; missing chain component cannot enter mediation. |
| G5 Intervention | Ablation and injection each change outcome ≥5 points, lower CI >0, beat all controls, and add <2 points parse failure. | One direction only → necessity or sufficiency claim; neither → kill causal claim. |
| G6 Mediation | TE lower CI >0; ordered NIE lower CI >0; mediated proportion ≥20%; specificity controls are null or smaller. | Otherwise report interventions without a mediation chain. |
| G7 OOD | OOD causal effect lower CI >0 and retention ≥50%; provenance remains 100%. | Otherwise claim is family-specific; a reversal forbids general mechanism language. |
| G8 Replication | Second checkpoint passes G3/G5, NIE lower CI >0, mediated proportion ≥20%, OOD effect lower CI >0, retention ≥50%. | Otherwise publish a checkpoint-specific result/null; no general mediation claim. |

Overrides create a new manifest lineage with written rationale and exploratory labeling; locked confirmatory records are never mutated.

## Machine-Executable Experiment Manifest

Every run is validated against the repository's draft 2020-12 JSON Schema. The resolved manifest is canonicalized to JSON and SHA-256 hashed; mutable remote references, executable expressions, interpolation, and post-lock defaults are forbidden.

Required sections are: identity/preregistration; budgets and H100 policy; deterministic environment; data families/splits; primary and secondary checkpoints; A–J/C′/training controls; T0–T4 extraction; probes; swaps; interventions/controls; mediation; exact metrics; Boolean-AST gates; ordered execution stages; and required artifacts.

Semantic validation additionally enforces no split leakage, in-range layers, no OOD fitting, matched condition IDs/tokens/parameters, declared metric paths, immutable model revisions, internally consistent ceilings, and H100 stage/evidence/approval consistency. Locking records owner approval; any confirmatory change creates a child manifest and new hash.

## Sequential and Resumable Execution

1. Mandatory order: preflight → data/scorer → behavior → world model → condition matrix → extraction → probes → swaps → interventions → mediation → OOD → replication → release.
2. At most one GPU stage runs at once. CPU work may overlap only when it cannot affect a pending gate or budget.
3. Stage/shard idempotency hashes manifest, code, inputs, checkpoint, stage, and shard IDs.
4. Outputs write to temporary content-addressed storage and are atomically promoted after validation.
5. Inference/extraction checkpoint every ≤25 episodes or ≤5 minutes; training checkpoints model/optimizer/scheduler, RNG, data cursor, and spend every ≤10 minutes.
6. Resume verifies all hashes, skips valid shards, discards partial shards, restores RNG/order, and records every retry cost.
7. Gate failure stops downstream allocation; pivot creates a new lineage; kill writes a termination report and blocks new GPU reservations.
8. Dry-run and fault-injection tests cover artifact write/promotion, ledger update, resume, and post-gate interruption.

## Required Artifacts

Retain the locked/resolved manifest and validation; code/environment/checkpoint hashes; split and prompt manifests; structured outputs and parse failures; world-model/latent/activation indexes; probe/swap/intervention/mediation/OOD/replication results; control/norm/orthogonality audits; gate and claim decisions; cost ledger; null/termination reports; and reproducibility/license records.

Each H100 stage also retains the cheaper smoke, memory/runtime benchmark, comparative cost forecast, stage caps, and signed approval. Conversation data, secrets, raw unrestricted logs, unlicensed derivatives, and third-party assets are never experiment artifacts.

## GPU and Cost Policy

No GPU job is part of this PRD update. Execution begins only after G0.

- Credits are $188, with $175 workspace hard ceiling, ~$165 environment ceiling, and $160 ordinary stop. Track H is ≤$90; Track R ordinary authorization is `min($98, $160 − actual Track H spend)`. Reserve use requires written approval.
- Every stage has dollar, GPU-second, wall-time, and retry caps. Forecast >80% of its envelope triggers pre-launch scope reduction; forecast >100% or above remaining credits blocks launch.
- Every GPU stage first runs a no-GPU plan and 1% smoke on T4/L4 or the cheapest class that fits. The smoke exercises real hooks, context, artifacts, resume, and caps.
- **H100 is permitted for justified fine-tuning and mechanistic experiments, but is not the default.** Paid H100 launch requires: cheaper smoke; measured peak memory and throughput/runtime; comparative L40S/A100/H100 cost forecast where applicable; hard stage caps; one-active-GPU-stage confirmation; and immutable approval by the research lead and operations/budget owner. Any change to checkpoint, batch/context, extraction sites, sample count, or sweep invalidates approval.

| Workload | Default | H100-worthy only when evidence shows |
|---|---|---|
| Simulation, scoring, statistics, artifact work | CPU | Never. |
| Probe/control fitting and tests | CPU/T4/L4/A10 | Rarely: only lower total cost within a gated deadline. |
| Baselines, cache generation, ordinary inference | L4/A10/L40S | Normally never; H100 is prohibited for public UI serving or convenience latency. |
| World-module/adapter fine-tuning | L4/A10/L40S/A100 | Optimizer/activation memory will not fit, or measured H100 total cost/runtime is materially better. |
| Preregistered extraction and matched swaps | L40S/A100 | Long context/multi-hook state breaches cheaper memory, cost, or gated runtime. |
| Ablation/injection and matched-control sweeps | L40S/A100 | Repeated hooked passes are cheaper overall or cannot fit on A100. |
| Mediation, OOD, second-checkpoint replication | L40S/A100 after prior gates | Locked repeated interventions or larger checkpoint breach the approved cheaper envelope; each phase needs fresh approval. |

Full-layer exploratory dumps, ungated sweeps, UI serving, data generation, and ad hoc debugging are not H100-eligible.

## Implementation Decisions

Deep modules are: deterministic benchmark/scorer; manifest validator/planner; sequential resumable runner; model instrumentation; probe/intervention controls; statistics/gates; artifact/provenance registry; and cached scoped demo. Interfaces are versioned and content-addressed. The base model stays frozen in E; directions are cross-fitted; confirmatory/exploratory namespaces are separate; the UI reads signed gate results and cannot upgrade claim language; generated DSL is grammar-parsed with executor-side resource caps and never dynamically evaluated.

## Testing Decisions

Tests assert external contracts using CPU fixtures or minimal smoke checkpoints, never routine paid GPUs:

1. simulator truth, determinism, nuisance independence, exact parsing/scoring, chance levels, and control worlds;
2. schema rejection for leakage, missing controls/OOD/replication, mutable references, invalid sites, nonsequential runs, bad budgets, and incomplete H100 evidence;
3. stable manifest hashes and new hashes after confirmatory changes;
4. sentinel alignment, hook shapes, zero-scale identity, projection ablation, injection norm, orthogonality, and cross-fitting;
5. swap matching/provenance and corruption detection;
6. metric/bootstrap/FDR/mediation/gate fixtures with pass, pivot, and kill paths;
7. fault-injected resume, atomic artifacts, retry accounting, delayed billing, hard caps, and H100 approval invalidation;
8. cached zero-backend UI, scope/privacy/cache labels, refusal, rate limit, and 100% provenance.

## Out of Scope

Track H causal claims; general scientific-assistant routing; uploads; chemistry/biology or real-world advice; persistent conversations; free-text/LLM judging; base-model fine-tuning inside E; full-layer dumps before gates; and release of checkpoint-derived artifacts before license verification.

## Further Notes

Use “swap demonstration,” never “causal proof,” for Track H. If C′≈E, position the work as benchmark/interface engineering. If OOD or replication fails, state the family/checkpoint limitation. “Preregistered” requires an immutable external timestamp or commit before confirmatory execution; otherwise use “pre-specified.”
