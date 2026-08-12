# Independent Validation Review: JUMP Complete Research Plan

Reviewer: AO worker scratch-6. Source: `1786528309242-JUMP_COMPLETE_RESEARCH_PLAN.md` (not modified).

## Verdict

The scientific core (controlled model organism for theory revision, baseline matrix A–J, gated adapter with exact-baseline recovery, preregistered conjunction metric, failure/pivot table) is unusually well-designed. The plan fails on three structural points: (1) an unreconciled 10x timeline contradiction between the research program and the hackathon window, (2) a confounded main treatment (the adapter's training signal is never specified), and (3) a missing tool-use baseline that is the obvious cheap alternative to the entire latent-injection architecture. Product scope is roughly a second full project and should be cut ~70%.

## What checks out (verified independently)

- Zahavy "LLMs can't jump" PDF exists at the cited author URL.
- `google/gemma-4-12B` exists on HF: encoder-free "Unified" multimodal (text/image/audio), 48 layers, 256K context, Apache-2.0, bf16 ~12B params — fits L40S 48GB with activation headroom, so the GPU policy and ledger are plausible for the hackathon slice. Note: the chat product should use the `-it` variant; the plan cites the base checkpoint.
- Reference #5 (R-lens) is correctly flagged as unverified — good hygiene. Remaining 2026 arXiv IDs were not independently verified; the plan's own "recheck before implementation" instruction should be executed.
- Baseline matrix A–J, false-abduction control, leakage-resistant joint holdouts, "no LLM judge," decision gates, and budget ledger are all genuinely strong and above typical hackathon rigor.

## Critical flaws (ordered by severity)

### 1. Timeline contradiction — the plan is two plans wearing one coat
Section 11's phases sum to ~28–45 working days. Section 17 is a 2-day hackathon ending Aug 13, 7 PM — and today is Aug 12. Nothing states which subset ships in the hackathon. As written, Phases 4–6 (localization, causal surgery, mediation, OOD) — the entire "research contribution" — cannot happen in the window. **Fix: declare explicitly that the hackathon deliverable is Phases 0–3 plus the swap demo (the "minimum viable research result" of §12), and that inadequacy-state localization/mediation is post-hackathon follow-up. Rewrite the demo narrative to not promise mechanism-level claims.**

### 2. The main treatment (E) is confounded by an unspecified training signal
The plan never says what loss trains the projection W and gate. If the adapter is trained end-to-end on task answers over episodes containing hidden-type structure, then E vs A/B/C measures task-specific fine-tuning, not "counterfactual execution helps." Control I (parameter count) does not control training-signal content. **Fix: pin the adapter objective in Phase 0 (ideally next-token loss on neutral description text, or task loss with an explicit disclosure), and add a control: same adapter objective trained on latents from a dynamics model fit to worlds without hidden-variable structure, or on permuted episode–latent pairs.**

### 3. Missing baseline: simulator-as-tool with textual results
Condition C ("coordinate/state text") is ambiguous — it appears to cover observed evidence, not counterfactual rollout *results*. The obvious cheap alternative to the whole latent interface is: run the same simulator as an ordinary tool and paste its counterfactual predictions into the prompt. If that matches E, the contribution collapses to engineering. Every serious reviewer will ask this first. **Fix: add condition C′ = Gemma + simulator counterfactual outputs as text, matched token budget. Pre-register E vs C′ as a principal comparison.**

### 4. The abduction task is information-theoretically tiny
Six objects, binary hidden type = 31 partitions up to label swap; sign-flip force law is a two-parameter rule. A 12B reasoning-tuned model with textual evidence will plausibly solve it symbolically (saturating B/C), or the vision-only variant will floor at ~0%. The "20–70% mixed regime" gate is correct but the plan underweights how narrow and prompt-fragile that regime may be, and that finding it consumes Phase 1 budget. **Fix: run Phase 1 baselines before any adapter/world-model spend (the plan's gate order allows this — enforce it), and pre-commit the pivot: if C/C′ saturate, the claim becomes "how injected state is used," not "world models enable discovery."**

### 5. "No LLM judge" is not actually achieved by the schema
`replacement_law` is free text; "exact recovery of sign/qualitative rule" from free text needs either a parser that will be gamed or a judge. **Fix: make the law field categorical/DSL-enumerated (e.g., `{same: attract|repel, different: attract|repel, exponent: n}`), which also sharpens law accuracy scoring.**

## Weak claims to soften

- **Decoded-latent provenance ≠ interpretability.** The caption ("decoded from the same latent supplied to the model") is provenance-true but implies the image shows what Gemma *uses*. The decoder is a separately trained consumer; it reflects its own training. The swap test partially validates this — keep the swap test, weaken the caption to "decoded from the same latent, validated by swap tests."
- **Latent-only readout** shows the pipeline can transmit information, not that "the latent contains the simulated consequence" in any model-internal sense. Frame accordingly.
- **Mediation as "decisive test."** Single-direction ablation mediation on one checkpoint is the most over-interpreted maneuver in current interpretability. The plan's specificity controls (surprise/uncertainty/negation) are good; add cross-checkpoint replication as a *requirement* for the mediation claim, not a "if compute permits."
- **Chance levels unreported.** With 31 possible partitions and 4 sign-rule variants, chance and shortcut baselines must appear next to every accuracy number.
- **§6 product metrics** (~13 metrics: intent classification accuracy, clarification rates, etc.) have no measurement plan, data source, or raters. In a 2-day build these are vaporware; delete or mark aspirational.

## Scope and opportunity cost

- The product (§6) — general scientific chat, query compiler, domain router, 8 adapter families, multi-turn investigation memory, safety DSL, conflict matrix, research view — is a second full project, mostly disjoint from the research claim. Building it competes directly with Phases 1–3 for the only two days available. **Cut to: one synthetic-dynamics adapter, the three canned demo conversations (precomputed + one live path), the swap toggle, and provenance labels. Drop the router/registry, uploaded-tabular adapter, statistical-hypothesis adapter, and most of the clarification policy.**
- **Budget:** $188 is plausible for the hackathon slice (~$60–90) but not for the full 7-phase program; activation extraction at T0–T4 across 48 layers for thousands of episodes on 12B will strain the $25 line item. The per-phase cost gates are good; add a rule that any phase forecast >80% of its ceiling triggers scope reduction before launch, not after overrun.
- **Opportunity cost not argued:** the plan never states why latent injection is worth building relative to (a) tool-use simulation (see flaw 3) and (b) fine-tuning on simulator traces. One paragraph of explicit positioning against both would materially strengthen the paper framing.

## Sharper plan (concrete)

1. **Split the document into two committed artifacts:** `HACKATHON.md` (48h: Phases 0–3 + swap demo + Space with cached demos, ≤$90) and `RESEARCH.md` (Phases 4–6, post-event). Judge story sells the swap result, not mediation.
2. **Reorder Day 1:** frozen-Gemma behavioral baselines (A/B/C/C′) on the pilot set *first*, before any adapter code. This is the cheapest experiment and determines whether the whole architecture is worth building.
3. **Add condition C′** (simulator-as-tool, text results) and the adapter training-signal control; pre-register E vs C′ and E vs I as principal comparisons alongside E vs G.
4. **Pin the adapter objective and the law-DSL answer schema in Phase 0** (both currently unspecified — the two biggest unspecified degrees of freedom).
5. **Add a power note:** with paired design and 200 episodes/condition, detectable effect ~10–12pp at 95% CI; size the full run (≥500/condition) for the preregistered comparisons.
6. **Use `gemma-4-12B-it` for the chat surface;** verify hook points survive quantization before committing the serving budget.
7. **Demote §6 metrics** to a post-hackathon appendix; keep only: valid compilation rate, tool execution success, counterfactual answer accuracy, and provenance-labeling correctness.
8. **Execute the plan's own reference-verification instruction** for the unverified 2026 arXiv IDs, and either find the R-lens primary source or remove it from the method list (it currently appears in deliverables §20 despite being unsourced).

## Bottom line

Keep: model organism design, A–J matrix, gates, ledger, swap-test-as-credibility-centerpiece, truthfulness requirements. Fix before building: timeline split, adapter training-signal specification + control, C′ tool-use baseline, categorical law schema. Cut: domain router/registry, most of the product surface, mediation claims from the hackathon narrative. The most likely failure mode is not scientific — it is attempting the full document in 48 hours and shipping neither the product nor the result.
