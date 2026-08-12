# JUMP — Research Repository

JUMP is a controlled model organism for mechanistically studying **theory revision** in multimodal language models: does injecting an executable counterfactual world representation cause a frozen model to reject an inadequate law and promote a hidden-variable explanation? The research core is wrapped in a conversational scientific thought-experiment product.

This repository currently holds the planning and review documents. Code, benchmark data, and evaluation scripts will land here as the project executes.

## Scope

Work is split into two committed tracks (see the revised plan for the full rationale):

- **Track H — hackathon (≤48h, ≤$90):** a benchmark and an instrument. Six-object 2D simulator with hidden binary type; frozen-Gemma behavioral baselines A/B/C/C′; one latent dynamics module with one adapter injection path; the matched World A/World B latent-swap demonstration; a scoped, cached-first HF Space chat UI with provenance badges. **No causal or mechanistic claims ship in this window.**
- **Track R — research program (4–7 weeks post-event):** localization, causal surgery, and mediation analysis under preregistered gates, with strengthened controls (C′ tool-use baseline, adapter training-signal control, mandatory second-checkpoint replication for any mediation claim).

Overall posture: **CONDITIONAL GO** — go for the vertical slice with honest labeling; no-go for any public causal/mechanistic claim inside the hackathon window.

## Documents

| Document | Description |
|---|---|
| [docs/JUMP_REVISED_RESEARCH_PLAN.md](docs/JUMP_REVISED_RESEARCH_PLAN.md) | The authoritative revised plan: two-track split, MVP definitions, architecture, hypotheses H1–H5, condition matrix, budget, phase gates |
| [docs/JUMP_RISK_REVIEW.md](docs/JUMP_RISK_REVIEW.md) | Red-team risk review (R1–R23, A1–A12): timeline, budget, security, compliance, and overclaim risks with adopted mitigations |
| [docs/JUMP_VALIDATION_REVIEW.md](docs/JUMP_VALIDATION_REVIEW.md) | Independent validation review (flaws 1–5, sharper plan items 1–8): confound analysis, baseline gaps, scope and metric critiques |

## What this repository will not contain

- Secrets, tokens, or credentials of any kind (CI secret scanning is planned per risk R8)
- Datasets or model weights (released via appropriate artifact channels, subject to license verification)
- Any Gemma-derived checkpoint before license pass-through verification (risk R12)
- Copyrighted or licensed third-party assets

## License

**License is pending.** No license has been selected yet because open questions remain — notably Gemma license pass-through obligations for any derived artifacts (risk R12) and the intended release posture for benchmark data versus code. Until a LICENSE file is added, all rights are reserved; do not reuse or redistribute the contents.
