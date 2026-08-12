# JUMP — Research Repository

JUMP is a controlled model organism for mechanistically studying **theory revision** in multimodal language models: does injecting an executable counterfactual world representation cause a frozen model to reject an inadequate law and promote a hidden-variable explanation? The research core is wrapped in a conversational scientific thought-experiment product.

The repository also includes a CPU-first mechanistic experiment suite: strict
activation capture, exact scoring, held-out/OOD probes, paired latent swaps,
causal interventions and controls, mediation, and second-checkpoint replication
hooks. It is callable through a versioned manifest/result contract for the Modal
runner; see [the automation guide](docs/EXPERIMENT_AUTOMATION.md).

## Scope

Work is split into two committed tracks (see the revised plan for the full rationale):

- **Track H — hackathon (≤48h, ≤$90):** a benchmark and an instrument. Six-object 2D simulator with hidden binary type; frozen-Gemma behavioral baselines A/B/C/C′; one latent dynamics module with one adapter injection path; the matched World A/World B latent-swap demonstration; a scoped, cached-first HF Space chat UI with provenance badges. **No causal or mechanistic claims ship in this window.**
- **Track R — research program (4–7 weeks post-event):** localization, causal surgery, and mediation analysis under preregistered gates, with strengthened controls (C′ tool-use baseline, adapter training-signal control, mandatory second-checkpoint replication for any mediation claim).

Overall posture: **CONDITIONAL GO** — go for the vertical slice with honest labeling; no-go for any public causal/mechanistic claim inside the hackathon window.

## Documents

| Document | Description |
|---|---|
| [docs/JUMP_PRD.md](docs/JUMP_PRD.md) | Implementation-ready PRD for the benchmark/instrument and gated mechanism study, including exact metrics, causal controls, artifacts, budgets, and pass/pivot/kill rules |
| [docs/JUMP_REVISED_RESEARCH_PLAN.md](docs/JUMP_REVISED_RESEARCH_PLAN.md) | The authoritative revised plan: two-track split, MVP definitions, architecture, hypotheses H1–H5, condition matrix, budget, phase gates |
| [docs/JUMP_RISK_REVIEW.md](docs/JUMP_RISK_REVIEW.md) | Red-team risk review (R1–R23, A1–A12): timeline, budget, security, compliance, and overclaim risks with adopted mitigations |
| [docs/JUMP_VALIDATION_REVIEW.md](docs/JUMP_VALIDATION_REVIEW.md) | Independent validation review (flaws 1–5, sharper plan items 1–8): confound analysis, baseline gaps, scope and metric critiques |
| [docs/EXPERIMENT_AUTOMATION.md](docs/EXPERIMENT_AUTOMATION.md) | Versioned contract, mechanistic task interface, hardware escalation policy, and safe sequential runner workflow |
| [schemas/experiment-manifest-v1.schema.json](schemas/experiment-manifest-v1.schema.json) | Authoritative experiment manifest schema |
| [schemas/run-result-v1.schema.json](schemas/run-result-v1.schema.json) | Authoritative immutable run-result schema |

## Experiment runner and mechanistic suite

The repository includes a safe sequential experiment runner plus CPU-first
mechanistic primitives for allowlisted activation capture, exact scoring,
held-out/OOD probes, matched latent swaps, causal interventions, mediation, and
second-checkpoint replication. Paid matrices and H100 escalation are fail-closed.

```bash
python -m pip install -e '.[test]'
pytest
jump-experiments dry-run examples/mechanistic-synthetic.manifest.json --smoke
```

See [the automation guide](docs/EXPERIMENT_AUTOMATION.md) before any Modal use.

## What this repository will not contain

- Secrets, tokens, or credentials of any kind (CI secret scanning is planned per risk R8)
- Datasets or model weights (released via appropriate artifact channels, subject to license verification)
- Any Gemma-derived checkpoint before license pass-through verification (risk R12)
- Copyrighted or licensed third-party assets

## License

**License is pending.** No license has been selected yet because open questions remain — notably Gemma license pass-through obligations for any derived artifacts (risk R12) and the intended release posture for benchmark data versus code. Until a LICENSE file is added, all rights are reserved; do not reuse or redistribute the contents.
