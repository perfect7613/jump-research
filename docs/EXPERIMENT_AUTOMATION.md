# Sequential experiment automation

This package turns a preregistered manifest into a durable, auditable sequence
of GPU runs. It is deliberately **not** a sweep launcher: the Modal controller
waits for each GPU function to finish before dispatching the next one, and the
checked-in example locks full-matrix submission off.

## Safety invariants

1. `dry-run` validates the whole manifest before any remote call. It rejects
   unknown GPUs, forward/missing phase dependencies, duplicate IDs,
   non-preregistered layers/timepoints, unsafe secret-like task parameters, and
   worst-case retry time/cost above a phase ceiling. GPU phases must declare
   `max_concurrent_gpus: 1`; CPU-only phases use `max_concurrent_gpus: 0`.
2. Both `launch_policy.allow_full_matrix: true` **and** `--confirm-paid` are
   needed for a non-smoke submit. `--smoke` selects only runs explicitly marked
   `smoke_test: true`.
3. Runs are serial. A failed run or failed phase gate stops the controller; a
   downstream phase starts only when every `depends_on` phase passed.
4. A completed run is never executed again. Retry creates a new numbered
   attempt and exposes the preceding `checkpoint/` through `JUMP_RESUME_FROM`.
5. Manifest, resolved run config, attempt start/result, stdout/stderr, artifacts,
   SHA-256 manifests, and final run result are immutable-by-contract. Only the
   experiment-level `status.json` checkpoint is atomically replaceable.

## Versioned v1 contract

The normative schemas are
[`schemas/experiment-manifest-v1.schema.json`](../schemas/experiment-manifest-v1.schema.json)
and [`schemas/run-result-v1.schema.json`](../schemas/run-result-v1.schema.json).
Unknown domain-specific fields are retained for forward compatibility.

Each run has a unique `id`, a safe `task.module` invoked as `python -m`, JSON
`task.parameters`, resource/retry policy, and optional explicit `selection`.
The task receives:

```text
python -m MODULE \
  --parameters /immutable/task-parameters.json \
  --output-dir /attempt/work \
  --checkpoint-dir /attempt/checkpoint
```

The same paths are exported as `JUMP_PARAMETERS_PATH`, `JUMP_OUTPUT_DIR`, and
`JUMP_CHECKPOINT_DIR`; a prior checkpoint path is exported as
`JUMP_RESUME_FROM`. A successful task must write `result.json` in its output
directory with a `metrics` array. Artifact files beside it are copied and
hashed. The runner adds status, timings, artifacts, and provenance.
Task modules under `src/` are bundled into the Modal image; add third-party task
dependencies to the image definition explicitly rather than installing at run
time.

Mechanistic tasks can carry `checkpoint_id`, `replication_id`, split seed,
world-pair IDs, interventions/controls, and OOD law families in parameters.
Metric records support `{name,value,split,condition,checkpoint_id,world_pair_id,
timepoint,layer,ci}`. Output layer/timepoint values are validated again against
the preregistration, so a buggy task cannot widen the study after launch.

### Mechanistic analysis task

`jump_mechanistic.runner` implements the subprocess contract above and checks
the immutable run config before capturing activations. Its CPU-first primitives
cover strict layer/T0--T4 capture, exact behavioral scoring, episode/world-group
held-out probes, leave-one-law-family-out evaluation, matched World A/B swaps,
ablation/injection with matched-norm, orthogonal and generic-error controls,
paired causal effects, mediation specificity, and two-checkpoint replication.
The packaged synthetic fixture verifies plumbing, not a scientific claim.

```json
{
  "task": {
    "module": "jump_mechanistic.runner",
    "parameters": {"task": "mechanistic_suite.synthetic", "seed": 17}
  },
  "selection": {
    "layers": ["model.layers.8"],
    "timepoints": ["T0", "T1", "T2", "T3", "T4"]
  }
}
```

## Local workflow (no paid compute)

```bash
python -m pip install -e '.[test]'
pytest
jump-experiments plan examples/smoke-manifest.yaml --smoke
jump-experiments dry-run examples/smoke-manifest.yaml --smoke
jump-experiments run-local examples/smoke-manifest.yaml --smoke --runs-dir /tmp/jump-runs
jump-experiments status examples/smoke-manifest.yaml --smoke --runs-dir /tmp/jump-runs
jump-experiments run-local examples/mechanistic-synthetic.manifest.json --smoke --runs-dir /tmp/jump-mechanistic
```

`run-local` is smoke-only. The examples above use deterministic CPU tasks:
the runner protocol mock and the mechanistic fixture suite. Verification is run
in a fresh local virtual environment and makes no GPU calls; this repository
does not consume hosted Actions minutes for the test suite.

## Modal deployment and smoke

Run the local tests and dry-run first. Then authenticate and inspect the active
profile without printing credentials:

```bash
modal --version
modal profile current
modal token info  # inspect identity only; do not copy its output into reports
```

Deploy the app and submit only the marked smoke run:

```bash
python -m pip install -e '.[modal]'
modal deploy -m jump_runner.modal_app
modal run -m jump_runner.modal_app::submit --manifest-path examples/smoke-manifest.yaml --smoke
modal run -m jump_runner.modal_app::status --manifest-path examples/smoke-manifest.yaml --smoke
```

The example reserves at most two 30-second T4 attempts and declares a planning
ceiling of $0.02. The forecast uses the manifest-pinned rate, not live pricing;
actual billing remains authoritative in Modal. Do not infer scientific evidence
from the protocol smoke.

## Secrets

Never place secret values in a manifest. Create named Modal Secret objects out
of band, then deploy with `JUMP_MODAL_SECRET_SPECS` as a JSON map from object
name to its exhaustive environment-key list (names only). A manifest declares
the same name/`required_keys`, and a run opts in with `secret: NAME`. The remote
boundary requires an exact match and mounts only that one Secret on that one
worker; secret-free runs receive none. Task subprocesses inherit only a small
environment allowlist plus declared keys. Before stdout/stderr become immutable,
the runner replaces declared values and common credential patterns with
`[REDACTED]`. Redaction is defense in depth, not permission to print secrets;
tasks remain responsible for avoiding sensitive output. Keep smoke tests
secret-free.

Every worker derives its durable path from the authorized manifest hash,
execution mode, phase ID, and run ID; callers cannot choose output roots. A
shared atomic Modal Dict lease serializes all worker functions, including direct
calls across GPU types. A crashed worker leaves the lease present and dispatch
fails closed until an operator inspects and clears the named lease Dict.

## Durable layout

```text
RUNS/EXPERIMENT/MANIFEST_SHA/{smoke|full}/
  manifest.json
  status.json                         # mutable atomic checkpoint
  phases/PHASE/result.json            # immutable gate/budget result
  phases/PHASE/runs/RUN/
    config.json
    result.json                       # immutable terminal result
    artifact_hashes.json
    hashes.sha256
    attempts/0001/
      started.json
      task-parameters.json
      stdout.log
      stderr.log
      result.json
      checkpoint/*
      artifacts/*
      artifact_hashes.json
      hashes.sha256
```

The manifest hash and separate `smoke`/`full` namespaces prevent configuration
drift or smoke completion from contaminating a paid run.

### CPU-only phases

Deterministic synthesis, scoring, and integration tasks can declare
`resources.gpu: cpu`, use `allowed_gpu_types: [cpu]`, set
`max_concurrent_gpus: 0`, and pin `gpu_hourly_cost_usd: {cpu: 0}`. Modal routes
these runs to a function without a GPU reservation while retaining the same
serial execution, evidence, retry, and gate contract.

### H100 escalation (opt-in only)

H100 is never a default and must be written explicitly on each H100 run. Its
phase must depend on an earlier gated `smoke_test` run using T4, L4, or L40S and
must include `h100_justification` with measured peak memory, measured runtime,
why a lower GPU is insufficient, a retry-aware cost forecast, and the remaining
budget. The runner checks that the forecast equals the planned H100 cost and
fits the remaining budget. Both phases require stop-on-failure gates. The H100
Modal function has a hard one-hour timeout, one container, and one concurrent
input.

Even after validation, launch remains locked. A selected H100 plan requires all
four approvals: `launch_policy.allow_full_matrix: true`, `--confirm-paid`,
`launch_policy.allow_h100: true`, and `--confirm-h100`. Keep both policy flags
false until the manifest, measurements, and forecast are reviewed and approved.
Declaring or deploying the H100 function does not execute it.

Use CPU for scoring/probes/mediation, L40S for the 1% hook and memory smoke,
and A100-80GB as the default large-memory tier for full allowlisted capture.
No current phase requires H100. The checked-in H100 template is intentionally
launch-locked and H100 is considered only when sealed L40S and completed A100
profiles show all of the following: peak memory at most 72 GiB, A100 misses the
runtime ceiling, projected H100 speedup is at least 1.25x, and retry-aware cost
fits remaining budget. Above 72 GiB, reduce batch/context, stream, checkpoint,
or shard; H100 and A100-80GB have the same nominal memory capacity.

The valid no-H100 profile plan is
`examples/mechanistic-gpu-profile.manifest.json`. After its gated profile,
materialize the separate locked template from the two completed runner results:

```bash
python -m jump_mechanistic.hardware \
  --template examples/mechanistic-h100-escalation.manifest.template.json \
  --l40s-result /runs/l40s-one-percent-hook-smoke/result.json \
  --a100-result /runs/a100-throughput-profile/result.json \
  --output /tmp/reviewed-h100-manifest.json
```

At the materialization boundary, the loader rereads both paths, recomputes the
referenced artifact hashes, verifies each result's runner checksum manifest and
immutable config, and requires matching manifest, code version, workload,
checkpoint revision, and distinct L40S/A100 run IDs. The immutable profile phase
result must exist, have `status: passed`, and report every declared gate passed.
Materialization records those bindings (including the phase-result digest),
calculates the exact forecast, and preserves both launch flags as false. It does
not accept a previously loaded or caller-modified evidence object. A human
review is still required before either flag is changed.

### Confirmatory statistics and gates

Behavioral outputs use the locked six-object partition denominator, explicit
exponent allowlist, and adequacy balanced accuracy. Probe sample IDs are unique;
each episode/world group belongs to one law family, and OOD train/test groups
must be disjoint. Confirmatory paired and mediation intervals are deterministic
95% percentile intervals from exactly 10,000 episode/world-seed cluster
resamples; artifacts record the seed, cluster count, and resample count.

G6 requires positive TE and both ordered NIE lower bounds, mediated proportion
at least 20%, and specificity. G7 requires positive OOD causal effect, at least
50% retention, and 100% provenance. G8 requires G3/G5 plus all of those
conditions on an immutable, independently revised second checkpoint. Missing,
partial, single-checkpoint, or aliased evidence fails closed.
