"""Evidence-gated hardware selection and H100 manifest materialization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


H100_WORKLOADS = frozenset({"activation_capture", "adapter_training", "full_finetune"})
L40S_SAFE_PEAK_GB = 44.0
EIGHTY_GB_SAFE_PEAK_GB = 72.0
MIN_H100_SPEEDUP = 1.25
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ProfileEvidence:
    """Measurements loaded only from two hashed, immutable runner results."""

    profile_phase_id: str
    profile_experiment_id: str
    profile_phase_result_sha256: str
    manifest_sha256: str
    code_version: str
    l40s_run_id: str
    a100_run_id: str
    l40s_artifact_sha256: str
    a100_artifact_sha256: str
    workload: str
    checkpoint_id: str
    checkpoint_revision: str
    measured_peak_memory_gb: float
    measured_runtime_seconds: float
    projected_l40s_runtime_seconds: float
    projected_a100_runtime_seconds: float
    projected_h100_runtime_seconds: float
    runtime_ceiling_seconds: float
    remaining_budget_usd: float
    _verified_digest: str | None = field(init=False, default=None, repr=False, compare=False)

    @classmethod
    def from_runner_results(
        cls,
        l40s_result_path: str | Path,
        a100_result_path: str | Path,
        *,
        profile_phase_id: str = "gpu-memory-profile",
    ) -> "ProfileEvidence":
        l40s = _load_profile_record(Path(l40s_result_path), expected_gpu="L40S")
        a100 = _load_profile_record(Path(a100_result_path), expected_gpu="A100-80GB")
        if l40s["phase_id"] != profile_phase_id or a100["phase_id"] != profile_phase_id:
            raise ValueError("profile results must be bound to the declared profile phase ID")
        if l40s["source_manifest_path"] != a100["source_manifest_path"]:
            raise ValueError("L40S and A100 results must share one immutable source manifest")
        for key in (
            "manifest_sha256",
            "experiment_id",
            "code_version",
            "workload",
            "checkpoint_id",
            "checkpoint_revision",
            "projected_l40s_runtime_seconds",
            "projected_a100_runtime_seconds",
            "projected_h100_runtime_seconds",
            "runtime_ceiling_seconds",
            "remaining_budget_usd",
        ):
            if l40s[key] != a100[key]:
                raise ValueError(f"L40S and A100 profile evidence mismatch for {key}")
        if l40s["run_id"] == a100["run_id"]:
            raise ValueError("L40S and A100 evidence must come from distinct runner run IDs")
        phase_result_sha256 = _verify_profile_phase(
            Path(l40s["source_manifest_path"]),
            profile_phase_id,
            l40s["declared_gates"],
        )
        evidence = cls(
            profile_phase_id=profile_phase_id,
            profile_experiment_id=l40s["experiment_id"],
            profile_phase_result_sha256=phase_result_sha256,
            manifest_sha256=l40s["manifest_sha256"],
            code_version=l40s["code_version"],
            l40s_run_id=l40s["run_id"],
            a100_run_id=a100["run_id"],
            l40s_artifact_sha256=l40s["artifact_sha256"],
            a100_artifact_sha256=a100["artifact_sha256"],
            workload=l40s["workload"],
            checkpoint_id=l40s["checkpoint_id"],
            checkpoint_revision=l40s["checkpoint_revision"],
            measured_peak_memory_gb=max(
                l40s["measured_peak_memory_gb"], a100["measured_peak_memory_gb"]
            ),
            measured_runtime_seconds=a100["measured_runtime_seconds"],
            projected_l40s_runtime_seconds=l40s["projected_l40s_runtime_seconds"],
            projected_a100_runtime_seconds=l40s["projected_a100_runtime_seconds"],
            projected_h100_runtime_seconds=l40s["projected_h100_runtime_seconds"],
            runtime_ceiling_seconds=l40s["runtime_ceiling_seconds"],
            remaining_budget_usd=l40s["remaining_budget_usd"],
        )
        object.__setattr__(evidence, "_verified_digest", _evidence_content_sha256(evidence))
        _validate_evidence(evidence)
        return evidence


@dataclass(frozen=True)
class HardwareDecision:
    selected_gpu: str | None
    passed: bool
    reason: str


def select_hardware(evidence: ProfileEvidence, *, h100_forecast_cost_usd: float) -> HardwareDecision:
    """Choose the cheapest adequate tier; H100 is a throughput escalation only."""
    _validate_evidence(evidence)
    if evidence.measured_peak_memory_gb > EIGHTY_GB_SAFE_PEAK_GB:
        return HardwareDecision(
            None,
            False,
            "Projected peak exceeds the 72 GiB safety ceiling shared by A100-80GB and H100; "
            "reduce batch/context, shard, or checkpoint instead of escalating.",
        )
    if (
        evidence.measured_peak_memory_gb <= L40S_SAFE_PEAK_GB
        and evidence.projected_l40s_runtime_seconds <= evidence.runtime_ceiling_seconds
    ):
        return HardwareDecision("L40S", True, "The workload fits the L40S memory and runtime ceilings.")
    if evidence.projected_a100_runtime_seconds <= evidence.runtime_ceiling_seconds:
        return HardwareDecision("A100-80GB", True, "A100-80GB meets the measured memory and runtime ceilings.")

    speedup = evidence.projected_a100_runtime_seconds / evidence.projected_h100_runtime_seconds
    if speedup < MIN_H100_SPEEDUP:
        return HardwareDecision(
            None,
            False,
            f"Projected H100 speedup {speedup:.3f}x is below the preregistered {MIN_H100_SPEEDUP:.2f}x threshold.",
        )
    if evidence.projected_h100_runtime_seconds > evidence.runtime_ceiling_seconds:
        return HardwareDecision(None, False, "Even H100 is projected to miss the runtime ceiling.")
    if not math.isfinite(h100_forecast_cost_usd) or h100_forecast_cost_usd < 0:
        return HardwareDecision(None, False, "The H100 forecast must be finite and nonnegative.")
    if h100_forecast_cost_usd > evidence.remaining_budget_usd:
        return HardwareDecision(None, False, "The retry-aware H100 forecast exceeds remaining budget.")
    return HardwareDecision(
        "H100",
        True,
        "A100-80GB misses the runtime ceiling, while profiled H100 projection meets it with "
        f"{speedup:.3f}x speedup and fits the remaining budget.",
    )


def materialize_h100_manifest(
    template: dict[str, Any],
    *,
    l40s_result_path: str | Path,
    a100_result_path: str | Path,
) -> tuple[dict[str, Any], HardwareDecision]:
    """Reload verified runner evidence at the safety boundary and preserve launch locks."""
    evidence = ProfileEvidence.from_runner_results(l40s_result_path, a100_result_path)
    manifest = copy.deepcopy(template)
    launch_policy = manifest.setdefault("launch_policy", {})
    if launch_policy.get("allow_full_matrix") is not False or launch_policy.get("allow_h100") is not False:
        raise ValueError("the H100 template must keep both launch locks false")
    h100_phases = [
        phase
        for phase in manifest.get("phases", [])
        if "H100" in phase.get("budget", {}).get("allowed_gpu_types", [])
    ]
    if len(h100_phases) != 1:
        raise ValueError("template must contain exactly one H100 phase")
    phase = h100_phases[0]
    if evidence.profile_phase_id not in phase.get("depends_on", []):
        raise ValueError("H100 phase must depend on the measured profile phase")
    rate = phase["budget"]["gpu_hourly_cost_usd"]["H100"]
    forecast = 0.0
    for run in phase.get("runs", []):
        if run.get("resources", {}).get("gpu") != "H100":
            raise ValueError("every H100 phase run must explicitly select H100")
        parameters = run.get("task", {}).get("parameters", {})
        for key in ("workload", "checkpoint_id", "checkpoint_revision"):
            if parameters.get(key) != getattr(evidence, key):
                raise ValueError(f"profile {key} does not match H100 run {key}")
        attempts = run.get("retry", {}).get("max_attempts", 1)
        forecast += run["resources"]["timeout_seconds"] * attempts / 3600.0 * rate
    decision = select_hardware(evidence, h100_forecast_cost_usd=forecast)
    if not decision.passed or decision.selected_gpu != "H100":
        raise ValueError(f"H100 escalation is not justified: {decision.reason}")
    phase["h100_justification"] = {
        "opt_in": True,
        "profile_phase_id": evidence.profile_phase_id,
        "profile_experiment_id": evidence.profile_experiment_id,
        "profile_phase_result_sha256": evidence.profile_phase_result_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "code_version": evidence.code_version,
        "l40s_run_id": evidence.l40s_run_id,
        "a100_run_id": evidence.a100_run_id,
        "l40s_artifact_sha256": evidence.l40s_artifact_sha256,
        "a100_artifact_sha256": evidence.a100_artifact_sha256,
        "checkpoint_id": evidence.checkpoint_id,
        "checkpoint_revision": evidence.checkpoint_revision,
        "measured_peak_memory_gb": evidence.measured_peak_memory_gb,
        "measured_runtime_seconds": evidence.measured_runtime_seconds,
        "why_lower_gpu_insufficient": decision.reason,
        "forecast_cost_usd": forecast,
        "remaining_budget_usd": evidence.remaining_budget_usd,
    }
    return manifest, decision


def _load_profile_record(result_path: Path, *, expected_gpu: str) -> dict[str, Any]:
    if not result_path.is_file():
        raise ValueError(f"runner result does not exist: {result_path}")
    result = _read_strict_json(result_path)
    config_path = result_path.parent / "config.json"
    if not config_path.is_file():
        raise ValueError("runner-produced config.json is required beside result.json")
    config = _read_strict_json(config_path)
    if result.get("schema_version") != "jump.run-result/v1" or result.get("status") != "completed":
        raise ValueError("profile runner result must be a completed jump.run-result/v1")
    if config.get("schema_version") != "jump.run-config/v1":
        raise ValueError("profile evidence requires a jump.run-config/v1")
    provenance = result.get("provenance", {})
    for key in ("manifest_sha256", "run_id", "code_version"):
        if provenance.get(key) != config.get(key):
            raise ValueError(f"runner result/config provenance mismatch for {key}")
    manifest_sha = provenance.get("manifest_sha256")
    code_version = provenance.get("code_version")
    if not isinstance(manifest_sha, str) or not _SHA256.fullmatch(manifest_sha):
        raise ValueError("runner manifest_sha256 is invalid")
    if not isinstance(code_version, str) or not code_version or code_version == "unknown":
        raise ValueError("runner code_version must be resolved")
    if config.get("resources", {}).get("gpu") != expected_gpu:
        raise ValueError(f"runner config must bind profile to {expected_gpu}")
    phase_id = config.get("phase_id")
    if not isinstance(phase_id, str) or not phase_id:
        raise ValueError("runner config must bind a profile phase ID")
    source = _verify_source_manifest(result_path, config, manifest_sha)

    matches = [artifact for artifact in result.get("artifacts", []) if artifact.get("name") == "hardware-profile.json"]
    if len(matches) != 1:
        raise ValueError("runner result must reference exactly one hardware-profile.json artifact")
    artifact = matches[0]
    relative = Path(artifact.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("profile artifact path must stay inside the runner directory")
    artifact_path = result_path.parent / relative
    if not artifact_path.is_file():
        raise ValueError("referenced hardware profile artifact is missing")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if digest != artifact.get("sha256"):
        raise ValueError("hardware profile artifact SHA-256 mismatch")
    _verify_runner_hash_manifest(result_path.parent, [Path("config.json"), Path("result.json"), relative])
    profile = _read_strict_json(artifact_path)
    required = {
        "schema_version", "run_id", "gpu", "workload", "checkpoint_id", "checkpoint_revision",
        "measured_peak_memory_gb", "measured_runtime_seconds", "projected_l40s_runtime_seconds",
        "projected_a100_runtime_seconds", "projected_h100_runtime_seconds", "runtime_ceiling_seconds",
        "remaining_budget_usd",
    }
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError(f"hardware profile must contain exactly {sorted(required)}")
    parameters = config.get("task", {}).get("parameters", {})
    if profile["schema_version"] != "jump.hardware-profile/v1":
        raise ValueError("hardware profile schema_version is invalid")
    if profile["run_id"] != provenance["run_id"] or profile["gpu"] != expected_gpu:
        raise ValueError("hardware profile run ID or GPU does not match runner evidence")
    for key in ("workload", "checkpoint_id", "checkpoint_revision"):
        if profile[key] != parameters.get(key):
            raise ValueError(f"hardware profile {key} does not match runner task config")
    if profile["workload"] not in H100_WORKLOADS:
        raise ValueError(f"workload must be one of {sorted(H100_WORKLOADS)}")
    numeric_keys = required - {
        "schema_version", "run_id", "gpu", "workload", "checkpoint_id", "checkpoint_revision"
    }
    for key in numeric_keys:
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"hardware profile {key} must be finite numeric")
        if key != "remaining_budget_usd" and value <= 0:
            raise ValueError(f"hardware profile {key} must be positive")
        if key == "remaining_budget_usd" and value < 0:
            raise ValueError("hardware profile remaining_budget_usd must be nonnegative")
    return {
        **profile,
        "manifest_sha256": manifest_sha,
        "code_version": code_version,
        "artifact_sha256": digest,
        "phase_id": phase_id,
        "experiment_id": source["experiment_id"],
        "source_manifest_path": source["manifest_path"],
        "declared_gates": source["declared_gates"],
    }


def _validate_evidence(evidence: ProfileEvidence) -> None:
    if not isinstance(evidence, ProfileEvidence):
        raise ValueError("profile evidence must be loaded from bound runner results")
    expected_digest = _evidence_content_sha256(evidence)
    if evidence._verified_digest is None or not _SHA256.fullmatch(evidence._verified_digest):
        raise ValueError("profile evidence must be loaded from bound runner results")
    if evidence._verified_digest != expected_digest:
        raise ValueError("profile evidence changed after runner-result verification")
    for digest in (
        evidence.manifest_sha256,
        evidence.profile_phase_result_sha256,
        evidence.l40s_artifact_sha256,
        evidence.a100_artifact_sha256,
    ):
        if not _SHA256.fullmatch(digest):
            raise ValueError("profile binding contains an invalid SHA-256 digest")
    if evidence.l40s_run_id == evidence.a100_run_id:
        raise ValueError("bound profile run IDs must be distinct")


def _evidence_content_sha256(evidence: ProfileEvidence) -> str:
    value = {
        field_name: getattr(evidence, field_name)
        for field_name, descriptor in evidence.__dataclass_fields__.items()
        if descriptor.init
    }
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _verify_runner_hash_manifest(root: Path, required_paths: list[Path]) -> None:
    checksum_path = root / "hashes.sha256"
    if not checksum_path.is_file():
        raise ValueError("runner-produced hashes.sha256 is required")
    records: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None or match.group(2) in records:
            raise ValueError("runner hashes.sha256 is malformed or contains duplicate paths")
        records[match.group(2)] = match.group(1)
    for relative in required_paths:
        name = relative.as_posix()
        path = root / relative
        if name not in records or not path.is_file():
            raise ValueError(f"runner hashes.sha256 does not bind {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != records[name]:
            raise ValueError(f"runner hashes.sha256 mismatch for {name}")


def _verify_source_manifest(
    result_path: Path, config: dict[str, Any], expected_sha256: str
) -> dict[str, Any]:
    manifest_path = next(
        (parent / "manifest.json" for parent in result_path.parents if (parent / "manifest.json").is_file()),
        None,
    )
    if manifest_path is None:
        raise ValueError("runner source manifest.json is required in the run ancestry")
    manifest = _read_strict_json(manifest_path)
    encoded = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("runner source manifest does not match manifest_sha256 provenance")
    phases = [phase for phase in manifest.get("phases", []) if phase.get("id") == config["phase_id"]]
    if len(phases) != 1:
        raise ValueError("runner source manifest does not contain the bound profile phase")
    runs = [run for run in phases[0].get("runs", []) if run.get("id") == config["run_id"]]
    if len(runs) != 1:
        raise ValueError("runner source manifest does not contain the bound profile run")
    run = runs[0]
    if run.get("resources", {}).get("gpu") != config["resources"]["gpu"]:
        raise ValueError("runner source manifest GPU does not match resolved run config")
    if run.get("task") != config.get("task"):
        raise ValueError("runner source manifest task does not match resolved run config")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("runner source manifest experiment_id is required")
    return {
        "experiment_id": experiment_id,
        "manifest_path": str(manifest_path.resolve()),
        "declared_gates": phases[0].get("gates", []),
    }


def _verify_profile_phase(
    source_manifest_path: Path,
    phase_id: str,
    declared_gates: list[dict[str, Any]],
) -> str:
    if not isinstance(declared_gates, list) or not declared_gates:
        raise ValueError("profile phase must declare stop-on-failure gates")
    phase_result_path = source_manifest_path.parent / "phases" / phase_id / "result.json"
    if not phase_result_path.is_file():
        raise ValueError("passed profile phase result is required before H100 materialization")
    result = _read_strict_json(phase_result_path)
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != "jump.phase-result/v1"
        or result.get("phase_id") != phase_id
        or result.get("status") != "passed"
    ):
        raise ValueError("profile phase result must be a passed jump.phase-result/v1")
    declared_ids = [gate.get("id") for gate in declared_gates]
    if any(not isinstance(gate_id, str) or not gate_id for gate_id in declared_ids):
        raise ValueError("profile manifest gate IDs must be nonempty strings")
    emitted = result.get("gates")
    if not isinstance(emitted, list) or len(emitted) != len(declared_ids):
        raise ValueError("profile phase result must report every declared gate")
    emitted_by_id = {
        gate.get("id"): gate for gate in emitted if isinstance(gate, dict) and gate.get("id")
    }
    if set(emitted_by_id) != set(declared_ids):
        raise ValueError("profile phase result gate IDs do not match the manifest")
    for declared in declared_gates:
        actual = emitted_by_id[declared["id"]]
        for key in ("metric", "aggregation", "operator", "threshold"):
            expected = declared.get(key, "mean" if key == "aggregation" else None)
            if actual.get(key) != expected:
                raise ValueError(f"profile gate {declared['id']} does not match declared {key}")
        if actual.get("filters", {}) != declared.get("filters", {}):
            raise ValueError(f"profile gate {declared['id']} filters do not match the manifest")
        if actual.get("passed") is not True:
            raise ValueError(f"profile gate {declared['id']} did not pass")
        gate_value = actual.get("actual")
        samples = actual.get("samples")
        if (
            isinstance(gate_value, bool)
            or not isinstance(gate_value, (int, float))
            or not math.isfinite(gate_value)
            or isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples <= 0
        ):
            raise ValueError(f"profile gate {declared['id']} lacks finite evaluated evidence")
    return hashlib.sha256(phase_result_path.read_bytes()).hexdigest()


def _read_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant is forbidden: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize a locked H100 manifest from bound profiles")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--l40s-result", type=Path, required=True)
    parser.add_argument("--a100-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    template = _read_strict_json(args.template)
    manifest, decision = materialize_h100_manifest(
        template,
        l40s_result_path=args.l40s_result,
        a100_result_path=args.a100_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"selected_gpu": decision.selected_gpu, "reason": decision.reason}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
