"""Mechanistic experiment primitives for the JUMP research program."""

from .capture import ActivationCapture, CapturePolicy, Timepoint
from .boundaries import BoundaryManifest, ResolvedPrompt, resolve_boundaries
from .runtime import HookManifest, MechanisticRuntime, PatchMode, WorldModelRuntimeBinding
from .stage_d import (
    LATENT_PERMUTATION_VERSION,
    STAGE_D_ARMS,
    STAGE_D_CONTROL_VERSION,
    STAGE_D_EXECUTION_CONTRACT_SHA256,
    StageDArmInput,
    StageDControlEvidence,
    StageDControlSpec,
    build_stage_d_control_result,
    execute_stage_d_control_set,
    identity_injection,
    no_z_injection,
    scrambled_injection,
)

__all__ = [
    "ActivationCapture",
    "CapturePolicy",
    "Timepoint",
    "BoundaryManifest",
    "ResolvedPrompt",
    "resolve_boundaries",
    "HookManifest",
    "MechanisticRuntime",
    "PatchMode",
    "WorldModelRuntimeBinding",
    "LATENT_PERMUTATION_VERSION",
    "STAGE_D_ARMS",
    "STAGE_D_CONTROL_VERSION",
    "STAGE_D_EXECUTION_CONTRACT_SHA256",
    "StageDArmInput",
    "StageDControlEvidence",
    "StageDControlSpec",
    "build_stage_d_control_result",
    "execute_stage_d_control_set",
    "identity_injection",
    "no_z_injection",
    "scrambled_injection",
]

__version__ = "0.1.0"
