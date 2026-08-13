"""Track H deterministic benchmark and exact behavioral baseline interfaces."""

from .simulator import DatasetSpec, EpisodeSpec, generate_dataset, generate_episode
from .experiment_spec import (
    EXPERIMENT_SPEC_CONTRACT_SHA256,
    build_planned_run,
    compile_experiment_intent,
    materialize_experiment,
    validate_experiment_run,
    validate_experiment_spec,
)

__all__ = [
    "DatasetSpec",
    "EpisodeSpec",
    "EXPERIMENT_SPEC_CONTRACT_SHA256",
    "build_planned_run",
    "compile_experiment_intent",
    "generate_dataset",
    "generate_episode",
    "materialize_experiment",
    "validate_experiment_run",
    "validate_experiment_spec",
]
