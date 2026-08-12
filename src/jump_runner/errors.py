class RunnerError(Exception):
    """Base error with a safe, user-facing message."""


class ManifestError(RunnerError):
    """The manifest is invalid or violates a preregistered constraint."""


class BudgetError(ManifestError):
    """The planned experiment exceeds a declared phase ceiling."""


class GateFailed(RunnerError):
    """A preregistered gate failed and downstream execution was stopped."""


class ImmutableOutputError(RunnerError):
    """An immutable output already exists with different content."""
