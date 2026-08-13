from pathlib import Path

import pytest

from jump_workbench.safety import SafetyError, validate_simulation_source
from jump_workbench.workflow import WorkbenchError, validate_user_intent


@pytest.mark.parametrize("name", ["math", "random", "statistics", "collections", "heapq"])
def test_fixed_toy_simulation_imports_are_allowed(name):
    validate_simulation_source(f"import {name}\n\ndef simulate(plan):\n    return {{'measurements': []}}\n")


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef simulate(plan):\n return {'measurements': []}",
        "def simulate(plan):\n return open('/tmp/x')",
        "def simulate(plan):\n return eval('1')",
        "def simulate(plan):\n return plan.__class__",
        "def simulate(plan):\n exec('x=1')\n return {'measurements': []}",
        "def simulate(plan):\n while True: pass",
    ],
)
def test_file_network_process_dynamic_and_dunder_surfaces_are_rejected(source):
    with pytest.raises(SafetyError):
        validate_simulation_source(source)


@pytest.mark.parametrize(
    "intent",
    [
        "Download current traffic data from https://example.com and analyze it",
        "```python\nimport requests\n```",
        "Run a clinical trial on patients",
        "Read ~/private.csv and simulate it",
    ],
)
def test_user_input_is_inert_and_simulation_only(intent):
    with pytest.raises(WorkbenchError):
        validate_user_intent(intent)


def test_modal_boundary_has_required_isolation_controls_and_no_mounts():
    source = (Path(__file__).parents[1] / "src/jump_workbench/modal_app.py").read_text()
    for control in (
        "restrict_modal_access=True", "single_use_containers=True", "block_network=True",
        "cpu=1.0", "memory=512", "timeout=30",
    ):
        assert control in source
    decorator = source[source.index("@app.function("):source.index("def execute_restricted_simulation")]
    assert "secrets=" not in decorator
    assert "volumes=" not in decorator
    assert ".pip_install(" not in source[:source.index("gateway_image")]
