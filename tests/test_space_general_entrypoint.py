from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_space_launches_general_workbench_with_contract_dependency():
    app = (ROOT / "space" / "app.py").read_text()
    requirements = (ROOT / "space" / "requirements.txt").read_text().splitlines()
    readme = (ROOT / "space" / "README.md").read_text()

    assert "from jump_ui.general_app import create_general_app" in app
    assert "demo = create_general_app()" in app
    assert "jsonschema==4.26.0" in requirements
    assert "supported idea" in readme
    assert "Unsupported requests fail closed" in readme
