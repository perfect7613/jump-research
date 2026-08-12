from __future__ import annotations

import json

import pytest

from jump_runner.cli import main


def write_manifest(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_dry_run_never_submits(manifest, tmp_path, capsys):
    path = write_manifest(tmp_path, manifest)
    assert main(["dry-run", str(path), "--smoke"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["will_submit"] is False
    assert output["mode"] == "smoke"


def test_full_submit_has_two_part_lock(manifest, tmp_path, capsys):
    path = write_manifest(tmp_path, manifest)
    assert main(["submit", str(path)]) == 2
    assert "full matrix submission is locked" in capsys.readouterr().err
