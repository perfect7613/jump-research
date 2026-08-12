from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .errors import RunnerError
from .executor import read_status, run_manifest
from .manifest import authorize_launch, load_manifest, make_plan

DEFAULT_RUNS_DIR = Path(os.environ.get("JUMP_RUNS_DIR", ".jump/runs"))
APP_NAME = os.environ.get("JUMP_MODAL_APP_NAME", "jump-sequential-experiments")


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    return load_manifest(args.manifest)


def cmd_plan(args: argparse.Namespace) -> int:
    _print(make_plan(_manifest(args), smoke=args.smoke))
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    manifest = _manifest(args)
    plan = make_plan(manifest, smoke=args.smoke)
    plan["valid"] = True
    plan["will_submit"] = False
    plan["full_matrix_launch_enabled"] = bool(manifest.get("launch_policy", {}).get("allow_full_matrix", False))
    _print(plan)
    return 0


def _submit(manifest: dict[str, Any], smoke: bool, confirm_paid: bool, confirm_h100: bool) -> dict[str, Any]:
    authorize_launch(
        manifest, smoke=smoke, confirm_paid=confirm_paid, confirm_h100=confirm_h100
    )
    try:
        import modal
    except ImportError as exc:
        raise RunnerError("install the Modal extra first: pip install -e '.[modal]'") from exc
    function = modal.Function.from_name(APP_NAME, "orchestrate")
    call = function.spawn(
        manifest,
        smoke=smoke,
        confirm_paid=confirm_paid,
        confirm_h100=confirm_h100,
    )
    registry = Path(".jump/submissions")
    registry.mkdir(parents=True, exist_ok=True)
    record = {"call_id": call.object_id, "app_name": APP_NAME, "experiment_id": manifest["experiment_id"], "smoke": smoke}
    (registry / f"{call.object_id}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def cmd_submit(args: argparse.Namespace) -> int:
    _print(_submit(_manifest(args), args.smoke, args.confirm_paid, args.confirm_h100))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    # The orchestrator is idempotent: immutable completed results are skipped and
    # incomplete/eligible failed attempts continue from the newest checkpoint.
    _print(_submit(_manifest(args), args.smoke, args.confirm_paid, args.confirm_h100))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manifest = _manifest(args)
    if args.remote:
        try:
            import modal
        except ImportError as exc:
            raise RunnerError("install the Modal extra first: pip install -e '.[modal]'") from exc
        function = modal.Function.from_name(APP_NAME, "get_status")
        _print(function.remote(manifest, smoke=args.smoke))
    else:
        _print(read_status(manifest, args.runs_dir, smoke=args.smoke))
    return 0


def cmd_run_local(args: argparse.Namespace) -> int:
    if not args.smoke:
        raise RunnerError("local execution is restricted to --smoke")
    _print(run_manifest(_manifest(args), args.runs_dir, smoke=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jump-experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("plan", cmd_plan, "show ordered runs and worst-case cost"),
        ("dry-run", cmd_dry_run, "validate gates, allowlists, and budgets without execution"),
        ("submit", cmd_submit, "submit a deployed Modal orchestrator"),
        ("status", cmd_status, "read durable local/volume-compatible status"),
        ("resume", cmd_resume, "resume through the idempotent Modal orchestrator"),
        ("run-local", cmd_run_local, "execute smoke fixtures on CPU (never a paid matrix)"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("manifest")
        command.add_argument("--smoke", action="store_true", help="select only runs explicitly marked smoke_test")
        if name in {"status", "run-local"}:
            command.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
        if name == "status":
            command.add_argument("--remote", action="store_true", help="read status from the deployed Modal Volume")
        if name in {"submit", "resume"}:
            command.add_argument("--confirm-paid", action="store_true", help="second lock for an explicitly enabled full matrix")
            command.add_argument(
                "--confirm-h100",
                action="store_true",
                help="confirm reviewed H100 justification and cost forecast",
            )
        command.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return args.handler(args)
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
