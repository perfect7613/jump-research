"""Fail-closed policy for model-produced toy simulation programs."""

from __future__ import annotations

import ast
import hashlib
from copy import deepcopy
from typing import Any

from jump_contracts.experiments import FIXED_SANDBOX, RESTRICTED_POLICY, RESTRICTED_POLICY_SHA256

MAX_SOURCE_BYTES = RESTRICTED_POLICY["source_bytes"]
MAX_AST_NODES = RESTRICTED_POLICY["ast_nodes"]
ALLOWED_IMPORTS = tuple(RESTRICTED_POLICY["allowed_imports"])
ALLOWED_BUILTINS = tuple(RESTRICTED_POLICY["allowed_builtins"])
ALLOWED_ATTRIBUTES = tuple(RESTRICTED_POLICY["allowed_attributes"])
BANNED_NAMES = tuple(RESTRICTED_POLICY["banned_names"])
BANNED_MODULES = tuple(RESTRICTED_POLICY["banned_modules"])
POLICY = RESTRICTED_POLICY
POLICY_SHA256 = RESTRICTED_POLICY_SHA256


class SafetyError(ValueError):
    """Raised when source or input crosses the restricted runtime boundary."""


def sandbox_declaration(source: str) -> dict[str, Any]:
    """Return the fixed server-owned sandbox declaration for validated source."""
    validate_simulation_source(source)
    encoded = source.encode("utf-8")
    declaration = deepcopy(FIXED_SANDBOX)
    declaration["source"] = {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "byte_length": len(encoded),
        "media_type": "text/x-python",
    }
    declaration["policy_sha256"] = POLICY_SHA256
    return declaration


def validate_simulation_source(source: str) -> ast.Module:
    if not isinstance(source, str) or not source.strip():
        raise SafetyError("simulation source must be nonempty text")
    try:
        encoded = source.encode("utf-8")
    except UnicodeError as exc:
        raise SafetyError("simulation source must be UTF-8") from exc
    if len(encoded) > MAX_SOURCE_BYTES:
        raise SafetyError("simulation source exceeds the byte limit")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SafetyError(f"simulation source is invalid Python: {exc.msg}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise SafetyError("simulation source exceeds the AST node limit")

    top_functions: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            if isinstance(statement, ast.FunctionDef):
                top_functions.add(statement.name)
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        raise SafetyError("top level may contain only imports and function definitions")
    if "simulate" not in top_functions:
        raise SafetyError("simulation source must define simulate(plan)")

    banned_nodes = (
        ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Delete, ast.Global, ast.Lambda, ast.Nonlocal,
        ast.While, ast.With, ast.AsyncWith, ast.Yield, ast.YieldFrom,
    )
    for node in nodes:
        if isinstance(node, banned_nodes):
            raise SafetyError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise SafetyError("decorators are not allowed")
            if node.name.startswith("_"):
                raise SafetyError("private or dunder function names are not allowed")
            if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
                raise SafetyError("variadic and keyword-only function arguments are not allowed")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in BANNED_NAMES or node.id in BANNED_MODULES:
                raise SafetyError(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ALLOWED_ATTRIBUTES:
                raise SafetyError(f"attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Call):
            _validate_call(node, top_functions)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, bytes)) and len(node.value) > 4_096:
                raise SafetyError("literal exceeds the size limit")
    simulate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "simulate")
    if len(simulate.args.args) != 1 or simulate.args.args[0].arg != "plan":
        raise SafetyError("simulate must have the exact signature simulate(plan)")
    return tree


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
        aliases = [alias.asname for alias in node.names]
    else:
        if node.level:
            raise SafetyError("relative imports are not allowed")
        names = [node.module or ""]
        aliases = [alias.asname for alias in node.names]
        if any(alias.name.startswith("_") or alias.name not in ALLOWED_ATTRIBUTES for alias in node.names):
            raise SafetyError("from-import name is not in the fixed allowlist")
    if any(name.split(".", 1)[0] not in ALLOWED_IMPORTS or "." in name for name in names):
        raise SafetyError("import is not in the fixed allowlist")
    if any(alias is not None and alias.startswith("_") for alias in aliases):
        raise SafetyError("private import aliases are not allowed")


def _validate_call(node: ast.Call, top_functions: set[str]) -> None:
    if any(keyword.arg is None or keyword.arg.startswith("_") for keyword in node.keywords):
        raise SafetyError("expanded or private call arguments are not allowed")
    if isinstance(node.func, ast.Name):
        if node.func.id not in set(ALLOWED_BUILTINS) | top_functions:
            raise SafetyError(f"call to {node.func.id!r} is not allowed")
    elif isinstance(node.func, ast.Attribute):
        if node.func.attr not in ALLOWED_ATTRIBUTES:
            raise SafetyError(f"method call {node.func.attr!r} is not allowed")
    else:
        raise SafetyError("indirect calls are not allowed")


__all__ = [
    "MAX_SOURCE_BYTES", "MAX_AST_NODES", "ALLOWED_IMPORTS", "ALLOWED_BUILTINS",
    "ALLOWED_ATTRIBUTES", "POLICY", "POLICY_SHA256", "SafetyError", "sandbox_declaration",
    "validate_simulation_source",
]
