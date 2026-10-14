# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Generate the inventory of Sigima computation functions
======================================================

Development tool (not part of the runtime): lists every computation function
exported by ``sigima.proc.signal`` and ``sigima.proc.image`` with its call shape,
output type, global options read, validation test and declared contract. With
``--desktop`` and ``--web``, it also reads (without importing them) the patterns
declared by DataLab Desktop (``register_*`` calls) and DataLab-Web (overrides)
and reports their differences with Sigima.

Usage::

    python scripts/contract_inventory.py [--desktop PATH] [--web PATH] [-o FILE]

The output is a generated Markdown report; it is never edited by hand.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os.path as osp
import re
import subprocess
import sys
from collections import Counter
from typing import Any, Callable

import sigima.proc.image as sipi
import sigima.proc.signal as sips
from sigima.proc.decorator import get_computation_metadata, is_computation_function

OPTION_RE = re.compile(r"options\.(\w+)\.(?:get|context)\(")
FAMILIES = (("signal", sips), ("image", sipi))


def _annotation(annot: Any) -> str:
    if annot is inspect.Signature.empty:
        return "?"
    if isinstance(annot, str):
        return annot
    return getattr(annot, "__name__", str(annot))


def _source(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""


def _options_read(func: Callable) -> tuple[list[str], list[str]]:
    """Return (options read by the function, options read by parameter defaults)."""
    raw = inspect.unwrap(func)
    in_func = sorted(set(OPTION_RE.findall(_source(raw))))
    in_params: set[str] = set()
    for param in inspect.signature(raw).parameters.values():
        annot = param.annotation
        if isinstance(annot, str):
            annot = raw.__globals__.get(annot.split(" ")[0], None)
        if inspect.isclass(annot):
            in_params.update(OPTION_RE.findall(_source(annot)))
    return in_func, sorted(in_params)


def _validation_tests() -> dict[str, str]:
    """Return {function_name@family: test_name} using sigima.proc.validation."""
    # pylint: disable=import-outside-toplevel
    import sigima.tests as tests_pkg
    from sigima.proc.validation import (
        generate_valid_test_names_for_function,
        get_validation_tests,
    )

    tests = get_validation_tests(tests_pkg)
    found: dict[str, str] = {}
    for family, module in FAMILIES:
        for name in module.__all__:
            names = set(
                generate_valid_test_names_for_function(f"sigima.proc.{family}", name)
            )
            for test, _path, _line in tests:
                if test in names:
                    found[f"{name}@{family}"] = test
                    break
    return found


def catalog() -> list[dict[str, Any]]:
    """Return one record per exported computation function."""
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    tests = _validation_tests()
    for family, module in FAMILIES:
        for name in module.__all__:
            func = getattr(module, name, None)
            if not (inspect.isfunction(func) and is_computation_function(func)):
                continue
            if id(func) in seen:
                continue
            seen.add(id(func))
            sig = inspect.signature(func)
            args = [
                f"{p.name}: {_annotation(p.annotation)}"
                for p in sig.parameters.values()
                if p.kind != inspect.Parameter.KEYWORD_ONLY
            ]
            metadata = get_computation_metadata(func)
            in_func, in_params = _options_read(func)
            records.append(
                {
                    "family": family,
                    "name": name,
                    "module": inspect.unwrap(func).__module__,
                    "args": args,
                    "returns": _annotation(sig.return_annotation),
                    "operation_id": metadata.operation_id,
                    "contract_version": metadata.contract_version,
                    "options": in_func,
                    "param_options": in_params,
                    "validation_test": tests.get(f"{name}@{family}"),
                }
            )
    return records


GENERIC_PARAM_RE = re.compile(r"\b(?!ROI[12]DParam\b)\w+Param\b")


def _shape(record: dict[str, Any], generic: bool = False) -> str:
    """Return the call shape; *generic* replaces parameter classes by ``Param``."""
    types = [arg.split(": ", 1)[1] for arg in record["args"]]
    shape = f"({', '.join(types)}) -> {record['returns']}"
    return GENERIC_PARAM_RE.sub("Param", shape) if generic else shape


def _contract_status() -> dict[str, str]:
    """Return {function_name@family: contract inference outcome}."""
    # pylint: disable=import-outside-toplevel
    from sigima.proc import contracts

    status: dict[str, str] = {}
    for family, module in FAMILIES:
        for name in module.__all__:
            func = getattr(module, name, None)
            if not (inspect.isfunction(func) and is_computation_function(func)):
                continue
            try:
                contract = contracts.build_contract_registry([func]).for_function(func)
            except contracts.ContractError as exc:
                status[f"{name}@{family}"] = f"error: {exc}".split(":", 2)[0]
                continue
            if contract is None:
                status[f"{name}@{family}"] = "no identity"
            else:
                status[f"{name}@{family}"] = f"{contract.pattern}" + (
                    " qualified" if getattr(contract, "qualified", False) else ""
                )
    return status


def _call_target(node: ast.expr) -> str | None:
    """Return the Sigima function name referenced by *node* (``sips.name``)."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr
    return None


def _literal(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)


def desktop_patterns(root: str) -> dict[str, dict[str, Any]]:
    """Read ``register_*`` calls of DataLab Desktop processors (AST only)."""
    found: dict[str, dict[str, Any]] = {}
    for family in ("signal", "image"):
        path = osp.join(root, "datalab", "gui", "processor", f"{family}.py")
        with open(path, encoding="utf-8") as fdesc:
            tree = ast.parse(fdesc.read())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith("register_")
                and node.args
            ):
                continue
            name = _call_target(node.args[0])
            if name is None:
                continue
            keywords = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
            found[f"{name}@{family}"] = {
                "pattern": node.func.attr.removeprefix("register_"),
                "skip_xarray_compat": bool(keywords.get("skip_xarray_compat", False)),
                "line": node.lineno,
            }
    return found


def web_patterns(root: str) -> dict[str, dict[str, Any]]:
    """Read DataLab-Web feature overrides (AST only)."""
    path = osp.join(root, "src", "runtime", "processor.py")
    with open(path, encoding="utf-8") as fdesc:
        tree = ast.parse(fdesc.read())
    found: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target not in ("SIGNAL_OVERRIDES", "IMAGE_OVERRIDES"):
            continue
        family = "signal" if target == "SIGNAL_OVERRIDES" else "image"
        assert isinstance(value, ast.Dict)
        for key, call in zip(value.keys, value.values):
            name = _literal(key)
            keywords = (
                {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg}
                if isinstance(call, ast.Call)
                else {}
            )
            found[f"{name}@{family}"] = {
                "pattern": keywords.get("pattern"),
                "skip_xarray_compat": bool(keywords.get("skip_xarray_compat", False)),
                "output_kind": keywords.get("output_kind"),
                "line": key.lineno,
            }
    return found


def _git_revision(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def _cell(value: Any) -> str:
    if value in (None, "", [], False):
        return "—"
    if value is True:
        return "yes"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value).replace("|", "\\|")


def render(
    records: list[dict[str, Any]],
    status: dict[str, str],
    desktop: dict[str, dict[str, Any]] | None,
    web: dict[str, dict[str, Any]] | None,
    revisions: dict[str, str],
) -> str:
    """Render the inventory as Markdown."""
    lines = ["# Inventaire du catalogue de calcul Sigima", ""]
    lines.append(
        "Généré par `Sigima/scripts/contract_inventory.py` — ne pas modifier à la "
        "main. Révisions : "
        + ", ".join(f"{repo} `{rev}`" for repo, rev in revisions.items())
        + "."
    )
    lines.append("")
    counts = Counter(r["family"] for r in records)
    with_test = sum(1 for r in records if r["validation_test"])
    with_id = sum(1 for r in records if r["operation_id"])
    lines.append(
        f"Fonctions : {len(records)} ({counts['signal']} signal, "
        f"{counts['image']} image) ; avec test de validation : {with_test} ; "
        f"avec identité déclarée : {with_id}."
    )
    lines += [
        "",
        "## Formes d’appel",
        "",
        "| Nombre | Famille | Forme |",
        "| --- | --- | --- |",
    ]
    shapes = Counter((r["family"], _shape(r, generic=True)) for r in records)
    for (family, shape), count in sorted(shapes.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {count} | {family} | `{_cell(shape)}` |")
    lines += ["", "## Options globales lues", ""]
    for r in records:
        if r["options"] or r["param_options"]:
            lines.append(
                f"- `{r['family']}.{r['name']}` : calcul {_cell(r['options'])} ; "
                f"défauts de paramètres {_cell(r['param_options'])}"
            )
    header = "| Fonction | Forme | Identité | Contrat | Test de validation |"
    sep = "| --- | --- | --- | --- | --- |"
    if desktop is not None:
        header += " Desktop | Desktop skip_x |"
        sep += " --- | --- |"
    if web is not None:
        header += " Web | Web skip_x | Web sortie |"
        sep += " --- | --- | --- |"
    lines += ["", "## Catalogue", "", header, sep]
    for r in records:
        key = f"{r['name']}@{r['family']}"
        identity = (
            f"{r['operation_id']} v{r['contract_version']}"
            if r["operation_id"]
            else None
        )
        row = (
            f"| `{r['family']}.{r['name']}` | `{_cell(_shape(r))}` | {_cell(identity)} "
            f"| {_cell(status.get(key))} | {_cell(r['validation_test'])} |"
        )
        if desktop is not None:
            d = desktop.get(key, {})
            row += (
                f" {_cell(d.get('pattern'))} | {_cell(d.get('skip_xarray_compat'))} |"
            )
        if web is not None:
            w = web.get(key)
            web_pattern = None if w is None else (w.get("pattern") or "inferred")
            row += (
                f" {_cell(web_pattern)} | {_cell((w or {}).get('skip_xarray_compat'))} "
                f"| {_cell((w or {}).get('output_kind'))} |"
            )
        lines.append(row)
    if desktop is not None or web is not None:
        names = {f"{r['name']}@{r['family']}" for r in records}
        lines += ["", "## Écarts entre éditions", ""]
        for label, data in (("Desktop", desktop), ("Web", web)):
            if data is None:
                continue
            unknown = sorted(set(data) - names)
            missing = sorted(names - set(data))
            lines.append(
                f"- {label} : {len(data)} fonctions déclarées ; inconnues de Sigima : "
                f"{_cell(unknown)} ; non exposées ({len(missing)}) : "
                f"{_cell([k.replace('@', ' ') for k in missing])}"
            )
        if desktop is not None and web is not None:
            for key in sorted(set(desktop) & set(web)):
                d_skip = desktop[key]["skip_xarray_compat"]
                w_skip = web[key]["skip_xarray_compat"]
                if d_skip != w_skip:
                    lines.append(
                        f"- `{key}` : skip_xarray_compat Desktop={d_skip}, Web={w_skip}"
                    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--desktop", help="DataLab Desktop repository root")
    parser.add_argument("--web", help="DataLab-Web repository root")
    parser.add_argument("-o", "--output", help="Output Markdown file (default: stdout)")
    args = parser.parse_args()
    records = catalog()
    status = _contract_status()
    sigima_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    revisions = {"Sigima": _git_revision(sigima_root)}
    desktop = web = None
    if args.desktop:
        desktop = desktop_patterns(args.desktop)
        revisions["DataLab"] = _git_revision(args.desktop)
    if args.web:
        web = web_patterns(args.web)
        revisions["DataLab-Web"] = _git_revision(args.web)
    text = render(records, status, desktop, web, revisions)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fdesc:
            fdesc.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
