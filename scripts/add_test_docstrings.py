"""Inject docstrings into test functions/classes that lack one.

This is a one-shot helper used to bring the new test files into compliance with
:mod:`pylint`'s ``missing-function-docstring`` / ``missing-class-docstring``
rules. The generated docstring is derived from:

1. The leading inline ``#`` comment block immediately preceding the ``def`` /
   ``class`` line (if any) -- this is typically the most informative summary.
2. Otherwise, the function/class name itself, converted to a human-readable
   sentence (``test_read_signal_real`` -> ``Test read signal real``).

The script must be idempotent and must not modify functions that already have
a docstring.
"""

from __future__ import annotations

import ast
import re
import sys
import tokenize
from io import BytesIO
from pathlib import Path


def _humanize(name: str) -> str:
    """Convert ``test_foo_bar`` / ``FooBar`` to ``foo bar`` (lower)."""
    if name.startswith("test_"):
        name = name[5:]
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return name.replace("_", " ").strip().lower()


def _docstring_for_function(name: str, leading_comment: str | None) -> str:
    """Build a Google-style one-line docstring for a test function."""
    if leading_comment:
        text = leading_comment.strip().rstrip(".") + "."
        # Capitalise first letter.
        return text[0].upper() + text[1:] if text else text
    sentence = _humanize(name)
    if name.startswith("_"):
        return f"Test helper: {sentence}."
    return f"Verify the ``{sentence}`` scenario."


def _docstring_for_class(name: str, leading_comment: str | None) -> str:
    """Build a one-line docstring for a (helper) class."""
    if leading_comment:
        text = leading_comment.strip().rstrip(".") + "."
        return text[0].upper() + text[1:] if text else text
    return f"Test helper class ``{name}``."


def _collect_leading_comments(source: str) -> dict[int, str]:
    """Map ``def``/``class`` lineno -> aggregated leading comment lines.

    A "leading comment" is a contiguous block of ``# ...`` lines immediately
    above the definition (no blank line in between).
    """
    lines = source.splitlines()
    out: dict[int, str] = {}
    for idx, raw in enumerate(lines):
        stripped = raw.lstrip()
        if not (stripped.startswith("def ") or stripped.startswith("class ")):
            continue
        # Walk upwards collecting contiguous comment lines.
        comments: list[str] = []
        j = idx - 1
        while j >= 0:
            prev = lines[j].lstrip()
            if prev.startswith("#"):
                # Skip section banners ("=========", "---------").
                body = prev.lstrip("#").strip()
                if body and not set(body) <= {"=", "-", "*"}:
                    comments.insert(0, body)
                j -= 1
            elif prev == "":
                break
            else:
                break
        if comments:
            out[idx + 1] = " ".join(comments)
    return out


def _has_docstring(node: ast.AST) -> bool:
    """Return ``True`` if the node already starts with a string literal."""
    body = getattr(node, "body", None)
    if not body:
        return False
    first = body[0]
    return (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )


def _process_file(path: Path) -> int:
    """Inject docstrings in ``path`` and return the number of insertions."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    leading = _collect_leading_comments(source)

    # Collect insertions as (line_to_insert_before, text).
    insertions: list[tuple[int, str, str]] = []  # (lineno, indent, doc)

    def _insert_lineno(body0: ast.AST) -> int:
        """Line at which to insert the docstring (1-based).

        If ``body0`` is a decorated def/class, its ``lineno`` is the def/class
        keyword line, so we must insert before the decorator instead.
        """
        deco_list = getattr(body0, "decorator_list", None) or []
        if deco_list:
            return int(min(d.lineno for d in deco_list))
        return int(body0.lineno)

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _has_docstring(child):
                    body0 = child.body[0]
                    indent = " " * (body0.col_offset)
                    comment = leading.get(child.lineno)
                    doc = _docstring_for_function(child.name, comment)
                    insertions.append((_insert_lineno(body0), indent, doc))
                visit(child)
            elif isinstance(child, ast.ClassDef):
                if not _has_docstring(child):
                    body0 = child.body[0]
                    indent = " " * (body0.col_offset)
                    comment = leading.get(child.lineno)
                    doc = _docstring_for_class(child.name, comment)
                    insertions.append((_insert_lineno(body0), indent, doc))
                visit(child)
            else:
                visit(child)

    visit(tree)
    if not insertions:
        return 0

    # Apply from bottom to top to keep line numbers stable.
    lines = source.splitlines(keepends=True)
    for lineno, indent, doc in sorted(insertions, key=lambda t: -t[0]):
        # Insert ``"""<doc>"""`` line at position ``lineno - 1`` (1-based).
        new_line = f'{indent}"""{doc}"""\n'
        lines.insert(lineno - 1, new_line)

    path.write_text("".join(lines), encoding="utf-8")
    return len(insertions)


def main(argv: list[str]) -> int:
    """Entry point: process every file passed on the command line."""
    total = 0
    for arg in argv[1:]:
        p = Path(arg)
        if not p.is_file():
            continue
        n = _process_file(p)
        if n:
            print(f"  + {n:3d}  {p}")
            total += n
    print(f"Inserted {total} docstrings.")
    # Use tokenize to silence unused-import warnings (kept for future extension).
    _ = tokenize, BytesIO
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
