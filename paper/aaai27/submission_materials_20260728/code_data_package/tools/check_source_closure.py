#!/usr/bin/env python3
"""Check that every internal Python import in the review snapshot is present."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


INTERNAL_PREFIXES = (
    "wan_sr",
    "changing_resolution",
    "changing_resolution_distill",
    "paper.aaai27.experiments",
)


def module_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_relative(current: str, level: int, target: str | None, is_package: bool) -> str:
    package = current.split(".") if is_package else current.split(".")[:-1]
    if level > 1:
        package = package[: -(level - 1)]
    if target:
        package.extend(target.split("."))
    return ".".join(package)


def exists_as_module(root: Path, name: str) -> bool:
    rel = Path(*name.split("."))
    return (root / rel).with_suffix(".py").is_file() or (root / rel / "__init__.py").is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "code",
    )
    args = parser.parse_args()
    root = args.code_root.resolve()
    missing: set[tuple[str, str]] = set()

    for path in sorted(root.rglob("*.py")):
        current = module_for(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = resolve_relative(
                        current,
                        node.level,
                        node.module,
                        path.name == "__init__.py",
                    )
                else:
                    base = node.module or ""
                names.append(base)
                for alias in node.names:
                    if alias.name != "*":
                        names.append(f"{base}.{alias.name}" if base else alias.name)
            for name in names:
                if not name.startswith(INTERNAL_PREFIXES):
                    continue
                if exists_as_module(root, name):
                    continue
                # `from package import symbol` may name an attribute rather than
                # a submodule; the containing module/package is sufficient.
                parent = name.rpartition(".")[0]
                if parent and exists_as_module(root, parent):
                    continue
                missing.add((current, name))

    if missing:
        for source, name in sorted(missing):
            print(f"MISSING\t{source}\t{name}")
        return 1
    count = sum(1 for _ in root.rglob("*.py"))
    print(f"source closure OK: {count} Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
