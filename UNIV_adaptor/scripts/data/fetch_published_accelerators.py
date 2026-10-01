"""Fetch pinned official source snapshots without modifying the main git index.

This script downloads *code*, not model weights or Python dependencies. Existing
checkouts are inspected and never updated, cleaned, or overwritten.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "UNIV_adaptor/configs/published_accelerators_v1.json"


def git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(f"git {args[0]} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def load_manifest() -> tuple[dict, Path]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    checkout_root = (ROOT / manifest["checkout_root"]).resolve()
    if checkout_root != (ROOT / "UNIV_adaptor/external").resolve():
        raise ValueError("Unexpected checkout root")
    for record in manifest["repositories"]:
        if not re.fullmatch(r"[a-z0-9_]+", record["name"]):
            raise ValueError(f"Invalid repository name: {record['name']}")
        if not re.fullmatch(r"[0-9a-f]{40}", record["commit"]):
            raise ValueError(f"Invalid commit hash: {record['name']}")
        if not record["url"].startswith("https://github.com/"):
            raise ValueError(f"Unexpected repository URL: {record['name']}")
    return manifest, checkout_root


def check_checkout(record: dict, target: Path) -> str:
    if not target.is_dir():
        return "missing"
    try:
        if Path(git("rev-parse", "--show-toplevel", cwd=target)).resolve() != target.resolve():
            return "wrong repository root"
        origin = git("remote", "get-url", "origin", cwd=target)
        if origin.rstrip("/").removesuffix(".git") != record["url"].rstrip("/").removesuffix(".git"):
            return f"wrong origin: {origin}"
        commit = git("rev-parse", "HEAD", cwd=target)
        if commit != record["commit"]:
            return f"wrong commit: {commit}"
        if git("status", "--porcelain", "--untracked-files=all", cwd=target):
            return "modified or untracked files"
        missing = [entry for entry in record["entrypoints"] if not (target / entry).exists()]
        if missing:
            return f"missing entrypoints: {missing}"
        return "ok"
    except (OSError, RuntimeError) as exc:
        return f"invalid checkout: {exc}"


def fetch(record: dict, target: Path) -> None:
    if target.exists():
        state = check_checkout(record, target)
        if state != "ok":
            raise RuntimeError(f"Refusing to change {target}: {state}")
        print(f"{record['name']}: already pinned and clean")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"{record['name']}: cloning {record['url']} at {record['commit']}", flush=True)
    git("clone", "--filter=blob:none", "--depth=1", "--no-checkout", record["url"], str(target))
    try:
        if record.get("sparse_dirs"):
            git("sparse-checkout", "set", "--cone", *record["sparse_dirs"], cwd=target)
        try:
            git("cat-file", "-e", f"{record['commit']}^{{commit}}", cwd=target)
        except RuntimeError:
            git("fetch", "--depth=1", "origin", record["commit"], cwd=target)
        git("checkout", "--detach", record["commit"], cwd=target)
        state = check_checkout(record, target)
        if state != "ok":
            raise RuntimeError(f"Checkout verification failed: {state}")
    except Exception:
        print(f"{record['name']}: incomplete checkout preserved at {target}")
        raise
    print(f"{record['name']}: verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="show pinned source availability")
    group.add_argument("--check", action="store_true", help="verify every local checkout")
    group.add_argument("--fetch", metavar="NAME", help="fetch one named repository or 'all'")
    args = parser.parse_args()
    manifest, root = load_manifest()
    records = manifest["repositories"]
    if args.list:
        for record in records:
            print(f"{record['name']:<16} {record['family']:<42} {record['commit'][:12]} {record['url']}")
        for record in manifest["unavailable"]:
            print(f"{record['name']:<16} unavailable: {record['reason']}")
        return
    if args.check:
        failures = []
        for record in records:
            state = check_checkout(record, root / record["name"])
            print(f"{record['name']:<16} {state}")
            if state != "ok":
                failures.append(record["name"])
        if failures:
            raise SystemExit(f"Not ready: {', '.join(failures)}")
        return
    selected = records if args.fetch == "all" else [r for r in records if r["name"] == args.fetch]
    if not selected:
        raise SystemExit(f"Unknown or unavailable method: {args.fetch}")
    for record in selected:
        fetch(record, root / record["name"])


if __name__ == "__main__":
    main()
