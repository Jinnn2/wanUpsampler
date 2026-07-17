from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "outputs/aaai27_experiments"
MEDIA_SUFFIXES = {".avi", ".gif", ".mkv", ".mov", ".mp4", ".webm"}


def main() -> None:
    args = parse_args()
    output = export_results(
        project_root=Path(args.project_root).resolve(),
        inventory_path=Path(args.inventory).resolve(),
        output_root=Path(args.output_root).resolve(),
        allowed_missing=set(args.allow_missing),
        include_videos=args.include_videos,
        include_checkpoints=args.include_checkpoints,
        include_private=args.include_private,
        include_logs=args.include_logs,
        include_code=not args.no_code,
    )
    print(f"Export directory: {output}")
    if args.archive:
        archive = make_archive(output)
        print(f"Export archive  : {archive}")


def export_results(
    *,
    project_root: Path,
    inventory_path: Path,
    output_root: Path,
    allowed_missing: set[str],
    include_videos: bool = False,
    include_checkpoints: bool = False,
    include_private: bool = False,
    include_logs: bool = False,
    include_code: bool = True,
) -> Path:
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite an existing export: {output_root}")
    inventory = load_json(inventory_path)
    if int(inventory.get("schema_version", 0)) < 2:
        raise SystemExit(f"Inventory schema_version must be at least 2: {inventory_path}")
    validate_issue_allowlist(inventory, allowed_missing)

    results_root = Path(inventory["canonical_results_root"]).resolve()
    if is_relative_to(output_root, results_root) or is_relative_to(output_root, project_root / "outputs"):
        raise SystemExit("Export output must be outside the result trees being copied")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as raw_staging:
        staging = Path(raw_staging) / output_root.name
        staging.mkdir()
        path_map: list[dict[str, str]] = []
        copy_core(inventory_path, staging, path_map)
        copy_source_trees(inventory, project_root, results_root, staging, path_map, include_videos, include_private)
        copy_factorials(inventory, staging, path_map, include_videos, include_private)
        copy_ablations(inventory, project_root, results_root, staging, path_map, include_videos, include_private)
        copy_task_state(results_root, staging, path_map, include_logs)
        if include_checkpoints:
            copy_checkpoints(inventory, staging, path_map)
        if include_code:
            copy_tracked_code(project_root, staging, path_map)

        exclusions = {
            "schema_version": 1,
            "decision": "export_with_intentional_omissions" if allowed_missing else "complete_export",
            "allowed_missing": sorted(allowed_missing),
            "reason": "Experiments intentionally not run before final export" if allowed_missing else "",
        }
        write_json(staging / "core/declared_exclusions.json", exclusions)
        provenance = staging / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        write_git_provenance(project_root, provenance)
        write_json(
            provenance / "environment.json",
            {
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "project_root": str(project_root),
                "inventory": str(inventory_path),
            },
        )
        write_json(provenance / "path_map.json", {"files": path_map})
        manifest = {
            "schema_version": 1,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "inventory_schema_version": inventory["schema_version"],
            "inventory_generated_at_utc": inventory.get("generated_at_utc"),
            "allowed_missing": sorted(allowed_missing),
            "options": {
                "include_videos": include_videos,
                "include_checkpoints": include_checkpoints,
                "include_private": include_private,
                "include_logs": include_logs,
                "include_code": include_code,
            },
            "copied_entries": len(path_map),
        }
        write_json(staging / "export_manifest.json", manifest)
        write_checksums(staging)
        staging.replace(output_root)
    return output_root


def validate_issue_allowlist(inventory: dict[str, Any], allowed_missing: set[str]) -> None:
    issue_items = {str(item.get("item", "")) for item in inventory.get("issues", [])}
    unexpected = sorted(issue_items - allowed_missing)
    absent = sorted(allowed_missing - issue_items)
    if unexpected or absent:
        details = []
        if unexpected:
            details.append("unexpected issues: " + ", ".join(unexpected))
        if absent:
            details.append("allowed issues not present: " + ", ".join(absent))
        raise SystemExit("Issue allowlist mismatch; " + "; ".join(details))


def copy_core(inventory_path: Path, staging: Path, path_map: list[dict[str, str]]) -> None:
    root = inventory_path.parent
    for name in ("result_inventory.json", "paper_tables.md"):
        source = root / name
        if not source.is_file():
            raise SystemExit(f"Required collection output is missing: {source}")
        copy_item(source, staging / "core" / name, staging, path_map)
    compiled = root / "compiled_tables"
    if not compiled.is_dir():
        raise SystemExit(f"Required compiled table directory is missing: {compiled}")
    copy_item(compiled, staging / "core/compiled_tables", staging, path_map)


def copy_source_trees(
    inventory: dict[str, Any],
    project_root: Path,
    results_root: Path,
    staging: Path,
    path_map: list[dict[str, str]],
    include_videos: bool,
    include_private: bool,
) -> None:
    copied: set[tuple[str, str]] = set()
    for name, source in inventory.get("sources", {}).items():
        if source.get("status") != "complete" or not source.get("path"):
            continue
        raw_path = str(source["path"])
        # Collection-generated summaries live in result_inventory.json and
        # compiled_tables; their path is descriptive, not a filesystem path.
        # The corresponding raw source is a separate entry and is copied by
        # this same loop.
        if raw_path.startswith("derived from "):
            continue
        path = Path(raw_path).resolve()
        if not path.exists():
            raise SystemExit(f"Inventory marks source complete but its path is missing: {name}: {path}")
        tree, destination = source_tree_destination(path, name, project_root, results_root, staging)
        key = (str(tree), str(destination))
        if key in copied:
            continue
        copy_item(
            tree,
            destination,
            staging,
            path_map,
            include_videos=include_videos,
            include_private=include_private,
        )
        copied.add(key)


def source_tree_destination(
    path: Path, name: str, project_root: Path, results_root: Path, staging: Path
) -> tuple[Path, Path]:
    if is_relative_to(path, results_root):
        relative = path.relative_to(results_root)
        if len(relative.parts) > 1:
            tree = results_root / relative.parts[0]
            return tree, staging / "evidence/canonical" / relative.parts[0]
        return path, staging / "evidence/canonical" / path.name
    legacy_root = (project_root / "outputs").resolve()
    if is_relative_to(path, legacy_root):
        relative = path.relative_to(legacy_root)
        if len(relative.parts) > 1:
            tree = legacy_root / relative.parts[0]
            return tree, staging / "evidence/legacy" / relative.parts[0]
    return path, staging / "evidence/external_sources" / safe_name(name) / path.name


def copy_factorials(
    inventory: dict[str, Any],
    staging: Path,
    path_map: list[dict[str, str]],
    include_videos: bool,
    include_private: bool,
) -> None:
    for family, result in inventory.get("factorials", {}).items():
        root = Path(result.get("root", ""))
        if result.get("status") != "complete":
            continue
        if not root.is_dir():
            raise SystemExit(f"Complete factorial root is missing: {family}: {root}")
        copy_item(
            root,
            staging / "evidence/factorials" / safe_name(family),
            staging,
            path_map,
            include_videos=include_videos,
            include_private=include_private,
        )


def copy_ablations(
    inventory: dict[str, Any],
    project_root: Path,
    results_root: Path,
    staging: Path,
    path_map: list[dict[str, str]],
    include_videos: bool,
    include_private: bool,
) -> None:
    for name, result in inventory.get("ablations", {}).items():
        if result.get("status") != "complete" or not result.get("root"):
            continue
        path = Path(result["root"]).resolve()
        if not path.exists():
            raise SystemExit(f"Complete ablation root is missing: {name}: {path}")
        tree, _ = source_tree_destination(path, name, project_root, results_root, staging)
        copy_item(
            tree,
            staging / "evidence/ablations" / safe_name(name),
            staging,
            path_map,
            include_videos=include_videos,
            include_private=include_private,
        )


def copy_task_state(
    results_root: Path, staging: Path, path_map: list[dict[str, str]], include_logs: bool
) -> None:
    state = results_root / "_state"
    if state.is_dir():
        copy_item(state, staging / "provenance/task_state", staging, path_map, include_logs=include_logs)


def copy_checkpoints(inventory: dict[str, Any], staging: Path, path_map: list[dict[str, str]]) -> None:
    candidates = {str(value) for value in inventory.get("final_configuration", {}).values() if looks_like_checkpoint(value)}
    for result in inventory.get("factorials", {}).values():
        root = Path(result.get("root", ""))
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = load_json(manifest_path)
        artifacts = list(manifest.get("lora_artifacts", {}).values()) + [manifest.get("stage2_artifact", {})]
        for artifact in artifacts:
            if isinstance(artifact, dict) and artifact.get("resolved_path"):
                candidates.add(str(artifact["resolved_path"]))
    for raw_path in sorted(candidates):
        source = Path(raw_path).resolve()
        if not source.is_file():
            raise SystemExit(f"Referenced checkpoint is missing: {source}")
        digest = sha256(source)
        destination = staging / "models" / f"{digest[:16]}_{source.name}"
        copy_item(source, destination, staging, path_map)


def copy_tracked_code(project_root: Path, staging: Path, path_map: list[dict[str, str]]) -> None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=project_root, check=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError):
        return
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        source = project_root / relative
        if source.is_file():
            copy_item(source, staging / "provenance/code" / relative, staging, path_map)


def copy_item(
    source: Path,
    destination: Path,
    staging: Path,
    path_map: list[dict[str, str]],
    *,
    include_videos: bool = True,
    include_private: bool = True,
    include_logs: bool = True,
) -> None:
    if source.is_file():
        if should_skip(source, include_videos, include_private, include_logs):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_file() and sha256(source) == sha256(destination):
                return
            raise SystemExit(f"Export destination collision: {destination}")
        shutil.copy2(source, destination)
        path_map.append({"source": str(source), "exported": destination.relative_to(staging).as_posix()})
        return
    if not source.is_dir():
        raise SystemExit(f"Cannot export missing path: {source}")
    for path in sorted(source.rglob("*")):
        if not path.is_file() or should_skip(path, include_videos, include_private, include_logs, source):
            continue
        copy_item(path, destination / path.relative_to(source), staging, path_map)


def should_skip(
    path: Path,
    include_videos: bool,
    include_private: bool,
    include_logs: bool,
    root: Path | None = None,
) -> bool:
    relative_parts = path.relative_to(root).parts if root and is_relative_to(path, root) else path.parts
    if not include_videos and path.suffix.lower() in MEDIA_SUFFIXES:
        return True
    if not include_private and "_private" in relative_parts:
        return True
    if not include_logs and path.suffix.lower() == ".log":
        return True
    return False


def write_git_provenance(project_root: Path, destination: Path) -> None:
    for name, command in (
        ("git_commit.txt", ["git", "rev-parse", "HEAD"]),
        ("git_status.txt", ["git", "status", "--short"]),
        ("git_diff.patch", ["git", "diff", "--binary", "HEAD"]),
    ):
        try:
            result = subprocess.run(command, cwd=project_root, check=True, capture_output=True)
            (destination / name).write_bytes(result.stdout)
        except (OSError, subprocess.CalledProcessError) as exc:
            (destination / name).write_text(f"UNAVAILABLE: {exc}\n", encoding="utf-8")


def write_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_archive(export_root: Path) -> Path:
    archive = export_root.with_suffix(".tar.gz")
    if archive.exists():
        raise SystemExit(f"Refusing to overwrite an existing archive: {archive}")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(export_root, arcname=export_root.name, recursive=True)
    return archive


def looks_like_checkpoint(value: Any) -> bool:
    return isinstance(value, str) and Path(value).suffix.lower() in {".ckpt", ".pt", ".pth", ".safetensors"}


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid or missing JSON {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a verified, self-describing AAAI-27 result bundle.")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--inventory", default=str(DEFAULT_RESULTS_ROOT / "result_inventory.json"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-missing", action="append", default=[])
    parser.add_argument("--include-videos", action="store_true")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--include-logs", action="store_true")
    parser.add_argument("--no-code", action="store_true")
    parser.add_argument("--archive", action="store_true", help="Also create a .tar.gz archive beside the export directory.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
