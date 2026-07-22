from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.aaai27.experiments.collect_results import (  # noqa: E402
    MIN_VALID_VIDEO_BYTES,
    inspect_factorial,
)


EXPECTED_CASES = (
    "full_hr50",
    "lightx2v_cr40",
    "lightx2v_cr45",
    "lightx2v_cr48",
    "talh40",
    "talh45",
    "full_lr50_stage2_0hr",
    "full_lr50_stage2_1hr",
    "full_lr50_stage2_2hr",
    "full_lr50_stage2_5hr",
    "ralu_quality",
)
OBSOLETE_RALU_CASES = ("ralu_nt40", "ralu_nt45", "ralu_nt48")
IMPLEMENTATION_FILES = (
    "changing_resolution/ralu_nt_math.py",
    "changing_resolution/ralu_wan_quality.py",
    "changing_resolution/ralu_wan_state.py",
    "wan_sr/schedulers/ralu_nt.py",
    "changing_resolution/scripts/bridge/run_lightx2v_clean_bridge_batch_infer.py",
    "paper/aaai27/experiments/run_final_quality_efficiency.py",
    "paper/aaai27/experiments/benchmark_quality_efficiency.py",
    "paper/aaai27/experiments/benchmark_warm_quality_efficiency.py",
    "paper/aaai27/experiments/collect_results.py",
    "paper/aaai27/experiments/collect_quality_efficiency.py",
)
SUMMARY_FILES = (
    "quality_efficiency.csv",
    "quality_efficiency_raw.csv",
    "quality_efficiency_warm.csv",
    "quality_efficiency_warm_raw.csv",
    "quality_efficiency_warm_pairs.csv",
)
WARM_ARTIFACTS = {
    "quality_efficiency_warm.csv",
    "quality_efficiency_warm_raw.csv",
    "quality_efficiency_warm_pairs.csv",
    "warm_timing_manifest.json",
    "protocol.json",
}


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else suite_root.parent / f"wan50_quality_efficiency_collection_{timestamp}"
    )
    output = collect_quality_efficiency(
        suite_root=suite_root,
        output_root=output_root,
        project_root=Path(args.project_root).resolve(),
        include_videos=args.include_videos,
        probe_videos=args.probe_videos,
        require_metrics=args.require_metrics,
        require_timing=args.require_timing,
        allow_incomplete=args.allow_incomplete,
    )
    print(f"Collection directory: {output}")
    if args.archive:
        archive = make_archive(output)
        print(f"Collection archive  : {archive}")


def collect_quality_efficiency(
    *,
    suite_root: Path,
    output_root: Path,
    project_root: Path,
    include_videos: bool,
    probe_videos: bool,
    require_metrics: bool,
    require_timing: bool,
    allow_incomplete: bool,
) -> Path:
    if not suite_root.is_dir():
        raise SystemExit(f"Suite root is missing: {suite_root}")
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing collection: {output_root}")
    if is_relative_to(output_root, suite_root):
        raise SystemExit("Collection output must be outside the suite root")

    manifest_path = suite_root / "run_manifest.json"
    manifest = load_json(manifest_path)
    manifest_issues = validate_manifest(manifest)
    factorial = inspect_factorial(suite_root, expected_family="wan50_quality_efficiency")
    video_rows, video_issues, obsolete_dirs = inventory_videos(
        suite_root,
        manifest,
        probe_videos=probe_videos,
    )
    summary_inventory, summary_issues = inspect_summaries(
        suite_root,
        expected_cases=set(EXPECTED_CASES),
        require_metrics=require_metrics,
        require_timing=require_timing,
    )
    issues = [*manifest_issues, *factorial.get("issues", []), *video_issues, *summary_issues]
    issues = list(dict.fromkeys(str(issue) for issue in issues))
    if issues and not allow_incomplete:
        details = "\n  ".join(issues)
        raise SystemExit(f"Quality-efficiency collection validation failed:\n  {details}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as raw:
        staging = Path(raw) / output_root.name
        staging.mkdir()
        copied: list[dict[str, Any]] = []

        copy_file(manifest_path, staging / "suite/run_manifest.json", staging, copied)
        for case in EXPECTED_CASES:
            source = suite_root / "configs" / f"{case}.json"
            if source.is_file():
                copy_file(source, staging / "suite/configs" / source.name, staging, copied)

        for name in ("benchmark_spec.json", "protocol.json", "warm_timing_manifest.json"):
            source = resolve_suite_artifact(suite_root, name)
            if source.is_file():
                copy_file(source, staging / "suite" / name, staging, copied)
        for name in SUMMARY_FILES:
            source = resolve_suite_artifact(suite_root, name)
            if source.is_file():
                copy_file(source, staging / "suite" / name, staging, copied)
        copy_tree_if_present(suite_root / "metrics", staging / "suite/metrics", staging, copied)
        for log_path in sorted(suite_root.glob("*.log")):
            copy_file(log_path, staging / "suite/logs" / log_path.name, staging, copied)

        for relative in IMPLEMENTATION_FILES:
            source = project_root / relative
            if source.is_file():
                copy_file(source, staging / "implementation" / relative, staging, copied)

        if include_videos:
            for row in video_rows:
                if row["status"] != "valid":
                    continue
                source = Path(row["path"])
                destination = staging / "suite/videos" / row["case"] / source.name
                copy_file(source, destination, staging, copied, known_sha256=row["sha256"])

        write_csv(staging / "video_inventory.csv", video_rows)
        validation = {
            "schema_version": 1,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "suite_root": str(suite_root),
            "status": "complete" if not issues else "incomplete",
            "issues": issues,
            "expected_cases": list(EXPECTED_CASES),
            "expected_videos": len(EXPECTED_CASES) * len(manifest.get("prompts", [])),
            "valid_videos": sum(row["status"] == "valid" for row in video_rows),
            "minimum_valid_video_bytes_exclusive": MIN_VALID_VIDEO_BYTES,
            "probe_videos": probe_videos,
            "obsolete_ralu_directories_ignored": obsolete_dirs,
            "factorial": factorial,
            "summaries": summary_inventory,
        }
        write_json(staging / "validation.json", validation)
        write_json(
            staging / "collection_manifest.json",
            {
                "schema_version": 1,
                "generated_at_utc": validation["generated_at_utc"],
                "source_suite": str(suite_root),
                "include_videos": include_videos,
                "probe_videos": probe_videos,
                "require_metrics": require_metrics,
                "require_timing": require_timing,
                "allow_incomplete": allow_incomplete,
                "copied_files": copied,
            },
        )
        write_checksums(staging)
        staging.replace(output_root)
    return output_root


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if manifest.get("family") != "wan50_quality_efficiency":
        issues.append(f"manifest family mismatch: {manifest.get('family')!r}")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return [*issues, "manifest cases must be a list"]
    names = [str(case.get("name", "")) for case in cases]
    if names != list(EXPECTED_CASES):
        issues.append(f"manifest case order/content mismatch: {names!r}")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 10:
        issues.append(f"manifest must contain exactly 10 prompts, found {len(prompts or [])}")
    if int(manifest.get("seed_base", -1)) != 9700:
        issues.append(f"manifest seed_base mismatch: {manifest.get('seed_base')!r}")
    ralu = next((case for case in cases if case.get("name") == "ralu_quality"), None)
    if not ralu:
        issues.append("ralu_quality case is missing")
    elif (
        int(ralu.get("lr_evaluations", -1)),
        int(ralu.get("mixed_evaluations", -1)),
        int(ralu.get("hr_evaluations", -1)),
        int(ralu.get("total_evaluations", -1)),
    ) != (5, 6, 7, 18):
        issues.append("ralu_quality budget must be LR/mixed/HR/total=(5,6,7,18)")
    return issues


def inventory_videos(
    suite_root: Path,
    manifest: dict[str, Any],
    *,
    probe_videos: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    prompts = list(manifest.get("prompts", []))
    seed_base = int(manifest.get("seed_base", 0))
    prompt_offset = int(manifest.get("prompt_offset", 0))
    manifest_cases = [str(case["name"]) for case in manifest.get("cases", [])]
    videos_root = suite_root / "videos"

    for case in manifest_cases:
        case_root = videos_root / case
        expected = {
            f"{case}_{index:02d}_seed{seed_base + index}.mp4"
            for index in range(prompt_offset, prompt_offset + len(prompts))
        }
        actual = {path.name: path for path in case_root.glob("*.mp4")} if case_root.is_dir() else {}
        for name in sorted(expected | set(actual)):
            path = actual.get(name)
            status = "missing"
            size = 0
            digest = ""
            probe_status = "not_requested"
            if path is not None:
                size = path.stat().st_size
                digest = sha256(path)
                if name not in expected:
                    status = "unexpected"
                elif size <= MIN_VALID_VIDEO_BYTES:
                    status = "undersized"
                else:
                    status = "valid"
                    if probe_videos:
                        probe_status = probe_video(path)
                        if probe_status != "valid":
                            status = "probe_failed"
            rows.append(
                {
                    "case": case,
                    "filename": name,
                    "expected": name in expected,
                    "status": status,
                    "size_bytes": size,
                    "sha256": digest,
                    "probe_status": probe_status,
                    "path": str(path or case_root / name),
                }
            )
            if status != "valid":
                issues.append(f"video {case}/{name}: {status}")

    directory_names = {path.name for path in videos_root.iterdir() if path.is_dir()} if videos_root.is_dir() else set()
    obsolete = sorted(directory_names & set(OBSOLETE_RALU_CASES))
    unexpected_dirs = sorted(directory_names - set(manifest_cases) - set(OBSOLETE_RALU_CASES))
    if unexpected_dirs:
        issues.append(f"unexpected video case directories: {unexpected_dirs!r}")
    return rows, issues, [str(videos_root / name) for name in obsolete]


def probe_video(path: Path) -> str:
    executable = shutil.which("ffprobe")
    if executable is None:
        return "ffprobe_unavailable"
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return "valid" if result.returncode == 0 and "video" in result.stdout else "invalid_media"


def inspect_summaries(
    suite_root: Path,
    *,
    expected_cases: set[str],
    require_metrics: bool,
    require_timing: bool,
) -> tuple[dict[str, Any], list[str]]:
    inventory: dict[str, Any] = {}
    issues: list[str] = []
    requirements = {
        # The paper-facing protocol is warm steady-state timing.  Retain a
        # historical cold-start table if present, but do not require it.
        "quality_efficiency.csv": False,
        "quality_efficiency_warm.csv": require_timing,
        "metrics/vbench_v1_custom.json": require_metrics,
    }
    for relative, required in requirements.items():
        path = resolve_suite_artifact(suite_root, relative)
        entry: dict[str, Any] = {
            "path": str(path),
            "canonical_relative_path": relative,
            "required": required,
            "status": "missing",
        }
        if path.is_file():
            entry.update({"status": "present", "size_bytes": path.stat().st_size, "sha256": sha256(path)})
            if path.suffix.lower() == ".csv":
                cases = csv_cases(path)
                entry["cases"] = sorted(cases)
                if cases != expected_cases:
                    entry["status"] = "case_mismatch"
                    if required:
                        issues.append(f"{relative} case coverage mismatch: {sorted(cases)!r}")
            elif path.suffix.lower() == ".json" and path.name == "vbench_v1_custom.json":
                payload = load_json(path)
                cases = set(payload.get("cases", {})) if isinstance(payload, dict) else set()
                entry["cases"] = sorted(cases)
                if cases != expected_cases:
                    entry["status"] = "case_mismatch"
                    if required:
                        issues.append(f"{relative} case coverage mismatch: {sorted(cases)!r}")
        elif required:
            issues.append(f"required result is missing: {relative}")
        inventory[relative] = entry
    return inventory, issues


def resolve_suite_artifact(suite_root: Path, relative: str) -> Path:
    """Resolve canonical outputs, including the warm benchmark's default subdirectory."""

    direct = suite_root / relative
    if direct.is_file() or relative not in WARM_ARTIFACTS:
        return direct
    nested = suite_root / "warm_quality_efficiency" / relative
    return nested if nested.is_file() else direct


def csv_cases(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "case" not in (reader.fieldnames or []):
            return set()
        return {str(row["case"]) for row in reader if row.get("case")}


def copy_file(
    source: Path,
    destination: Path,
    staging: Path,
    copied: list[dict[str, Any]],
    *,
    known_sha256: str | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(
        {
            "source": str(source),
            "collected": destination.relative_to(staging).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": known_sha256 or sha256(source),
        }
    )


def copy_tree_if_present(
    source: Path,
    destination: Path,
    staging: Path,
    copied: list[dict[str, Any]],
) -> None:
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file():
            copy_file(path, destination / path.relative_to(source), staging, copied)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["case", "filename", "expected", "status", "size_bytes", "sha256", "probe_status", "path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid or missing JSON {path}: {exc}") from exc


def write_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum_path
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_archive(root: Path) -> Path:
    archive = root.with_suffix(".tar.gz")
    if archive.exists():
        raise SystemExit(f"Refusing to overwrite existing archive: {archive}")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(root, arcname=root.name, recursive=True)
    return archive


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and collect the final 11-case Wan quality-efficiency suite."
    )
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--include-videos", action="store_true")
    parser.add_argument("--probe-videos", action="store_true")
    parser.add_argument("--require-metrics", action="store_true")
    parser.add_argument("--require-timing", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--archive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
