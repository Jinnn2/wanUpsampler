from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MIN_VIDEO_BYTES = 1024
MAIN_CASES = (
    "native_hr4",
    "interp2",
    "interp3",
    "taa_interp3",
    "cll3",
    "talh3",
    "endpoint_stage2_0hr",
    "endpoint_stage2_1hr",
    "endpoint_stage2_2hr",
    "endpoint_stage2_4hr",
    "endpoint_interp_0hr",
    "endpoint_interp_1hr",
    "endpoint_interp_2hr",
    "endpoint_interp_4hr",
    "endpoint_rgb_0hr",
    "endpoint_rgb_1hr",
    "endpoint_rgb_2hr",
    "endpoint_rgb_4hr",
)
QUALITY5 = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
)
MAIN_REQUIRED = (
    "run_manifest.json",
    "generation_schedule.json",
    "configs/endpoint_rgb_1hr.json",
    "metrics/vbench_v1_custom.json",
    "metrics/vbench_paired_statistics.csv",
    "metrics/vbench_temporal_flickering.json",
    "metrics/vbench_temporal_flickering_paired_statistics.csv",
)
P3_REQUIRED = (
    "run_manifest.json",
    "generation_schedule.json",
    "metrics/vbench_v1_custom.json",
    "metrics/talh_validation_selection.json",
    "metrics/talh_validation_selection.csv",
)
IMPLEMENTATION_FILES = (
    "changing_resolution_distill/lightx2v_distill_bridge.py",
    "changing_resolution_distill/scripts/eval/run_distill4_final_18case_4gpu.sh",
    "changing_resolution_distill/scripts/eval/run_distill4_p1_temporal_flickering_4gpu.sh",
    "changing_resolution_distill/scripts/eval/run_distill4_p3_talh_validation_4gpu.sh",
    "changing_resolution_distill/scripts/eval/export_distill4_p0_p1_p3_final.sh",
    "changing_resolution_distill/scripts/eval/refresh_distill4_p0_metrics_and_export.sh",
    "paper/aaai27/experiments/run_distill4_quality_efficiency.py",
    "paper/aaai27/experiments/run_vbench_factorials.py",
    "paper/aaai27/experiments/compile_vbench_paired_statistics.py",
    "paper/aaai27/experiments/run_distill4_talh_validation_sweep.py",
    "paper/aaai27/experiments/distill4_talh_validation_prompts_8.txt",
    "paper/aaai27/experiments/export_distill4_p0_p1_p3.py",
    "paper/aaai27/experiments/refresh_distill4_p0_results.py",
)
OPTIONAL_MAIN_FILES = (
    "artifact_fingerprints.json",
    "benchmark_spec.json",
    "protocol.json",
    "quality_efficiency.csv",
    "quality_efficiency_raw.csv",
    "quality_efficiency_warm.csv",
    "quality_efficiency_warm_raw.csv",
    "quality_efficiency_warm_pairs.csv",
    "warm_timing_manifest.json",
    "warm_quality_efficiency/benchmark_spec.json",
    "warm_quality_efficiency/protocol.json",
    "warm_quality_efficiency/quality_efficiency_warm.csv",
    "warm_quality_efficiency/quality_efficiency_warm_raw.csv",
    "warm_quality_efficiency/quality_efficiency_warm_pairs.csv",
    "warm_quality_efficiency/warm_timing_manifest.json",
    "metrics/p0_vbench_refresh_manifest.json",
    "warm_quality_efficiency/p0_warm_refresh_manifest.json",
)


def main() -> None:
    args = parse_args()
    output = export_bundle(
        main_root=Path(args.main_root).resolve(),
        validation_root=Path(args.validation_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        project_root=Path(args.project_root).resolve(),
        include_videos=args.include_videos,
    )
    print(f"Export directory: {output}")
    print(f"Export archive  : {output.with_suffix('.tar.gz')}")


def export_bundle(
    *,
    main_root: Path,
    validation_root: Path,
    output_root: Path,
    project_root: Path,
    include_videos: bool,
) -> Path:
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing export: {output_root}")
    archive = output_root.with_suffix(".tar.gz")
    if archive.exists():
        raise SystemExit(f"Refusing to overwrite existing archive: {archive}")
    if not main_root.is_dir() or not validation_root.is_dir():
        raise SystemExit(
            f"Missing input root: main={main_root.is_dir()}, "
            f"validation={validation_root.is_dir()}"
        )

    main_manifest = load_json(main_root / "run_manifest.json")
    p3_manifest = load_json(validation_root / "run_manifest.json")
    validate_manifest(
        main_manifest,
        family="distill4_quality_efficiency",
        expected_cases=MAIN_CASES,
        prompt_count=10,
    )
    p3_cases = tuple(str(case["name"]) for case in p3_manifest.get("cases", []))
    validate_manifest(
        p3_manifest,
        family="distill4_talh_validation_sweep",
        expected_cases=p3_cases,
        prompt_count=8,
    )
    if len(p3_cases) != 8:
        raise SystemExit(f"P3 must contain eight sweep cases, found {len(p3_cases)}")

    require_files(main_root, MAIN_REQUIRED)
    require_files(validation_root, P3_REQUIRED)
    validate_p0(main_root)
    main_videos = validate_videos(main_root, main_manifest, MAIN_CASES)
    p3_videos = validate_videos(validation_root, p3_manifest, p3_cases)
    validate_metrics(
        main_root / "metrics/vbench_v1_custom.json",
        MAIN_CASES,
        QUALITY5,
    )
    validate_metrics(
        main_root / "metrics/vbench_temporal_flickering.json",
        MAIN_CASES,
        ("temporal_flickering",),
    )
    validate_metrics(
        validation_root / "metrics/vbench_v1_custom.json",
        p3_cases,
        (*QUALITY5, "temporal_flickering"),
    )
    validate_p3_selection(validation_root, p3_cases)
    validate_freshness(main_root, main_videos)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.", dir=output_root.parent
    ) as raw_staging:
        staging = Path(raw_staging) / output_root.name
        staging.mkdir()
        copied: list[dict[str, Any]] = []
        for relative in MAIN_REQUIRED:
            if relative.startswith("configs/"):
                continue
            copy_file(
                main_root / relative,
                staging / "main_suite" / relative,
                staging,
                copied,
            )
        for relative in OPTIONAL_MAIN_FILES:
            source = main_root / relative
            if source.is_file():
                copy_file(
                    source,
                    staging / "main_suite" / relative,
                    staging,
                    copied,
                )
        copy_tree(
            main_root / "configs",
            staging / "main_suite/configs",
            staging,
            copied,
        )
        for relative in P3_REQUIRED:
            copy_file(
                validation_root / relative,
                staging / "p3_validation" / relative,
                staging,
                copied,
            )
        copy_tree(
            validation_root / "configs",
            staging / "p3_validation/configs",
            staging,
            copied,
        )
        for relative in IMPLEMENTATION_FILES:
            source = project_root / relative
            if not source.is_file():
                raise SystemExit(f"Missing implementation file: {source}")
            copy_file(
                source,
                staging / "implementation" / relative,
                staging,
                copied,
            )
        if include_videos:
            copy_video_inventory(
                main_videos,
                staging / "main_suite/videos",
                staging,
                copied,
            )
            copy_video_inventory(
                p3_videos,
                staging / "p3_validation/videos",
                staging,
                copied,
            )

        export_manifest = {
            "schema_version": 1,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": ["P0_mrflow_sigma_0.12", "P1_temporal_flickering", "P3_talh_validation_sweep"],
            "source_roots": {
                "main_suite": str(main_root),
                "p3_validation": str(validation_root),
            },
            "include_videos": include_videos,
            "validated_video_counts": {
                "main_suite": len(main_videos),
                "p3_validation": len(p3_videos),
            },
            "p0": {
                "case": "endpoint_rgb_1hr",
                "direct_sigma": 0.12,
                "metrics_freshness_checked": True,
            },
            "p3_selected": load_json(
                validation_root / "metrics/talh_validation_selection.json"
            )["selected"],
            "copied_files": copied,
        }
        write_json(staging / "export_manifest.json", export_manifest)
        write_checksums(staging)
        staging.replace(output_root)

    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_root, arcname=output_root.name, recursive=True)
    return output_root


def validate_manifest(
    payload: dict[str, Any],
    *,
    family: str,
    expected_cases: tuple[str, ...],
    prompt_count: int,
) -> None:
    if payload.get("family") != family:
        raise SystemExit(
            f"Manifest family mismatch: {payload.get('family')!r}, expected {family!r}"
        )
    cases = tuple(str(case.get("name", "")) for case in payload.get("cases", []))
    if cases != expected_cases:
        raise SystemExit(f"{family} case mismatch: {cases!r}")
    prompts = payload.get("prompts", [])
    if not isinstance(prompts, list) or len(prompts) != prompt_count:
        raise SystemExit(
            f"{family} must contain {prompt_count} prompts, found {len(prompts)}"
        )


def validate_p0(root: Path) -> None:
    config = load_json(root / "configs/endpoint_rgb_1hr.json")
    if int(config.get("wan_final_refine_steps", -1)) != 1:
        raise SystemExit("Endpoint-RGB-1HR must use one HR refinement")
    if abs(float(config.get("wan_final_refine_sigma", -1.0)) - 0.12) > 1e-12:
        raise SystemExit("Endpoint-RGB-1HR must use direct sigma=0.12")
    if config.get("wan_rgb_sr_backend") != "realesrgan":
        raise SystemExit("Endpoint-RGB-1HR must use Real-ESRGAN")


def validate_videos(
    root: Path, manifest: dict[str, Any], cases: tuple[str, ...]
) -> list[Path]:
    prompt_count = len(manifest["prompts"])
    offset = int(manifest.get("prompt_offset", 0))
    seed_base = int(manifest["seed_base"])
    videos: list[Path] = []
    for case in cases:
        for position in range(prompt_count):
            index = offset + position
            path = (
                root
                / "videos"
                / case
                / f"{case}_{index:02d}_seed{seed_base + index}.mp4"
            )
            if not path.is_file() or path.stat().st_size <= MIN_VIDEO_BYTES:
                raise SystemExit(f"Missing or undersized video: {path}")
            videos.append(path)
    return videos


def validate_metrics(
    path: Path, cases: tuple[str, ...], dimensions: tuple[str, ...]
) -> None:
    payload = load_json(path)
    metric_cases = payload.get("cases", {})
    if set(metric_cases) != set(cases):
        raise SystemExit(f"Metric case mismatch in {path}")
    declared = set(payload.get("dimensions", []))
    missing_dimensions = set(dimensions) - declared
    if missing_dimensions:
        raise SystemExit(
            f"Missing dimensions in {path}: {sorted(missing_dimensions)}"
        )
    for case in cases:
        numeric = metric_cases[case].get("numeric_metrics", {})
        for dimension in dimensions:
            if not any(key.endswith(f".{dimension}.0") for key in numeric):
                raise SystemExit(
                    f"Missing aggregate {dimension} for {case} in {path}"
                )


def validate_p3_selection(root: Path, cases: tuple[str, ...]) -> None:
    path = root / "metrics/talh_validation_selection.json"
    payload = load_json(path)
    selected = payload.get("selected", {})
    if selected.get("case") not in cases or not selected.get("selected"):
        raise SystemExit(f"Invalid P3 selected case in {path}")
    ranking = payload.get("ranking", [])
    if len(ranking) != len(cases) or {row.get("case") for row in ranking} != set(
        cases
    ):
        raise SystemExit(f"Invalid P3 ranking coverage in {path}")
    vbench = root / "metrics/vbench_v1_custom.json"
    if path.stat().st_mtime_ns < vbench.stat().st_mtime_ns:
        raise SystemExit("P3 selection is older than its VBench input; rerun select")


def validate_freshness(main_root: Path, videos: list[Path]) -> None:
    rgb1 = [
        path
        for path in videos
        if path.parent.name == "endpoint_rgb_1hr"
    ]
    newest_rgb1 = max(path.stat().st_mtime_ns for path in rgb1)
    required_outputs = [
        main_root / "metrics/vbench_v1_custom.json",
        main_root / "metrics/vbench_temporal_flickering.json",
    ]
    warm_candidates = [
        main_root / "quality_efficiency_warm.csv",
        main_root / "warm_quality_efficiency/quality_efficiency_warm.csv",
    ]
    warm = next((path for path in warm_candidates if path.is_file()), None)
    if warm is None:
        raise SystemExit("Missing quality_efficiency_warm.csv")
    required_outputs.append(warm)
    stale = [str(path) for path in required_outputs if path.stat().st_mtime_ns < newest_rgb1]
    if stale:
        raise SystemExit(
            "These outputs are older than the P0 RGB-1HR videos and must be rerun:\n  "
            + "\n  ".join(stale)
        )


def require_files(root: Path, relative_paths: tuple[str, ...]) -> None:
    missing = [
        str(root / relative)
        for relative in relative_paths
        if not (root / relative).is_file()
    ]
    if missing:
        raise SystemExit("Missing required export inputs:\n  " + "\n  ".join(missing))


def copy_video_inventory(
    videos: list[Path],
    destination: Path,
    staging: Path,
    copied: list[dict[str, Any]],
) -> None:
    for source in videos:
        copy_file(
            source,
            destination / source.parent.name / source.name,
            staging,
            copied,
        )


def copy_tree(
    source: Path,
    destination: Path,
    staging: Path,
    copied: list[dict[str, Any]],
) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            copy_file(
                path,
                destination / path.relative_to(source),
                staging,
                copied,
            )


def copy_file(
    source: Path,
    destination: Path,
    staging: Path,
    copied: list[dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(
        {
            "source": str(source),
            "exported": destination.relative_to(staging).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum_path
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid or missing JSON {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Strictly validate and export completed Distill4 P0/P1/P3 results."
    )
    parser.add_argument(
        "--main-root",
        default=str(
            REPO_ROOT
            / "outputs/aaai27_experiments/quality_efficiency_distill4"
        ),
    )
    parser.add_argument(
        "--validation-root",
        default=str(
            REPO_ROOT
            / "outputs/aaai27_experiments/distill4_talh_validation_sweep"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(
            REPO_ROOT
            / f"exports/distill4_p0_p1_p3_final_{timestamp}"
        ),
    )
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Also include all 244 MP4 files; omitted by default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
