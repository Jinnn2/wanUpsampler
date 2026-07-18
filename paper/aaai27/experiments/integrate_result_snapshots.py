from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import mean, pstdev
from typing import Any, Iterable


QUALITY5 = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
)


def main() -> None:
    args = parse_args()
    old_core = Path(args.base_core).resolve()
    archive_path = Path(args.incremental_archive).resolve()
    output_root = Path(args.output_root).resolve()

    if not (old_core / "result_inventory.json").is_file():
        raise SystemExit(f"Old result inventory not found: {old_core}")
    if not archive_path.is_file():
        raise SystemExit(f"Incremental archive not found: {archive_path}")

    compiled_root = output_root / "compiled_tables"
    compiled_root.mkdir(parents=True, exist_ok=True)

    old_inventory = load_json_file(old_core / "result_inventory.json")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive_root = find_archive_root(archive)
        checksum_report = verify_archive(archive, archive_root)
        new_inventory = load_archive_json(
            archive, f"{archive_root}/core/result_inventory.json"
        )
        merge_report = merge_compiled_tables(
            old_core / "compiled_tables", archive, archive_root, compiled_root
        )
        derive_step45_endpoint_statistics(archive, archive_root, compiled_root)
        derive_final_quality5_paired_statistics(archive, archive_root, compiled_root)

    vbench_rows = extract_vbench_rows(old_inventory, "base_20260717")
    vbench_rows.extend(extract_vbench_rows(new_inventory, "incremental_20260718"))
    vbench_rows = deduplicate_vbench_rows(vbench_rows)
    write_csv(compiled_root / "vbench_case_summary.csv", vbench_rows)

    factorial_effects = derive_factorial_effects(vbench_rows)
    write_csv(compiled_root / "factorial_vbench_effects.csv", factorial_effects)
    efficiency_rows = derive_efficiency_summary(
        read_csv(compiled_root / "wan50_final_quality_efficiency.csv")
    )
    write_csv(compiled_root / "quality_efficiency_summary.csv", efficiency_rows)

    old_issues = {item["item"] for item in old_inventory.get("issues", [])}
    new_issues = {item["item"] for item in new_inventory.get("issues", [])}
    remaining_gaps = sorted(old_issues & new_issues)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "base_core": str(old_core),
            "base_inventory_generated_at_utc": old_inventory.get("generated_at_utc"),
            "incremental_archive": str(archive_path),
            "incremental_archive_sha256": sha256_file(archive_path),
            "incremental_inventory_generated_at_utc": new_inventory.get(
                "generated_at_utc"
            ),
        },
        "archive_verification": checksum_report,
        "compiled_table_merge": merge_report,
        "derived_tables": [
            "compiled_tables/wan50_step45_final_endpoint_paired_statistics.csv",
            "compiled_tables/vbench_case_summary.csv",
            "compiled_tables/factorial_vbench_effects.csv",
            "compiled_tables/quality_efficiency_summary.csv",
            "compiled_tables/wan50_final_quality5_paired_statistics.csv",
        ],
        "final_configuration": new_inventory.get("final_configuration", {}),
        "reported_training_resources": {
            "gpu_model": "NVIDIA H100",
            "gpu_count": 4,
            "lora_wall_clock_hours": 33,
            "stage2_wall_clock_hours": 8,
            "source": "author-reported on 2026-07-18",
        },
        "paper_terminology": {
            "lora_internal": "Trajectory Alignment Adapter (TAA)",
            "stage2_internal": "Clean Latent Lifter (CLL)",
            "stage3_internal": "Joint Trajectory-Scale Lifter (JTSL)",
            "renoise_hr_suffix": "High-Resolution Trajectory Re-entry (HTR)",
            "full_hr_internal": "Native-HR Sampling",
            "quality5_internal": "VBench-5",
        },
        "intentionally_omitted_experiments": remaining_gaps,
        "interpretation_constraints": [
            "Step45 final TAA statistics use strength 0.75 raw paired samples from the LoRA implementation, not the legacy strength 1.0 paired table.",
            "VBench-5 (internal quality5) is the unweighted mean of subject consistency, background consistency, motion smoothness, aesthetic quality, and imaging quality.",
            "Human-review inference uses ten prompt-level majority outcomes; thirty individual ratings are clustered descriptive observations.",
            "TAA endpoint improvement and end-to-end perceptual improvement are distinct claims.",
            "Peak GPU memory is approximately unchanged across the final Wan50 efficiency cases.",
        ],
    }
    (output_root / "integration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(output_root, manifest, vbench_rows, factorial_effects, efficiency_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the AAAI-27 base core and closure archive into one paper result set."
    )
    parser.add_argument("--base-core", required=True)
    parser.add_argument("--incremental-archive", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def find_archive_root(archive: tarfile.TarFile) -> str:
    roots = {
        PurePosixPath(member.name).parts[0]
        for member in archive.getmembers()
        if member.name and PurePosixPath(member.name).parts
    }
    if len(roots) != 1:
        raise ValueError(f"Expected one archive root, found: {sorted(roots)}")
    return next(iter(roots))


def verify_archive(archive: tarfile.TarFile, archive_root: str) -> dict[str, Any]:
    checksum_member = archive.getmember(f"{archive_root}/SHA256SUMS")
    checksum_text = archive.extractfile(checksum_member).read().decode("utf-8-sig")
    expected: dict[str, str] = {}
    for line in checksum_text.splitlines():
        if not line.strip():
            continue
        digest, relative_path = line.split(maxsplit=1)
        expected[relative_path.strip()] = digest

    missing: list[str] = []
    mismatched: list[str] = []
    illegal_on_windows = 0
    for relative_path, expected_digest in expected.items():
        member_name = f"{archive_root}/{relative_path}"
        try:
            member = archive.getmember(member_name)
        except KeyError:
            missing.append(relative_path)
            continue
        if ":" in PurePosixPath(relative_path).name:
            illegal_on_windows += 1
        actual_digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        if actual_digest != expected_digest:
            mismatched.append(relative_path)
    if missing or mismatched:
        raise ValueError(
            f"Archive verification failed: missing={len(missing)}, mismatched={len(mismatched)}"
        )
    return {
        "checksummed_files": len(expected),
        "missing": 0,
        "mismatched": 0,
        "windows_illegal_colon_members": illegal_on_windows,
        "status": "verified",
    }


def merge_compiled_tables(
    old_root: Path,
    archive: tarfile.TarFile,
    archive_root: str,
    output_root: Path,
) -> dict[str, Any]:
    old_names = {path.name for path in old_root.glob("*.csv")}
    prefix = f"{archive_root}/core/compiled_tables/"
    new_members = {
        PurePosixPath(member.name).name: member
        for member in archive.getmembers()
        if member.isfile() and member.name.startswith(prefix) and member.name.endswith(".csv")
    }
    report: dict[str, Any] = {"tables": {}, "counts": defaultdict(int)}
    for name in sorted(old_names | set(new_members)):
        old_bytes = (old_root / name).read_bytes() if (old_root / name).is_file() else None
        new_bytes = (
            archive.extractfile(new_members[name]).read() if name in new_members else None
        )
        if name == "factorial_coverage.csv":
            data = merge_factorial_coverage(old_bytes, new_bytes)
            source = "union"
        elif csv_has_data(new_bytes):
            data = new_bytes
            source = "incremental"
        elif csv_has_data(old_bytes):
            data = old_bytes
            source = "base_fallback"
        elif new_bytes is not None:
            data = new_bytes
            source = "incremental_empty"
        elif old_bytes is not None:
            data = old_bytes
            source = "base_empty"
        else:
            continue
        (output_root / name).write_bytes(data)
        report["tables"][name] = source
        report["counts"][source] += 1
    report["counts"] = dict(report["counts"])
    return report


def merge_factorial_coverage(old_bytes: bytes | None, new_bytes: bytes | None) -> bytes:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    fieldnames: list[str] = []
    for data in (old_bytes, new_bytes):
        if not data:
            continue
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if reader.fieldnames:
            for field in reader.fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)
        for row in reader:
            rows[(row.get("family", ""), row.get("case", ""))] = row
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for key in sorted(rows):
        writer.writerow(rows[key])
    return stream.getvalue().encode("utf-8")


def derive_step45_endpoint_statistics(
    archive: tarfile.TarFile, archive_root: str, output_root: Path
) -> None:
    suffix = (
        "evidence/legacy/changing_resolution_tail_skip_lora_strength_sweep_360p_368x640/"
        "metrics/lora_45_s0p75/original_lora_teacher_metrics.csv"
    )
    member = archive.getmember(f"{archive_root}/{suffix}")
    rows = list(
        csv.DictReader(
            io.StringIO(archive.extractfile(member).read().decode("utf-8-sig"))
        )
    )
    output: list[dict[str, Any]] = []
    for metric, better in (
        ("l1", "lower"),
        ("mse", "lower"),
        ("psnr", "higher"),
        ("temporal_l1", "lower"),
    ):
        pairs = [
            (float(row[f"original_{metric}"]), float(row[f"lora_{metric}"]))
            for row in rows
        ]
        output.append(
            paired_summary(
                pairs,
                metric=metric,
                better=better,
                extra={"step": 45, "strength": 0.75, "case": "lora_45_s0p75"},
            )
        )
    write_csv(output_root / "wan50_step45_final_endpoint_paired_statistics.csv", output)


def derive_final_quality5_paired_statistics(
    archive: tarfile.TarFile, archive_root: str, output_root: Path
) -> None:
    member_name = (
        f"{archive_root}/evidence/canonical/quality_efficiency_final/metrics/"
        "vbench_v1_custom.json"
    )
    payload = json.loads(archive.extractfile(archive.getmember(member_name)).read())
    comparisons = (
        ("full_hr50_vs_talh40", "full_hr50", "talh40"),
        ("full_hr50_vs_talh45", "full_hr50", "talh45"),
        (
            "full_hr50_vs_full_lr50_stage2_1hr",
            "full_hr50",
            "full_lr50_stage2_1hr",
        ),
        ("talh45_vs_talh40", "talh45", "talh40"),
    )
    rows: list[dict[str, Any]] = []
    for comparison, case_a, case_b in comparisons:
        values_a = quality5_per_video(payload, case_a)
        values_b = quality5_per_video(payload, case_b)
        common = sorted(set(values_a) & set(values_b))
        if not common:
            raise ValueError(f"No paired Quality5 values for {case_a} and {case_b}")
        a = [values_a[index] for index in common]
        b = [values_b[index] for index in common]
        deltas = [right - left for left, right in zip(a, b)]
        rng = random.Random(f"202707:{comparison}:quality5")
        boot = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000)]
        boot.sort()
        wins = sum(value > 0 for value in deltas)
        losses = sum(value < 0 for value in deltas)
        rows.append(
            {
                "comparison": comparison,
                "case_a": case_a,
                "case_b": case_b,
                "metric": "quality5_mean",
                "samples": len(common),
                "a_mean": mean(a),
                "b_mean": mean(b),
                "delta_b_minus_a_mean": mean(deltas),
                "relative_delta_vs_a": mean(deltas) / mean(a),
                "delta_std": pstdev(deltas),
                "bootstrap_ci_low": quantile(boot, 0.025),
                "bootstrap_ci_high": quantile(boot, 0.975),
                "wins": wins,
                "losses": losses,
                "ties": len(deltas) - wins - losses,
                "two_sided_sign_test_p": sign_test_p(wins, losses),
            }
        )
    write_csv(output_root / "wan50_final_quality5_paired_statistics.csv", rows)


def quality5_per_video(payload: dict[str, Any], case: str) -> dict[int, float]:
    numeric = payload.get("cases", {}).get(case, {}).get("numeric_metrics", {})
    per_dimension: dict[str, dict[int, float]] = {}
    for dimension in QUALITY5:
        pattern = re.compile(
            rf"\.{re.escape(dimension)}\.1\.(?P<index>\d+)\.video_results$"
        )
        values: dict[int, float] = {}
        for key, raw_value in numeric.items():
            match = pattern.search(key)
            if not match:
                continue
            value = float(raw_value)
            if dimension == "imaging_quality" and value > 1.0:
                value /= 100.0
            values[int(match.group("index"))] = value
        per_dimension[dimension] = values
    common = set.intersection(*(set(values) for values in per_dimension.values()))
    return {
        index: mean(per_dimension[dimension][index] for dimension in QUALITY5)
        for index in sorted(common)
    }


def paired_summary(
    pairs: list[tuple[float, float]],
    metric: str,
    better: str,
    extra: dict[str, Any] | None = None,
    bootstrap_samples: int = 10000,
    seed: int = 202707,
) -> dict[str, Any]:
    deltas = [b - a for a, b in pairs]
    oriented = [-value if better == "lower" else value for value in deltas]
    rng = random.Random(seed)
    boot = [mean(rng.choices(oriented, k=len(oriented))) for _ in range(bootstrap_samples)]
    boot.sort()
    wins = sum(value > 0 for value in oriented)
    losses = sum(value < 0 for value in oriented)
    result: dict[str, Any] = dict(extra or {})
    result.update(
        {
            "metric": metric,
            "better": better,
            "samples": len(pairs),
            "original_mean": mean(a for a, _ in pairs),
            "lora_mean": mean(b for _, b in pairs),
            "oriented_improvement_mean": mean(oriented),
            "relative_improvement": mean(oriented) / mean(a for a, _ in pairs),
            "oriented_improvement_std": pstdev(oriented),
            "bootstrap_ci_low": quantile(boot, 0.025),
            "bootstrap_ci_high": quantile(boot, 0.975),
            "wins": wins,
            "losses": losses,
            "ties": len(oriented) - wins - losses,
            "two_sided_sign_test_p": sign_test_p(wins, losses),
        }
    )
    return result


def extract_vbench_rows(inventory: dict[str, Any], source_snapshot: str) -> list[dict[str, Any]]:
    accum: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    pattern = re.compile(
        r"^cases\.([^.]+)\.numeric_metrics\..+_eval_results\.([^.]+)\.0$"
    )
    for family, group in inventory.get("external", {}).get("vbench", {}).items():
        for file_entry in group.get("files", []):
            for key, value in file_entry.get("numeric_metrics", {}).items():
                match = pattern.match(key)
                if match and isinstance(value, (int, float)):
                    case, metric = match.groups()
                    accum[(family, case)][metric] = float(value)
    rows: list[dict[str, Any]] = []
    for (family, case), metrics in sorted(accum.items()):
        if not all(metric in metrics for metric in QUALITY5):
            continue
        row: dict[str, Any] = {
            "family": family,
            "case": case,
            "source_snapshot": source_snapshot,
            "samples": 10,
        }
        row.update({metric: metrics[metric] for metric in QUALITY5})
        row["dynamic_degree"] = metrics.get("dynamic_degree", "")
        row["quality5_mean"] = mean(metrics[metric] for metric in QUALITY5)
        rows.append(row)
    return rows


def deduplicate_vbench_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["case"]))
        if key not in preferred or row["source_snapshot"] == "incremental_20260718":
            preferred[key] = row
    return [preferred[key] for key in sorted(preferred)]


def derive_factorial_effects(vbench_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case = {str(row["case"]): row for row in vbench_rows}
    comparisons = (
        ("wan50_step40", "Stage2 effect", "step40_base_interp", "step40_base_stage2"),
        (
            "wan50_step40",
            "LoRA effect within Stage2 (strength 0.75)",
            "step40_base_stage2",
            "step40_lora_s0p75_stage2",
        ),
        (
            "wan50_step40",
            "TALH effect vs interpolation",
            "step40_base_interp",
            "step40_lora_s0p75_stage2",
        ),
        ("wan50_step45", "Stage2 effect", "step45_base_interp", "step45_base_stage2"),
        (
            "wan50_step45",
            "LoRA effect within Stage2 (strength 0.75)",
            "step45_base_stage2",
            "step45_lora_stage2",
        ),
        (
            "wan50_step45",
            "TALH effect vs interpolation",
            "step45_base_interp",
            "step45_lora_stage2",
        ),
        ("distill4", "Stage2 effect", "step3_base_interp", "step3_base_stage2"),
        (
            "distill4",
            "LoRA effect within Stage2",
            "step3_base_stage2",
            "step3_lora_stage2",
        ),
        (
            "distill4",
            "TALH effect vs interpolation",
            "step3_base_interp",
            "step3_lora_stage2",
        ),
    )
    output: list[dict[str, Any]] = []
    for family, effect, case_a, case_b in comparisons:
        if case_a not in by_case or case_b not in by_case:
            continue
        a, b = by_case[case_a], by_case[case_b]
        row: dict[str, Any] = {
            "family": family,
            "effect": effect,
            "case_a": case_a,
            "case_b": case_b,
            "quality5_a": a["quality5_mean"],
            "quality5_b": b["quality5_mean"],
            "delta_quality5_b_minus_a": b["quality5_mean"] - a["quality5_mean"],
        }
        for metric in QUALITY5:
            row[f"delta_{metric}"] = float(b[metric]) - float(a[metric])
        output.append(row)
    return output


def derive_efficiency_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    full_hr = next(row for row in rows if row["case"] == "full_hr50")
    base_time = float(full_hr["elapsed_mean_s"])
    base_quality = float(full_hr["quality_value"])
    output: list[dict[str, Any]] = []
    for row in rows:
        elapsed = float(row["elapsed_mean_s"])
        quality = float(row["quality_value"])
        output.append(
            {
                "case": row["case"],
                "lr_evaluations": row["lr_evaluations"],
                "hr_evaluations": row["hr_evaluations"],
                "elapsed_mean_s": elapsed,
                "elapsed_std_s": float(row["elapsed_std_s"]),
                "speedup_vs_full_hr": base_time / elapsed,
                "latency_reduction_vs_full_hr": 1.0 - elapsed / base_time,
                "peak_memory_gib": float(row["peak_memory_gib"]),
                "quality5_mean": quality,
                "quality5_drop_vs_full_hr": base_quality - quality,
                "quality5_relative_drop_vs_full_hr": (base_quality - quality)
                / base_quality,
            }
        )
    return output


def write_readme(
    output_root: Path,
    manifest: dict[str, Any],
    vbench_rows: list[dict[str, Any]],
    factorial_effects: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
) -> None:
    gaps = manifest["intentionally_omitted_experiments"]
    lines = [
        "# AAAI-27 统一结果集",
        "",
        "该目录由 `integrate_result_snapshots.py` 生成，合并 2026-07-17 base core 与 2026-07-18 closure incremental archive。",
        "",
        "- 增量归档校验：全部 "
        f"{manifest['archive_verification']['checksummed_files']} 个文件通过 SHA-256；"
        f"其中 {manifest['archive_verification']['windows_illegal_colon_members']} 个原始文件名含冒号，只能从 tar 流读取。",
        f"- 汇总 VBench case：{len(vbench_rows)} 个。",
        f"- 论文因子效应：{len(factorial_effects)} 组。",
        f"- 最终质量—效率 case：{len(efficiency_rows)} 个。",
        "- 最终 TAA 配置（LoRA 实现）：step40 strength=0.75，step45 strength=0.75。",
        "- 训练资源：4×NVIDIA H100；TAA 约 33 小时，CLL 约 8 小时（wall-clock）。",
        "",
        "## 有意不纳入的实验",
        "",
    ]
    lines.extend(f"- `{gap}`" for gap in gaps)
    lines.extend(
        [
            "",
            "## 使用约束",
            "",
            "- `wan50_step45_final_endpoint_paired_statistics.csv` 从归档内 TAA strength=0.75 原始逐样本表重新计算；不要引用旧的 strength=1.0 paired table 作为最终配置。",
            "- TAA endpoint 指标与最终视频质量必须分别表述。",
            "- 人工盲评以 10 个 prompt-majority 为统计单位；30 个 individual votes 仅作描述。",
            "- 最终效率表不支持显存下降主张。",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def csv_has_data(data: bytes | None) -> bool:
    if not data:
        return False
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    rows = list(reader)
    if not rows:
        return False
    if reader.fieldnames == ["status"] and all(
        row.get("status", "").strip().upper() == "MISSING" for row in rows
    ):
        return False
    return True


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    if not row_list:
        path.write_text("status,detail\nmissing,no rows\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in row_list:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_list)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_archive_json(archive: tarfile.TarFile, member_name: str) -> dict[str, Any]:
    return json.loads(archive.extractfile(archive.getmember(member_name)).read())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


if __name__ == "__main__":
    main()
