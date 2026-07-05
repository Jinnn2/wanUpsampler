from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.data.video_io import read_video_frames


DEFAULT_OUT_ROOT = "/mnt/afs_2/houze/wanUpsampler/outputs/changing_resolution_distill_last_step_skip_lora_clean_pred_compare_480p"
DEFAULT_CASES = {
    "original": "original3_clean_pred",
    "lora": "lora3_step3_clean_pred",
    "teacher": "teacher4",
}
LOWER_IS_BETTER = {"l1", "mse", "temporal_l1", "lpips"}
HIGHER_IS_BETTER = {"psnr", "ssim"}
SUPPORTED_METRICS = tuple(sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER))


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    pair_rows = find_video_triples(out_root, args)
    if not pair_rows:
        raise SystemExit(f"No matched original/LoRA/teacher triples found under: {out_root}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    similarity = VideoSimilarityMetrics(args.metrics, device=device)

    result_dir = Path(args.result_dir) if args.result_dir else out_root / "metrics"
    result_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = result_dir / args.jsonl_name
    csv_path = result_dir / args.csv_name
    summary_json_path = result_dir / args.summary_json_name
    summary_csv_path = result_dir / args.summary_csv_name

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for row_index, paths in enumerate(pair_rows, start=1):
            metrics = evaluate_triple(
                paths=paths,
                metric_names=args.metrics,
                similarity=similarity,
                device=device,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
                resize_mode=args.resize_mode,
                metric_batch_size=args.metric_batch_size,
            )
            rows.append(metrics)
            jsonl_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            print(format_progress(row_index, len(pair_rows), metrics), flush=True)

    write_csv(csv_path, rows, sample_columns(args.metrics))
    summary = build_summary(rows, args.metrics, args.primary_metric)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv_path, summary["metrics"], summary_columns())

    print("")
    print(f"Matched triples : {len(rows)}")
    print(f"Primary metric  : {args.primary_metric} ({metric_direction(args.primary_metric)} is better)")
    print(f"Overall winner  : {summary['overall_winner']}")
    print(f"JSONL           : {jsonl_path}")
    print(f"CSV             : {csv_path}")
    print(f"Summary JSON    : {summary_json_path}")
    print(f"Summary CSV     : {summary_csv_path}")


def find_video_triples(out_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    original_dir = out_root / "videos" / args.original_case
    lora_dir = out_root / "videos" / args.lora_case
    teacher_dir = out_root / "videos" / args.teacher_case
    for case_dir in (original_dir, lora_dir, teacher_dir):
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case video directory not found: {case_dir}")

    original = index_case_videos(original_dir, args.original_case)
    lora = index_case_videos(lora_dir, args.lora_case)
    teacher = index_case_videos(teacher_dir, args.teacher_case)
    keys = sorted(set(original) & set(lora) & set(teacher))
    if args.limit > 0:
        keys = keys[: args.limit]

    missing = {
        "original_only_missing": sorted((set(lora) | set(teacher)) - set(original)),
        "lora_only_missing": sorted((set(original) | set(teacher)) - set(lora)),
        "teacher_only_missing": sorted((set(original) | set(lora)) - set(teacher)),
    }
    missing_count = sum(len(value) for value in missing.values())
    if missing_count and not args.quiet_missing:
        print(f"Warning: {missing_count} unmatched video keys skipped.", file=sys.stderr)

    return [
        {
            "key": key,
            "sample_index": key[0],
            "seed": key[1],
            "original_path": original[key],
            "lora_path": lora[key],
            "teacher_path": teacher[key],
        }
        for key in keys
    ]


def index_case_videos(case_dir: Path, case_name: str) -> dict[tuple[int, int | None], Path]:
    pattern = re.compile(rf"^{re.escape(case_name)}_(?P<index>\d+)(?:_seed(?P<seed>-?\d+))?\.mp4$")
    indexed: dict[tuple[int, int | None], Path] = {}
    for path in sorted(case_dir.glob("*.mp4")):
        match = pattern.match(path.name)
        if not match:
            continue
        key = (int(match.group("index")), int(match.group("seed")) if match.group("seed") is not None else None)
        indexed[key] = path
    return indexed


def evaluate_triple(
    *,
    paths: dict[str, Any],
    metric_names: list[str],
    similarity: "VideoSimilarityMetrics",
    device: torch.device,
    max_frames: int | None,
    frame_stride: int,
    resize_mode: str,
    metric_batch_size: int,
) -> dict[str, Any]:
    teacher = prepare_video(read_video_frames(paths["teacher_path"], max_frames=max_frames), frame_stride)
    original = prepare_video(read_video_frames(paths["original_path"], max_frames=max_frames), frame_stride)
    lora = prepare_video(read_video_frames(paths["lora_path"], max_frames=max_frames), frame_stride)
    original, lora, teacher = align_videos(original, lora, teacher, resize_mode=resize_mode)

    original_metrics = similarity.compute(original, teacher, batch_size=metric_batch_size)
    lora_metrics = similarity.compute(lora, teacher, batch_size=metric_batch_size)

    wins = {}
    deltas = {}
    for metric_name in metric_names:
        original_value = original_metrics[metric_name]
        lora_value = lora_metrics[metric_name]
        wins[metric_name] = winner(original_value, lora_value, metric_name)
        deltas[f"{metric_name}_delta_lora_minus_original"] = lora_value - original_value

    result: dict[str, Any] = {
        "sample_index": paths["sample_index"],
        "seed": paths["seed"],
        "original_path": str(paths["original_path"]),
        "lora_path": str(paths["lora_path"]),
        "teacher_path": str(paths["teacher_path"]),
        "num_frames": int(teacher.shape[0]),
        "height": int(teacher.shape[1]),
        "width": int(teacher.shape[2]),
        "primary_metric": metric_names[0],
        "primary_winner": wins[metric_names[0]],
    }
    for metric_name in metric_names:
        result[f"original_{metric_name}"] = original_metrics[metric_name]
        result[f"lora_{metric_name}"] = lora_metrics[metric_name]
        result[f"{metric_name}_winner"] = wins[metric_name]
    result.update(deltas)
    return result


def prepare_video(video: torch.Tensor, frame_stride: int) -> torch.Tensor:
    if frame_stride <= 1:
        return video
    return video[::frame_stride]


def align_videos(*videos: torch.Tensor, resize_mode: str) -> tuple[torch.Tensor, ...]:
    min_frames = min(video.shape[0] for video in videos)
    videos = tuple(video[:min_frames] for video in videos)
    target_h = min(video.shape[1] for video in videos)
    target_w = min(video.shape[2] for video in videos)
    if resize_mode == "teacher":
        target_h, target_w = videos[-1].shape[1:3]
    elif resize_mode == "first":
        target_h, target_w = videos[0].shape[1:3]
    elif resize_mode != "min":
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")

    aligned = []
    for video in videos:
        if video.shape[1] == target_h and video.shape[2] == target_w:
            aligned.append(video)
            continue
        frames = video.permute(0, 3, 1, 2).contiguous()
        frames = F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)
        aligned.append(frames.permute(0, 2, 3, 1).contiguous())
    return tuple(aligned)


class VideoSimilarityMetrics:
    def __init__(self, metric_names: list[str], device: torch.device) -> None:
        self.metric_names = metric_names
        self.device = device
        self.torchmetrics = self._build_torchmetrics(metric_names, device)

    def compute(self, gen_video: torch.Tensor, ref_video: torch.Tensor, batch_size: int) -> dict[str, float]:
        gen_video = gen_video.float().clamp(0, 1)
        ref_video = ref_video.float().clamp(0, 1)
        if gen_video.shape != ref_video.shape:
            raise ValueError(f"gen/ref video shape mismatch: {tuple(gen_video.shape)} vs {tuple(ref_video.shape)}")

        result: dict[str, float] = {}
        if "l1" in self.metric_names:
            result["l1"] = float((gen_video - ref_video).abs().mean())
        if "mse" in self.metric_names or "psnr" in self.metric_names:
            mse = float(F.mse_loss(gen_video, ref_video))
            result["mse"] = mse
            result["psnr"] = psnr(mse)
        if "temporal_l1" in self.metric_names:
            result["temporal_l1"] = temporal_l1(gen_video, ref_video)

        if self.torchmetrics:
            gen_frames = video_to_nchw(gen_video).to(self.device)
            ref_frames = video_to_nchw(ref_video).to(self.device)
            batch_size = max(1, int(batch_size))
            with torch.no_grad():
                for metric in self.torchmetrics.values():
                    metric.reset()
                for start in range(0, gen_frames.shape[0], batch_size):
                    gen_batch = gen_frames[start : start + batch_size]
                    ref_batch = ref_frames[start : start + batch_size]
                    for metric in self.torchmetrics.values():
                        metric.update(gen_batch, ref_batch)
                for name, metric in self.torchmetrics.items():
                    result[name] = float(metric.compute().item())

        return {name: result[name] for name in self.metric_names}

    def _build_torchmetrics(self, metric_names: list[str], device: torch.device) -> dict[str, Any]:
        names = [name for name in metric_names if name in {"ssim", "lpips"}]
        if not names:
            return {}
        try:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
        except Exception as exc:
            raise RuntimeError(
                "SSIM/LPIPS require torchmetrics, and LPIPS also requires lpips. "
                "Install with: pip install torchmetrics lpips"
            ) from exc

        metrics = {}
        for name in names:
            if name == "ssim":
                metrics[name] = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
            elif name == "lpips":
                metrics[name] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        return metrics


def temporal_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.shape[0] < 2:
        return 0.0
    pred_delta = pred[1:] - pred[:-1]
    target_delta = target[1:] - target[:-1]
    return float((pred_delta - target_delta).abs().mean())


def video_to_nchw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video must be [T,H,W,C], got {tuple(video.shape)}")
    return video.permute(0, 3, 1, 2).contiguous()


def psnr(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def winner(original: float, lora: float, metric_name: str) -> str:
    if not math.isfinite(original) or not math.isfinite(lora):
        return "tie"
    eps = 1e-12
    if abs(original - lora) <= eps:
        return "tie"
    if metric_name in HIGHER_IS_BETTER:
        return "lora" if lora > original else "original"
    return "lora" if lora < original else "original"


def build_summary(rows: list[dict[str, Any]], metrics: list[str], primary_metric: str) -> dict[str, Any]:
    metric_rows = []
    for metric_name in metrics:
        original_values = finite_values(row[f"original_{metric_name}"] for row in rows)
        lora_values = finite_values(row[f"lora_{metric_name}"] for row in rows)
        delta_values = finite_values(row[f"{metric_name}_delta_lora_minus_original"] for row in rows)
        lora_wins = sum(1 for row in rows if row[f"{metric_name}_winner"] == "lora")
        original_wins = sum(1 for row in rows if row[f"{metric_name}_winner"] == "original")
        ties = len(rows) - lora_wins - original_wins
        metric_rows.append(
            {
                "metric": metric_name,
                "better": metric_direction(metric_name),
                "samples": len(rows),
                "original_mean": mean(original_values),
                "lora_mean": mean(lora_values),
                "delta_lora_minus_original_mean": mean(delta_values),
                "delta_lora_minus_original_std": pstdev(delta_values) if len(delta_values) > 1 else 0.0,
                "lora_wins": lora_wins,
                "original_wins": original_wins,
                "ties": ties,
                "lora_win_rate": lora_wins / len(rows),
            }
        )

    primary = next(row for row in metric_rows if row["metric"] == primary_metric)
    if primary["lora_wins"] > primary["original_wins"]:
        overall_winner = "lora"
    elif primary["original_wins"] > primary["lora_wins"]:
        overall_winner = "original"
    else:
        overall_winner = "tie"

    return {
        "num_samples": len(rows),
        "primary_metric": primary_metric,
        "overall_winner": overall_winner,
        "metrics": metric_rows,
    }


def finite_values(values: Any) -> list[float]:
    result = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    if not result:
        raise ValueError("Expected at least one finite value")
    return result


def metric_direction(metric_name: str) -> str:
    if metric_name in HIGHER_IS_BETTER:
        return "higher"
    if metric_name in LOWER_IS_BETTER:
        return "lower"
    raise ValueError(f"Unsupported metric: {metric_name}")


def format_progress(index: int, total: int, row: dict[str, Any]) -> str:
    primary = row["primary_metric"]
    return (
        f"[{index}/{total}] idx={row['sample_index']:02d} seed={row['seed']} "
        f"{primary} original={row[f'original_{primary}']:.6f} "
        f"lora={row[f'lora_{primary}']:.6f} winner={row['primary_winner']}"
    )


def sample_columns(metrics: list[str]) -> list[str]:
    columns = [
        "sample_index",
        "seed",
        "num_frames",
        "height",
        "width",
        "primary_metric",
        "primary_winner",
    ]
    for metric_name in metrics:
        columns.extend(
            [
                f"original_{metric_name}",
                f"lora_{metric_name}",
                f"{metric_name}_delta_lora_minus_original",
                f"{metric_name}_winner",
            ]
        )
    columns.extend(["original_path", "lora_path", "teacher_path"])
    return columns


def summary_columns() -> list[str]:
    return [
        "metric",
        "better",
        "samples",
        "original_mean",
        "lora_mean",
        "delta_lora_minus_original_mean",
        "delta_lora_minus_original_std",
        "lora_wins",
        "original_wins",
        "ties",
        "lora_win_rate",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether original3_clean_pred or LoRA clean prediction is closer to teacher4 videos."
    )
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT, help="Compare output root containing videos/<case> dirs.")
    parser.add_argument("--original_case", default=DEFAULT_CASES["original"])
    parser.add_argument("--lora_case", default=DEFAULT_CASES["lora"])
    parser.add_argument("--teacher_case", default=DEFAULT_CASES["teacher"])
    parser.add_argument("--result_dir", help="Directory for metric outputs. Defaults to <out_root>/metrics.")
    parser.add_argument("--metrics", nargs="+", choices=SUPPORTED_METRICS, default=["l1", "mse", "psnr", "ssim", "temporal_l1"])
    parser.add_argument("--primary_metric", choices=SUPPORTED_METRICS, default="l1")
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=0, help="Limit decoded frames per video. 0 means all frames.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Use every Nth frame for faster evaluation.")
    parser.add_argument("--resize_mode", choices=["min", "teacher", "first"], default="teacher")
    parser.add_argument("--limit", type=int, default=0, help="Limit matched triples. 0 means all.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quiet_missing", action="store_true")
    parser.add_argument("--jsonl_name", default="original_lora_teacher_metrics.jsonl")
    parser.add_argument("--csv_name", default="original_lora_teacher_metrics.csv")
    parser.add_argument("--summary_json_name", default="original_lora_teacher_summary.json")
    parser.add_argument("--summary_csv_name", default="original_lora_teacher_summary.csv")
    args = parser.parse_args()
    if args.primary_metric not in args.metrics:
        args.metrics = [args.primary_metric, *args.metrics]
    else:
        args.metrics = [args.primary_metric, *[metric for metric in args.metrics if metric != args.primary_metric]]
    if args.max_frames <= 0:
        args.max_frames = None
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    return args


if __name__ == "__main__":
    main()
