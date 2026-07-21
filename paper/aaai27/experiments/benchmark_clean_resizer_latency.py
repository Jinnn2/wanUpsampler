from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_sr.models import build_clean_latent_resizer, infer_clean_resizer_model_type  # noqa: E402
from wan_sr.training.config import load_yaml  # noqa: E402
from wan_sr.training.ema import EMA  # noqa: E402


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the operator latency benchmark")
    device = torch.device("cuda:0")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    load_started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_config = load_yaml(args.train_config)
    model_config = checkpoint.get("config", {}).get("model", train_config.get("model", {}))
    if not model_config:
        raise SystemExit("No model config found in checkpoint or train config")
    model_config = dict(model_config)
    if args.model_class != "auto":
        model_config["model_type"] = args.model_class
    model = build_clean_latent_resizer(model_config)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    if args.use_ema and "ema" in checkpoint:
        ema = EMA(model)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model = model.to(device=device, dtype=dtype).eval()
    torch.cuda.synchronize()
    initialization_s = time.perf_counter() - load_started

    latent = torch.randn(
        args.batch_size,
        args.channels,
        args.frames,
        args.low_height,
        args.low_width,
        device=device,
        dtype=dtype,
    )
    output_size = (args.high_height, args.high_width)

    def interpolation() -> torch.Tensor:
        return F.interpolate(
            latent,
            size=(args.frames, args.high_height, args.high_width),
            mode="trilinear",
            align_corners=False,
        )

    def clean_lifter() -> torch.Tensor:
        return model(latent, output_size=output_size)

    raw = measure_cuda_pair(
        {"trilinear_ms": interpolation, "cll_ms": clean_lifter},
        args.warmup,
        args.repeats,
    )
    trilinear = summarize(raw["trilinear_ms"])
    cll = summarize(raw["cll_ms"])
    payload = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(0),
        "cuda_visible_device_logical_index": 0,
        "precision": args.precision,
        "shape_lr": [args.batch_size, args.channels, args.frames, args.low_height, args.low_width],
        "shape_hr": [args.batch_size, args.channels, args.frames, args.high_height, args.high_width],
        "model_type": infer_clean_resizer_model_type(model_config),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "initialization_s": initialization_s,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "trilinear": trilinear,
        "cll": cll,
        "delta_mean_ms": cll["mean_ms"] - trilinear["mean_ms"],
        "ratio_mean": cll["mean_ms"] / trilinear["mean_ms"],
        "raw": raw,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "raw"}, ensure_ascii=False, indent=2))
    print(f"Operator latency JSON: {output}")


def measure_cuda_pair(
    operations: dict[str, Callable[[], torch.Tensor]],
    warmup: int,
    repeats: int,
) -> dict[str, list[float]]:
    durations = {name: [] for name in operations}
    names = list(operations)
    with torch.inference_mode():
        for _ in range(warmup):
            for operation in operations.values():
                result = operation()
        del result
        torch.cuda.synchronize()

        for repeat in range(repeats):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                result = operations[name]()
                finished.record()
                finished.synchronize()
                durations[name].append(float(started.elapsed_time(finished)))
        del result
    return durations


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA latency of the learned clean latent lifter versus trilinear interpolation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-class", choices=["auto", "stage2", "unet"], default="stage2")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--low-height", type=int, default=46)
    parser.add_argument("--low-width", type=int, default=80)
    parser.add_argument("--high-height", type=int, default=90)
    parser.add_argument("--high-width", type=int, default=156)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("--warmup must be >= 1 and --repeats must be >= 2")
    return args


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
