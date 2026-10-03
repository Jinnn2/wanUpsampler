"""Reuse the published worker; attach endpoint adapter ONLY to custom native arms."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from UNIV_adaptor.scripts.data import published_wan21_worker as worker
from UNIV_adaptor.scripts.data import published_wan21_study as study
from UNIV_adaptor.wan21_endpoint_runtime import EndpointRuntime


def run(args):
    frozen = study.load_plan(args.out)
    all_arms = frozen["config"]["arms"] + frozen["config"]["disabled_arms"]
    arm = next(a for a in all_arms if a["id"] == args.arm)
    original_load, original_immutable = worker.load_entrypoint, worker.immutable
    controller = None

    def load_entrypoint(source):
        nonlocal controller
        module = original_load(source)
        case = arm.get("endpoint")
        if not case:
            return module
        if source != "wan21":
            raise ValueError("Never wrap an official cached forward with the custom endpoint adapter")
        protocol = frozen["config"]["endpoint_protocol"]
        if args.probe and case["transition"] == "rgb_sr_vae":
            import torch
            from UNIV_adaptor.rgb_super_resolution import build_univ_rgb_super_resolver
            sr = build_univ_rgb_super_resolver(protocol["sr"])
            restored = sr.resize(torch.zeros(1, 32, 32, 3), target_height=64, target_width=64)
            if tuple(restored.shape) != (1, 64, 64, 3) or not torch.isfinite(restored).all():
                raise RuntimeError("RealESRGAN CUDA/BasicSR smoke test failed")
            print("RealESRGAN x2 smoke test passed; no bicubic fallback", flush=True)
            del sr, restored
        original_constructor = module.wan.WanT2V

        def constructor(*pos, **kw):
            nonlocal controller
            pipeline = original_constructor(*pos, **kw)
            controller = EndpointRuntime(pipeline, case, protocol, args.out, arm["id"], frozen["config"]["warmup"])
            pipeline.generate = controller.generate
            return pipeline

        module.wan.WanT2V = constructor
        return module

    def save_receipt(path, row):
        if Path(path).parent.name == "records" and arm.get("endpoint"):
            if controller is None or controller.record is None:
                raise RuntimeError("Missing custom endpoint runtime metadata")
            row["endpoint"] = controller.record
            runtime = row["runtime"]
            runtime["pipeline_seconds"] -= controller.instrument_seconds
            runtime["instrumentation_seconds"] += controller.instrument_seconds
            runtime["scope"] += "; custom arms include RGB SR/restore, VAE re-encode, all four full HR steps; exclude clean-endpoint audit/save only"
            study.validate_endpoint_receipt(row)
            if runtime["pipeline_seconds"] <= 0:
                raise RuntimeError("Invalid instrument-subtracted endpoint latency")
        original_immutable(path, row)

    worker.load_plan, worker.receipt_valid = study.load_plan, study.receipt_valid
    worker.load_entrypoint, worker.immutable = load_entrypoint, save_receipt
    worker.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--probe", action="store_true")
    run(parser.parse_args())
