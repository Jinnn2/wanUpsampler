"""Inspect saved endpoint stages without regenerating or modifying frozen assets.

Loads only the native VAE (and SR for S), never the DiT or text encoder. PNGs
are the lossless inspection assets; MP4s are navigation previews, not scores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from UNIV_adaptor.scripts.data import published_wan21_study as study
from UNIV_adaptor.transition import WanDVGAnchorTransition, WanRGBSRTransition
from UNIV_adaptor.wan21_endpoint_runtime import NativeCodec, tensor_hash


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def png_indices(frames):
    # Include non-anchor positions in 5->21 temporal latent reconstruction.
    return sorted({min(i, frames - 1) for i in (0, 4, 8, 16, 24, 40, 56, 80)})


def load_inputs(root, job_id):
    if Path(job_id).name != job_id or not job_id.startswith("pilot_"):
        raise ValueError("Use an exact pilot job ID, not a path")
    frozen = study.load_plan(root)
    record_path = root / "records" / (job_id + ".json")
    row = study.base.read(record_path)
    jobs = [j for j in frozen["jobs"] if j["id"] == job_id]
    if len(jobs) != 1 or row["job"] != jobs[0] or row["plan_sha256"] != frozen["plan_sha256"]:
        raise ValueError("Record does not belong to the frozen plan")
    if not row["job"]["arm"].get("endpoint"):
        raise ValueError("This diagnostic is for endpoint arms only")
    study.validate_endpoint_receipt(row)
    source = Path(row["video_path"])
    if not source.is_file() or file_hash(source) != row["video_sha256"]:
        raise ValueError("Original final MP4 is missing or changed")
    return frozen, row


def native_vae_module():
    # The pinned VAE source is standalone; avoid importing unrelated T5/DiT.
    path = ROOT / "UNIV_adaptor/external/wan21/wan/modules/vae.py"
    spec = importlib.util.spec_from_file_location("diagnostic_native_wan_vae", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    frozen, row = load_inputs(args.study_root.resolve(), args.job)
    endpoint = row["endpoint"]
    metadata = {"job": row["job"], "plan_sha256": frozen["plan_sha256"],
                "diagnostic_implementation_sha256": file_hash(__file__),
                "source_receipt_sha256": file_hash(args.study_root / "records" / (args.job + ".json")),
                "source_final_mp4": row["video_path"], "source_final_sha256": row["video_sha256"],
                "hr": endpoint["hr"], "frozen_environment": row["environment"]}
    if args.inspect_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if args.out is None:
        raise ValueError("--out must name a NEW diagnostic directory")
    out = args.out.resolve()
    study_root = args.study_root.resolve()
    if out == study_root or out in study_root.parents or study_root in out.parents:
        raise ValueError("Diagnostic output must be outside the frozen study directory and cannot be its ancestor")
    if out.exists():
        raise FileExistsError(f"No overwriting diagnostic outputs: {out}")

    import imageio.v2 as imageio
    import torch
    from UNIV_adaptor.rgb_super_resolution import build_univ_rgb_super_resolver

    if not torch.cuda.is_available():
        raise RuntimeError("Run in the existing remote generation venv with CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    artifact = Path(endpoint["artifacts"][0]["path"])
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    clean = payload["main_clean"]
    if payload["prompt"] != row["job"]["prompt"] or payload["seed"] != row["job"]["seed"] or payload["arm"] != row["job"]["arm"]["id"]:
        raise ValueError("Saved endpoint prompt/seed/arm mismatch")
    if not torch.isfinite(clean).all() or tensor_hash(clean) != endpoint["main_clean"]:
        raise ValueError("Saved clean latent differs from the generation receipt")
    model_root = Path(frozen["model_root"])
    vae = native_vae_module().WanVAE(vae_pth=str(model_root / "Wan2.1_VAE.pth"),
                                   dtype=torch.float32, device=device)
    clean = clean.to(device)
    out.mkdir(parents=True, exist_ok=False)
    stages = {}

    def save_rgb(name, rgb):
        # [T,H,W,3], preserving geometry. No sharpening/resizing/normalization.
        rgb = rgb.detach().float().cpu().contiguous()
        if not torch.isfinite(rgb).all():
            raise ValueError(f"Non-finite RGB at {name}")
        frames = rgb.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        with imageio.get_writer(out / (name + ".mp4"), fps=16, codec="libx264",
                                quality=8, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        indices = png_indices(len(frames))
        for index in indices:
            imageio.imwrite(out / f"{name}_frame{index:03d}.png", frames[index])
        stages[name] = {"shape_THWC": list(rgb.shape), "png_frame_indices": indices,
                        "rgb_mean": float(rgb.mean()), "rgb_std": float(rgb.std()),
                        "rgb_min": float(rgb.min()), "rgb_max": float(rgb.max())}
        print(f"Saved {name}: {tuple(rgb.shape)}", flush=True)

    def decode_stage(name, latent):
        rgb = (vae.decode([latent])[0].float() + 1) * .5
        save_rgb(name, rgb.permute(1, 2, 3, 0))

    class CaptureSR:
        def __init__(self, resolver):
            self.resolver = resolver

        def resize(self, video, **kw):
            result = self.resolver.resize(video, **kw)
            save_rgb("02_rgb_sr_before_vae_encode", result)
            return result

    case = row["job"]["arm"]["endpoint"]
    target = tuple(endpoint["target_latent_shape"])
    with torch.no_grad():
        decode_stage("01_main_before_restore", clean)
        if case["transition"] == "dvg_latent_anchor":
            lifted = WanDVGAnchorTransition().lift(clean, target_latent_shape=target)
            restored = lifted.clean_hr
        elif case["transition"] == "rgb_sr_vae":
            sr = CaptureSR(build_univ_rgb_super_resolver(frozen["config"]["endpoint_protocol"]["sr"]))
            lifted = WanRGBSRTransition(vae_codec=NativeCodec(vae), spatial_resolver=sr,
                                       target_height=480, target_width=832).lift(clean, target_latent_shape=target)
            restored = lifted.clean_hr
        elif case["transition"] == "identity":
            restored = clean
        elif case["transition"] == "vae_roundtrip":
            codec = WanRGBSRTransition(vae_codec=NativeCodec(vae), spatial_resolver=None,
                                       target_height=480, target_width=832)
            decoded = codec._decode(clean)
            rgb = ((decoded.float().clamp(-1, 1) + 1) * .5).permute(1, 2, 3, 0).contiguous().cpu()
            restored = codec._encode(rgb, device=device, dtype=clean.dtype)
        else:
            raise ValueError("Unknown endpoint transition")
        if not torch.isfinite(restored).all():
            raise ValueError("Non-finite restored latent")
        restored_hash = tensor_hash(restored)
        match = restored_hash == endpoint["restored_clean"]
        decode_stage("03_restored_before_hr4", restored)
    metadata.update(stages=stages, restored_clean_hash=restored_hash,
                    restored_clean_matches_frozen_receipt=match,
                    diagnostic_torch=torch.__version__, vae_dtype=str(vae.dtype),
                    scope="Saved main endpoint + existing restore + VAE decode only. No 50-step or HR4 rerun; no original assets changed.",
                    caution="PNG/preview inspection, not quality scores. Main short clips are native duration, not matched human pairs. If restored hash mismatches, investigate replay environment before attributing differences to HR4.")
    (out / "diagnostic.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Restored latent hash match: {match}; compare stage 03 with original final MP4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    run(parser.parse_args())
