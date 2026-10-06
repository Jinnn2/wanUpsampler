"""Asset-only FlashVSR diagnostic. No DiT/T5 generation or frozen-study writes.

Plan/report/fetch are stdlib-only. Prepare uses the existing native VAE venv;
FlashVSR runs in a NEW isolated environment. Never treat native-LR vs FULL
pixel distance as ground-truth reconstruction quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "UNIV_adaptor/configs/flashvsr_asset_diagnostic_v1.json"
# The author commit stores this as an ordinary 4 MiB Git blob, not an LFS
# pointer. Pin its RAW bytes independently of machine-specific clean filters.
FLASH_AUTHOR_COMMIT = "cf910c61a60733e610e9c6e8b607f80c3a6c202b"
FLASH_CONTEXT = {
    "relative_path": "examples/WanVSR/prompt_tensor/posi_prompt.pth",
    "git_blob": "95c840b20b8d6aa0d16504ee04e49f1c01a7f209",
    "bytes": 4195504,
    "sha256": "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path, value):
    """Immutable, exclusive creation; interrupted files are never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing output differs; choose a new directory: {path}")
        return
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def outside(out, source):
    out, source = Path(out).resolve(), Path(source).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Diagnostic output must be outside the source and cannot be its ancestor")


def bound_file(path, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required original asset not found: {path}")
    identity = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}
    if expected and identity["sha256"] != expected:
        raise ValueError(f"Asset changed: {path}")
    return identity


def verify_files(identities):
    for identity in identities:
        bound_file(identity["path"], identity["sha256"])


def git_changes(path):
    """NUL parsing preserves unstaged/index distinction and arbitrary names."""
    output = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain=v1",
                                      "-z", "--untracked-files=all"], text=True)
    entries = iter(output.split("\0"))
    changes = []
    for entry in entries:
        if not entry:
            continue
        if len(entry) < 4 or entry[2] != " ":
            raise ValueError("Malformed Git status")
        row = {"status": entry[:2], "path": entry[3:]}
        if "R" in row["status"] or "C" in row["status"]:
            row["original_path"] = next(entries, "")
        changes.append(row)
    return changes


def author_context_bytes(path):
    """Read the canonical object, bypassing checkout/clean/smudge conversions."""
    raw = subprocess.check_output(["git", "-C", str(path), "cat-file", "blob", FLASH_CONTEXT["git_blob"]])
    actual = hashlib.sha256(raw).hexdigest()
    if actual != FLASH_CONTEXT["sha256"] or len(raw) != FLASH_CONTEXT["bytes"]:
        raise ValueError("Author prompt tensor Git object RAW bytes differ: "
                         f"expected_sha256={FLASH_CONTEXT['sha256']}, actual_sha256={actual}, "
                         f"expected_bytes={FLASH_CONTEXT['bytes']}, actual_bytes={len(raw)}. "
                         "The canonical Git object is invalid; inspect object storage/transfer. "
                         "The working-tree file will not be used as a fallback.")
    return raw


def verify_author_context(path):
    relative = FLASH_CONTEXT["relative_path"]
    head_blob = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD:" + relative], text=True).strip()
    stage = subprocess.check_output(["git", "-C", str(path), "ls-files", "--stage", "-z", "--", relative], text=True)
    expected_stage = f"100644 {FLASH_CONTEXT['git_blob']} 0\t{relative}\0"
    if head_blob != FLASH_CONTEXT["git_blob"] or stage != expected_stage:
        raise ValueError("Author prompt tensor HEAD/index differs; do not overwrite or ignore staged changes")
    tensor_path = Path(path) / relative
    if tensor_path.is_symlink():
        raise ValueError("Author prompt tensor is unexpectedly a symlink")
    author_context_bytes(path)
    worktree = bound_file(tensor_path)
    return {"input_source": "git_cat_file_blob", "git_blob": FLASH_CONTEXT["git_blob"],
            "bytes": FLASH_CONTEXT["bytes"], "sha256": FLASH_CONTEXT["sha256"],
            "worktree": worktree,
            "worktree_matches_author": worktree["sha256"] == FLASH_CONTEXT["sha256"]
                                       and worktree["bytes"] == FLASH_CONTEXT["bytes"]}


def checkout(path, commit):
    path = Path(path).resolve()
    top = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != path:
        raise ValueError(f"Expected an independent source checkout, not a parent repository: {path}")
    current = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if current != commit:
        raise ValueError(f"Pinned commit mismatch: {path}; do not reset user changes. HEAD={current}")
    context = verify_author_context(path) if commit == FLASH_AUTHOR_COMMIT else None
    changes = git_changes(path)
    verified = []
    rejected = []
    for change in changes:
        if context and change == {"status": " M", "path": FLASH_CONTEXT["relative_path"]}:
            # This ONE unstaged binary is never consumed. Preserve it and bind
            # the canonical Git object instead; all source/staged edits fail.
            attributes = subprocess.check_output(["git", "-C", str(path), "check-attr",
                                                  "filter", "text", "eol", "working-tree-encoding",
                                                  "--", FLASH_CONTEXT["relative_path"]], text=True).strip()
            verified.append(dict(change, verification="HEAD/index/canonical Git RAW bytes match; worktree is NOT executed",
                                 execution_sha256=context["sha256"], worktree=context["worktree"],
                                 worktree_matches_author=context["worktree_matches_author"], git_attributes=attributes))
        else:
            rejected.append(change)
    if rejected:
        raise ValueError(f"Pinned checkout dirty: {path}; do not reset user changes. status={rejected}")
    result = {"path": str(path), "commit": current}
    if context:
        result["author_prompt_tensor"] = context
        result["isolated_context_worktree_changes"] = verified
    return result


def fetch(args):
    cfg = read(args.config)
    for name, path in (("flash", args.flash_root), ("kernel", args.kernel_root)):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", cfg[name + "_repository"], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", cfg[name + "_commit"]], check=True)
            subprocess.run(["git", "-C", str(path), "submodule", "update", "--init", "--recursive"], check=True)
        identity = checkout(path, cfg[name + "_commit"])
        print(f"Verified {name}: {path}", flush=True)
        if identity.get("isolated_context_worktree_changes"):
            print(json.dumps(identity["isolated_context_worktree_changes"], indent=2), flush=True)
        context = identity.get("author_prompt_tensor")
        if context and not context["worktree_matches_author"]:
            print("WARNING: Worktree prompt tensor differs and is preserved, NOT loaded. "
                  f"Inference uses verified Git blob SHA256={context['sha256']}; "
                  f"unused worktree SHA256={context['worktree']['sha256']}. "
                  "The cause of the worktree difference is not established.", flush=True)


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    cfg = read(args.config)
    manifest = args.weights / "diagnostic_weight_binding.json"
    if manifest.exists():
        binding = read(manifest)
        if binding["repository"] != cfg["weights_repository"]:
            raise ValueError("Wrong existing weight binding")
        verify_files(binding["files"])
        print(f"Verified existing weights at revision {binding['revision']}")
        return
    # Resolve main ONCE, then download the immutable revision and hash all weights.
    revision = HfApi().model_info(cfg["weights_repository"], revision=args.weight_revision).sha
    snapshot_download(repo_id=cfg["weights_repository"], revision=revision,
                      local_dir=str(args.weights), allow_patterns=cfg["weights_files"])
    files = [bound_file(args.weights / name) for name in cfg["weights_files"]]
    for entry in files:
        entry["mtime_ns"] = Path(entry["path"]).stat().st_mtime_ns
    write_new(manifest, {"repository": cfg["weights_repository"], "revision": revision, "files": files})
    print(f"Downloaded/hashed author weights: {revision}")


def implementation():
    return {"driver": file_hash(__file__)}


def freeze_pairs(source, cfg):
    source = Path(source).resolve()
    frozen = read(source / "plan.json")
    if frozen.get("study_schema") != "published_wan21_study_plan_v2":
        raise ValueError("Expected published_wan21_study_v2, not a sparse Phase3 directory")
    if digest({k: v for k, v in frozen.items() if k != "plan_sha256"}) != frozen["plan_sha256"]:
        raise ValueError("Source frozen plan hash mismatch")
    sampling = frozen["config"]["sampling"]
    width, height = map(int, sampling["size"].split("*"))
    if width % 4 or height % 4 or not 25 <= cfg["frames"] <= sampling["frame_num"]:
        raise ValueError("Use 25..source-frame-count frames; FULL canvas must support exact x4 downsampling")
    if not cfg["prompt_ids"] or not cfg["seeds"] or not cfg["spatial_arms"]:
        raise ValueError("Empty selection")
    for key in ("prompt_ids", "seeds", "spatial_arms"):
        if len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"Duplicate selection: {key}")
    jobs = {j["id"]: j for j in frozen["jobs"]}
    pairs = []
    for prompt_id in cfg["prompt_ids"]:
        for seed in cfg["seeds"]:
            for arm in cfg["spatial_arms"]:
                if arm not in ("S_B025", "S_B050"):
                    raise ValueError("Only saved SPATIAL endpoints; do not use temporal/final HR4 clips as LR")
                rows = []
                for method in ("FULL50", arm):
                    job_id = f"pilot_{prompt_id}_s{seed}_{method}"
                    path = source / "records" / (job_id + ".json")
                    row = read(path)
                    if job_id not in jobs or row["job"] != jobs[job_id] or row["plan_sha256"] != frozen["plan_sha256"]:
                        raise ValueError(f"Source receipt/plan mismatch: {job_id}")
                    rows.append(row)
                full, spatial = rows
                case, endpoint = spatial["job"]["arm"]["endpoint"], spatial["endpoint"]
                if (full["job"]["prompt"] != spatial["job"]["prompt"] or
                        case["transition"] != "rgb_sr_vae" or case["frames"] != sampling["frame_num"] or
                        endpoint["main_steps"] != 50 or endpoint["main_terminal_sigma"] != 0):
                    raise ValueError("Not a completed same-prompt, full-duration clean SPATIAL endpoint")
                expected_shape = [16, (case["frames"] - 1) // 4 + 1, case["height"] // 8, case["width"] // 8]
                if endpoint["main_latent_shape"] != expected_shape or endpoint["main_clean"]["shape"] != expected_shape:
                    raise ValueError("Saved main endpoint shape does not match the source arm")
                artifacts = endpoint["artifacts"]
                if len(artifacts) != 1 or artifacts[0]["kind"] != "main_clean_sigma_zero":
                    raise ValueError("Missing unrefined main endpoint")
                receipt_files = [bound_file(source / "records" / (r["job"]["id"] + ".json")) for r in rows]
                pairs.append({"id": f"{prompt_id}_s{seed}_{arm}", "prompt_id": prompt_id, "seed": seed,
                              "prompt": full["job"]["prompt"], "arm": arm,
                              "width": width, "height": height, "frames": cfg["frames"],
                              "source_frames": sampling["frame_num"],
                              "fps": full["sampling_identity"]["fps"],
                              "lr_width": case["width"], "lr_height": case["height"],
                              "full_video": bound_file(full["video_path"], full["video_sha256"]),
                              "endpoint": bound_file(artifacts[0]["path"], artifacts[0]["sha256"]),
                              "main_clean": endpoint["main_clean"], "receipt_files": receipt_files,
                              "lr_main_seconds": endpoint["timing"]["main_seconds"],
                              "full_pipeline_seconds": full["runtime"]["pipeline_seconds"],
                              "source_noise_caveat": "Same seed/shared field does not imply identical geometry/trajectory/subject."})
    return frozen, pairs


def plan(args):
    outside(args.out, args.study_root)
    for source in (args.flash_root, args.kernel_root, args.weights):
        outside(args.out, source)
    cfg = read(args.config)
    if cfg["schema"] != "flashvsr_asset_diagnostic_config_v1" or cfg["flash_scale"] != 4:
        raise ValueError("This diagnostic locks the author's x4 model; no silent x2 substitution")
    for key in ("prompt_ids", "seeds", "spatial_arms", "frames"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    frozen, pairs = freeze_pairs(args.study_root, cfg)
    native_source = ROOT / "UNIV_adaptor/external/wan21/wan/modules/vae.py"
    weights = read(args.weights / "diagnostic_weight_binding.json")
    if weights["repository"] != cfg["weights_repository"] or not weights.get("revision"):
        raise ValueError("Weights must be revision-bound; run download first")
    verify_files(weights["files"])
    body = {"schema": "flashvsr_asset_diagnostic_plan_v1", "config": cfg, "pairs": pairs,
            "source_root": str(args.study_root.resolve()), "source_plan": bound_file(args.study_root / "plan.json"),
            "source_plan_sha256": frozen["plan_sha256"], "implementation": implementation(),
            "native_vae_source": bound_file(native_source),
            "native_vae_weights": bound_file(Path(frozen["model_root"]) / "Wan2.1_VAE.pth"),
            "weights": weights, "weight_binding": bound_file(args.weights / "diagnostic_weight_binding.json"),
            "flash_source": checkout(args.flash_root, cfg["flash_commit"]),
            "kernel_source": checkout(args.kernel_root, cfg["kernel_commit"])}
    body["plan_sha256"] = digest(body)
    if args.out.exists() and not (args.out / "plan.json").exists() and any(args.out.iterdir()):
        raise ValueError("Unplanned nonempty output; use a NEW diagnostic directory")
    write_new(args.out / "plan.json", body)
    print(f"Frozen {len(pairs)} matched pairs, {2*len(pairs)} SR jobs, first {cfg['frames']} frames. No new video generation.")


def load_plan(out):
    p = read(Path(out) / "plan.json")
    if p["schema"] != "flashvsr_asset_diagnostic_plan_v1" or digest({k: v for k, v in p.items() if k != "plan_sha256"}) != p["plan_sha256"]:
        raise ValueError("Diagnostic plan hash mismatch")
    if p["implementation"] != implementation():
        raise ValueError("Diagnostic code changed after planning; use a new directory")
    outside(out, p["source_root"])
    return p


def padding_spec(width, height, frames, scale=4):
    if min(width, height) < 1 or frames < 25 or scale != 4:
        raise ValueError("x4 Tiny requires a positive canvas and at least 25 source frames")
    pw, ph = math.ceil(width / 32) * 32, math.ceil(height / 32) * 32
    # Tiny returns F-4 frames. Official example floors F and loses tail frames.
    padded_frames = math.ceil((frames + 3) / 8) * 8 + 1
    return {"source_width": width, "source_height": height, "source_frames": frames,
            "lr_padded_width": pw, "lr_padded_height": ph,
            "flash_width": pw * scale, "flash_height": ph * scale,
            "content_width": width * scale, "content_height": height * scale,
            "flash_frames": padded_frames, "expected_raw_output_frames": padded_frames - 4,
            "tail_pad_frames": padded_frames - frames,
            "policy": "right-bottom edge padding and last-frame repeat; no crop or temporal interpolation"}


def resize_frames(frames, size, resample):
    import numpy as np
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(frame).resize(size, resample), dtype=np.uint8) for frame in frames])


def decode_video(path, frames):
    import imageio.v2 as imageio
    import numpy as np
    with imageio.get_reader(path) as reader:
        fps = float(reader.get_meta_data()["fps"])
        array = np.stack([reader.get_data(i) for i in range(frames)])
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Unexpected original video RGB/fps")
    return array, fps


def save_frames(out, stem, frames, fps):
    import imageio.v2 as imageio
    import numpy as np
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    npz, video = out / (stem + ".npz"), out / (stem + ".mp4")
    pngs = [out / f"{stem}_{index:03d}.png" for index in sorted({0, len(frames)//2, len(frames)-1})]
    if npz.exists() or video.exists() or any(path.exists() for path in pngs):
        raise FileExistsError(f"Unreceipted output exists; inspect/move it, do not overwrite: {stem}")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Save only audited uint8 THWC frames")
    with npz.open("xb") as handle:
        np.savez_compressed(handle, frames=frames)
    with imageio.get_writer(video, fps=fps, codec="libx264", macro_block_size=1,
                            ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p"]) as writer:
        for frame in frames:
            writer.append_data(frame)
    # PNG is authoritative for fine detail; MP4 is a common-encoding preview.
    for index in sorted({0, len(frames)//2, len(frames)-1}):
        imageio.imwrite(out / f"{stem}_{index:03d}.png", frames[index])
    check, actual_fps = decode_video(video, len(frames))
    if check.shape != frames.shape or abs(actual_fps - fps) > .01:
        raise ValueError("Encoded preview geometry/fps changed")
    return {"npz": bound_file(npz), "video": bound_file(video), "shape": list(frames.shape), "fps": fps}


def load_frames(identity):
    import numpy as np
    verify_files([identity["npz"]])
    with np.load(identity["npz"]["path"], allow_pickle=False) as archive:
        frames = archive["frames"]
    if frames.dtype != np.uint8 or list(frames.shape) != identity["shape"]:
        raise ValueError("Prepared frame tensor mismatch")
    return frames


def validate_clean(payload, pair):
    import torch
    clean = payload["main_clean"]
    if payload["prompt"] != pair["prompt"] or payload["seed"] != pair["seed"] or payload["arm"] != pair["arm"]:
        raise ValueError("Endpoint prompt/seed/arm mismatch")
    if not torch.isfinite(clean).all():
        raise ValueError("Non-finite saved clean latent")
    array = clean.detach().float().cpu().contiguous().numpy()
    if {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()} != pair["main_clean"]:
        raise ValueError("Clean endpoint differs from source receipt")
    return clean


def prepare(args):
    import numpy as np
    from PIL import Image
    import torch
    p = load_plan(args.out)
    verify_files([p["source_plan"], p["native_vae_source"], p["native_vae_weights"]])
    if not torch.cuda.is_available():
        raise RuntimeError("Prepare requires the existing native generation CUDA venv; no DiT/T5 needed")
    spec = importlib.util.spec_from_file_location("diagnostic_native_vae", p["native_vae_source"]["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vae = module.WanVAE(vae_pth=p["native_vae_weights"]["path"], dtype=torch.float32, device="cuda:0")
    for pair in p["pairs"]:
        receipt = args.out / "prepared" / (pair["id"] + ".json")
        verify_files(pair["receipt_files"] + [pair["full_video"], pair["endpoint"]])
        if receipt.exists():
            prior = read(receipt)
            if prior["plan_sha256"] != p["plan_sha256"]:
                raise ValueError("Prepared receipt plan mismatch")
            for item in prior["assets"].values():
                verify_files([item["npz"], item["video"]])
            continue
        full, fps = decode_video(pair["full_video"]["path"], pair["frames"])
        if full.shape[1:3] != (pair["height"], pair["width"]) or abs(fps - pair["fps"]) > .01:
            raise ValueError("Original FULL geometry/fps mismatch")
        payload = torch.load(pair["endpoint"]["path"], map_location="cpu", weights_only=True)
        clean = validate_clean(payload, pair).to("cuda:0")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            decoded = vae.decode([clean])[0].float()
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - start
        if not torch.isfinite(decoded).all() or tuple(decoded.shape) != (3, pair["source_frames"], pair["lr_height"], pair["lr_width"]):
            raise ValueError("Native main VAE decode geometry/non-finite error")
        lr = ((decoded[:, :pair["frames"]].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        control = resize_frames(full, (pair["width"]//4, pair["height"]//4), Image.Resampling.BOX)
        folder = args.out / "media" / pair["id"]
        assets = {}
        for name, frames in (("FULL", full), ("HR_DOWN4", control), ("NATIVE_LR", lr),
                             ("HR_DOWN4_BICUBIC", resize_frames(control, (pair["width"], pair["height"]), Image.Resampling.BICUBIC)),
                             ("NATIVE_LR_BICUBIC", resize_frames(lr, (pair["width"], pair["height"]), Image.Resampling.BICUBIC))):
            assets[name] = save_frames(folder, name, frames, fps)
        write_new(receipt, {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"], "assets": assets,
                            "native_decode_full_clip_seconds": decode_seconds,
                            "timing_scope": "Decode entire original LR endpoint, even when reviewing only a prefix; exclude saving/audit."})
        print(f"Prepared {pair['id']}: FULL and unrefined native LR; no HR4 or DiT rerun", flush=True)
        del decoded, clean, payload


def verify_sr_sources(p):
    checkout(p["flash_source"]["path"], p["flash_source"]["commit"])
    checkout(p["kernel_source"]["path"], p["kernel_source"]["commit"])
    verify_files([p["weight_binding"]])
    # Files were SHA256-verified at download/plan; each worker checks frozen stat
    # rather than redundantly reading multi-GB weights on all eight GPUs.
    for item in p["weights"]["files"]:
        stat = Path(item["path"]).stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("Weight stat changed after SHA256 binding; re-audit weights")


def init_flash(p):
    import torch
    verify_sr_sources(p)
    if not torch.cuda.is_available():
        raise RuntimeError("FlashVSR requires CUDA; no interpolation/dense-attention fallback")
    root = Path(p["flash_source"]["path"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "examples/WanVSR"))
    from block_sparse_attn import block_sparse_attn_func  # required official LCSA backend
    from diffsynth import ModelManager, FlashVSRTinyPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder
    from diffsynth.models import wan_video_dit
    if block_sparse_attn_func is None:
        raise RuntimeError("Missing block sparse backend")
    files = {Path(item["path"]).name: item["path"] for item in p["weights"]["files"]}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    manager.load_models([files["diffusion_pytorch_model_streaming_dmd.safetensors"]])
    pipe = FlashVSRTinyPipeline.from_model_manager(manager, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(files["LQ_proj_in.ckpt"], map_location="cpu", weights_only=True), strict=True)
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16+768)
    # Upstream uses strict=False; require no missing/unexpected keys instead of
    # silently evaluating a partially initialized restoration model.
    mismatch = pipe.TCDecoder.load_state_dict(torch.load(files["TCDecoder.ckpt"], map_location="cpu", weights_only=True), strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f"TCDecoder checkpoint mismatch: {mismatch}")
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    # Explicitly pass the independently verified canonical tensor. Never load
    # the possibly transformed/modified working-tree .pth, even if status clean.
    context = torch.load(io.BytesIO(author_context_bytes(root)), map_location="cpu", weights_only=True)
    pipe.init_cross_kv(context_tensor=context)
    pipe.load_models_to_device(["dit", "vae"])
    pipe.denoising_model().eval()
    pipe.TCDecoder.eval()
    # Read-only counters on the actual upstream call path: an import alone does
    # not prove LCSA executed. No changes to masks/kernels/numerical behavior.
    counter = {"calls": 0}
    original_sparse = wan_video_dit.block_sparse_attn_func

    def counted_sparse(*pos, **kw):
        counter["calls"] += 1
        return original_sparse(*pos, **kw)

    wan_video_dit.block_sparse_attn_func = counted_sparse
    pipe._diagnostic_sparse_counter = counter
    return pipe


def flash_infer(pipe, frames, cfg, target):
    import numpy as np
    from PIL import Image
    import torch
    spec = padding_spec(frames.shape[2], frames.shape[1], len(frames))
    padded = np.pad(frames, ((0, spec["tail_pad_frames"]), (0, spec["lr_padded_height"]-frames.shape[1]),
                             (0, spec["lr_padded_width"]-frames.shape[2]), (0, 0)), mode="edge")
    # Author preprocessing also bicubic-upsamples the LQ conditioning to HQ size.
    tensors = []
    for frame in padded:
        array = np.array(Image.fromarray(frame).resize((spec["flash_width"], spec["flash_height"]), Image.Resampling.BICUBIC), copy=True)
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    # Match the author: normalize in FP32 BEFORE casting. BF16 arithmetic on
    # raw uint8 makes midpoint values noticeably wrong (e.g. RGB=128).
    lq = torch.stack(tensors, dim=1).unsqueeze(0).to(device="cuda", dtype=torch.float32)
    lq = lq.div_(255.0).mul_(2.0).sub_(1.0).to(dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    sparse_before = pipe._diagnostic_sparse_counter["calls"]
    color_calls = []
    hook = pipe.ColorCorrector.register_forward_hook(lambda *unused: color_calls.append(True))
    try:
        with torch.inference_mode():
            result = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
                          seed=cfg["flash_seed"], LQ_video=lq, num_frames=spec["flash_frames"],
                          height=spec["flash_height"], width=spec["flash_width"],
                          is_full_block=False, if_buffer=True,
                          topk_ratio=cfg["sparse_ratio"]*768*1280/(spec["flash_height"]*spec["flash_width"]),
                          kv_ratio=cfg["kv_ratio"], local_range=cfg["local_range"], color_fix=cfg["color_fix"])
    finally:
        hook.remove()
    torch.cuda.synchronize()
    model_seconds = time.perf_counter() - started
    expected = (3, spec["expected_raw_output_frames"], spec["flash_height"], spec["flash_width"])
    if tuple(result.shape) != expected or not torch.isfinite(result).all():
        raise RuntimeError(f"FlashVSR output failed shape/finite gate: {tuple(result.shape)} != {expected}. No silent tail truncation.")
    sparse_calls = pipe._diagnostic_sparse_counter["calls"] - sparse_before
    if sparse_calls <= 0 or len(color_calls) != int(cfg["color_fix"]):
        raise RuntimeError("Actual sparse/color-correction path did not complete; no silent fallback accepted")
    result = result[:, :len(frames), :spec["content_height"], :spec["content_width"]]
    rgb = ((result.float().clamp(-1, 1) + 1)*127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
    if (rgb.shape[2], rgb.shape[1]) != target:
        rgb = resize_frames(rgb, target, Image.Resampling.LANCZOS)
    return rgb, {"padding": spec, "sr_model_seconds": model_seconds,
                 "actual_sparse_attention_calls": sparse_calls, "successful_color_correction_calls": len(color_calls),
                 "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                 "topk_ratio_argument": cfg["sparse_ratio"]*768*1280/(spec["flash_height"]*spec["flash_width"]),
                 "output_policy": "x4 -> remove padded borders/tail -> optional Lanczos to FULL canvas"}


def jobs(p):
    return [(pair, kind) for pair in p["pairs"] for kind in ("HR_DOWN4", "NATIVE_LR")]


def validate_result(row, p, pair, kind, prepared):
    expected_shape = [pair["frames"], pair["height"], pair["width"], 3]
    timing = row["timing"]
    if (row["plan_sha256"] != p["plan_sha256"] or row["pair_id"] != pair["id"] or row["kind"] != kind or
            row["input_npz_sha256"] != prepared["assets"][kind]["npz"]["sha256"] or
            row["asset"]["shape"] != expected_shape or abs(row["asset"]["fps"] - pair["fps"]) > .01 or
            timing["actual_sparse_attention_calls"] <= 0 or
            timing["successful_color_correction_calls"] != int(p["config"]["color_fix"])):
        raise ValueError("SR receipt input/geometry/backend/plan mismatch")
    for key in ("sr_model_seconds",):
        if not math.isfinite(timing[key]) or timing[key] <= 0:
            raise ValueError("Invalid measured restoration latency")
    width, height = (pair["width"]//4, pair["height"]//4) if kind == "HR_DOWN4" else (pair["lr_width"], pair["lr_height"])
    if timing["padding"] != padding_spec(width, height, pair["frames"]):
        raise ValueError("SR padding/canvas protocol mismatch")


def worker(args):
    import torch
    p = load_plan(args.out)
    selected = [j for i, j in enumerate(jobs(p)) if i % args.ngpus == args.gpu]
    pending = []
    for pair, kind in selected:
        receipt = args.out / "results" / f"{pair['id']}_{kind}.json"
        if receipt.exists():
            row = read(receipt)
            prepared = read(args.out / "prepared" / (pair["id"] + ".json"))
            validate_result(row, p, pair, kind, prepared)
            verify_files([row["asset"]["npz"], row["asset"]["video"]])
        else:
            folder = args.out / "media" / pair["id"]
            stem = kind + "_FLASH"
            if any(folder.glob(stem + "*")):
                raise FileExistsError(f"Unreceipted SR artifacts: {folder}/{stem}; inspect/move before resume")
            pending.append((pair, kind))
    if not pending:
        return
    started = time.perf_counter()
    pipe = init_flash(p)
    load_seconds = time.perf_counter() - started
    warmup_seconds = {}
    for pair, kind in pending:
        prepared = read(args.out / "prepared" / (pair["id"] + ".json"))
        if prepared["plan_sha256"] != p["plan_sha256"]:
            raise ValueError("Prepared plan mismatch")
        frames = load_frames(prepared["assets"][kind])
        target = (pair["width"], pair["height"])
        warm_key = (frames.shape[1], frames.shape[2], frames.shape[0])
        if warm_key not in warmup_seconds:
            start = time.perf_counter()
            flash_infer(pipe, frames, p["config"], target)
            warmup_seconds[warm_key] = time.perf_counter() - start
            print(f"Passed/excluded full-shape warmup {warm_key}", flush=True)
        start = time.perf_counter()
        output, timing = flash_infer(pipe, frames, p["config"], target)
        postprocess_seconds = time.perf_counter() - start
        asset = save_frames(args.out / "media" / pair["id"], kind + "_FLASH", output, pair["fps"])
        write_new(args.out / "results" / f"{pair['id']}_{kind}.json",
                  {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"], "kind": kind, "asset": asset,
                   "input_npz_sha256": prepared["assets"][kind]["npz"]["sha256"], "timing": timing,
                   "sr_preprocess_model_postprocess_seconds": postprocess_seconds,
                   "model_load_seconds": load_seconds, "excluded_warmup_seconds": warmup_seconds[warm_key],
                   "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name(0)},
                   "timing_caution": "Restoration only, excludes original generation and file encoding/audit. Prefix diagnostics cannot prove full-pipeline speedup."})
        print(f"Completed {pair['id']}/{kind}: restoration {postprocess_seconds:.2f}s", flush=True)


def launch(args):
    p = load_plan(args.out)
    for pair in p["pairs"]:
        prepared = read(args.out / "prepared" / (pair["id"] + ".json"))
        if prepared["plan_sha256"] != p["plan_sha256"]:
            raise ValueError("Prepared plan mismatch")
    processes = []
    try:
        for gpu in range(min(args.ngpus, len(jobs(p)))):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1")
            log = args.out / "logs" / f"gpu{gpu}_{time.time_ns()}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("x", encoding="utf-8")
            command = [args.sr_python, str(Path(__file__).resolve()), "worker", "--out", str(args.out.resolve()),
                       "--gpu", str(gpu), "--ngpus", str(args.ngpus)]
            try:
                process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            except BaseException:
                handle.close()
                raise
            processes.append((process, handle, log))
            print(f"GPU {gpu}: {log}", flush=True)
        last_update = 0.0
        while any(process.poll() is None for process, _, _ in processes):
            if time.monotonic() - last_update > 30:
                completed = len(list((args.out / "results").glob("*.json")))
                active = sum(process.poll() is None for process, _, _ in processes)
                print(f"FlashVSR: {completed}/{len(jobs(p))} SR receipts; {active} active GPUs. Progress: logs/", flush=True)
                last_update = time.monotonic()
            time.sleep(5)
        failures = [str(log) for process, _, log in processes if process.returncode != 0]
        if failures:
            raise RuntimeError(f"FlashVSR worker failures; inspect actual logs: {failures}")
    finally:
        for process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            handle.close()


def reconstruction(reference, candidate):
    import numpy as np
    if reference.shape != candidate.shape:
        raise ValueError("Unaligned reconstruction diagnostic")
    delta = reference.astype(np.float64)/255 - candidate.astype(np.float64)/255
    mse = float(np.mean(delta**2))
    return {"mae_0_1": float(np.mean(np.abs(delta))), "psnr_db": 10*math.log10(1/mse) if mse else None,
            "psnr_exact_match": mse == 0,
            "scope": "Valid only for SAME FULL frames downsampled and reconstructed; not native-LR vs FULL."}


def signal_stats(frames):
    import numpy as np
    gray = np.mean(frames.astype(np.float64)/255, axis=-1)
    return {"mean_spatial_gradient": float((np.mean(np.abs(np.diff(gray, axis=1))) + np.mean(np.abs(np.diff(gray, axis=2))))/2),
            "mean_temporal_pixel_change": float(np.mean(np.abs(np.diff(gray, axis=0)))),
            "caution": "Signal diagnostics, NOT quality scores. Sharpening/artifacts increase gradients; motion changes temporal differences."}


def report(args):
    p = load_plan(args.out)
    rows, cards, human = [], [], []
    for pair in p["pairs"]:
        prepared = read(args.out / "prepared" / (pair["id"] + ".json"))
        if prepared["plan_sha256"] != p["plan_sha256"]:
            raise ValueError("Prepared plan mismatch")
        assets = dict(prepared["assets"])
        timings = {}
        for kind in ("HR_DOWN4", "NATIVE_LR"):
            result = read(args.out / "results" / f"{pair['id']}_{kind}.json")
            validate_result(result, p, pair, kind, prepared)
            verify_files([result["asset"]["video"]])
            assets[kind + "_FLASH"] = result["asset"]
            timings[kind] = {"sr_model_seconds": result["timing"]["sr_model_seconds"],
                             "sr_preprocess_model_postprocess_seconds": result["sr_preprocess_model_postprocess_seconds"],
                             "excluded_warmup_seconds": result["excluded_warmup_seconds"]}
        arrays = {name: load_frames(assets[name]) for name in ("FULL", "HR_DOWN4_BICUBIC", "HR_DOWN4_FLASH", "NATIVE_LR_BICUBIC", "NATIVE_LR_FLASH")}
        verify_files([assets[name]["video"] for name in arrays])
        rows.append({"pair_id": pair["id"], "prompt": pair["prompt"],
                     "restoration_timings": timings,
                     "source_native_lr_full_main_seconds": pair["lr_main_seconds"],
                     "native_vae_full_decode_seconds": prepared["native_decode_full_clip_seconds"],
                     "end_to_end_speedup": None,
                     "hr_down4_bicubic_reconstruction": reconstruction(arrays["FULL"], arrays["HR_DOWN4_BICUBIC"]),
                     "hr_down4_flash_reconstruction": reconstruction(arrays["FULL"], arrays["HR_DOWN4_FLASH"]),
                     "signal_diagnostics": {name: signal_stats(array) for name, array in arrays.items()},
                     "native_lr_vs_full_pixel_quality": None,
                     "native_lr_vs_full_reason": "Different generation trajectories/subjects are possible; FULL is not pixel-aligned ground truth."})
        tags = []
        for name in arrays:
            video = Path(assets[name]["video"]["path"]).relative_to(args.out.resolve()).as_posix()
            tags.append(f'<div><p>{html.escape(name)}</p><video controls loop preload="metadata" src="{html.escape(video)}"></video></div>')
        cards.append(f'<section><h2>{html.escape(pair["id"])}</h2><p>{html.escape(pair["prompt"])}</p><div class="grid">{"".join(tags)}</div></section>')
        for comparison in ("FULL_vs_HR_DOWN4_FLASH", "NATIVE_LR_BICUBIC_vs_NATIVE_LR_FLASH", "FULL_vs_NATIVE_LR_FLASH"):
            human.append({"pair_id": pair["id"], "comparison": comparison, "subject_matches_prompt": "",
                          "fine_details": "", "motion_and_flicker": "", "overall_preference": "", "notes": ""})
    result = {"schema": "flashvsr_asset_diagnostic_report_v1", "plan_sha256": p["plan_sha256"], "rows": rows,
              "claim_status": "Diagnosis only. No prompt learnability, method superiority or VBench failure established.",
              "limitations": ["Fixed prefix; small, preselected prompts; original FULL is decoded from lossy MP4.",
                              "Native LR is re-decoded from verified clean endpoints; no original HR4 or RealESRGAN reused.",
                              "Only HR-downsample controls have aligned reconstruction targets.",
                              "Official x4 -> padding removal -> target resize is an explicitly adapted composite pipeline.",
                              "No aggregate invented quality score; no speedup claim from restoration-only timing."]}
    write_new(args.out / "report.json", result)
    page = '<!doctype html><meta charset="utf-8"><title>FlashVSR asset diagnosis</title><style>body{font-family:sans-serif;margin:24px}video{width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}section{border-top:1px solid #aaa;margin-top:32px}</style><h1>Labelled diagnostic review — not a blind human study</h1><p>Compare prompt fulfillment, fine detail, motion and new artifacts separately. FULL is not pixel-aligned ground truth for native LR. PNG/NPZ are authoritative; MP4 is a preview.</p>' + ''.join(cards)
    target = args.out / "review.html"
    if target.exists() and target.read_text(encoding="utf-8") != page:
        raise ValueError("Existing review changed")
    if not target.exists():
        with target.open("x", encoding="utf-8") as handle:
            handle.write(page)
    ratings = args.out / "manual_review.csv"
    if not ratings.exists():
        with ratings.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(human[0]))
            writer.writeheader()
            writer.writerows(human)
    print(f"Saved {args.out / 'report.json'} and review.html. Human diagnosis still required.")


def export(args):
    load_plan(args.out)
    if not (args.out / "report.json").is_file():
        raise ValueError("Run report before export")
    archive = args.out.parent / (args.out.name + "_analysis.tgz")
    if archive.exists():
        raise FileExistsError(f"Will not overwrite export: {archive}")
    with tarfile.open(archive, "x:gz") as bundle:
        for path in sorted(args.out.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".log", ".html", ".csv", ".png", ".mp4"):
                bundle.add(path, arcname=(Path(args.out.name) / path.relative_to(args.out)).as_posix())
    print(f"Exported {archive} ({archive.stat().st_size/1024**2:.1f} MiB). No weights, endpoint tensors or NPZ arrays.")


def check(args):
    p = load_plan(args.out)
    pipe = init_flash(p)
    print(f"FlashVSR v1.1 Tiny loaded with required sparse backend: {type(pipe).__name__}. This is NOT a successful inference receipt.")


def status(args):
    p = load_plan(args.out)
    completed = []
    missing = []
    for pair, kind in jobs(p):
        receipt = args.out / "results" / f"{pair['id']}_{kind}.json"
        if receipt.exists():
            prepared = read(args.out / "prepared" / (pair["id"] + ".json"))
            row = read(receipt)
            validate_result(row, p, pair, kind, prepared)
            verify_files([row["asset"]["video"], row["asset"]["npz"]])
            completed.append(receipt.stem)
        else:
            missing.append(receipt.stem)
    print(json.dumps({"completed_sr_jobs": len(completed), "expected_sr_jobs": len(jobs(p)),
                      "missing": missing, "scope": "Validated SR diagnostic receipts, not complete pipeline generation."}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "download", "plan", "prepare", "check", "run", "worker", "status", "report", "export"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--study-root", type=Path, default=ROOT / "outputs/published_wan21_study_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/flashvsr_asset_diagnostic_v1")
    parser.add_argument("--flash-root", type=Path, default=ROOT / "UNIV_adaptor/external/flashvsr")
    parser.add_argument("--kernel-root", type=Path, default=ROOT / "UNIV_adaptor/external/flashvsr_block_sparse")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/flashvsr_v11")
    parser.add_argument("--weight-revision", default="main")
    parser.add_argument("--prompt-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--spatial-arms", nargs="+")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--sr-python", default=sys.executable)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.ngpus < 1 or not 0 <= args.gpu < args.ngpus:
        parser.error("Positive ngpus and gpu in [0, ngpus) required")
    (launch if args.mode == "run" else globals()[args.mode])(args)


if __name__ == "__main__":
    main()
