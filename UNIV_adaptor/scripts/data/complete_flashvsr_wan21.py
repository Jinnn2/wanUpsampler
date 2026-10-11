"""Complete 24 native-LR -> FlashVSR full clips, with four independent GPUs.

Reuse the proven diagnostic's numerical helpers WITHOUT changing its frozen code.
Orchestration/finalization are stdlib-only. Native and SR workers use existing,
separate environments. No generator or old output is modified.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flashvsr_asset_diagnostic as diag

CONFIG = ROOT / "UNIV_adaptor/configs/flashvsr_wan21_complete_v1.json"
METHOD = "LR_FLASH_B025"
BASE_ARMS = ("FULL50", "STEP25", "TEA008", "SCALING10")


def gpu_ids(value):
    ids = value.split(",")
    if len(ids) != 4 or len(set(ids)) != 4 or any(not x.isdecimal() for x in ids):
        raise ValueError("GPU_IDS must contain exactly four unique physical GPU indices, e.g. 0,1,2,3")
    return [str(int(x)) for x in ids] if len({int(x) for x in ids}) == 4 else _duplicate_gpu()


def _duplicate_gpu():
    raise ValueError("Duplicate physical GPU indices")


def implementation():
    return {"driver": diag.file_hash(__file__), "diagnostic_helpers": diag.file_hash(diag.__file__)}


def validate_config(cfg):
    expected = diag.read(CONFIG)
    locked = {"schema": "flashvsr_wan21_complete_config_v1", "frames": 81,
              "prompt_ids": [f"p{i:02d}" for i in range(12)], "seeds": [42, 3407],
              "spatial_arms": ["S_B025"], "flash_scale": 4, "flash_seed": 0,
              "sparse_ratio": 2.0, "kv_ratio": 3.0, "local_range": 11, "color_fix": True,
              "flash_commit": diag.FLASH_AUTHOR_COMMIT,
              "kernel_commit": "49d6c39e4dc0303442cda3bb758b3925d4399c49"}
    if cfg != expected or any(cfg.get(key) != value for key, value in locked.items()):
        raise ValueError("This completion protocol is locked to all 12 prompts x 2 seeds, 81 frames, S_B025 and proven SR parameters")


def load_plan(out):
    p = diag.read(out / "plan.json")
    if (p["schema"] != "flashvsr_wan21_complete_plan_v1" or
            diag.digest({k: v for k, v in p.items() if k != "plan_sha256"}) != p["plan_sha256"] or
            p["implementation"] != implementation()):
        raise ValueError("Completion plan/code hash mismatch; do not reuse diagnostic or changed-code output")
    validate_config(p["config"])
    if len(p["pairs"]) != 24 or len({x["id"] for x in p["pairs"]}) != 24:
        raise ValueError("Expected exactly 24 distinct full-duration pairs")
    for source in (p["source_root"], p["flash_source"]["path"], p["kernel_source"]["path"], Path(p["weight_binding"]["path"]).parent):
        diag.outside(out, source)
    return p


def plan(args):
    cfg = diag.read(args.config)
    validate_config(cfg)
    for source in (args.study_root, args.flash_root, args.kernel_root, args.weights):
        diag.outside(args.out, source)
    if (args.out / "plan.json").exists():
        p = load_plan(args.out)
        if (p["config"] != cfg or p["source_root"] != str(args.study_root.resolve()) or
                p["flash_source"]["path"] != str(args.flash_root.resolve()) or
                p["kernel_source"]["path"] != str(args.kernel_root.resolve()) or
                p["weight_binding"]["path"] != str((args.weights / "diagnostic_weight_binding.json").resolve())):
            raise ValueError("Existing plan belongs to different inputs")
        diag.verify_files([p["source_plan"], p["weight_binding"]])
        print("Resuming existing frozen 24-clip completion plan", flush=True)
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Unplanned nonempty output; choose a NEW completion directory")
    frozen, pairs = diag.freeze_pairs(args.study_root, cfg)
    jobs = {j["id"]: j for j in frozen["jobs"]}
    expected = {f"pilot_{pid}_s{seed}_FULL50" for pid in cfg["prompt_ids"] for seed in cfg["seeds"]}
    full_jobs = {key for key in jobs if key.startswith("pilot_") and key.endswith("_FULL50")}
    if full_jobs != expected or frozen["config"]["sampling"]["frame_num"] != 81:
        raise ValueError("Source is not the expected complete 12-prompt, 2-seed study")
    for pair in pairs:
        if (pair["width"], pair["height"], pair["lr_width"], pair["lr_height"], pair["fps"]) != (832, 480, 416, 240, 16):
            raise ValueError("Source canvas/fps differs from the completion protocol")
        pair["comparison"] = {}
        for arm in BASE_ARMS:
            key = f"pilot_{pair['prompt_id']}_s{pair['seed']}_{arm}"
            receipt = args.study_root / "records" / (key + ".json")
            row = diag.read(receipt)
            if (row["plan_sha256"] != frozen["plan_sha256"] or row["job"] != jobs[key] or
                    row["job"]["prompt"] != pair["prompt"] or row["sampling_identity"]["fps"] != pair["fps"]):
                raise ValueError(f"Comparison receipt mismatch: {key}")
            pair["comparison"][arm] = {"receipt": diag.bound_file(receipt),
                                        "video": diag.bound_file(row["video_path"], row["video_sha256"])}
    binding = args.weights / "diagnostic_weight_binding.json"
    weights = diag.read(binding)
    if (weights["repository"] != cfg["weights_repository"] or not weights.get("revision") or
            sorted(Path(x["path"]).name for x in weights["files"]) != sorted(cfg["weights_files"])):
        raise ValueError("Missing/wrong author weights; reuse the completed diagnostic setup")
    diag.verify_files(weights["files"])
    p = {"schema": "flashvsr_wan21_complete_plan_v1", "config": cfg, "pairs": pairs,
         "source_root": str(args.study_root.resolve()), "source_plan": diag.bound_file(args.study_root / "plan.json"),
         "source_plan_sha256": frozen["plan_sha256"], "implementation": implementation(),
         "native_vae_source": diag.bound_file(ROOT / "UNIV_adaptor/external/wan21/wan/modules/vae.py"),
         "native_vae_weights": diag.bound_file(Path(frozen["model_root"]) / "Wan2.1_VAE.pth"),
         "weights": weights, "weight_binding": diag.bound_file(binding),
         "flash_source": diag.checkout(args.flash_root, cfg["flash_commit"]),
         "kernel_source": diag.checkout(args.kernel_root, cfg["kernel_commit"])}
    p["plan_sha256"] = diag.digest(p)
    diag.write_new(args.out / "plan.json", p)
    print("Frozen 24 full 81-frame native-LR FlashVSR clips; no new DiT generation", flush=True)


def prepared_path(out, pair):
    return out / "prepared" / (pair["id"] + ".json")


def result_path(out, pair):
    return out / "results" / (pair["id"] + "_NATIVE_LR.json")


def validate_prepared(row, p, pair):
    asset = row["assets"]["NATIVE_LR"]
    if (row["plan_sha256"] != p["plan_sha256"] or row["pair_id"] != pair["id"] or
            row["source_endpoint_sha256"] != pair["endpoint"]["sha256"] or
            set(row["assets"]) != {"NATIVE_LR"} or
            asset["shape"] != [81, pair["lr_height"], pair["lr_width"], 3] or
            abs(asset["fps"] - pair["fps"]) > .01):
        raise ValueError("Prepared native clip plan/endpoint/geometry mismatch; 33-frame receipts cannot resume here")
    seconds = row["native_decode_full_clip_seconds"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid native decode latency")
    diag.verify_files([asset["npz"], asset["video"]])


def completed_result(out, p, pair):
    prepared = diag.read(prepared_path(out, pair))
    validate_prepared(prepared, p, pair)
    row = diag.read(result_path(out, pair))
    diag.validate_result(row, p, pair, "NATIVE_LR", prepared)
    diag.verify_files([row["asset"]["npz"], row["asset"]["video"]])
    return row


def pending_pairs(out, p, phase, slot):
    pending = []
    for i, pair in enumerate(p["pairs"]):
        if i % 4 != slot:
            continue
        path = prepared_path(out, pair) if phase == "prepare" else result_path(out, pair)
        if path.exists():
            if phase == "prepare":
                validate_prepared(diag.read(path), p, pair)
            else:
                completed_result(out, p, pair)
            continue
        stem = "NATIVE_LR" if phase == "prepare" else "NATIVE_LR_FLASH"
        folder = out / "media" / pair["id"]
        if any(folder.glob(stem + "*")):
            raise FileExistsError(f"Unreceipted artifacts: {folder}/{stem}; inspect/move before resuming; never overwritten")
        pending.append(pair)
    return pending


def prepare_worker(args):
    p = load_plan(args.out)
    pending = pending_pairs(args.out, p, "prepare", args.slot)
    if not pending:
        return
    import torch
    diag.verify_files([p["source_plan"], p["native_vae_source"], p["native_vae_weights"]])
    spec = importlib.util.spec_from_file_location("completion_native_vae", p["native_vae_source"]["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vae = module.WanVAE(vae_pth=p["native_vae_weights"]["path"], dtype=torch.float32, device="cuda:0")
    for pair in pending:
        diag.verify_files(pair["receipt_files"] + [pair["endpoint"]])
        payload = torch.load(pair["endpoint"]["path"], map_location="cpu", weights_only=True)
        clean = diag.validate_clean(payload, pair).to("cuda:0")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            decoded = vae.decode([clean])[0].float()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if not torch.isfinite(decoded).all() or tuple(decoded.shape) != (3, 81, pair["lr_height"], pair["lr_width"]):
            raise ValueError("Native decode failed full-duration finite/shape gate")
        frames = ((decoded.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        asset = diag.save_frames(args.out / "media" / pair["id"], "NATIVE_LR", frames, pair["fps"])
        diag.write_new(prepared_path(args.out, pair), {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"],
                        "source_endpoint_sha256": pair["endpoint"]["sha256"], "assets": {"NATIVE_LR": asset},
                        "native_decode_full_clip_seconds": seconds, "native_vae_dtype": "float32"})
        print(f"Prepared {pair['id']}: 81 frames, no HR4/DiT/T5", flush=True)
        del clean, decoded, frames, payload


def sr_worker(args):
    p = load_plan(args.out)
    pending = pending_pairs(args.out, p, "run", args.slot)
    if not pending:
        return
    import torch
    started = time.perf_counter()
    pipe = diag.init_flash(p)
    load_seconds = time.perf_counter() - started
    warmup_seconds = None
    for pair in pending:
        prepared = diag.read(prepared_path(args.out, pair))
        validate_prepared(prepared, p, pair)
        frames = diag.load_frames(prepared["assets"]["NATIVE_LR"])
        target = (pair["width"], pair["height"])
        if warmup_seconds is None:
            started = time.perf_counter()
            diag.flash_infer(pipe, frames, p["config"], target)
            warmup_seconds = time.perf_counter() - started
            print("Passed/excluded full 81-frame shape warmup (89 padded input frames)", flush=True)
        started = time.perf_counter()
        output, timing = diag.flash_infer(pipe, frames, p["config"], target)
        elapsed = time.perf_counter() - started
        asset = diag.save_frames(args.out / "media" / pair["id"], "NATIVE_LR_FLASH", output, pair["fps"])
        row = {"plan_sha256": p["plan_sha256"], "pair_id": pair["id"], "kind": "NATIVE_LR", "asset": asset,
               "input_npz_sha256": prepared["assets"]["NATIVE_LR"]["npz"]["sha256"], "timing": timing,
               "sr_preprocess_model_postprocess_seconds": elapsed, "model_load_seconds": load_seconds,
               "excluded_warmup_seconds": warmup_seconds,
               "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
               "end_to_end_speedup": None,
               "timing_caution": "Restoration only. Original LR generation reused; no fresh end-to-end timing/speedup."}
        diag.validate_result(row, p, pair, "NATIVE_LR", prepared)
        diag.write_new(result_path(args.out, pair), row)
        print(f"Completed {pair['id']}: 81 frames, SR {elapsed:.2f}s", flush=True)
        del frames, output


def worker_command(args, phase, slot, physical):
    python = args.native_python if phase == "prepare" else args.sr_python
    command = [python, str(Path(__file__).resolve()), phase + "-worker", "--out", str(args.out), "--slot", str(slot)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=physical, PYTHONDONTWRITEBYTECODE="1")
    return command, env


def check(args):
    p = load_plan(args.out)
    diag.verify_files([p["source_plan"], p["native_vae_source"], p["native_vae_weights"]])
    diag.verify_sr_sources(p)
    # Check both interpreters without rebuilding environments or loading generators.
    for python, packages in ((args.native_python, "import imageio, numpy"),
                             (args.sr_python, "import imageio, numpy; from block_sparse_attn import block_sparse_attn_func")):
        subprocess.run([python, "-c", "import torch; " + packages +
                        "; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids(args.gpu_ids)[0]), check=True)
    print("Environment/source checks passed. Actual 81-frame inference is checked by each SR worker's excluded warmup.", flush=True)


def launch(args, phase):
    p = load_plan(args.out)
    selected_ids = gpu_ids(args.gpu_ids)
    # Validate all preparation before starting any SR process.
    if phase == "run":
        for pair in p["pairs"]:
            validate_prepared(diag.read(prepared_path(args.out, pair)), p, pair)
    selections = {slot: pending_pairs(args.out, p, phase, slot) for slot in range(4)}
    active = []
    try:
        for slot, physical in enumerate(selected_ids):
            if not selections[slot]:
                print(f"{phase} GPU {physical}: already complete", flush=True)
                continue
            command, env = worker_command(args, phase if phase == "prepare" else "sr", slot, physical)
            log = args.out / "logs" / f"{phase}_gpu{physical}_{time.time_ns()}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("x", encoding="utf-8")
            try:
                process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            except BaseException:
                handle.close()
                raise
            active.append((process, handle, log))
            print(f"{phase} GPU {physical}, slot {slot}: {log}", flush=True)
        last = 0.0
        while any(process.poll() is None for process, _, _ in active):
            if time.monotonic() - last >= 30:
                folder = "prepared" if phase == "prepare" else "results"
                n = len(list((args.out / folder).glob("*.json")))
                print(f"{phase}: {n}/24 receipts; progress in logs/", flush=True)
                last = time.monotonic()
            time.sleep(2)
        failures = [str(log) for process, _, log in active if process.returncode]
        if failures:
            raise RuntimeError(f"Worker failures; inspect these logs: {failures}")
    finally:
        for process, handle, _ in active:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            handle.close()


def status(args):
    p = load_plan(args.out)
    complete, missing = [], []
    for pair in p["pairs"]:
        if result_path(args.out, pair).exists():
            completed_result(args.out, p, pair)
            complete.append(pair["id"])
        else:
            missing.append(pair["id"])
    print(f"Validated full-length FlashVSR videos: {len(complete)}/24; 81 frames, 832x480, 16 fps")
    if missing:
        print("Missing: " + ", ".join(missing))


def finalize(args):
    p = load_plan(args.out)
    diag.verify_files([p["source_plan"]])
    rows = []
    for pair in p["pairs"]:
        if not result_path(args.out, pair).exists():
            raise RuntimeError(f"Incomplete 24-clip dataset: missing {pair['id']}; run prepare/run then finalize")
        result = completed_result(args.out, p, pair)
        for binding in pair["comparison"].values():
            diag.verify_files([binding["receipt"], binding["video"]])
        rows.append({"prompt_id": pair["prompt_id"], "prompt": pair["prompt"], "seed": pair["seed"],
                     "method": METHOD, "frames": 81, "fps": pair["fps"], "width": 832, "height": 480,
                     "asset": result["asset"], "source_endpoint": pair["endpoint"],
                     "source_main_seconds": pair["lr_main_seconds"], "timing": result["timing"],
                     "restoration_seconds": result["sr_preprocess_model_postprocess_seconds"],
                     "comparison": pair["comparison"], "end_to_end_speedup": None})
    manifest = {"schema": "flashvsr_wan21_complete_dataset_v1", "plan_sha256": p["plan_sha256"],
                "videos": rows, "completed_flash_videos": 24, "matched_groups": 24,
                "methods": list(BASE_ARMS) + [METHOD], "matched_comparison_videos": 120,
                "claim": p["config"]["claim"],
                "caveats": ["LR+FlashVSR is an adapted composite spatial baseline, not an official Wan acceleration preset.",
                            "Same prompt/seed does not imply pixel-aligned FULL and native LR trajectories.",
                            "Existing 33-frame screening clips are not counted as full videos.",
                            "No full-pipeline speedup or quality/routing claim follows from completing these assets."]}
    diag.write_new(args.out / "dataset_manifest.json", manifest)
    print(f"Finalized 24 full FlashVSR videos; five-method matched dataset = 120 videos. {args.out / 'dataset_manifest.json'}")


def export(args):
    finalize(args)
    archive = args.out.parent / (args.out.name + "_videos.tgz")
    receipt = args.out / "export_receipt.json"
    if archive.exists():
        if not receipt.exists():
            raise FileExistsError(f"Unreceipted/incomplete export; inspect/move before resuming: {archive}")
        prior = diag.read(receipt)
        if prior["plan_sha256"] != load_plan(args.out)["plan_sha256"]:
            raise ValueError("Export plan mismatch")
        diag.verify_files([prior["archive"]])
        print(f"Verified existing export preserved: {archive}")
        return
    if receipt.exists():
        raise FileNotFoundError("Export receipt exists but archive is missing; inspect/move receipt before re-export")
    paths = [args.out / "plan.json", args.out / "dataset_manifest.json"]
    paths += sorted((args.out / "results").glob("*.json"))
    for pair in load_plan(args.out)["pairs"]:
        folder = args.out / "media" / pair["id"]
        paths += [folder / "NATIVE_LR_FLASH.mp4"] + sorted(folder.glob("NATIVE_LR_FLASH_*.png"))
    with tarfile.open(archive, "x:gz") as bundle:
        for path in paths:
            bundle.add(path, arcname=(Path(args.out.name) / path.relative_to(args.out)).as_posix())
    diag.write_new(receipt, {"plan_sha256": load_plan(args.out)["plan_sha256"], "archive": diag.bound_file(archive)})
    print(f"Exported {archive} ({archive.stat().st_size / 1024**2:.1f} MiB): 24 final MP4s plus provenance/PNGs; no weights/NPZ or redundant baseline copies")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "check", "prepare", "run", "prepare-worker", "sr-worker", "status", "finalize", "export", "all"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--study-root", type=Path, default=ROOT / "outputs/published_wan21_study_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/flashvsr_wan21_complete_v1")
    parser.add_argument("--flash-root", type=Path, default=ROOT / "UNIV_adaptor/external/flashvsr")
    parser.add_argument("--kernel-root", type=Path, default=ROOT / "UNIV_adaptor/external/flashvsr_block_sparse")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/flashvsr_v11")
    parser.add_argument("--native-python", default=str(ROOT / ".venv_published_wan21/bin/python"))
    parser.add_argument("--sr-python", default=str(ROOT / ".venv_flashvsr_diagnostic/bin/python"))
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--slot", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    gpu_ids(args.gpu_ids)
    if args.mode == "all":
        plan(args)
        check(args)
        launch(args, "prepare")
        launch(args, "run")
        finalize(args)
        export(args)
    elif args.mode in ("prepare", "run"):
        launch(args, args.mode)
    else:
        globals()[args.mode.replace("-", "_")](args)


if __name__ == "__main__":
    main()
