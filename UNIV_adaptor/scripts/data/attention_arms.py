"""Attention-level accelerator arms for Wan2.1-T2V-1.3B through LightX2V's attention backends.

Machine (2026-11-05): 4x A800 (sm_80). RADIAL was dropped: its MagiAttention kernel needs Hopper.

Every arm, including its own dense reference FULL50_LX, runs on the same LightX2V code path with the same sampling as
the platform (832x480, 81 frames, 50 steps, shift 8, CFG 6, Wan's default negative prompt). Only the video
self-attention backend changes; cross-attention stays FlashAttention-2 in every arm. Platform v1 showed that two code
paths give different FULL clips (like two seeds), so these arms are scored only against FULL50_LX.

  configs  write one LightX2V config JSON per arm (derived from LightX2V configs/wan/wan_t2v.json)
  probe    run every arm once on one prompt and seed, each arm on its own GPU, plus FULL50_LX twice to check that
           the dense path is deterministic. Reports per arm: import / run errors, wall time, LightX2V's own timing
           lines, attention sparsity lines, output sha256. Read-only apart from its output folder.

Arms (see ACCELERATOR_EVAL_SURVEY.md section 4 for the choice):
  FULL50_LX  flash_attn2                 dense reference on this path
  SAGE2      sage_attn2                  quantized attention (INT8 QK / FP8 PV), content-independent
  SVG2       svg2_attn                   dynamic semantic (k-means) sparse attention (needs flashinfer)
  SPARGE     spas_sage_attn2, sparge_mode  SpargeAttn: dynamic block mask from mean-similarity of Q/K blocks, run with
                                         the SageAttention2 block-sparse kernel, fixed 80% sparsity (LightX2V default);
                                         training-free, no tuned sparge checkpoint
LightX2V applies the sparse backends at every step and layer. The SVG2 paper keeps the first 30% of steps dense, so
SVG2 here is "LightX2V's SVG2", not the authors' exact recipe; the probe decides whether that matters for quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Wan2.1's default negative prompt (wan/configs/shared_config.py sample_neg_prompt), as used by the platform driver.
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
       "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
BASE = {"infer_steps": 50, "target_video_length": 81, "text_len": 512, "target_height": 480, "target_width": 832,
        "self_attn_1_type": "flash_attn2", "cross_attn_1_type": "flash_attn2", "cross_attn_2_type": "flash_attn2",
        "sample_guide_scale": 6, "sample_shift": 8, "enable_cfg": True, "cpu_offload": False}
ARMS = {
    "FULL50_LX": {},
    "SAGE2": {"self_attn_1_type": "sage_attn2"},
    "SVG2": {"self_attn_1_type": "svg2_attn"},
    "SPARGE": {"self_attn_1_type": "spas_sage_attn2",
               "spas_sage_attn2_setting": {"sparsity_ratio": 0.8, "sparse_mode": "sparge_mode"}},
}
PROBE_PROMPT = ("A red sports car drifts around a sharp corner on a wet mountain road at dusk, spraying water, "
                "while the camera tracks alongside it.")


# Which kernel libraries the arm interpreter actually loads (version and file), for provenance.
LIBS = """import importlib, json
out = {}
for name in ("torch", "flash_attn", "sageattention", "spas_sage_attn", "flashinfer"):
    try:
        m = importlib.import_module(name); out[name] = [getattr(m, "__version__", "?"), getattr(m, "__file__", "?")]
    except Exception as e:
        out[name] = ["missing", type(e).__name__ + ": " + str(e)[:120]]
print(json.dumps(out))"""


def arm_config(arm):
    return {**BASE, **ARMS[arm]}


def configs(args):
    out = Path(args.out).resolve()  # the child runs with cwd=LightX2V, so every path it gets must be absolute
    out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        (out / f"{arm}.json").write_text(json.dumps(arm_config(arm), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(args.arms)} configs to {out}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def launch(arm, tag, gpu, args, out):
    cfg = out / "configs" / f"{arm}.json"
    video = out / "videos" / f"{tag}.mp4"
    log = out / "logs" / f"{tag}.log"
    cmd = [args.python, "-m", "lightx2v.infer", "--model_cls", "wan2.1", "--task", "t2v", "--model_path", args.model,
           "--config_json", str(cfg), "--prompt", args.prompt, "--negative_prompt", NEG, "--seed", str(args.seed),
           "--save_result_path", str(video)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": os.pathsep.join([args.lightx2v, os.environ.get("PYTHONPATH", "")])}
    handle = open(log, "w", encoding="utf-8")
    return {"arm": arm, "tag": tag, "gpu": gpu, "video": video, "log": log, "t0": time.time(), "handle": handle,
            "proc": subprocess.Popen(cmd, cwd=args.lightx2v, env=env, stdout=handle, stderr=subprocess.STDOUT)}


def summarize(job):
    text = job["log"].read_text(encoding="utf-8", errors="replace").splitlines()
    keep = [line.strip()[-160:] for line in text
            if any(k in line.lower() for k in ("sparsity", "cost", "time", "error", "not found", "traceback"))]
    ok = job["proc"].returncode == 0 and job["video"].exists()
    return {"arm": job["arm"], "tag": job["tag"], "gpu": job["gpu"], "returncode": job["proc"].returncode, "ok": ok,
            "wall_seconds": round(job["seconds"], 1), "video_sha256": sha256(job["video"]) if ok else None,
            "log_lines": keep[-25:], "log_tail": [line[-200:] for line in text[-8:]]}


def probe(args):
    out = Path(args.out).resolve()  # the child runs with cwd=LightX2V, so every path it gets must be absolute
    for d in ("configs", "videos", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        (out / "configs" / f"{arm}.json").write_text(json.dumps(arm_config(arm), indent=2) + "\n", encoding="utf-8")
    runs = [(arm, arm) for arm in args.arms] + [("FULL50_LX", "FULL50_LX_repeat")]
    gpus = [int(g) for g in args.gpus.split(",")]
    pending, running, results = list(runs), [], []
    while pending or running:
        free = [g for g in gpus if g not in {j["gpu"] for j in running}]
        while pending and free:
            arm, tag = pending.pop(0)
            running.append(launch(arm, tag, free.pop(0), args, out))
            print(f"[probe] started {tag}", flush=True)
        time.sleep(5)
        for job in [j for j in running if j["proc"].poll() is not None]:
            job["seconds"] = time.time() - job["t0"]
            job["handle"].close()
            running.remove(job)
            results.append(summarize(job))
            r = results[-1]
            print(f"[probe] {r['tag']}: {'OK' if r['ok'] else 'FAILED rc=' + str(r['returncode'])} {r['wall_seconds']}s", flush=True)
    full = {r["tag"]: r for r in results if r["arm"] == "FULL50_LX"}
    deterministic = (len(full) == 2 and all(r["ok"] for r in full.values())
                     and full["FULL50_LX"]["video_sha256"] == full["FULL50_LX_repeat"]["video_sha256"])
    libs = subprocess.run([args.python, "-c", LIBS], capture_output=True, text=True).stdout
    report = {"model": args.model, "lightx2v": args.lightx2v, "prompt": args.prompt, "seed": args.seed,
              "python": args.python, "libraries": json.loads(libs) if libs.strip() else None,
              "full_deterministic": deterministic, "runs": sorted(results, key=lambda r: r["tag"])}
    (out / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base = full.get("FULL50_LX", {}).get("wall_seconds")
    print("\narm               ok    wall s   vs FULL50_LX (incl. model load)")
    for r in report["runs"]:
        ratio = f"{base / r['wall_seconds']:.2f}x" if base and r["ok"] else "-"
        print(f"{r['tag']:17} {str(r['ok']):5} {r['wall_seconds']:8} {ratio}")
    print(f"FULL50_LX deterministic across two GPUs: {deterministic}")
    print(f"details: {out / 'probe.json'}; logs in {out / 'logs'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("configs")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p = sub.add_parser("probe")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--model", default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--lightx2v", default="/mnt/afs_2/houze/LightX2V")
    p.add_argument("--python", default=sys.executable, help="interpreter with the attention kernels (ATTN_PYTHON from install_attention_env.sh)")
    p.add_argument("--prompt", default=PROBE_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args(argv)
    {"configs": configs, "probe": probe}[args.mode](args)


if __name__ == "__main__":
    main()
