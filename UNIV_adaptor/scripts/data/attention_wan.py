"""Attention-level arms on the platform's own official-Wan2.1 code path (no LightX2V).

Why: the three-layer metrics are paired against FULL, and two code paths give different FULL clips (like two seeds).
Here every arm runs through exactly the pipeline that produced the platform's FULL50 videos: the frozen worker's
load_entrypoint("wan21"), the FA2 lock, argv_for(), TF32 off, the same warm-up, the same seeds. The only change is the
video self-attention inside wan.modules.model; cross-attention, scheduler, text encoder and VAE are untouched. The frozen
v1 driver/worker files are imported, never edited.

Arms
  DENSE     no attention change. Gate: must reproduce the platform FULL50 videos (sha256) before any FULL50 is reused.
  SPARGE50/65/80  SpargeAttn, training-free plug-in spas_sage2_attn_meansim_topk_cuda, topk = 1 - sparsity, every
            step and layer (as the SpargeAttn plug-in is used); the sparsity sweep picks one operating point.
  SVG2      Sparse-VideoGen 2 (semantic-aware permutation), settings of the official Wan2.1 T2V SAP script
            (scripts/wan/wan_t2v_720p_sap.sh): first 20% of steps dense, first floor(0.03 * layers) layers dense
            (0 of 30 for 1.3B), 300 q / 1000 k centroids, top-p 0.9, min_kc_ratio 0.1, k-means 50 init / 2 per step
            iterations. As in the official processor, each layer keeps one k-means state shared by the cond and uncond
            passes, initialised at its first sparse call; here it is also reset per video (the official script runs one
            video per process). Kernels: svg2_kernels.py (vendored); block-sparse attention via flashinfer.

Modes
  run     python attention_wan.py run --config CFG --out OUT --arms DENSE SPARGE65 --gpus 0,1,2,3 [--prompt-ids ...]
          [--seeds 42] [--first N]   one worker process per GPU, jobs (prompt, seed, arm) dealt round-robin;
          resumable: a job with a receipt is skipped.
  gate    python attention_wan.py gate --out OUT --reference PLATFORM_ROOT   DENSE vs platform FULL50 receipts
  report  python attention_wan.py report --out OUT [--reference PLATFORM_ROOT]   time and PSNR per arm vs DENSE
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"
SVG2_OFFICIAL = {"num_q_centroids": 300, "num_k_centroids": 1000, "top_p_kmeans": 0.9, "min_kc_ratio": 0.10,
                 "kmeans_iter_init": 50, "kmeans_iter_step": 2, "first_times_fp": 0.2, "first_layers_fp": 0.03}
ARMS = {
    "DENSE": {"kind": "dense"},
    "SPARGE50": {"kind": "sparge", "sparsity": 0.50},
    "SPARGE65": {"kind": "sparge", "sparsity": 0.65},
    "SPARGE80": {"kind": "sparge", "sparsity": 0.80},
    "SVG2": {"kind": "svg2", **SVG2_OFFICIAL},
}
STEPS = 50


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def job_id(prompt_id, seed, arm):
    return f"pilot_{prompt_id}_s{seed}_{arm}"  # the platform's naming, so later scoring can treat both alike


# ----------------------------------------------------------------------------------------------- attention patch
class Svg2Layer:
    """Per-layer k-means state of the official SVG2 processor (shared by cond and uncond)."""

    def __init__(self, cfg):
        self.cfg, self.q_centroids, self.k_centroids = cfg, None, None

    def __call__(self, q, k, v, kern, flashinfer, buffers):
        # q, k, v: [1, H, S, D]
        c = self.cfg
        _, H, S, D = q.shape
        init = self.q_centroids is None
        qx, kx = q.reshape(H, S, D), k.reshape(H, S, D)
        ql, qc, qs, _ = kern.batch_kmeans_Euclid(qx, n_clusters=c["num_q_centroids"],
                                                 max_iters=c["kmeans_iter_init" if init else "kmeans_iter_step"],
                                                 init_centroids=None if init else self.q_centroids)
        kl, kc, ks, _ = kern.batch_kmeans_Euclid(kx, n_clusters=c["num_k_centroids"],
                                                 max_iters=c["kmeans_iter_init" if init else "kmeans_iter_step"],
                                                 init_centroids=None if init else self.k_centroids)
        self.q_centroids, self.k_centroids = qc, kc
        qs, ks = qs.view(1, H, -1), ks.view(1, H, -1)
        dyn = kern.identify_dynamic_map(qc.view(1, H, -1, D), kc.view(1, H, -1, D), qs, ks, c["top_p_kmeans"], c["min_kc_ratio"])
        qp, q_idx = kern.permute_tensor_by_labels_triton(q, ql, dim=2)
        kp, k_idx = kern.permute_tensor_by_labels_triton(k, kl, dim=2)
        vp, _ = kern.permute_tensor_by_labels_triton(v, kl, dim=2, sorted_indices=k_idx)
        if "wrapper" not in buffers:  # the same buffers as LightX2V's launcher, allocated once instead of per call
            torch = torch_module()
            w = flashinfer.sparse.VariableBlockSparseAttentionWrapper(torch.empty(128 * 1024 * 1024, device=q.device), backend="auto")
            w.reset_workspace_buffer(float_workspace_buffer=w._float_workspace_buffer, int_workspace_buffer=w._int_workspace_buffer,
                                     vector_sparse_indices_buffer=torch.empty(1024 * 1024 * 1024, device=q.device),
                                     vector_sparse_indptr_buffer=w._vector_sparse_indptr_buffer)
            buffers["wrapper"] = w
        w = buffers["wrapper"]
        w.plan(block_mask_map=dyn.reshape(H, dyn.shape[-2], dyn.shape[-1]), block_row_sz=qs.reshape(H, -1),
               block_col_sz=ks.reshape(H, -1), num_qo_heads=H, num_kv_heads=H, head_dim=D,
               q_data_type=qp.dtype, kv_data_type=kp.dtype)
        o = w.run(qp.reshape(H, S, D), kp.reshape(H, S, D), vp.reshape(H, S, D)).reshape(1, H, S, D)
        density = float(kern.density_calculation(dyn, qs, ks).mean()) if hasattr(kern, "density_calculation") else float("nan")
        return kern.apply_inverse_permutation_triton(o, q_idx, dim=2), density


def torch_module():
    import torch
    return torch


class SelfAttentionPatch:
    """Replaces wan.modules.model.flash_attention; only video self-attention (Lq == Lk, no window) is changed."""

    def __init__(self, model_module, num_layers):
        self.dense, self.L = model_module.flash_attention, num_layers
        self.arm, self.calls, self.stats, self.layers = None, 0, [], {}
        self.kern = self.flashinfer = self.sparge = None
        self.buffers = {}
        model_module.flash_attention = self

    def start(self, arm_id):
        self.arm_id, self.arm = arm_id, ARMS[arm_id]
        self.calls, self.stats, self.layers, self.checked = 0, [], {}, False
        kind = self.arm["kind"]
        if kind == "sparge" and self.sparge is None:
            import spas_sage_attn
            fn = getattr(spas_sage_attn, "spas_sage2_attn_meansim_topk_cuda", None) or spas_sage_attn.core.spas_sage2_attn_meansim_topk_cuda
            self.sparge = (fn, set(inspect.signature(fn).parameters))
        if kind == "svg2" and self.kern is None:
            import flashinfer
            sys.path.insert(0, str(Path(__file__).parent))
            import svg2_kernels
            self.kern, self.flashinfer = svg2_kernels, flashinfer

    def finish(self):
        expected = STEPS * 2 * self.L
        if self.calls != expected:
            raise RuntimeError(f"{self.arm_id}: {self.calls} self-attention calls, expected {expected} (steps x cfg x layers)")
        sparse = [s for s in self.stats if s is not None]
        return {"self_attention_calls": self.calls, "sparse_calls": len(sparse),
                "mean_kept_fraction": (sum(sparse) / len(sparse)) if sparse else None}

    def __call__(self, *args, **kw):
        if len(args) > 3:
            raise RuntimeError("Unexpected positional flash_attention arguments; refusing to guess their meaning")
        kw.update(zip(("q", "k", "v"), args))
        q, k, v = kw["q"], kw["k"], kw["v"]
        is_self = q.shape[1] == k.shape[1] and tuple(kw.get("window_size", (-1, -1))) == (-1, -1) and q.shape[1] >= 1024
        if not is_self or self.arm is None:
            return self.dense(**kw)
        index, self.calls = self.calls, self.calls + 1
        layer, step = index % self.L, index // (2 * self.L)
        kind = self.arm["kind"]
        dense = kind == "dense" or (kind == "svg2" and (step < math.floor(self.arm["first_times_fp"] * STEPS)
                                                        or layer < math.floor(self.arm["first_layers_fp"] * self.L)))
        if dense:
            self.stats.append(None)
            return self.dense(**kw)
        k_lens = kw.get("k_lens")
        if not self.checked:
            if q.shape[0] != 1 or (k_lens is not None and int(k_lens.max()) != k.shape[1]):
                raise RuntimeError("Sparse self-attention needs batch 1 and no key padding")
            self.checked = True
        torch = torch_module()
        out_dtype = q.dtype
        qh, kh, vh = (t.to(torch.bfloat16).transpose(1, 2).contiguous() for t in (q, k, v))  # [1, H, S, D]
        if kind == "sparge":
            fn, params = self.sparge
            extra = {key: value for key, value in (("tensor_layout", "HND"), ("is_causal", False), ("return_sparsity", True))
                     if key in params}
            res = fn(qh, kh, vh, topk=1 - self.arm["sparsity"], **extra)
            o, sparsity = res if isinstance(res, tuple) else (res, None)
            self.stats.append(1 - float(sparsity) if sparsity is not None else 1 - self.arm["sparsity"])
        else:
            if layer not in self.layers:
                self.layers[layer] = Svg2Layer(self.arm)
            o, density = self.layers[layer](qh, kh, vh, self.kern, self.flashinfer, self.buffers)
            self.stats.append(density)
        return o.transpose(1, 2).to(out_dtype).contiguous()


# ----------------------------------------------------------------------------------------------- worker
def worker(args):
    jobs = read(args.jobs)
    sys.path.insert(0, str(ROOT))
    from UNIV_adaptor.scripts.data import published_wan21_worker as frozen
    cfg = read(args.config)
    plan_like = {"config": cfg, "model_root": args.model_root}
    module = frozen.load_entrypoint("wan21")  # chdir + sys.path to the pinned official Wan checkout
    sys.argv = frozen.argv_for(plan_like, {"arm": {"steps": STEPS, "flags": []}, "prompt": "x", "seed": 0}, Path("/dev/null"))
    module._parse_args()
    import random
    import numpy as np
    import torch
    attention, backend = frozen.configure_attention_backend()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    wan_commit = subprocess.run(["git", "-C", str(Path(sys.modules["wan"].__file__).parents[1]), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    env = frozen.versions() | {"gpu_name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                               "wan_module": str(sys.modules["wan"].__file__), "wan_commit": wan_commit}
    state = {"pipeline": None, "load": 0.0, "save": 0.0}
    original_constructor, original_save = module.wan.WanT2V, module.cache_video

    def cached_constructor(*pos, **kw):
        if state["pipeline"] is None:
            start = time.perf_counter()
            state["pipeline"] = original_constructor(*pos, **kw)
            torch.cuda.synchronize()
            state["load"] = time.perf_counter() - start
        return state["pipeline"]

    def timed_save(*pos, **kw):
        torch.cuda.synchronize()
        if state.get("warmup"):
            return None
        start = time.perf_counter()
        original_save(*pos, **kw)
        state["save"] += time.perf_counter() - start

    module.wan.WanT2V, module.cache_video = cached_constructor, timed_save
    patch = None

    def execute(job, warmup=False):
        nonlocal patch
        out = Path(args.out)
        video = out / "videos" / job["arm"] / (job["id"] + ".mp4")
        video.parent.mkdir(parents=True, exist_ok=True)
        state.update(warmup=warmup, save=0.0)
        load_before = state["load"]  # nonzero only after the first (warm-up) generate loaded the model
        if patch is not None:
            patch.start(job["arm"])
        sys.argv = frozen.argv_for(plan_like, {"arm": {"steps": STEPS, "flags": []}, "prompt": job["prompt"], "seed": job["seed"]}, video)
        options = module._parse_args()
        module.args = options
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        module.generate(options)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        loaded = state["load"] - load_before
        if patch is None:  # the model exists only after the first generate (the warm-up, always dense)
            import wan.modules.model as model_module
            patch = SelfAttentionPatch(model_module, len(state["pipeline"].model.blocks))
            return None
        attention_stats = patch.finish()
        if warmup:
            return None
        receipt = {"schema": "attention_wan_record_v1", "job": job, "arm_spec": ARMS[job["arm"]],
                   "video_path": str(video.resolve()), "video_sha256": sha256(video),
                   "runtime": {"pipeline_seconds": elapsed - loaded - state["save"], "encoding_seconds": state["save"],
                               "model_load_seconds": loaded, "outer_seconds": elapsed,
                               "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                               "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                               "scope": "as the platform worker: prompt encoding + denoising + VAE; excludes model load and MP4 encoding"},
                   "attention": attention_stats, "environment": env, "attention_backend": backend}
        path = out / "records" / (job["id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"[gpu {args.gpu}] {job['id']}: {receipt['runtime']['pipeline_seconds']:.1f}s "
              f"kept={attention_stats['mean_kept_fraction']}", flush=True)

    cal = cfg["calibration"]
    warm = {"id": "warmup", "arm": "DENSE", "prompt": cal["prompts"][0]["prompt"], "seed": cal["seed"]}
    execute(warm, warmup=True)          # loads the model, installs the patch (excluded, as in the platform worker)
    execute(warm, warmup=True)          # the same full-shape warm-up through the patched path
    for arm in sorted({j["arm"] for j in jobs} - {"DENSE"}):
        execute(dict(warm, arm=arm), warmup=True)  # kernel JIT / autotune per arm, excluded from timing
    for job in jobs:
        execute(job)


# ----------------------------------------------------------------------------------------------- driver
def run(args):
    cfg = read(args.config)
    prompts = [p for p in cfg["prompts"] if not args.prompt_ids or p["id"] in args.prompt_ids]
    if args.first:
        prompts = prompts[:args.first]
    seeds = args.seeds or cfg["seeds"]
    out = Path(args.out).resolve()
    jobs = [{"id": job_id(p["id"], s, a), "prompt_id": p["id"], "prompt": p["prompt"], "seed": s, "arm": a}
            for p in prompts for s in seeds for a in args.arms]
    todo = [j for j in jobs if not (out / "records" / (j["id"] + ".json")).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run, arms {args.arms}, {len(prompts)} prompts x {len(seeds)} seeds")
    kinds = {ARMS[a]["kind"] for a in args.arms}
    need = ["flash_attn"] + (["spas_sage_attn"] if "sparge" in kinds else []) + (["flashinfer", "triton"] if "svg2" in kinds else [])
    check = subprocess.run([args.python, "-c", f"import {', '.join(need)}"], capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit(f"{args.python} cannot import {need}:\n{check.stderr[-800:]}\nPass --python (or PYTHON_WAN21) with these libraries.")
    print(f"python {args.python}: imports {need} OK")
    gpus = args.gpus.split(",")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for i, gpu in enumerate(gpus):
        mine = todo[i::len(gpus)]
        if not mine:
            continue
        jf = out / "logs" / f"jobs_gpu{gpu}.json"
        jf.write_text(json.dumps(mine, ensure_ascii=False, indent=1), encoding="utf-8")
        cmd = [args.python, str(Path(__file__).resolve()), "worker", "--jobs", str(jf), "--out", str(out), "--gpu", gpu,
               "--config", str(Path(args.config).resolve()), "--model-root", args.model_root]
        log = open(out / "logs" / f"worker_gpu{gpu}.log", "a", encoding="utf-8")
        procs.append((gpu, subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONDONTWRITEBYTECODE": "1"},
                                            stdout=log, stderr=subprocess.STDOUT)))
    # Watchdog: tqdm writes every denoising step (~1 s), so a log that stops changing for HANG_SECONDS is a hung kernel
    # (seen with SpargeAttn at 65% sparsity). Kill that worker; its unfinished jobs have no receipt and rerun next time.
    hung = []
    while any(p.poll() is None for _, p in procs):
        time.sleep(30)
        for gpu, p in procs:
            log = out / "logs" / f"worker_gpu{gpu}.log"
            if p.poll() is None and time.time() - log.stat().st_mtime > args.hang_seconds:
                p.kill()
                hung.append(gpu)
                print(f"GPU {gpu}: no progress for {args.hang_seconds}s, killed (hung kernel); rerun to retry its jobs", flush=True)
    bad = [gpu for gpu, p in procs if p.returncode != 0]
    print("all workers finished" if not bad else f"workers failed on GPUs {bad} (hung: {hung}); see {out / 'logs'}")
    sys.exit(1 if bad else 0)


def records(out):
    return {r["job"]["id"]: r for r in (read(p) for p in sorted(Path(out).glob("records/*.json")))}


def gate(args):
    rows, ok = [], True
    for jid, r in sorted(records(args.out).items()):
        if r["job"]["arm"] != "DENSE":
            continue
        ref_path = Path(args.reference) / "records" / (job_id(r["job"]["prompt_id"], r["job"]["seed"], "FULL50") + ".json")
        if not ref_path.exists():
            rows.append((jid, "no platform FULL50 receipt", None)); ok = False
            continue
        ref = read(ref_path)
        same = ref["video_sha256"] == r["video_sha256"]
        psnr = None if same else video_psnr(r["video_path"], ref["video_path"])
        ok &= same
        rows.append((jid, "IDENTICAL" if same else "DIFFERENT", psnr, r["runtime"]["pipeline_seconds"], ref["runtime"]["pipeline_seconds"]))
    for row in rows:
        print(*row)
    if not rows:
        print("GATE NOT RUN: no DENSE records yet (see the worker logs)")
        return
    new = next(r for r in records(args.out).values() if r["job"]["arm"] == "DENSE")["environment"]
    old = read(Path(args.reference) / "records" / (job_id(rows[0][0].split("_")[1], 42, "FULL50") + ".json")).get("environment", {})
    for key in sorted(set(new) | set(old)):
        if key in ("wan_module", "wan_commit") or new.get(key) != old.get(key):
            print(f"  env {key}: platform={old.get(key)}  now={new.get(key)}")
    print("GATE PASSED: platform FULL50 can be reused" if ok and rows else
          "GATE FAILED: DENSE does not reproduce the platform FULL50; attention arms need their own DENSE reference")


def video_frames(path):
    import imageio
    import numpy as np
    return np.asarray(imageio.mimread(path, memtest=False), dtype=np.float32)


def video_psnr(a, b):
    import numpy as np
    x, y = video_frames(a), video_frames(b)
    n = min(len(x), len(y))
    mse = float(((x[:n] - y[:n]) ** 2).mean())
    return round(10 * math.log10(255 ** 2 / mse), 2) if mse > 0 else float("inf")


def report(args):
    recs = records(args.out)
    by = {}
    for r in recs.values():
        by.setdefault(r["job"]["arm"], {})[(r["job"]["prompt_id"], r["job"]["seed"])] = r
    dense = by.get("DENSE", {})
    lines = ["arm        n   pipeline s (mean)  speedup vs DENSE  kept fraction  PSNR vs DENSE (mean)"]
    for arm in sorted(by, key=lambda a: list(ARMS).index(a)):
        rs = by[arm]
        t = sum(r["runtime"]["pipeline_seconds"] for r in rs.values()) / len(rs)
        pairs = [k for k in rs if k in dense]
        sp = (sum(dense[k]["runtime"]["pipeline_seconds"] for k in pairs) / sum(rs[k]["runtime"]["pipeline_seconds"] for k in pairs)) if pairs else float("nan")
        kept = [r["attention"]["mean_kept_fraction"] for r in rs.values() if r["attention"]["mean_kept_fraction"] is not None]
        ps = [video_psnr(rs[k]["video_path"], dense[k]["video_path"]) for k in pairs] if arm != "DENSE" else []
        lines.append(f"{arm:10} {len(rs):3} {t:10.1f}          {sp:6.2f}x            "
                     f"{(sum(kept) / len(kept)) if kept else float('nan'):.3f}          {(sum(ps) / len(ps)) if ps else float('nan'):.2f}")
    text = "\n".join(lines)
    print(text)
    Path(args.out, "report.txt").write_text(text + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--arms", nargs="+", required=True, choices=list(ARMS))
    r.add_argument("--gpus", default="0,1,2,3")
    r.add_argument("--prompt-ids", nargs="*")
    r.add_argument("--seeds", nargs="*", type=int)
    r.add_argument("--first", type=int, help="only the first N prompts of the config (after --prompt-ids)")
    r.add_argument("--model-root", default=DEFAULT_MODEL)
    r.add_argument("--hang-seconds", type=int, default=600)
    r.add_argument("--python", default=os.environ.get("PYTHON_WAN21", sys.executable))
    w = sub.add_parser("worker")
    for name in ("--jobs", "--out", "--gpu", "--config", "--model-root"):
        w.add_argument(name, required=True)
    g = sub.add_parser("gate")
    g.add_argument("--out", required=True)
    g.add_argument("--reference", required=True)
    o = sub.add_parser("report")
    o.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"run": run, "worker": worker, "gate": gate, "report": report}[args.mode](args)


if __name__ == "__main__":
    main()
