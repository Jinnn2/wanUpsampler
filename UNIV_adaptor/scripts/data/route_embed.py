"""Router from prompt vectors: predict each method's scores per prompt, then route without hand-made categories.

Every input representation goes through the same mechanism, so they compare only on what the input knows:

  1. On the training folds, fit a predictor from prompt features to each arm's (time fraction, retention,
     quality_kept) per prompt.
  2. On the held-out fold, predict those for every prompt and solve one linear program over per-prompt arm
     distributions: minimize mean predicted time subject to mean predicted retention >= tau and mean predicted
     quality_kept >= tau (FULL is an arm with 1, 1, 1). Only predictions enter; labels only score the result.

Feature sets (all fit inside each fold):
- blind:  intercept only -> the best prompt-blind mixture
- cell:   one-hot designed motion x detail cell (the hand-made categories; upper bound for categories)
- bow:    cells predicted by in-fold bag-of-words (route_v1)
- t5:     Wan2.1's own umT5 encoder, attention-masked mean pooled (what the generator conditions on)
- qwen / qwenmid: Qwen3-VL-32B last / middle hidden layer, mean pooled over the prompt tokens
Vector features use ridge regression (alpha by in-fold generalized cross-validation) and k-nearest neighbours.

Usage (see run_univ_route.sh):
  python route_embed.py embed-t5   --out ROUTE_ROOT          # GPU, Wan checkpoint + LightX2V
  python route_embed.py embed-qwen --out ROUTE_ROOT --model M  # GPU
  python route_embed.py evaluate   --out ROUTE_ROOT
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402
from UNIV_adaptor.scripts.data import route_v1 as rt  # noqa: E402

KNN = 10


def prompt_texts(paths):
    from UNIV_adaptor.scripts.data import requirements_v2 as rv2

    return {pid: p["prompt"] for pid, p in rv2.load_prompts(paths).items() if p["split"] != "test"}


def save_vectors(path, pids, x, meta):
    np.savez(path, prompt_ids=np.asarray(pids), x=np.asarray(x, np.float32))
    write_json_atomic(Path(path).with_suffix(".json"), meta | {"n": len(pids), "dim": int(np.asarray(x).shape[1]),
                                                               "npz_sha256": sha256_file(path)})


# ---------------------------------------------------------------- embeddings

def embed_t5(args):
    from changing_resolution_uni.scripts.data.extract_prompt_t5_embeddings import directory_file_inventory

    texts = prompt_texts(args.prompts)
    pids = sorted(texts)
    work = Path(args.out) / "t5_cache"
    work.mkdir(parents=True, exist_ok=True)
    (work / "prompts.json").write_text(json.dumps([texts[p] for p in pids], ensure_ascii=False), encoding="utf-8")
    model = Path(args.model_root)
    subprocess.run([sys.executable, str(ROOT / "changing_resolution_uni/scripts/data/extract_prompt_t5_embeddings.py"),
                    "--prompts_file", str(work / "prompts.json"), "--out_dir", str(work), "--model_path", str(model),
                    "--required_backend", "wan_native", "--precision", "bf16", "--device", args.device],
                   env=os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [
                       str(Path(args.lightx2v_repo).resolve()), str(ROOT), os.environ.get("PYTHONPATH")]))}, check=True)
    x = []
    for i, pid in enumerate(pids):
        meta = json.loads((work / f"prompt_{i:06d}.json").read_text(encoding="utf-8"))
        if meta["prompt_text"] != texts[pid].strip():
            raise ValueError(f"T5 cache prompt mismatch at {i}")
        with np.load(work / f"prompt_{i:06d}.npz", allow_pickle=False) as z:
            x.append(z["pooled_embedding"].astype(np.float32).reshape(-1))
    save_vectors(Path(args.out) / "emb_t5.npz", pids, x, {
        "encoder": "wan2.1 umT5-xxl (masked mean pool)",
        "checkpoint_sha256": sha256_file(model / "models_t5_umt5-xxl-enc-bf16.pth"),
        "tokenizer_files": directory_file_inventory(model / "google/umt5-xxl")})
    print(f"wrote emb_t5.npz: {len(pids)} x {len(x[0])}")


def embed_qwen(args):
    import torch
    import transformers

    texts = prompt_texts(args.prompts)
    pids = sorted(texts)
    tok = transformers.AutoProcessor.from_pretrained(args.model).tokenizer
    model = transformers.AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                                     device_map=args.device).eval()
    last, mid = [], []
    with torch.inference_mode():
        for pid in pids:
            ids = tok(texts[pid], return_tensors="pt").to(model.device)
            hidden = model(**ids, output_hidden_states=True).hidden_states
            last.append(hidden[-1][0].float().mean(0).cpu().numpy())
            mid.append(hidden[len(hidden) // 2][0].float().mean(0).cpu().numpy())
    for name, x, layer in (("qwen", last, "last"), ("qwenmid", mid, "middle")):
        save_vectors(Path(args.out) / f"emb_{name}.npz", pids, x, {"encoder": f"{args.model} {layer} hidden layer (mean pool)"})
        print(f"wrote emb_{name}.npz: {len(pids)} x {len(x[0])}")


def load_vectors(out, name, pids):
    path = Path(out) / f"emb_{name}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        index = {int(p): i for i, p in enumerate(z["prompt_ids"])}
        missing = [p for p in pids if p not in index]
        if missing:
            raise ValueError(f"{path.name} lacks prompts {missing[:5]}")
        return z["x"][[index[p] for p in pids]].astype(np.float64)


# ---------------------------------------------------------------- predictors

def targets(data):
    """(P, 3, A) per-prompt time fraction, retention, quality_kept."""
    return np.stack([data.tf, data.ret, data.q], 1)


def ridge_gcv(xtr, ytr, xte, alphas=np.logspace(-3, 3, 25)):
    """Ridge on centered features with one alpha per target column chosen by generalized cross-validation."""
    mx, my = xtr.mean(0), ytr.mean(0)
    u, s, vt = np.linalg.svd(xtr - mx, full_matrices=False)
    uty = u.T @ (ytr - my)
    n = len(xtr)
    best_err, best_coef = np.full(ytr.shape[1], np.inf), np.zeros((xtr.shape[1], ytr.shape[1]))
    for a in alphas * (s[0] ** 2) / n:
        f = s ** 2 / (s ** 2 + a)
        fit = u @ (f[:, None] * uty)
        gcv = ((ytr - my - fit) ** 2).mean(0) / (1 - f.sum() / n) ** 2
        better = gcv < best_err
        coef = vt.T @ ((s / (s ** 2 + a))[:, None] * uty)
        best_err[better], best_coef[:, better] = gcv[better], coef[:, better]
    return (xte - mx) @ best_coef + my


def knn(xtr, ytr, xte, k=KNN):
    def unit(m):
        m = m - xtr.mean(0)
        return m / np.linalg.norm(m, axis=1, keepdims=True)
    sim = unit(xte) @ unit(xtr).T
    nn = np.argsort(-sim, 1)[:, :k]
    return ytr[nn].mean(1)


def one_hot(cells):
    return np.eye(len(rt.CELLS))[cells]


def predictor(name, data, texts, vectors):
    """Returns f(train idx, test idx) -> predicted (len(test), 3, A)."""
    y = targets(data)
    flat = y.reshape(len(y), -1)

    def wrap(fn):
        return lambda tr, te: fn(tr, te).reshape(len(te), *y.shape[1:])

    if name == "blind":
        return wrap(lambda tr, te: np.tile(flat[tr].mean(0), (len(te), 1)))
    if name == "cell":
        return wrap(lambda tr, te: one_hot(data.cell[te]) @ np.stack([flat[tr][data.cell[tr] == c].mean(0)
                                                                      for c in range(len(rt.CELLS))]))
    if name == "bow":
        labels = [(rt.CELLS[c][0] == "h", rt.CELLS[c][1] == "h") for c in data.cell]

        def bow(tr, te):
            ctr, cte = rt.bow_cells(texts, labels, tr, tr), rt.bow_cells(texts, labels, tr, te)
            means = np.stack([flat[tr][ctr == c].mean(0) if (ctr == c).any() else flat[tr].mean(0)
                              for c in range(len(rt.CELLS))])
            return one_hot(cte) @ means
        return wrap(bow)
    feat, model = name.rsplit("_", 1)
    x = vectors[feat]
    if model == "ridge":
        return wrap(lambda tr, te: ridge_gcv(x[tr], flat[tr], x[te]))
    if model == "knn":
        return wrap(lambda tr, te: knn(x[tr], flat[tr], x[te]))
    raise ValueError(name)


# ---------------------------------------------------------------- policy

def route(pred, tau):
    """Per-prompt arm distributions (P, A) minimizing mean predicted time s.t. mean predicted ret, q >= tau."""
    from scipy.optimize import linprog

    p, _, a = pred.shape
    tf, ret, q = (np.clip(pred[:, k], 1e-3, None) for k in range(3))
    a_eq = np.kron(np.eye(p), np.ones((1, a)))
    res = linprog(tf.ravel() / p, A_ub=-np.stack([ret.ravel(), q.ravel()]) / p, b_ub=[-tau, -tau],
                  A_eq=a_eq, b_eq=np.ones(p), bounds=(0, None), method="highs")
    if res.status != 0:  # predictions too pessimistic to meet the floor: everything goes to FULL
        x = np.zeros((p, a))
        x[:, -1] = 1
        return x
    x = np.clip(res.x.reshape(p, a), 0, None)
    return x / x.sum(1, keepdims=True)


def evaluate(args):
    rng = np.random.default_rng(rt.SEED)
    out = Path(args.out)
    data = rt.Data(out / "route_table.csv")
    texts_by_pid = prompt_texts(args.prompts)
    texts = [texts_by_pid[p] for p in data.pids]
    vectors = {n: v for n in ("t5", "qwen", "qwenmid") if (v := load_vectors(out, n, data.pids)) is not None}
    names = ["blind", "cell", "bow"] + [f"{n}_{m}" for n in vectors for m in ("ridge", "knn")]
    if not vectors:
        print("NOTE: no emb_*.npz yet (run embed-t5 / embed-qwen); comparing blind, cell and bow only")
    y = targets(data)
    idx = np.arange(len(data.pids))
    splits = []
    for _ in range(args.repeats):
        f = rt.folds(list(data.family), args.folds, rng)
        splits.append([(idx[f != k], idx[f == k]) for k in range(args.folds)])
    # Out-of-fold predictions once per repeat and feature set (the policy is re-solved per tau).
    preds = {n: [] for n in names}
    for n in names:
        fn = predictor(n, data, texts, vectors)
        for rep in splits:
            p = np.zeros_like(y)
            for tr, te in rep:
                p[te] = fn(tr, te)
            preds[n].append(p)
        r2 = 1 - ((np.mean(preds[n], 0) - y)[:, 1:, :-1] ** 2).sum() / ((y[:, 1:, :-1] - y[:, 1:, :-1].mean(0)) ** 2).sum()
        print(f"[{n}] out-of-fold R^2 of retention/quality_kept over arms: {r2:.3f}", flush=True)
    # Realized out-of-fold (time, retention, quality_kept) per prompt for every internal target on the grid.
    grid = np.round(np.arange(args.grid_low, 1.0 + 1e-9, 0.02), 2)
    realized = {}
    for n in names:
        vals = np.zeros((len(grid), len(idx), 3))
        for g, tau in enumerate(grid):
            for rep, p in zip(splits, preds[n]):
                for _, te in rep:
                    x = route(p[te], tau)
                    vals[g, te] += np.stack([(x * y[te, k]).sum(1) for k in range(3)], 1)
        realized[n] = vals / len(splits)
        print(f"[{n}] routed on {len(grid)} internal targets", flush=True)
    fam = sorted(set(data.family))
    sums = {n: np.stack([v[:, data.family == f].sum(1) for f in fam]) for n, v in realized.items()}  # (F, G, 3)
    sizes = np.array([(data.family == f).sum() for f in fam])
    taus = np.array(args.taus)

    def curves(rows):
        return {n: s[rows].sum(0) / sizes[rows].sum() for n, s in sums.items()}  # (G, 3) mean tf, ret, q

    def stats(c):
        # matched: least time reaching realized min(ret, Q) >= tau, mixing two grid policies if needed (time and
        # both scores are linear in the mixing weight); nominal: the policy whose predicted floor is tau.
        out = {}
        for n, m in c.items():
            tf, qual = m[:, 0], m[:, 1:].min(1)
            t = taus[:, None, None]
            w = np.clip((t - qual[None, None, :]) / (qual[None, :, None] - qual[None, None, :] + 1e-12), 0, 1)
            mixed = w * tf[None, :, None] + (1 - w) * tf[None, None, :]
            ok = w * qual[None, :, None] + (1 - w) * qual[None, None, :] >= t - 1e-9
            matched = np.where(ok, mixed, np.inf).min((1, 2))
            nominal = np.array([m[np.argmin(abs(grid - tau))] for tau in taus])
            out[n] = np.stack([1 / matched, 1 / nominal[:, 0], nominal[:, 1:].min(1) - taus], 1)  # (T, 3)
        return out

    point = stats(curves(np.arange(len(fam))))
    boot = [stats(curves(rng.integers(0, len(fam), len(fam)))) for _ in range(args.boot)]
    records = []
    for n in names:
        for i, tau in enumerate(taus):
            rec = {"features": n, "tau": tau, "speed_matched": point[n][i, 0], "speed_nominal": point[n][i, 1],
                   "slack_nominal": point[n][i, 2]}
            for ref in ("blind", "cell"):
                g = np.array([b[n][i, 0] / b[ref][i, 0] - 1 for b in boot])
                rec |= {f"gain_vs_{ref}": point[n][i, 0] / point[ref][i, 0] - 1,
                        f"gain_vs_{ref}_lo": np.nanpercentile(g, 2.5), f"gain_vs_{ref}_hi": np.nanpercentile(g, 97.5)}
            records.append(rec)
    with (out / "route_embed_eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    head = "| features | " + " | ".join(f"tau {t:.2f}" for t in taus) + " |\n| --- |" + " --- |" * len(taus)
    lines = ["# Router from prompt vectors vs hand-made cells", "",
             f"{len(idx)} prompts, {args.folds}-fold by family x {args.repeats} repeats. Each feature set predicts "
             "per-prompt (time, retention, quality_kept) for every arm in-fold; one LP over the held-out fold's "
             "predictions picks per-prompt arm distributions (internal floor on a grid). Speed is compared at "
             "matched realized quality: the fastest policy (or two-policy mix) whose out-of-fold min(mean ret, "
             "mean Q) reaches tau, so a predictor that overpromises gets no credit. 95% CI: family bootstrap.", ""]
    for ref in ("blind", "cell"):
        lines += [f"## speed at realized quality tau, gain vs {ref}", "", head]
        for n in names:
            rs = [r for r in records if r["features"] == n]
            lines.append(f"| {n} | " + " | ".join(
                f"{r['speed_matched']:.2f}x {r[f'gain_vs_{ref}']:+.0%} [{r[f'gain_vs_{ref}_lo']:+.0%}, "
                f"{r[f'gain_vs_{ref}_hi']:+.0%}]" for r in rs) + " |")
        lines.append("")
    lines += ["## nominal: internal floor = tau (speed, realized min(ret, Q) - tau)", "", head]
    for n in names:
        rs = [r for r in records if r["features"] == n]
        lines.append(f"| {n} | " + " | ".join(f"{r['speed_nominal']:.2f}x s{r['slack_nominal']:+.3f}" for r in rs) + " |")
    text = "\n".join(lines) + "\n"
    (out / "route_embed_report.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("embed-t5", "embed-qwen", "evaluate"):
        p = sub.add_parser(mode)
        p.add_argument("--out", required=True)
        p.add_argument("--prompts", nargs="+", default=[str(x) for x in rt.DEFAULT_PROMPTS])
        if mode == "embed-t5":
            p.add_argument("--model-root", default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B")
            p.add_argument("--lightx2v-repo", default="/mnt/afs_2/houze/LightX2V")
            p.add_argument("--device", default="cuda")
        if mode == "embed-qwen":
            p.add_argument("--model", required=True)
            p.add_argument("--device", default="cuda")
        if mode == "evaluate":
            p.add_argument("--taus", nargs="+", type=float, default=[0.95, 0.90, 0.85, 0.80, 0.70, 0.60])
            p.add_argument("--folds", type=int, default=8)
            p.add_argument("--repeats", type=int, default=10)
            p.add_argument("--grid-low", type=float, default=0.5)
            p.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args(argv)
    {"embed-t5": embed_t5, "embed-qwen": embed_qwen, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
