"""Offline prompt -> quality gain -> budgeted action experiment for Phase 2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic
from UNIV_adaptor.scripts.data.phase2_analysis import (
    DIMENSIONS, NATIVE, aggregate_prompts, bootstrap_ci, csv_write, enrich,
)
from UNIV_adaptor.scripts.data.score_phase2_dataset import SCORE_SCHEMA, output_lock
from UNIV_adaptor.scripts.router.phase2_gain_model import (
    cross_validate, normalize, predict, select_actions, text_features, train_mixture,
)

CACHE, SKIP, SPATIAL = "P2_B30_CACHE", "P2_B25_SKIP", "P2_B30_SPATIAL"
ACTION_SETS = {"pair": [CACHE, SKIP], "three": [CACHE, SKIP, SPATIAL],
               "five": [CACHE, SKIP, SPATIAL, "P2_B25_SPATIAL", "P2_B25_TEMPORAL"]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_data(quality_dir, actions):
    inputs, scores = read(quality_dir/"evaluation_inputs.json"), read(quality_dir/"scores.json")
    identity = {k: inputs[k] for k in ("generation_manifest_sha256", "plan_sha256")}
    digest = canonical_sha256({"identity": identity, "rows": inputs["rows"]})
    if digest != inputs["input_sha256"] or digest != scores["input_sha256"]:
        raise ValueError("evaluation inputs/scores identity mismatch")
    if scores.get("schema") != SCORE_SCHEMA or scores.get("dimensions") != list(DIMENSIONS):
        raise ValueError("unsupported score schema/dimensions")
    if scores.get("payload_sha256") != canonical_sha256({k:v for k,v in scores.items() if k!="payload_sha256"}):
        raise ValueError("score payload hash mismatch")
    rows = enrich(inputs["rows"], scores["scores"])
    for row in rows:
        if row["prompt_sha256"] != canonical_sha256(row["prompt"]):
            raise ValueError("prompt identity mismatch")
    prompts = aggregate_prompts(rows)
    samples = []
    for split, pid in sorted({(r["split"], r["prompt_id"]) for r in prompts}):
        items = {r["action_id"]:r for r in prompts if r["split"]==split and r["prompt_id"]==pid}
        samples.append({"split":split, "prompt_id":pid, "prompt":items[NATIVE]["prompt"],
                        "prompt_sha256":canonical_sha256(items[NATIVE]["prompt"]),
                        "quality":[items[a]["vbench5"] for a in actions],
                        "seconds":[items[a]["pipeline_seconds"] for a in actions],
                        "native_seconds":items[NATIVE]["pipeline_seconds"],
                        "dimensions":{d:[items[a][d] for a in actions] for d in DIMENSIONS},
                        "seed_seconds":[[r["pipeline_seconds"] for r in rows if r["split"]==split and r["prompt_id"]==pid and r["action_id"]==a] for a in actions]})
    return samples, rows, {"input_sha256":digest, "scores_sha256":scores["payload_sha256"]}


def load_t5(directory, samples):
    manifest = read(directory/"t5_manifest.json")
    if manifest.get("schema") != "prompt_t5_embeddings_manifest_v2" or not manifest.get("complete") or manifest.get("backend")!="wan_native":
        raise ValueError("expected complete frozen Wan native T5 manifest")
    if canonical_sha256({k:v for k,v in manifest.items() if k not in ("schema","manifest_sha256")}) != manifest.get("manifest_sha256"):
        raise ValueError("T5 manifest hash mismatch")
    by_text = {p["prompt_text"]:p for p in manifest["prompts"]}
    if len(by_text)!=len(manifest["prompts"]) or set(by_text)!={s["prompt"] for s in samples}:
        raise ValueError("T5 prompt coverage mismatch")
    vectors = []
    for s in samples:
        item = by_text[s["prompt"]]
        if hashlib.sha256(s["prompt"].encode("utf-8")).hexdigest()!=item["prompt_sha256"]:
            raise ValueError("T5 prompt hash mismatch")
        # Basenames permit downloading the whole directory from the remote host.
        path = directory/Path(item["npz_file"]).name
        if sha256_file(path)!=item["npz_sha256"]:
            raise ValueError(f"T5 feature hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            vector = np.asarray(data["pooled_embedding"], dtype=np.float64)
        if vector.shape!=(4096,) or not np.isfinite(vector).all():
            raise ValueError("invalid T5 pooled vector")
        vectors.append(vector)
    return np.stack(vectors), manifest["manifest_sha256"]


def extract_t5(args, samples):
    from changing_resolution_uni.scripts.data.extract_prompt_t5_embeddings import directory_file_inventory
    directory = Path(args.t5_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    model = Path(args.model_root).resolve()
    extractor = ROOT/"changing_resolution_uni/scripts/data/extract_prompt_t5_embeddings.py"
    texts = [s["prompt"] for s in samples]
    request = {"prompts":texts, "checkpoint_sha256":sha256_file(model/"models_t5_umt5-xxl-enc-bf16.pth"),
               "tokenizer_files":directory_file_inventory(model/"google/umt5-xxl"),
               "extractor_sha256":sha256_file(extractor), "precision":"bf16", "max_seq_len":512}
    with output_lock(directory):
        request_path = directory/"phase2_embedding_request.json"
        if request_path.exists() and read(request_path)!=request:
            raise ValueError("T5 extraction request changed; use a new --t5-dir")
        if not request_path.exists() and list(directory.glob("prompt_*.npz")):
            raise ValueError("existing unbound T5 cache; use a new --t5-dir")
        write_json_atomic(request_path, request)
        if (directory/"t5_manifest.json").exists():
            load_t5(directory, samples)
            print(f"Reusing verified T5 features: {directory}")
            return
        prompts_path = directory/"prompts.json"
        write_json_atomic(prompts_path, texts)
        subprocess.run([sys.executable, str(extractor), "--prompts_file", str(prompts_path),
                        "--out_dir", str(directory), "--model_path", str(model),
                        "--required_backend", "wan_native", "--precision", "bf16",
                        "--device", args.device], check=True)
        load_t5(directory, samples)


def calibrated_costs(rows, actions):
    return np.asarray([np.quantile([r["pipeline_seconds"] for r in rows if r["split"]=="train" and r["action_id"]==a], .95, method="higher") for a in actions])


def evaluate(samples, predictions, actions, train_q, train_cost, caps, budgets, split, rng_seed):
    q = np.asarray([s["quality"] for s in samples])
    seconds = np.asarray([s["seconds"] for s in samples])
    dimension_values = {d:np.asarray([s["dimensions"][d] for s in samples]) for d in DIMENSIONS}
    summary, details = [], []
    for budget in budgets:
        chosen, eligible = select_actions(predictions, caps, budget)
        if chosen is None:
            continue
        fixed = sorted(eligible, key=lambda a:(-train_q[a], train_cost[a], a))[0]
        histogram = np.bincount(chosen, minlength=len(actions))/len(chosen)
        target = float(histogram@train_cost)
        mix = train_mixture(train_q, train_cost, eligible, target)
        uniform = np.zeros(len(actions)); uniform[eligible] = 1/len(eligible)
        oracle = eligible[np.argmax(q[:,eligible], axis=1)]
        eye = np.eye(len(actions))
        policies = {"fixed_train":np.tile(eye[fixed],(len(q),1)),
                    "uniform_random_expected":np.tile(uniform,(len(q),1)),
                    "prompt_gain":eye[chosen],
                    "shuffled_router_hist_expected":np.tile(histogram,(len(q),1)),
                    "train_frontier_mixture_expected":np.tile(mix,(len(q),1)),
                    "quality_oracle_hindsight":eye[oracle]}
        qualities = {name:(w*q).sum(axis=1) for name,w in policies.items()}
        # Conditional randomization: same chosen action counts, shuffled across prompts.
        rng = np.random.default_rng(rng_seed)
        observed = float(qualities["prompt_gain"].mean())
        null = [float(q[np.arange(len(q)),rng.permutation(chosen)].mean()) for _ in range(2000)]
        permutation_p = (1+sum(x >= observed-1e-12 for x in null))/(len(null)+1)
        for name,weights in policies.items():
            quality = qualities[name]
            runtime = (weights*seconds).sum(axis=1)
            fixed_gain = quality-qualities["fixed_train"]
            shuffle_gain = quality-qualities["shuffled_router_hist_expected"]
            frontier_gain = quality-qualities["train_frontier_mixture_expected"]
            ci = bootstrap_ci(fixed_gain.tolist())
            shci = bootstrap_ci(shuffle_gain.tolist())
            fci = bootstrap_ci(frontier_gain.tolist())
            summary.append({"split":split, "budget_seconds":float(budget), "policy":name,
                            "prompts":len(q), "eligible_actions":[actions[a] for a in eligible],
                            "vbench5":float(quality.mean()), "pipeline_seconds":float(runtime.mean()),
                            "native_speedup_ratio_of_means":float(np.mean([s["native_seconds"] for s in samples])/runtime.mean()),
                            "calibrated_mean_seconds":float((weights@train_cost).mean()),
                            "prompt_mean_cap_violation_rate":float((weights*(seconds>budget+1e-9)).sum(axis=1).mean()),
                            "video_cap_violation_rate":float(np.mean([sum(weights[i,a]*np.mean(np.asarray(s["seed_seconds"][a])>budget+1e-9) for a in range(len(actions))) for i,s in enumerate(samples)])),
                            "gain_vs_fixed":float(fixed_gain.mean()), "gain_fixed_ci_low":ci[0], "gain_fixed_ci_high":ci[1],
                            "gain_vs_histogram_shuffle":float(shuffle_gain.mean()), "gain_shuffle_ci_low":shci[0], "gain_shuffle_ci_high":shci[1],
                            "gain_vs_train_frontier":float(frontier_gain.mean()), "gain_frontier_ci_low":fci[0], "gain_frontier_ci_high":fci[1],
                            "router_permutation_p":permutation_p if name=="prompt_gain" else None,
                            "action_fractions":{a:float(weights[:,j].mean()) for j,a in enumerate(actions)},
                            **{d:float((weights*values).sum(axis=1).mean()) for d,values in dimension_values.items()}})
            for i,s in enumerate(samples):
                details.append({"split":split, "prompt_id":s["prompt_id"], "budget_seconds":float(budget),
                                "policy":name, "action_weights":weights[i].tolist(), "vbench5":float(quality[i]),
                                "pipeline_seconds":float(runtime[i]), "gain_vs_fixed":float(fixed_gain[i])})
    return summary, details


def train(args, samples, rows, identity):
    actions = ACTION_SETS[args.action_set]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_features, feature_digest = load_t5(Path(args.t5_dir).resolve(), samples) if args.features=="t5" else (None,None)
    tr = np.asarray([i for i,s in enumerate(samples) if s["split"]=="train"])
    va = np.asarray([i for i,s in enumerate(samples) if s["split"]=="validation"])
    texts = [s["prompt"] for s in samples]
    q = np.asarray([s["quality"] for s in samples])
    # Predict only gains to CACHE; native is a reporting baseline, not a label target.
    target = q[:,1:]-q[:,[0]]
    request = {**identity, "actions":actions, "features":args.features, "t5_manifest_sha256":feature_digest,
               "folds":args.folds, "seed":args.seed, "alphas":args.alphas,
               "max_features":args.max_features, "budgets_seconds":args.budgets_seconds,
               "source_sha256":{p.name:sha256_file(p) for p in (Path(__file__), Path(__file__).with_name("phase2_gain_model.py"))}}
    with output_lock(out):
        if (out/"training_request.json").exists() and read(out/"training_request.json")!=request:
            raise ValueError("training request changed; use a new --out-dir to preserve previous results")
        write_json_atomic(out/"training_request.json", request)
        started = time.perf_counter()
        model, state, oof, folds, cv = cross_validate([texts[i] for i in tr], target[tr],
            embeddings=all_features[tr] if all_features is not None else None,
            folds=args.folds, seed=args.seed, alphas=args.alphas, max_features=args.max_features)
        xval = text_features([texts[i] for i in va], state) if state is not None else normalize(all_features[va])
        vp = predict(xval, model)
        duration = time.perf_counter()-started
        train_q = q[tr].mean(axis=0)
        train_cost = np.asarray([samples[i]["seconds"] for i in tr]).mean(axis=0)
        caps = calibrated_costs(rows, actions)
        budgets = sorted(set(args.budgets_seconds or caps.tolist()))
        if any(b < float(caps.min())-1e-9 for b in budgets):
            raise ValueError(f"budget below minimum calibrated cost {caps.min():.6f}s; no eligible action (no silent fallback)")
        np.savez_compressed(out/"model.npz", **model)
        meta = {"schema":"phase2_prompt_gain_ridge_v1", "actions":actions, "reference_action":CACHE,
                "features":args.features, "text_state":state, "selected_alpha":next(r["alpha"] for r in cv if r["selected"]),
                "train_mean_quality":train_q.tolist(), "train_mean_seconds":train_cost.tolist(),
                "train_p95_seconds":caps.tolist(), "budgets_seconds":budgets,
                "training_seconds":duration, "model_sha256":sha256_file(out/"model.npz"),
                "request_sha256":canonical_sha256(request)}
        write_json_atomic(out/"model.json", meta)
        csv_write(out/"cross_validation.csv",cv)
        prediction_rows = []
        summary, details = [], []
        for ids, pred, split in ((tr,oof,"train_oof_selection"),(va,vp,"validation_development")):
            gains = np.column_stack([np.zeros(len(pred)),pred])
            for pos,i in enumerate(ids):
                prediction_rows.append({"split":split, "prompt_id":samples[i]["prompt_id"], "prompt":texts[i],
                                        "fold":int(folds[pos]) if split=="train_oof_selection" else None,
                                        "predicted_gains":gains[pos].tolist(), "observed_gains":(q[i]-q[i,0]).tolist()})
            sr, dr = evaluate([samples[i] for i in ids], gains, actions, train_q, train_cost, caps,budgets,split,args.seed)
            summary.extend(sr); details.extend(dr)
        csv_write(out/"gain_predictions.csv",prediction_rows)
        csv_write(out/"policy_summary.csv",summary)
        csv_write(out/"policy_by_prompt.csv",details)
        write_json_atomic(out/"policy_summary.json", summary)
        lines = ["# Phase 2 prompt gain prior", "",
                 f"Features: {args.features}; actions: {actions}; selected ridge alpha: {meta['selected_alpha']} (null = mean-only).",
                 "Train-only prompt CV selects alpha by gain MSE; vocabulary and centering are fit inside each fold. Native is not a selectable action.",
                 "Train OOF results reuse CV for alpha selection and full-train cost/baseline calibration: they are selection diagnostics, not nested unbiased evaluation.",
                 "Validation was already inspected during research design and is a DEVELOPMENT set, not a fresh confirmation set.",
                 "Budget eligibility uses train P95 per-video latency. Prompt selection never reads validation quality or observed latency.",
                 "Report all prompts and actual cap violations; no post-hoc feasibility filtering or fallbacks. Times exclude text encoding/router overhead.",
                 "Shuffled-router expectation randomizes the router's action histogram over prompts, matching calibrated mean cost exactly, not necessarily actual measured cost.",
                 "Train-frontier mixture maximizes TRAIN mean quality under the router's calibrated mean cost, with prompt-independent probabilities. Random policy rows are exact expectations.",
                 "Mixture weights depend on the incoming batch's predicted histogram, not its quality labels. These are batch comparison controls, not an online per-request baseline.",
                 "Oracle observes evaluation quality within the calibrated action set; it does not enforce observed-time feasibility. Its violation rate is reported.",
                 "Bootstrap intervals condition on trained model, selected histogram/mixture and budget; they do not refit the controller or adjust for multiple comparisons.",
                 "TF-IDF tests lexical predictability only; a negative result does not rule out semantic prompt features. SKIP remains solver-grid cache reuse.", "",
                 "| Split | Budget s | Policy | Quality | Seconds | Cap violation | Gain vs fixed | Gain vs shuffled histogram |",
                 "|---|---:|---|---:|---:|---:|---:|---:|"]
        for r in summary:
            if r["split"]=="validation_development":
                lines.append(f"| {r['split']} | {r['budget_seconds']:.2f} | {r['policy']} | {r['vbench5']:.5f} | {r['pipeline_seconds']:.2f} | {r['video_cap_violation_rate']:.1%} | {r['gain_vs_fixed']:.5f} | {r['gain_vs_histogram_shuffle']:.5f} |")
        lines += ["", "Decision: require reproducible quality gains at comparable measured cost against both fixed and prompt-independent mixtures, then confirm on new prompts."]
        (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        print(f"Selected alpha={meta['selected_alpha']}; CPU fitting {duration:.2f}s; report: {out/'report.md'}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("check","embed","train"))
    parser.add_argument("--quality-dir",required=True,help="downloaded phase2_quality folder; videos not required")
    parser.add_argument("--out-dir")
    parser.add_argument("--action-set",choices=tuple(ACTION_SETS),default="three")
    parser.add_argument("--features",choices=("tfidf","t5"),default="tfidf")
    parser.add_argument("--t5-dir")
    parser.add_argument("--model-root",default="/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--seed",type=int,default=20260918)
    parser.add_argument("--max-features",type=int,default=4096)
    parser.add_argument("--alphas",nargs="+",type=float,default=[.1,1.,10.,100.])
    parser.add_argument("--budgets-seconds",nargs="+",type=float)
    args = parser.parse_args()
    if args.folds<2 or args.max_features<1 or any(not math.isfinite(a) or a<=0 for a in args.alphas+(args.budgets_seconds or [])):
        parser.error("folds >= 2, max-features >= 1, finite positive alphas/budgets required")
    quality = Path(args.quality_dir).resolve()
    args.out_dir = args.out_dir or str(quality/f"gain_prior_{args.features}_{args.action_set}")
    args.t5_dir = args.t5_dir or str(quality/"t5_phase2")
    samples, rows, identity = load_data(quality,ACTION_SETS[args.action_set])
    print(f"Verified {len(rows)} scored videos; {sum(s['split']=='train' for s in samples)} train prompts, {sum(s['split']=='validation' for s in samples)} validation prompts",flush=True)
    if args.mode=="check":
        return
    if args.mode=="embed":
        extract_t5(args,samples)
    else:
        train(args,samples,rows,identity)


if __name__=="__main__":
    main()
