"""Read-only analysis of the user's metadata-only archive; no GPU assets loaded.

Outputs descriptive paired statistics and explicit cross-seed diagnostics.
These are not a trained prompt router or a human validation of metric failure.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

import numpy as np


def emit(name, obj):
    print(name + " " + json.dumps(obj, ensure_ascii=False, allow_nan=False))


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def ci(values, indices):
    means = np.asarray(values)[indices].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def correlation(left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main(path):
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        assert len(names) == len(set(names)), "Duplicate archive members"
        assert all(not n.startswith("/") and ".." not in Path(n).parts for n in names)

        def raw(name):
            return archive.extractfile(name).read()

        def read(name):
            return json.loads(raw(name).decode("utf-8-sig"))

        def table(name):
            return list(csv.DictReader(io.StringIO(raw(name).decode("utf-8-sig"))))

        plan = read("plan.json")
        dataset = read("dataset_manifest.json")
        audit = read("calibration_audit.json")
        provenance = read("metrics/score_provenance.json")
        rows = table("metrics/quality_by_video.csv")
        review = table("base_observability_review.csv")
        records = dataset["records"]
        cfg = plan["config"]
        assert plan["plan_sha256"] == hash_json({k:v for k,v in plan.items() if k != "plan_sha256"})
        assert dataset["plan_sha256"] == audit["plan_sha256"] == provenance["plan_sha256"] == plan["plan_sha256"]
        assert hashlib.sha256(raw("metrics/quality_by_video.csv")).hexdigest() == provenance["quality_csv_sha256"]
        assert audit["passed"] and len(audit["checks"]) == 26 and all(c["passed"] for c in audit["checks"])
        assert not dataset["partial_exploratory"]
        assert len(rows) == len(records) == 240
        assert len({r["observation_id"] for r in rows}) == 240
        recs = {r["job"]["id"]:r for r in records}
        jobs = {j["id"]:j for j in plan["jobs"]}
        quality = cfg["evaluation"]["quality_dimensions"]
        dimensions = quality + cfg["evaluation"]["diagnostic_dimensions"] + ["vbench5"]
        for row in rows:
            rec = recs[row["observation_id"]]
            assert rec == read("records/" + row["observation_id"] + ".json")
            assert rec["job"] == jobs[row["observation_id"]]
            assert rec["plan_sha256"] == plan["plan_sha256"]
            assert row["video_sha256"] == rec["video_sha256"]
            assert row["prompt"] == rec["job"]["prompt"]
            assert row["group_id"] == rec["job"]["group_id"]
            assert row["action_id"] == rec["job"]["arm"]["id"]
            assert int(row["seed"]) == rec["job"]["seed"]
            assert float(row["pipeline_seconds"]) == rec["runtime"]["pipeline_seconds"]
            assert math.isclose(float(row["vbench5"]), np.mean([float(row[d]) for d in quality]), abs_tol=1e-12)
            assert all(math.isfinite(float(row[d])) for d in dimensions)
            runtime = rec["runtime"]
            assert math.isclose(runtime["outer_seconds"], runtime["pipeline_seconds"] + runtime["model_load_seconds"] + runtime["instrumentation_seconds"] + runtime["encoding_seconds"], abs_tol=1e-6)
            assert runtime["warmed"]
        for method, metadata in provenance["vbench"].items():
            assert metadata["video_count"] == 24 and metadata["dimensions"] == dimensions[:-1]
            assert metadata["vbench"]["git_commit"] == "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
            assert not metadata["vbench"]["tracked_dirty"]

        prompts = cfg["prompts"]
        seeds = cfg["seeds"]
        arms = [a["id"] for a in cfg["arms"]]
        pidx = {p["id"]:i for i,p in enumerate(prompts)}
        sidx = {s:i for i,s in enumerate(seeds)}
        aidx = {a:i for i,a in enumerate(arms)}
        didx = {d:i for i,d in enumerate(dimensions)}
        values = np.full((len(prompts), len(seeds), len(arms), len(dimensions)), np.nan)
        seconds = np.full(values.shape[:-1], np.nan)
        for row in rows:
            p,s,a = pidx[row["prompt_key"]],sidx[int(row["seed"])],aidx[row["action_id"]]
            assert math.isnan(seconds[p,s,a])
            values[p,s,a] = [float(row[d]) for d in dimensions]
            seconds[p,s,a] = float(row["pipeline_seconds"])
        assert np.isfinite(values).all() and np.isfinite(seconds).all()
        full = aidx["FULL50"]
        qidx = didx["vbench5"]
        deltas = values - values[:,:,full:full+1,:]
        times = seconds / seconds[:,:,full:full+1]
        rng = np.random.default_rng(20261005)
        indices = rng.integers(0,len(prompts),size=(20000,len(prompts)))
        for prompt in prompts:
            for seed in seeds:
                group = [r for r in records if r["job"]["prompt_id"] == prompt["id"] and r["job"]["seed"] == seed]
                assert len(group) == 10
                assert all(r["noise"] == group[0]["noise"] and r["sampling_identity"] == group[0]["sampling_identity"] and r["environment"] == group[0]["environment"] for r in group)
                endpoints = [r["endpoint"] for r in group if r.get("endpoint")]
                assert all(e["hr"]["noise"] == endpoints[0]["hr"]["noise"] for e in endpoints)
                assert all(e["main_steps"] == 50 and e["main_terminal_sigma"] == 0 and e["model_forward_counts"] == {"main":100,"hr":8} and e["hr"]["fresh_solver_history"] and e["hr"]["shift"] == 1 and e["hr"]["timesteps"] == [200,150,100,50] for e in endpoints)
                controls = [r["endpoint"]["main_clean"] for r in group if r["job"]["arm"]["id"] in ("FULL50_HR4", "FULL50_RT_HR4")]
                assert controls[0] == controls[1]
        emit("INTEGRITY", {"plan":plan["plan_sha256"],"videos":len(records),"groups":len(dataset["complete_groups"]),"prompts":len(prompts),"seeds":seeds,"arms":arms,
             "calibration_checks":len(audit["checks"]),"review_filled":sum(bool(r["actual_motion"] or r["actual_detail"] or r["base_failure"]) for r in review),
             "verified":"archive metadata/CSV hash/noise/receipts/protocol; video/NPZ/PT bytes unavailable and NOT independently verified"})

        summary = []
        for arm in arms:
            a = aidx[arm]
            delta = deltas[:,:,a,qidx]
            per_prompt = delta.mean(axis=1)
            item = {"arm":arm,"seconds":float(seconds[:,:,a].mean()),"speedup_mean_paired":float((1/times[:,:,a]).mean()),
                    "score_mean":float(values[:,:,a,qidx].mean()),"delta_q":float(delta.mean()),"delta_q_ci_prompt_bootstrap":ci(per_prompt,indices),
                    "range_delta_q":[float(delta.min()),float(delta.max())],
                    "fraction_delta_abs_le_001":float((np.abs(delta)<=.001).mean()),
                    "fraction_delta_abs_le_005":float((np.abs(delta)<=.005).mean()),
                    "positive_q_groups":int((delta>0).sum()),
                    "delta_dimensions":{d:float(deltas[:,:,a,didx[d]].mean()) for d in dimensions[:-1]},
                    "dimension_means":{d:float(values[:,:,a,didx[d]].mean()) for d in dimensions[:-1]}}
            endpoints = [r["endpoint"]["timing"] for r in records if r["job"]["arm"]["id"]==arm and r.get("endpoint")]
            if endpoints:
                item["stage_seconds"] = {key:float(np.mean([e[key] for e in endpoints])) for key in endpoints[0]}
            summary.append(item)
        emit("METHOD_SUMMARY",summary)

        cell_summary = []
        for motion in ("low","high"):
            for detail in ("low","high"):
                selected = [i for i,p in enumerate(prompts) if p["motion"]==motion and p["detail"]==detail]
                cell_summary.append({"cell":motion+"/"+detail,"prompts":[prompts[i]["id"] for i in selected],
                    "delta_q":{arm:float(deltas[selected,:,aidx[arm],qidx].mean()) for arm in arms},
                    "delta_motion":{arm:float(deltas[selected,:,aidx[arm],didx["motion_smoothness"]].mean()) for arm in arms},
                    "dynamic_means":{arm:float(values[selected,:,aidx[arm],didx["dynamic_degree"]].mean()) for arm in arms}})
        emit("CELLS",cell_summary)

        # Descriptive paired method contrasts; values are Q5 units, NOT percent.
        comparisons = []
        for left,right in [("T_B050","S_B050"),("T_B025","S_B025"),("S_B025","S_B050"),("T_B025","T_B050"),("TEA008","SCALING10"),
                           ("FULL50_HR4","FULL50"),("FULL50_RT_HR4","FULL50_HR4")]:
            diff = values[:,:,aidx[left],qidx] - values[:,:,aidx[right],qidx]
            signs = np.sign(diff)
            item = {"left":left,"right":right,"mean_q_difference":float(diff.mean()),"ci":ci(diff.mean(axis=1),indices),
                    "cross_seed_corr":correlation(diff[:,0],diff[:,1]),"cross_seed_same_sign_prompts":int((signs[:,0]==signs[:,1]).sum()),
                    "left_wins_groups":int((diff>0).sum()),"right_wins_groups":int((diff<0).sum()),
                    "per_prompt":[{"id":p["id"],"family":p["family_id"],"cell":p["motion"]+"/"+p["detail"],
                                   "difference_by_seed":diff[i].tolist(),"mean":float(diff[i].mean())} for i,p in enumerate(prompts)]}
            # Balanced 2x2 prompt labels; bootstrap prompts within each label,
            # not frames/videos, and conditions on the observed two seeds.
            mean_diff = diff.mean(axis=1)
            low = [i for i,p in enumerate(prompts) if p["motion"]=="low"]
            high = [i for i,p in enumerate(prompts) if p["motion"]=="high"]
            effect = mean_diff[high].mean() - mean_diff[low].mean()
            boot = mean_diff[rng.choice(high,size=(20000,len(high)))].mean(axis=1) - mean_diff[rng.choice(low,size=(20000,len(low)))].mean(axis=1)
            item["high_minus_low_motion_difference"] = float(effect)
            item["motion_difference_ci"] = np.quantile(boot,[.025,.975]).tolist()
            comparisons.append(item)
        emit("CONTRASTS",comparisons)

        seed_diagnostics = []
        for arm in arms:
            a = aidx[arm]
            diff = deltas[:,:,a,qidx]
            between = float(np.var(diff.mean(axis=1),ddof=1))
            within = float(np.mean((diff[:,0]-diff[:,1])**2)/2)
            raw = values[:,:,a,qidx]
            seed_diagnostics.append({"arm":arm,"q_by_seed":raw.mean(axis=0).tolist(),"delta_q_by_seed":diff.mean(axis=0).tolist(),
                "raw_q_cross_seed_corr":correlation(raw[:,0],raw[:,1]),"relative_degradation_cross_seed_corr":correlation(diff[:,0],diff[:,1]),
                "delta_cross_seed_mean_absolute_difference":float(np.mean(np.abs(diff[:,0]-diff[:,1]))),
                "delta_prompt_mean_sd":math.sqrt(between),"delta_seed_residual_sd_estimate":math.sqrt(within),
                "signal_variance_moment_estimate":between-within/2})
        emit("SEED_DIAGNOSTICS",seed_diagnostics)

        allowed = ["STEP25","TEA008","SCALING10","S_B050","S_B025","T_B050","T_B025"]
        sel = [aidx[a] for a in allowed]
        scores = values[:,:,sel,qidx]
        normalized_times = times[:,:,sel]
        routing = []
        for lam in [0,.02,.05,.1]:
            utility = scores - lam*normalized_times
            fixed = int(np.argmax(utility.mean(axis=(0,1))))
            per_prompt = utility.mean(axis=1)
            prompt_choice = np.argmax(per_prompt,axis=1)
            noise_choice = np.argmax(utility,axis=2)
            fixed_values = utility[:,:,fixed].mean(axis=1)
            prompt_values = per_prompt[np.arange(len(prompts)),prompt_choice]
            seed_values = utility.max(axis=2).mean(axis=1)
            cross = []
            for train,test in [(0,1),(1,0)]:
                prior = utility[:,train,:]
                pred = np.argmax(prior,axis=1)
                fixed_train = int(np.argmax(prior.mean(axis=0)))
                held = utility[:,test,:]
                gain = held[np.arange(len(prompts)),pred] - held[:,fixed_train]
                cross.append({"train_seed":seeds[train],"test_seed":seeds[test],"train_selected_fixed":allowed[fixed_train],
                    "transfer_gain":float(gain.mean()),"ci":ci(gain,indices),
                    "chosen_actions":dict(Counter(allowed[i] for i in pred))})
            gain = prompt_values - fixed_values
            routing.append({"lambda":lam,"time_basis":"paired T/T_FULL; descriptive retrospectively-selected bounds, NOT previous studies' exact lambda basis",
                "fixed":allowed[fixed],"seed_oracle_gain":float((seed_values-fixed_values).mean()),
                "prompt_mean_hindsight_oracle_gain":float(gain.mean()),"prompt_oracle_gain_ci":ci(gain,indices),
                "fraction_seed_oracle_gain_captured_by_prompt_oracle":float(gain.mean()/(seed_values-fixed_values).mean()),
                "same_best_action_across_seeds":int((noise_choice[:,0]==noise_choice[:,1]).sum()),
                "prompt_choice":dict(Counter(allowed[i] for i in prompt_choice)),
                "cross_seed_lookup":cross,
                "per_prompt":[{"id":p["id"],"best_by_seed":[allowed[j] for j in noise_choice[i]],"best_mean":allowed[prompt_choice[i]],
                    "gain_mean":float(gain[i]),"utility_by_action_mean":dict(zip(allowed,per_prompt[i].tolist()))} for i,p in enumerate(prompts)]})
        emit("ROUTING",routing)

        prompt_results = []
        for i,p in enumerate(prompts):
            prompt_results.append({"id":p["id"],"family":p["family_id"],"origin":p["origin"],"cell":p["motion"]+"/"+p["detail"],"prompt":p["prompt"],
                "full_q_by_seed":values[i,:,full,qidx].tolist(),"full_dynamic":values[i,:,full,didx["dynamic_degree"]].tolist(),
                "delta_q_by_arm":{arm:deltas[i,:,aidx[arm],qidx].tolist() for arm in arms[1:]},
                "dynamic_by_arm":{arm:values[i,:,aidx[arm],didx["dynamic_degree"]].tolist() for arm in arms}})
        emit("PROMPTS",prompt_results)

        motion_cases = []
        high_motion = [i for i,p in enumerate(prompts) if p["motion"] == "high"]
        for i in high_motion:
            for s,seed in enumerate(seeds):
                a = aidx["T_B025"]
                if values[i,s,full,didx["dynamic_degree"]] == 1 and values[i,s,a,didx["dynamic_degree"]] == 0:
                    motion_cases.append({"prompt":prompts[i]["id"],"family":prompts[i]["family_id"],"seed":seed,
                        "full_q":float(values[i,s,full,qidx]),"t025_q":float(values[i,s,a,qidx]),
                        "deltas":{d:float(deltas[i,s,a,didx[d]]) for d in dimensions}})
        emit("MOTION_DIAGNOSTIC", {"high_motion_detected_counts":{arm:int(values[high_motion,:,aidx[arm],didx["dynamic_degree"]].sum()) for arm in arms},
                                  "full_detected_t025_not_detected":motion_cases})

        # Preregistered all pairs stay primary. These are diagnostic examples,
        # never a replacement for representative human evaluation.
        examples = []
        for row in rows:
            if row["action_id"] in ("TEA008","SCALING10"):
                i,s,a = pidx[row["prompt_key"]],sidx[int(row["seed"])],aidx[row["action_id"]]
                effects = {d:float(deltas[i,s,a,didx[d]]) for d in dimensions[:-1]}
                examples.append({"prompt":row["prompt_key"],"seed":row["seed"],"arm":row["action_id"],"delta_q":float(deltas[i,s,a,qidx]),"deltas":effects})
        examples.sort(key=lambda e:abs(e["delta_q"]))
        emit("CACHE_CLOSE_SCORE_EXAMPLES",examples[:12])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive",type=Path)
    main(parser.parse_args().archive)
