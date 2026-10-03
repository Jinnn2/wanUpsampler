"""Add the previously implemented clean-endpoint S/T arms without altering v3.

Shared orchestration/scoring comes from the frozen published pilot. Overrides
exist only in this new process; old drivers and frozen assets are untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from UNIV_adaptor.scripts.data import published_wan21_pilot as base
from UNIV_adaptor.wan21_endpoint_runtime import geometry
from UNIV_adaptor.transition import dvg_rounded_anchors

WORKER = Path(__file__).with_name("published_wan21_study_worker.py")
CONFIG = ROOT / "UNIV_adaptor/configs/published_wan21_study_v2.json"
LEGACY_WORKER = Path(base.__file__).with_name("published_wan21_worker.py")
BASE_RECEIPT_VALID = base.receipt_valid
BASE_BUILD_PLAN = base.build_plan
BASE_VALIDATE_OUTPUT = base.validate_output_root
BASE_FINALIZE, BASE_REPORT = base.finalize, base.report


def implementation():
    paths = [Path(__file__), WORKER, Path(base.__file__), LEGACY_WORKER,
             ROOT / "UNIV_adaptor/wan21_endpoint_runtime.py",
             ROOT / "UNIV_adaptor/flow.py", ROOT / "UNIV_adaptor/hr_refinement.py",
             ROOT / "UNIV_adaptor/transition.py", ROOT / "UNIV_adaptor/rgb_super_resolution.py",
             ROOT / "changing_resolution_distill/rgb_super_resolution.py",
             ROOT / "changing_resolution_distill/realesrgan_compat.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): base.file_hash(p) for p in paths}


def validate_output_root(out):
    BASE_VALIDATE_OUTPUT(out)
    path = Path(out) / "plan.json"
    if path.exists() and base.read(path).get("study_schema") != "published_wan21_study_plan_v2":
        raise ValueError("This is an older published pilot. Use a NEW study root; do not overwrite v3.")


def sr_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Existing RealESRGAN_x2plus.pth required: {path}. Set SR_CHECKPOINT; no bicubic fallback.")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": base.file_hash(path)}


def resolve_config(path, checkpoint):
    extra = base.read(path)
    if extra["schema"] != "published_wan21_study_config_v2":
        raise ValueError("Wrong study config schema")
    cfg = copy.deepcopy(base.validate_config(base.read(ROOT / extra["base_config"])))
    cfg["arms"] = [a for a in cfg["arms"] if a["id"] not in extra["exclude_from_main"]] + extra["extra_arms"]
    cfg["disabled_arms"] = [a for a in cfg["disabled_arms"] if a["id"] not in extra["exclude_from_calibration"]] + [extra["adapter_control"]]
    cfg["endpoint_protocol"] = {"hr": extra["hr"], "sr": extra["sr"] | {"wan_rgb_sr_checkpoint": str(checkpoint)},
                                "calibration_seed": cfg["calibration"]["seed"],
                                "calibration_prompts": [p["prompt"] for p in cfg["calibration"]["prompts"]]}
    cfg["method_scope"] = {"published": ["TEA008", "SCALING10"],
                           "custom_not_official_dvg": ["S_B050", "S_B025", "T_B050", "T_B025"],
                           "controls": ["FULL50", "STEP25", "FULL50_HR4", "FULL50_RT_HR4"],
                           "excluded_separate_stratum": extra["exclusion_reason"]}
    cfg["study_config_binding"] = {"path": str(path.resolve()), "sha256": base.file_hash(path),
                                    "base_config_sha256": base.file_hash(ROOT / extra["base_config"])}
    validate_endpoint_config(cfg)
    return base.validate_config(cfg)


def validate_endpoint_config(cfg):
    if cfg["sampling"]["size"] != "832*480" or cfg["sampling"]["frame_num"] != 81 or cfg["sampling"]["sample_solver"] != "unipc" or cfg["sampling"]["offload_model"]:
        raise ValueError("Study geometry/solver/offload differs from the existing Wan endpoint protocol")
    protocol = cfg["endpoint_protocol"]
    if protocol["hr"] != {"sigma": .2, "steps": 4, "noise_seed_offset": 1000000007}:
        raise ValueError("This study locks sigma=0.2, four independent HR steps and shared repair noise")
    if protocol["sr"]["wan_rgb_sr_backend"] != "realesrgan":
        raise ValueError("No silent bicubic substitution in the RGB-SR experiment")
    for arm in cfg["arms"] + cfg["disabled_arms"]:
        case = arm.get("endpoint")
        if not case:
            continue
        if arm["source"] != "wan21" or arm["steps"] != 50 or arm["flags"] or type(case["refine"]) is not bool:
            raise ValueError("Endpoint arms must use native full-compute 50-step Wan")
        geometry(case, (832, 480, 81))
        transition = case["transition"]
        if transition == "rgb_sr_vae":
            if case["frames"] != 81 or case["height"] * 2 < 480 or case["width"] * 2 < 832 or (case["width"], case["height"]) == (832, 480):
                raise ValueError("Spatial arms must change space only within RealESRGAN x2 reach")
        elif transition == "dvg_latent_anchor":
            if (case["width"], case["height"]) != (832, 480) or case["frames"] >= 81:
                raise ValueError("Temporal arms must change time only")
        elif transition in ("identity", "vae_roundtrip"):
            if (case["width"], case["height"], case["frames"]) != (832, 480, 81):
                raise ValueError("Full controls cannot change geometry")
        else:
            raise ValueError("Unknown endpoint transition")
        if not case["refine"] and (transition != "identity" or arm["id"] != "NATIVE_ADAPTER_OFF"):
            raise ValueError("Only the full adapter equivalence control skips HR4")


def plan(args):
    validate_output_root(args.out)
    identity = sr_identity(args.sr_checkpoint)
    cfg = resolve_config(args.config, identity["path"])
    commits = base.check_sources(cfg)
    source = ROOT / "UNIV_adaptor/external" / cfg["prompt_source"]["repository"] / cfg["prompt_source"]["path"]
    public = base.read(source)
    metadata = {"source_sha256": base.file_hash(source)}
    for prompt in cfg["prompts"]:
        if prompt["origin"] == "vbench":
            matches = [p for p in public if p["prompt_en"] == prompt["prompt"]]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous public prompt: {prompt['id']}")
            metadata[prompt["id"]] = matches[0]
    frozen = BASE_BUILD_PLAN(cfg, args.model_root.resolve(), args.ngpus, commits, metadata, base.weight_inventory(args.model_root))
    frozen.pop("plan_sha256")
    frozen.update(study_schema="published_wan21_study_plan_v2", implementation=implementation(), sr_checkpoint=identity)
    frozen["plan_sha256"] = base.digest(frozen)
    base.immutable(args.out / "plan.json", frozen)
    print(f"Frozen {frozen['plan_sha256'][:12]}: {sum(j['phase']=='pilot' for j in frozen['jobs'])} main videos, {sum(j['phase']=='calibration' for j in frozen['jobs'])} calibration videos. Old v3 untouched.")


def load_plan(out, verify_implementation=True):
    validate_output_root(out)
    frozen = base.read(Path(out) / "plan.json")
    if base.digest({k:v for k,v in frozen.items() if k != "plan_sha256"}) != frozen["plan_sha256"]:
        raise ValueError("Study plan hash mismatch")
    if verify_implementation:
        if frozen["implementation"] != implementation() or base.check_sources(frozen["config"]) != frozen["source_commits"]:
            raise ValueError("Study implementation/pinned sources changed after freezing; use a new output")
        if sr_identity(frozen["sr_checkpoint"]["path"]) != frozen["sr_checkpoint"]:
            raise ValueError("RealESRGAN checkpoint changed after freezing")
    return frozen


def validate_endpoint_receipt(row):
    import numpy as np
    arm = row["job"]["arm"]
    case = arm.get("endpoint")
    if not case:
        return
    endpoint = row.get("endpoint")
    if not endpoint or endpoint["schema"] != "native_wan21_endpoint_receipt_v1":
        raise ValueError("Endpoint receipt missing")
    expected = geometry(case, (832, 480, 81))
    if any(endpoint[k] != v for k,v in expected.items()):
        raise ValueError("Endpoint geometry/density differs from the frozen arm")
    if endpoint["main_steps"] != 50 or endpoint["main_terminal_sigma"] != 0 or endpoint["model_forward_counts"]["main"] != 100:
        raise ValueError("Incomplete full-compute main endpoint")
    if endpoint["executed_main_noise"]["shape"] != expected["main_latent_shape"] or endpoint["main_clean"]["shape"] != expected["main_latent_shape"] or endpoint["restored_clean"]["shape"] != expected["target_latent_shape"]:
        raise ValueError("Noise/clean endpoint shape mismatch")
    if row["noise"]["shape"] != expected["target_latent_shape"]:
        raise ValueError("Shared canonical noise field must have FULL geometry")
    alignment = endpoint["noise_alignment"]
    expected_anchors = {str(axis):list(dvg_rounded_anchors(expected["main_latent_shape"][axis], expected["target_latent_shape"][axis])) for axis in (1, 2, 3)}
    if alignment["policy"] != "nested_iid_full_field_anchor_subsample_v1" or alignment["anchors"] != expected_anchors:
        raise ValueError("Executed noise anchor alignment differs from frozen geometry")
    if endpoint["transition"]["baseline"] != case["transition"]:
        raise ValueError("Transition receipt mismatch")
    if case["refine"]:
        hr = endpoint["hr"]
        if not hr or not hr["fresh_solver_history"] or hr["shift"] != 1 or hr["formula"] != "(1-sigma)*clean_hr + sigma*noise" or not np.allclose(hr["sigmas"], [.2, .15, .1, .05, 0], atol=1e-7, rtol=0) or hr["timesteps"] != [200, 150, 100, 50] or endpoint["model_forward_counts"]["hr"] != 8 or hr["noise"]["shape"] != expected["target_latent_shape"] or hr["noise_seed"] != (row["job"]["seed"] + 1000000007) % (2**63 - 1):
            raise ValueError("Incorrect direct sigma=0.2/HR4 refinement receipt")
    elif endpoint["hr"] is not None or endpoint["model_forward_counts"]["hr"] != 0:
        raise ValueError("Adapter OFF unexpectedly refined")
    for artifact in endpoint["artifacts"]:
        if base.file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Persisted clean endpoint changed")
    if len(endpoint["artifacts"]) != 1 or endpoint["artifacts"][0]["kind"] != "main_clean_sigma_zero":
        raise ValueError("Missing reusable clean endpoint artifact")


def receipt_valid(out, frozen, job):
    if not BASE_RECEIPT_VALID(out, frozen, job):
        return False
    row = base.read(Path(out) / "records" / (job["id"] + ".json"))
    validate_endpoint_receipt(row)
    provenance = row.get("reused_from")
    if provenance:
        if base.file_hash(provenance["receipt_path"]) != provenance["receipt_sha256"] or base.file_hash(provenance["plan_path"]) != provenance["plan_sha256_file"]:
            raise ValueError("Reused calibration provenance changed")
    return True


def reuse_calibration(args):
    """Rebind verified compatible calibration assets by reference, never v3 edits."""
    frozen = load_plan(args.out)
    old_root = args.reuse_calibration_root
    if old_root is None or not (old_root / "plan.json").exists():
        print("No old calibration root found; calibrate will generate every missing record.")
        return
    if old_root.resolve() == args.out.resolve():
        raise ValueError("Cannot import calibration from the new output itself")
    old_path = old_root / "plan.json"
    old = base.read(old_path)
    if old.get("schema") != "published_wan21_plan_v1" or old.get("study_schema") or base.digest({k:v for k,v in old.items() if k != "plan_sha256"}) != old["plan_sha256"]:
        raise ValueError("Old calibration plan is not an intact original published pilot")
    if old["implementation"] != {p.name: base.file_hash(p) for p in (Path(base.__file__), LEGACY_WORKER)}:
        raise ValueError("Original published calibration implementation differs")
    if old["model_root"] != frozen["model_root"] or old["weight_inventory"] != frozen["weight_inventory"] or old["config"]["sampling"] != frozen["config"]["sampling"] or any(old["source_commits"].get(k) != v for k,v in frozen["source_commits"].items()):
        raise ValueError("Old calibration model/sampling/source identity differs")
    old_jobs = {j["id"]:j for j in old["jobs"] if j["phase"] == "calibration"}
    count = 0
    for job in frozen["jobs"]:
        if job["phase"] != "calibration" or job["arm"].get("endpoint") or receipt_valid(args.out, frozen, job):
            continue
        prior = old_jobs.get(job["id"])
        if not prior or any(prior[k] != job[k] for k in job if k != "gpu") or not BASE_RECEIPT_VALID(old_root, old, prior):
            continue
        receipt = old_root / "records" / (prior["id"] + ".json")
        row = base.read(receipt)
        row.update(job=job, plan_sha256=frozen["plan_sha256"], reused_from={
            "receipt_path": str(receipt.resolve()), "receipt_sha256": base.file_hash(receipt),
            "plan_path": str(old_path.resolve()), "plan_sha256_file": base.file_hash(old_path),
            "original_plan_sha256": old["plan_sha256"], "scope": "calibration only; measured runtime retained, video/sample reused by reference"})
        base.immutable(args.out / "records" / (job["id"] + ".json"), row)
        count += 1
    print(f"Reused {count} original calibration videos by reference. Old failed Jenga audit is neither copied nor overridden.")


def audit(args):
    import numpy as np
    frozen = load_plan(args.out)
    rows = base.collect(args.out, frozen, "calibration")
    expected = [j for j in frozen["jobs"] if j["phase"] == "calibration"]
    if len(rows) != len(expected):
        raise RuntimeError(f"Calibration incomplete: {len(rows)}/{len(expected)}; run calibrate")
    groups = defaultdict(dict)
    for row in rows:
        groups[row["job"]["group_id"]][row["job"]["arm"]["id"]] = row
    checks = []
    tolerance = frozen["config"]["calibration"]
    for group, arms in groups.items():
        full, adapter = arms["FULL50"], arms["NATIVE_ADAPTER_OFF"]
        with np.load(full["sample_path"]) as sample:
            baseline = sample["frames"]
        repair_hash = None
        for name, row in arms.items():
            item = {"group": group, "arm": name, "noise_match": row["noise"] == full["noise"],
                    "sampling_match": row["sampling_identity"] == full["sampling_identity"],
                    "environment_match": {k:v for k,v in row["environment"].items() if k != "wan_module"} == {k:v for k,v in full["environment"].items() if k != "wan_module"},
                    "pipeline_seconds": row["runtime"]["pipeline_seconds"]}
            item["passed"] = item["noise_match"] and item["sampling_match"] and item["environment_match"]
            if name.endswith("_OFF"):
                with np.load(row["sample_path"]) as sample:
                    diff = np.abs(sample["frames"] - baseline)
                item.update(sample_mae=float(diff.mean()), sample_max=float(diff.max()))
                item["passed"] &= item["sample_mae"] <= tolerance["sample_mae_tolerance"] and item["sample_max"] <= tolerance["sample_max_tolerance"]
            endpoint = row.get("endpoint")
            if endpoint and endpoint["hr"]:
                current = endpoint["hr"]["noise"]
                repair_hash = repair_hash or current
                item["shared_hr_noise"] = current == repair_hash
                item["passed"] &= item["shared_hr_noise"]
                if name.startswith("FULL50_"):
                    item["full_main_clean_matches_adapter"] = endpoint["main_clean"] == adapter["endpoint"]["main_clean"]
                    item["passed"] &= item["full_main_clean_matches_adapter"]
            checks.append(item)
    result = {"schema": "published_wan21_study_audit_v2", "plan_sha256": frozen["plan_sha256"],
              "passed": all(c["passed"] for c in checks), "checks": checks,
              "caveat": "Baseline/OFF subsample equivalence and endpoint/HR protocol checks, NOT equal-quality requirements on accelerated outputs or proof of prompt routing benefit."}
    base.write(args.out / "calibration_audit.json", result)
    print(f"Calibration audit passed={result['passed']}; details: {args.out / 'calibration_audit.json'}")
    if not result["passed"]:
        failed = [c for c in checks if not c["passed"]]
        raise RuntimeError(f"Calibration failed; do not relax thresholds: {failed}")


def blind(args):
    from UNIV_adaptor.scripts.data import acceleration_blind_audit as human
    frozen, dataset = base.finalized(args)
    base.finalized_score_rows(args, frozen, dataset)
    if dataset["partial_exploratory"]:
        raise ValueError("Preregistered human study requires all groups")
    cfg = frozen["config"]
    epsilon = {m:cfg["evaluation"]["metric_epsilon"] for m in ["vbench5"] + cfg["evaluation"]["quality_dimensions"] + cfg["evaluation"]["diagnostic_dimensions"]}
    epsilon["dynamic_degree"] = 0.0
    source, identity = human.load_source("published_wan21", args.out / "metrics", epsilon)
    # Include ALL pairs, including refinement/roundtrip controls; no score-based
    # selection. Role labels remain private and separate the scientific strata.
    spec = {"schema": "acceleration_blind_audit_v1", "seed": 20261003,
        "minimum_raters": cfg["evaluation"]["minimum_raters"], "presentation": cfg["evaluation"]["presentation"],
        "metric_epsilon": epsilon,
        "real_strata": [{"source": "published_wan21", "left": "FULL50", "right": a["id"], "count": len(dataset["complete_groups"])} for a in cfg["arms"] if a["id"] != "FULL50"],
        "synthetic": {"source": "published_wan21", "action": "FULL50", "bases": 0},
        "seed_controls": {"source": "published_wan21", "action": "FULL50", "count": 0}}
    pairs = human.make_plan(spec, {"published_wan21": source})
    roles = {a["id"]:a["role"] for a in cfg["arms"]}
    for pair in pairs:
        pair["method_role"] = roles[pair["b"]["action"]]
    count = len(pairs)
    for pair in random.Random(20261003).sample(pairs, min(6, count)):
        pairs.append(pair | {"id": human.digest(["repeat", pair["id"]])[:20], "kind": "reliability_repeat"})
    body = {"schema": "acceleration_blind_plan_v1", "config": spec, "sources": {"published_wan21": identity}, "pairs": pairs}
    body["plan_sha256"] = human.digest(body)
    base.immutable(args.out / "blind/private/plan.json", body)
    human.package(argparse.Namespace(out=args.out / "blind", path_map=[]))
    print(f"Packaged {count} primary + {len(pairs)-count} repeats. Controls and custom/DVG-inspired arms are separate private roles, not official published reproductions.")


def finalize(args):
    frozen = load_plan(args.out)
    groups = defaultdict(list)
    for row in base.collect(args.out, frozen, "pilot"):
        groups[row["job"]["group_id"]].append(row)
    for group, rows in groups.items():
        hr_noise = [r["endpoint"]["hr"]["noise"] for r in rows if r.get("endpoint", {}).get("hr")]
        if hr_noise and any(n != hr_noise[0] for n in hr_noise):
            raise ValueError(f"Unmatched paired HR repair noise in {group}")
        full_clean = [r["endpoint"]["main_clean"] for r in rows if r["job"]["arm"]["id"] in ("FULL50_HR4", "FULL50_RT_HR4")]
        if len(full_clean) == 2 and full_clean[0] != full_clean[1]:
            raise ValueError(f"Full controls have different main clean endpoints in {group}")
    BASE_FINALIZE(args)


def report(args):
    BASE_REPORT(args)
    path = args.out / "metrics/report.json"
    result = base.read(path)
    cfg = load_plan(args.out)["config"]
    roles = {a["id"]:a["role"] for a in cfg["arms"]}
    for item in result["summary"]:
        item["method_role"] = roles[item["arm"]]
    result["method_scope"] = cfg["method_scope"]
    result["caveats"] += ["S/T arms are our previously implemented endpoint pipelines, not an official DVG reproduction.",
        "Main-pass budgets 0.5/0.25 do not include SR/VAE/HR4 cost; wall speedups include those stages.",
        "FULL50_HR4 and FULL50_RT_HR4 are mechanistic controls, not acceleration methods."]
    base.write(path, result)


def activate():
    # Never mutate the frozen original files. Replace only this process's
    # orchestration hooks; workers have a separate, explicit adapter script.
    base.WORKER = WORKER
    base.load_plan, base.plan, base.validate_output_root = load_plan, plan, validate_output_root
    base.receipt_valid, base.audit, base.blind = receipt_valid, audit, blind
    base.finalize, base.report = finalize, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "reuse-calibration", "check", "calibrate", "audit", "generate", "status", "finalize", "score", "report", "blind", "blind-report", "export", "diagnose"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/published_wan21_study_v2")
    parser.add_argument("--model-root", type=Path, default=Path("/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B"))
    parser.add_argument("--sr-checkpoint", type=Path, required=True)
    parser.add_argument("--reuse-calibration-root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--snapshot-out", type=Path)
    parser.add_argument("--vbench-root", type=Path, default=Path("/mnt/afs_2/houze/VBench"))
    parser.add_argument("--vbench-python", default="/opt/conda/bin/python")
    parser.add_argument("--expected-vbench-commit", default="fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490")
    args = parser.parse_args()
    args.out, args.vbench_root = args.out.resolve(), args.vbench_root.resolve()
    activate()
    if args.mode == "plan":
        plan(args)
    elif args.mode == "reuse-calibration":
        reuse_calibration(args)
    elif args.mode in ("check", "calibrate", "generate"):
        if not (args.out / "plan.json").exists():
            plan(args)
        reuse_calibration(args)
        if args.mode == "check":
            import shutil
            for name in ("ffmpeg", "ffprobe"):
                if not shutil.which(name):
                    raise RuntimeError(f"Missing {name}")
            base.launch(args, calibration=True, probe=True)
        else:
            base.launch(args, calibration=args.mode == "calibrate")
    elif args.mode == "audit":
        audit(args)
    elif args.mode == "blind":
        blind(args)
    elif args.mode == "diagnose":
        # The legacy diagnose expects a resolved pilot config, not this overlay.
        print(f"Study root: {args.out}; frozen protocol: {load_plan(args.out)['plan_sha256']}")
        for path in sorted((args.out / "logs").glob("*.log"), key=lambda p:p.stat().st_mtime_ns)[-8:]:
            print(f"\n{path}\n{base.log_excerpt(path)}")
    else:
        getattr(base, args.mode.replace("-", "_"))(args)


if __name__ == "__main__":
    main()
