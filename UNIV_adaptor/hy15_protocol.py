"""CPU-only protocol, immutable plans and integrity checks for HY1.5 pilots."""
from __future__ import annotations

import json
import math
from pathlib import Path

from UNIV_adaptor.data_protocol import canonical_sha256, sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "UNIV_adaptor/configs/hy15_endpoint_prior_v1.json"
DEFAULT_PROMPTS = ROOT / "prompts/univ_controlled_factor_v1.jsonl"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Refusing different content at {path}; use a fresh output root")
    else:
        write_json_atomic(path, value)


def refresh_indices(steps, count):
    if not 2 <= count <= steps:
        raise ValueError("Refresh count must be between 2 and steps")
    result = [int(math.floor(i * (steps - 1) / (count - 1) + 0.5)) for i in range(count)]
    if len(set(result)) != count:
        raise ValueError("Duplicate refresh indices")
    return result


def density(case, protocol):
    spatial = case["height"] * case["width"] / (protocol["height"] * protocol["width"])
    temporal = ((case["frames"] - 1) // 4 + 1) / ((protocol["frames"] - 1) // 4 + 1)
    main = spatial * temporal * case["refreshes"] / protocol["main_steps"]
    return {"main_proxy": main, "total_proxy_excluding_rgb": main + (0.08 if case["refine"] else 0),
            "spatial_area_ratio": spatial, "temporal_token_ratio": temporal}


def validate_protocol(p):
    if p["schema"] != "hy15_endpoint_prior_v1" or p["main_steps"] != 50:
        raise ValueError("Expected HY15 endpoint 50-step protocol")
    if p["refine_sigmas"] != [0.2, 0.15, 0.1, 0.05, 0.0]:
        raise ValueError("Refinement is frozen at sigma=0.2, HR4")
    if len(set(p["base_seeds"])) != len(p["base_seeds"]):
        raise ValueError("Duplicate base seeds")
    if len({c["id"] for c in p["cases"]}) != len(p["cases"]):
        raise ValueError("Duplicate cases")
    for c in p["cases"]:
        if any(c[k] % 16 for k in ("height", "width")) or (c["frames"] - 1) % 4:
            raise ValueError(f"Illegal VAE shape: {c}")
        if not 1 < c["frames"] <= p["frames"]:
            raise ValueError("Invalid temporal length")
        for k in ("height", "width"):
            if not p[k] / 2 <= c[k] <= p[k]:
                raise ValueError("Current spatial restoration only supports up to x2")
        refresh_indices(50, c["refreshes"])


def make_plan(protocol_path=DEFAULT_PROTOCOL, prompts_path=DEFAULT_PROMPTS):
    p = read(protocol_path)
    validate_protocol(p)
    prompts = [json.loads(line) for line in Path(prompts_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [r for r in prompts if r["family_id"] in p["families"]]
    if any(r["split"] == "test" for r in prompts):
        raise ValueError("This development experiment must not access test prompts")
    if len({r["prompt_id"] for r in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt ids")
    for family in p["families"]:
        rows = [r for r in prompts if r["family_id"] == family]
        if len(rows) != 4 or {(r["motion_level"], r["detail_level"]) for r in rows} != {
            (m, d) for m in ("low", "high") for d in ("low", "high")
        }:
            raise ValueError(f"Family {family} lacks the four distinct factor cells")
    jobs = []
    for r in prompts:
        for base in p["base_seeds"]:
            group = f"p{r['prompt_id']:04d}_b{base}"
            for case in p["cases"]:
                jobs.append({"id": f"{group}__{case['id']}", "group": group,
                             "prompt": r, "base_seed": base, "seed": base + r["prompt_id"],
                             "case": case, "density": density(case, p)})
    body = {"protocol": p, "prompts": prompts, "jobs": jobs,
            "prompt_source_sha256": sha256_file(prompts_path)}
    return {**body, "plan_sha256": canonical_sha256(body)}


def verify_plan(plan):
    if canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Plan integrity mismatch")


def record_paths(root, job):
    root = Path(root)
    return root / "videos" / (job["id"] + ".mp4"), root / "records" / (job["id"] + ".json")


def verify_record(root, job, plan, environment=None):
    video, path = record_paths(root, job)
    r = read(path)
    if r["plan_sha256"] != plan["plan_sha256"] or r["job"] != job:
        raise ValueError(f"Record identity mismatch: {path}")
    if environment is not None and r["environment_sha256"] != environment:
        raise ValueError("Generation environment changed")
    if r["video_sha256"] != sha256_file(video):
        raise ValueError(f"Video hash mismatch: {video}")
    if not math.isfinite(r["timing_seconds"]["candidate_total"]) or r["timing_seconds"]["candidate_total"] <= 0:
        raise ValueError("Invalid candidate timing")
    return r
