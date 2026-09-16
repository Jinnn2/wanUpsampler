"""Generate, score, and analyze the 8-case UNIV online-policy pilot."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.core import UniversalAction  # noqa: E402
from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    load_prompts,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.online_policy import (  # noqa: E402
    DIAGNOSTIC_DIMENSIONS,
    QUALITY_DIMENSIONS,
    action_counts,
    bootstrap_mean_ci,
    bootstrap_oracle_gain_ci,
    full_compute_steps,
    mean_quality,
    validate_spec,
)
from UNIV_adaptor.schedule import resolve_schedule  # noqa: E402
from UNIV_adaptor.validation import base_runtime_config  # noqa: E402


MANIFEST_SCHEMA = "univ_online_policy_generation_manifest_v1"
SCORE_SCHEMA = "univ_online_policy_vbench_v1"
ANALYSIS_SCHEMA = "univ_online_policy_existence_analysis_v1"


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_checkout_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit = None
        tracked_status = None
    return {
        "root": str(root),
        "git_commit": commit,
        "tracked_status": tracked_status,
    }


def source_identity(lightx2v_repo: str | Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = (
        REPO_ROOT / "UNIV_adaptor/online_policy.py",
        REPO_ROOT / "UNIV_adaptor/online_policy_runner.py",
        REPO_ROOT / "UNIV_adaptor/core.py",
        REPO_ROOT / "UNIV_adaptor/data_protocol.py",
        REPO_ROOT / "UNIV_adaptor/diagnostics.py",
        REPO_ROOT / "UNIV_adaptor/flow.py",
        REPO_ROOT / "UNIV_adaptor/hr_ablation_runner.py",
        REPO_ROOT / "UNIV_adaptor/hr_refinement.py",
        REPO_ROOT / "UNIV_adaptor/mrflow_ablation_runner.py",
        REPO_ROOT / "UNIV_adaptor/noise.py",
        REPO_ROOT / "UNIV_adaptor/schedule.py",
        REPO_ROOT / "UNIV_adaptor/transition.py",
        REPO_ROOT / "UNIV_adaptor/validation.py",
        REPO_ROOT / "UNIV_adaptor/wan_runner.py",
        REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_batch.py",
        REPO_ROOT
        / "changing_resolution_uni/scripts/data/batch_vbench_score_dataset.py",
        Path(__file__).resolve(),
    )
    return {
        "git_commit": commit,
        "files": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths
        },
        "lightx2v": git_checkout_identity(lightx2v_repo),
    }


def case_runtime_config(
    template: dict[str, Any], spec: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    config = base_runtime_config(template)
    action = case["initial_action"]
    config.update(
        {
            "univ_action": {
                "spatial_ratio": action["spatial_ratio"],
                "temporal_ratio": action["temporal_ratio"],
                "lr_nfe_ratio": 1.0,
                "switch_ratio": 1.0,
            },
            "univ_cache_mode": "residual",
            "univ_transition_baseline": "dvg_latent_anchor",
            "univ_enable_transition_diagnostics": False,
            "univ_mrflow_lr_steps": spec["reference_nfe"],
            "univ_mrflow_refine_sigma": case["hr_refine_sigma"],
            "univ_mrflow_hr_steps": case["hr_steps"],
            "univ_mrflow_reuse_endpoint": False,
            "univ_mrflow_boundary_path": "",
            "univ_mrflow_endpoint_state_dtype": "fp16",
            "univ_online_decision_step": spec["decision_step"],
            "univ_online_remaining_lr_full_compute": case["remaining_lr_full_compute"],
            "univ_online_case_id": case["name"],
            "univ_online_initial_group": case["initial_group"],
            "univ_online_case_role": case["role"],
            "univ_low_budget_artifact_id": case["name"],
        }
    )
    return config


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = Path(args.spec).resolve()
    template_path = Path(args.template_config).resolve()
    prompts_path = Path(args.prompts).resolve()
    out_root = Path(args.out_root).resolve()
    spec = validate_spec(load_json(spec_path))
    template = load_json(template_path)
    if int(template.get("infer_steps", 0)) != spec["reference_nfe"]:
        raise ValueError("template infer_steps differs from policy spec")
    prompts = load_prompts(prompts_path)
    selected = prompts[args.prompt_offset : args.prompt_offset + args.limit]
    if len(selected) != args.limit:
        raise ValueError(
            f"requested {args.limit} prompts at offset {args.prompt_offset}, "
            f"found {len(selected)}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("selected prompts must be unique")
    if not 0 <= args.timing_warmup < len(selected):
        raise ValueError("timing_warmup must be non-negative and smaller than limit")

    cases = []
    configs: list[tuple[Path, dict[str, Any]]] = []
    target_shape = (
        16,
        (int(template["target_video_length"]) - 1) // 4 + 1,
        int(template["target_height"]) // 8,
        int(template["target_width"]) // 8,
    )
    target_tokens = math.prod(target_shape[1:])
    for worker_slot, case in enumerate(spec["cases"]):
        config = case_runtime_config(template, spec, case)
        path = out_root / "configs" / f"{case['name']}.json"
        action = UniversalAction(**config["univ_action"])
        resolved = resolve_schedule(
            action,
            reference_nfe=spec["reference_nfe"],
            target_latent_shape=target_shape,
        )
        compute = full_compute_steps(
            decision_step=spec["decision_step"],
            reference_nfe=spec["reference_nfe"],
            remaining_full_compute=case["remaining_lr_full_compute"],
        )
        cases.append(
            {
                **case,
                "worker_slot": worker_slot,
                "model_cls": "wan2.1_univ_online_policy_existence",
                "config_path": str(path),
                "config_sha256": canonical_sha256(config),
                "low_latent_shape": list(resolved.low_latent_shape),
                "low_latent_token_ratio": (
                    math.prod(resolved.low_latent_shape[1:]) / target_tokens
                ),
                "total_lr_full_compute": len(compute),
                "remaining_lr_compute_steps": [
                    step for step in compute if step >= spec["decision_step"]
                ],
            }
        )
        configs.append((path, config))

    balanced_case = next(case for case in cases if case["role"] == "continue")
    balanced_ratio = balanced_case["low_latent_token_ratio"]
    for case in cases:
        if case["role"] != "restart":
            continue
        ratio = case["low_latent_token_ratio"] / balanced_ratio
        if not 0.95 <= ratio <= 1.05:
            raise ValueError(
                f"resolved token ratio for {case['name']} differs from balanced by "
                f"more than 5%: {ratio:.6f}"
            )

    body = {
        "spec": spec,
        "spec_path": str(spec_path),
        "spec_file_sha256": sha256_file(spec_path),
        "template_config": str(template_path),
        "template_file_sha256": sha256_file(template_path),
        "prompts_file": str(prompts_path),
        "prompts_file_sha256": sha256_file(prompts_path),
        "selected_prompts": selected,
        "prompt_offset": args.prompt_offset,
        "prompt_count": args.limit,
        "timing_warmup": args.timing_warmup,
        "seed_base": args.seed,
        "target_video_length": int(template["target_video_length"]),
        "model_root": str(Path(args.model_root).resolve()),
        "out_root": str(out_root),
        "source": source_identity(args.lightx2v_repo),
        "cases": cases,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_sha256": canonical_sha256(body),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **body,
    }
    manifest_path = out_root / "generation_manifest.json"
    if manifest_path.is_file():
        previous = validate_manifest(load_json(manifest_path))
        if previous["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError(
                f"experiment changed under {out_root}; use a fresh OUT_ROOT"
            )
        manifest = previous
    for path, config in configs:
        if path.is_file() and canonical_sha256(load_json(path)) != canonical_sha256(
            config
        ):
            raise RuntimeError(f"prepared config changed: {path}")
        if not path.is_file():
            write_json_atomic(path, config)
    if not manifest_path.is_file():
        write_json_atomic(manifest_path, manifest)
    print(f"Prepared {len(cases)} cases x {len(selected)} prompts at {out_root}")
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported online-policy generation manifest")
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"schema", "manifest_sha256", "created_at_utc"}
    }
    if canonical_sha256(body) != manifest.get("manifest_sha256"):
        raise ValueError("manifest hash mismatch")
    validate_spec(manifest["spec"])
    if len(manifest.get("cases", [])) != 8:
        raise ValueError("manifest must contain eight cases")
    return manifest


def expected_video(
    manifest: dict[str, Any], case: dict[str, Any], prompt_index: int
) -> Path:
    seed = int(manifest["seed_base"]) + prompt_index
    return (
        Path(manifest["out_root"])
        / "videos"
        / case["name"]
        / f"{case['name']}_{prompt_index:02d}_seed{seed}.mp4"
    )


def timing_rows(manifest: dict[str, Any], case: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(manifest["out_root"]) / "timings" / f"{case['name']}.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_sidecar(
    path: Path, manifest: dict[str, Any], case: dict[str, Any], prompt_index: int
) -> dict[str, Any]:
    runtime = load_json(path)
    online = runtime.get("online_decision", {})
    endpoint = runtime.get("endpoint_state", {})
    expected_seed = int(manifest["seed_base"]) + prompt_index
    expected_remaining_steps = [
        step
        for step in full_compute_steps(
            decision_step=manifest["spec"]["decision_step"],
            reference_nfe=manifest["spec"]["reference_nfe"],
            remaining_full_compute=case["remaining_lr_full_compute"],
        )
        if step >= manifest["spec"]["decision_step"]
    ]
    if (
        runtime.get("schema") != "wan_univ_online_policy_existence_v1"
        or runtime.get("artifact_id") != case["name"]
        or int(runtime.get("seed", -1)) != expected_seed
        or online.get("schema") != "univ_online_decision_observation_v1"
        or online.get("case_id") != case["name"]
        or online.get("case_role") != case["role"]
        or online.get("initial_group") != case["initial_group"]
        or int(online.get("decision_step", -1)) != manifest["spec"]["decision_step"]
        or int(online.get("remaining_lr_full_compute", -1))
        != case["remaining_lr_full_compute"]
        or online.get("remaining_lr_compute_steps") != expected_remaining_steps
        or endpoint.get("schema") != "univ_mrflow_clean_transition_v1"
    ):
        raise ValueError(f"runtime sidecar contract mismatch: {path}")
    endpoint_path = Path(endpoint.get("path", ""))
    if not endpoint_path.is_file() or endpoint_path.stat().st_size < 1024:
        raise FileNotFoundError(f"endpoint archive is missing: {endpoint_path}")
    digests = {
        "state_sha256": str(online.get("state_sha256", "")),
        "predicted_clean_sha256": str(online.get("predicted_clean_sha256", "")),
        "clean_lr_sha256": str(
            runtime.get("lr_endpoint", {}).get("clean_lr_sha256", "")
        ),
    }
    for key, digest in digests.items():
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"invalid runtime hash {key}: {path}")
    return runtime


def case_complete(manifest: dict[str, Any], case: dict[str, Any]) -> bool:
    try:
        rows = timing_rows(manifest, case)
        if len([row for row in rows if row.get("kind") == "initialization"]) != 1:
            return False
        videos = [row for row in rows if row.get("kind") == "video"]
        if len(videos) != manifest["prompt_count"]:
            return False
        by_index = {int(row["prompt_index"]): row for row in videos}
        expected_indices = range(
            manifest["prompt_offset"],
            manifest["prompt_offset"] + manifest["prompt_count"],
        )
        if set(by_index) != set(expected_indices):
            return False
        for prompt_index in expected_indices:
            output = expected_video(manifest, case, prompt_index).resolve()
            row = by_index[prompt_index]
            if (
                Path(row["output"]).resolve() != output
                or not output.is_file()
                or output.stat().st_size < 1024
            ):
                return False
            validate_sidecar(
                output.with_suffix(output.suffix + ".univ.json"),
                manifest,
                case,
                prompt_index,
            )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def reset_incomplete_case(manifest: dict[str, Any], case: dict[str, Any]) -> None:
    out_root = Path(manifest["out_root"]).resolve()
    video_dir = (out_root / "videos" / case["name"]).resolve()
    timing_path = (out_root / "timings" / f"{case['name']}.jsonl").resolve()
    if out_root not in video_dir.parents or out_root not in timing_path.parents:
        raise RuntimeError("refusing to reset a case outside the experiment root")
    if video_dir.exists():
        shutil.rmtree(video_dir)
    timing_path.unlink(missing_ok=True)


def generate_case(args: argparse.Namespace) -> None:
    manifest = validate_manifest(load_json(args.manifest))
    cases = {case["name"]: case for case in manifest["cases"]}
    if args.case_name not in cases:
        raise ValueError(f"unknown case: {args.case_name}")
    case = cases[args.case_name]
    if args.resume:
        if case_complete(manifest, case):
            print(f"[resume] {case['name']}")
            return
        reset_incomplete_case(manifest, case)
        print(f"[resume-reset] {case['name']} from prompt zero")
    if sha256_file(manifest["prompts_file"]) != manifest["prompts_file_sha256"]:
        raise RuntimeError("prompt file changed after planning")
    if source_identity(args.lightx2v_repo) != manifest["source"]:
        raise RuntimeError("UNIV or LightX2V source changed after planning")
    if canonical_sha256(load_json(case["config_path"])) != case["config_sha256"]:
        raise RuntimeError(f"case config changed: {case['config_path']}")
    environment = dict(os.environ)
    environment["LIGHTX2V_REPO"] = str(Path(args.lightx2v_repo).resolve())
    roots = [environment["LIGHTX2V_REPO"], str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    out_root = Path(manifest["out_root"])
    command = [
        args.wan_python,
        str(REPO_ROOT / "UNIV_adaptor/scripts/bridge/run_wan_univ_batch.py"),
        "--seed",
        str(manifest["seed_base"]),
        "--model_cls",
        case["model_cls"],
        "--model_path",
        manifest["model_root"],
        "--config_json",
        case["config_path"],
        "--prompts_file",
        manifest["prompts_file"],
        "--out_dir",
        str(out_root / "videos" / case["name"]),
        "--name_prefix",
        case["name"],
        "--limit",
        str(manifest["prompt_count"]),
        "--prompt-offset",
        str(manifest["prompt_offset"]),
        "--timing-jsonl",
        str(out_root / "timings" / f"{case['name']}.jsonl"),
        "--timing-warmup",
        str(manifest["timing_warmup"]),
        "--target_video_length",
        str(manifest["target_video_length"]),
        "--negative_prompt",
        args.negative_prompt,
    ]
    print(f"[generate] {case['name']}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    if not case_complete(manifest, case):
        raise RuntimeError(f"case finished without complete artifacts: {case['name']}")


def stage_vbench_inputs(manifest: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(manifest["out_root"])
    inputs = root / "metrics" / "vbench_inputs" / "all_cases"
    inputs.mkdir(parents=True, exist_ok=True)
    prompt_map: dict[str, str] = {}
    expected_names: set[str] = set()
    for case in manifest["cases"]:
        if not case_complete(manifest, case):
            raise RuntimeError(f"generation is incomplete: {case['name']}")
        for position, prompt in enumerate(manifest["selected_prompts"]):
            index = manifest["prompt_offset"] + position
            source = expected_video(manifest, case, index).resolve()
            destination = inputs / source.name
            expected_names.add(destination.name)
            if destination.is_file():
                if sha256_file(destination) != sha256_file(source):
                    raise RuntimeError(f"staged VBench input changed: {destination}")
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            prompt_map[str(destination.resolve())] = prompt
    unexpected = {path.name for path in inputs.glob("*.mp4")} - expected_names
    if unexpected:
        raise RuntimeError(f"unexpected staged VBench videos: {sorted(unexpected)[:5]}")
    map_path = inputs.parent / "prompt_map.json"
    write_json_atomic(map_path, prompt_map)
    return inputs, map_path


def run_vbench(args: argparse.Namespace) -> None:
    manifest = validate_manifest(load_json(args.manifest))
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout,
        score_case_directory,
        warmup_vbench_cache,
    )

    vbench_root = Path(args.vbench_root).resolve()
    identity = inspect_vbench_checkout(
        vbench_root, expected_commit=args.vbench_commit or None
    )
    if not args.skip_vbench_warmup:
        warmup_vbench_cache(args.vbench_python, vbench_root)
    inputs, prompt_map = stage_vbench_inputs(manifest)
    dimensions = [*QUALITY_DIMENSIONS, *DIAGNOSTIC_DIMENSIONS]
    bundle = score_case_directory(
        vbench_root,
        args.vbench_python,
        inputs,
        prompt_map,
        Path(manifest["out_root"]) / "metrics" / "vbench_run",
        dimensions,
        list(QUALITY_DIMENSIONS),
        list(DIAGNOSTIC_DIMENSIONS),
        args.vbench_ngpus,
        args.force,
        identity,
    )
    cases: dict[str, dict[str, Any]] = {}
    for case in manifest["cases"]:
        per_video = {}
        for index in range(
            manifest["prompt_offset"],
            manifest["prompt_offset"] + manifest["prompt_count"],
        ):
            stem = expected_video(manifest, case, index).stem
            if stem not in bundle.scores:
                raise RuntimeError(f"VBench score missing for {stem}")
            per_video[stem] = bundle.scores[stem]
        aggregate = {
            dimension: statistics.mean(
                float(scores[dimension]) for scores in per_video.values()
            )
            for dimension in dimensions
        }
        cases[case["name"]] = {
            "aggregate": aggregate,
            "quality5_mean": statistics.mean(
                aggregate[key] for key in QUALITY_DIMENSIONS
            ),
            "per_video": per_video,
        }
    payload = {
        "schema": SCORE_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "quality_dimensions": list(QUALITY_DIMENSIONS),
        "diagnostic_dimensions": list(DIAGNOSTIC_DIMENSIONS),
        "vbench_provenance": bundle.provenance,
        "cases": cases,
    }
    output = Path(manifest["out_root"]) / "metrics" / "vbench_scores.json"
    write_json_atomic(output, payload)
    print(f"VBench scores: {output}")


def _case_sample_rows(
    manifest: dict[str, Any], scores: dict[str, Any]
) -> tuple[dict[int, dict[str, dict[str, Any]]], list[int]]:
    samples: dict[int, dict[str, dict[str, Any]]] = {}
    measured_indices: list[int] = []
    warmup_stop = manifest["prompt_offset"] + manifest["timing_warmup"]
    for case in manifest["cases"]:
        by_index = {
            int(row["prompt_index"]): row
            for row in timing_rows(manifest, case)
            if row.get("kind") == "video"
        }
        for index, timing in by_index.items():
            if index < warmup_stop:
                continue
            if index not in measured_indices:
                measured_indices.append(index)
            video = expected_video(manifest, case, index)
            runtime = validate_sidecar(
                video.with_suffix(video.suffix + ".univ.json"),
                manifest,
                case,
                index,
            )
            stem = video.stem
            dimensions = scores["cases"][case["name"]]["per_video"][stem]
            segment = float(timing["segment_elapsed_s"])
            stage_timing = runtime["timing_seconds"]
            candidate = float(stage_timing.get("candidate_denoise", 0.0))
            prefix = float(stage_timing.get("decision_prefix_lr", 0.0))
            observation = float(stage_timing.get("online_observation", 0.0))
            remaining = float(stage_timing.get("remaining_lr", 0.0))
            transition = float(stage_timing.get("transition", 0.0))
            hr_compute = float(stage_timing.get("hr_full_compute", 0.0))
            if (
                not math.isfinite(segment)
                or not math.isfinite(candidate)
                or segment <= 0
                or candidate <= 0
                or not math.isfinite(observation)
                or observation <= 0
                or any(
                    not math.isfinite(value) or value < 0
                    for value in (prefix, remaining, transition, hr_compute)
                )
                or prefix <= 0
                or not math.isclose(
                    candidate,
                    prefix + remaining + transition + hr_compute,
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                )
            ):
                raise ValueError(f"invalid timing for {case['name']} prompt {index}")
            samples.setdefault(index, {})[case["name"]] = {
                "quality": mean_quality(dimensions),
                "dynamic_degree": float(dimensions["dynamic_degree"]),
                "overall_consistency": float(dimensions["overall_consistency"]),
                "segment_seconds": segment,
                "candidate_denoise_seconds": candidate,
                "prefix_seconds": prefix,
                "observation_seconds": observation,
                "remaining_lr_seconds": remaining,
                "transition_seconds": transition,
                "hr_compute_seconds": hr_compute,
                "state_sha256": runtime["online_decision"]["state_sha256"],
                "predicted_clean_sha256": runtime["online_decision"][
                    "predicted_clean_sha256"
                ],
                "clean_lr_sha256": runtime["lr_endpoint"]["clean_lr_sha256"],
                "video": str(video),
            }
    measured_indices.sort()
    expected_cases = {case["name"] for case in manifest["cases"]}
    for index in measured_indices:
        if set(samples[index]) != expected_cases:
            raise RuntimeError(f"incomplete measured sample {index}")
    return samples, measured_indices


def _verify_counterfactual_contract(
    manifest: dict[str, Any], samples: dict[int, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    continues = [case for case in manifest["cases"] if case["role"] == "continue"]
    for index, rows in samples.items():
        for key in ("state_sha256", "predicted_clean_sha256"):
            hashes = {rows[case["name"]][key] for case in continues}
            if len(hashes) != 1:
                raise RuntimeError(
                    f"continuation cases do not share exact {key} for prompt {index}; "
                    "the online counterfactual is invalid"
                )
        by_lr: dict[int, list[dict[str, Any]]] = {}
        for case in continues:
            by_lr.setdefault(case["remaining_lr_full_compute"], []).append(case)
        for lr_level, cases in by_lr.items():
            hashes = {rows[case["name"]]["clean_lr_sha256"] for case in cases}
            if len(hashes) != 1:
                raise RuntimeError(
                    f"HR branches do not share LR{lr_level} endpoint for prompt {index}"
                )
    return {
        "exact_common_decision_state": True,
        "exact_common_predicted_clean": True,
        "exact_common_lr_endpoint_within_hr_pairs": True,
    }


def analyze_payload(
    manifest: dict[str, Any], scores: dict[str, Any], *, bootstrap_repetitions: int
) -> dict[str, Any]:
    if scores.get("schema") != SCORE_SCHEMA:
        raise ValueError("unsupported VBench score file")
    if scores.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("VBench scores refer to a different manifest")
    samples, indices = _case_sample_rows(manifest, scores)
    contract = _verify_counterfactual_contract(manifest, samples)
    if not indices:
        raise RuntimeError("no measured prompts remain after timing warmup")
    cases = {case["name"]: case for case in manifest["cases"]}
    continue_names = [
        name for name, case in cases.items() if case["role"] == "continue"
    ]
    restart_names = [name for name, case in cases.items() if case["role"] == "restart"]
    restart_suffix = (
        cases[restart_names[0]]["remaining_lr_full_compute"],
        cases[restart_names[0]]["hr_steps"],
        cases[restart_names[0]]["hr_refine_sigma"],
    )
    initial_names = [
        name
        for name, case in cases.items()
        if (
            case["remaining_lr_full_compute"],
            case["hr_steps"],
            case["hr_refine_sigma"],
        )
        == restart_suffix
    ]
    if len(initial_names) != 3:
        raise RuntimeError(
            "initial-strategy comparison must contain three matched cases"
        )
    reference_name = manifest["spec"]["reference_case"]
    case_summary = {}
    for name, case in cases.items():
        case_summary[name] = {
            "role": case["role"],
            "initial_group": case["initial_group"],
            "initial_action": case["initial_action"],
            "low_latent_shape": case["low_latent_shape"],
            "low_latent_token_ratio": case["low_latent_token_ratio"],
            "remaining_lr_full_compute": case["remaining_lr_full_compute"],
            "hr_steps": case["hr_steps"],
            "hr_refine_sigma": case["hr_refine_sigma"],
            "quality5_mean": statistics.mean(
                samples[index][name]["quality"] for index in indices
            ),
            "overall_consistency_mean": statistics.mean(
                samples[index][name]["overall_consistency"] for index in indices
            ),
            "dynamic_degree_mean": statistics.mean(
                samples[index][name]["dynamic_degree"] for index in indices
            ),
            "candidate_denoise_mean_s": statistics.mean(
                samples[index][name]["candidate_denoise_seconds"] for index in indices
            ),
            "online_observation_mean_ms": 1000.0
            * statistics.mean(
                samples[index][name]["observation_seconds"] for index in indices
            ),
            "segment_mean_s": statistics.mean(
                samples[index][name]["segment_seconds"] for index in indices
            ),
        }
    reference_cost = statistics.mean(
        samples[index][reference_name]["candidate_denoise_seconds"] for index in indices
    )
    near_tie = float(manifest["spec"]["near_tie_utility"])
    lambda_results = []
    per_sample_rows = []
    for lambda_value in manifest["spec"]["lambda_values"]:
        single_attempt_utilities: dict[int, dict[str, float]] = {}
        continuation_utilities: dict[int, dict[str, float]] = {}
        restart_utilities: dict[int, dict[str, float]] = {}
        expanded_decision_utilities: dict[int, dict[str, float]] = {}
        for index in indices:
            prefix = statistics.median(
                samples[index][name]["prefix_seconds"] for name in continue_names
            )
            observation = statistics.median(
                samples[index][name]["observation_seconds"] for name in continue_names
            )
            single_attempt_utilities[index] = {
                name: samples[index][name]["quality"]
                - lambda_value
                * samples[index][name]["candidate_denoise_seconds"]
                / reference_cost
                for name in cases
            }
            continuation_utilities[index] = {
                name: samples[index][name]["quality"]
                - lambda_value
                * (
                    prefix
                    + observation
                    + samples[index][name]["remaining_lr_seconds"]
                    + samples[index][name]["transition_seconds"]
                    + samples[index][name]["hr_compute_seconds"]
                )
                / reference_cost
                for name in continue_names
            }
            restart_utilities[index] = {
                name: samples[index][name]["quality"]
                - lambda_value
                * (
                    samples[index][name]["candidate_denoise_seconds"]
                    + prefix
                    + observation
                )
                / reference_cost
                for name in restart_names
            }
            expanded_decision_utilities[index] = {
                **continuation_utilities[index],
                **restart_utilities[index],
            }

        mean_by_case = {
            name: statistics.mean(
                single_attempt_utilities[index][name] for index in indices
            )
            for name in cases
        }
        mean_continue_by_case = {
            name: statistics.mean(
                continuation_utilities[index][name] for index in indices
            )
            for name in continue_names
        }
        mean_decision_by_case = {
            name: statistics.mean(
                expanded_decision_utilities[index][name] for index in indices
            )
            for name in cases
        }
        fixed_global = max(mean_by_case, key=mean_by_case.__getitem__)
        fixed_continue = max(
            continue_names, key=lambda name: mean_continue_by_case[name]
        )
        fixed_initial = max(initial_names, key=lambda name: mean_by_case[name])
        fixed_decision = max(
            mean_decision_by_case, key=mean_decision_by_case.__getitem__
        )
        online_names = []
        online_values = []
        single_names = []
        single_values = []
        initial_names_selected = []
        initial_values = []
        restart_aware_names = []
        restart_aware_values = []
        online_gains = []
        single_gains = []
        initial_gains = []
        restart_policy_gains = []
        restart_incremental_gains = []
        online_margins = []
        single_margins = []
        initial_margins = []
        restart_margins = []
        for index in indices:
            ranked_online = sorted(
                (
                    (continuation_utilities[index][name], name)
                    for name in continue_names
                ),
                reverse=True,
            )
            ranked_single = sorted(
                ((single_attempt_utilities[index][name], name) for name in cases),
                reverse=True,
            )
            ranked_initial = sorted(
                (
                    (single_attempt_utilities[index][name], name)
                    for name in initial_names
                ),
                reverse=True,
            )
            restart_choices = [
                (expanded_decision_utilities[index][name], name) for name in cases
            ]
            ranked_restart = sorted(restart_choices, reverse=True)
            online_values.append(ranked_online[0][0])
            online_names.append(ranked_online[0][1])
            single_values.append(ranked_single[0][0])
            single_names.append(ranked_single[0][1])
            initial_values.append(ranked_initial[0][0])
            initial_names_selected.append(ranked_initial[0][1])
            restart_aware_values.append(ranked_restart[0][0])
            restart_aware_names.append(ranked_restart[0][1])
            online_gains.append(
                ranked_online[0][0] - continuation_utilities[index][fixed_continue]
            )
            single_gains.append(
                ranked_single[0][0] - single_attempt_utilities[index][fixed_global]
            )
            initial_gains.append(
                ranked_initial[0][0] - single_attempt_utilities[index][fixed_initial]
            )
            fixed_decision_utility = expanded_decision_utilities[index][fixed_decision]
            restart_policy_gains.append(ranked_restart[0][0] - fixed_decision_utility)
            restart_incremental_gains.append(ranked_restart[0][0] - ranked_online[0][0])
            online_margins.append(ranked_online[0][0] - ranked_online[1][0])
            single_margins.append(ranked_single[0][0] - ranked_single[1][0])
            initial_margins.append(ranked_initial[0][0] - ranked_initial[1][0])
            restart_margins.append(restart_incremental_gains[-1])
            per_sample_rows.append(
                {
                    "lambda": lambda_value,
                    "prompt_index": index,
                    "seed": int(manifest["seed_base"]) + index,
                    "prompt": manifest["selected_prompts"][
                        index - manifest["prompt_offset"]
                    ],
                    "online_observation_case": reference_name,
                    "online_observation_sidecar": str(
                        expected_video(manifest, cases[reference_name], index)
                        .with_suffix(".mp4.univ.json")
                        .resolve()
                    ),
                    "fixed_continue": fixed_continue,
                    "online_lr_hr_oracle": ranked_online[0][1],
                    "online_margin": online_margins[-1],
                    "fixed_global": fixed_global,
                    "single_attempt_oracle": ranked_single[0][1],
                    "single_attempt_margin": single_margins[-1],
                    "fixed_initial": fixed_initial,
                    "initial_strategy_oracle": ranked_initial[0][1],
                    "initial_strategy_margin": initial_margins[-1],
                    "fixed_restart_aware_decision": fixed_decision,
                    "restart_aware_oracle": ranked_restart[0][1],
                    "restart_aware_margin": restart_margins[-1],
                    "online_gain": online_gains[-1],
                    "single_attempt_gain": single_gains[-1],
                    "initial_strategy_gain": initial_gains[-1],
                    "restart_aware_gain_vs_fixed_decision": restart_policy_gains[-1],
                    "restart_incremental_gain_vs_online_oracle": (
                        restart_incremental_gains[-1]
                    ),
                }
            )
        online_ci = bootstrap_oracle_gain_ci(
            [continuation_utilities[index] for index in indices],
            continue_names,
            repetitions=bootstrap_repetitions,
        )
        restart_policy_ci = bootstrap_oracle_gain_ci(
            [expanded_decision_utilities[index] for index in indices],
            list(cases),
            repetitions=bootstrap_repetitions,
            seed=20260908,
        )
        restart_incremental_ci = bootstrap_mean_ci(
            restart_incremental_gains,
            repetitions=bootstrap_repetitions,
            seed=20260911,
        )
        single_ci = bootstrap_oracle_gain_ci(
            [single_attempt_utilities[index] for index in indices],
            list(cases),
            repetitions=bootstrap_repetitions,
            seed=20260909,
        )
        initial_ci = bootstrap_oracle_gain_ci(
            [single_attempt_utilities[index] for index in indices],
            initial_names,
            repetitions=bootstrap_repetitions,
            seed=20260910,
        )
        decisive_online = [
            name
            for name, margin in zip(online_names, online_margins)
            if margin > near_tie
        ]
        decisive_restart = [
            name
            for name, margin in zip(restart_aware_names, restart_margins)
            if name in restart_names and margin > near_tie
        ]
        decisive_initial = [
            name
            for name, margin in zip(initial_names_selected, initial_margins)
            if margin > near_tie
        ]
        online_counts = action_counts(decisive_online)
        initial_counts = action_counts(decisive_initial)
        restart_counts = action_counts(decisive_restart)
        initial_groups = {cases[name]["initial_group"] for name in initial_counts}
        lr_levels = {cases[name]["remaining_lr_full_compute"] for name in online_counts}
        hr_levels = {cases[name]["hr_steps"] for name in online_counts}
        lambda_results.append(
            {
                "lambda": lambda_value,
                "reference_segment_seconds": reference_cost,
                "best_fixed_continue": fixed_continue,
                "best_fixed_initial": fixed_initial,
                "best_fixed_global": fixed_global,
                "best_fixed_restart_aware_decision": fixed_decision,
                "best_fixed_continue_value": mean_continue_by_case[fixed_continue],
                "best_fixed_global_value": mean_by_case[fixed_global],
                "best_fixed_restart_aware_decision_value": mean_decision_by_case[
                    fixed_decision
                ],
                "online_lr_hr_oracle_value": statistics.mean(online_values),
                "single_attempt_oracle_value": statistics.mean(single_values),
                "initial_strategy_oracle_value": statistics.mean(initial_values),
                "restart_aware_oracle_value": statistics.mean(restart_aware_values),
                "online_gain_vs_fixed_continue": statistics.mean(online_gains),
                "online_gain_ci95": list(online_ci),
                "single_attempt_gain_vs_fixed_global": statistics.mean(single_gains),
                "single_attempt_gain_ci95": list(single_ci),
                "initial_strategy_gain_vs_fixed_initial": statistics.mean(
                    initial_gains
                ),
                "initial_strategy_gain_ci95": list(initial_ci),
                "restart_aware_gain_vs_fixed_decision": statistics.mean(
                    restart_policy_gains
                ),
                "restart_aware_gain_vs_fixed_decision_ci95": list(restart_policy_ci),
                "restart_incremental_gain_vs_online_oracle": statistics.mean(
                    restart_incremental_gains
                ),
                "restart_incremental_gain_ci95": list(restart_incremental_ci),
                "online_decisive_action_counts": online_counts,
                "initial_strategy_decisive_action_counts": initial_counts,
                "restart_decisive_action_counts": restart_counts,
                "decisive_online_samples": len(decisive_online),
                "decisive_initial_strategy_samples": len(decisive_initial),
                "decisive_restart_samples": len(decisive_restart),
                "distinct_initial_strategy_groups": sorted(initial_groups),
                "distinct_online_lr_levels": sorted(lr_levels),
                "distinct_online_hr_levels": sorted(hr_levels),
                "restart_selected_decisively": sum(
                    restart_counts.get(name, 0) for name in restart_names
                ),
                "online_strategy_exists_in_sampled_space": (
                    online_ci[0] > 0 and (len(lr_levels) > 1 or len(hr_levels) > 1)
                ),
                "initial_strategy_exists_in_sampled_space": (
                    initial_ci[0] > 0 and len(initial_groups) > 1
                ),
                "restart_strategy_exists_in_sampled_space": (
                    restart_incremental_ci[0] > 0 and bool(restart_counts)
                ),
                "case_mean_utility": mean_by_case,
                "decision_case_mean_utility": mean_decision_by_case,
            }
        )
    return {
        "schema": ANALYSIS_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "score_file": str(
            Path(manifest["out_root"]) / "metrics" / "vbench_scores.json"
        ),
        "measured_prompt_indices": indices,
        "measured_prompt_count": len(indices),
        "timing_scope": (
            "synchronized_lr_transition_hr_and_online_observation_compute_"
            "excluding_text_encode_decode_and_experiment_checkpoint_io"
        ),
        "restart_cost_rule": (
            "alternative full denoising plus the abandoned balanced decision-prefix "
            "LR and online-observation time; prompt embeddings may be reused and only "
            "the final candidate is decoded"
        ),
        "cost_models": {
            "initial_strategy": "candidate full denoising; no online observation",
            "continuation": (
                "common median balanced prefix and observation plus candidate "
                "remaining LR, transition and HR"
            ),
            "restart": (
                "common median abandoned balanced prefix and observation plus "
                "alternative candidate full denoising"
            ),
        },
        "quality_definition": "arithmetic_mean_of_vbench5_working_dimensions",
        "canonical_online_observation_case": reference_name,
        "case_summary": case_summary,
        "counterfactual_contract": contract,
        "near_tie_utility": near_tie,
        "bootstrap_repetitions": bootstrap_repetitions,
        "lambda_results": lambda_results,
        "per_sample": per_sample_rows,
        "interpretation": (
            "These are sampled-oracle headroom tests. They establish whether useful "
            "decisions exist in the tested action set, not whether a learned online "
            "policy can predict them."
        ),
    }


def write_analysis_reports(manifest: dict[str, Any], payload: dict[str, Any]) -> None:
    reports = Path(manifest["out_root"]) / "reports"
    write_json_atomic(reports / "policy_existence.json", payload)
    rows = payload["per_sample"]
    if rows:
        path = reports / "policy_existence_per_sample.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    lines = [
        "# UNIV online policy existence pilot",
        "",
        f"Measured prompts: {payload['measured_prompt_count']}",
        "",
        "This report measures sampled-oracle headroom. It does not report learned-policy performance.",
        "Restart cost adds the abandoned balanced LR prefix and one online observation to the alternative generation segment.",
        "VBench-5 is a working quality proxy; Overall Consistency and Dynamic Degree are diagnostics and are not included in utility.",
        f"Canonical controller observation: `{payload['canonical_online_observation_case']}` balanced-prefix sidecar.",
        "",
        "## Candidate summary",
        "",
        "| case | role | initial group | requested S/T | LR latent CxTxHxW | token ratio | remaining LR compute | HR | quality5 | overall consistency | dynamic degree | observation ms | denoise s |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["case_summary"].items():
        lines.append(
            f"| {name} | {row['role']} | {row['initial_group']} | "
            f"{row['initial_action']['spatial_ratio']:.3f}/"
            f"{row['initial_action']['temporal_ratio']:.3f} | "
            f"{'x'.join(str(value) for value in row['low_latent_shape'])} | "
            f"{row['low_latent_token_ratio']:.4f} | "
            f"{row['remaining_lr_full_compute']} | {row['hr_steps']} | "
            f"{row['quality5_mean']:.6f} | "
            f"{row['overall_consistency_mean']:.6f} | "
            f"{row['dynamic_degree_mean']:.6f} | "
            f"{row['online_observation_mean_ms']:.3f} | "
            f"{row['candidate_denoise_mean_s']:.3f} |"
        )
    lines += [
        "",
        "## Sampled-oracle headroom",
        "",
        "| lambda | fixed continuation | online LR+HR gain (95% CI) | fixed matched initial strategy | initial-strategy gain (95% CI) | fixed expanded decision | expanded-policy gain (95% CI) | restart-only increment (95% CI) | initial exists | online exists | restart exists |",
        "| ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["lambda_results"]:
        online_ci = row["online_gain_ci95"]
        initial_ci = row["initial_strategy_gain_ci95"]
        restart_policy_ci = row["restart_aware_gain_vs_fixed_decision_ci95"]
        restart_incremental_ci = row["restart_incremental_gain_ci95"]
        lines.append(
            f"| {row['lambda']:.3f} | {row['best_fixed_continue']} | "
            f"{row['online_gain_vs_fixed_continue']:+.6f} "
            f"[{online_ci[0]:+.6f}, {online_ci[1]:+.6f}] | "
            f"{row['best_fixed_initial']} | "
            f"{row['initial_strategy_gain_vs_fixed_initial']:+.6f} "
            f"[{initial_ci[0]:+.6f}, {initial_ci[1]:+.6f}] | "
            f"{row['best_fixed_restart_aware_decision']} | "
            f"{row['restart_aware_gain_vs_fixed_decision']:+.6f} "
            f"[{restart_policy_ci[0]:+.6f}, {restart_policy_ci[1]:+.6f}] | "
            f"{row['restart_incremental_gain_vs_online_oracle']:+.6f} "
            f"[{restart_incremental_ci[0]:+.6f}, {restart_incremental_ci[1]:+.6f}] | "
            f"{row['initial_strategy_exists_in_sampled_space']} | "
            f"{row['online_strategy_exists_in_sampled_space']} | "
            f"{row['restart_strategy_exists_in_sampled_space']} |"
        )
    primary = next(
        row
        for row in payload["lambda_results"]
        if row["lambda"] == manifest["spec"]["primary_lambda"]
    )
    lines += [
        "",
        f"## Primary lambda = {primary['lambda']}",
        "",
        f"Online decisive actions: `{json.dumps(primary['online_decisive_action_counts'], sort_keys=True)}`",
        "",
        f"Initial-strategy decisive actions: `{json.dumps(primary['initial_strategy_decisive_action_counts'], sort_keys=True)}`",
        "",
        f"Restart-aware decisive actions: `{json.dumps(primary['restart_decisive_action_counts'], sort_keys=True)}`",
        "",
        "The restart flag uses only the incremental gain from adding restart actions beyond the per-sample continuation oracle. A positive flag also requires decisive action use. A false flag is a pilot result for this action set, not a proof that no useful policy exists anywhere.",
    ]
    report = reports / "POLICY_EXISTENCE.md"
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report)
    print("\n".join(lines))
    print(f"\nReport: {report}")


def analyze(args: argparse.Namespace) -> None:
    manifest = validate_manifest(load_json(args.manifest))
    score_path = Path(manifest["out_root"]) / "metrics" / "vbench_scores.json"
    payload = analyze_payload(
        manifest,
        load_json(score_path),
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    write_analysis_reports(manifest, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--spec", required=True)
    prepare_parser.add_argument("--template-config", required=True)
    prepare_parser.add_argument("--prompts", required=True)
    prepare_parser.add_argument("--out-root", required=True)
    prepare_parser.add_argument("--model-root", required=True)
    prepare_parser.add_argument("--lightx2v-repo", required=True)
    prepare_parser.add_argument("--prompt-offset", type=int, default=0)
    prepare_parser.add_argument("--limit", type=int, default=16)
    prepare_parser.add_argument("--timing-warmup", type=int, default=1)
    prepare_parser.add_argument("--seed", type=int, default=9700)

    list_parser = sub.add_parser("list-cases")
    list_parser.add_argument("--manifest", required=True)

    generate_parser = sub.add_parser("generate-case")
    generate_parser.add_argument("--manifest", required=True)
    generate_parser.add_argument("--case-name", required=True)
    generate_parser.add_argument("--wan-python", required=True)
    generate_parser.add_argument("--lightx2v-repo", required=True)
    generate_parser.add_argument("--negative-prompt", default="")
    generate_parser.add_argument("--resume", action="store_true")

    vbench_parser = sub.add_parser("vbench")
    vbench_parser.add_argument("--manifest", required=True)
    vbench_parser.add_argument("--vbench-root", required=True)
    vbench_parser.add_argument("--vbench-python", required=True)
    vbench_parser.add_argument("--vbench-ngpus", type=int, default=1)
    vbench_parser.add_argument("--vbench-commit", default="")
    vbench_parser.add_argument("--skip-vbench-warmup", action="store_true")
    vbench_parser.add_argument("--force", action="store_true")

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--manifest", required=True)
    analyze_parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    for name in ("prompt_offset", "limit", "timing_warmup"):
        if hasattr(args, name) and getattr(args, name) < 0:
            parser.error(f"{name} must be non-negative")
    if hasattr(args, "limit") and args.limit < 2:
        parser.error("limit must be at least 2")
    if hasattr(args, "vbench_ngpus") and args.vbench_ngpus < 1:
        parser.error("vbench-ngpus must be positive")
    if hasattr(args, "bootstrap_repetitions") and args.bootstrap_repetitions < 100:
        parser.error("bootstrap-repetitions must be at least 100")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "list-cases":
        manifest = validate_manifest(load_json(args.manifest))
        for case in sorted(manifest["cases"], key=lambda row: row["worker_slot"]):
            print(case["name"])
    elif args.command == "generate-case":
        generate_case(args)
    elif args.command == "vbench":
        run_vbench(args)
    elif args.command == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
