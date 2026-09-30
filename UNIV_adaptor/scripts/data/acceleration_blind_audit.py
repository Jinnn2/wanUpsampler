"""Reuse existing assets for a frozen, score-independent video preference pilot.

Plan/package/report/serve use stdlib; package requires ffmpeg/ffprobe.
Score calls the existing NumPy-based strict VBench runner.
Private manifest, media, and ratings are separate. Never serve the study root
with a generic static HTTP server: it contains unblinding information.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tarfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "UNIV_adaptor/configs/acceleration_blind_audit_v1.json"
DIMENSIONS = ("detail", "temporal", "prompt", "overall")
CHOICES = ("A", "B", "tie", "uncertain")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen output differs; use a new output directory: {path}")
    else:
        write(path, value)


def csv_write(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_text(path, filename, required=True):
    path = Path(path)
    if path.is_dir():
        match = path / filename
        if match.exists():
            return match.read_text(encoding="utf-8-sig")
    elif path.is_file() and tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            matches = [m for m in archive.getmembers() if m.isfile() and Path(m.name).name == filename]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous {filename} in {path}")
            if matches:
                return archive.extractfile(matches[0]).read().decode("utf-8-sig")
    if required:
        raise FileNotFoundError(f"{filename} missing from {path}")
    return None


def load_source(name, path, metrics):
    raw = source_text(path, "quality_by_video.csv")
    extra = source_text(path, "evaluation_inputs.json", required=False)
    identities = {}
    if extra:
        for row in json.loads(extra)["rows"]:
            identities[row.get("observation_id", row.get("stem"))] = row
    out = []
    for row in csv.DictReader(io.StringIO(raw)):
        key = row.get("observation_id", row.get("stem"))
        merged = {**identities.get(key, {}), **row}
        if merged.get("split") == "test" or "test" in merged.get("cohort", "").split("_"):
            continue
        if not merged.get("video_path") or not merged.get("video_sha256"):
            raise ValueError(f"{name}:{key}: missing content-bound video path/hash")
        prompt = merged["prompt"]
        out.append({"source": name, "id": key, "group": merged.get("group", merged.get("group_id")),
                    "action": merged.get("case", merged.get("case_id", merged.get("action_id"))),
                    "prompt": prompt, "prompt_key": digest(prompt),
                    "seed": str(merged["seed"]), "path": merged["video_path"],
                    "sha256": merged["video_sha256"],
                    "cell": merged.get("prompt_group") or (str(merged.get("motion", merged.get("motion_level", "unknown"))) + "/" + str(merged.get("detail", merged.get("detail_level", "unknown")))),
                    "family": merged.get("family_id", "unknown"),
                    "scores": {m: float(merged[m]) for m in metrics if merged.get(m) not in (None, "")}})
    if not out or len({r["id"] for r in out}) != len(out):
        raise ValueError(f"Empty or duplicate records: {name}")
    if any(not math.isfinite(v) for r in out for v in r["scores"].values()):
        raise ValueError(f"Non-finite source score: {name}")
    return out, {"quality_csv_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                 "input_json_sha256": hashlib.sha256(extra.encode()).hexdigest() if extra else None,
                 "eligible_records": len(out)}


def select_balanced(candidates, count, rng, used):
    """Balance cells within a stratum and prompts across strata; never read scores."""
    candidates = list(candidates)
    rng.shuffle(candidates)
    cells, local, selected = Counter(), Counter(), []
    if len(candidates) < count:
        raise ValueError(f"Need {count} pairs but only {len(candidates)} eligible; no silent quota reduction")
    for _ in range(count):
        i = min(range(len(candidates)), key=lambda j: (
            cells[candidates[j][0]["cell"]], local[candidates[j][0]["prompt_key"]],
            used[candidates[j][0]["prompt_key"]]))
        pair = candidates.pop(i)
        cells[pair[0]["cell"]] += 1
        local[pair[0]["prompt_key"]] += 1
        used[pair[0]["prompt_key"]] += 1
        selected.append(pair)
    return selected


def make_plan(config, sources):
    rng, used = random.Random(config["seed"]), Counter()
    pairs = []
    def add(kind, a, b, transform=None):
        if a["prompt"] != b["prompt"]:
            raise ValueError("Paired prompts differ")
        pairs.append({"kind": kind, "a": a, "b": b, "transform_b": transform,
                      "cluster": a["prompt_key"], "source": a["source"], "cell": a["cell"]})
    for spec in config["real_strata"]:
        groups = defaultdict(dict)
        for row in sources[spec["source"]]:
            if row["action"] in groups[row["group"]]:
                raise ValueError("Duplicate action within group")
            groups[row["group"]][row["action"]] = row
        candidates = [(g[spec["left"]], g[spec["right"]]) for g in groups.values()
                      if spec["left"] in g and spec["right"] in g]
        for a, b in select_balanced(candidates, spec["count"], rng, used):
            if a["seed"] != b["seed"]:
                raise ValueError("Real action comparison has mismatched seeds")
            add("real", a, b)
    spec = config["synthetic"]
    bases = [r for r in sources[spec["source"]] if r["action"] == spec["action"]]
    # Require distinct source prompts so each base supplies six correlated variants.
    distinct = {}
    shuffled = list(bases)
    rng.shuffle(shuffled)
    for row in shuffled:
        distinct.setdefault(row["prompt_key"], row)
    for a, _ in select_balanced([(r, r) for r in distinct.values()], spec["bases"], rng, used):
        for kind, levels in (("spatial", [0.5, 0.25]), ("temporal", [12, 6]), ("freeze", [0.5, 1.5])):
            for level in levels:
                add("synthetic", a, a, {"kind": kind, "level": level})
    spec = config["seed_controls"]
    groups = defaultdict(list)
    for row in sources[spec["source"]]:
        if row["action"] == spec["action"]:
            groups[row["prompt_key"]].append(row)
    candidates = [pair for group in groups.values() for pair in itertools.combinations(group, 2)
                  if pair[0]["seed"] != pair[1]["seed"]]
    for a, b in select_balanced(candidates, spec["count"], rng, used):
        add("seed_control", a, b)
    for pair in pairs:
        # Scores and filenames cannot affect pair identifiers or selection.
        pair["id"] = digest({"kind": pair["kind"], "a": pair["a"]["sha256"],
                             "b": pair["b"]["sha256"], "transform": pair["transform_b"]})[:20]
    if len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate pair")
    return pairs


def plan(args):
    config = read(args.config)
    paths = {k: str(ROOT / v) for k, v in config["sources"].items()}
    for value in args.source:
        name, path = value.split("=", 1)
        if name not in paths:
            raise ValueError(f"Unknown source {name}")
        paths[name] = path
    sources, provenance = {}, {}
    for name, path in paths.items():
        sources[name], provenance[name] = load_source(name, path, config["metric_epsilon"])
    body = {"schema": "acceleration_blind_plan_v1", "config": config, "sources": provenance,
            "pairs": make_plan(config, sources)}
    body["plan_sha256"] = digest(body)
    immutable(args.out / "private/plan.json", body)
    rows = [{"pair": p["id"], "kind": p["kind"], "source": p["source"], "cell": p["cell"],
             "prompt_key": p["cluster"], "seed_a": p["a"]["seed"], "seed_b": p["b"]["seed"],
             "action_a": p["a"]["action"], "action_b": p["b"]["action"],
             "transform_b": json.dumps(p["transform_b"]), "path_a": p["a"]["path"], "path_b": p["b"]["path"]}
            for p in body["pairs"]]
    csv_write(args.out / "private/pair_review.csv", rows)
    print(json.dumps({"plan_sha256": body["plan_sha256"], "pairs": len(rows),
                      "kinds": dict(Counter(p["kind"] for p in body["pairs"])),
                      "sources": dict(Counter(p["source"] for p in body["pairs"])),
                      "unique_prompts": len({p["cluster"] for p in body["pairs"]})}, indent=2))


def load_plan(out):
    plan = read(out / "private/plan.json")
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if digest(body) != plan["plan_sha256"]:
        raise ValueError("Plan hash mismatch")
    return plan


def mapped_path(value, maps):
    for mapping in maps:
        old, new = mapping.split("=", 1)
        if value == old or value.startswith(old.rstrip("/") + "/"):
            return Path(new) / value[len(old):].lstrip("/")
    return Path(value)


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height:format=duration", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return {**data["streams"][0], "duration": float(data["format"]["duration"])}


def render_filter(pres, transform, info):
    w, h, fps = pres["width"], pres["height"], pres["fps"]
    if info["width"] > w or info["height"] > h:
        raise ValueError("Presentation would downscale source detail; use a larger frozen canvas")
    filters = ["setpts=PTS-STARTPTS", f"fps={fps}"]
    # Spatial controls are defined on the native source, before display upscaling.
    if transform and transform["kind"] == "spatial":
        ratio = transform["level"]
        sw, sh = max(2, int(info["width"] * ratio) // 2 * 2), max(2, int(info["height"] * ratio) // 2 * 2)
        filters += [f"scale={sw}:{sh}:flags=area", f"scale={info['width']}:{info['height']}:flags=lanczos"]
    if transform and transform["kind"] == "temporal":
        filters += [f"fps={transform['level']}", f"fps={fps}"]
    if transform and transform["kind"] == "freeze":
        first = int(info["duration"] * fps * 0.35)
        last = min(int(info["duration"] * fps) - 2, first + int(transform["level"] * fps) - 1)
        filters += [f"split=2[live][hold];[live][hold]freezeframes=first={first}:last={last}:replace={first}"]
    filters += [f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos",
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2", "setsar=1", "format=yuv420p"]
    return ",".join(filters)


def package(args):
    plan = load_plan(args.out)
    pres, assets, pairs = plan["config"]["presentation"], {}, []
    for pair in plan["pairs"]:
        ids = []
        for row, transform in ((pair["a"], None), (pair["b"], pair["transform_b"])):
            clip = digest({"source": row["sha256"], "transform": transform, "presentation": pres})[:24]
            assets[clip] = {"row": row, "transform": transform}
            ids.append(clip)
        pairs.append({"id": pair["id"], "a": ids[0], "b": ids[1], "prompt": pair["a"]["prompt"]})
    # Validate all originals before launching any rendering; no resampling missing assets.
    checked, infos = {}, {}
    for asset in assets.values():
        row = asset["row"]
        path = mapped_path(row["path"], args.path_map)
        if row["path"] not in checked:
            if not path.is_file() or file_hash(path) != row["sha256"]:
                raise ValueError(f"Missing/changed source video: {path}")
            checked[row["path"]] = str(path)
            infos[row["path"]] = probe(path)
        render_filter(pres, asset["transform"], infos[row["path"]])
    for pair in plan["pairs"]:
        if abs(infos[pair["a"]["path"]]["duration"] - infos[pair["b"]["path"]]["duration"]) > 0.15:
            raise ValueError(f"Pair duration mismatch: {pair['id']}")
    implementation = file_hash(__file__)
    encoder = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout
    manifest = {"plan_sha256": plan["plan_sha256"], "implementation_sha256": implementation,
                "ffmpeg_version": encoder, "presentation": pres, "pairs": pairs, "clips": {}}
    media = args.out / "media"
    media.mkdir(parents=True, exist_ok=True)
    for index, (clip, asset) in enumerate(assets.items(), 1):
        row, transform = asset["row"], asset["transform"]
        video = media / f"{clip}.mp4"
        receipt = args.out / f"private/receipts/{clip}.json"
        request = digest({"implementation": implementation, "encoder": encoder, "clip": clip})
        if receipt.exists():
            saved = read(receipt)
            if saved.get("request_sha256") != request:
                raise ValueError("Rendering implementation/encoder changed; use a new study directory")
            if not video.exists() or file_hash(video) != saved["sha256"]:
                raise ValueError(f"Rendered asset changed: {video}")
        else:
            if video.exists():
                raise ValueError(f"Unreceipted output exists: {video}; inspect before removing")
            temp = media / f"{clip}.partial.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", checked[row["path"]], "-an",
                            "-map_metadata", "-1", "-filter_complex", "[0:v]" + render_filter(pres, transform, infos[row["path"]]) + "[presented]",
                            "-map", "[presented]",
                            "-t", str(infos[row["path"]]["duration"]), "-c:v", "libx264", "-crf", str(pres["crf"]),
                            "-preset", "medium", "-movflags", "+faststart", str(temp)], check=True)
            info = probe(temp)
            if abs(info["duration"] - infos[row["path"]]["duration"]) > 0.15:
                raise ValueError(f"Rendered duration mismatch: {temp}")
            temp.replace(video)
            saved = {"sha256": file_hash(video), "source_sha256": row["sha256"], "transform": transform,
                     "info": info, "request_sha256": request}
            immutable(receipt, saved)
        manifest["clips"][clip] = saved
        print(f"Prepared clip {index}/{len(assets)}", flush=True)
    manifest["package_sha256"] = digest(manifest)
    immutable(args.out / "private/package.json", manifest)
    csv_write(args.out / "private/presented_metric_inputs.csv", [
        {"clip_id": clip, "video_path": str((media / f"{clip}.mp4").resolve()),
         "video_sha256": manifest["clips"][clip]["sha256"], "prompt": asset["row"]["prompt"]}
        for clip, asset in assets.items()])
    print(f"Packaged {len(pairs)} pairs; serve this directory with this script only.")


def session(plan, package, participant):
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,40}", participant):
        raise ValueError("Participant ID: 2-40 letters, digits, underscore or hyphen")
    rng = random.Random(digest([plan["plan_sha256"], participant]))
    pairs = list(package["pairs"])
    rng.shuffle(pairs)
    out = []
    for pair in pairs:
        swap = bool(rng.getrandbits(1))
        out.append({"id": pair["id"], "prompt": pair["prompt"],
                    "A": pair["b"] if swap else pair["a"], "B": pair["a"] if swap else pair["b"], "swap": swap})
    return out


def load_package(out, plan):
    package = read(out / "private/package.json")
    if package["plan_sha256"] != plan["plan_sha256"] or digest({k:v for k,v in package.items() if k != "package_sha256"}) != package["package_sha256"]:
        raise ValueError("Package identity mismatch")
    return package


def score(args):
    """Score normalized copies, including derivatives, using the existing strict runner."""
    sys.path.insert(0, str(ROOT))
    from changing_resolution_uni.scripts.data.batch_vbench_score_dataset import (
        inspect_vbench_checkout, score_case_directory,
    )
    from UNIV_adaptor.scripts.data.score_phase2_dataset import output_lock
    plan = load_plan(args.out)
    package = load_package(args.out, plan)
    prompt_map = {}
    for pair in package["pairs"]:
        for clip in (pair["a"], pair["b"]):
            if clip + ".mp4" in prompt_map and prompt_map[clip + ".mp4"] != pair["prompt"]:
                raise ValueError("One clip has conflicting prompts")
            prompt_map[clip + ".mp4"] = pair["prompt"]
    for clip, record in package["clips"].items():
        if file_hash(args.out / f"media/{clip}.mp4") != record["sha256"]:
            raise ValueError(f"Media hash mismatch: {clip}")
    private = args.out / "private"
    with output_lock(private):
        identity = inspect_vbench_checkout(args.vbench_root, expected_commit=args.expected_vbench_commit)
        immutable(private / "presented_prompt_map.json", prompt_map)
        quality = ["subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality"]
        diagnostics = ["dynamic_degree", "overall_consistency"]
        scores = {clip: {} for clip in package["clips"]}
        provenance = {}
        for dimension in quality + diagnostics:
            bundle = score_case_directory(args.vbench_root, args.vbench_python, args.out / "media",
                private / "presented_prompt_map.json", private / "vbench" / dimension, [dimension],
                [dimension] if dimension in quality else [], [dimension] if dimension in diagnostics else [],
                args.ngpus, False, identity)
            if set(bundle.scores) != set(scores):
                raise ValueError("Presented-score coverage mismatch")
            for clip in scores:
                scores[clip][dimension] = bundle.scores[clip][dimension]
            provenance[dimension] = bundle.provenance
        rows = [{"clip_id": clip, "video_sha256": package["clips"][clip]["sha256"], **values,
                 "vbench5": statistics.mean(values[d] for d in quality)} for clip, values in scores.items()]
        immutable(private / "presented_scores.json", {"package_sha256": package["package_sha256"],
                  "vbench_identity": identity, "scores": rows, "provenance": provenance})
        csv_write(private / "presented_scores.csv", rows)
        print(private / "presented_scores.csv")


def serve(args):
    plan = load_plan(args.out)
    package = load_package(args.out, plan)
    for clip, record in package["clips"].items():
        if file_hash(args.out / f"media/{clip}.mp4") != record["sha256"]:
            raise ValueError(f"Media hash mismatch: {clip}")
    def ratings_path(participant):
        session(plan, package, participant)
        return args.out / f"private/ratings/{participant}.json"
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                data = Path(__file__).with_name("acceleration_blind_audit.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif re.fullmatch(r"/media/[a-f0-9]{24}\.mp4", path) and Path(path).stem in package["clips"]:
                file = args.out / path.lstrip("/")
                size, start, end = file.stat().st_size, 0, file.stat().st_size - 1
                header = self.headers.get("Range")
                if header:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
                    if not match:
                        self.send_error(416)
                        return
                    start = int(match[1])
                    end = min(int(match[2]) if match[2] else end, end)
                    if start > end:
                        self.send_error(416)
                        return
                self.send_response(206 if header else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if header:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                try:
                    with file.open("rb") as handle:
                        handle.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            block = handle.read(min(65536, remaining))
                            self.wfile.write(block)
                            remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 32768:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                participant = request["participant"]
                ordered = session(plan, package, participant)
                file = ratings_path(participant)
                saved = read(file) if file.exists() else {"participant": participant,
                    "plan_sha256": plan["plan_sha256"], "package_sha256": package["package_sha256"], "answers": {}}
                if saved["package_sha256"] != package["package_sha256"]:
                    raise ValueError("Ratings belong to another package")
                if self.path == "/api/session":
                    self.send_json({"pairs": [{k:v for k,v in p.items() if k != "swap"} for p in ordered],
                                    "answers": saved["answers"]})
                elif self.path == "/api/rate":
                    if request["pair"] not in {p["id"] for p in ordered}:
                        raise ValueError("Unknown pair")
                    votes = request["votes"]
                    if set(votes) != set(DIMENSIONS) or any(v not in CHOICES for v in votes.values()):
                        raise ValueError("All four dimensions require a valid vote")
                    saved["answers"][request["pair"]] = votes
                    write(file, saved)
                    self.send_json({"saved": True})
                else:
                    self.send_error(404)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)
    print(f"Open http://127.0.0.1:{args.port}; use SSH forwarding for remote access. Ctrl+C stops serving.", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


def report(args):
    plan = load_plan(args.out)
    package = load_package(args.out, plan)
    votes, raters = defaultdict(list), []
    for file in sorted((args.out / "private/ratings").glob("*.json")):
        data = read(file)
        if data["plan_sha256"] != plan["plan_sha256"] or data["package_sha256"] != package["package_sha256"]:
            raise ValueError(f"Ratings identity mismatch: {file}")
        if data["participant"] in raters:
            raise ValueError("Duplicate participant")
        raters.append(data["participant"])
        mapping = {p["id"]: p for p in session(plan, package, data["participant"])}
        for key, answers in data["answers"].items():
            for dim in DIMENSIONS:
                choice = answers[dim]
                if choice not in CHOICES:
                    raise ValueError("Invalid stored vote")
                if mapping[key]["swap"] and choice in ("A", "B"):
                    choice = "B" if choice == "A" else "A"
                votes[key, dim].append(choice)
    # Optional scores computed on the EXACT presented clips: strict hash binding.
    presented = {}
    score_path = args.presented_scores or args.out / "private/presented_scores.csv"
    if args.presented_scores and not score_path.exists():
        raise FileNotFoundError(score_path)
    if score_path.exists():
        for row in csv.DictReader(io.StringIO(score_path.read_text(encoding="utf-8-sig"))):
            clip = row["clip_id"]
            if clip in presented or clip not in package["clips"] or row["video_sha256"] != package["clips"][clip]["sha256"]:
                raise ValueError("Duplicate, unknown or changed presented clip")
            presented[clip] = {m: float(row[m]) for m in plan["config"]["metric_epsilon"] if row.get(m) not in (None, "")}
            if any(not math.isfinite(v) for v in presented[clip].values()):
                raise ValueError("Non-finite presented score")
    public_pairs = {p["id"]: p for p in package["pairs"]}
    human, comparisons = [], []
    minimum = plan["config"]["minimum_raters"]
    for pair in plan["pairs"]:
        for dim in DIMENSIONS:
            observed = votes[pair["id"], dim]
            counts = Counter(observed)
            consensus = "unresolved"
            if len(observed) >= minimum:
                for choice in ("A", "B", "tie"):
                    if counts[choice] / len(observed) >= 2 / 3:
                        consensus = choice
            valid = [v for v in observed if v != "uncertain"]
            matches = [a == b for a,b in itertools.combinations(valid, 2)]
            human.append({"pair": pair["id"], "kind": pair["kind"], "source": pair["source"],
                          "cluster": pair["cluster"], "dimension": dim, "raters": len(observed),
                          "A": counts["A"], "B": counts["B"], "tie": counts["tie"],
                          "uncertain": counts["uncertain"], "consensus": consensus,
                          "pairwise_agreement": statistics.mean(matches) if matches else None})
            if consensus not in ("A", "B"):
                continue
            for metric, epsilon in plan["config"]["metric_epsilon"].items():
                pp = public_pairs[pair["id"]]
                options = [("presented", presented.get(pp["a"], {}), presented.get(pp["b"], {}))]
                if pair["kind"] != "synthetic":
                    options.append(("original_exploratory", pair["a"]["scores"], pair["b"]["scores"]))
                for scope, a, b in options:
                    if metric not in a or metric not in b:
                        continue
                    delta = a[metric] - b[metric]
                    predicted = "tie" if abs(delta) <= epsilon else ("A" if delta > 0 else "B")
                    comparisons.append({"pair": pair["id"], "cluster": pair["cluster"],
                        "kind": pair["kind"], "source": pair["source"], "dimension": dim,
                        "metric": metric, "scope": scope, "human": consensus, "prediction": predicted,
                        "delta": delta, "correct": int(predicted == consensus),
                        "miss": int(predicted == "tie"), "reversed": int(predicted not in (consensus, "tie"))})
    target = args.out / "analysis"
    target.mkdir(exist_ok=True)
    csv_write(target / "human_consensus.csv", human)
    csv_write(target / "metric_pairs.csv", comparisons,
              ["pair", "cluster", "kind", "source", "dimension", "metric", "scope", "human", "prediction", "delta", "correct", "miss", "reversed"])
    buckets = defaultdict(list)
    for row in comparisons:
        buckets[row["kind"],row["source"],row["dimension"],row["metric"],row["scope"]].append(row)
    summary = []
    for key, rows in sorted(buckets.items()):
        clusters = defaultdict(list)
        for row in rows:
            clusters[row["cluster"]].append(row["correct"])
        means = [statistics.mean(v) for v in clusters.values()]
        ci = None
        if len(means) >= 2:
            rng = random.Random(20260930)
            boot = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(2000))
            ci = [boot[49], boot[1949]]
        summary.append(dict(zip(("kind","source","dimension","metric","scope"), key)) | {
            "pairs": len(rows), "prompt_clusters": len(means),
            "diagnostic_only": key[3] == "dynamic_degree",
            "pair_accuracy": statistics.mean(r["correct"] for r in rows),
            "prompt_macro_accuracy": statistics.mean(means), "prompt_bootstrap_ci": ci,
            "metric_tie_rate": statistics.mean(r["miss"] for r in rows),
            "reversal_rate": statistics.mean(r["reversed"] for r in rows)})
    result = {"claim_status": "pilot; no automatic claim of metric failure or router benefit",
              "primary_comparison": "presented VBench5 versus human overall preference on real pairs, separately by source; other comparisons exploratory",
              "plan_sha256": plan["plan_sha256"], "participants": len(raters),
              "minimum_raters": minimum, "fully_rated_pairs": sum(
                  all(len(votes[p["id"],d]) >= minimum for d in DIMENSIONS) for p in plan["pairs"]),
              "planned_pairs": len(plan["pairs"]), "metric_summary": summary,
              "caveats": ["Agreement excludes uncertain votes; consensus includes them in its denominator.",
                  "Human ties/unresolved pairs are excluded from directional metric accuracy; see human_consensus.csv.",
                  "Bootstrap resamples prompt clusters, conditional on observed raters and selected pairs.",
                  "Original scores are exploratory because presentation is normalized and re-encoded.",
                  "Synthetic metrics are missing unless presented_scores is supplied; original scores are never copied.",
                  "Dynamic degree is a motion diagnostic, not a universal higher-is-better quality criterion.",
                  "Different-seed controls can genuinely differ in quality; no equal-quality label is imposed."]}
    write(target / "report.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "package", "score", "serve", "report"])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/acceleration_blind_audit_v1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", action="append", default=[], help="NAME=directory_or_tgz, plan only")
    parser.add_argument("--path-map", action="append", default=[], help="OLD_PREFIX=NEW_PREFIX, package only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--presented-scores", type=Path, help="CSV: clip_id,video_sha256,metric columns")
    parser.add_argument("--vbench-root", type=Path, default=Path("/mnt/afs_2/houze/VBench"))
    parser.add_argument("--vbench-python", default="/opt/conda/bin/python")
    parser.add_argument("--expected-vbench-commit", default="fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490")
    parser.add_argument("--ngpus", type=int, default=8)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.vbench_root = args.vbench_root.resolve()
    if args.ngpus < 1:
        parser.error("--ngpus must be positive")
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
