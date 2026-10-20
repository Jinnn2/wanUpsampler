"""Requirement checklist v2: schema, judge routing and validation (see UNIV_adaptor/REQUIREMENTS_V2_SPEC.md).

Each prompt gets atomic yes/no questions (yes = requirement satisfied), each tied to an exact span of the
prompt, with dependencies between them. The question type alone decides which judge scores it.

Usage:
  python UNIV_adaptor/scripts/data/requirements_v2.py validate --checklist prompts/univ_requirements_v2_pilot.jsonl \
      --prompts prompts/univ_controlled_factor_v1.jsonl prompts/univ_controlled_factor_dev2.jsonl
"""
import argparse, collections, json, sys
from pathlib import Path

# type -> judge. vlm_frames: native-resolution stills; vlm_video: dense clip; flow: RAFT on a grounded region;
# luma: brightness statistics over time; sharpness: per-frame sharpness over time.
JUDGE = {
    "entity": "vlm_frames", "count": "vlm_frames", "attribute": "vlm_frames", "detail": "vlm_frames",
    "spatial": "vlm_frames", "negative": "vlm_frames", "integrity": "vlm_video", "action": "vlm_video",
    "motion": "flow", "static": "flow", "camera": "flow", "focus": "sharpness", "lighting": "luma",
}
# Questions a VLM judges are also asked pairwise against the same-seed FULL video.
PAIRWISE = {t for t, j in JUDGE.items() if j.startswith("vlm")}
REGIONS = {"subject", "background", "global"}
EXPECT = {"motion": {"moving", "fast"}, "static": {"still"}, "camera": {"still"}}


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_record(rec, prompt):
    """Return a list of problems for one checklist record (empty = valid)."""
    errs, seen = [], set()
    pid = rec.get("prompt_id")
    if prompt is None:
        return [f"p{pid}: no such prompt"]
    if prompt["split"] == "test":
        return [f"p{pid}: locked test prompt must not get a checklist yet"]
    for key in ("family_id", "motion_level", "detail_level"):
        if rec.get(key) != prompt[key]:
            errs.append(f"p{pid}: {key} {rec.get(key)!r} != prompt {prompt[key]!r}")
    questions = rec.get("questions") or []
    if not questions:
        errs.append(f"p{pid}: no questions")
    for i, q in enumerate(questions):
        tag = f"p{pid}/{q.get('id')}"
        if q.get("id") != f"q{i}":
            errs.append(f"{tag}: ids must run q0, q1, ... in order")
        t = q.get("type")
        if t not in JUDGE:
            errs.append(f"{tag}: unknown type {t!r}")
            continue
        text = q.get("question", "")
        if not text.endswith("?"):
            errs.append(f"{tag}: question must end with '?'")
        if q.get("evidence", "") not in prompt["prompt"] or not q.get("evidence"):
            errs.append(f"{tag}: evidence is not an exact span of the prompt: {q.get('evidence')!r}")
        for d in q.get("depends_on", []):
            if d not in seen:
                errs.append(f"{tag}: depends_on {d} is not an earlier question")
        if JUDGE[t] == "flow":
            if q.get("region") not in REGIONS:
                errs.append(f"{tag}: flow question needs region in {sorted(REGIONS)}")
            if q.get("expect") not in EXPECT[t]:
                errs.append(f"{tag}: {t} needs expect in {sorted(EXPECT[t])}")
            if q.get("region") == "subject" and not q.get("grounding"):
                errs.append(f"{tag}: subject region needs a grounding phrase")
        if t == "integrity" and not q.get("grounding"):
            errs.append(f"{tag}: integrity needs a grounding phrase")
        seen.add(q.get("id"))
    types = {q.get("type") for q in questions}
    text = prompt["prompt"].lower()
    if "camera" in text or "locked" in text or "stationary" in text:
        if "camera" not in types:
            errs.append(f"p{pid}: prompt constrains the camera but no camera question")
    if prompt["motion_level"] == "high" and not types & {"motion", "action"}:
        errs.append(f"p{pid}: high-motion prompt without a motion/action question")
    if prompt["motion_level"] == "low" and "static" not in types:
        errs.append(f"p{pid}: low-motion prompt without a static question")
    return errs


def validate(args):
    prompts = {}
    for path in args.prompts:
        for p in load_jsonl(path):
            prompts[p["prompt_id"]] = p
    records = load_jsonl(args.checklist)
    errs, ids = [], collections.Counter(r.get("prompt_id") for r in records)
    errs += [f"p{pid}: duplicated" for pid, n in ids.items() if n > 1]
    for rec in records:
        errs += validate_record(rec, prompts.get(rec.get("prompt_id")))
    types = collections.Counter(q["type"] for r in records for q in r["questions"])
    judges = collections.Counter(JUDGE[t] for t in types.elements() if t in JUDGE)
    n = sum(types.values())
    print(f"{len(records)} prompts, {n} questions ({n / max(len(records), 1):.1f} per prompt)")
    print("by type: " + ", ".join(f"{t} {c}" for t, c in types.most_common()))
    print("by judge: " + ", ".join(f"{j} {c}" for j, c in judges.most_common()))
    for e in errs:
        print("ERROR", e)
    return 1 if errs else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--checklist", required=True)
    v.add_argument("--prompts", nargs="+", required=True)
    args = ap.parse_args(argv)
    return validate(args)


if __name__ == "__main__":
    sys.exit(main())
