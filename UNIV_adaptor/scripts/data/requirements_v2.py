"""Requirement checklist v2: schema, judge routing and validation (see UNIV_adaptor/REQUIREMENTS_V2_SPEC.md).

Each prompt gets atomic yes/no questions (yes = requirement satisfied), each tied to an exact span of the
prompt, with dependencies between them. The question type alone decides which judge scores it.

Usage:
  python UNIV_adaptor/scripts/data/requirements_v2.py validate --checklist prompts/univ_requirements_v2_pilot.jsonl \
      --prompts prompts/univ_controlled_factor_v1.jsonl prompts/univ_controlled_factor_dev2.jsonl
  python .../requirements_v2.py generate --prompts ... --examples prompts/univ_requirements_v2_pilot.jsonl \
      --model /path/to/Qwen3-VL-32B-Instruct --out outputs/requirements_v2_gen [--only 1 11 12 ...]
  python .../requirements_v2.py collect --prompts ... --out outputs/requirements_v2_gen
  python .../requirements_v2.py compare --generated outputs/requirements_v2_gen/checklist.jsonl \
      --reference prompts/univ_requirements_v2_pilot.jsonl --prompts ...
"""
import argparse, collections, json, re, sys
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


# ---------------------------------------------------------------- generate (M1)

# Spec sections 1-2, written for the question-writing model.
RULES = """You write evaluation checklists for text-to-video prompts. A checklist is a list of atomic yes/no
questions about the generated video; answering Yes means the requirement is satisfied.

Fields of each question:
- id: q0, q1, q2, ... in order.
- type: one of the types below. The type alone decides how the question is judged.
- question: one English yes/no question ending with "?". Phrase negative constraints so that Yes means
  satisfied, e.g. "Is the frame free of human hands?".
- evidence: a span copied character-for-character from the prompt, naming the requirement the question checks.
- depends_on: ids of earlier questions that must hold for this one to make sense (e.g. an attribute question
  depends on its entity question). Use [] when there is none.
- region, expect, grounding: see motion, static, camera and integrity below.

Types:
- entity: the main subject or an object is present.
- count: number or quantity ("a single", "dozens of", "hundreds of").
- attribute: color, material or shape of something.
- detail: small structures are clearly visible (fine patterns, thin lines, crisp detail).
- spatial: position, framing, viewpoint, occlusion, filling the frame.
- negative: something must be absent (no other objects, no text, no patterns, no hands).
- action: a semantic action (swinging, tumbling, circling, folding).
- integrity: a moving object keeps a coherent shape (does not melt, smear or split). Implied for every moving
  subject even if the prompt does not say it; evidence is the noun phrase of that object; give grounding.
- motion: amount or speed of motion ("fast", "sweeps across most of the frame"); expect is "moving" or "fast".
- static: a subject or the whole scene stays still; expect is "still".
- camera: the camera stays locked; expect is "still", region is "background" or "global".
- focus: focus stays constant.
- lighting: lighting stays constant.

motion, static and camera questions need region ("subject", "background" or "global") and expect. When region
is "subject", also give grounding: a short noun phrase naming what to segment (e.g. "the brass pendulum").

Rules:
- Only ask about requirements the prompt states explicitly; never ask about aesthetics. The one exception is
  integrity (see above).
- One question checks one thing. "A silver pocket watch filling the frame" becomes an attribute question and
  a spatial question.
- Every negative or stillness constraint gets a question ("no other objects", "nothing moves", "the lighting
  never changes"). These are what acceleration methods break most often.
- If the prompt mentions the camera (camera, locked, stationary), include a camera question.
- If the prompt describes a still scene, include a static question. If it describes motion, include a motion
  or action question.
- Typically 7 to 12 questions. Do not ask about the same requirement twice.

Output exactly one JSON object {"questions": [...]} and nothing else."""

EXAMPLE_KEYS = ("id", "type", "question", "evidence", "depends_on", "region", "expect", "grounding")


def user_message(prompt):
    return f"Prompt:\n{prompt['prompt']}\n\nWrite the checklist."


def build_messages(prompt, examples, prompts):
    """Chat messages: rules, each example as a user/assistant turn, then the target prompt."""
    msgs = [{"role": "system", "content": RULES}]
    for ex in examples:
        qs = [{k: q[k] for k in EXAMPLE_KEYS if k in q} for q in ex["questions"]]
        msgs += [{"role": "user", "content": user_message(prompts[ex["prompt_id"]])},
                 {"role": "assistant", "content": json.dumps({"questions": qs}, ensure_ascii=False)}]
    msgs.append({"role": "user", "content": user_message(prompt)})
    return msgs


def parse_reply(text):
    """The first JSON object in the reply (code fences tolerated), or None."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and isinstance(obj.get("questions"), list) else None


def generate_one(prompt, examples, prompts, chat, repairs=2):
    """Draft, validate, and feed the errors back up to `repairs` times. Keeps every attempt for audit."""
    msgs = build_messages(prompt, examples, prompts)
    attempts, rec = [], None
    for _ in range(repairs + 1):
        reply = chat(msgs)
        obj = parse_reply(reply)
        if obj is None:
            errs = ["reply is not a JSON object with a 'questions' list"]
        else:
            # Prompt metadata comes from the prompt file, never from the model.
            rec = {"prompt_id": prompt["prompt_id"], "family_id": prompt["family_id"],
                   "motion_level": prompt["motion_level"], "detail_level": prompt["detail_level"],
                   "questions": obj["questions"]}
            try:
                errs = validate_record(rec, prompt)
            except (AttributeError, TypeError) as exc:  # e.g. a question that is not an object
                errs = [f"malformed questions: {exc}"]
        attempts.append({"reply": reply, "errors": errs})
        if not errs:
            break
        msgs = msgs + [{"role": "assistant", "content": reply},
                       {"role": "user", "content": "The checklist has these problems:\n- " + "\n- ".join(errs)
                        + "\nOutput the full corrected JSON object only."}]
    return {"prompt_id": prompt["prompt_id"], "ok": not errs, "record": rec, "attempts": attempts}


class TextChat:
    """Greedy text-only chat with a Hugging Face (vision-)language model, sharded over the visible GPUs."""

    def __init__(self, model_id, max_new_tokens=2048):
        import torch
        import transformers

        self.torch = torch
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        cls = getattr(transformers, "AutoModelForImageTextToText", transformers.AutoModelForCausalLM)
        self.model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages):
        # The chat template lives on the processor for Qwen3-VL; multimodal processors want list-form content.
        msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def load_prompts(paths):
    return {p["prompt_id"]: p for path in paths for p in load_jsonl(path)}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def generate(args, chat=None):
    prompts = load_prompts(args.prompts)
    examples = [r for r in load_jsonl(args.examples) if r["family_id"] in args.example_family]
    if not examples:
        raise SystemExit(f"no examples of family {args.example_family} in {args.examples}")
    todo = [p for pid, p in sorted(prompts.items()) if p["split"] != "test" and (not args.only or pid in args.only)]
    raw = Path(args.out) / "raw"
    todo = [p for p in todo[args.shard_index::args.shard_count] if not (raw / f"{p['prompt_id']}.json").exists()]
    if not todo:
        return 0
    chat = chat or TextChat(args.model, args.max_new_tokens)
    for i, p in enumerate(todo):
        result = generate_one(p, examples, prompts, chat, args.repairs)
        result.update(model=str(args.model), examples=[e["prompt_id"] for e in examples])
        write_json(raw / f"{p['prompt_id']}.json", result)
        print(f"[generate {args.shard_index}/{args.shard_count}] {i + 1}/{len(todo)} p{p['prompt_id']} "
              f"{'ok' if result['ok'] else 'FAILED'} after {len(result['attempts'])} attempt(s)", flush=True)
    return 0


def collect(args):
    """Merge raw/*.json into checklist.jsonl (valid records only) and list the prompts that need a human."""
    prompts = load_prompts(args.prompts)
    out = Path(args.out)
    results = sorted((json.loads(f.read_text(encoding="utf-8")) for f in (out / "raw").glob("*.json")),
                     key=lambda r: r["prompt_id"])
    ok = [r["record"] for r in results if r["ok"]]
    (out / "checklist.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ok),
                                         encoding="utf-8")
    done = {r["prompt_id"] for r in results}
    missing = sorted(pid for pid, p in prompts.items() if p["split"] != "test" and pid not in done)
    tries = collections.Counter(len(r["attempts"]) for r in results if r["ok"])
    print(f"{len(ok)} valid, {len(results) - len(ok)} failed, {len(missing)} not generated")
    print("attempts needed: " + ", ".join(f"{k}: {v}" for k, v in sorted(tries.items())))
    for r in results:
        if not r["ok"]:
            print(f"FAILED p{r['prompt_id']}: " + "; ".join(r["attempts"][-1]["errors"]))
    if missing:
        print("not generated: " + " ".join(map(str, missing)))
    if ok:
        validate(argparse.Namespace(checklist=str(out / "checklist.jsonl"), prompts=args.prompts))
    return 0


def span(text, evidence):
    i = text.find(evidence)
    return (i, i + len(evidence)) if i >= 0 else None


def overlaps(a, b):
    return a is not None and b is not None and a[0] < b[1] and b[0] < a[1]


def compare(args):
    """Recall of reviewed reference questions by generated ones (same type, overlapping evidence), plus the
    generated questions that match nothing, for a human to mark."""
    prompts = load_prompts(args.prompts)
    gen = {r["prompt_id"]: r for r in load_jsonl(args.generated)}
    ref = [r for r in load_jsonl(args.reference) if r["family_id"] not in args.exclude_family]
    n_t, hit_t, hit_any, missed, extras = collections.Counter(), collections.Counter(), 0, [], []
    for r in ref:
        text = prompts[r["prompt_id"]]["prompt"]
        gq = gen.get(r["prompt_id"], {}).get("questions", [])
        gs = [(q["type"], span(text, q["evidence"])) for q in gq]
        rs = [(q["type"], span(text, q["evidence"])) for q in r["questions"]]
        for q, (t, s) in zip(r["questions"], rs):
            n_t[t] += 1
            hit = any(gt == t and overlaps(s, g) for gt, g in gs)
            hit_t[t] += hit
            hit_any += any(overlaps(s, g) for _, g in gs)
            if not hit:
                missed.append(f"| p{r['prompt_id']} | {t} | {q['question']} | {q['evidence']} |")
        for q, (t, s) in zip(gq, gs):
            if not any(rt == t and overlaps(s, x) for rt, x in rs):
                extras.append(f"| p{r['prompt_id']} | {t} | {q['question']} | {q['evidence']} | |")
    n, hits = sum(n_t.values()), sum(hit_t.values())
    lines = ["# Generated vs reviewed checklist", "",
             f"Reference prompts: {len(ref)} (example families excluded: {', '.join(args.exclude_family) or 'none'}); "
             f"generated: {sum(r['prompt_id'] in gen for r in ref)}.", "",
             f"- Recall, same type and overlapping evidence: {hits}/{n} = {hits / max(n, 1):.1%}",
             f"- Recall, overlapping evidence of any type: {hit_any}/{n} = {hit_any / max(n, 1):.1%}",
             f"- Generated questions matching no reference question: {len(extras)}", "",
             "| type | recall |", "| --- | --- |"]
    lines += [f"| {t} | {hit_t[t]}/{c} |" for t, c in n_t.most_common()]
    lines += ["", "## Missed reference questions", "", "| prompt | type | question | evidence |",
              "| --- | --- | --- | --- |", *missed]
    lines += ["", "## Extra generated questions (mark each: reasonable / wrong)", "",
              "| prompt | type | question | evidence | verdict |", "| --- | --- | --- | --- | --- |", *extras]
    report = "\n".join(lines) + "\n"
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    print(report)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--checklist", required=True)
    v.add_argument("--prompts", nargs="+", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompts", nargs="+", required=True)
    g.add_argument("--examples", required=True, help="reviewed checklist to take few-shot examples from")
    g.add_argument("--example-family", nargs="+", default=["watchmaking"])
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--only", nargs="*", type=int, default=[], help="prompt ids (default: every non-test prompt)")
    g.add_argument("--repairs", type=int, default=2)
    g.add_argument("--max-new-tokens", type=int, default=2048)
    g.add_argument("--shard-index", type=int, default=0)
    g.add_argument("--shard-count", type=int, default=1)
    c = sub.add_parser("collect")
    c.add_argument("--prompts", nargs="+", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--generated", required=True)
    m.add_argument("--reference", required=True)
    m.add_argument("--prompts", nargs="+", required=True)
    m.add_argument("--exclude-family", nargs="*", default=["watchmaking"], help="families used as examples")
    m.add_argument("--report")
    args = ap.parse_args(argv)
    return {"validate": validate, "generate": generate, "collect": collect, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
