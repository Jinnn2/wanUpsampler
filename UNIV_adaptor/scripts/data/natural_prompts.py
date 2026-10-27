"""Natural prompt set v1: real user prompts from VidProM, to test whether routing regions exist beyond the designed cells.

The controlled-factor prompts vary only motion and detail, so a router can only rediscover those two factors there.
This set is drawn from VidProM (Wang & Yang 2024; real prompts users wrote for Pika, CC BY-NC 4.0) without looking
at any generation outcome:

  candidates  download VidProM_semantic_unique.csv, keep English, 6-70 word, SFW (all six Detoxify probabilities
              <= 0.02 plus a keyword block list) prompts without URLs, handles or Pika flags; deduplicate; draw
              CANDIDATES prompts stratified over three length bins (seed SEED)
  screen      Qwen3-VL-32B text-only: P(renderable, safe scene) and the route_v1 motion/detail questions (unchanged)
  select      keep P(ok) >= 0.5 minus the manual EXCLUDE list, draw N per length bin in candidate order -> univ_natural_v1.jsonl
              (prompt ids 1000+, family = the prompt itself, split "natural", motion/detail = the screen's labels
              with cell_source "predicted")
  configs     generation configs for the platform (FULL50 STEP20 TEA015 SCALING10) and two-stage (LR_FLASH F41_RIFE
              RCM4 LR41_FLASH_RIFE) drivers, one seed, built from the dev2 configs with only prompts/seeds/arms changed

Usage: see run_univ_natural_8gpu.sh.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402

REPO_ID, FILE = "WenhaoWang/VidProM", "VidProM_semantic_unique.csv"
NSFW = ("toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit")
NSFW_MAX = 0.02
BLOCK = re.compile(r"\b(nude|naked|nsfw|sex\w*|porn\w*|boob\w*|breast\w*|nipple\w*|bikini|lingerie|erotic\w*|"
                   r"blood\w*|gore|corpse|dead body|suicide|murder\w*|rape|torture|cocaine|nazi|hitler|terroris\w*|"
                   r"child|children|kid|kids|loli\w*|teen\w*|underage)\b", re.I)
JUNK = re.compile(r"https?://|www\.|@|#|<|>|\{|\}|\[|\]|\battachment\b|\bmessage:", re.I)
FLAG = re.compile(r"\s-{1,2}[a-z]+(\s|$)", re.I)  # Pika parameters (-gs 12 -motion 2 -ar 16:9 -neg ...) end a prompt
# Pika parameters left inside the text after the first cut (-fps24 -motion0, ,--ar 16:9, --ar3:2, (ar-16:9), ,-camera)
PARAM = re.compile(r"\(ar-\d+:\d+\)|-{1,2}(?:ar|gs|motion|fps|seed|neg)\s*\d+(?:\s*:\s*\d*)?", re.I)
CAMERA_FLAG = re.compile(r"(?<!\w)-{1,2}camera\b", re.I)
# Manual review of the first draw (2026-10-27), before any generation; the next candidate in the bin takes the slot.
EXCLUDE = {
    "073c59f5-7dbd-5ea8-8be8-3199841f9987": "named real person (Walt Disney)",
    "06914f78-2e37-5971-82f0-88b53bb27b7a": "swimsuit focus on a character who is a minor in canon (Spider-Gwen)",
    "c815e358-665e-58a2-b3c5-53882bb4d175": "likeness of a named real actress",
    "00fc5a95-1c0f-587e-aa31-03d3786e5a9b": "named real person (Joe Coulombe)",
    "5358894f-b5ff-5891-99f8-6adf713d240c": "sexualized body imagery",
}
BINS = ((6, 12), (12, 25), (25, 71))  # word-count bins [lo, hi)
CANDIDATES = 768
N = 256
SEED = 20261027
FIRST_ID = 1000
OK_QUESTION = ("Here is a text prompt a user wrote for a text-to-video generator:\n\"{prompt}\"\n"
               "Is this a safe-for-work, coherent description of a visual scene that can be shown as a short video - "
               "not mainly text or a logo, not a meme or a named real person, and not gibberish? "
               "Answer with only Yes or No.")
PLATFORM_ARMS = ("FULL50", "STEP20", "TEA015", "SCALING10")
TWOSTAGE_ARMS = ("FULL50", "LR_FLASH", "F41_RIFE", "RCM4", "LR41_FLASH_RIFE")
SEEDS = [42]


def clean(text):
    text = " ".join(html.unescape(text).split())  # VidProM stores quotes as &quot;
    m = FLAG.search(text)
    if m:
        text = text[:m.start()]
    return text.strip(" ,;")


def tidy(text):
    """Second pass on drawn candidates only (eligibility and the draw use clean()): drop leftover Pika parameters."""
    text = CAMERA_FLAG.sub("camera", PARAM.sub(" ", text))
    text = re.sub(r",(?:\s*,)+", ",", " ".join(text.split()))  # commas left around a removed parameter
    return text.strip(" ,;-")


def keep(text, row):
    words = text.split()
    if not BINS[0][0] <= len(words) < BINS[-1][1] or JUNK.search(text) or BLOCK.search(text):
        return False
    if not text.isascii() or sum(c.isalpha() for c in text) < 0.7 * len(text.replace(" ", "")):
        return False
    return all(float(row[k]) <= NSFW_MAX for k in NSFW)


def bin_of(text):
    n = len(text.split())
    return next(i for i, (lo, hi) in enumerate(BINS) if lo <= n < hi)


def candidates(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(args.csv) if args.csv else None
    if path is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(REPO_ID, FILE, repo_type="dataset", local_dir=out / "vidprom"))
    seen, pools, total = set(), [[] for _ in BINS], 0
    csv.field_size_limit(1 << 20)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"uuid", "prompt", *NSFW} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path.name} lacks columns {sorted(missing)}; has {reader.fieldnames}")
        for row in reader:
            total += 1
            text = clean(row["prompt"])
            key = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
            head = " ".join(key.split()[:6])
            if not keep(text, row) or key in seen or head in seen:
                continue
            seen.update((key, head))
            pools[bin_of(text)].append({"uuid": row["uuid"], "prompt": text, "raw_prompt": row["prompt"]})
    rng = random.Random(SEED)
    per = CANDIDATES // len(BINS)
    chosen = []
    for b, pool in enumerate(pools):
        pool.sort(key=lambda r: r["uuid"])  # file order is not a promise; uuid order is
        chosen += [{**r, "prompt": tidy(r["prompt"]), "length_bin": b} for r in rng.sample(pool, per)]
    (out / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    write_json_atomic(out / "candidates_meta.json", {
        "source": f"{REPO_ID}/{FILE}", "csv_sha256": sha256_file(path), "rows": total,
        "eligible_per_bin": [len(p) for p in pools], "bins": BINS, "nsfw_max": NSFW_MAX, "seed": SEED,
        "candidates": len(chosen)})
    print(f"{total} rows, eligible per length bin {[len(p) for p in pools]} -> {len(chosen)} candidates")


def screen(args):
    from UNIV_adaptor.scripts.data import route_v1 as rt
    from UNIV_adaptor.scripts.data.score_dynamic import Judge

    out = Path(args.out)
    rows = [json.loads(line) for line in (out / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = {"ok": OK_QUESTION, **rt.QUESTIONS}
    judge = Judge(args.model)
    result = {}
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        ps = {k: judge.p_yes([[q.format(prompt=r["prompt"])] for r in chunk]) for k, q in questions.items()}
        for j, r in enumerate(chunk):
            result[r["uuid"]] = {f"p_{k}": round(float(ps[k][j]), 5) for k in questions}
        print(f"[screen] {min(i + args.batch, len(rows))}/{len(rows)}", flush=True)
    write_json_atomic(out / "screen.json", {"model": str(args.model), "questions": questions, "prompts": result})


def select(args):
    from UNIV_adaptor.scripts.data import route_v1 as rt

    out = Path(args.out)
    rows = [json.loads(line) for line in (out / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    value = json.loads((out / "screen.json").read_text(encoding="utf-8"))
    if value["questions"] != {"ok": OK_QUESTION, **rt.QUESTIONS}:
        raise ValueError("screen.json was made with different questions; rerun screen")
    s = value["prompts"]
    ok = [r for r in rows if s[r["uuid"]]["p_ok"] >= 0.5 and r["uuid"] not in EXCLUDE]
    print(f"manually excluded: {sum(r['uuid'] in EXCLUDE for r in rows)} of {len(EXCLUDE)} listed")
    per = N // len(BINS) + 1
    picked = []
    for b in range(len(BINS)):
        picked += [r for r in ok if r["length_bin"] == b][:per]  # candidates are already in random order
    picked = picked[:N]
    if len(picked) < N:
        raise ValueError(f"only {len(picked)} prompts pass the screen; raise CANDIDATES")
    records = []
    for i, r in enumerate(picked):
        p = s[r["uuid"]]
        records.append({"prompt_id": FIRST_ID + i, "family_id": f"nat{FIRST_ID + i}", "split": "natural",
                        "motion_level": "high" if p["p_motion"] >= 0.5 else "low",
                        "detail_level": "high" if p["p_detail"] >= 0.5 else "low", "cell_source": "predicted",
                        "prompt": r["prompt"], "vidprom_uuid": r["uuid"], "length_bin": r["length_bin"],
                        **{k: p[k] for k in ("p_ok", "p_motion", "p_detail")}})
    args.prompts_out = args.prompts_out or str(out / "univ_natural_v1.jsonl")
    Path(args.prompts_out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    cells = {}
    for r in records:
        c = r["motion_level"][0] + r["detail_level"][0]
        cells[c] = cells.get(c, 0) + 1
    print(f"{len(ok)}/{len(rows)} candidates pass the screen; wrote {len(records)} prompts to {args.prompts_out}")
    print(f"predicted cells {dict(sorted(cells.items()))}; per length bin "
          f"{[sum(r['length_bin'] == b for r in records) for b in range(len(BINS))]}")
    rejected = [r["prompt"] for r in rows if s[r["uuid"]]["p_ok"] < 0.5][:15]
    print("examples rejected by the screen:\n  " + "\n  ".join(rejected))


def configs(args):
    prompts = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = [{"id": f"nt{p['prompt_id']}", "origin": "natural_vidprom_v1", "motion": p["motion_level"],
                "detail": p["detail_level"], "family_id": p["family_id"], "split": p["split"], "prompt": p["prompt"]}
               for p in prompts]
    note = (f"The {len(entries)} prompts come from prompts/univ_natural_v1.jsonl: real VidProM user prompts "
            "(CC BY-NC 4.0) drawn by natural_prompts.py without looking at any outcome. motion/detail are Qwen3-VL "
            "zero-shot predictions (route_v1 questions), used only as nominal cells.")
    cfg_dir = ROOT / "UNIV_adaptor/configs"
    plat = json.loads((cfg_dir / "published_wan21_platform_dev2.json").read_text(encoding="utf-8"))
    plat["claim"] = ("Natural prompt set v1: frontier platform accelerators on real user prompts, one seed, as a test "
                     "of whether prompt->method regions extend beyond the designed motion/detail cells.")
    plat["seeds"] = SEEDS
    plat["disabled_arms"] += [a for a in plat["arms"] if a["id"] not in PLATFORM_ARMS]
    plat["arms"] = [a for a in plat["arms"] if a["id"] in PLATFORM_ARMS]
    plat["prompt_source"]["description"] = note
    plat["prompts"] = entries
    two = json.loads((cfg_dir / "published_wan21_twostage_dev2.json").read_text(encoding="utf-8"))
    two["claim"] = ("Natural prompt set v1: frontier two-stage and distillation arms on real user prompts, one seed. "
                    "FULL50 is reused from the platform-natural run on the same machine.")
    two["seeds"] = SEEDS
    two["arms"] = [a for a in two["arms"] if a["id"] in TWOSTAGE_ARMS]
    used = {a["generation"] for a in two["arms"]}
    two["generations"] = [g for g in two["generations"] if g["id"] in used]
    two["prompt_source"]["description"] = note
    two["prompts"] = entries
    two["reference"]["expected_root"] = "outputs/published_wan21_platform_natural"
    for name, cfg in (("published_wan21_platform_natural", plat), ("published_wan21_twostage_natural", two)):
        Path(args.config_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.config_dir) / f"{name}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote configs/{name}.json: arms {[a['id'] for a in cfg['arms']]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("candidates")
    p.add_argument("--out", required=True)
    p.add_argument("--csv", help="local copy of the VidProM csv (default: download from the Hub)")
    p = sub.add_parser("screen")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, default=16)
    p = sub.add_parser("select")
    p.add_argument("--out", required=True)
    p.add_argument("--prompts-out", help="default: OUT/univ_natural_v1.jsonl (committed to prompts/ after review)")
    p = sub.add_parser("configs")
    p.add_argument("--prompts", required=True)
    p.add_argument("--config-dir", required=True, help="where to write the two configs (committed to configs/ after review)")
    args = parser.parse_args(argv)
    {"candidates": candidates, "screen": screen, "select": select, "configs": configs}[args.mode](args)


if __name__ == "__main__":
    main()
