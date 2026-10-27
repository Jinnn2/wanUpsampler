"""Copy the summary files of a server result tgz into results/<name>/.

    python results/ingest.py E:/Downloads/natural_v1_results.tgz            # name = tgz stem minus "_analysis"
    python results/ingest.py E:/Downloads/foo_analysis.tgz --name layers/foo

Keeps small text files (csv/json/jsonl/md/txt/log) and skips per-item folders (cache, records, vlm, pair, raw ...),
media, embeddings and large plans; nested .tgz files are unpacked into subfolders. Writes SOURCE.json with the
archive's sha256 and the kept file list, so the server path can be traced back. Re-running replaces the folder.
"""
import argparse
import datetime
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

RESULTS = Path(__file__).resolve().parent
TEXT = {".csv", ".json", ".jsonl", ".md", ".txt", ".log"}
SKIP_DIRS = {"cache", "records", "logs", "vlm", "pair", "signals", "receipts", "raw", "vbench", "inputs", "videos",
             "render", "frames", "clips"}
SKIP_NAMES = {"dataset_manifest.json", "plan.json", "evaluation_inputs.json"}  # path manifests; stay on the server
MAX_BYTES = 8 << 20  # dynamic_by_question.csv (12 MB) is not kept


def keep(path, size):
    p = PurePosixPath(path)
    return (p.suffix in TEXT and size <= MAX_BYTES and p.name not in SKIP_NAMES
            and not SKIP_DIRS.intersection(p.parts[:-1]))


def strip_root(names):
    """Drop shared top folders (e.g. "requirements_v2_gen/full/") so files land directly in the target."""
    parts = [[x for x in PurePosixPath(n).parts if x != "."] for n in names]
    drop = 0
    while parts and len({p[drop] for p in parts}) == 1 and all(len(p) > drop + 1 for p in parts):
        drop += 1
    return lambda n: str(PurePosixPath(*[x for x in PurePosixPath(n).parts if x != "."][drop:]))


def extract(tar, out, kept, skipped, prefix=""):
    members = [m for m in tar.getmembers() if m.isfile()]
    rel = strip_root([m.name for m in members])
    for m in members:
        name = rel(m.name)
        if name.endswith(".tgz") or name.endswith(".tar.gz"):
            sub = PurePosixPath(name)
            sub = str(sub.with_name(sub.name.split(".")[0].removesuffix("_analysis")))
            with tarfile.open(fileobj=io.BytesIO(tar.extractfile(m).read()), mode="r:gz") as inner:
                extract(inner, out / sub, kept, skipped, f"{prefix}{sub}/")
        elif keep(name, m.size):
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(m).read())
            kept.append(prefix + name)
        else:
            skipped[PurePosixPath(name).suffix or "(none)"] = skipped.get(PurePosixPath(name).suffix or "(none)", 0) + 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tgz")
    ap.add_argument("--name", help="target folder under results/ (default: tgz stem minus _analysis)")
    ap.add_argument("--note", default="", help="one line stored in SOURCE.json")
    args = ap.parse_args(argv)
    src = Path(args.tgz)
    name = args.name or src.name.split(".")[0].removesuffix("_analysis")
    out = RESULTS / name
    if out.resolve() == RESULTS or RESULTS not in out.resolve().parents:
        sys.exit(f"bad target {out}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    kept, skipped = [], {}
    with tarfile.open(src, "r:gz") as tar:
        extract(tar, out, kept, skipped)
    (out / "SOURCE.json").write_text(json.dumps({
        "archive": src.name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "ingested": datetime.date.today().isoformat(), "note": args.note,
        "kept": sorted(kept), "skipped_by_suffix": skipped}, indent=1, ensure_ascii=False), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{src.name} -> results/{name}: {len(kept)} files, {size / 1e6:.1f} MB; skipped {skipped}")


if __name__ == "__main__":
    main()
