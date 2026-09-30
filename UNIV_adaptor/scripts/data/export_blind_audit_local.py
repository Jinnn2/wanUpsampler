"""Export the rendered study for offline/local annotation; no model or re-encoding."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

if __package__:
    from .acceleration_blind_audit import ROOT, digest, file_hash, immutable, load_package, load_plan, read
else:
    from acceleration_blind_audit import ROOT, digest, file_hash, immutable, load_package, load_plan, read


LAUNCHER = '''from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from types import SimpleNamespace

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "UNIV_adaptor/scripts/data"))
from acceleration_blind_audit import serve_local

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
url = f"http://127.0.0.1:{port}/"
def open_when_ready():
    for _ in range(600):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(.5)
threading.Thread(target=open_when_ready, daemon=True).start()
print("Verifying videos, then opening your browser. Keep this window open.", flush=True)
try:
    serve_local(SimpleNamespace(out=root / "study", port=port))
except KeyboardInterrupt:
    print("Stopped. Saved answers remain in study/private/ratings.")
'''

CMD = '''@echo off
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
  py -3 run_local.py
) else (
  python run_local.py
)
pause
'''

README = '''本机视频盲评包

1. 完整解压 ZIP 到普通文件夹，不要在压缩包中直接运行。
2. Windows 安装 Python 3.10 或更高版本后，双击 START_WINDOWS.cmd。
   本包不需要额外 Python 库、GPU、VBench 或 ffmpeg。
   也可以在本目录执行：python run_local.py
3. 校验视频后会自动打开浏览器；请保持命令行窗口运行。
4. 每位评价者使用唯一编号，例如 rater01。重新输入相同编号可继续。
5. 答案自动保存在 study/private/ratings/。结束后把该目录下的 JSON
   文件交回研究者，在原服务器的同名目录汇总。请勿覆盖他人同名文件。

可以离线运行，无需 VS Code Tunnel。不要直接双击 HTML 页面。
不含原评分、方法名或原视频路径；请勿自行尝试从匿名标识推断方法。
这是人工标注包，评分和统计仍在原研究目录进行。
'''


ANALYSIS = '''from pathlib import Path
import sys
from types import SimpleNamespace
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "UNIV_adaptor/scripts/data"))
from acceleration_blind_audit import report
report(SimpleNamespace(out=root / "study", presented_scores=None))
'''

RESEARCH_README = '''研究者完整数据包（包含未盲化的信息，不要直接分发给评价者）

本包保留全部展示视频、private/ 中的清单、原评分、新评分、VBench 原始
评分记录和已有标注，以及 study 目录中的其他结果文件。
绝对服务器路径仅用于追溯，本机盲评和统计不依赖它们。

完整解压后，双击 START_WINDOWS.cmd 启动本机盲评。
双击 ANALYZE_WINDOWS.cmd 汇总现有标注与评分。
两者只需 Python 3.10+，不需要额外库或 GPU。

未达到三人评价的配对不会被当作已验证的人工共识。
分数已完成不代表人工验证已完成。结果写入 study/analysis/。
如果需要给其他评价者发送数据，请单独使用不加 --research 的匿名导出。
'''


def export(study, destination, research=False, metadata_only=False):
    if metadata_only and not research:
        raise ValueError("Metadata-only export requires research mode")
    study, destination = Path(study).resolve(), Path(destination).resolve()
    plan = load_plan(study)
    package = load_package(study, plan)
    public = {"schema": "blind_local_v1", "plan_sha256": plan["plan_sha256"],
              "package_sha256": package["package_sha256"],
              "pairs": [{k: p[k] for k in ("id", "a", "b", "prompt")} for p in package["pairs"]],
              "clips": {k: {"sha256": v["sha256"]} for k, v in package["clips"].items()}}
    public["bundle_sha256"] = digest(public)
    if destination.is_relative_to(study):
        raise ValueError("Place the exported ZIP outside the source study directory")
    if research:
        scored = read(study / "private/presented_scores.json")
        if scored["package_sha256"] != package["package_sha256"]:
            raise ValueError("Presented scores belong to a different media package")
        with (study / "private/presented_scores.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(public["clips"]) or {r["clip_id"] for r in rows} != set(public["clips"]):
            raise ValueError("Presented score coverage is incomplete")
        if any(r["video_sha256"] != public["clips"][r["clip_id"]]["sha256"] for r in rows):
            raise ValueError("Presented score video hashes differ")
    # Verify every clip before opening an output archive. Only declared media enter it.
    for clip, record in ({} if metadata_only else public["clips"]).items():
        if file_hash(study / f"media/{clip}.mp4") != record["sha256"]:
            raise ValueError(f"Changed media: {clip}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).parent
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED if metadata_only else zipfile.ZIP_STORED, allowZip64=True) as archive:
        prefix = "blind_audit_research/" if research else "blind_audit_local/"
        for name in ("acceleration_blind_audit.py", "acceleration_blind_audit.html"):
            archive.write(base / name, prefix + "UNIV_adaptor/scripts/data/" + name)
        if not metadata_only:
            archive.writestr(prefix + "run_local.py", LAUNCHER)
            archive.writestr(prefix + "START_WINDOWS.cmd", CMD.replace("\n", "\r\n"))
        intro = "本包仅含分析数据，没有视频，不能用于播放或盲评。使用 ANALYZE_WINDOWS.cmd 汇总。\n\n" if metadata_only else ""
        archive.writestr(prefix + "README.txt", (intro + (RESEARCH_README if research else README)).encode("utf-8-sig"))
        archive.writestr(prefix + "study/public_study.json", json.dumps(public, ensure_ascii=False, indent=2))
        for clip in ([] if metadata_only else public["clips"]):
            archive.write(study / f"media/{clip}.mp4", prefix + f"study/media/{clip}.mp4")
        if research:
            archive.writestr(prefix + "run_analysis.py", ANALYSIS)
            archive.writestr(prefix + "ANALYZE_WINDOWS.cmd", CMD.replace("run_local.py", "run_analysis.py").replace("\n", "\r\n"))
            written = {Path("media") / f"{clip}.mp4" for clip in public["clips"]} | {Path("public_study.json")}
            for path in sorted(study.rglob("*")):
                relative = path.relative_to(study)
                if path.is_file() and relative not in written:
                    if metadata_only and ("media" in relative.parts or path.suffix.lower() not in (".json", ".csv", ".md", ".txt")):
                        continue
                    archive.write(path, prefix + "study/" + relative.as_posix())
    print(f"Ready: {destination}\n{len(public['pairs'])} pairs; videos {'excluded' if metadata_only else len(public['clips'])}; {destination.stat().st_size / 1024**2:.2f} MiB")


MERGER = '''from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
manifest = json.loads((root / "parts.json").read_text())
output = root.parent / manifest["filename"]
for part in manifest["parts"]:
    path = root / part["name"]
    if not path.is_file() or path.stat().st_size != part["bytes"]:
        raise RuntimeError("Missing/incomplete part: " + str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
        raise RuntimeError("Checksum mismatch: " + str(path))
with output.open("xb") as target:
    total = hashlib.sha256()
    for part in manifest["parts"]:
        data = (root / part["name"]).read_bytes()
        target.write(data)
        total.update(data)
if total.hexdigest() != manifest["sha256"]:
    raise RuntimeError("Combined checksum mismatch")
print("Verified: " + str(output))
'''


def split_zip(path, part_mib):
    path = Path(path).resolve()
    if not 1 <= part_mib <= 128:
        raise ValueError("part-mib must be between 1 and 128")
    folder = path.with_name(path.name + ".parts")
    folder.mkdir(exist_ok=True)
    records, total = [], hashlib.sha256()
    with path.open("rb") as source:
        while data := source.read(part_mib * 1024 * 1024):
            name = f"part-{len(records):05d}.bin"
            item = folder / name
            hashed = hashlib.sha256(data).hexdigest()
            if item.exists():
                if file_hash(item) != hashed:
                    raise ValueError(f"Existing part differs: {item}; use a new destination ZIP name")
            else:
                with item.open("xb") as handle:
                    handle.write(data)
            total.update(data)
            records.append({"name": name, "bytes": len(data), "sha256": hashed})
            print(f"Prepared {name}: {len(data) / 1024**2:.1f} MiB", flush=True)
    immutable(folder / "parts.json", {"filename": path.name, "sha256": total.hexdigest(), "parts": records})
    (folder / "merge.py").write_text(MERGER, encoding="utf-8")
    print(f"Download parts.json, merge.py and all {len(records)} parts from {folder}; run python merge.py locally.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs/acceleration_blind_audit_v1")
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--research", action="store_true", help="Include ALL study artifacts and local analysis; contains unblinding information")
    parser.add_argument("--metadata-only", action="store_true", help="Small compressed analysis export; no media reading/copying")
    parser.add_argument("--split-zip", type=Path, help="Split an existing ZIP; does not export or re-encode")
    parser.add_argument("--part-mib", type=int, default=8)
    args = parser.parse_args()
    if args.split_zip:
        split_zip(args.split_zip, args.part_mib)
    else:
        destination = args.zip or ROOT / ("outputs/acceleration_blind_audit_analysis.zip" if args.metadata_only else "outputs/acceleration_blind_audit_research.zip" if args.research else "outputs/acceleration_blind_audit_local.zip")
        export(args.study, destination, args.research, args.metadata_only)
