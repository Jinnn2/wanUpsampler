"""Export the rendered study for offline/local annotation; no model or re-encoding."""
import argparse
import csv
import json
from pathlib import Path
import zipfile

if __package__:
    from .acceleration_blind_audit import ROOT, digest, file_hash, load_package, load_plan, read
else:
    from acceleration_blind_audit import ROOT, digest, file_hash, load_package, load_plan, read


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


def export(study, destination, research=False):
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
    for clip, record in public["clips"].items():
        if file_hash(study / f"media/{clip}.mp4") != record["sha256"]:
            raise ValueError(f"Changed media: {clip}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).parent
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        prefix = "blind_audit_research/" if research else "blind_audit_local/"
        for name in ("acceleration_blind_audit.py", "acceleration_blind_audit.html"):
            archive.write(base / name, prefix + "UNIV_adaptor/scripts/data/" + name)
        archive.writestr(prefix + "run_local.py", LAUNCHER)
        archive.writestr(prefix + "START_WINDOWS.cmd", CMD.replace("\n", "\r\n"))
        archive.writestr(prefix + "README.txt", (RESEARCH_README if research else README).encode("utf-8-sig"))
        archive.writestr(prefix + "study/public_study.json", json.dumps(public, ensure_ascii=False, indent=2))
        for clip in public["clips"]:
            archive.write(study / f"media/{clip}.mp4", prefix + f"study/media/{clip}.mp4")
        if research:
            archive.writestr(prefix + "run_analysis.py", ANALYSIS)
            archive.writestr(prefix + "ANALYZE_WINDOWS.cmd", CMD.replace("run_local.py", "run_analysis.py").replace("\n", "\r\n"))
            written = {Path("media") / f"{clip}.mp4" for clip in public["clips"]} | {Path("public_study.json")}
            for path in sorted(study.rglob("*")):
                relative = path.relative_to(study)
                if path.is_file() and relative not in written:
                    archive.write(path, prefix + "study/" + relative.as_posix())
    print(f"Ready: {destination}\n{len(public['pairs'])} pairs, {len(public['clips'])} clips; {destination.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs/acceleration_blind_audit_v1")
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--research", action="store_true", help="Include ALL study artifacts and local analysis; contains unblinding information")
    args = parser.parse_args()
    destination = args.zip or ROOT / ("outputs/acceleration_blind_audit_research.zip" if args.research else "outputs/acceleration_blind_audit_local.zip")
    export(args.study, destination, args.research)
