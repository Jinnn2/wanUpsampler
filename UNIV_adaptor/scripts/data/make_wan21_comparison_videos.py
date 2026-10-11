"""Create score-labelled horizontal five-method videos from completed scores.

CPU-only presentation, not new inference/scoring or blinded human evaluation.
Frozen source videos/scores are read only. Default panels retain native 832x480.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_flashvsr_wan21_complete as scoring

diag, helpers = scoring.diag, scoring.helpers
METHODS = (*scoring.generation.BASE_ARMS, scoring.generation.METHOD)
NAMES = {"FULL50": "FULL50 (Dense)", "STEP25": "STEP25 (25 steps)",
         "TEA008": "TeaCache (TEA008)", "SCALING10": "ScalingCache (SCALING10)",
         "LR_FLASH_B025": "LR + FlashVSR (B025)"}


def implementation():
    return {"driver": diag.file_hash(__file__), "scoring_reader": diag.file_hash(scoring.__file__),
            "helpers": diag.file_hash(helpers.__file__), "hash_helpers": diag.file_hash(diag.__file__)}


def find_font(explicit=None):
    candidates = [explicit] if explicit else [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        *Path(sys.prefix).glob("lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError("No usable TrueType font. Set FONT_FILE or --font to an existing .ttf file; no packages are installed automatically.")


def raw_quality5(values):
    return sum(values[d] for d in scoring.QUALITY) / len(scoring.QUALITY)


def group_clips(clips, scores):
    lookup = {}
    for clip in clips:
        group = helpers.safe_id(clip["group_id"])
        method = clip["method"]
        if method not in METHODS or (group, method) in lookup:
            raise ValueError("Duplicate/unexpected comparison method")
        lookup[group, method] = clip
    groups = []
    for group in sorted({k[0] for k in lookup}):
        rows = []
        for method in METHODS:
            clip = lookup[group, method]
            if (clip["frames"], clip["width"], clip["height"], clip["fps"]) != (81, 832, 480, 16):
                raise ValueError("Only complete 81-frame matching videos; no prefix/geometry substitution")
            rows.append(clip | {"scores": scores[clip["clip_id"]], "quality5_raw_mean": raw_quality5(scores[clip["clip_id"]])})
        if len({(r["prompt_id"], r["seed"], r["prompt"]) for r in rows}) != 1:
            raise ValueError("Mixed prompt/seed within a horizontal comparison")
        groups.append({"group_id": group, "prompt_id": rows[0]["prompt_id"], "seed": rows[0]["seed"],
                       "prompt": rows[0]["prompt"], "clips": rows})
    if len(groups) != 24:
        raise ValueError("Expected 24 full five-method prompt-seed groups")
    return groups


def layout(width):
    if width < 416 or width > 832 or width % 2:
        raise ValueError("Even panel width in [416,832] required; default 832 retains native pixels")
    ratio = width / 832
    video_h = 2 * round(480 * ratio / 2)
    # All dimensions remain even for common H264/yuv420p playback.
    top, header, footer = [2 * round(x * ratio / 2) for x in (144, 280, 64)]
    return {"panel_width": width, "video_height": video_h, "top": top, "header": header,
            "footer": footer, "video_y": top + header, "width": width * 5,
            "height": top + header + video_h + footer}


def plan(args):
    diag.outside(args.out, args.score_root)
    geometry = layout(args.panel_width)
    if (args.out / "plan.json").exists():
        p = load(args)
        if (p["score_root"] != str(args.score_root.resolve()) or p["layout"] != geometry or
                p["encoding"]["crf"] != args.crf or p["selection"] != args.group_ids or
                (args.font is not None and p["font"]["path"] != str(args.font.resolve()))):
            raise ValueError("Existing presentation uses different settings; choose a new output directory")
        print(f"Resuming {len(p['groups'])} frozen comparisons", flush=True)
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Unplanned nonempty output; choose a NEW comparison directory")
    source_args = argparse.Namespace(out=args.score_root)
    original = scoring.load(source_args)
    saved = scoring.load_scores(source_args, original)
    for source in (original["source_root"], original["study_root"]):
        diag.outside(args.out, source)
    groups = group_clips(original["clips"], saved["scores"])
    if args.group_ids:
        if len(set(args.group_ids)) != len(args.group_ids) or not set(args.group_ids) <= {g["group_id"] for g in groups}:
            raise ValueError("Unknown/duplicate --group-ids")
        groups = [g for g in groups if g["group_id"] in args.group_ids]
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        raise FileNotFoundError(f"FFmpeg unavailable: {args.ffmpeg}")
    version = subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]
    p = {"schema": "wan21_labelled_comparison_plan_v1", "implementation": implementation(),
         "score_root": str(args.score_root.resolve()), "scoring_plan_sha256": original["plan_sha256"],
         "scores_sha256": saved["scores_sha256"], "source_roots": [original["source_root"], original["study_root"]],
         "source_files": [diag.bound_file(args.score_root / name) for name in ("plan.json", "scores.json")],
         "groups": groups, "selection": args.group_ids, "layout": geometry,
         "font": diag.bound_file(find_font(args.font)), "ffmpeg": diag.bound_file(ffmpeg), "ffmpeg_version": version,
         "encoding": {"codec": "libx264", "crf": args.crf, "preset": "fast", "pix_fmt": "yuv420p"},
         "presentation_scope": "Labelled exploratory review, NOT blinded evaluation or rescored montage.",
         "metric_scope": "VBench-5 raw mean = arithmetic mean of five existing quality dimensions; display aid, NOT official VBench Total. Seven dimensions also shown individually.",
         "claim": "Inspect whether similar scores hide perceptible prompt/detail failures; visual impressions here are not independent blinded evidence."}
    p["plan_sha256"] = diag.digest(p)
    diag.write_new(args.out / "plan.json", p)
    print(f"Planned {len(groups)} horizontal videos; method order: {', '.join(METHODS)}; canvas {geometry['width']}x{geometry['height']}", flush=True)


def load(args):
    p = diag.read(args.out / "plan.json")
    if (p["schema"] != "wan21_labelled_comparison_plan_v1" or p["implementation"] != implementation() or
            diag.digest({k: v for k, v in p.items() if k != "plan_sha256"}) != p["plan_sha256"]):
        raise ValueError("Presentation plan/code hash mismatch")
    for source in [p["score_root"], *p["source_roots"]]:
        diag.outside(args.out, source)
    diag.verify_files(p["source_files"] + [p["font"], p["ffmpeg"]])
    diag.verify_files([r["video"] for g in p["groups"] for r in g["clips"]])
    return p


def wrap_text(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + ([line] if line else [])


def make_header(p, group, target):
    from PIL import Image, ImageDraw, ImageFont
    if target.exists():
        raise FileExistsError(f"Unreceipted header exists: {target}; inspect/move before resume")
    target.parent.mkdir(parents=True, exist_ok=True)
    size, ratio = p["layout"], p["layout"]["panel_width"] / 832
    image = Image.new("RGBA", (size["width"], size["height"]), (20, 24, 31, 255))
    draw = ImageDraw.Draw(image)
    fonts = {n: ImageFont.truetype(p["font"]["path"], max(12, round(n * ratio))) for n in (24, 26, 28, 32, 34)}
    draw.rectangle((0, size["video_y"], size["width"] - 1, size["video_y"] + size["video_height"] - 1), fill=(0, 0, 0, 0))
    def text(x, y, value, n=26, color="#e7edf4"):
        draw.text((round(x * ratio), round(y * ratio)), value, font=fonts[n], fill=color)
    text(20, 10, f"{group['prompt_id']} | seed {group['seed']} | 81 frames / 16 fps | LABELLED REVIEW", 34)
    lines = wrap_text(draw, "Prompt: " + group["prompt"], fonts[26], size["width"] - round(40 * ratio))
    if len(lines) > 2:
        raise ValueError("Prompt exceeds frozen header space; no silent truncation")
    for i, line in enumerate(lines):
        text(20, 64 + i * 34, line, 26)
    reference = group["clips"][0]["quality5_raw_mean"]
    top = size["top"] / ratio
    for i, row in enumerate(group["clips"]):
        x = i * 832
        v = row["scores"]
        text(x + 20, top + 10, NAMES[row["method"]], 32, "#a7d5ff")
        text(x + 20, top + 56, f"VBench-5 raw mean: {row['quality5_raw_mean']:.4f}", 28)
        lines = [f"Subj {v['subject_consistency']:.4f}    Bg {v['background_consistency']:.4f}",
                 f"Motion {v['motion_smoothness']:.4f}    Aes {v['aesthetic_quality']:.4f}",
                 f"Img {v['imaging_quality']:.4f}    Dyn {v['dynamic_degree']:.4f}",
                 f"Align {v['overall_consistency']:.4f}"]
        for j, line in enumerate(lines):
            text(x + 20, top + 104 + j * 32, line, 26)
        text(x + 20, top + 240, f"Mean delta vs FULL: {row['quality5_raw_mean'] - reference:+.4f}", 24)
        if i:
            draw.line((i * size["panel_width"], size["top"], i * size["panel_width"], size["video_y"] - 1), fill="#596373", width=2)
    footer_y = (size["video_y"] + size["video_height"]) / ratio
    text(20, footer_y + 14, "Raw mean is NOT official VBench Total. Dyn is motion degree, NOT higher-is-better. Same seed is NOT pixel-aligned ground truth.", 24)
    with target.open("xb") as handle:
        image.save(handle, format="PNG")


def render_command(p, group, header, output, threads):
    command = [p["ffmpeg"]["path"], "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
               "-filter_complex_threads", "1"]
    for row in group["clips"]:
        command += ["-threads", "1", "-i", row["video"]["path"]]
    command += ["-loop", "1", "-framerate", "16", "-i", str(header)]
    size = p["layout"]
    filters = []
    for i in range(5):
        transform = "" if size["panel_width"] == 832 else f"scale={size['panel_width']}:{size['video_height']}:flags=lanczos,"
        # Exact frame-index alignment at the validated original 16 fps. This
        # avoids a high MP4 track timebase propagating as encoder frame rate.
        filters.append(f"[{i}:v]{transform}settb=expr=1/16,setpts=N,setsar=1[v{i}]")
    filters.append("".join(f"[v{i}]" for i in range(5)) + "hstack=inputs=5:shortest=1[row]")
    filters.append(f"[row]pad={size['width']}:{size['height']}:0:{size['video_y']}:color=0x14181f[canvas]")
    filters.append("[canvas][5:v]overlay=0:0:shortest=1:format=auto,format=yuv420p[out]")
    enc = p["encoding"]
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-sn", "-dn",
                "-frames:v", "81", "-r", "16", "-enc_time_base", "1:16", "-fps_mode", "cfr",
                "-c:v", enc["codec"], "-preset", enc["preset"],
                "-crf", str(enc["crf"]), "-pix_fmt", enc["pix_fmt"], "-threads", str(threads),
                "-movflags", "+faststart", str(output)]
    return command


def validate_receipt(args, p, group):
    receipt = args.out / "receipts" / (group["group_id"] + ".json")
    row = diag.read(receipt)
    if (row["plan_sha256"] != p["plan_sha256"] or row["group_id"] != group["group_id"] or
            row["clip_ids"] != [r["clip_id"] for r in group["clips"]] or
            row["geometry"] != {"width": p["layout"]["width"], "height": p["layout"]["height"], "frames": 81, "fps": 16}):
        raise ValueError("Comparison receipt/source order/geometry mismatch")
    diag.verify_files([row["video"], row["header"]])
    return row


def render_one(args, p, group):
    gid = group["group_id"]
    receipt = args.out / "receipts" / (gid + ".json")
    if receipt.exists():
        validate_receipt(args, p, group)
        return gid + " (verified existing)"
    video = args.out / "videos" / (gid + ".mp4")
    temporary = video.with_name(gid + ".partial.mp4")
    header = args.out / "headers" / (gid + ".png")
    log = args.out / "logs" / (gid + ".log")
    if any(path.exists() for path in (video, temporary, header, log)):
        raise FileExistsError(f"Unreceipted output for {gid}; inspect/move its video/partial/header/log before resume")
    for clip in group["clips"]:
        observed = helpers.probe(clip["video"]["path"], args.ffprobe)
        if observed != {"width": 832, "height": 480, "frames": 81, "fps": 16}:
            raise ValueError(f"Source frame/fps/canvas changed: {clip['clip_id']}")
    make_header(p, group, header)
    video.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(render_command(p, group, header, temporary, args.threads), stdout=handle, stderr=subprocess.STDOUT, check=True)
    geometry = helpers.probe(temporary, args.ffprobe)
    expected = {"width": p["layout"]["width"], "height": p["layout"]["height"], "frames": 81, "fps": 16}
    if geometry != expected:
        raise ValueError(f"Rendered frame/fps/canvas mismatch: {geometry} != {expected}")
    # Exclusive hard-link commit: no replacement of an existing final output.
    os.link(temporary, video)
    temporary.unlink()
    diag.write_new(receipt, {"plan_sha256": p["plan_sha256"], "group_id": gid,
                   "clip_ids": [r["clip_id"] for r in group["clips"]], "video": diag.bound_file(video),
                   "header": diag.bound_file(header), "geometry": expected})
    return gid


def render(args):
    p = load(args)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(render_one, args, p, group) for group in p["groups"]]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"Rendered {i}/{len(futures)}: {future.result()}", flush=True)


def index(args):
    p = load(args)
    rows, cards, outputs = [], [], []
    for group in p["groups"]:
        receipt = validate_receipt(args, p, group)
        gid = group["group_id"]
        outputs.append(receipt)
        for clip in group["clips"]:
            rows.append({"group_id": gid, "prompt_id": group["prompt_id"], "seed": group["seed"],
                         "method": clip["method"], "source_clip_id": clip["clip_id"],
                         "source_video_sha256": clip["video"]["sha256"],
                         "vbench5_raw_mean_display_only": clip["quality5_raw_mean"]} | clip["scores"])
        cards.append(f'<section id="{gid}"><h2>{group["prompt_id"]} / seed {group["seed"]}</h2>'
                     f'<p>{html.escape(group["prompt"])}</p><a href="videos/{gid}.mp4" download>下载这一组</a>'
                     f'<div class="strip"><video controls preload="none" width="{p["layout"]["width"]}" '
                     f'poster="headers/{gid}.png" src="videos/{gid}.mp4"></video></div></section>')
    links = ' '.join(f'<a href="#{g["group_id"]}">{g["group_id"]}</a>' for g in p["groups"])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Wan2.1 五方法对比</title>
<style>body{font-family:system-ui,sans-serif;background:#14181f;color:#e7edf4;margin:24px}a{color:#a7d5ff}nav a{display:inline-block;margin:6px}section{border-top:1px solid #596373;padding:20px 0}.strip{overflow-x:auto}video{display:block;width:100%;height:auto}body.native video{width:auto;max-width:none}button{padding:10px;margin:8px 0}</style>
<h1>Wan2.1：五方法横向对比（有标签的探索性查看）</h1>
<p>从左到右：FULL50 → STEP25 → TeaCache → ScalingCache → LR+FlashVSR。同 prompt、seed；按时间同步，不代表像素级配准。</p>
<p>VBench-5 raw mean 仅为五个质量维度的原始算术均值，不是官方 VBench Total。Subj=主体一致性，Bg=背景一致性，Motion=运动平滑度，Aes=美学，Img=成像质量，Dyn=运动程度，Align=文本视频整体一致性。Dyn 并非越高越好。</p>
<p>分数来自原始视频，而不是重新压缩的拼接视频。已显示方法和分数，不能作为盲评证据。</p>
<button onclick="document.body.classList.toggle('native');this.textContent=document.body.classList.contains('native')?'切换：适应窗口':'切换：原尺寸（横向滚动）'">切换：原尺寸（横向滚动）</button>
<p><a href="scores_by_panel.csv">逐面板分数 CSV</a></p><nav>''' + links + '</nav>' + ''.join(cards) + '</html>'
    helpers.write_text_new(args.out / "index.html", page)
    scoring.write_csv(args.out / "scores_by_panel.csv", rows)
    diag.write_new(args.out / "manifest.json", {"plan_sha256": p["plan_sha256"], "n_groups": len(outputs),
                   "method_order": list(METHODS), "layout": p["layout"], "outputs": outputs,
                   "metric_scope": p["metric_scope"], "presentation_scope": p["presentation_scope"]})
    print(f"Offline review index: {args.out / 'index.html'}", flush=True)


def export(args):
    index(args)
    p = load(args)
    archive = args.out.parent / (args.out.name + "_review.tgz")
    receipt = args.out / "export_receipt.json"
    if archive.exists():
        if not receipt.exists():
            raise FileExistsError(f"Unreceipted/partial archive; inspect/move before resume: {archive}")
        prior = diag.read(receipt)
        if prior["plan_sha256"] != p["plan_sha256"]:
            raise ValueError("Export plan mismatch")
        diag.verify_files([prior["archive"]])
        print(f"Verified existing review package: {archive}")
        return
    if receipt.exists():
        raise FileNotFoundError("Export receipt exists but archive missing")
    with tarfile.open(archive, "x:gz") as bundle:
        for path in sorted(args.out.rglob("*")):
            if path.is_file() and path.suffix in (".mp4", ".json", ".csv", ".png", ".html"):
                bundle.add(path, arcname=(Path(args.out.name) / path.relative_to(args.out)).as_posix())
    diag.write_new(receipt, {"plan_sha256": p["plan_sha256"], "archive": diag.bound_file(archive)})
    print(f"Download {archive} ({archive.stat().st_size / 1024**2:.1f} MiB); extract and open index.html locally", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "plan", "render", "index", "export"))
    parser.add_argument("--score-root", type=Path, default=ROOT / "outputs/flashvsr_wan21_complete_score_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/wan21_five_method_comparisons_v1")
    parser.add_argument("--group-ids", nargs="+")
    parser.add_argument("--panel-width", type=int, default=832)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--crf", type=int, choices=range(0, 19), default=12)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.score_root, args.out = args.score_root.resolve(), args.out.resolve()
    if args.jobs < 1 or args.threads < 1:
        parser.error("Positive jobs/threads required")
    if args.mode == "all":
        for function in (plan, render, index, export):
            function(args)
    else:
        globals()[args.mode](args)


if __name__ == "__main__":
    main()
