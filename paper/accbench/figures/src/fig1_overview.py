"""Figure 1 (full width): one accelerator for every prompt vs AccBench -> region profiles -> prompt-guided routing.

Layout mirrors the agreed wireframe: two bands with the same four columns (prompt cells | processing | outputs |
scorecard). Band (a) runs the same prompts through fixed arms; band (b) runs them through AccBench, the
method x cell profiles and the budgeted router.

Data (all exported by the main line on 2026-10-27 from outputs/fig1_v1, see ../README.md)
- profiles inset:      data/route_table_20261027.json   r[arm][cell] = [retention, quality_kept, rel. adv., speedup]
- (a) per-cell numbers: data/fixed_arms_by_cell_dev2_20261027.csv   (dev2, 61 prompts, 3 seeds averaged)
- scorecards:           data/scoreboard_dev2_20261027.csv            (dev2; router/blind policies fit on dev only)
- (b) routed arms:      data/scoreboard_dev2_policies_20261027.json  (dev-fit per-cell distributions, argmax shown)
- frame strips: TODO, main line decodes them on the server (median sample by min(retention, quality), dev2 seed 42).

Usage: python fig1_overview.py [--tau 0.8]  -> ../fig1_overview.pdf and .png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "fig1_overview"
SAMPLES = DATA / "fig1_samples_v1"   # median clips per (arm, cell), dev2 seed 42; 416-px copies of the server export

CELLS = ["hh", "hl", "lh", "ll"]
CELL_LABEL = {"hh": "high motion\nhigh detail", "hl": "high motion\nlow detail",
              "lh": "low motion\nhigh detail", "ll": "low motion\nlow detail"}
ARM_NAME = {"RCM4": "rCM 4-step", "LR_FLASH": "LR + FlashVSR", "LR_BICUBIC": "LR + bicubic",
            "F41_RIFE": "41f + RIFE", "DVG_D050": "DVG", "TEA015": "TeaCache", "TEA008": "TeaCache (0.08)",
            "SCALING10": "SCALING10",            # TODO: paper name for this cache_step arm
            "LR41_FLASH_RIFE": "LR 41f + FlashVSR + RIFE", "FULL_NATIVE_50": "FULL (50 steps)"}
# representative arm per family for the profile inset (full 15-arm table goes in the experiments section)
PROFILE_ARMS = ["RCM4", "LR_FLASH", "F41_RIFE", "DVG_D050", "TEA015"]
FIXED_ARMS = ["RCM4", "LR_FLASH"]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

C_BOX = "#F4F3EE"; C_EDGE = "#8A8880"; C_OK_E = "#1D9E75"; C_BAD_E = "#D85A30"
C_OURS = "#ECEAFB"; C_OURS_E = "#534AB7"; C_BENCH = "#E1F5EE"; C_TXT = "#2C2C2A"; C_MUTE = "#5F5E5A"
GREENS = plt.get_cmap("Greens")


def load():
    d = json.loads((DATA / "route_table_20261027.json").read_text(encoding="utf-8"))
    by_cell = {}
    with (DATA / "fixed_arms_by_cell_dev2_20261027.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_cell[(r["arm"], r["cell"])] = {k: float(v) for k, v in r.items() if k not in ("arm", "cell")}
    score = {}
    with (DATA / "scoreboard_dev2_20261027.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["policy"], float(r["tau"])) if r["tau"] else r["policy"]
            score[key] = {k: float(v) for k, v in r.items() if k not in ("policy", "tau")}
    pol = json.loads((DATA / "scoreboard_dev2_policies_20261027.json").read_text(encoding="utf-8"))
    return d, by_cell, score, pol["policies_dev_fit"]


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=7, weight="normal", sub=None, color=C_TXT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3", fc=fc, ec=ec, lw=0.6))
    if sub is None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color=color)
    else:
        ax.text(x + w / 2, y + h * 0.66, text, ha="center", va="center", fontsize=fs, weight=weight, color=color)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=fs - 1, color=C_MUTE)


def arrow(ax, p, q, color=C_EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.6, color=color,
                                 shrinkA=0, shrinkB=0))


def strip(ax, x, y, w, h, label, kept, label_color=None, frames=None, header=False):
    """One output row, tinted by min(retention, quality_kept) like the profile inset.
    frames = [FULL frame, arm frame 0, arm frame 40, arm frame 80] (paths): the FULL reference at the left, then
    three frames of the accelerated clip. Without frames an empty 4-frame placeholder is drawn."""
    ax.add_patch(Rectangle((x, y), w, h, fc=GREENS(0.1 + 0.6 * kept), ec=C_EDGE, lw=0.5))
    fw = (w - 8) / 4; fh = fw / (832 / 480); fy = y + h * 0.92 - fh
    for i in range(4):
        fx = x + 2 + i * (fw + 1.5) + (1.5 if i else 0)        # extra gap between FULL and the arm frames
        if frames is None:
            ax.add_patch(Rectangle((fx, fy), fw, fh, fc="#FFFFFF", ec="#C9C7BE", lw=0.4))
        else:
            ax.imshow(plt.imread(frames[i]), extent=(fx, fx + fw, fy, fy + fh), aspect="auto",
                      interpolation="lanczos", zorder=2)
    if header:
        ax.text(x + 2 + fw / 2, y + h + 1.5, "FULL", ha="center", va="bottom", fontsize=4.3, color=C_MUTE)
        ax.text(x + 2 + fw + 3 + 1.5 * fw, y + h + 1.5, "accel. f0 / f40 / f80", ha="center",
                va="bottom", fontsize=4.3, color=C_MUTE)
    ax.text(x + w / 2, y + h * 0.13, label, ha="center", va="center", fontsize=4.8,
            color=label_color or (C_TXT if kept < 0.6 else "#F7FBF5"))


def band(ax, y0, title):
    ax.add_patch(FancyBboxPatch((40, y0), 600, 160, boxstyle="round,pad=0,rounding_size=6", fc="none",
                                ec="#B4B2A9", lw=0.6, ls=(0, (3, 2))))
    ax.text(48, y0 + 150, title, fontsize=7.5, weight="bold", va="center", color=C_TXT)


def cells_column(ax, rows):
    for c, y in zip(CELLS, rows):
        box(ax, 48, y, 70, 24, CELL_LABEL[c], fs=5.5)
    ax.plot([118, 128, 128, 118], [rows[0] + 12, rows[0] + 12, rows[-1] + 12, rows[-1] + 12], color=C_EDGE, lw=0.6)
    ax.plot([118, 128], [rows[1] + 12, rows[1] + 12], color=C_EDGE, lw=0.6)
    ax.plot([118, 128], [rows[2] + 12, rows[2] + 12], color=C_EDGE, lw=0.6)


def fmt_pp(v):
    return f"{v:+.1f} pp"


FIXED_POINTS = ["SCALING10", "TEA015", "STEP20", "F41_RIFE", "LR_FLASH", "LR_BICUBIC", "LR41_FLASH_RIFE", "RCM4"]


def frontier(fig, x, y, w, h, score, highlight, curves, tau_mark=None):
    """Mini speed vs min(semantic, quality)-kept plot in canvas coordinates (680 x 350), dev2 numbers.
    Fixed arms are points; with curves=True the router and prompt-blind frontiers are drawn on top."""
    a = fig.add_axes([x / 680, y / 350, w / 680, h / 350])
    a.set_xlim(0.8, 8.2); a.set_ylim(0, 1.04)
    a.set_xticks([1, 2, 4, 6, 8]); a.set_yticks([0, 0.5, 1.0])
    a.tick_params(labelsize=4.8, length=1.5, pad=1.5, colors=C_MUTE)
    for s in a.spines.values():
        s.set_linewidth(0.5); s.set_color(C_EDGE)
    a.set_xlabel("speedup (x)", fontsize=5, color=C_MUTE, labelpad=1)
    a.set_ylabel("min(sem., qual.) kept", fontsize=5, color=C_MUTE, labelpad=1)
    a.set_title("dev2 (61 prompts)", fontsize=5.2, color=C_MUTE, pad=2)
    a.plot([1], [1], marker="*", ms=4, color=C_TXT, ls="none")
    a.annotate("FULL", (1, 1), xytext=(2, -4), textcoords="offset points", fontsize=4.5, color=C_MUTE)
    for arm in FIXED_POINTS:
        s = score[arm]
        hi = arm in highlight
        a.plot([s["speedup"]], [s["min_ret_q"]], marker="o", ms=3 if hi else 2,
               color=C_OURS_E if hi else "#B4B2A9", ls="none", zorder=3)
        if hi:
            up = arm != "RCM4"
            a.annotate(f"{ARM_NAME[arm]}\nΔVB {fmt_pp(s['d_vbench5_pp'])}", (s["speedup"], s["min_ret_q"]),
                       xytext=(0 if up else 3, 4 if up else -4), textcoords="offset points", fontsize=4.5,
                       color=C_OURS_E, ha="center" if up else "right", va="bottom" if up else "top")
    if curves:
        taus = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]
        for pol, col, ls, lab in [("router", C_OURS_E, "-", "router"), ("blind", C_MUTE, "--", "blind mixture")]:
            xs = [score[(pol, t)]["speedup"] for t in taus]; ys = [score[(pol, t)]["min_ret_q"] for t in taus]
            a.plot(xs, ys, color=col, ls=ls, lw=0.9, marker="o", ms=1.6, zorder=4, label=lab)
        if tau_mark is not None:
            s = score[("router", tau_mark)]
            a.annotate(f"τ={tau_mark:.2f}: {s['speedup']:.1f}x\nΔVB {fmt_pp(s['d_vbench5_pp'])}",
                       (s["speedup"], s["min_ret_q"]), xytext=(6, 2), textcoords="offset points",
                       fontsize=4.5, color=C_OURS_E, va="center")
        a.legend(fontsize=4.5, loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    else:
        a.text(1.0, 0.04, "grey: other fixed arms", ha="left", va="bottom", fontsize=4.5, color=C_MUTE)


def main(tau: float):
    d, by_cell, score, pol = load()
    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 680); ax.set_ylim(0, 350); ax.axis("off")
    ax.set_autoscale_on(False)

    rows = [292, 262, 232, 202]          # band (a) row baselines (y of the row's bottom), top to bottom
    rows_b = [r - 172 for r in rows]     # band (b)

    # ---------------- band (a): status quo ----------------
    band(ax, 180, "(a) Status quo: one accelerator for every prompt")
    cells_column(ax, rows)
    arrow(ax, (128, 247), (150, 247))
    sa = {a: score[a] for a in FIXED_ARMS}
    box(ax, 152, 266, 96, 22, f"{ARM_NAME['RCM4']}, {sa['RCM4']['speedup']:.1f}x", fc=C_OURS, ec=C_OURS_E, fs=6)
    box(ax, 152, 236, 96, 22, f"{ARM_NAME['LR_FLASH']}, {sa['LR_FLASH']['speedup']:.1f}x", fc=C_OURS,
        ec=C_OURS_E, fs=6)
    ax.text(200, 262, "or", ha="center", va="center", fontsize=6, color=C_MUTE)
    # both arms join one junction: whichever arm is picked, it is applied to every prompt
    ax.plot([248, 262, 262], [277, 277, 247], color=C_EDGE, lw=0.6)
    ax.plot([248, 270], [247, 247], color=C_EDGE, lw=0.6)
    ax.plot([262], [247], marker="o", ms=2.2, color=C_EDGE)
    # outputs: two columns of strips (one per fixed arm), four rows (one per cell); tint and numbers from dev2
    for j, arm in enumerate(FIXED_ARMS):
        x = 296 + j * 102
        ax.text(x + 48, 331, ARM_NAME[arm], ha="center", fontsize=5.5, color=C_MUTE)
        for k, (c, y) in enumerate(zip(CELLS, rows)):
            r = by_cell[(arm, c)]
            sd = SAMPLES / f"{arm}_{c}"
            fr = [sd / "FULL_f040.jpg"] + [sd / f"{arm}_f{f:03d}.jpg" for f in (0, 40, 80)]
            strip(ax, x, y, 96, 26, f"kept {r['min_ret_q']:.2f}  ·  ΔVB {r['d_vbench5_pp']:+.1f} pp",
                  r["min_ret_q"], frames=fr if all(p.exists() for p in fr) else None, header=(k == 0))
    for y in rows:
        arrow(ax, (270, 247), (294, y + 13))
    # scorecard: speed vs min(semantic, quality) kept on dev2; fixed arms are points, VBench deltas annotated
    frontier(fig, 538, 198, 94, 100, score, highlight=FIXED_ARMS, curves=False)

    # ---------------- band (b): ours ----------------
    band(ax, 8, "(b) Ours: AccBench  →  method × region profiles  →  prompt-guided routing")
    cells_column(ax, rows_b)
    arrow(ax, (128, 75), (150, 75))
    box(ax, 152, 52, 68, 46, "AccBench", fc=C_BENCH, ec=C_OK_E, fs=7, weight="bold",
        sub="paired to FULL\n3 layers")
    arrow(ax, (220, 75), (232, 75))

    # profile inset: representative arms x cells, min(retention, quality_kept)
    ix, iy, cw, ch = 234, 40, 11, 11
    vals = np.array([[min(d["r"][a][c][0], d["r"][a][c][1]) for c in CELLS] for a in PROFILE_ARMS])
    for i, a in enumerate(PROFILE_ARMS):
        for j, c in enumerate(CELLS):
            ax.add_patch(Rectangle((ix + j * cw, iy + (len(PROFILE_ARMS) - 1 - i) * ch), cw, ch,
                                   fc=GREENS(0.15 + 0.8 * vals[i, j]), ec="white", lw=0.4))
        ax.text(ix + 4 * cw + 2, iy + (len(PROFILE_ARMS) - 1 - i) * ch + ch / 2, ARM_NAME[a], va="center",
                fontsize=4.8, color=C_MUTE)
    for j, c in enumerate(CELLS):
        ax.text(ix + j * cw + cw / 2, iy + len(PROFILE_ARMS) * ch + 3, c, ha="center", fontsize=4.8, color=C_MUTE)
    ax.text(ix + 2 * cw, iy - 8, "min(semantic, quality) kept", ha="center", fontsize=5, color=C_MUTE)
    ax.text(ix + 2 * cw, 114, "profiles", ha="center", fontsize=6.5, color=C_TXT)

    arrow(ax, (318, 75), (338, 75))
    box(ax, 340, 52, 70, 46, "Router", fc=C_OURS, ec=C_OURS_E, fs=7, weight="bold",
        sub=f"min time s.t.\nsem., qual. ≥ τ = {tau:.2f}")
    # outputs: one strip per cell, labelled with the dev-fit routed arm (argmax of the per-cell distribution)
    routed = pol[f"{tau:g}"]["router"]
    for c, y in zip(CELLS, rows_b):
        arm, p = max(routed[c].items(), key=lambda kv: kv[1])
        arrow(ax, (410, 75), (438, y + 13))
        strip(ax, 440, y, 70, 26, "→ " + ARM_NAME[arm] + ("" if p > 0.95 else f" ({p:.0%})"), 1.0,
              label_color="#F7FBF5")
    # same axes as band (a): fixed arms in grey, router and blind-mixture frontiers (tau 0.95 -> 0.60) on top
    frontier(fig, 538, 26, 94, 100, score, highlight=[], curves=True, tau_mark=tau)

    fig.savefig(OUT.with_suffix(".pdf")); fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.8)
    main(ap.parse_args().tau)
