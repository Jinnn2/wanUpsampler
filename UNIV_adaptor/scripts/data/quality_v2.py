"""Quality layer v2: the v1 VLM judgments plus fixed-layer temporal checks the VLM cannot see.

The synthetic-degradation control showed the v1 quality layer (including its "temporal" aspect) is blind to frame
repetition (judder) and per-frame brightness flicker, while the fixed-layer signals flow_alternation and flicker
detect them. v2 flags a clip when a signal is both

    above the 99th percentile of that signal over all FULL clips (outside what real FULL videos show), and
    above the same-seed FULL clip's own value (worse than its reference),

so the thresholds come from FULL clips alone, never from any accelerated method's scores. Then

    kept_temporal_v2 = kept_temporal and not judder and not flicker
    quality_kept     = mean(kept_sharp, kept_artifact, kept_aesthetic, kept_temporal_v2, kept_acceptable)
    quality_kept_gate = quality_kept_v1 if neither flag else 0          (stricter sensitivity variant)

Usage:
  python quality_v2.py --quality Q/quality_by_video.csv --fixed F/fixed_by_video.csv \
      [--calibration-fixed MAIN/fixed_by_video.csv] --out Q2/quality_by_video.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UNIV_adaptor.data_protocol import sha256_file, write_json_atomic  # noqa: E402

FULL = "FULL_NATIVE_50"
SIGNALS = {"judder": "flow_alternation", "flicker": "flicker"}
QUANTILE = 0.99
ASPECTS = ("sharp", "artifact", "aesthetic", "temporal", "acceptable")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def thresholds(calibration):
    full = [r for r in calibration if r["case_id"] == FULL]
    if len(full) < 50:
        raise ValueError(f"only {len(full)} FULL clips to calibrate on")
    return {flag: float(np.quantile([float(r[col]) for r in full if r[col] not in ("", "nan")], QUANTILE))
            for flag, col in SIGNALS.items()}, len(full)


def flags(row, limits):
    """A clip's own value and its FULL's (value - paired delta) against the FULL-population thresholds."""
    out = {}
    for flag, col in SIGNALS.items():
        value, delta = float(row[col]), float(row[f"d_{col}"])
        out[flag] = int(value > limits[flag] and delta > 0)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quality", required=True, help="v1 quality_by_video.csv")
    parser.add_argument("--fixed", required=True, help="fixed_by_video.csv covering the same clips")
    parser.add_argument("--calibration-fixed", help="fixed_by_video.csv whose FULL clips set the thresholds "
                        "(default: --fixed)")
    parser.add_argument("--out", required=True, help="output quality_by_video.csv (v2)")
    args = parser.parse_args(argv)
    calibration = args.calibration_fixed or args.fixed
    limits, n_full = thresholds(read_csv(calibration))
    fixed = {(r["source"], r["stem"]): r for r in read_csv(args.fixed)}
    rows = []
    for r in read_csv(args.quality):
        out = dict(r)
        out["quality_kept_v1"] = r["quality_kept"]
        if r["case_id"] != FULL and r["quality_kept"] != "":
            f = flags(fixed[(r["source"], r["stem"])], limits)
            kept = {a: int(r[f"kept_{a}"]) for a in ASPECTS}
            kept["temporal"] = int(kept["temporal"] and not f["judder"] and not f["flicker"])
            out.update(f, kept_temporal_v2=kept["temporal"], quality_kept=float(np.mean(list(kept.values()))),
                       quality_kept_gate=float(r["quality_kept"]) * (1 - max(f.values())))
        rows.append(out)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(path.with_name("quality_v2_thresholds.json"), {
        "quantile": QUANTILE, "signals": SIGNALS, "thresholds": limits, "full_clips": n_full,
        "calibration_fixed": str(Path(calibration).resolve()), "calibration_sha256": sha256_file(calibration),
        "quality_v1_sha256": sha256_file(args.quality)})
    print(f"thresholds (FULL q{QUANTILE:g} over {n_full} clips): " + ", ".join(f"{k} {v:.3f}" for k, v in limits.items()))
    by = collections.defaultdict(list)
    for r in rows:
        if "judder" in r:
            by[r["case_id"]].append(r)
    print("case".ljust(18) + "judder  flicker  kept_v1  kept_v2   gate")
    for case, sub in sorted(by.items()):
        m = lambda k: np.mean([float(x[k]) for x in sub])  # noqa: E731
        print(case.ljust(18) + f"{m('judder'):6.2f} {m('flicker'):8.2f} {m('quality_kept_v1'):8.3f} "
              f"{m('quality_kept'):8.3f} {m('quality_kept_gate'):6.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
