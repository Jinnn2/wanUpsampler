# Acceleration degradation blind pilot

Claim under test: human-visible acceleration degradation may be missed or
misranked by existing metrics. This pilot does not assume that FULL wins, that
different seeds have equal quality, or that a new metric will rescue routing.

All assets are reused. No diffusion model is called. The default plan selects
48 real method comparisons (24 HY15, 12 Wan targeted S/T, 12 Wan matched-star),
24 controlled derivatives (four existing HY15 FULL videos x three degradation
types x two levels), and eight HY15 FULL different-seed controls. The Wan star
REFERENCE is already accelerated, not native HR. Seed controls and derivatives
currently cover HY15 only; they cannot establish cross-model generalization.

Selection is deterministic and score-independent: balance available factor
cells within strata, then prompts, with a fixed random seed. Missing quotas or
missing/changed source videos fail instead of silently selecting substitutes.
Test records explicitly tagged test are excluded. All currently selected sets
are development data. Review private/pair_review.csv BEFORE recruiting raters.
Do not cherry-pick visually striking metric failures into this stratified set.

## Plan and reuse on the server

Run from `/mnt/afs_2/houze/wanUpsampler`. Python 3.10+ is sufficient.

```bash
python UNIV_adaptor/scripts/data/acceleration_blind_audit.py plan
python UNIV_adaptor/scripts/data/acceleration_blind_audit.py package
python UNIV_adaptor/scripts/data/acceleration_blind_audit.py score --ngpus 8
python UNIV_adaptor/scripts/data/acceleration_blind_audit.py serve --port 8765
```

Plan/serve/report use only the Python standard library. Score uses the existing
NumPy-based strict VBench runner and the existing VBench environment. It pins
the known HY15 evaluator commit fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490;
do not override it merely to bypass a checkout mismatch. Scoring and human
annotation may be performed independently after packaging. No GPU is needed
to serve or annotate. Only score uses the eight GPUs.

Plan needs only score exports; package needs the original videos and `ffmpeg`
and `ffprobe` in PATH, with the libx264 encoder. It verifies source SHA-256s,
renders anonymous copies, and leaves originals untouched. Work is resumable
through verified per-clip receipts. This is CPU video encoding, not 8-GPU
generation. A source must be a directory containing quality_by_video.csv and
evaluation_inputs.json, or a tgz with exactly one of each. HY15 exports have
video paths/hashes directly in quality_by_video.csv and do not need the JSON.

The source roots are in configs/acceleration_blind_audit_v1.json. Override a
root without editing the protocol using, for example:

```bash
python UNIV_adaptor/scripts/data/acceleration_blind_audit.py plan \
  --source wan_targeted=/mnt/afs_2/houze/wanUpsampler/outputs/univ_targeted_st_temporal_calibration_v2/metrics/targeted_st_vbench
```

If the source export has moved, provide its actual root. If the VIDEOS have
moved, use `package --path-map OLD_PREFIX=NEW_PREFIX`; hashes still must match.
An existing frozen plan cannot be replaced; changed settings require a new
`--out` directory. The locally reviewed plan can instead be copied as the
entire `outputs/acceleration_blind_audit_v1` directory to the server and packaged
directly, without rerunning plan. Its original `/mnt/...` video paths are kept.

The web server binds only to localhost. On each evaluator's computer, forward
the port using the same SSH host/login as normal (replace SERVER_LOGIN):

```bash
ssh -N -L 8765:127.0.0.1:8765 SERVER_LOGIN
```

Then open `http://127.0.0.1:8765`. Give each evaluator a distinct pseudonymous
ID such as rater01. The order and A/B sides are independently randomized by ID
and stable on resume. Three IDs should represent three different people, not
three sessions of the same person. No invitation or external message is sent
by the scripts. Never serve the root with `python -m http.server`: private/
contains original paths, method names, metrics, and individual answers. This
tool is for a trusted local/SSH pilot, not public Internet deployment.

## Presentation and controlled distortions

Both sides use a 1280x720 letterboxed canvas, 24 fps, no audio, identical H.264
CRF16 encoding, and no embedded source metadata. A source larger than the
canvas is rejected to avoid hiding detail by downscaling. The page offers
synchronized replay and equal zoom controls. Use a desktop display; establish
the same viewing conditions across raters. Browser playback is not frame-lock
hardware; replay for temporal comparisons, especially on a slow connection.

Spatial derivatives reduce each native dimension to 0.5 or 0.25, then restore
with Lanczos. Temporal derivatives retain 12 or 6 fps and repeat frames back
to 24 fps (not a learned interpolator). Freeze derivatives hold a frame for
0.5 or 1.5 seconds starting at 35% of the clip. These are sensitivity controls,
not simulated cache implementations. Severity does not impose a quality label:
freezing an already static video may be imperceptible. All derivatives of a
source share a prompt cluster; none are independent prompt observations.

Normalization can alter the perceived video and metric values. Existing scores
are therefore labelled `original_exploratory`. For primary metric validation,
score the exact anonymous files listed in private/presented_metric_inputs.csv.
The `score` command does this using the seven existing VBench dimensions and
also computes VBench5. It includes synthetic clips and reuses content-matched
scoring caches. Results remain private and never appear in the annotation UI.
Supply a CSV with `clip_id,video_sha256` and supported metric columns; video
hashes are checked. Missing derivative scores stay missing, never zero or
copied from the original. Additional metrics such as DOVER/LPIPS are not yet
integrated; their orientation and applicability need a declared protocol.

## Download for local annotation (no tunnel)

After `package` has completed on the server, run:

```bash
python UNIV_adaptor/scripts/data/export_blind_audit_local.py
```

Download outputs/acceleration_blind_audit_local.zip. Extract it completely on
Windows and double-click START_WINDOWS.cmd inside blind_audit_local. Python
3.10+ is required; no extra libraries, ffmpeg, GPU or VBench are needed locally.
It verifies media, chooses an available localhost port and opens the browser
only after the server is ready. Keep the terminal running. The ZIP contains
anonymous media and a minimal manifest, not the original plan, scores, source
paths or prior raters' answers. This does not rerender or regenerate videos.

Use distinct participant IDs across computers. Send back the JSON files from
study/private/ratings to the researcher; preserve these original IDs and do
not overwrite a different participant's file. They retain the original plan
and package identities and can be aggregated in the original study directory.
The local package intentionally cannot run score/report without the private
researcher manifest. Keep researcher-only files out of participant packages.

To migrate the completed scoring study for local research, stop annotations
while taking the snapshot and use `python
UNIV_adaptor/scripts/data/export_blind_audit_local.py --research`. This requires
complete presented_scores.json/csv and exports ALL study files to
outputs/acceleration_blind_audit_research.zip, including raw score runs and any
existing ratings. Extract on the researcher computer. START_WINDOWS.cmd runs
annotation; ANALYZE_WINDOWS.cmd runs local aggregation using existing scores.
Both require only Python. This larger ZIP contains unblinding information and
must not be distributed as the anonymous participant package. Output ZIPs must
be outside the study directory; existing ZIPs are never overwritten.

## Results and aggregation

If downloading the video ZIP stalls, export the small compressed analysis-only
archive first (does not read/copy video bytes):

```bash
python UNIV_adaptor/scripts/data/export_blind_audit_local.py --research --metadata-only
```

Download outputs/acceleration_blind_audit_analysis.zip. It supports local
analysis but not video playback. To transfer the existing full ZIP in smaller
independently retryable chunks:

```bash
python UNIV_adaptor/scripts/data/export_blind_audit_local.py \
  --split-zip outputs/acceleration_blind_audit_research.zip --part-mib 8
```

Download all files in outputs/acceleration_blind_audit_research.zip.parts,
including parts.json and merge.py, into one local folder. Run `python merge.py`
inside it. The merger checks every part and the complete ZIP checksum before
the archive is used. A previously downloaded incomplete ZIP with the same name
in the parent folder must be moved aside first; the merger never overwrites it.
Splitting is resumable for intact existing parts. If even the small metadata
ZIP cannot transfer, investigate the transfer channel rather than file size.

```bash
python UNIV_adaptor/scripts/data/acceleration_blind_audit.py report
# Once the displayed clips have been scored:
python UNIV_adaptor/scripts/data/acceleration_blind_audit.py report \
  --presented-scores /path/to/presented_scores.csv
```

Report automatically uses private/presented_scores.csv when produced by score;
the explicit path is only for an alternative content-bound score export.

Outputs: analysis/human_consensus.csv, metric_pairs.csv (when comparisons are
available), and report.json. A directional/tie consensus requires at least
three responses and >=2/3 of ALL votes for that choice. Uncertain votes remain
in the denominator. Pairwise human agreement excludes uncertain votes and is
descriptive, not chance-corrected reliability. Metric tie and reversal rates
are reported only on decisive human pairs, separately by source/type/dimension.
Prompt-macro accuracy and prompt-cluster bootstrap intervals are conditional on
these raters and pair selection; this small pilot is not a formal benchmark.
Dynamic Degree ranking is diagnostic; larger motion is not universally better.

Freeze metric calibration on development data, then validate on new prompt
families and held-out methods/models before claiming a new evaluator. No
prompt-router benefit follows automatically from metric disagreement.
