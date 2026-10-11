# Five-method horizontal video review with VBench labels

After full FlashVSR scoring has completed on the server:

```bash
bash UNIV_adaptor/scripts/run_univ_wan21_comparison_videos.sh all
```

Default: **24 groups**, 12 prompts × seeds 42/3407, **five horizontally arranged
panels** in a fixed order: FULL50 → STEP25 → TeaCache → ScalingCache → native
LR+FlashVSR. Old custom S/T and refinement controls are not included in this
main five-method comparison. Each original video retains **832×480**, with all
81 frames at 16 fps. Output is **4160×968**, including separate text bands;
labels never cover video pixels. There is no crop, enhancement, interpolation,
generation or scoring. Original files remain untouched.

Each panel displays method name, **VBench-5 raw mean**, all **seven individual
dimension scores**, and the raw-mean difference from matched FULL50. Values
have four decimal places. The displayed mean is an explicitly labelled
arithmetic mean of the five existing quality dimensions, **not official VBench
Total**. Dynamic degree is not uniformly higher-is-better. Source score/video
hashes and prompt/seed identities must match the finished scoring plan.

The panorama is re-encoded using libx264 CRF12, so its scores are labels of the
**original individual videos**, not scores of the derived montage. Inspect
original videos/lossless frames for definitive fine-detail judgments. Frame
timestamps start together; same seed does not imply FULL/LR pixel alignment.

## Download and view

Download `outputs/wan21_five_method_comparisons_v1_review.tgz`, extract it, and
open **`index.html`** locally. It works offline without a web server or port
forwarding. The page supports fit-to-window and native-size horizontal scroll;
individual videos are at `videos/p00_s42.mp4`, etc. `scores_by_panel.csv` maps
all 120 source videos to displayed scores. `plan.json`, receipts and the
manifest preserve source-video hashes, scores, layout and encoding provenance.

This is **labelled exploratory review**, not blinded preference collection.
It helps inspect whether similar scores hide prompt/detail failures, but it
does not independently establish metric insufficiency or routing benefit.

## Requirements and optional settings

Reuse the existing VBench Python with Pillow, FFmpeg and FFprobe. It runs on
CPU with two parallel jobs and two encoding threads per job by default; no GPU
is used. Fonts are discovered from system DejaVu/Liberation or matplotlib.
Set `FONT_FILE=/absolute/path/to/existing/font.ttf` if needed; nothing is installed.

- `FLASH_SCORE_ROOT`: finished full-length scoring directory.
- `COMPARISON_ROOT`: new independent presentation output directory.
- `JOBS`, `ENCODE_THREADS`: CPU concurrency.
- `FFMPEG`, `FFPROBE`: existing executable paths.
- `PANEL_WIDTH`: even value 416–832. Default 832 preserves native resolution;
  smaller values explicitly downsample with Lanczos. Use a **new output root**
  for a different width or CRF; do not mix variants into one frozen plan.

To inspect one group before rendering everything:

```bash
COMPARISON_ROOT=outputs/wan21_comparison_smoke \
bash UNIV_adaptor/scripts/run_univ_wan21_comparison_videos.sh all --group-ids p02_s3407
```

Completed hash-verified receipts are skipped when rerunning `all`. If interrupted
mid-output, inspect/move the named unreceipted header/video/partial/log artifacts
before resuming; they are never overwritten automatically. Archived packages
are also checked rather than overwritten. No score-ranked selection or quality
color-coding is performed.
