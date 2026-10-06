# FlashVSR existing-asset diagnosis

This is **diagnosis, not a new method benchmark or a trained selector**. Maintain
the claim: different prompts may have different sensitivity to acceleration
mechanisms/budgets; stable, predictable routing gains still need evidence.
Do not choose a metric or keep only prompts that make FlashVSR look good/bad.

## Scope and author sources

- Author repository: https://github.com/OpenImagingLab/FlashVSR
- Recipe: `examples/WanVSR/infer_flashvsr_v1.1_tiny.py`, pinned at
  `cf910c61a60733e610e9c6e8b607f80c3a6c202b`.
- Block-Sparse-Attention pinned at `49d6c39e4dc0303442cda3bb758b3925d4399c49`.
- Author v1.1 weights: https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
  Download resolves a commit once, downloads that revision and hashes the three
  required files. Workers check the frozen size/mtime identity. To fully rehash,
  run `download` again. No generation model training/download is needed.
- Source assets: `outputs/published_wan21_study_v2` ONLY. Frozen study/record/MP4
  and saved clean endpoint hashes are checked. Source assets are never changed.

Author `examples/WanVSR/prompt_tensor/posi_prompt.pth` is a regular 4,195,504-byte
Git blob in this commit, not a Git-LFS pointer. The diagnostic loads it directly
from `git cat-file blob` into an in-memory stream, bypassing working-tree
checkout/clean/smudge conversions. It verifies the pinned HEAD blob, unchanged
index, canonical object size and raw SHA256
`4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef`.
Only an unstaged modification of this specific binary can be isolated: the
working file is preserved but NEVER loaded, even if byte-different. Its actual
hash, whether it matches, and filter attributes (when marked `M`) are recorded
in the plan; a mismatch prints a warning. This does NOT establish the cause of
the modification or claim byte differences are metadata-only. Staged changes,
missing/symlink files, corrupt Git objects, source edits and untracked additions
remain errors. No reset, assume-unchanged flag, file replacement or global Git
config change is made. Thus the executed tensor remains the author's exact
input, not the differing working-tree tensor or a relaxed hash check.

New environment `.venv_flashvsr_diagnostic` uses the author Torch 2.6/CUDA 12.4
dependencies. **Do not install these into the native Wan/VBench environments.**
Setup needs Python 3.11, a matching CUDA compiler/toolkit, a working NVIDIA
driver, disk space for a separate Torch install and the author VSR weights.
Compiling the sparse backend can take time; `MAX_JOBS=2` bounds build RAM.
Kernel import success is NOT CUDA inference success. `check` loads weights;
`run` requires a full-shape warmup/output-shape gate before saving results.

## Five matched display tracks

| Track | Input and purpose |
|---|---|
| FULL | Existing FULL50 original MP4, fixed first N frames |
| HR_DOWN4_BICUBIC | FULL -> BOX x1/4 each axis -> bicubic FULL; interpolation control |
| HR_DOWN4_FLASH | EXACT same degraded FULL -> author x4 FlashVSR; SR-module fidelity |
| NATIVE_LR_BICUBIC | Verified `main_clean` from S_B025/S_B050 -> native VAE -> bicubic FULL |
| NATIVE_LR_FLASH | SAME decoded main LR -> author x4 -> Lanczos FULL; composite diagnostic |

FULL is already lossy MP4. Native LR is re-decoded from the saved unrefined
latent; no old RealESRGAN, VAE re-encode or HR4 is included. These different
provenances are recorded and are a limitation, not hidden pixel equivalence.

**Native-LR vs FULL is not pixel-aligned reconstruction ground truth.** Their
subjects/trajectories can differ even at the same seed. PSNR/MAE are computed
only for the HR-downsample controls. Gradient/temporal-change statistics are
signal diagnostics, not quality scores. There is no invented total score.

Author demo preprocessing center-crops to 128-multiple dimensions and floors
the temporal length, potentially losing view/tail frames. This adapter instead:

1. Pads LR right/bottom with edge pixels to multiples of 32, then bicubic x4.
2. Repeats the last frame to a sufficiently long 8n+1 input. Tiny's raw output
   must have exactly `F-4` frames. No short output is silently accepted.
3. Removes ONLY padded spatial borders and tail frames.
4. Resizes x4 output to the original FULL canvas when needed.

This preserves the inspection content, but is an **explicitly adapted x4 then
resize pipeline**, NOT an official/native x2 recipe or reproduction of paper
numbers. `topk_ratio` follows the author's size-dependent formula. Required
LCSA/block-sparse attention is retained; there is no dense/CPU/bicubic fallback.
The author's sparse ratio 2, KV ratio 3, local range 11 and color correction are
fixed. The restoration noise seed is fixed at 0, separate from generation seed.

## Remote commands

Run from `/mnt/afs_2/houze/wanUpsampler`. Setup downloads/builds only VSR assets.

```bash
bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh setup
bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh plan
bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh prepare
bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh check
bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh run
bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh report
bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh export
```

Stop if any command fails; do not blindly paste later commands. Actual worker
tracebacks are in `outputs/flashvsr_asset_diagnostic_v1/logs`. Existing outputs
without receipts are never overwritten: inspect/move them before resuming.
Valid receipts allow resume. Code/protocol changes require a fresh output root.
Use `status` to verify completed receipts while/after running. Kernel builds
use a separate retained checkout under outputs, keeping the audited source clean.

Defaults: p02/p05/p08/p11 (the four planned motion/detail cells), seeds42/3407,
S_B025, **first33 frames**. 8 matched pairs/16 FlashVSR restoration jobs. All
prompt choices are nominal design labels, not verified generated contents.
33-frame screening can miss late motion/defects and cannot establish a paper
claim. Full-duration follow-up must use a NEW plan/output:

```bash
FLASH_DIAG_ROOT="$PWD/outputs/flashvsr_asset_diagnostic_full_v1" \
bash UNIV_adaptor/scripts/run_univ_flashvsr_asset_diagnostic_8gpu.sh plan --frames 81
# Set the SAME FLASH_DIAG_ROOT for prepare/check/run/report/export afterwards.
```

To first inspect the previously discussed p00/s3407 spatial counterpart, use
`plan --prompt-ids p00 --seeds 3407 --spatial-arms S_B025` in another fresh root.
Never substitute its T_B025 clip for a spatial LR endpoint. Include S_B050 via
`--spatial-arms S_B025 S_B050`; its x4 restoration canvas is more expensive.
`NGPUS=1` is useful for a tiny initial smoke experiment.

## Review and handoff

`review.html` is a **labelled diagnostic viewer**, not blinded human evaluation.
Download/extract the export and open the HTML locally; no VSCode tunnel needed.
`manual_review.csv` records subject/prompt fulfillment, fine detail, motion/
flicker and overall preference separately. Do not overwrite filled reviews.
Use PNGs at native scale for fine details; MP4s use the same CRF16 and are
navigation previews. Different tracks are not secretly sharpened or normalized.

Export: `outputs/flashvsr_asset_diagnostic_v1_analysis.tgz` includes JSON receipts,
logs, viewer, CSV, PNGs and MP4 previews. It excludes huge model/latent/NPZ files.
Upload that archive for analysis; original raw NPZ remains available remotely.
The export contains methods/prompt labels; it is NOT a blinded rater bundle.

Interpretation gates:

- If HR-downsample control fails, investigate/reject this VSR configuration
  before blaming native low-resolution generation.
- If the SR control works but native LR has already changed the subject, the
  bottleneck is upstream generation; do not claim SR can repair it.
- If native LR preserves subject/motion but loses detail, compare bicubic and
  FlashVSR for genuine restoration AND hallucination/flicker.
- If visual quality is similar, that is a valid negative result; do not tune
  selection/metrics to manufacture content-dependent degradation.

Timing receipts separately report model load, excluded per-shape warmup,
restoration inference, preprocessing/postprocessing and original main denoising.
VAE preparation decodes the entire saved LR clip even for prefix review.
**No end-to-end speedup is computed** from these disjoint/prefix timings.
Full pipeline comparison must include text encoding, LR denoising, decoding,
VSR and the chosen output policy, with matched warmup/timing boundaries.
