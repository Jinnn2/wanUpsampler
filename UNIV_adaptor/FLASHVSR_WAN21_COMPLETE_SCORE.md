# Score the complete FlashVSR comparison (four GPUs)

After `run_univ_flashvsr_wan21_complete_4gpu.sh all` has completed:

```bash
bash UNIV_adaptor/scripts/run_univ_flashvsr_wan21_complete_score_4gpu.sh all
```

Or select four other cards:

```bash
GPU_IDS=4,5,6,7 bash UNIV_adaptor/scripts/run_univ_flashvsr_wan21_complete_score_4gpu.sh all
```

This scores **only the 24 new 81-frame FlashVSR videos**, using the existing
VBench environment/checkout. Seven dimensions are evaluated sequentially,
each with four distributed ranks. Exactly matching completed dimension runs
can be reused after interruption; rerun `all` to resume. No setup, compilation,
generation, shared-cache cleanup or overwrite of the generation study occurs.

VBench commit remains `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`, `custom_input`
mode, `imaging_quality_preprocessing_mode=longer`. The dimensions are
subject consistency, background consistency, motion smoothness, aesthetic
quality, imaging quality, dynamic degree and overall consistency. These are
**seven selected dimensions, not all 16 dimensions or official VBench Total**.
They are reported separately; no new aggregate score is invented.

The prior FULL50, STEP25, TEA008 and SCALING10 **96 scores** are reused from
`published_wan21_study_v2/metrics`: frozen CSV/provenance, raw result/request/
full-info hashes, video inventory, exact prompt/seed, scoring code, VBench
checkout and preprocessing must agree. Raw values must match CSV values.
Incompatible/missing results stop the process rather than silently rescoring,
filling zeros, or mixing 33-frame screening scores into the complete comparison.
All 120 original videos are checked for 81 frames, 832×480 and 16 fps.

Defaults:

- Source: `outputs/flashvsr_wan21_complete_v1` (`FLASH_COMPLETE_ROOT`).
- Independent scoring: `outputs/flashvsr_wan21_complete_score_v1` (`FLASH_SCORE_ROOT`).
- VBench: `/mnt/afs_2/houze/VBench` (`VBENCH_ROOT`).
- Scoring Python: `/opt/conda/bin/python` (`VBENCH_PYTHON`); orchestrator
  `BASE_PYTHON` defaults to the same existing interpreter.

## Results to send back

Download **`outputs/flashvsr_wan21_complete_score_v1_analysis.tgz`**. It is a
small analysis-only package, containing scores, CSVs and the original/raw
scoring evidence, without videos, model weights or endpoint arrays. The
separate generation `_videos.tgz` remains unchanged.

Main outputs:

- `per_video_vbench.csv`: 120 videos × seven dimensions.
- `method_summary.csv`: five method means, all based on 24 matched videos.
- `paired_deltas_vs_full.csv`: four methods relative to matched FULL50.
- `flash_vs_methods.csv`: FlashVSR relative to each of the four existing methods.
- `per_prompt_deltas.csv`: per-prompt mean deltas across the two seeds.
- `scores.json`, `report.json`, `vbench/`, `reused_vbench/`: provenance and raw scores.

Deltas are method minus FULL, or Flash minus the named existing method.
Dynamic degree is not uniformly higher-is-better. Flash restoration-only
time does not establish complete-pipeline speedup; no speedup is fabricated.
These comparisons provide matched exploratory evidence for the paper claim
that similar scores may hide explicit prompt/detail failures. Visual or human
validation is still required; score proximity alone proves neither metric
insufficiency nor prompt-only routing benefit.
