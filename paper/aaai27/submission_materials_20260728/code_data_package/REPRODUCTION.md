# Reproduction protocol

## External dependencies

The implementation expects:

- Wan2.1 T2V-1.3B for the 50-step suite;
- Wan2.1 T2V-14B StepDistill-CfgDistill for the 4-step suite;
- LightX2V and DiffSynth-Studio;
- VBench in a separate CUDA 12.1 environment for metric evaluation.

Minimum Python package requirements are provided in `requirements.txt`.
`reproduction_assets.json` pins both public model revisions and the size and
SHA-256 of every final custom checkpoint. Exact experiment-machine versions
must still be recovered from the remote environment.

Plan or download the pinned public assets with:

```bash
python tools/download_public_assets.py --output-root /path/to/public_models
python tools/download_public_assets.py --output-root /path/to/public_models --execute
```

On the original experiment machine, export the custom weights, source snapshot,
Git state, and full software/hardware environment with:

```bash
bash tools/export_full_repro_bundle.sh \
  --project-root "$PROJECT_ROOT" \
  --output "$EXPORT_ROOT/intrascale_full_repro"
```

The exporter rejects missing, truncated, or wrong checkpoints before copying.
Validate the result with:

```bash
python tools/verify_repro_bundle.py \
  --bundle-root "$EXPORT_ROOT/intrascale_full_repro" \
  --require-checkpoints
```

For a smaller evidence-only export that omits weights and latent arrays:

```bash
bash tools/export_missing_repro_metadata.sh \
  --project-root "$PROJECT_ROOT" \
  --output "$EXPORT_ROOT/intrascale_missing_metadata"
```

## Data construction

ITU uses content-aligned clean latent pairs. A target-resolution video generated
by the frozen model is downsampled in RGB space and both resolutions are encoded
independently with the frozen VAE. TTD uses matched prompt, seed, initial noise,
scheduler, and guidance settings to cache the pre-transition state and the
completed low-resolution endpoint.

The full latent LMDBs are not included. The dataset builders and final configs
are included under `code/` and `configs/`. `DATA_APPENDIX.md` records their
construction targets, schema, and deterministic split protocol. The metadata
exporter reads actual shard headers and emits counts, shapes, prompt hashes,
seed ranges, and exact validation indices without copying tensors.

## Training

ITU uses AdamW-style settings encoded in the YAML files: batch size 1, effective
batch size 8, bf16, learning rate `1e-4`, weight decay `0.01`, 50k maximum
steps, EMA `0.9999`, and seed `1234`.

TTD freezes the base denoiser and trains rank-32 LoRA updates on `q,k,v,o` and
the two feed-forward projections. The final configs use bf16, learning rate
`5e-5`, weight decay `0.01`, 10k maximum steps, and seed `1234`. The inference
strength is selected from `{0.25, 0.5, 0.75, 1.0}` on disjoint validation
prompts; `0.75` is fixed for test evaluation.

`data/final_parameters.json` is the complete final inventory.
`data/development_search.json` distinguishes parameters that were swept from
parameters fixed before the final development experiments.

## Evaluation

Wan50 evaluates ten prompt/seed pairs with seeds 9700–9709. Distill4 evaluates
ten prompt/seed pairs with seeds 9800–9809. The state-reconstruction sweep uses
eight disjoint prompts and seed base 16000.

Each reported quality difference uses prompt/seed pairs as the statistical
unit. The package includes paired bootstrap confidence intervals and two-sided
sign tests. Timing is reported separately from quality and uses five warm
repeats after one warm-up.

Metric definitions and motivations are given in the Technical Supplement.
VBench-5 is the unweighted mean of the five named VBench dimensions; it is not
an official VBench aggregate.

## Evidence recovery

The package includes the checksum-verified 50-sample
`368x640 -> 720x1248` ITU operator JSONL. Machine-specific paths are sanitized.
The accompanying sample and summary CSV files are recomputed directly from
those records. The source JSONL SHA-256 is
`e9dccf84dc386b91616e3151d43d5ef19c29f5e4bf8dcb33eb4b862eceaf2c85`.

## Remaining reproduction gaps

1. Exact hardware and software versions are missing from the archived manifests.
2. The five complete custom ITU/TTD checkpoints still need to be exported from
   the original experiment machine.
3. Full from-scratch retraining additionally requires the latent LMDBs, exact
   experiment-machine shard metadata, and training-state archives. The split
   algorithm is supplied; the exporter recovers realized indices from lengths.

These gaps are deliberately not replaced by inferred or fabricated values.
