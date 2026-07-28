# InTraScale Supplementary Code and Data

Anonymous review package for the paper
“InTraScale: Accelerating Video Generation via In-Trajectory Resolution Scaling.”

## Scope

This package contains the method implementation, training and inference
configurations, evaluation scripts, prompt lists, per-sample metric exports,
paired statistics, and the compact tables used in the paper and supplement.

The package intentionally does not contain:

- the public Wan/LightX2V base-model weights;
- custom ITU or TTD checkpoints;
- full generated-video sets or latent-pair LMDBs;
- machine-specific absolute paths.

The five exact custom-checkpoint sizes and SHA-256 values are recorded in
`reproduction_assets.json`. The `tools/` directory provides pinned public-model
download, experiment-machine export, raw-evidence recovery, and bundle
verification scripts. Consequently, the package supports code and result
auditing and provides a deterministic path to an end-to-end reproduction
bundle, but the final custom weights and experiment-machine environment export
still need to be collected.

## Layout

- `code/`: method, training, inference, and evaluation sources.
- `configs/`: final training recipes for Wan50 and Distill4.
- `data/prompts/`: the 10 matched test prompts and 8 disjoint validation prompts.
- `data/final_parameters.json`: complete final method/training/inference settings.
- `data/development_search.json`: values tried and selection criteria.
- `data/derived/`: compact tables and paired statistical tests.
- `data/raw_metrics/`: sanitized per-sample VBench metric JSONs.
- `data/operator_368p/`: 50-sample raw ITU operator evidence, sample table,
  recomputed summary, and provenance.
- `tools/`: public-asset downloader, remote export, and integrity validation.
- `reproduction_assets.json`: pinned public revisions and exact custom weights.
- `CASE_NAME_MAP.csv`: immutable internal case IDs mapped to paper terminology.
- `REPRODUCTION.md`: protocol and dependency instructions.
- `DATA_APPENDIX.md`: generated-data construction, counts, schemas, and splits.
- `CODE_TO_PAPER_MAP.md`: implementation-to-method-section mapping.

## Evidence conventions

- All system comparisons use matched prompts and seeds within each suite.
- `VBench-5` is the unweighted mean of Subject Consistency, Background
  Consistency, Motion Smoothness, Aesthetic Quality, and Imaging Quality. It is
  not an official VBench aggregate.
- Warm latency uses one untimed warm-up followed by five CUDA-synchronized
  generations in one resident process.
- Legacy internal identifiers such as `talh`, `cll`, and `stage2` are retained
  in raw files for provenance. Their manuscript names are InTraScale, ITU-only,
  and ITU, respectively.

Run `python tools/check_source_closure.py` to verify that every paper-specific
internal Python import referenced by the supplied snapshot is present.
