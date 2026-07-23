# Distill4 Final Experiment Archive (2026-07-23)

This directory is the Git-visible archive for the final four-step experiments
used by the current AAAI 2027 paper.

## Canonical data

Use `distill4_p0_p1_p3_final_20260723T064558Z/` for all paper claims and
tables. It contains the refreshed:

- P0 MrFlow-style RGB endpoint result with direct correction at
  \(\sigma=0.12\);
- P1 VBench temporal-flickering evaluation and paired statistics;
- P3 independent \(4\times2\) TrajScale validation sweep;
- warm-model pipeline and denoising timing tables;
- benchmark specifications, configurations, artifact fingerprints, manifests,
  and implementation snapshots.

The selected P3 configuration is `talh3_s0p75_random`: EAA strength 0.75 with
random re-noising. The final paper comparison uses `interp3`, `talh3`,
`endpoint_stage2_2hr`, and `endpoint_rgb_1hr`.

`paper_tables/` contains the compact CSVs copied directly into the paper:

- `distill4_p0_p1_p3_final_table_20260723.csv`;
- `distill4_paired_statistics_20260723.csv`;
- `distill4_p3_validation_20260723.csv`.

## Earlier complete 18-case snapshot

`distill4_quality_efficiency_final_20260722T174918Z/` is retained for
traceability. It is the earlier complete 18-case export before the final P0
RGB-1HR metric refresh and P1/P3 additions. Do not use it in place of the
canonical 2026-07-23 results.

## Original exports and integrity

`source_archives/` contains the two byte-identical tarballs downloaded from
the evaluation machine. Each extracted export also contains its original
`SHA256SUMS` and manifest.

Archive SHA256:

```text
6b8aac733b631468d2ae1adde5522ff36001f5ef56e78689d74d3ed6dabc4493  distill4_quality_efficiency_final_20260722T174918Z.tar.gz
d69aae7a1427fce98d879bc0dfb170229e283323f8b790d2446ac57e0de366e4  distill4_p0_p1_p3_final_20260723T064558Z.tar.gz
```

## Scope limitation

These exports intentionally contain no generated videos, model checkpoints,
or pretrained weights. The final manifest records 180 validated main-suite
videos and 64 validation videos on the evaluation machine, but
`include_videos` is false. This repository archive therefore preserves all
exported metrics and provenance, not the large remote generation artifacts.
