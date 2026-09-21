# Targeted equal-density spatial versus temporal contrast

This development-only existence test asks whether a deliberately controlled
prompt grouping exposes the expected spatial/temporal preference. It does not
generate Native-HR or the accelerated Phase4 anchor.

- Four low-motion prompts contain dense fine detail and many small objects. The
  expected winner is temporal compression.
- Four high-motion prompts contain one large, simple subject against a plain
  background. The expected winner is spatial compression.
- Every prompt uses base seeds 42, 100, and 2024.
- Spatial-only and temporal-only both have planning proxy density 0.5.
- Total generation is 8 prompts x 3 seeds x 2 cases = 48 videos.

The primary estimand is paired `Q_temporal - Q_spatial`. Lambda and HR quality
do not enter this direct equal-density comparison. Measured end-to-end latency
must nevertheless match within 5%; otherwise the result is a calibration pilot
rather than a clean method comparison.

```bash
cd /mnt/afs_2/houze/wanUpsampler

bash UNIV_adaptor/scripts/run_univ_targeted_st_contrast_8gpu.sh check
bash UNIV_adaptor/scripts/run_univ_targeted_st_contrast_8gpu.sh plan
bash UNIV_adaptor/scripts/run_univ_targeted_st_contrast_8gpu.sh generate
bash UNIV_adaptor/scripts/run_univ_targeted_st_contrast_8gpu.sh finalize
bash UNIV_adaptor/scripts/run_univ_targeted_st_contrast_8gpu.sh score
```

Important outputs are under
`outputs/univ_targeted_st_contrast_v1/metrics/targeted_st_vbench`:

- `paired_st.csv`: seed-level paired S/T differences;
- `prompt_summary.csv`: three-seed prompt means and sign consistency;
- `class_summary.csv`: directional accuracy for the two controlled groups;
- `analysis.json` and `report.md`: difference-in-differences, prompt bootstrap
  interval, and measured-latency gate.

## Temporal latency calibration follow-up

If the density-0.5 temporal arm is more than 5% slower than the spatial arm,
reuse the 24 spatial videos and generate only a more aggressive temporal arm:

```bash
bash UNIV_adaptor/scripts/run_univ_targeted_st_temporal_calibration_8gpu.sh check
bash UNIV_adaptor/scripts/run_univ_targeted_st_temporal_calibration_8gpu.sh plan
bash UNIV_adaptor/scripts/run_univ_targeted_st_temporal_calibration_8gpu.sh generate
bash UNIV_adaptor/scripts/run_univ_targeted_st_temporal_calibration_8gpu.sh finalize
bash UNIV_adaptor/scripts/run_univ_targeted_st_temporal_calibration_8gpu.sh score
```

The follow-up uses requested temporal ratio 0.36, which resolves to actual
temporal ratio 0.35 and actual token proxy density 0.48. It writes a new output
root, reuses every source spatial artifact by content hash, and applies the same
measured-latency gate to the resulting 48-video paired dataset.
