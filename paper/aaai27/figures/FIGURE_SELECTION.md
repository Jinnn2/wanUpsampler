# Final Main-Paper Figure Selection

Status: **locked on 2026-07-19**.

The manuscript should reference only the canonical filenames in the second column. Unselected renderings are retained for provenance under `_archive/unused_alternatives/` and must not be used as paper inputs.

| Figure | Canonical manuscript files | Selected rendering | Authoritative provenance |
|---|---|---|---|
| Fig. 1: Real-frame teaser and acceleration mechanism | `fig_teaser.png`, `fig_teaser.pdf` | Deterministic frame layout plus locked overview crop | `gen_fig_teaser.py`, `fig_teaser_manifest.json`, audited real video frames, and the upper inference panel of `fig_talh_overview.png` |
| Fig. 2: Model-internal supervision | `fig_talh_overview.png`, `fig_talh_overview.pdf` | Cropped ImageGen supervision panel | `fig_talh_overview_imagegen.png`, `imagegen_prompt_talh_overview.md`, and `postprocess_fig_talh_overview.py` |
| Fig. 3: Quality--efficiency | `fig_quality_efficiency.png`, `fig_quality_efficiency.pdf` | ImageGen | `fig_quality_efficiency_imagegen.png`, `imagegen_prompt_quality_efficiency.md` |
| Fig. 4: Component evidence | `fig_component_evidence.png`, `fig_component_evidence.pdf` | Deterministic native plot | `gen_fig_component_evidence.py` and integrated result tables |
| Fig. S1: Extended qualitative comparison | `supplementary/fig_qualitative.png`, `supplementary/fig_qualitative.pdf` | Deterministic native composite | `gen_fig_qualitative.py`, `supplementary/fig_qualitative_manifest.json`, and audited real video frames |

## Publication Notes

- The canonical PNG and PDF of each figure contain the same selected rendering.
- The PDFs for Figs. 2 and 3 are raster-backed wrappers created from the selected ImageGen PNGs; the PNGs are authoritative. Fig. 2 deterministically retains only the supervision panel because its inference panel is reproduced in Fig. 1(b).
- The deterministic Fig. 3 alternative remains the numerical audit source. Verify every displayed ImageGen label and point/bar location against it before submission.
- The experimental video pixels in Fig. 1 and Fig. S1 must remain deterministic. Fig. 1(b) directly reuses the locked inference half of Fig. 2. `Native-HR (estimated)` is rendered from its actual content-aligned 368p trajectory with the same normalized crop coordinates as the 720p handoff outputs; it is not a placeholder.
- Running `gen_fig_talh_overview.py` or `gen_fig_quality_efficiency.py` writes only to `_archive/unused_alternatives/` and therefore cannot overwrite the selected canonical assets.
