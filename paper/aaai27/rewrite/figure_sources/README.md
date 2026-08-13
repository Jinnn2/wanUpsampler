# Editable figure sources

Run the terminology-sensitive figure build from `rewrite` with:

```powershell
python figure_sources/render_ttd_figures.py
```

The script regenerates:

- `figures/fig_overall_framework.pdf`
- `figures/fig_challenge_alignment.pdf`

`overall_framework/fig_overall_framework_template.pdf` is a label-free vector
template derived from the canonical framework figure. The accompanying HTML,
manifest, render script, and image assets are the recovered editable layout
for future structural redesigns.

`challenge_alignment/fig_challenge_alignment_template.png` is a label-free
template derived from the canonical 300-DPI comparison figure. It avoids the
original generator's dependency on external source videos.

Render Figure 3 from the matched prompt-08/seed-9708 video group with:

```powershell
python figure_sources/render_fig_challenge_interpolation.py
```

The script deterministically extracts the midpoint frame, applies the canonical
centered `3x` crop, and regenerates:

- `figure_sources/fig_challenge_interpolation_source.png`
- `figure_sources/fig_challenge_interpolation_source.pdf`
- `figure_sources/fig_challenge_interpolation_source_manifest.json`
- `figures/fig_challenge_interpolation.png`
- `figures/fig_challenge_interpolation.pdf`

Run the LR-endpoint label revision for Figure 1 from `rewrite` with:

```powershell
python figure_sources/edit_fig_teaser.py
```

This script regenerates:

- `figures/fig_teaser.pdf`

The corresponding `fig_teaser_source.pdf` preserves the original raster pixels.
The script replaces only the legacy first-row label with `LR endpoint`; decoded
frames and crop regions are unchanged. The older
`edit_fig_challenge_interpolation.py` remains available for label-only edits to
legacy Figure 3 sources.
