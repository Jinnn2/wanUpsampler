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

Run the LR-endpoint label revisions for Figures 1 and 3 from `rewrite` with:

```powershell
python figure_sources/edit_fig_teaser.py
python figure_sources/edit_fig_challenge_interpolation.py
```

These scripts regenerate:

- `figures/fig_teaser.pdf`
- `figures/fig_challenge_interpolation.pdf`

The corresponding `*_source.pdf` files preserve the original raster pixels.
The scripts replace only the legacy first-row/first-column labels with
`LR endpoint`; decoded frames and crop regions are unchanged.
