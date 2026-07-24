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
