# ImageGen Prompt: Component Evidence Alternative

Use case: infographic-diagram

Asset type: full-width AAAI paper ablation figure; pure English; landscape canvas close to 3.0:1.

Input image: the deterministic source is authoritative for all data and semantics. Produce a visually refined alternative while preserving all values, labels, encodings, row/column orders, and conclusions.

Style: modern minimal vector-like scientific plotting, white background, crisp fine axes, restrained serif typography, compact but breathable spacing, no gradients, shadows, 3D effects, pictograms, logos, watermarks, or caption text.

Palette: CLL dark green `#009E73` and light green `#6CC4A8`; unaligned gray `#B0B0B0`; TAA magenta `#CC79A7`; interaction heatmap sequential blue; negative cell outlined in orange `#E69F00`; main text `#242424`.

Panel (a) title: `(a) CLL lifting`

- Y-axis: `Error reduction vs. trilinear (%)`, range 0 to 100.
- Categories in exact order: `Latent L1`, `LPIPS`, `Temporal L1`.
- Legend series: `480x832 -> 720p` and `368x640 -> 720p`.
- Exact values for `480x832 -> 720p`: 48.6, 73.3, 30.8.
- Exact values for `368x640 -> 720p`: 33.4, 51.1, 15.5.
- Use paired green bars with distinct diagonal hatch directions and print every value above its bar.

Panel (b) title: `(b) TAA alignment`

- X-axis: `Endpoint L1 (lower is better)`.
- Rows in exact order: `Wan50 @ 40`, `Wan50 @ 45`, `Distill4 @ 3/4`.
- For each row show a thin horizontal connector from gray `Unaligned` circle to magenta `TAA-aligned` triangle.
- Exact endpoint pairs and annotations:
  - `Wan50 @ 40`: 0.03215 -> 0.02385, `-25.8%`.
  - `Wan50 @ 45`: 0.02363 -> 0.01866, `-21.0%`.
  - `Distill4 @ 3/4`: 0.04286 -> 0.04070, `-5.0%`.
- Legend labels: `Unaligned`, `TAA-aligned`.

Panel (c) title: `(c) Factorial interaction`

- Heatmap Y-axis rows in exact order: `Wan50 @ 40`, `Wan50 @ 45`, `Distill4 @ 3/4`.
- Heatmap X-axis columns in exact order: `Unaligned + Trilinear`, `TAA + Trilinear`, `Unaligned + CLL`, `TAA + CLL (TALH)`.
- Cell values, row by row:
  - 0.000, +0.0009, +0.031, +0.032.
  - 0.000, -0.0004, +0.041, +0.040.
  - 0.000, +0.001, +0.039, +0.039.
- Colorbar label: `Delta VBench-5`, spanning 0.000 to 0.045 for nonnegative cells.
- Show each value inside its cell. Keep the -0.0004 cell white and emphasize it with a thin orange border.

Constraints:

- All text must be English and rendered exactly as specified; use the Greek delta glyph in the colorbar only if it remains crisp, otherwise spell `Delta`.
- Do not change, add, omit, round away, or reorder any value.
- Keep all three panels aligned and visually balanced, with panel (c) slightly wider for its four columns.
- No external caption, footnote, figure number, decorative annotation, or unrequested text.

