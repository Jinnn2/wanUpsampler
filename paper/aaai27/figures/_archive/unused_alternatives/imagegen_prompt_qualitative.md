# ImageGen Prompt: Qualitative Comparison Alternative

Use case: precise-object-edit

Asset type: full-width AAAI qualitative comparison figure; pure English; landscape canvas close to 1.8:1.

Input image: this is a scientific evidence plate assembled from real experiment frames. Preserve every embedded full frame, crop, temporal frame, bounding box, and row-to-row visual difference exactly. Change only the surrounding layout, typography, spacing, dividers, borders, and label hierarchy. Never regenerate, repaint, retouch, sharpen, blur, color-grade, crop differently, or replace any video content.

Primary request: refine the current two-panel evidence plate into a cohesive, modern, minimal publication figure with stronger alignment and compact AAAI-friendly typography.

Style: white background, restrained serif typography, thin neutral rules, consistent margins, flat vector-like labels. No shadows, gradients, icons, illustrations, logos, watermarks, artificial textures, or caption block beyond the specified footer.

Color semantics: Native-HR gray `#7B8794`; Trilinear light gray `#B0B0B0`; CLL green `#009E73`; TALH-Q blue `#0072B2`; TALH-E orange `#E69F00`; Crop A green `#009E73`; Crop B magenta `#CC79A7`; temporal ROI orange `#E69F00`; main text `#242424`.

Left panel title: `(a) Spatial detail -- prompt 05, seed 9705`

- Column headers: `Full frame`, `Crop A`, `Crop B`.
- Rows in exact order: `Native-HR (estimated)`, `Trilinear @ 40`, `CLL-only @ 40`, `TALH-Q @ 40`.
- Keep the green Crop A and magenta Crop B boxes and crop borders attached to exactly the same image regions.
- In the Native-HR row, the two crop cells remain neutral placeholders reading `Context reference only`.
- Under the Native-HR row label retain `368p aligned`.

Right panel title: `(b) Temporal behavior -- prompt 07, seed 9707`

- Column headers: `Context`, `t-4`, `t-2`, `t`, `t+2`, `t+4`.
- Rows in exact order: `Native-HR (estimated)`, `Trilinear @ 45`, `CLL-only @ 45`, `TALH-E @ 45`.
- Preserve the orange ROI box at exactly the same location and preserve every temporal frame in the same column and row.
- Under the Native-HR row label retain `368p aligned`.

Footer, rendered exactly once:

`Native-HR (estimated) is a content-aligned 368p reference; all other rows are 720p outputs.`

Constraints:

- Preserve all real video pixels exactly; do not synthesize any visual content.
- All text must be English and rendered exactly as specified.
- Align row labels, image cells, crop cells, temporal strips, and panel baselines precisely.
- Use color only for method accents and ROI/crop identities; keep the rest neutral.
- No unrequested text, caption, figure number, arrows, quality claims, or decorative annotations.

