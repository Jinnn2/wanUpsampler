# ImageGen Prompt: Quality--Efficiency Alternative

Use case: infographic-diagram

Asset type: full-width AAAI paper result figure; pure English; landscape canvas close to 2.9:1.

Input image: treat the deterministic chart as the authoritative source for every value, label, ordering, and relationship. Redesign only the visual presentation. The output is a style candidate and must remain numerically auditable against the deterministic source.

Primary request: produce a clean, publication-quality two-panel scientific figure showing TALH quality--latency operating points and per-dimension VBench-5 changes.

Style: modern minimal vector-like statistical graphic, white background, crisp axes, restrained serif typography, generous whitespace, no gradients, shadows, 3D effects, icons, illustrations, logos, watermarks, or caption text. Preserve grayscale readability through marker shapes and bar hatching.

Palette: Native-HR gray `#7B8794`; TALH-Q blue `#0072B2`; TALH-E orange `#E69F00`; Endpoint Re-entry dark gray `#5F6B73`; main text `#242424`; secondary guides `#B8B8B8`.

Panel (a) title: `(a) Quality--latency operating points`

- X-axis: `End-to-end latency per video (s)  <-  faster`.
- Y-axis: `VBench-5 (higher is better)`.
- Plot exactly four points with these coordinates and marker identities:
  - `Native-HR`: 253.10 s, 0.82836, gray square.
  - `TALH-Q  1.83x`: 138.36 s, 0.80983, blue circle.
  - `TALH-E  2.22x`: 114.26 s, 0.80792, orange triangle.
  - `Endpoint Re-entry`: 86.45 s, 0.80093, dark-gray diamond.
- Connect the four points with one thin light-gray dashed trend line in increasing latency order.
- Keep labels clearly separated from points and axes. Do not imply an interpolation curve or uncertainty band.

Panel (b) title: `(b) VBench change from Native-HR`

- Y-axis: `Absolute score change` with a strong zero baseline.
- X-axis categories, in this exact order: `Subject`, `Background`, `Motion`, `Aesthetic`, `Imaging`.
- Two grouped bars per category: TALH-Q in blue with forward diagonal hatching and TALH-E in orange with backward diagonal hatching.
- Exact TALH-Q values: `-0.034267`, `-0.010140`, `-0.005021`, `-0.000415`, `-0.042824`.
- Exact TALH-E values: `-0.037810`, `-0.011166`, `-0.005203`, `-0.014609`, `-0.033432`.
- Legend labels: `TALH-Q` and `TALH-E`.
- Show compact value labels with four decimal places where legible; never round a nonzero value to zero.

Constraints:

- All text must be English and rendered exactly as specified.
- Do not alter, invent, omit, or swap any numerical value, coordinate, category, or legend mapping.
- Panel (a) and panel (b) must have equal visual weight and aligned top/bottom bounds.
- No figure number, external caption, footnote, decorative annotation, or unrequested text.

