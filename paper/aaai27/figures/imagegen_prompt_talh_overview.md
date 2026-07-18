# ImageGen Prompt: TALH Overview Alternative

Use case: infographic-diagram

Asset type: full-width AAAI paper architecture figure; pure English; landscape canvas with an aspect ratio close to 2.2:1.

Input image: the existing deterministic TALH overview is a content and color reference only. Create a visually distinct, more polished alternative rather than tracing it pixel for pixel.

Primary request: create a clean, publication-quality architecture diagram for the TALH video-generation acceleration framework. Preserve the exact scientific structure, directions, schedules, and module relationships below.

Style/medium: modern minimal academic infographic with vector-like geometry, crisp flat shapes, fine lines, generous whitespace, pure white background, restrained Times-compatible or highly legible serif typography. No shadows, gradients, decorative icons, clip art, 3D effects, or photorealistic elements.

Color palette:

- Native-HR and frozen modules: gray `#7B8794`.
- Low-resolution prefix and LR paths: blue `#4A90D9`.
- TAA: magenta `#CC79A7`.
- CLL: green `#009E73`.
- HTR and HR suffix: vermillion `#D55E00`.
- Main text: charcoal `#242424`.
- Secondary text: gray `#666666`.
- Section background: very pale warm gray `#F7F7F5`.

Composition: two wide horizontal bands, both read left to right.

Upper band title, rendered exactly once: `INFERENCE: HYBRID-RESOLUTION TRAJECTORY`

Upper-left schedule block:

- Progression label: `Structure & Motion` followed by a right arrow followed by `Texture & Detail`.
- Row `Native-HR`: 50 HR evaluations.
- Row `TALH-Q`: 40 LR evaluations followed by 10 HR evaluations; show the handoff at step 40.
- Row `TALH-E`: 45 LR evaluations followed by 5 HR evaluations; show the handoff at step 45.
- Show step labels `0`, `40`, `45`, and `50`.

Upper-right handoff pipeline, rendered in exactly this order:

`LR Prefix` -> `TAA` -> `Aligned Clean LR` -> `CLL` -> `Lifted Clean HR` -> `HTR` -> `HR Suffix`

Small module annotations:

- Under `TAA`: `LoRA; base frozen` and `E_traj(s)`.
- Under `CLL`: `Clean LR -> Clean HR` and `E_lift(s)`.
- Under `HTR`: `Target-resolution re-noise` and `E_refine(s)`.

Lower band title, rendered exactly once: `MODEL-INTERNAL SUPERVISION`

Lower-left subpanel title: `TAA: trajectory-alignment pair`

Lower-left flow:

- `Frozen Wan LR rollout` branches to `Cached state x_s^L` and `Full endpoint z_T^L`.
- Both connect to `TAA training pair`.
- Add the condition line `same prompt / seed / scheduler / CFG`.
- Indicate that the cached state is the inference state.

Lower-right subpanel title: `CLL: cross-resolution lifting pair`

Lower-right flow:

- `Frozen Wan` -> `HR video`.
- The upper branch is `Wan VAE` -> `z_0^H`.
- The lower branch is `RGB downsample` -> `same Wan VAE` -> `z_0^L`.
- Visually pair `z_0^L` and `z_0^H` as the CLL supervision pair.

Footer, rendered exactly once: `No external paired videos, SR weights, or extra teacher`

Constraints:

- All text must be English and rendered verbatim; do not add, omit, paraphrase, or duplicate labels.
- Scientific notation must remain legible: `x_s^L`, `z_T^L`, `z_0^H`, `z_0^L`, `E_traj(s)`, `E_lift(s)`, `E_refine(s)`.
- Keep every arrow direction and branch exactly as specified.
- Make the handoff pipeline the main visual focus.
- Use color only as a secondary encoding; preserve borders and spatial grouping for grayscale readability.
- No figure number, caption, logo, watermark, decorative illustration, or unrequested text.
- Do not insert neural-network layer diagrams or video frames.
