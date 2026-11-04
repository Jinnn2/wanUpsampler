# How the accelerators we compare were evaluated by their own authors

Compiled 2026-11-04 from the papers and official READMEs. Items marked **[unverified]** came from a summary of the
page or a third-party listing, not from the paper's own table; check them against the PDF before quoting.

## 1. Methods already in the benchmark

| Method (arm) | Base models in the paper | Wan2.1-1.3B? | VBench as reported | Other metrics | Prompt set / seeds | Per-prompt or failure analysis |
| --- | --- | --- | --- | --- | --- | --- |
| DVG (DVG_D050/D070), arXiv 2605.21042 | HunyuanVideo 720p; Wan2.2-14B 480p×81; HunyuanVideo-1.5 T2V/I2V | **no** | Quality / Semantic / Total only | PSNR/SSIM/LPIPS vs the un-accelerated output (I2V table only); "small-scale user study" mentioned, no results | standard 946 prompts, seed 42 [unverified for the seed]; no error bars | none; limitations paragraph only |
| TeaCache (TEA008/TEA015), arXiv 2411.19108 | Open-Sora, Open-Sora-Plan, Latte | no (Wan support added later in the repo) | VBench (headline "−0.07%" on Open-Sora-Plan at 4.41×) | PSNR, SSIM, LPIPS vs the original model's output; latency | VBench prompts (official eval README) | not found [table values unverified: the arXiv page could not be fetched] |
| ScalingCache (SCALING10), ICLR 2026 (README only; no arXiv found) | Wan2.1-1.3B and 14B, HunyuanVideo, FLUX | **yes** | one "VBench (%)" number; Total/Quality not stated | PSNR, SSIM, LPIPS vs the original output | not stated | none |
| rCM (RCM4/2/1), arXiv 2510.08431 | Wan2.1 T2V 1.3B and 14B at 832×480×81; Cosmos-Predict2 | **yes** | Total / Quality / Semantic only | GenEval (T2I); throughput (FPS); diversity judged qualitatively only | "retested with Diffusers and our augmented prompts"; prompt count and seeds not stated | qualitative remarks only (1-step blurry; distilled not "strictly superior… in diversity and physical consistency") |
| FlashVSR (LR_FLASH, chain), arXiv 2510.12747 | VSR on real video; built on Wan2.1-1.3B + LoRA | as backbone, not as an accelerator | **no VBench** | PSNR/SSIM/LPIPS vs ground-truth HR (YouHQ40, REDS, SPMCS); NIQE, MUSIQ, CLIPIQA, DOVER (no-reference, also on VideoLQ and AIGC30); user study (20 raters, 32 sets); runtime and memory | real-video benchmarks; AIGC30 with no-reference metrics only | none |
| RIFE (F41_RIFE, chain), arXiv 2011.06294 | frame interpolation on real video | no | **no VBench** | PSNR, SSIM, interpolation error vs ground-truth frames (Vimeo90K, UCF101, HD, X4K, MiddleBury); runtime | real video only; never AI-generated | none |
| Step reduction (STEP25/20) | baseline, no paper | — | — | — | — | — |

### Wan2.1-1.3B numbers the papers report

| Source | Setting | Speed | VBench | Fidelity vs original |
| --- | --- | --- | --- | --- |
| rCM Table 2 | teacher 50×2 NFE | 0.72 FPS | Total 83.02 / Quality 83.95 / Semantic 79.26 | — |
| rCM Table 2 | rCM 4 steps | 14.6 FPS | 84.43 / 85.38 / 80.63 | — |
| rCM Table 2 | rCM 2 steps | 23.0 FPS | 84.09 / 84.90 / 80.86 | — |
| rCM Table 2 | rCM 1 step | 32.3 FPS | 82.65 / 83.60 / 78.82 | — |
| ScalingCache README | original, T=50 | 1× | 83.31 | — |
| ScalingCache README | 40% steps | 2.5× | 80.30 | PSNR 14.50 / SSIM 0.523 / LPIPS 0.437 |
| ScalingCache README | TeaCache 0.08 | 2.0× | 81.04 | 22.57 / 0.806 / 0.128 |
| ScalingCache README | ScalingCache | 2.5× | 82.92 | 26.61 / 0.890 / 0.071 |
| VSA paper [unverified] | original Wan2.1-1.3B | — | Total 82.56 | — |

## 2. What this means for the paper

1. **The protocol shifts the score more than the claimed losses.** The same base model (Wan2.1-1.3B, 50 steps) is
   reported at VBench 83.02 (rCM), 83.31 (ScalingCache) and 82.56 (VSA, unverified). That spread of 0.75 points
   comes from prompt sets, prompt augmentation, resolution and seeds. It is as large as or larger than the losses the
   papers claim (ScalingCache −0.39; DVG on Wan2.2 −0.4%). Each paper's comparison is internally paired, but no two
   papers' numbers can be compared.
2. **Only aggregate scores are reported.** Every paper gives a mean: Total, Quality/Semantic, or one number. None
   reports per-dimension VBench on Wan, per-prompt or per-category results, error bars, or more than one seed. None
   analyses failure cases. This is the gap our evaluation fills, not a claim that VBench's dimensions are wrong.
3. **VBench can rank a distilled model above its teacher.** rCM 4-step 1.3B scores 84.43 against the teacher's 83.02.
   On our dev2 prompts, RCM4 keeps only 25% of FULL-satisfied static requirements on high-detail prompts and changes
   VBench5 by only −0.21. The rCM authors note qualitatively that the student is not superior in diversity and
   physical consistency, but no metric they report shows it.
4. **Pixel fidelity cannot compare across method families.**
   - Cache and sparse-attention papers use PSNR/SSIM/LPIPS against the original output.
   - That is meaningful only when the accelerated sampler follows almost the same trajectory. Step reduction scores
     PSNR 14.5 in ScalingCache's own table while its VBench drop is moderate, and we measure retention 0.85 for STEP20.
   - Distillation, DVG and two-stage pipelines produce a different but possibly equally valid video, so these papers
     do not report pixel fidelity for T2V.
   - Our retention and quality_kept are paired with FULL at the requirement and quality level, not the pixel level.
     That is what lets one evaluation cover every family.
5. **Each family uses its own yardstick.**
   - cache: VBench + pixel fidelity;
   - distillation: VBench (+ GenEval);
   - VSR: restoration metrics on real video;
   - frame interpolation: PSNR on real frames.
   - Neither FlashVSR nor RIFE has been evaluated as a stage inside a T2V acceleration pipeline. FlashVSR has only
     no-reference metrics on AIGC video.

## 3. Attention-level candidates (to add)

Target setting: Wan2.1-T2V-1.3B, 832×480×81, 50 steps.

**Integration point:** LightX2V, which is already on the server, ships attention backends for most of these:
`sage_attn2/3`, `svg_attn`, `svg2_attn`, `radial_attn`, `sparge_attn` / `spas_sage_attn2`, `sla_attn`, `nbhd_attn`,
`draft_attn`. It also has T2V configs `configs/attentions/wan_t2v_{sla_480p,sparge,draft}.json`; the other configs
are I2V and would need a T2V copy.

**FULL pairing:** if the attention arms run on the LightX2V path, they must be paired with a FULL50 generated on that
same path. See the platform v1 finding: the official-Wan and DVG-path FULL clips differ like two seeds.

| Method | Kind | Training-free | Official Wan2.1-1.3B 480p | Reported speed on Wan | Authors' metrics |
| --- | --- | --- | --- | --- | --- |
| SageAttention2 (arXiv 2411.10958) | quantized attention (INT4 QK, FP8 PV), drop-in kernel | yes | generic kernel; no Wan results in paper | not reported for Wan (CogVideoX-1.5 about 1.8× end-to-end) | CLIPSIM, CLIP-T, VQA-a/t, FScore [unverified] |
| SVG2 (arXiv 2505.18875) | semantic k-means sparse attention, 30% dense warm-up | yes | no (Wan 14B, 720p scripts only) | 1.89× end-to-end, Wan 14B 720p | PSNR/SSIM/LPIPS vs dense; VBench 5 quality dimensions + average; Penguin benchmark prompts |
| SVG1 (arXiv 2502.01776) | spatial/temporal head classification | yes | no (Wan only in appendix visuals) | 1.51× [unverified] | PSNR/SSIM/LPIPS; VBench 2 dimensions |
| Radial Attention (arXiv 2506.19852) | static O(n log n) energy-decay mask | yes at default length | no (14B only) | about 1.8×, Wan 14B 768p | PSNR/SSIM/LPIPS, Vision Reward |
| SpargeAttn (`spas_sage_attn`) | sparse + Sage quantization | yes | via LightX2V (`wan_t2v_sparge.json`) | 1.78× on HunyuanVideo (DVG's baseline table) | — |
| Jenga (arXiv 2505.16864) | block sparse attention + progressive resolution | yes | **yes** (`wan_1.3B_jenga_base.sh`, 832×480×81) | DiT-only 4.26× base / 6.16× turbo | VBench (abstract) |
| STA (arXiv 2502.04507) | sliding-tile window | yes (mask search) | no (Wan only in appendix, at HunyuanVideo resolution) | 1.60× [unverified] | VBench Total, PSNR/SSIM, CD-FVD, human study (200 MovieGen prompts) |
| VSA (arXiv 2505.13389) | trainable hierarchical sparse attention | **no** (fine-tuning) | paper retrofits 1.3B at 480p; checkpoint release unverified | 31 s → 18 s on 1.3B | VBench Total, human study |
| SLA (arXiv 2509.24006) | sparse + linear attention | **no** (fine-tunable; fine-tuning needed for quality) | the paper reports on Wan2.1-1.3B; LightX2V has `wan_t2v_sla_480p.json` | 13.7× attention kernel, 2.2× end-to-end | not checked |

**Recommendation:** add three or four training-free arms that span the attention design space, all through LightX2V
on one code path:

- **SageAttention2.** Quantization: the near-lossless reference point, at a different speed tier.
- **SVG2.** Dynamic, semantic sparsity. This is the strongest published training-free method. We would port it to
  1.3B 480p, so the setting must be called "our port", not the authors'.
- **Radial Attention.** A static mask. It gives the content-independent sparse baseline: does a fixed spatio-temporal
  decay fail on high-motion prompts?
- **Optional: Jenga.** It is the only one with official 1.3B 480p support. It mixes sparse attention with progressive
  resolution, so it lands between the attention and spatial-bottleneck families.

VSA and SLA need fine-tuning and are out of scope for a training-free comparison. SLA could only enter through a
released fine-tuned checkpoint.

**Prediction to pre-register:** sparse attention drops long-range spatio-temporal interactions. It should therefore
lose most on prompts with large motion or camera movement (a motion × arm interaction, opposite in sign to rCM).
Quantized attention should show no content interaction.

## Sources

- DVG: https://arxiv.org/html/2605.21042
- TeaCache: https://arxiv.org/pdf/2411.19108 , https://liewfeng.github.io/TeaCache , eval README in github.com/ali-vilab/TeaCache
- ScalingCache: https://github.com/KlingAIResearch/ScalingCache , ICLR 2026 poster https://www.iclr.cc/virtual/2026/poster/10006887
- rCM: https://arxiv.org/html/2510.08431v2 , https://github.com/NVlabs/rcm
- FlashVSR: https://arxiv.org/html/2510.12747v1
- RIFE: https://arxiv.org/abs/2011.06294
- SVG1/SVG2: https://arxiv.org/abs/2502.01776 , https://arxiv.org/html/2505.18875v3 , https://github.com/svg-project/Sparse-VideoGen
- SageAttention: https://arxiv.org/abs/2411.10958 , https://github.com/thu-ml/SageAttention
- Radial Attention: https://arxiv.org/abs/2506.19852 , https://github.com/mit-han-lab/radial-attention
- STA / VSA: https://arxiv.org/abs/2502.04507 , https://arxiv.org/abs/2505.13389 , https://github.com/hao-ai-lab/FastVideo
- Jenga: https://arxiv.org/abs/2505.16864 , https://github.com/JIA-Lab-research/Jenga
- SLA: https://arxiv.org/html/2509.24006v2

## 4. Mainstream attention acceleration: a taxonomy (2026-11-04)

Written from the survey above and general knowledge of the literature. Speedups are the authors' own figures where §3
gives them. End-to-end gain is capped by attention's share of DiT time (Amdahl). At 832×480×81 the sequence is about
32K tokens. Measure the dense FA2 attention share on our H100 before predicting any end-to-end number.

| Family | How it saves time | Representative methods | Training | Lossy? | Content-dependent? | In LightX2V |
| --- | --- | --- | --- | --- | --- | --- |
| Exact IO-aware kernels | tiling / fusion, same math | FlashAttention 2/3/4, FlexAttention | none | no | no | flash_attn2/3/4 (our FULL baseline) |
| Quantized attention | low-bit QK / PV matmuls | SageAttention 1/2/2++/3 (INT8/INT4/FP8/FP4; SA3 needs Blackwell) | none | slightly | no (in principle) | sage_attn2/3 |
| Static sparse masks | fixed local / decaying pattern | sliding-tile STA, neighborhood attention, Radial Attention | none (STA mask search; Radial LoRA for long videos) | yes | **no**: the same mask for every prompt | radial_attn, nbhd_attn |
| Dynamic (online) sparse | predicts important blocks per input, then block-sparse kernel | SpargeAttn, SVG1 (spatial/temporal heads), SVG2 (k-means semantic permutation, top-p), draft attention (low-resolution attention map → mask), XAttention-style antidiagonal scoring | none | yes | **yes**: the mask follows the content | sparge_attn / spas_sage_attn2/3, svg_attn, svg2_attn, draft_attn |
| Trainable sparse / sparse+linear | learned routing or a linear branch for the dropped blocks | VSA, SLA (TurboDiffusion), MoBA-style | fine-tuning required | small after training | yes | sla_attn (needs the fine-tuned weights) |
| Attention reuse across steps | skip or broadcast attention outputs between timesteps / CFG branches | PAB (pyramid attention broadcast), DiTFastAttn (window + step/CFG sharing) | none | yes | partly | — (overlaps with feature caching: TeaCache / ScalingCache) |
| Token reduction | shorter sequence (merge / prune tokens) | token merging / pruning for video DiTs | mostly none | yes | yes | — |
| Multi-GPU sequence parallel | split the sequence over GPUs, exact | Ulysses, Ring attention, xDiT | none | no | no | ulysses, ring (out of scope: changes the hardware budget) |

**What the benchmark gains from each family:**

- **Quantized (SageAttention2).** The near-lossless reference point. It should show no prompt × arm interaction; if it
  does, that is a finding in itself.
- **Static sparse (Radial / sliding-tile).** The cleanest test of the content hypothesis. A fixed local mask should
  lose most where long-range spatio-temporal dependencies matter: large motion and camera movement.
- **Dynamic sparse (SVG2, SpargeAttn).** The adaptive counterpart. Does adapting the mask to content remove the
  failures of the static mask, or move them to other content?
- **Out of scope for a training-free, single-GPU comparison:** trainable sparse (VSA / SLA), sequence-parallel
  methods, and attention reuse (redundant with the caching arms already in the benchmark).

**Suggested arms:**

| Arm | Family | Settings |
| --- | --- | --- |
| SAGE2 | quantized | SageAttention2 8-bit |
| RADIAL | static sparse | Radial Attention, default settings |
| SVG2 | dynamic sparse | SVG2, authors' default top-p, 30% dense warm-up |
| SPARGE (optional) | dynamic sparse | SpargeAttn, if the budget allows |

All arms run on the LightX2V path, with a FULL50 from the same path.
