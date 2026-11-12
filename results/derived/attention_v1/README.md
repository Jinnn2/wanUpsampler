# Attention v1: verdict

- **Pre-registration:** `UNIV_adaptor/ATTENTION_V1.md`.
- **Data:** `results/attention/attention_v1_results`. The data are SVG2 and SPARGE50 on the v1 and dev2 prompts,
  3 seeds each. They were generated on H100 through the platform's official-Wan path, scored against the platform
  FULL50, which is reused bitwise (the gate passed on both sets).
- **Script and output:** `analyze.py` writes `analyze.txt`.
- **Excluded prompts:** dev2 prompts 103, 139 and 151 have no checklist, so they have no retention score. They are
  dropped for every arm, as the router drops them. That leaves 125 prompts.

## Pre-registered hypotheses

| | hypothesis | result | verdict |
| --- | --- | --- | --- |
| A1 | SPARGE50: low-motion score minus high-motion score > 0 | +0.416 [+0.346, +0.478] | **holds** |
| A2 | SVG2: motion and detail deltas inside ±0.05 | motion +0.446 [+0.382, +0.509]; detail +0.027 [−0.040, +0.091] | **does not hold** (strong motion effect) |
| A3 | SPARGE50 fails judder/flicker gates more often than SVG2 | fail rates 2.3% vs 2.1%; discordant pairs 6 vs 5, p = 1.0 | **does not hold** |

**A1 holds, but A1 as written cannot tell a SPARGE-specific effect from the shared one.**
- High-motion prompts are harder for every arm. The mean motion delta over the 15 old arms is +0.257.
- Post hoc, not pre-registered: subtracting that shared delta still leaves an interaction for both sparse-attention
  arms (family bootstrap).
  - SPARGE50: +0.158 [+0.106, +0.209].
  - SVG2: +0.189 [+0.123, +0.256].
- So sparse attention does lose more on motion than the other accelerators. But SVG2 does too, which is the
  opposite of A2.

**Where the loss is.**
- Almost all of it is in quality, not in semantics. On high-motion prompts:
  - SVG2: retention 0.88, quality_kept 0.45.
  - SPARGE50: retention 0.78, quality_kept 0.48.
- SVG2 on high motion is judged less sharp than FULL in 83% of pairs (kept_sharp 0.17); on low motion this is 0.86.
- The v2 temporal gates rarely fire for either arm (about 2%). The loss is blur and artifacts, not judder or flicker.
  That is why A3 has no power.

## Descriptive

**Both attention arms are dominated on Wan2.1-1.3B at 480p.**

| arm | speed | min(ret, Q) |
| --- | --- | --- |
| SPARGE50 | 1.47x | 0.673 |
| SVG2 | 1.06x | 0.666 |
| SCALING10 | 1.76x | 0.876 |
| TEA008 | 1.87x | 0.812 |

- SCALING10 and TEA008 are faster and better than both attention arms.
- The router never selects SVG2 or SPARGE50, at any τ, in any cell. The cell routing results with them in the
  pool are in `route_attn/route_report.md`.

**VBench misses the loss again.**
- dev2 Δvbench5, in percentage points against FULL:
  - SVG2: −0.70.
  - SPARGE50: −0.59.
  - RCM4: −0.50, although its min(ret, Q) is only 0.34 (from `fig1_v1/scoreboard_dev2.csv`).
- VBench therefore ranks SVG2, layered score 0.67, below RCM4, layered score 0.34. The layered score ranks them the
  other way. All five VBench quality dimensions move by about 0.01 for both attention arms.

**Speed** (H100, vs FULL50 at 102 s):
- SPARGE50 1.47x (50% of blocks kept).
- SVG2 1.05x. Of the 40 sparse steps, 21% of the attention is kept. k-means and the per-call flashinfer planning
  cost almost the whole saving at 32,760 tokens.

## Notes

- **SpargeAttn hangs on H100.** The hangs match thu-ml/SpargeAttn#127.
  - With the default `pvthreshd`, 14 SPARGE50 clips in dev2 hung once or more (`hangs_pvthreshd50.json` on the
    server).
  - With `pvthreshd=1e30` there were no hangs, and the output is bitwise identical (4/4 clips).
- **One SVG2 worker crashed** with a CUDA illegal memory access in dev2. Its job was rerun successfully; all 768
  clips exist.

## Follow-up: quality-layer null controls (`results/attention/quality_null`)

The decision rules were pre-registered in ATTENTION_V1.md before the run.

**Ceiling.** Identity pairs are kept on 0.97–1.00 of aspects. The judge works on identical clips.

**N1 is triggered.** Two FULL clips of the same prompt, seeds 42 and 100, are equally good by construction, but
their `quality_kept` is only 0.711.

| | quality_kept | `worse` share |
| --- | --- | --- |
| seed null, low motion | 0.916 | 0.07 |
| seed null, high motion | 0.506 | 0.32 |

So the layer charges for "different", and much more on high-motion clips. Under N1, `not_worse` becomes the
primary quality score, and all arms must be rescored with the reverse question.

**Arm minus seed null, `worse` share** (family bootstrap, 95% CI):

| arm | high motion | low motion | reading |
| --- | --- | --- | --- |
| SVG2 | +0.102 [+0.020, +0.183] | +0.028 [−0.010, +0.068] | **N2 holds, barely**: a real excess loss on high motion of about 10 pp beyond being different; SVG2 is closer to FULL (PSNR 23) than a reseed (10) |
| SPARGE50 | +0.000 [−0.090, +0.089] | +0.011 [−0.025, +0.045] | indistinguishable from a reseed: its "loss" is that it makes a different video |
| TEA008 | −0.162 [−0.243, −0.082] | −0.039 [−0.080, +0.001] | better than a reseed (closer to FULL) |
| SCALING10 | −0.258 [−0.333, −0.184] | −0.047 [−0.089, −0.008] | better than a reseed (closer to FULL) |

**What this means.** The judge alone already produces most of the shared high-vs-low motion gap in `quality_kept`.
The seed null's own gap is 0.41.
- So the cell × arm pattern of every arm partly reflects how far the arm's clip drifts from FULL on high-motion
  prompts, where the judge cannot confirm "at least as good" for two different clips.
- How much of the dev/dev2 routing result survives with `not_worse` has to be checked by rescoring the old arms.
  The retention layer has the same one-directional pairwise design and needs the same null.
