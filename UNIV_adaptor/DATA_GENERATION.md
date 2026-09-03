# UNIV Sparse Data Generation Protocol

This document defines the data required to train the prompt-and-latent compute
utility controller described in `README.md`. It is deliberately separate from
`VALIDATION.md`: the validation suite tests fixed actions, while this protocol
collects counterfactual trajectories from one content-independent common probe.

## 1. Scientific contract

The controller predicts action quality, not a pre-combined utility:

```text
Q_hat(prompt feature, common-probe observation, candidate action)
```

The deterministic selector applies measured cost afterward:

```text
argmax_a Q_hat(state, a) - lambda * C(a) / C(full)
subject to C(a) <= remaining budget
```

Consequently:

- quality labels are stored without a fixed `lambda`;
- every record keeps raw quality dimensions and measured cost;
- budget feasibility comes from a hardware-specific measured CostProfile;
- the cheap proxy density is used only to design and balance collection;
- prompt, probe observation, and action are separate fields;
- train/validation/test are split by prompt before generation.

## 2. Why the current fixed-action runner is insufficient

`WanUniversalRGBPipelineRunner` chooses spatial, temporal, NFE, and switch at
step zero. A controller cannot inspect a latent and then retroactively choose
the grid on which that latent was generated. Controller data therefore needs a
new branchable execution path:

```text
coordinate-aligned native noise
  -> fixed common probe
  -> save probe state and extract DVG/learned observation
  -> select candidate slots
  -> clone the same probe state for every candidate
  -> reshape probe state to the candidate working grid at the same sigma
  -> candidate LR/cache suffix
  -> selected transition to native HR
  -> HR suffix
  -> decode and score
```

This introduces two explicit reshapes:

```text
common probe grid -> candidate LR grid -> native HR grid
```

Fixed-action validation videos remain useful baselines, but they are not valid
controller training rows because their pre-decision histories differ.

## 3. Collection phases

### E2-A: probe feasibility

Run a small technical set over the configured probe candidates. Check that:

- every probe ends before reference step 30;
- its last position is a fresh full DiT evaluation;
- clean extrapolation and DVG demand features are finite;
- at least one legal future action remains at every budget;
- probe cost leaves a useful acceleration margin;
- repeated seeds have stable feature scales.

The checked-in pilot protocol proposes, but does not select, probes with
`2/3/4/6` full DiT evaluations at reference step 10 and an additional
four-evaluation probe at step 15.

### E2-B: probe information value

For each surviving probe, train the same small action-conditioned predictor on
the same prompt split and sparse action observations. Select one probe using
validation policy regret minus probe cost. Do not select a probe by latent
reconstruction error alone.

Write the selected immutable probe object into `common_probe.selected`. Only
then may the controller collection planner run.

### Sparse controller training

Each train trajectory contains:

```text
one Native-HR teacher
one runtime DVG action
one deterministic space-filling exploration action
```

Later active-learning rounds may append one uncertainty or hard-negative action
to an existing trajectory. They must never overwrite the original candidates.

### Dense sampled-Oracle audit

Validation and test use a deterministic spread-out subset of the budget-feasible
action pool. This is a sampled Oracle, not an exhaustive global Oracle. Test is
generated and scored only after all probe, model, loss, and threshold choices
are frozen on validation.

## 4. Action and budget representation

The initial legal grid is:

```text
spatial:  0.50, 0.625, 0.75, 0.875, 1.00
temporal: 0.50, 0.67,  0.80, 1.00
LR NFE:   0.40, 0.55,  0.70, 0.85, 1.00
switch:   0.60, 0.80,  1.00
```

Requested ratios and resolved Wan shapes/step masks are both stored. The
planning-only density is:

```text
rho_proxy(a) = switch * spatial^2 * temporal * LR_NFE + (1 - switch)
```

It excludes cache updates, transition, VAE, SR, decode, and controller costs.
Formal feasibility must use synchronized measurements from the deployment GPU,
precision, kernels, model, and transition implementation.

Use one transition per formal protocol. Compare `dvg_latent_anchor` and
`rgb_sr_vae` under E3, then lock the selected baseline. Pooling both transitions
would double data and make transition identity an undeclared action axis.

## 5. Runtime DVG slot

The DVG candidate is deferred until the common probe exists. Runtime extraction
computes at least:

```text
spatial demand: normalized high-frequency latent energy
temporal demand: latent motion / temporal-difference energy
NFE demand: velocity curvature and cache residual drift
switch demand: clean-estimate stabilization across probe checkpoints
```

The original DVG spatial-temporal matching rule selects the first candidate.
UNIV-specific NFE and switch extensions must be named `univ_dvg_heuristic`, not
paper DVG. The record stores raw features, normalization version, selected
action, feasible set hash, and tie-breaking rule.

## 6. Required trajectory record

Every completed `univ_sparse_trajectory_record_v1` contains:

```text
identity
  plan hash, trajectory key, split, prompt id/text/hash, seed, budget id

common probe
  probe id/config, x_sigma, velocity, predicted-clean state, feature vector
  boundary step/sigma/logSNR, elapsed time, artifact hashes

native teacher
  final video, video hash, warm latency, quality vector
  optional HR states at reference boundaries 30/40/50

candidates[]
  selection source and propensity
  requested action and resolved schedule
  transition id and implementation hash
  switch clean state, transition clean HR, re-noised HR
  final video and hash
  quality vector and transition diagnostics
  warm full-pipeline and per-stage timings

provenance
  Git commit, dirty state, model/config/checkpoint hashes
  CUDA/GPU/PyTorch/LightX2V/VBench versions
```

Large tensors use fp16 or bf16 storage, while diagnostics and quality values use
float32/float64. Tensor files are content-addressed; JSON records store paths,
shape, dtype, and SHA256 rather than embedding tensors.

## 7. Quality labels

The primary model predicts a vector:

```text
subject consistency
background consistency
motion smoothness
aesthetic quality
imaging quality
paired native fidelity
```

`dynamic_degree` remains diagnostic and is not averaged into VBench-5. Paired
native fidelity must use the same prompt, seed, coordinate noise, target shape,
and frame count. Keep individual metrics so later experiments can change their
weights without regenerating videos.

Use three fidelity levels:

```text
L0: transition/native HR state distance and spectral/temporal diagnostics
L1: fixed decoded keyframes with paired perceptual/semantic metrics
L2: complete MP4, VBench-5, and synchronized warm latency
```

L0 may reject numerically broken candidates. It must not silently replace L2
for candidates used as formal quality labels.

## 8. Sparse sampling and propensity

The checked-in pilot uses two candidates for each train trajectory:

1. runtime DVG action;
2. deterministic space-filling exploration within the assigned budget tier.

Across prompts, budget tiers are assigned deterministically and approximately
balanced. Future randomized collection should use a logged mixture such as:

```text
0.40 DVG-neighbor perturbation
0.30 uniform/LHS feasible exploration
0.20 one-axis counterfactual
0.10 uncertainty or hard-negative action
```

The exact selection probability must be logged. Without action overlap and
propensity, off-policy comparison of collection policies is not defensible.

## 9. Split and seed policy

The pilot protocol contains 500 prompt-disjoint sources:

```text
train:      300 prompts, base seed 42, sparse two-action collection
validation: 100 prompts, base seeds 42/100/2024, 8-action Oracle subset
test:       100 prompts, base seeds 42/100/2024, 8-action Oracle subset
```

This resolves to 900 prompt-seed trajectories: 600 sparse-train candidate runs,
4,800 validation/test candidate runs, and 900 shared Native-HR teachers, for a
maximum of 6,300 complete videos before L0/L1 screening. Probe selection uses
30 prompts, three seeds, five probe candidates, and two matched downstream
actions: at most 900 candidate videos plus 90 shared teachers.

The actual sampling seed is `base_seed + prompt_id`. Prompt order, text hashes,
split assignment, budget assignment, action slots, and the complete protocol
are frozen in an immutable collection plan before generation.

## 10. Output layout and resume rules

```text
<OUT_ROOT>/
  collection_plan.json
  cost_profile.json
  prompts/
  probe/
    states/
    features/
  native/
    states/
    videos/
  candidates/
    <action_key>/states/
    <action_key>/videos/
  records/
    <trajectory_key>.json
  metrics/
  logs/
  coverage.json
```

Generation is append-only. A candidate is complete only when its video, sidecar,
metrics, and declared tensor artifacts exist, are non-empty, and match hashes.
Resume skips only complete candidates. A changed prompt, probe, action space,
transition, model, or cost profile requires a new output root.

## 11. Planning commands

The checked-in protocol intentionally has `common_probe.selected=null`:

```bash
python UNIV_adaptor/scripts/data/plan_univ_sparse_dataset.py \
  check-protocol \
  --protocol UNIV_adaptor/configs/univ_sparse_controller_pilot.json \
  --allow-pending-probe
```

Create the immutable E2 plan before selecting a probe:

```bash
PROMPTS_FILE=/path/to/500_prompts.txt \
PLAN_PATH=/path/to/output/probe_selection_plan.json \
bash UNIV_adaptor/scripts/run_univ_data_plan.sh plan-probes
```

After E2 selects a probe, copy its exact object into `common_probe.selected`
and freeze a new protocol file. Then create the immutable plan:

```bash
python UNIV_adaptor/scripts/data/plan_univ_sparse_dataset.py plan \
  --protocol /path/to/selected_protocol.json \
  --prompts /path/to/500_prompts.txt \
  --output /path/to/output/collection_plan.json

python UNIV_adaptor/scripts/data/plan_univ_sparse_dataset.py check \
  --plan /path/to/output/collection_plan.json
```

There is intentionally no `generate` command yet. It will be enabled only after
the common-probe branch runner, runtime DVG selector, measured CostProfile gate,
and append-only trajectory writer satisfy this contract.
