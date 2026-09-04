# UNIV Prompt-Budget Data Generation v2

This protocol trains one prompt-conditioned budget quality curve. It does not
train a four-axis action controller. Each budget id is frozen to one concrete
space/time/NFE/switch action before video collection.

## Scientific contract

The learned model predicts quality for five fixed budget presets:

```text
Q_hat(prompt feature, budget id)
```

The runtime selector applies the measured cost and user preference afterward:

```text
argmax_j Q_hat(prompt, B_j) - lambda * C(B_j) / C(native)
subject to C(B_j) <= user hard budget
```

Quality labels never contain a fixed lambda. Raw quality dimensions and actual
latency remain separate so the same model can be evaluated under different
preferences.

## Scope boundary

The executable v2 protocol is explicitly:

```text
observation_mode = prompt_only
trajectory_origin = independent_step0
```

The existing Wan runner fixes space, time, NFE, and switch before latent
initialization. Consequently, every budget trajectory starts independently
from step zero but uses the same prompt, seed, target shape, and coordinate
Gaussian field. This is valid for the prompt-budget model because no generated
latent is an input to that model.

These records must not be used to train or claim a Prompt+latent common-probe
controller. That extension requires a branchable runner that clones one exact
probe tensor before candidate execution and must use a new protocol schema.

## Fixed budget curve

The calibration action grid is:

```text
space:  0.50, 0.625, 0.75, 0.875, 1.00
time:   0.50, 0.67,  0.80, 1.00
NFE:    0.40, 0.55,  0.70, 0.85, 1.00
switch: 0.80, 0.90,  1.00
```

`configs/univ_prompt_budget_pilot.json` freezes five initial
DVG-inspired balanced presets. Their `target_cost_ratio` values are design
targets, not measured facts. Before a formal full run, use a small pilot to
measure warm end-to-end latency and revise the concrete actions if their costs
do not form the intended B30/B40/B50/B60/B70 curve.

Every prompt-seed trajectory contains exactly:

```text
Native-HR50 teacher
B30
B40
B50
B60
B70
```

The transition is fixed to `dvg_latent_anchor`. Transition comparison remains
a separate E3 experiment; transition identity is not a hidden action axis in
this dataset.

## Splits and scale

Prompts are split before generation and must be unique:

```text
train:      300 prompts x 1 seed x 6 videos = 1800
validation: 100 prompts x 3 seeds x 6 videos = 1800
test:       100 prompts x 3 seeds x 6 videos = 1800
total:                                          5400
```

The actual generator seed is `base_seed + prompt_id`. Test generation is not
part of the default launcher selection and should run only after preset,
transition, scoring, and model choices are frozen on validation.

## Planning and execution

Default remote paths:

```text
project:    /mnt/afs_2/houze/wanUpsampler
Wan:        /mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B
LightX2V:   /mnt/afs_2/houze/LightX2V
Python:     /opt/conda/bin/python
prompts:    <project>/prompts/univ_controller_pilot_500.txt
```

Preflight and inspect the immutable 8-GPU plan:

```bash
cd /mnt/afs_2/houze/wanUpsampler

bash UNIV_adaptor/scripts/run_univ_prompt_budget_data_8gpu.sh check

OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_pilot_v2 \
bash UNIV_adaptor/scripts/run_univ_prompt_budget_data_8gpu.sh plan
```

The full protocol resolves to 54 jobs with the default 100-prompt chunk. Jobs
are greedily assigned to eight worker slots using native cost or preset proxy
density as a load estimate. The immutable manifest fixes the prompt file,
plan, generated configs, job boundaries, and worker assignment.

Run a separate eight-video execution smoke with one prompt per job and one job
per worker:

```bash
SPLITS=train,validation \
JOB_CHUNK_SIZE=1 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_smoke_v2 \
bash UNIV_adaptor/scripts/run_univ_prompt_budget_data_8gpu.sh plan

SPLITS=train,validation \
JOB_CHUNK_SIZE=1 \
MAX_JOBS_PER_WORKER=1 \
ALLOW_PILOT_PRESETS=1 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_smoke_v2 \
bash UNIV_adaptor/scripts/run_univ_prompt_budget_data_8gpu.sh generate
```

Generate train and validation after inspecting smoke videos and measured cost:

```bash
# First update preset actions and set preset_status=frozen_after_measured_cost.
SPLITS=train,validation \
JOB_CHUNK_SIZE=100 \
RESUME=1 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_pilot_v2 \
bash UNIV_adaptor/scripts/run_univ_prompt_budget_data_8gpu.sh all
```

After every design choice is frozen, generate test into the same immutable
root:

```bash
SPLITS=test \
JOB_CHUNK_SIZE=100 \
RESUME=1 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_pilot_v2 \
bash UNIV_adaptor/scripts/run_univ_prompt_budget_data_8gpu.sh all
```

Changing `JOB_CHUNK_SIZE`, prompts, protocol, template, model path, or generated
case config under an existing output root is rejected. Use a new output root.
The launcher also refuses full generation while the protocol remains
`frozen_for_pilot_cost_calibration`; `ALLOW_PILOT_PRESETS=1` is reserved for a
bounded smoke run.

## Output layout

```text
<OUT_ROOT>/
  collection_plan.json
  generation_manifest.json
  configs/
    native_hr50.json
    B30.json ... B70.json
  videos/
    train|validation|test/
      native_hr50/
      B30/ ... B70/
  timings/
    <job_id>.jsonl
  logs/8gpu_data/
    gpu_<id>.log
  records/
    train|validation|test/
      <trajectory_key>.json
```

Generation records are finalized as `generated_unscored`. They contain video
and sidecar SHA256 hashes, concrete requested/resolved actions, synchronized
pipeline timing, segment timing, and peak GPU memory. VBench and paired native
fidelity are a separate scoring phase; a record is not train-ready until all
six videos have the declared quality vector.

## Resume and failure semantics

The launcher uses one output-root lock and one log per GPU. Completed jobs are
skipped only when their timing file has exactly one initialization row, every
expected prompt/seed is present, every MP4 is non-empty, and every budget video
has a UNIV runtime sidecar. A partial job is regenerated as a whole; completed
jobs on other GPUs remain reusable.

## Independent reserve shard and zero-copy merge

A second eight-GPU machine should generate a separate 500-prompt shard. Do not
duplicate the primary prompt file and do not write into the primary output
root. Select a deterministic reserve shard from the same source corpus while
excluding all primary prompts:

```bash
cd /mnt/afs_2/houze/wanUpsampler

python UNIV_adaptor/scripts/data/select_prompt_shard.py \
  --source /mnt/afs_2/houze/wanUpsampler/prompts/vidprom_filtered_extended.txt \
  --exclude /mnt/afs_2/houze/wanUpsampler/prompts/univ_controller_pilot_500.txt \
  --output /mnt/afs_2/houze/wanUpsampler/prompts/univ_controller_reserve_500.txt \
  --count 500 \
  --seed univ-prompt-budget-reserve-v1
```

The selector removes exact duplicates, selects by stable SHA256 rank rather
than source order, reshuffles deterministically for split assignment, writes a
metadata sidecar, and refuses to replace a different existing selection.

Use the exact same frozen protocol as the primary shard. The reserve machine
can use local GPU ids 0-7:

```bash
PROTOCOL=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_protocols/frozen_after_smoke_v1.json \
PROMPTS_FILE=/mnt/afs_2/houze/wanUpsampler/prompts/univ_controller_reserve_500.txt \
SPLITS=train,validation \
JOB_CHUNK_SIZE=25 \
MAX_JOBS_PER_WORKER=0 \
RESUME=1 \
GPU_IDS=0,1,2,3,4,5,6,7 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_reserve_v1 \
bash UNIV_adaptor/scripts/run_univ_prompt_budget_data_8gpu.sh all
```

Once both roots have finalized trajectory records, create a merged index. This
operation does not copy MP4 files:

```bash
python UNIV_adaptor/scripts/data/merge_prompt_budget_datasets.py \
  --shard primary=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_full_v3 \
  --shard reserve=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_reserve_v1 \
  --splits train validation \
  --output /mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_merged/merged_train_validation.json
```

The expected train/validation merge contains 800 unique prompts, 1200
prompt-seed trajectories, and 7200 videos. After both test splits are generated
and frozen, create a new all-split index:

```bash
python UNIV_adaptor/scripts/data/merge_prompt_budget_datasets.py \
  --shard primary=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_full_v3 \
  --shard reserve=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_reserve_v1 \
  --splits train validation test \
  --output /mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_merged/merged_all.json
```

The full merge contains 1000 unique prompts, 1800 trajectories, and 10800
videos. The merger requires identical frozen protocols, budget presets, model
path, generated case configs, and hashes of all code that participates in
generation. It rejects prompt overlap, missing records, plan-record mismatch,
missing artifacts, and incompatible shards. Add `--verify-artifact-hashes` for
a full MP4/sidecar rehash audit. Add `--require-scores` only after all six
videos in every record have their final quality vector.
