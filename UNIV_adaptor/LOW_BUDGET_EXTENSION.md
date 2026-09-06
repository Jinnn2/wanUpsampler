# UNIV low-budget extension

This extension adds four true-LR25 MRFlow-style actions to an immutable UNIV
prompt-budget v2 shard. It does not regenerate or modify Native-HR50 or the
existing B30--B70 videos and records.

## Frozen actions

Artifact ids use an `LB` prefix because the new B30 action is not the v2 B30
action. `display_budget` retains the paper-facing budget name.

| Artifact id | Display | Space | Time | True LR | Re-noise | HR | Proxy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `LB10_LR25_S0200_HR02` | B10 | .50 | .50 | 25 | .20 | 2 | .1025 |
| `LB15_LR25_S0300_HR04` | B15 | .50 | .56 | 25 | .30 | 4 | .1500 |
| `LB20_LR25_S0300_HR04` | B20 | .55 | .80 | 25 | .30 | 4 | .2010 |
| `LB30_LR25_S0300_HR04` | B30 | .75 | .80 | 25 | .30 | 4 | .3050 |

The proxy is

```text
space^2 * time * true_lr_steps / 50 + hr_steps / 50
```

It excludes transition, codec, and fixed runtime overhead. Budget names remain
planning targets until a measured warm-latency profile is produced.

## Endpoint state contract

Every generated video has two sidecars:

```text
<video>.mp4.univ.json
<video>.mp4.endpoint.pt
```

The fp16 endpoint archive contains:

- exact prompt, prompt hash, and actual `base_seed + prompt_id` seed;
- the true LR25 sigma and model-timestep grid;
- an fp16 copy of the exact runtime sigma-zero LR endpoint `clean_lr`;
- the DVG-restored `clean_hr` and coordinate-aligned `hr_noise`;
- archive-tensor hashes and original runtime-tensor hashes;
- spatial/temporal action, transition identity, re-noise sigma, and HR steps.

The runtime sidecar binds the endpoint path, seed, tensor hashes, realized
LR/HR grids, and stage timings. Finalization hashes the video, runtime sidecar,
endpoint file, and immutable base record.

This is endpoint-conditioned data, not a common-probe counterfactual dataset.
The four actions start independently because their spatial and temporal
geometries differ. It supports a prompt action prior and an endpoint-conditioned
recovery model; it does not prove that one endpoint can retrospectively select
another action's initial geometry.

## Primary shard

Run train and validation only. The default paths target the existing primary
v2 shard and its 500-prompt file.

```bash
cd /mnt/afs_2/houze/wanUpsampler

bash UNIV_adaptor/scripts/run_univ_low_budget_extension_8gpu.sh check
bash UNIV_adaptor/scripts/run_univ_low_budget_extension_8gpu.sh plan
bash UNIV_adaptor/scripts/run_univ_low_budget_extension_8gpu.sh all
```

The default output is:

```text
outputs/univ_low_budget_extension_primary_v1
```

## Reserve shard

Run the reserve extension on the machine holding the reserve prompt file. It
must point to the matching immutable reserve base root.

```bash
cd /mnt/afs_2/houze/wanUpsampler

BASE_DATASET_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_reserve_v1 \
PROMPTS_FILE=/mnt/afs_2/houze/wanUpsampler/prompts/univ_controller_reserve_500.txt \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_low_budget_extension_reserve_v1 \
bash UNIV_adaptor/scripts/run_univ_low_budget_extension_8gpu.sh plan

BASE_DATASET_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_reserve_v1 \
PROMPTS_FILE=/mnt/afs_2/houze/wanUpsampler/prompts/univ_controller_reserve_500.txt \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_low_budget_extension_reserve_v1 \
bash UNIV_adaptor/scripts/run_univ_low_budget_extension_8gpu.sh all
```

Each shard adds 2,400 train/validation videos and 2,400 endpoint archives:

```text
train:       300 prompts * 1 seed  * 4 actions = 1200
validation:  100 prompts * 3 seeds * 4 actions = 1200
```

Across primary and reserve, the extension adds 4,800 videos. Test remains
excluded by the launcher's default `SPLITS=train,validation` and should be run
only after validation freezes the actions, model, and decision rule.

Inspect both extension roots with the existing multi-root progress tool:

```bash
python UNIV_adaptor/scripts/data/check_prompt_budget_progress.py \
  /mnt/afs_2/houze/wanUpsampler/outputs/univ_low_budget_extension_primary_v1 \
  /mnt/afs_2/houze/wanUpsampler/outputs/univ_low_budget_extension_reserve_v1 \
  --detail
```

## Resume and records

`RESUME=1` is the default. A video counts as complete only when the MP4,
runtime sidecar, endpoint archive, actual seed, and three endpoint tensor hashes
are present. Partial jobs rerun as a whole without touching the base dataset.

Finalization writes:

```text
records/<split>/<trajectory>.json
combined_records/<split>/<trajectory>.json
```

The extension record contains four low-budget candidates. The combined v3
record references the original Native-HR50, the five v2 candidates, and the
four new candidates, for nine unique `artifact_id` values. The old and new B30
remain distinct as `V2_B30` and `LB30_LR25_S0300_HR04`.
