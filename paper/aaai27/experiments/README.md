# AAAI-27 experiment suite

This directory provides a resumable layer over the existing training and
evaluation scripts. Run it on the Linux GPU host where `/mnt/afs_2/houze`
and the existing `outputs/` tree are available.

## 1. Audit existing remote results

```bash
cd /mnt/afs_2/houze/wanUpsampler
python paper/aaai27/experiments/run_experiments.py audit
```

Paths can be overridden without editing the manifest. For example:

```bash
WAN50_LORA40_CKPT=/other/run/step40.safetensors \
python paper/aaai27/experiments/run_experiments.py audit
```

The audit counts only non-empty expected artifacts. Existing videos in the
old four-way result folders are hard-linked into the canonical factorial
folder when prompt, seed, method, and handoff step match.

## 2. Run experiments incrementally

Start with paired Stage2 operator measurements:

```bash
python paper/aaai27/experiments/run_experiments.py run --group operator
```

Then generate the complete Base/LoRA x interpolation/Stage2 factorial:

```bash
python paper/aaai27/experiments/run_experiments.py run --group factorial
```

For a persistent remote run:

```bash
bash paper/aaai27/experiments/tmux_run_experiments.sh operator
bash paper/aaai27/experiments/tmux_run_experiments.sh factorial
```

Run one task or preview commands:

```bash
python paper/aaai27/experiments/run_experiments.py run --task wan50_factorial --dry-run
python paper/aaai27/experiments/run_experiments.py run --task wan50_factorial
```

Commands are sequential and fail closed. Re-running is safe: completed
evidence is reused, and individual video runners skip non-empty files. Logs
and task state are written to `outputs/aaai27_experiments/_state/`.

## 3. Blinded review and collection

```bash
python paper/aaai27/experiments/run_experiments.py run --task blind_review_package
python paper/aaai27/experiments/run_experiments.py collect
```

Give raters only `review/human_ratings.csv` and `review/blinded/`. Keep
`_private/human_review_key.csv` hidden. After rating, place the completed file
at `review/human_ratings_completed.csv`.

VBench remains an external task because the repository does not vendor an
official VBench environment. Put its per-video and aggregate JSON outputs in
each factorial folder's `metrics/` directory; `audit` will then mark that
evidence complete.

For paired metric columns, produce a bootstrap confidence interval and exact
sign-test result with:

```bash
python paper/aaai27/experiments/paired_statistics.py \
  --input path/to/metrics.jsonl \
  --a-field base_lpips --b-field talh_lpips \
  --output path/to/lpips_stats.json --lower-is-better
```
