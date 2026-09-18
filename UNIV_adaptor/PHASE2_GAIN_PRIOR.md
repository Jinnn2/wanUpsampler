# Phase 2 prompt quality-gain prior

Train a small regressor on existing Phase 2 scores. Target: the per-prompt mean VBench5 difference between each action and `P2_B30_CACHE`. Native HR50 is only the speedup baseline. No videos are generated or scored again.

Default actions: CACHE, SKIP, B30_SPATIAL (`ACTION_SET=three`). `pair` uses CACHE/SKIP; `five` retains all five as a control. Candidate selection was informed by the current validation results, so this is development evaluation, not a fresh test.

## Run remotely

```bash
cd /mnt/afs_2/houze/wanUpsampler
export QUALITY_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_phase2_20260918_chunk25/metrics/phase2_quality

# CPU lexical baseline; only Python + NumPy required
bash UNIV_adaptor/scripts/run_univ_phase2_gain_prior.sh all

# Frozen Wan T5 semantic features: GPU 0 for extraction, CPU for regression
FEATURES=t5 GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_phase2_gain_prior.sh all

# Pairwise follow-up, reusing verified T5 embeddings
FEATURES=t5 ACTION_SET=pair bash UNIV_adaptor/scripts/run_univ_phase2_gain_prior.sh all
```

Modes: `check` validates the downloaded input/score hash bindings and prompt/seed coverage; `embed` only extracts features; `train` fits and evaluates (T5 features must already exist); `all` checks, optionally embeds, and trains. `QUALITY_DIR` may point to the downloaded evaluation folder: original video files are not needed. This validates score artifacts, not the bytes of videos that are absent locally.

Environment overrides: `WAN_PYTHON`, `LIGHTX2V_REPO`, `MODEL_ROOT`, `T5_DIR`, `PRIOR_OUT`, `BUDGETS_SECONDS` (space-separated absolute seconds). Unlike generation, this task does not benefit from eight GPUs: T5 encodes only 160 prompts, then the ridge regression uses CPU. The T5 path requires the existing LightX2V Wan native encoder, tokenizer and checkpoint; it cannot silently fall back to a different encoder.

On Windows:

```powershell
python UNIV_adaptor/scripts/router/train_phase2_gain_prior.py train --quality-dir E:/Downloads/eval_0918
```

TF-IDF requires no scikit-learn or model download. T5 extraction reuses the repository's attention-mask mean-pooling implementation. Frozen features are L2-normalized; no validation-dependent PCA/scaling is applied. An extraction request binds exact prompt text, encoder checkpoint, tokenizer files and extractor source before resuming. A complete extraction is verified before reuse. Download the whole T5 directory if using it locally.

## Model selection and costs

- Average each prompt's seeds before forming gain targets; no prompt is split across CV folds.
- Five-fold train-only CV selects ridge alpha from 0.1, 1, 10, 100 plus a mean-only control by gain MSE. TF-IDF vocabulary/IDF and ridge centering fit on each fold's training prompts only. Alpha null means the mean-only model wins CV; it is not an error.
- Ridge predicts gains, not hard winner labels. Joint prediction of gains to a shared reference also defines all pairwise gain differences.
- Default eligibility/budget knots use train per-video P95 latency (`numpy.quantile(..., method='higher')`). Selection uses only prompt predictions and this calibration, never validation measured latency or quality. P95 is a calibration heuristic, not a hard latency guarantee.
- Evaluation retains all prompts and reports both prompt-mean and per-video actual latency violations. No action is changed after observing a validation failure or slow runtime. Quality is seed-averaged; validation has 20 prompt units.
- Train OOF results reuse the alpha-selection CV and full-train baseline/cost calibration, so they are selection diagnostics rather than nested unbiased estimates. The validation split is explicitly named `validation_development`.

## Baselines

1. `fixed_train`: highest train mean quality among budget-eligible actions.
2. `uniform_random_expected`: exact expectation of uniform action sampling, not one lucky RNG run.
3. `prompt_gain`: predicted highest quality gain among eligible actions.
4. `shuffled_router_hist_expected`: randomly reassign the router's action counts across prompts. This preserves action usage and matches calibrated average cost exactly. It isolates whether matching actions to prompts helps. Actual measured average latency can differ and is reported.
5. `train_frontier_mixture_expected`: prompt-independent action probabilities maximizing **train** mean quality at or below the router's calibrated average cost. Solves over pure actions and two-action mixtures. The target cost comes from the current batch's text-derived router choices, not its quality labels. This is a batch comparison control.
6. `quality_oracle_hindsight`: best observed quality in the train-calibrated action set. This intentionally uses quality labels and is only an upper bound; actual latency violations remain visible.

Main comparisons are quality/cost against both the fixed baseline and the two prompt-independent mixture controls. Compare actual measured time as well as calibrated cost; matching a budget upper bound alone does not ensure equal resource use. Reported generation times exclude feature extraction and router overhead. T5 extraction is reusable but still has deployment cost.

Bootstrap intervals resample prompts while holding the trained model, action histogram and mixture fixed. A 2000-permutation test shuffles the router's chosen actions over prompts, preserving the histogram. Neither test corrects for model/candidate development on this validation set, multiple budgets, or feature/action-set comparisons. Do not choose the best reported validation configuration and call it a locked confirmation result.

## Outputs

Under `QUALITY_DIR/gain_prior_FEATURES_ACTIONSET/`:

- `training_request.json`: input scores, settings and source hashes; changed requests require a new output folder.
- `cross_validation.csv`: train CV gain MSE including mean-only control.
- `model.json`, `model.npz`: vocabulary/IDF (TF-IDF), action order, calibration, centered ridge weights and intercept. For T5, reuse the verified 4096-dimensional pooled feature extractor. `phase2_gain_model.predict` and `select_actions` apply the model and calibrated budget mask.
- `gain_predictions.csv`: OOF/train and development predictions/observed differences with fold assignments.
- `policy_summary.csv/json`, `policy_by_prompt.csv`: quality dimensions, costs, violations, action fractions, paired gains and uncertainty.
- `report.md`: readable policy comparison.

TF-IDF is a lexical baseline. Failure does not rule out semantic conditioning; run the frozen T5 version before drawing that conclusion. `SKIP` remains cache reuse on a retained solver grid. This experiment does not claim an independent timestep-skipping mechanism.

The proposed 144-video multiseed supplement is not launched by these scripts. First assess whether the existing labels and frozen text features support gains; supplement training seeds only if label noise prevents a useful test. A later independent prompt set is needed for formal confirmation.
