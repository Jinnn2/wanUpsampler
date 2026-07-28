# Generated-data appendix

## Scope

The experiments do not train or evaluate on an external video benchmark
dataset. All videos and latent records used by the paper are generated or
constructed during the experiments with frozen Wan-family generators. Public
base-model weights are software/model dependencies, not experimental datasets.

## ITU clean-latent pairs

For each prompt and generation seed, the frozen generator first produces an
81-frame, 16-fps target-resolution video at 720x1248. The video is center
cropped/resized to the target grid, downsampled in RGB space to 368x640 (the
main setting; 480x832 is retained for the operator study), and both versions
are independently encoded with the same frozen Wan VAE. The resulting
`z0_lr`/`z0_hr` tensors, prompt, and construction metadata are stored in
sharded LMDB files.

- Wan50 construction target: 1,000 generated samples.
- Distill4 construction target: 5,000 generated samples.
- Clip length: 81 frames.
- Frame rate: 16 fps.
- Resize degradation: resize-only, with the selected kernel recorded per
  sample.
- Storage: float16 latent arrays in shards of 100 samples by default.

## TTD trajectory pairs

TTD records are constructed from the corresponding clean-pair LMDB. Each
record reuses the source prompt and generation seed (or a deterministic
base-seed-plus-index fallback recorded in metadata), the same initial noise,
scheduler, and guidance settings. It stores the cached pre-transition state,
the frozen teacher's completed low-resolution endpoint, the target clean
latent, prompt, seed, and the scheduler recipe needed by training.

- Wan50 target: 1,000 pairs for each trained transition (steps 40 and 45).
- Distill4 target: 5,000 step-3-to-step-4 pairs.
- Wan50 fallback base seed: 9400.
- Distill4 fallback base seed: 9500.

## Splits

Training scripts create deterministic index splits with Python
`random.Random(1234)`. ITU uses a 5% validation fraction capped at 100
examples. TTD uses a 2% validation fraction capped at 64 examples. The shuffled
validation prefix and training suffix are sorted before constructing
`Subset`s, so restarts and distributed launches use identical membership.

The final test prompts and seeds are disjoint from the Distill4 development
set. Wan50 test seeds are 9700--9709; Distill4 test seeds are 9800--9809. The
Distill4 development sweep uses eight different prompts with seed base 16000.

## Why ordinary public video datasets are not substitutes

The supervision requires paired internal states from the same frozen
generator: matched prompt, seed, initial noise, scheduler state, clean
low-/high-resolution VAE latents, and teacher trajectory endpoints. Ordinary
public video datasets contain rendered RGB videos but not these aligned
generator-internal quantities. Replacing the generated pairs with unrelated
public videos would therefore change the scientific question and would not
supervise the transition operation studied in the paper.

The large LMDBs are not included in the review ZIP. The complete builders are
under `code/`, and `tools/export_missing_repro_metadata.sh` exports exact shard,
sample, schema, shape, split, and environment records from the original
experiment machine without copying latent tensors.

## Realized original-machine inventory

The 2026-07-28 export verified 1,000 Wan50 ITU samples (10 shards), 1,000
Wan50 step-40 TTD samples (12 shards), 1,000 Wan50 step-45 TTD samples
(16 shards), and 5,000 Distill4 ITU samples (50 shards). Their realized
train/validation counts are 950/50, 980/20, 980/20, and 4,900/100,
respectively. The Wan50 collections contain 998 distinct prompt hashes; the
Distill4 ITU collection contains 5,000. The source-video inventories contain
1,002 Wan50 MP4 files (1,881,594,977 bytes) and 5,000 Distill4 MP4 files
(6,866,319,417 bytes); the realized Wan50 LMDB uses 1,000 records.

The first export followed a canonical Distill4 TTD3 directory that existed but
contained no LMDB shards. A complete legacy-path training-output inventory
exists through step 10,000, but that does not prove the latent-pair count.
Accordingly, `data/generated_dataset_realized_manifest.json` marks this one
collection as pending until the corrected candidate-path exporter is run; it
does not promote the empty directory to dataset evidence.
