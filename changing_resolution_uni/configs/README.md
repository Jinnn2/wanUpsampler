# Universal clean-latent upsampler

`train_universal_clean.yaml` is the first independent U-ITU stage. It trains
one checkpoint on all LR variants stored in a `wan_uni_clean_v1` LMDB. The
model changes only spatial latent resolution and keeps the Wan temporal latent
length unchanged.
