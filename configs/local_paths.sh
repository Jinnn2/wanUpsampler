#!/usr/bin/env bash

# Local machine paths for /mnt/afs_2/houze.
# Scripts source this file by default. Override with:
#   PATH_CONFIG=/path/to/local_paths.sh bash changing_resolution/scripts/train/run_clean_480p720p_stage2_lmdb_training.sh check

JIN_ROOT="${JIN_ROOT:-/mnt/afs_2/houze}"

PROJECT_ROOT="${PROJECT_ROOT:-${JIN_ROOT}/wanUpsampler}"
WAN_REPO="${WAN_REPO:-${JIN_ROOT}/Wan2.1}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-${JIN_ROOT}/LightX2V}"

MODEL_ROOT="${MODEL_ROOT:-${JIN_ROOT}/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"

# Stage 2 changing_resolution clean-latent paths.
CR_RAW_VIDEO_DIR="${CR_RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p}"
CR_RAW_VIDEO_DIR_1K="${CR_RAW_VIDEO_DIR_1K:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p_1k}"
CR_LMDB_DIR="${CR_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
CR_STAGE2_CONFIG="${CR_STAGE2_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage2.yaml}"
CR_STAGE2_OUT_DIR="${CR_STAGE2_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p_stage2_lmdb}"
CR_GENERATE_CONFIG="${CR_GENERATE_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p.json}"
CR_PROMPTS_FILE="${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}"
CR_HF_PROMPTS_FILE="${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}"
CR_STAGE2_OPERATOR_COMPARE_CKPT="${CR_STAGE2_OPERATOR_COMPARE_CKPT:-${CR_STAGE2_OUT_DIR}/latest.pt}"
CR_STAGE2_OPERATOR_COMPARE_DIR="${CR_STAGE2_OPERATOR_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_operator_compare_stage2}"
CR_STAGE2_CHAIN_COMPARE_CKPT="${CR_STAGE2_CHAIN_COMPARE_CKPT:-${CR_STAGE2_OUT_DIR}/latest.pt}"
CR_STAGE2_CHAIN_COMPARE_DIR="${CR_STAGE2_CHAIN_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_chain_ab_stage2}"
