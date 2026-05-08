#!/usr/bin/env bash

# Local machine paths for /mnt/afs_2/houze.
# Scripts source this file by default. Override with:
#   PATH_CONFIG=/path/to/local_paths.sh bash scripts/run_lightx2v_training.sh build

JIN_ROOT="${JIN_ROOT:-/mnt/afs_2/houze}"

PROJECT_ROOT="${PROJECT_ROOT:-${JIN_ROOT}/wanUpsampler}"
WAN_REPO="${WAN_REPO:-${JIN_ROOT}/Wan2.1}"
LIGHTX2V_REPO="${LIGHTX2V_REPO:-${JIN_ROOT}/LightX2V}"

MODEL_ROOT="${MODEL_ROOT:-${JIN_ROOT}/Wan-AI/Wan2.1-T2V-1.3B}"
VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/Wan2.1_VAE.pth}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/datasets}"
DAVIS_ZIP="${DAVIS_ZIP:-${DATASET_ROOT}/DAVIS-2017-trainval-480p.zip}"
# Leave DAVIS_DIR empty to auto-detect the extracted folder containing JPEGImages/480p.
DAVIS_DIR="${DAVIS_DIR:-}"

RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/raw_videos}"
LATENT_DIR="${LATENT_DIR:-${PROJECT_ROOT}/data/latent_pairs_wan21_512}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/wan_traj_upsampler_x2}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/configs/train_wan21_x2_512.yaml}"

# V2 changing_resolution clean-latent training paths.
CR_RAW_VIDEO_DIR="${CR_RAW_VIDEO_DIR:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p}"
CR_LATENT_DIR="${CR_LATENT_DIR:-${PROJECT_ROOT}/data/changing_resolution/latent_pairs_480p720p}"
CR_RAW_VIDEO_DIR_1K="${CR_RAW_VIDEO_DIR_1K:-${PROJECT_ROOT}/data/changing_resolution/raw_wan21_720p_1k}"
CR_LMDB_DIR="${CR_LMDB_DIR:-${PROJECT_ROOT}/data/changing_resolution/lmdb_480p720p_1k}"
CR_OUT_DIR="${CR_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p}"
CR_CONFIG="${CR_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p.yaml}"
CR_STAGE1_CONFIG="${CR_STAGE1_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/train_clean_480p_to_720p_lmdb_stage1.yaml}"
CR_STAGE1_OUT_DIR="${CR_STAGE1_OUT_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_clean_480p720p_stage1_lmdb}"
CR_GENERATE_CONFIG="${CR_GENERATE_CONFIG:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p.json}"
CR_PROMPTS_FILE="${CR_PROMPTS_FILE:-${PROJECT_ROOT}/changing_resolution/configs/wan_t2v_generate_720p_prompts.txt}"
CR_HF_PROMPTS_FILE="${CR_HF_PROMPTS_FILE:-${PROJECT_ROOT}/prompts/vidprom_filtered_extended.txt}"
CR_COMPARE_CKPT="${CR_COMPARE_CKPT:-${CR_OUT_DIR}/step_0100000.pt}"
CR_COMPARE_DIR="${CR_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_compare_step100000}"
CR_OPERATOR_COMPARE_CKPT="${CR_OPERATOR_COMPARE_CKPT:-${CR_STAGE1_OUT_DIR}/best_val.pt}"
CR_OPERATOR_COMPARE_DIR="${CR_OPERATOR_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_operator_compare_stage1}"
CR_CHAIN_COMPARE_CKPT="${CR_CHAIN_COMPARE_CKPT:-${CR_STAGE1_OUT_DIR}/best_val.pt}"
CR_CHAIN_COMPARE_DIR="${CR_CHAIN_COMPARE_DIR:-${PROJECT_ROOT}/outputs/changing_resolution_chain_ab_stage1}"
