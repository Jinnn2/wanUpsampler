#!/usr/bin/env bash

# Local machine paths for /data/yongyang/Jin.
# Scripts source this file by default. Override with:
#   PATH_CONFIG=/path/to/local_paths.sh bash scripts/run_lightx2v_training.sh build

JIN_ROOT="${JIN_ROOT:-/data/yongyang/Jin}"

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
