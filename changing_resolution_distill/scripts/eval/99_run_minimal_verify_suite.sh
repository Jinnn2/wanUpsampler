#!/usr/bin/env bash
set -euo pipefail

# 一键最小验证套件：
# 00 baseline
# 01 interp random / resize_flow
# 02 step2 ckpt random / resize_flow, EMA off / on
#
# 注意：这个脚本会连续生成多个视频，比较耗时。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/00_run_baseline_distill_720p.sh"
bash "${SCRIPT_DIR}/01_run_interp_renoise_ab.sh"
bash "${SCRIPT_DIR}/02_run_step2_ckpt_renoise_ema_ab.sh"
