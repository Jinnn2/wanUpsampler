#!/bin/bash

# set path firstly
lightx2v_path=/mnt/afs_2/houze/LightX2V
model_path=/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B
wanupsampler_path=/mnt/afs_2/houze/wanUpsampler

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export PYTHONPATH=${wanupsampler_path}:$PYTHONPATH

config_json=${wanupsampler_path}/configs/changing_resolution/wan_t2v_wanupsampler_v1.json
prompts_file=${wanupsampler_path}/configs/changing_resolution/wan_t2v_wanupsampler_v1_prompts_20.txt
output_root=${wanupsampler_path}/outputs/lightx2v_compare/batch20

negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

mkdir -p ${output_root}

index=1
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
  if [[ -z "$prompt" ]]; then
    continue
  fi

  sample_id=$(printf "%02d" ${index})
  seed=$((41 + index))
  save_result_path=${output_root}/${sample_id}_comparison.mp4

  python ${wanupsampler_path}/scripts/run_lightx2v_wanupsampler_compare.py \
  --seed ${seed} \
  --model_cls wan2.1 \
  --task t2v \
  --model_path $model_path \
  --config_json ${config_json} \
  --prompt "$prompt" \
  --negative_prompt "$negative_prompt" \
  --save_result_path ${save_result_path}

  index=$((index + 1))
done < ${prompts_file}
