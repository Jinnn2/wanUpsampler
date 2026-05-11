# changing_resolution scripts

脚本按职责分组。远程长任务优先使用 tmux 入口。

更完整的训练方法说明见：

```text
changing_resolution/TRAINING_METHODS.md
```

## data

构建训练数据，包括 Wan2.1 生成 720p 原始视频、480p/720p latent pair
编码、LMDB 打包和多卡构建。

常用入口：

```bash
bash changing_resolution/scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh
```

## train

训练 clean latent 480p -> 720p resizer。

常用入口：

```bash
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage1_lmdb_training.sh
```

## eval

评估模型效果。

Operator compare 使用 LMDB validation 的 `ori720_decode` 作为参考，
比较 `interp720_decode` 和 `trained720_decode` 的 PSNR、SSIM、LPIPS。

Generation-chain A/B 在真实 LightX2V `changing_resolution` 链路中比较
`interp720` 和 `trained720`，不使用 native 720p 作为参考。

常用入口：

```bash
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_operator_compare_multigpu.sh
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_chain_ab_compare_multigpu.sh
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh
```

Operator compare 完成后生成 CSV 和 Markdown 表格：

```bash
python changing_resolution/scripts/eval/summarize_operator_compare_table.py \
  --input outputs/changing_resolution_operator_compare_stage1 \
  --split val
```

## bridge

LightX2V `changing_resolution` 推理桥接脚本。

## legacy

早期文件式训练、批量 compare、单视频应用模型等历史脚本。
保留用于回溯和兼容，不作为当前主线入口。
