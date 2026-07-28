# 原实验机证据位置与导出方法

## 远程入口

原实验机通过名为 `temp` 的 Remote Tunnel 进入。导出脚本不处理 tunnel
连接，也不调用 SSH；在 `temp` 远程窗口的终端中直接执行即可。

## 已由本地配置和运行清单定位的目录

- 项目：`/mnt/afs_2/houze/wanUpsampler`
- Wan50 ITU LMDB：`/mnt/afs_2/houze/wanUpsampler/data/changing_resolution/lmdb_368x640_720x1248_1k`
- Wan50 TTD step-40 LMDB：`/mnt/afs_2/houze/wanUpsampler/data/changing_resolution/lmdb_tail_skip_lora_step40_to_step50`
- Wan50 TTD step-45 LMDB：`/mnt/afs_2/houze/wanUpsampler/data/changing_resolution/lmdb_tail_skip_lora_step45_to_step50`
- Distill4 原始 5k 视频：`/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_distill/raw_wan21_14b_cfgdistill_720p_5k`
- Distill4 ITU LMDB：`/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_distill/lmdb_clean_368x640_720x1248_14b_cfgdistill_5k`
- Distill4 TTD step-3 LMDB：`/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_distill/lmdb_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3`
- LightX2V：`/mnt/afs_2/houze/LightX2V`
- DiffSynth-Studio：`/mnt/afs_2/houze/DiffSynth-Studio`
- VBench：`/mnt/afs_2/houze/VBench`
- Wan2.1 1.3B：`/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B`
- Wan2.1 14B Distill：`/mnt/afs_2/houze/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`

## 推荐的轻量导出

已经增加固定路径专用脚本：

- `remote_export/export_temp_fixed_paths.sh`：在远端运行，所有原实验机路径
  已固定；支持 `metadata`、`full`、`all` 三种模式。

推荐先运行轻量 `metadata` 模式。它不复制 checkpoint、视频或 LMDB
张量，只导出硬件/软件版本、Git 状态、LMDB shard 元数据、总样本数、
shape、seed 范围、确定性 split index、原始视频文件清单以及小型训练
配置/日志：

```bash
cd /mnt/afs_2/houze/wanUpsampler
bash paper/aaai27/submission_materials_20260728/remote_export/export_temp_fixed_paths.sh metadata
```

结果写入固定目录：

```text
/mnt/afs_2/houze/wanUpsampler/outputs/aaai27_repro_exports/
```

脚本会输出三个可直接查找的末行字段：`EXPORT_DIR=...`、
`ARCHIVE=...` 和 `ARCHIVE_SHA256=...`。

完整五个 checkpoint 和源码/环境包使用：

```bash
bash paper/aaai27/submission_materials_20260728/remote_export/export_temp_fixed_paths.sh full
```

或者同时导出两类证据：

```bash
bash paper/aaai27/submission_materials_20260728/remote_export/export_temp_fixed_paths.sh all
```

固定路径脚本和未经匿名化的导出结果都不能放入 AAAI 上传 ZIP。
