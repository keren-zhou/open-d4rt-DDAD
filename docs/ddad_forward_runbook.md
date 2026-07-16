# DDAD 前向重建评测运行流程

这份 runbook 只记录如何运行当前已经实现的 DDAD forward-only 重建评测。
DDAD 训练代码设计见 `docs/ddad_training_plan.md`。

## 1. 环境准备

进入项目目录并激活环境：

```bash
cd /home/jhc/zkr/Open-d4rt
conda activate d4rt
```

确认关键文件存在：

```bash
ls /data/jhc/ddad_train_val
ls checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml
ls checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
```

当前默认使用：

```text
DDAD root: /data/jhc/ddad_train_val
split: all
camera: CAMERA_01
frames: 48
checkpoint: checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
```

本地 `/data/jhc/ddad_train_val` 不是 `train/val` 两级目录，而是 200 个数字
scene 平铺。每个 scene 的 `scene_*.json` 里用 `description` 标记 split：

```text
ddad_train: 150 scenes, 000000-000149
ddad_val:    50 scenes, 000150-000199
```

脚本支持：

```text
--split all    跑 train+val，共 200 个 scene
--split train  只跑 ddad_train，共 150 个 scene
--split val    只跑 ddad_val，共 50 个 scene
--vis          保存三联视频、raw npz 和 PLY 点云
--no-vis       不保存视频、raw npz 和点云，只保存指标
```

## 2. 单卡 smoke test

先用 1 张卡跑 1 个 scene，确认数据解析、LiDAR 投影、模型前向和输出文件都正常：

```bash
CUDA_VISIBLE_DEVICES=0 python eval_reconstruction_in_ddad.py \
  --model-config checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml \
  --ckpt-path checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --data-root /data/jhc/ddad_train_val \
  --output-dir output/ddad_reconstruction_smoke \
  --camera CAMERA_01 \
  --num-frames 48 \
  --query-chunk-size 4096 \
  --max-lidar-queries-per-frame 2048 \
  --limit-scenes 1 \
  --device cuda \
  --save-visualizations \
  --save-per-frame-npz
```

检查输出：

```bash
ls output/ddad_reconstruction_smoke
ls output/ddad_reconstruction_smoke/vis
```

重点看：

```text
summary_shard00.json
per_scene_metrics_shard00.jsonl
vis/*_triplet.mp4
vis/*_triplet_raw.npz
vis/*_pred_dense_ref0.ply
vis/*_pred_gt_sparse_ref0_compare.ply
```

## 3. 4 卡 smoke test

每张 4090 跑 1 个 scene，总共最多 4 个 scene：

```bash
bash scripts/eval_ddad_forward_4gpu.sh \
  --output-dir output/ddad_reconstruction_smoke_4gpu \
  --split val \
  --limit-scenes 1 \
  --vis
```

查看日志：

```bash
tail -n 80 output/ddad_reconstruction_smoke_4gpu/logs/shard_0.log
tail -n 80 output/ddad_reconstruction_smoke_4gpu/logs/shard_1.log
tail -n 80 output/ddad_reconstruction_smoke_4gpu/logs/shard_2.log
tail -n 80 output/ddad_reconstruction_smoke_4gpu/logs/shard_3.log
```

查看合并结果：

```bash
cat output/ddad_reconstruction_smoke_4gpu/summary.json
```

## 4. 4 卡完整评测

确认 smoke 没问题后，如果只是想给当前 checkpoint 做一次全量前向统计，可以跑完整
200 个 scene：

```bash
bash scripts/eval_ddad_forward_4gpu.sh \
  --output-dir output/ddad_reconstruction_eval_all \
  --split all \
  --camera CAMERA_01 \
  --num-frames 48 \
  --query-chunk-size 4096 \
  --max-lidar-queries-per-frame 2048 \
  --no-vis
```

如果后面要训练，建议同时保留一个只在官方 val 上的 pretrain baseline，作为训练后
对比的 held-out 指标：

```bash
bash scripts/eval_ddad_forward_4gpu.sh \
  --output-dir output/ddad_reconstruction_eval_before \
  --split val \
  --camera CAMERA_01 \
  --num-frames 48 \
  --query-chunk-size 4096 \
  --max-lidar-queries-per-frame 2048 \
  --no-vis
```

如果需要记录 train split 上的拟合前基线：

```bash
bash scripts/eval_ddad_forward_4gpu.sh \
  --output-dir output/ddad_reconstruction_eval_train \
  --split train \
  --camera CAMERA_01 \
  --num-frames 48 \
  --query-chunk-size 4096 \
  --max-lidar-queries-per-frame 2048 \
  --no-vis
```

输出目录：

```text
output/ddad_reconstruction_eval_all/
  summary.json
  summary_shard00.json
  summary_shard01.json
  summary_shard02.json
  summary_shard03.json
  per_scene_metrics_shard00.jsonl
  per_scene_metrics_shard01.jsonl
  per_scene_metrics_shard02.jsonl
  per_scene_metrics_shard03.jsonl
  logs/
```

`summary.json` 里会记录 `split_counts`，用于确认这次实际合并了多少
`ddad_train` / `ddad_val` scene。

## 5. 可视化输出

完整评测默认不保存可视化，避免输出过大。

如果要保存可视化：

```bash
rm -rf output/ddad_reconstruction_vis

bash scripts/eval_ddad_forward_4gpu.sh \
  --output-dir output/ddad_reconstruction_vis \
  --split val \
  --limit-scenes 1 \
  --vis \
  --vis-fps 6
```

如果重新跑同一个 `OUTPUT_DIR`，建议先删掉旧目录，避免上一次失败留下的空
MP4 或旧点云文件混在一起。

可视化文件：

```text
vis/*_triplet.mp4
vis/*_triplet_raw.npz
vis/*_pred_dense_ref0.ply
vis/*_pred_gt_sparse_ref0_compare.ply
```

说明：

- `*_triplet.mp4`：三联视频，顺序是 `RGB | pred dense depth | sparse error overlay`。右侧只在 LiDAR 投影 GT 有效的位置画点，绿色表示误差小，红色表示误差接近或超过 `ERROR_VIS_MAX_M`。默认 `VIS_FPS=6`，48 帧约 8 秒。
- `*_triplet_raw.npz`：三联视频对应的原始 dense depth、`vmin`、`vmax`，以及 sparse 点上的 `frame_idx/u_norm/v_norm/gt_depth/pred_depth/abs_error`，用于排查颜色范围和误差。
- `*_pred_dense_ref0.ply`：对每帧 dense grid 使用 `t_src=t_tgt=当前帧, t_cam=0`，模型直接输出第 0 帧参考系下的 dense 点云；再乘 global scale，并按 `DEPTH_VIS_MAX_M` 过滤太远点。优先用 CloudCompare、MeshLab 或 Open3D 看这个文件。
- `*_pred_gt_sparse_ref0_compare.ply`：预测和 GT 放在同一个第 0 帧参考系里；蓝色是 LiDAR 投影 uv 上用 `t_cam=0` 预测的 sparse 点，橙色是同一批 uv 对应并用 DDAD pose 变到第 0 帧参考系的 LiDAR sparse GT。这个文件和评估点一一对应。

如果你想点云更稠密，可以把 `DEPTH_VIS_GRID=128` 或 `DEPTH_VIS_GRID=256`。
模型输入默认是 `256x256`，所以 `256` 基本就是模型输入尺度的每像素 query；
不要按 DDAD 原始 `1936x1216` 每像素 query，代价会非常大。

注意：这里的 PLY 是普通点云 PLY，字段是 `x y z red green blue`。
SuperSplat 这类 Gaussian Splat 查看器通常期待 3DGS 专用 PLY 属性，
不一定能打开普通点云文件；如果它提示损坏，先用 CloudCompare、MeshLab
或 Open3D 验证。

## 6. 指标含义

主要看 `summary.json` 里的：

```text
local_depth_abs_rel_global
local_xyz_epe_global_m
ref0_xyz_epe_global_m
scale_global
total_queries
```

解释：

- `global`：沿用项目 WorldTrack 风格，每个 scene 只估计一个 scale。
- `local` 和 `ref0` 共用该 scale，ref0 不单独对齐。
- raw 和 Sim3 不作为主评测指标。
- `total_queries`：实际参与评测的 LiDAR 投影点数量。

## 7. 调参建议

默认参数：

```text
QUERY_CHUNK_SIZE=4096
MAX_LIDAR_QUERIES_PER_FRAME=2048
NUM_FRAMES=48
CAMERA=CAMERA_01
```

如果显存没吃满，可以尝试：

```bash
bash scripts/eval_ddad_forward_4gpu.sh \
  --output-dir output/ddad_reconstruction_eval_q8192 \
  --query-chunk-size 8192 \
  --max-lidar-queries-per-frame 4096 \
  --no-vis
```

如果显存不够，先降：

```bash
bash scripts/eval_ddad_forward_4gpu.sh \
  --output-dir output/ddad_reconstruction_eval_safe \
  --query-chunk-size 2048 \
  --max-lidar-queries-per-frame 1024 \
  --no-vis
```

## 8. 常见问题

如果报 `ModuleNotFoundError: No module named 'cv2'`，说明当前 Python 环境不是项目环境，先激活 `d4rt` 或安装 requirements。

如果某些 scene 被跳过，先看对应日志：

```bash
grep -R "Skip scene\\|Failed scene" output/ddad_reconstruction_eval/logs
```

如果 `valid_queries` 很低，优先检查 LiDAR 到相机投影和相机名是否正确。

## 9. DDAD 训练入口

训练设计和命令见：

```text
docs/ddad_training_plan.md
```

第一阶段训练入口：

```bash
bash scripts/train_ddad_reconstruction_4gpu.sh --help
```
