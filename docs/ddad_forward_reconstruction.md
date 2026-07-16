# DDAD 前向重建评测计划

这份文档先记录 DDAD 的前向评测方案。当前阶段只做 forward-only
evaluation，不写训练流程；等前向指标和可视化可信之后，再讨论训练。

## 范围

- 数据集：`/data/jhc/ddad_train_val`。
- 数据格式：DDAD / DGP 风格目录。
- 硬件目标：4 张 RTX 4090，每张 48 GB。
- 输入：一段自动驾驶视频 clip。
- 第一版相机：`CAMERA_01`。
- 第一版帧数：48 帧，匹配已发布的 48-frame OpenD4RT 模型。
- 第一版 checkpoint：
  `checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt`，除非命令行显式指定别的 checkpoint。
- 第一阶段不做 tracking 评测，因为 DDAD 没有 D4RT 所需的点轨迹 GT。

## 本地 DDAD 数据内容

本地目录示例：

```text
/data/jhc/ddad_train_val/000166/
  scene_*.json
  calibration/*.json
  rgb/CAMERA_01/*.png
  rgb/CAMERA_05/*.png
  rgb/CAMERA_06/*.png
  rgb/CAMERA_07/*.png
  rgb/CAMERA_08/*.png
  rgb/CAMERA_09/*.png
  point_cloud/LIDAR/*.npz
```

已经确认的本地属性：

- 共有 200 个 scene 目录。
- 目录是数字 scene 平铺，不是 `train/val` 两级目录。
- `scene_*.json` 的 `description` 字段标记 split：
  `ddad_train` 有 150 个 scene，`000000-000149`；
  `ddad_val` 有 50 个 scene，`000150-000199`。
- 每个 scene 大约 50 个同步 sample。
- 每个 sample 有 6 路 RGB 相机和 1 路 LiDAR。
- RGB 原图尺寸是 `1936x1216`。
- LiDAR `.npz` 里有 `data` 数组，列含义是 `X,Y,Z,INTENSITY`。
- scene JSON 存 datum 文件名和每个 datum 的 pose。
- calibration JSON 存 sensor 名称、内参和外参。

前向脚本支持 `--split all/train/val`。训练前后的 held-out 对比应优先看
`SPLIT=val` 的指标；`SPLIT=all` 适合只想粗看 checkpoint 在本地全部 DDAD
scene 上的整体前向表现。

## 评测目标

DDAD 在这里应该作为重建 benchmark 使用，不作为 tracking benchmark 使用。
第一版主目标是：

```text
CAMERA_01 第 t 帧上的 query pixel
  -> 模型预测 CAMERA_01 第 t 帧相机坐标系下的 xyz_3d
  -> 和 LiDAR 投影得到的同相机坐标系 GT xyz 对比
```

第一版 query 方式：

```text
t_src = t_tgt = t_cam = t
```

这样避免假设跨帧静态对应关系，也避免把动态物体误当成可跟踪 GT。

## GT 构造

对每个评测帧：

1. 读取同步的 `CAMERA_01` RGB 和 `LIDAR` 点云。
2. 用 DDAD calibration 和 datum pose，把 LiDAR 点转换到目标相机坐标系。
3. 保留相机深度为正的点。
4. 用 resize 后的相机内参投影到模型输入图像。
5. 保留落在图像内的点。
6. 如果多个 LiDAR 点落到同一个像素，用 z-buffer 或最近深度规则去重。
7. 剩下的投影点作为 sparse GT query。

坐标和缩放要跟仓库现有约定一致：

- 相机坐标系：`+x` 向右，`+y` 向下，`+z` 向前。
- RGB 和内参从 `1936x1216` resize 到模型输入尺寸。
- query 的 normalized UV 参考现有代码：除以 `W - 1` 和 `H - 1`。
- 模型输入 RGB 是 `[T,3,H,W]`，数值范围 `[0,1]`。
- `aspect_ratio` 使用 resize/crop 前的原始宽高比。

## 复用现有模型接口

DDAD 不单独开一套模型前向路径，应该复用仓库已有 evaluation helper：

- `src.eval.tasks._encode_model_memory`
- `src.eval.tasks._run_model_for_queries`
- `infer_track_3d._resize_video`
- `infer_track_3d._resolve_device`
- `eval_track3d_in_worldtrack._unwrap_state_dict`

模型输出字段包括：

- `xyz_3d`
- `uv_2d`
- `visibility`
- `displacement`
- `normal`
- `confidence`

第一版 DDAD forward eval 主要使用 `xyz_3d`，可以记录 `confidence` 作为诊断信息。

## 指标

仓库当前 WorldTrack 评测使用 sequence-global scale alignment。DDAD 同样对每个
scene 的全部 local 稀疏点只估计一次 scale，并把它固定用于 local 和 reference-0
分支；不要按帧重新估计 scale。

必需指标：

- `local_depth_abs_rel_global`：scene-global scale alignment 后的 local depth AbsRel。
- `local_xyz_epe_global_m`：scene-global scale alignment 后的 local XYZ EPE。
- `ref0_xyz_epe_global_m`：使用同一 scene scale 的 reference-0 点云 EPE。
- `valid_queries`：实际用于评测的 LiDAR 投影 query 数。
- `scale_global`：该 scene 使用的对齐 scale，仅作协议与可视化诊断。

可选指标：

- 按距离分桶，例如 `0-20m`、`20-40m`、`40-80m`、`80m+`。
- confidence-weighted summary，用模型的 `confidence` head 做诊断。

注意：DDAD 重建结果不要命名为 WorldTrack APD。可以报 fixed-threshold
accuracy，但名字要体现它是 reconstruction threshold accuracy。

## 相机评测

相机指标有价值，但第一版不能作为主成功标准。原因是模型并不直接输出 `K`
或 `T_wc`，相机量需要从 query 预测后处理估计。

内参评测可以复用：

```text
src.eval.tasks._estimate_intrinsics_params_from_predictions
```

用预测 `xyz_3d` 和 query UV 反推 `[fx, fy, cx, cy]`，再和 DDAD resize 后的
内参比较。

候选内参指标：

- `fx_rel_error`
- `fy_rel_error`
- `cx_abs_error_px`
- `cy_abs_error_px`

位姿评测建议放在单独开关后面，因为它更脆弱。如果实现，可以用多帧预测 3D
query，通过仓库已有 Umeyama/rigid alignment 工具估计相对相机运动，再和 DDAD
GT 相对 pose 比较。

候选位姿指标：

- rotation error，单位 degree。
- translation direction error。
- raw translation error。
- scale-aligned translation error。

位姿指标会混合模型重建误差、后处理对齐误差、动态物体影响和 DDAD 坐标转换误差，
所以不作为第一阶段验收标准。

## 尺度处理

主报告只使用 aligned 指标，因为训练的 XYZ loss 对预测和 GT 分别做 mean-depth
normalization，并不监督绝对米制尺度。raw 指标不作为模型选择或结论依据。

第一版 aligned 路径应贴近仓库已有 global-scale 风格：

```text
pred_aligned = pred * global_scale
```

一个 scene 只计算一次 `global_scale`。reference-0 EPE 必须复用从 local 点估计的
同一个 scale，不能为点云分支重新对齐，也不使用 Sim3 掩盖参考系误差。

## 可视化输出

前向评测除了指标，也应该输出可视化结果。

推荐输出目录：

```text
output/ddad_reconstruction_eval/
  summary.json
  per_scene_metrics.jsonl
  vis/
    000166_CAMERA_01_triplet.mp4
    000166_CAMERA_01_triplet_raw.npz
    000166_CAMERA_01_pred_dense_ref0.ply
    000166_CAMERA_01_pred_gt_sparse_ref0_compare.ply
```

### 深度视频

对选定 scene 输出视频，包含：

- RGB 原图。
- 预测 depth。

D4RT 是 query 模型，不是直接 dense depth head，所以生成深度图需要网格 query。
当前 DDAD 脚本对每一帧规则网格使用 `t_src=t_tgt=t_cam=当前帧`
生成当前相机坐标系下的 dense `xyz_3d`，直接取 `z * scale_global`
作为预测 depth，整段视频按 5/95 percentile 统一上色。
只输出 `*_triplet.mp4` 三联视频：

```text
RGB | pred dense depth | sparse error overlay
```

右侧 sparse error overlay 只在 LiDAR 投影 GT 有效的位置画误差点。
绿色表示误差小，红色表示误差接近或超过 `--error-vis-max-m`。
DDAD 没有 dense depth GT，因此不能生成 dense GT depth 差值图。
默认 `--vis-fps=6`，48 帧约 8 秒。

- smoke 可视化：`64x64` grid。
- 更清晰可视化：`96x96` 或 `128x128` grid。
- 完整 `256x256` dense query 可做，但成本更高，适合离线可视化。

### ref0 点云

首选输出是 `*_pred_dense_ref0.ply`：

- 对每帧 dense grid 使用 `t_src=t_tgt=当前帧, t_cam=0`。
- 模型直接输出第 0 帧参考系下的 dense `xyz_3d`，不再用 DDAD pose 拼预测 dense 点云。
- 乘当前 scene 的 global median scale。
- 按 `--depth-vis-max-m` 过滤太远点。
- 保存为标准 ASCII 点云 PLY，建议用 CloudCompare、MeshLab 或 Open3D 打开。

如果要肉眼比较预测和真值，使用：

- `*_pred_gt_sparse_ref0_compare.ply`：蓝色是 LiDAR 投影 uv 上用 `t_cam=0` 预测的 sparse 点，橙色是同一批 uv 对应并用 DDAD pose 变到第 0 帧参考系的 LiDAR sparse GT。

这个对比和评估口径一致：同一个图像 uv 上，模型预测点和 LiDAR GT 点
组成一一对应的 pair。DDAD 没有 dense GT，因此 dense 预测点云只用于看完整
重建效果，不和 GT 直接混成一一对应对比。

这个可视化适合检查 48 帧内道路、建筑、整体场景是否连贯，但解释时要小心：

- raw 累积点云暴露米制尺度误差。
- scale-aligned 累积点云暴露形状一致性。
- 动态物体可能拖影，因为没有 tracking GT。

## 多 GPU 运行规划

使用 4 张 GPU 时，按 scene 分片，每个进程绑定一张卡：

```text
process 0 -> CUDA_VISIBLE_DEVICES=0 -> scene shard 0
process 1 -> CUDA_VISIBLE_DEVICES=1 -> scene shard 1
process 2 -> CUDA_VISIBLE_DEVICES=2 -> scene shard 2
process 3 -> CUDA_VISIBLE_DEVICES=3 -> scene shard 3
```

第一版推荐参数：

- `num_frames=48`
- `camera=CAMERA_01`
- `query_chunk_size=4096`
- `max_lidar_queries_per_frame=2048`，用于指标。
- `depth_vis_grid=64`，用于 smoke 可视化。

如果显存明显没吃满，可以提高：

- `query_chunk_size` 到 `8192` 或 `16384`。
- `max_lidar_queries_per_frame`。
- 可视化 grid 到 `96` 或 `128`。

## 当前实现入口

前向评测脚本：

```text
eval_reconstruction_in_ddad.py
```

4-GPU 启动脚本：

```text
scripts/eval_ddad_forward_4gpu.sh
```

### 单卡 smoke

先在激活好的 `d4rt` 环境里跑一个最小 smoke：

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

### 4 卡 smoke

每张卡跑一个 scene：

```bash
LIMIT_SCENES=1 \
SAVE_VISUALIZATIONS=true \
SAVE_PER_FRAME_NPZ=true \
OUTPUT_DIR=output/ddad_reconstruction_smoke_4gpu \
bash scripts/eval_ddad_forward_4gpu.sh
```

注意：`LIMIT_SCENES=1` 是每个 shard 1 个 scene，所以 4 张卡总共最多 4 个
scene。

### 4 卡完整前向评测

```bash
OUTPUT_DIR=output/ddad_reconstruction_eval \
CAMERA=CAMERA_01 \
NUM_FRAMES=48 \
QUERY_CHUNK_SIZE=4096 \
MAX_LIDAR_QUERIES_PER_FRAME=2048 \
bash scripts/eval_ddad_forward_4gpu.sh
```

如果显存余量明显，可以尝试：

```bash
QUERY_CHUNK_SIZE=8192 \
MAX_LIDAR_QUERIES_PER_FRAME=4096 \
OUTPUT_DIR=output/ddad_reconstruction_eval_q8192 \
bash scripts/eval_ddad_forward_4gpu.sh
```

### 输出文件

单卡或单 shard 会输出：

```text
per_scene_metrics_shard00.jsonl
summary_shard00.json
```

4-GPU 脚本会在所有 shard 完成后自动合并：

```text
summary.json
```

如果开启可视化，还会输出：

```text
vis/*_triplet.mp4
vis/*_triplet_raw.npz
vis/*_pred_dense_ref0.ply
vis/*_pred_gt_sparse_ref0_compare.ply
```

其中 `*_triplet.mp4` 是 `RGB | pred dense depth | sparse error overlay`；
`*_triplet_raw.npz` 保存三联视频对应的 raw depth、`vmin`、`vmax`
和 sparse 点误差表。点云优先看 `*_pred_dense_ref0.ply`。

## 里程碑

### Milestone 1: forward smoke

先跑小规模 forward-only：

- 总共 4 个 scene。
- 48 帧。
- `CAMERA_01`。
- 指标为主。
- 输出 1 个深度可视化视频。

验收标准：

- LiDAR 投影点落在正确图像位置。
- 大多数帧 `valid_queries` 非零。
- raw 和 global-scale 指标都是 finite。
- 预测深度视频不是空白。

### Milestone 2: full DDAD forward

再跑完整 200 个 scene：

- 4-GPU scene sharding。
- 每个 scene 输出指标。
- 只对固定少量 scene 输出可视化。

验收标准：

- 产出 `summary.json` 和 `per_scene_metrics.jsonl`。
- 失败 scene 记录 scene id 和原因。
- 聚合 raw/aligned depth 和 xyz 指标稳定。

### Milestone 3: 训练计划

DDAD 训练设计已经整理到：

```text
docs/ddad_training_plan.md
```

核心原则：

- 复用项目现有 `train.py`、dataset builder、query decoder 和 `D4RTLoss`。
- 新增 DDAD dataset adapter，把 LiDAR 投影 sparse 点转换成现有
  `query/target/mask/camera` schema。
- 第一阶段只用 LiDAR sparse `xyz_3d/depth` 监督，不做 tracking loss。
- 训练前后都用 `SPLIT=val` 的 DDAD forward metrics 对比。
