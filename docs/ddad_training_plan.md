# DDAD 训练设计和运行流程

这份文档记录 DDAD fine-tuning 的代码设计和运行流程。当前仓库已经有 DDAD
前向评测，也已经接入第一阶段 DDAD sparse LiDAR reconstruction 训练入口。

## 1. 训练目标

DDAD 没有 D4RT 需要的长期点轨迹 GT，所以不做 tracking 监督。训练目标聚焦
重建：

- 用 LiDAR 投影到 `CAMERA_01` 的 sparse 点监督 `xyz_3d`。
- 用同一批 LiDAR 点的目标帧投影监督 `uv_2d`，仅在静态深度一致时打开。
- 用 LiDAR/相机投影关系监督 `visibility`，第一版只对可见正样本置 1。
- `displacement` 第一版只在静态一致点上监督为 0。
- `normal` 第一版关闭；DDAD LiDAR 太稀疏，不适合直接从 sparse depth 稳定估计法线。
- `confidence` 保持开启，因为论文和当前项目都把它作为 3D 误差加权和相机/几何稳定性的辅助项。

论文的 query 机制和 DDAD 对齐方式：

- 深度图：`t_src=t_tgt=t_cam=t`，取输出 `xyz_3d[..., 2]`。
- 共享参考系点云：`t_src=t_tgt=t`，`t_cam=0` 或随机参考帧。
- 训练 sparse 重建：从 LiDAR 投影到 source 帧的有效像素采样 `(u,v,t_src)`，
  用 DDAD pose 把该 3D 点变到 `t_cam` 坐标系，作为 `target.xyz_3d`。

## 2. 不做什么

第一版不要做这些：

- 不训练 point-track 轨迹，因为没有动态点轨迹 GT。
- 不用 DDAD 的相机 pose 当作直接 pose regression loss；项目本身是通过 query 出来的
  3D 点集合再估相机位姿，前向评测里也按这个逻辑做诊断。
- 不把 LiDAR sparse depth 插值成 dense depth 再当真值；这会把空洞和遮挡错误带进训练。
- 不改模型结构；复用 `train.py`、`D4RTLoss`、query decoder 和当前 checkpoint。

## 3. 需要新增的代码

### 3.1 DDAD Dataset

已新增文件：

```text
src/data/ddad_raw_dataset.py
```

建议配置类：

```text
DdadRawConfig
  root: Path = /data/jhc/ddad_train_val
  split: train|val
  camera: str = CAMERA_01
  clip_frames: int = 48
  image_size: tuple[int, int] = (256, 256)
  queries_per_clip: int = 4096
  hard_query_ratio: float = 0.2
  prob_t_tgt_equals_t_cam: float = 0.4
  t_src_tgt_delta_choices / probs
  max_scenes: optional int
  max_lidar_points_per_frame: int
  depth_consistency_abs_m: float = 0.5
  depth_consistency_rel: float = 0.05
  augment: RawAugmentConfig
```

Dataset `__getitem__` 返回现有 trainer/loss 已经认识的 schema：

```text
video:        FloatTensor [T, 3, H, W], RGB in [0,1]
query:
  u, v:       FloatTensor [M], normalized to [0,1]
  t_src:      LongTensor [M]
  t_tgt:      LongTensor [M]
  t_cam:      LongTensor [M]
target:
  xyz_3d:     FloatTensor [M,3], in t_cam camera coordinates
  uv_2d:      FloatTensor [M,2], target frame normalized uv
  visibility: FloatTensor [M]
  displacement: FloatTensor [M,3]
  normal:     FloatTensor [M,3], zeros in v1
mask:
  xyz_3d:     BoolTensor [M]
  uv_2d:      BoolTensor [M]
  visibility: BoolTensor [M]
  displacement: BoolTensor [M]
  normal:     BoolTensor [M], all false in v1
camera:
  K:          FloatTensor [T,3,3], resized intrinsics
  T_wc:       FloatTensor [T,4,4], camera-to-world
  valid:      BoolTensor [T]
query_stats:
  is_hard_query: BoolTensor [M]
meta:
  dataset, scene_id, split, camera
```

### 3.2 Query 构造

已新增：

```text
src/data/ddad_query_builder.py
```

核心逻辑：

1. 对 clip 内每帧读取 RGB、LiDAR、相机内参、`T_wc`。
2. 将 LiDAR 点变换到当前 camera 坐标系并投影到 resize 后的模型图像。
3. 对每帧保留图像内、`z>0`、有限值的点；同一像素多点时保留最近深度。
4. 按 `queries_per_clip` 采样 query：
   - `t_src` 均匀采样。
   - `t_tgt/t_cam` 复用项目现有 `sample_t_tgt_t_cam`。
   - `hard_query_ratio` 可用 sparse depth 的深度边界近似；第一版可以先设 0 或轻量实现。
5. 对每个 query：
   - source 点来自 `t_src` 的 LiDAR 投影。
   - `xyz_3d`：把 source LiDAR 3D 点从 `t_src` camera/world 变到 `t_cam` camera。
   - `uv_2d`：把同一个 world 点投影到 `t_tgt` camera；如果落在图像内且与
     `t_tgt` LiDAR 最近深度一致，则打开 `mask.uv_2d`。
   - `visibility`：第一版只对上述可见一致点置 1；负样本可后续再加。
   - `displacement`：静态一致点监督为 0；动态车/行人没有轨迹 GT，不强行监督。

DDAD 是自驾场景，动态物体会破坏“同一 LiDAR world 点跨帧静态一致”的假设。所以：

- `xyz_3d` 可以对 source 点本身监督，因为这是单帧重建。
- 跨帧 `uv_2d/displacement` 必须经过 target depth consistency 过滤。
- 如果训练不稳定，第一轮先把 `prob_t_tgt_equals_t_cam=1.0`，只训练本帧深度/局部重建。

### 3.3 Builder 注册

已修改：

```text
src/data/builder.py
```

已注册：

```text
from .ddad_raw_dataset import DdadRawConfig, DdadRawDataset

@DATASET_REGISTRY.register("ddad_raw")
def _build_ddad_raw(split: str, cfg: Any, manifest_paths: list[str] | None = None):
    ...
```

`split=train` 映射 `ddad_train`，`split=val` 映射 `ddad_val`。

## 4. 配置文件

已新增：

```text
configs/train_ddad_reconstruction.yaml
scripts/train_ddad_reconstruction_4gpu.sh
```

训练配置建议从 `configs/train_effective.yaml` 拷贝后最小化修改：

```yaml
experiment:
  name: d4rt_ddad_reconstruction_clip48_cam01
  output_dir: output/exp_ddad_reconstruction/d4rt_ddad_clip48_cam01

runtime:
  mixed_precision: true
  train_batch_size: 1
  val_batch_size: 1
  train_num_workers: 4
  val_num_workers: 2
  find_unused_parameters: true

data:
  clip_frames: 48
  image_size: [256, 256]
  train_dataset_type: ddad_raw
  val_dataset_type: ddad_raw
  ddad:
    root: /data/jhc/ddad_train_val
    camera: CAMERA_01
    max_lidar_points_per_frame: 20000
    depth_consistency_abs_m: 0.5
    depth_consistency_rel: 0.05

train_sampling:
  queries_per_clip: 4096
  hard_query_ratio: 0.0
  timestep_sampling:
    prob_t_tgt_equals_t_cam: 1.0

loss:
  xyz_3d:
    enabled: true
    normalize_by_mean_depth: true
    value_transform: sign_x_log1p_abs_x
    weight_lambda_3d: 1.0
  uv_2d:
    enabled: false
  visibility:
    enabled: false
  displacement:
    enabled: false
  normal:
    enabled: false
  confidence:
    enabled: true
    mode: main_text
    weight_lambda_conf: 0.2
    confidence_penalty: -log(c)
    confidence_weights_xyz_error: true
```

这是第一阶段“稳”的设置，只训练单帧 LiDAR sparse depth/xyz。跑通后再进入第二阶段：

因为第一阶段关闭了 `uv_2d/visibility/displacement/normal` loss，对应 heads 的参数不会参与
loss，DDP 需要 `runtime.find_unused_parameters=true`。原项目默认是 false；这个配置只用于
DDAD reconstruction fine-tuning。

```yaml
train_sampling:
  hard_query_ratio: 0.2
  timestep_sampling:
    prob_t_tgt_equals_t_cam: 0.4
    t_src_tgt_delta_mode: static_local_global
loss:
  uv_2d.enabled: true
  visibility.enabled: true
  displacement.enabled: true
```

## 5. 4 卡训练命令

训练脚本应使用 DDP，不要用 4 个独立进程各训各的。

先做 dataset smoke，只取一个 sample，检查 tensor shape、finite 和 mask：

```bash
conda activate d4rt
cd /home/jhc/zkr/Open-d4rt

python scripts/check_ddad_dataset_smoke.py \
  --train-config configs/train_ddad_reconstruction.yaml \
  --split train \
  --max-scenes 2 \
  --queries-per-clip 512
```

先做 dry run，只检查命令展开和关键路径，不启动训练：

```bash
conda activate d4rt
cd /home/jhc/zkr/Open-d4rt

bash scripts/train_ddad_reconstruction_4gpu.sh \
  --output-dir output/ddad_reconstruction_train \
  --init-model checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --dry-run
```

推荐默认从当前 48-frame checkpoint 初始化：

```bash
conda activate d4rt
cd /home/jhc/zkr/Open-d4rt

bash scripts/train_ddad_reconstruction_4gpu.sh \
  --output-dir output/ddad_reconstruction_train \
  --init-model checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --total-steps 20000 \
  --peak-lr 4e-6 \
  --train-batch-size 1 \
  --val-batch-size 1
```

`scripts/train_ddad_reconstruction_4gpu.sh` 默认会额外传入：

```text
--override model.encoder.pretrained.enabled=false
```

原因是 DDAD fine-tuning 使用完整 `--init-model`：

```text
checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
```

这个 checkpoint 会初始化 encoder、decoder、query embedding 和 heads；构建模型时不需要先加载
VideoMAE2 encoder 权重。

如果以后不用完整 OpenD4RT checkpoint，而是想先加载 VideoMAE2 encoder，可以显式加：

```bash
--load-encoder-pretrained \
--videomae2-ckpt videomae2/mae-g/vit_g_hybrid_pt_1200e.pth
```

脚本内部应等价于：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port 29714 \
  train.py \
  --tb_log \
  --model-config configs/model_effective.yaml \
  --train-config configs/train_ddad_reconstruction.yaml \
  --init-model checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
```

4090 48GB 的第一版保守建议：

- `train_batch_size=1` per GPU。
- `queries_per_clip=4096`。
- `clip_frames=48`。
- 如果显存明显富余，再尝试 `queries_per_clip=8192` 或 batch size 2。

## 6. 训练前后评测闭环

DDAD 训练逻辑参考原项目：

- 固定 `schedule.total_steps` 训练，不默认 early stopping。
- 每 `logging.validate_every_steps` 做一次 validation。
- `checkpoints/best.ckpt` 按 `checkpoint.keep_best_by=val_loss_total` 保存，越低越好。
- `checkpoints/last.ckpt` 保存最近状态，用于 resume。
- `checkpoints/step_*.ckpt` 每 `checkpoint.step_save_every_steps` 保存一次，用于外部评测。

和原项目一致，正式训练产物里的最佳模型统一认：

```text
checkpoints/best.ckpt
```

它按 `val_loss_total` 最低保存。WorldTrack auto-eval 对 DDAD 不适用，DDAD
配置里已经关闭 `checkpoint.auto_eval_worldtrack_step.enabled`。后面的 step
checkpoint DDAD 评测只作为分析报告，不改变 `best.ckpt` 的含义。

训练前 baseline 已经跑过：

```text
output/ddad_reconstruction_eval_before/summary.json
```

当前 pretrain val 主指标：

```text
xyz_epe_global_m:      9.3239
depth_mae_global_m:    7.3101
depth_abs_rel_global:  0.1514
```

训练后评测命令：

```bash
bash scripts/eval_ddad_forward_4gpu.sh \
  --output-dir output/ddad_reconstruction_eval_after \
  --split val \
  --ckpt-path output/ddad_reconstruction_train/checkpoints/best.ckpt \
  --camera CAMERA_01 \
  --num-frames 48 \
  --query-chunk-size 4096 \
  --max-lidar-queries-per-frame 2048 \
  --no-vis
```

如果想像原项目一样额外分析 step checkpoint 的 DDAD 指标，跑：

```bash
bash scripts/eval_ddad_step_ckpts.sh \
  --checkpoint-dir output/ddad_reconstruction_train/checkpoints \
  --output-root output/ddad_reconstruction_step_eval \
  --split val \
  --metric depth_abs_rel_global
```

输出：

```text
output/ddad_reconstruction_step_eval/ddad_step_eval_report.json
```

这里的 `best.step` 是分析报告里的 DDAD 指标最优 step，不会覆盖训练正式产物
`checkpoints/best.ckpt`。

对比：

- `depth_abs_rel_global` 是否低于 `0.1514`。
- `depth_mae_global_m` 是否低于 `7.3101`。
- `xyz_epe_global_m` 是否低于 `9.3239`。
- `scale_global` 是否更接近稳定，不要只看 raw 尺度。

## 7. 验收标准

训练代码实现完成后，先做三步：

1. Dataset smoke：单进程取 2 个 batch，检查 tensor shape、finite、mask 比例。
2. 4 GPU dry run：DDP 启动 10 step，确认 loss finite、没有 dataloader 卡死。
3. 小训练：500-1000 step，确认 `loss_xyz_3d` 下降，再跑 4 个 val scene 的前向指标。

只有这些通过后，再跑完整 20k step 或更长。
