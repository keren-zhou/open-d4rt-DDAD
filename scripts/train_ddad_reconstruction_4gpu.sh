#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model_effective.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_ddad_reconstruction.yaml}"
INIT_MODEL="${INIT_MODEL:-checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt}"
VIDEOMAE2_CKPT="${VIDEOMAE2_CKPT:-videomae2/mae-g/vit_g_hybrid_pt_1200e.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ddad_reconstruction_train}"
DATA_ROOT="${DATA_ROOT:-/data/jhc/ddad_train_val}"
CAMERA="${CAMERA:-CAMERA_01}"
GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29714}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
PEAK_LR="${PEAK_LR:-4e-6}"
FINAL_LR="${FINAL_LR:-4e-7}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
QUERIES_PER_CLIP="${QUERIES_PER_CLIP:-4096}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
VALIDATE_EVERY_STEPS="${VALIDATE_EVERY_STEPS:-2000}"
VALIDATE_MAX_SAMPLES_GLOBAL="${VALIDATE_MAX_SAMPLES_GLOBAL:-256}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
STEP_SAVE_EVERY_STEPS="${STEP_SAVE_EVERY_STEPS:-1000}"
TB_LOG="${TB_LOG:-true}"
DRY_RUN="${DRY_RUN:-false}"
LOAD_ENCODER_PRETRAINED="${LOAD_ENCODER_PRETRAINED:-false}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_ddad_reconstruction_4gpu.sh [options]

Options:
  --model-config PATH
  --train-config PATH
  --init-model PATH
  --videomae2-ckpt PATH            Encoder pretrained path used only with --load-encoder-pretrained.
  --output-dir PATH
  --data-root PATH
  --camera NAME
  --gpus IDS                      Comma-separated GPU ids. Default: 0,1,2,3
  --master-port PORT              torchrun master port. Default: 29714
  --total-steps N                 Default: 20000
  --peak-lr LR                    Default: 4e-6
  --final-lr LR                   Default: 4e-7
  --warmup-steps N                Default: 500
  --train-batch-size N            Per-GPU batch size. Default: 1
  --val-batch-size N              Per-GPU batch size. Default: 1
  --queries-per-clip N            Default: 4096
  --train-num-workers N           Default: 4
  --val-num-workers N             Default: 2
  --validate-every-steps N        Default: 2000
  --validate-max-samples-global N Default: 256
  --save-every-steps N            Default: 1000
  --step-save-every-steps N       Default: 1000
  --no-tb                         Disable TensorBoard logging.
  --load-encoder-pretrained       Keep model encoder.pretrained enabled before --init-model.
  --dry-run                       Print command and exit.
  -h, --help                      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
    --init-model) INIT_MODEL="$2"; shift 2 ;;
    --videomae2-ckpt) VIDEOMAE2_CKPT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --camera) CAMERA="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
    --peak-lr) PEAK_LR="$2"; shift 2 ;;
    --final-lr) FINAL_LR="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val-batch-size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --queries-per-clip) QUERIES_PER_CLIP="$2"; shift 2 ;;
    --train-num-workers) TRAIN_NUM_WORKERS="$2"; shift 2 ;;
    --val-num-workers) VAL_NUM_WORKERS="$2"; shift 2 ;;
    --validate-every-steps) VALIDATE_EVERY_STEPS="$2"; shift 2 ;;
    --validate-max-samples-global) VALIDATE_MAX_SAMPLES_GLOBAL="$2"; shift 2 ;;
    --save-every-steps) SAVE_EVERY_STEPS="$2"; shift 2 ;;
    --step-save-every-steps) STEP_SAVE_EVERY_STEPS="$2"; shift 2 ;;
    --no-tb) TB_LOG=false; shift ;;
    --load-encoder-pretrained) LOAD_ENCODER_PRETRAINED=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for path in "$MODEL_CONFIG" "$TRAIN_CONFIG" "$INIT_MODEL"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Required file not found: $path" >&2
    exit 1
  fi
done
if [[ "$LOAD_ENCODER_PRETRAINED" == "true" && ! -f "$VIDEOMAE2_CKPT" ]]; then
  echo "[ERROR] VideoMAE2 checkpoint not found: $VIDEOMAE2_CKPT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DDAD data root not found: $DATA_ROOT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${#GPU_LIST[@]}"

export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export D4RT_CV2_WORKER_THREADS="${D4RT_CV2_WORKER_THREADS:-0}"
export D4RT_TORCH_WORKER_THREADS="${D4RT_TORCH_WORKER_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
  torchrun
  --nnodes=1
  --nproc_per_node="$NPROC_PER_NODE"
  --master_addr=127.0.0.1
  --master_port="$MASTER_PORT"
  train.py
  --model-config "$MODEL_CONFIG"
  --train-config "$TRAIN_CONFIG"
  --init-model "$INIT_MODEL"
  --override "experiment.output_dir=${OUTPUT_DIR}"
  --override "data.ddad.root=${DATA_ROOT}"
  --override "data.ddad.camera=${CAMERA}"
  --override "schedule.total_steps=${TOTAL_STEPS}"
  --override "optimizer.learning_rate.peak_lr=${PEAK_LR}"
  --override "optimizer.learning_rate.final_lr=${FINAL_LR}"
  --override "optimizer.learning_rate.warmup_steps=${WARMUP_STEPS}"
  --override "runtime.train_batch_size=${TRAIN_BATCH_SIZE}"
  --override "runtime.batch_size=${TRAIN_BATCH_SIZE}"
  --override "runtime.val_batch_size=${VAL_BATCH_SIZE}"
  --override "runtime.train_num_workers=${TRAIN_NUM_WORKERS}"
  --override "runtime.val_num_workers=${VAL_NUM_WORKERS}"
  --override "train_sampling.queries_per_clip=${QUERIES_PER_CLIP}"
  --override "logging.validate_every_steps=${VALIDATE_EVERY_STEPS}"
  --override "logging.validate_max_samples_global=${VALIDATE_MAX_SAMPLES_GLOBAL}"
  --override "checkpoint.save_every_steps=${SAVE_EVERY_STEPS}"
  --override "checkpoint.step_save_every_steps=${STEP_SAVE_EVERY_STEPS}"
)

if [[ "$LOAD_ENCODER_PRETRAINED" != "true" ]]; then
  CMD+=(--override "model.encoder.pretrained.enabled=false")
else
  CMD+=(--override "model.encoder.pretrained.enabled=true")
  CMD+=(--override "model.encoder.pretrained.path=${VIDEOMAE2_CKPT}")
fi

if [[ "$TB_LOG" == "true" ]]; then
  CMD+=(--tb_log)
fi

echo "================================================================================"
echo "OpenD4RT DDAD reconstruction training"
echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "INIT_MODEL=${INIT_MODEL}"
echo "VIDEOMAE2_CKPT=${VIDEOMAE2_CKPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CAMERA=${CAMERA}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "TOTAL_STEPS=${TOTAL_STEPS}"
echo "QUERIES_PER_CLIP=${QUERIES_PER_CLIP}"
echo "LOAD_ENCODER_PRETRAINED=${LOAD_ENCODER_PRETRAINED}"
echo "================================================================================"
printf '%q ' "${CMD[@]}"
echo

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

"${CMD[@]}"
