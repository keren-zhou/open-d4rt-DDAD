#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/jhc/ddad_train_val}"
MODEL_CONFIG="${MODEL_CONFIG:-checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml}"
CKPT_PATH="${CKPT_PATH:-checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ddad_reconstruction_eval}"
SPLIT="${SPLIT:-all}"
CAMERA="${CAMERA:-CAMERA_01}"
NUM_FRAMES="${NUM_FRAMES:-48}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-4096}"
MAX_LIDAR_QUERIES_PER_FRAME="${MAX_LIDAR_QUERIES_PER_FRAME:-2048}"
DEPTH_VIS_GRID="${DEPTH_VIS_GRID:-64}"
DEPTH_VIS_MAX_M="${DEPTH_VIS_MAX_M:-80}"
ERROR_VIS_MAX_M="${ERROR_VIS_MAX_M:-10}"
VIS_FPS="${VIS_FPS:-6}"
LIMIT_SCENES="${LIMIT_SCENES:-0}"
GPUS="${GPUS:-0,1,2,3}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-false}"
SAVE_PER_FRAME_NPZ="${SAVE_PER_FRAME_NPZ:-false}"
SAVE_LOCAL_PLY="${SAVE_LOCAL_PLY:-false}"
SAVE_WORLD_PLY="${SAVE_WORLD_PLY:-false}"

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_ddad_forward_4gpu.sh [options]

Options:
  --data-root PATH             DDAD root. Default: /data/jhc/ddad_train_val
  --model-config PATH          Model yaml path.
  --ckpt-path PATH             Checkpoint path.
  --output-dir PATH            Output directory.
  --split all|train|val        DDAD split to evaluate. Default: all
  --camera NAME                Camera name. Default: CAMERA_01
  --num-frames N               Frames per scene. Default: 48
  --query-chunk-size N         Model query chunk size. Default: 4096
  --max-lidar-queries-per-frame N
                               Sparse LiDAR eval queries per frame. Default: 2048
  --limit-scenes N             Limit scenes per shard after split/shard. Default: 0
  --gpus IDS                   Comma-separated GPU ids. Default: 0,1,2,3
  --vis                        Save triplet videos, raw npz, and PLY files.
  --no-vis                     Do not save videos, raw npz, or PLY files.
  --save-triplet-video         Save triplet mp4 visualizations.
  --save-raw-npz               Save raw triplet npz payloads.
  --save-local-ply             Save ref0 PLY point clouds.
  --save-world-ply             Save world PLY point clouds.
  --depth-vis-grid N           Dense visualization grid side length. Default: 64
  --depth-vis-max-m M          Max depth for visualization/PLY filtering. Default: 80
  --error-vis-max-m M          Max sparse overlay error color range. Default: 10
  --vis-fps FPS                Visualization fps. Default: 6
  -h, --help                   Show this help.

Environment variables with the same names are still supported, but command-line
options take precedence.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --model-config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --ckpt-path)
      CKPT_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --camera)
      CAMERA="$2"
      shift 2
      ;;
    --num-frames)
      NUM_FRAMES="$2"
      shift 2
      ;;
    --query-chunk-size)
      QUERY_CHUNK_SIZE="$2"
      shift 2
      ;;
    --max-lidar-queries-per-frame)
      MAX_LIDAR_QUERIES_PER_FRAME="$2"
      shift 2
      ;;
    --limit-scenes)
      LIMIT_SCENES="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --vis)
      SAVE_VISUALIZATIONS=true
      SAVE_PER_FRAME_NPZ=true
      SAVE_LOCAL_PLY=true
      shift
      ;;
    --no-vis)
      SAVE_VISUALIZATIONS=false
      SAVE_PER_FRAME_NPZ=false
      SAVE_LOCAL_PLY=false
      SAVE_WORLD_PLY=false
      shift
      ;;
    --save-triplet-video)
      SAVE_VISUALIZATIONS=true
      shift
      ;;
    --save-raw-npz)
      SAVE_PER_FRAME_NPZ=true
      shift
      ;;
    --save-local-ply)
      SAVE_LOCAL_PLY=true
      shift
      ;;
    --save-world-ply)
      SAVE_WORLD_PLY=true
      shift
      ;;
    --depth-vis-grid)
      DEPTH_VIS_GRID="$2"
      shift 2
      ;;
    --depth-vis-max-m)
      DEPTH_VIS_MAX_M="$2"
      shift 2
      ;;
    --error-vis-max-m)
      ERROR_VIS_MAX_M="$2"
      shift 2
      ;;
    --vis-fps)
      VIS_FPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
SHARD_COUNT="${#GPU_LIST[@]}"

mkdir -p "${OUTPUT_DIR}/logs"

for SHARD_INDEX in "${!GPU_LIST[@]}"; do
  GPU_ID="${GPU_LIST[$SHARD_INDEX]}"
  EXTRA_ARGS=()
  if [[ "${SAVE_VISUALIZATIONS}" == "true" ]]; then
    EXTRA_ARGS+=(--save-visualizations)
  fi
  if [[ "${SAVE_PER_FRAME_NPZ}" == "true" ]]; then
    EXTRA_ARGS+=(--save-per-frame-npz)
  fi
  if [[ "${SAVE_LOCAL_PLY}" == "true" ]]; then
    EXTRA_ARGS+=(--save-local-ply)
  fi
  if [[ "${SAVE_WORLD_PLY}" == "true" ]]; then
    EXTRA_ARGS+=(--save-world-ply)
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python eval_reconstruction_in_ddad.py \
    --model-config "${MODEL_CONFIG}" \
    --ckpt-path "${CKPT_PATH}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --camera "${CAMERA}" \
    --num-frames "${NUM_FRAMES}" \
    --query-chunk-size "${QUERY_CHUNK_SIZE}" \
    --max-lidar-queries-per-frame "${MAX_LIDAR_QUERIES_PER_FRAME}" \
    --depth-vis-grid "${DEPTH_VIS_GRID}" \
    --depth-vis-max-m "${DEPTH_VIS_MAX_M}" \
    --error-vis-max-m "${ERROR_VIS_MAX_M}" \
    --vis-fps "${VIS_FPS}" \
    --limit-scenes "${LIMIT_SCENES}" \
    --scene-shard-index "${SHARD_INDEX}" \
    --scene-shard-count "${SHARD_COUNT}" \
    --device cuda \
    "${EXTRA_ARGS[@]}" \
    > "${OUTPUT_DIR}/logs/shard_${SHARD_INDEX}.log" 2>&1 &
done

wait
python eval_reconstruction_in_ddad.py \
  --output-dir "${OUTPUT_DIR}" \
  --merge-shards-only
echo "DDAD forward reconstruction shards finished. Output: ${OUTPUT_DIR}"
