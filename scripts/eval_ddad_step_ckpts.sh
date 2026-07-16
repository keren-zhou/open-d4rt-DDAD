#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-output/ddad_reconstruction_train/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/ddad_reconstruction_step_eval}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_effective.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/jhc/ddad_train_val}"
SPLIT="${SPLIT:-val}"
CAMERA="${CAMERA:-CAMERA_01}"
NUM_FRAMES="${NUM_FRAMES:-48}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-4096}"
MAX_LIDAR_QUERIES_PER_FRAME="${MAX_LIDAR_QUERIES_PER_FRAME:-2048}"
GPUS="${GPUS:-0,1,2,3}"
MIN_STEP="${MIN_STEP:-0}"
MAX_STEP="${MAX_STEP:-999999999}"
METRIC="${METRIC:-local_depth_abs_rel_global}"

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_ddad_step_ckpts.sh [options]

Options:
  --checkpoint-dir PATH     Directory containing step_*.ckpt. Default: output/ddad_reconstruction_train/checkpoints
  --output-root PATH        Evaluation output root.
  --model-config PATH       Model config. Default: configs/model_effective.yaml
  --data-root PATH          DDAD root. Default: /data/jhc/ddad_train_val
  --split all|train|val     DDAD split. Default: val
  --camera NAME             Camera. Default: CAMERA_01
  --gpus IDS                Comma-separated GPUs. Default: 0,1,2,3
  --min-step N              Minimum step checkpoint to evaluate.
  --max-step N              Maximum step checkpoint to evaluate.
  --metric NAME             Metric used to report the best step. Default: local_depth_abs_rel_global
  -h, --help                Show this help.

This script runs DDAD forward eval for saved step checkpoints and writes an analysis report:
  ${OUTPUT_ROOT}/ddad_step_eval_report.json

It does not overwrite training checkpoints or change checkpoints/best.ckpt.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --camera) CAMERA="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --min-step) MIN_STEP="$2"; shift 2 ;;
    --max-step) MAX_STEP="$2"; shift 2 ;;
    --metric) METRIC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_ROOT"
REPORT_PATH="${OUTPUT_ROOT}/ddad_step_eval_report.json"
MANIFEST_PATH="${OUTPUT_ROOT}/evaluated_checkpoints.tsv"
>"$MANIFEST_PATH"

mapfile -t CKPTS < <(
  find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'step_*.ckpt' | sort
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[ERROR] No step_*.ckpt found under ${CHECKPOINT_DIR}" >&2
  exit 1
fi

for CKPT in "${CKPTS[@]}"; do
  base="$(basename "$CKPT")"
  step="${base#step_}"
  step="${step%.ckpt}"
  step_num=$((10#$step))
  if (( step_num < MIN_STEP || step_num > MAX_STEP )); then
    continue
  fi
  out_dir="${OUTPUT_ROOT}/step_${step}"
  printf '%s\t%s\t%s\n' "$step_num" "$CKPT" "$out_dir" >> "$MANIFEST_PATH"
  echo "[DDAD eval] step=${step_num} ckpt=${CKPT} out=${out_dir}"
  bash scripts/eval_ddad_forward_4gpu.sh \
    --model-config "$MODEL_CONFIG" \
    --ckpt-path "$CKPT" \
    --data-root "$DATA_ROOT" \
    --output-dir "$out_dir" \
    --split "$SPLIT" \
    --camera "$CAMERA" \
    --num-frames "$NUM_FRAMES" \
    --query-chunk-size "$QUERY_CHUNK_SIZE" \
    --max-lidar-queries-per-frame "$MAX_LIDAR_QUERIES_PER_FRAME" \
    --gpus "$GPUS" \
    --no-vis
done

python - "$OUTPUT_ROOT" "$METRIC" "$REPORT_PATH" "$MANIFEST_PATH" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
metric = sys.argv[2]
report_path = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])

ckpt_by_step = {}
if manifest_path.exists():
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                ckpt_by_step[int(parts[0])] = parts[1]
            except ValueError:
                pass

records = []
for summary_path in sorted(root.glob("step_*/summary.json")):
    step_token = summary_path.parent.name.removeprefix("step_")
    try:
        step = int(step_token)
    except ValueError:
        continue
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    value = summary.get(metric)
    record = {
        "step": step,
        "checkpoint": ckpt_by_step.get(step, ""),
        "output_dir": str(summary_path.parent),
        "metric": metric,
        "metric_value": value,
        "summary": summary,
    }
    records.append(record)

finite = [r for r in records if isinstance(r.get("metric_value"), (int, float)) and math.isfinite(float(r["metric_value"]))]
best = min(finite, key=lambda r: float(r["metric_value"])) if finite else None
report = {"metric": metric, "best": best, "records": records}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"metric": metric, "best_step": None if best is None else best["step"], "best_value": None if best is None else best["metric_value"], "report": str(report_path)}, ensure_ascii=False, indent=2))
PY
