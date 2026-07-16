#!/usr/bin/env python3
"""Forward-only DDAD reconstruction evaluation for OpenD4RT."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from eval_track3d_in_worldtrack import _unwrap_state_dict
from infer_track_3d import _resize_video, _resolve_device
from src.core import build_logger, load_checkpoint, load_yaml_config, seed_everything
from src.eval.tasks import _encode_model_memory, _run_model_for_queries
from src.model import build_model
from vis.build_like_demo import _export_video_from_frames, _sample_rgb_from_uv_sequence


@dataclass
class DdadDatum:
    key: str
    name: str
    filename: str
    timestamp: str
    sensor_to_world: np.ndarray
    width: int | None = None
    height: int | None = None


@dataclass
class DdadSample:
    index: int
    calibration_key: str
    datums: dict[str, DdadDatum]


@dataclass
class DdadCalibration:
    names: list[str]
    intrinsics: dict[str, np.ndarray]
    sensor_to_rig: dict[str, np.ndarray]


@dataclass
class DdadScene:
    scene_id: str
    scene_dir: Path
    split: str
    samples: list[DdadSample]
    calibrations: dict[str, DdadCalibration]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenD4RT reconstruction on DDAD LiDAR-projected points.")
    parser.add_argument("--model-config", default="checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml")
    parser.add_argument("--ckpt-path", default="checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt")
    parser.add_argument("--data-root", default="/data/jhc/ddad_train_val")
    parser.add_argument("--output-dir", default="output/ddad_forward_reconstruction")
    parser.add_argument("--split", default="all", choices=("all", "train", "val", "ddad_train", "ddad_val"))
    parser.add_argument("--camera", default="CAMERA_01")
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--max-lidar-queries-per-frame", type=int, default=2048)
    parser.add_argument("--depth-vis-grid", type=int, default=64)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--scene-shard-index", type=int, default=0)
    parser.add_argument("--scene-shard-count", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--save-per-frame-npz", action="store_true")
    parser.add_argument("--save-local-ply", action="store_true")
    parser.add_argument("--save-world-ply", action="store_true")
    parser.add_argument("--depth-vis-max-m", type=float, default=80.0)
    parser.add_argument("--error-vis-max-m", type=float, default=10.0)
    parser.add_argument("--vis-fps", type=float, default=6.0)
    parser.add_argument("--merge-shards-only", action="store_true", help="Merge per-shard JSONL files and exit.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _quat_to_rot(q: dict[str, Any]) -> np.ndarray:
    qw = float(q["qw"])
    qx = float(q["qx"])
    qy = float(q["qy"])
    qz = float(q["qz"])
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _pose_to_mat(pose: dict[str, Any]) -> np.ndarray:
    rot = _quat_to_rot(pose["rotation"])
    tr = pose["translation"]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = [float(tr["x"]), float(tr["y"]), float(tr["z"])]
    return out


def _load_calibration(path: Path) -> DdadCalibration:
    raw = json.loads(path.read_text(encoding="utf-8"))
    names = [str(item) for item in raw["names"]]
    intrinsics: dict[str, np.ndarray] = {}
    sensor_to_rig: dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        intr = raw["intrinsics"][idx]
        intrinsics[name] = np.asarray(
            [
                [float(intr.get("fx", 0.0)), float(intr.get("skew", 0.0)), float(intr.get("cx", 0.0))],
                [0.0, float(intr.get("fy", 0.0)), float(intr.get("cy", 0.0))],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        sensor_to_rig[name] = _pose_to_mat(raw["extrinsics"][idx])
    return DdadCalibration(names=names, intrinsics=intrinsics, sensor_to_rig=sensor_to_rig)


def _datum_pose(raw_datum: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "image" in raw_datum:
        return "image", raw_datum["image"]
    if "point_cloud" in raw_datum:
        return "point_cloud", raw_datum["point_cloud"]
    raise ValueError(f"Unsupported DDAD datum payload keys: {sorted(raw_datum.keys())}")


def _load_scene(scene_dir: Path) -> DdadScene:
    scene_files = sorted(scene_dir.glob("scene_*.json"))
    if not scene_files:
        raise FileNotFoundError(f"No scene_*.json under {scene_dir}")
    raw = json.loads(scene_files[0].read_text(encoding="utf-8"))

    datum_by_key: dict[str, DdadDatum] = {}
    for item in raw.get("data", []):
        datum_kind, payload = _datum_pose(item["datum"])
        datum_id = item["id"]
        name = str(datum_id["name"])
        datum_by_key[str(item["key"])] = DdadDatum(
            key=str(item["key"]),
            name=name,
            filename=str(payload["filename"]),
            timestamp=str(datum_id.get("timestamp", "")),
            sensor_to_world=_pose_to_mat(payload["pose"]),
            width=int(payload["width"]) if datum_kind == "image" and "width" in payload else None,
            height=int(payload["height"]) if datum_kind == "image" and "height" in payload else None,
        )

    samples: list[DdadSample] = []
    for sample_raw in raw.get("samples", []):
        datums: dict[str, DdadDatum] = {}
        for key in sample_raw.get("datum_keys", []):
            datum = datum_by_key.get(str(key))
            if datum is not None:
                datums[datum.name] = datum
        samples.append(
            DdadSample(
                index=int(sample_raw.get("id", {}).get("index", len(samples))),
                calibration_key=str(sample_raw["calibration_key"]),
                datums=datums,
            )
        )
    samples.sort(key=lambda item: item.index)

    calibrations: dict[str, DdadCalibration] = {}
    for p in sorted((scene_dir / "calibration").glob("*.json")):
        calibrations[p.stem] = _load_calibration(p)
    split = str(raw.get("description", ""))
    return DdadScene(scene_id=scene_dir.name, scene_dir=scene_dir, split=split, samples=samples, calibrations=calibrations)


def _normalize_split(split: str) -> str:
    split = str(split).strip().lower()
    if split in ("", "all"):
        return "all"
    if split in ("train", "ddad_train"):
        return "ddad_train"
    if split in ("val", "valid", "validation", "ddad_val"):
        return "ddad_val"
    raise ValueError(f"Unsupported DDAD split: {split}")


def _scene_split(scene_dir: Path) -> str:
    scene_files = sorted(scene_dir.glob("scene_*.json"))
    if not scene_files:
        return ""
    raw = json.loads(scene_files[0].read_text(encoding="utf-8"))
    return str(raw.get("description", ""))


def _list_scene_dirs(data_root: Path, shard_index: int, shard_count: int, limit: int, split: str) -> list[Path]:
    scene_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    wanted_split = _normalize_split(split)
    if wanted_split != "all":
        scene_dirs = [p for p in scene_dirs if _scene_split(p) == wanted_split]
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index) % shard_count
    scene_dirs = [p for idx, p in enumerate(scene_dirs) if idx % shard_count == shard_index]
    if int(limit) > 0:
        scene_dirs = scene_dirs[: int(limit)]
    return scene_dirs


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _transform_points(t_dst_src: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    hom = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    out = (np.asarray(t_dst_src, dtype=np.float64) @ hom.T).T
    return out[:, :3]


def _project_lidar_to_camera(
    *,
    scene: DdadScene,
    sample: DdadSample,
    camera_name: str,
    model_hw: tuple[int, int],
    max_queries: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    camera = sample.datums.get(camera_name)
    lidar = sample.datums.get("LIDAR")
    if camera is None or lidar is None:
        return None
    calib = scene.calibrations.get(sample.calibration_key)
    if calib is None or camera_name not in calib.intrinsics:
        return None

    lidar_path = scene.scene_dir / lidar.filename
    if not lidar_path.exists():
        return None
    points = np.load(lidar_path)["data"].astype(np.float64, copy=False)
    if points.ndim != 2 or points.shape[1] < 3:
        return None

    t_cam_world = np.linalg.inv(camera.sensor_to_world)
    points_cam = _transform_points(t_cam_world @ lidar.sensor_to_world, points[:, :3])
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    points_cam = points_cam[valid]
    if points_cam.shape[0] <= 0:
        return None

    src_h = int(camera.height or 1216)
    src_w = int(camera.width or 1936)
    model_h, model_w = model_hw
    k = calib.intrinsics[camera_name].copy()
    k[0, 0] *= float(model_w) / float(max(src_w, 1))
    k[0, 2] *= float(model_w) / float(max(src_w, 1))
    k[1, 1] *= float(model_h) / float(max(src_h, 1))
    k[1, 2] *= float(model_h) / float(max(src_h, 1))

    proj = (k @ points_cam.T).T
    u = proj[:, 0] / np.maximum(points_cam[:, 2], 1e-8)
    v = proj[:, 1] / np.maximum(points_cam[:, 2], 1e-8)
    in_img = np.isfinite(u) & np.isfinite(v) & (u >= 0.0) & (u <= model_w - 1) & (v >= 0.0) & (v <= model_h - 1)
    if not np.any(in_img):
        return None

    u = u[in_img]
    v = v[in_img]
    points_cam = points_cam[in_img]

    # z-buffer by rounded model pixel so duplicate LiDAR projections use the nearest point.
    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    linear = py * int(model_w) + px
    order = np.lexsort((points_cam[:, 2], linear))
    linear_sorted = linear[order]
    keep_sorted = np.ones((order.shape[0],), dtype=bool)
    keep_sorted[1:] = linear_sorted[1:] != linear_sorted[:-1]
    keep = order[keep_sorted]

    if int(max_queries) > 0 and keep.shape[0] > int(max_queries):
        keep = np.sort(rng.choice(keep, size=int(max_queries), replace=False))

    uv_norm = np.stack(
        [
            u[keep] / float(max(model_w - 1, 1)),
            v[keep] / float(max(model_h - 1, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "uv_norm": np.clip(uv_norm, 0.0, 1.0),
        "xyz_cam": points_cam[keep].astype(np.float32),
        "depth": points_cam[keep, 2].astype(np.float32),
        "K_model": k.astype(np.float32),
        "T_wc": camera.sensor_to_world.astype(np.float32),
        "src_hw": np.asarray([src_h, src_w], dtype=np.int32),
    }


def _make_query(uv_norm: np.ndarray, frame_idx: int, device: torch.device) -> dict[str, torch.Tensor]:
    n = int(uv_norm.shape[0])
    t = np.full((n,), int(frame_idx), dtype=np.int64)
    return {
        "u": torch.from_numpy(uv_norm[:, 0]).to(device=device, dtype=torch.float32),
        "v": torch.from_numpy(uv_norm[:, 1]).to(device=device, dtype=torch.float32),
        "t_src": torch.from_numpy(t).to(device=device, dtype=torch.long),
        "t_tgt": torch.from_numpy(t).to(device=device, dtype=torch.long),
        "t_cam": torch.from_numpy(t).to(device=device, dtype=torch.long),
    }


def _make_query_with_t_cam(uv_norm: np.ndarray, frame_idx: int, t_cam_idx: int, device: torch.device) -> dict[str, torch.Tensor]:
    n = int(uv_norm.shape[0])
    t = np.full((n,), int(frame_idx), dtype=np.int64)
    t_cam = np.full((n,), int(t_cam_idx), dtype=np.int64)
    return {
        "u": torch.from_numpy(uv_norm[:, 0]).to(device=device, dtype=torch.float32),
        "v": torch.from_numpy(uv_norm[:, 1]).to(device=device, dtype=torch.float32),
        "t_src": torch.from_numpy(t).to(device=device, dtype=torch.long),
        "t_tgt": torch.from_numpy(t).to(device=device, dtype=torch.long),
        "t_cam": torch.from_numpy(t_cam).to(device=device, dtype=torch.long),
    }


def _local_metric_payload(pred: np.ndarray, gt: np.ndarray, *, scale_global: float) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1) & (gt[:, 2] > 1e-6)
    if not np.any(valid):
        return {
            "valid_queries": 0,
            "local_xyz_epe_global_m": float("nan"),
            "local_depth_abs_rel_global": float("nan"),
            "scale_global": float("nan"),
        }
    pred = pred[valid]
    gt = gt[valid]
    pred_global = pred * float(scale_global)
    global_dist = np.linalg.norm(pred_global - gt, axis=1)
    gt_z = gt[:, 2]
    global_abs_rel = float(np.mean(np.abs(pred_global[:, 2] - gt_z) / np.maximum(gt_z, 1e-6)))
    return {
        "valid_queries": int(pred.shape[0]),
        "local_xyz_epe_global_m": float(np.mean(global_dist)),
        "local_depth_abs_rel_global": global_abs_rel,
        "scale_global": float(scale_global),
    }


def _ref0_metric_payload(pred: np.ndarray, gt: np.ndarray, *, scale_global: float) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if not np.any(valid):
        return {"valid_ref0_queries": 0, "ref0_xyz_epe_global_m": float("nan")}
    pred_global = pred[valid] * float(scale_global)
    epe = np.linalg.norm(pred_global - gt[valid], axis=1)
    return {
        "valid_ref0_queries": int(np.count_nonzero(valid)),
        "ref0_xyz_epe_global_m": float(np.mean(epe)),
    }


def _compute_scale_factor_global(gt_points: np.ndarray, pred_points: np.ndarray) -> float:
    gt_flat = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    pred_flat = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(gt_flat).all(axis=-1) & np.isfinite(pred_flat).all(axis=-1)
    if not np.any(finite):
        return 1.0
    gt_norm = np.linalg.norm(gt_flat[finite], axis=-1)
    pred_norm = np.linalg.norm(pred_flat[finite], axis=-1)
    eps = 1e-12
    if gt_norm.size <= 0 or pred_norm.size <= 0:
        return 1.0
    gt_norm = np.maximum(gt_norm, eps)
    pred_norm = np.maximum(pred_norm, eps)
    return float(np.median(gt_norm) / max(float(np.median(pred_norm)), eps))


def _weighted_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_scenes": int(len(items))}
    total_queries = int(sum(_query_weight(item) for item in items))
    summary["total_queries"] = total_queries
    keys = [
        "local_xyz_epe_global_m",
        "local_depth_abs_rel_global",
        "ref0_xyz_epe_global_m",
        "scale_global",
    ]
    for key in keys:
        vals = []
        weights = []
        for item in items:
            value = float(item.get(key, float("nan")))
            weight = (
                int(item.get("valid_ref0_queries", 0))
                if key == "ref0_xyz_epe_global_m"
                else _query_weight(item)
            )
            if np.isfinite(value) and weight > 0:
                vals.append(value)
                weights.append(weight)
        summary[key] = float(np.average(vals, weights=weights)) if vals else float("nan")
    return summary


def _query_weight(item: dict[str, Any]) -> int:
    if "valid_queries" in item:
        return int(item.get("valid_queries", 0))
    return int(item.get("total_queries", 0))


def _merge_shard_outputs(output_dir: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("per_scene_metrics_shard*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
    split_counts: dict[str, int] = {}
    for item in results:
        split = str(item.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
    summary = {
        "summary": _weighted_summary(results),
        "num_scene_records": int(len(results)),
        "split_counts": split_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _colorize_depth_map_project_style(depth_hw: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    depth = np.asarray(depth_hw, dtype=np.float32)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return out
    norm = np.clip((depth - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap(np.rint(norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[valid] = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)[valid]
    return out


def _make_normalized_uv_grid(grid_size: int) -> tuple[np.ndarray, int, int]:
    size = max(2, int(grid_size))
    coords = np.linspace(0.0, 1.0, num=size, dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords, indexing="xy"), axis=-1).reshape(-1, 2)
    return grid.astype(np.float32), size, size


def _normalized_grid_to_model_pixels(uv_norm: np.ndarray, model_hw: tuple[int, int]) -> np.ndarray:
    h, w = model_hw
    uv = np.asarray(uv_norm, dtype=np.float32)
    out = np.empty_like(uv, dtype=np.float32)
    out[:, 0] = uv[:, 0] * float(max(w - 1, 1))
    out[:, 1] = uv[:, 1] * float(max(h - 1, 1))
    return out


def _infer_dense_points(
    *,
    model: torch.nn.Module,
    video_tensor: torch.Tensor,
    aspect_tensor: torch.Tensor,
    memory: torch.Tensor | None,
    uv_grid: np.ndarray,
    query_chunk_size: int,
    t_cam_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    num_frames = int(video_tensor.shape[1])
    num_points = int(uv_grid.shape[0])
    repeated_uv = np.tile(np.asarray(uv_grid, dtype=np.float32), (num_frames, 1))
    t = np.repeat(np.arange(num_frames, dtype=np.int64), num_points)
    if str(t_cam_mode) == "local":
        t_cam = t.copy()
    elif str(t_cam_mode) == "ref0":
        t_cam = np.zeros_like(t)
    else:
        raise ValueError(f"Unsupported t_cam_mode={t_cam_mode!r}")
    query = {
        "u": torch.from_numpy(repeated_uv[:, 0]).to(device=device, dtype=torch.float32),
        "v": torch.from_numpy(repeated_uv[:, 1]).to(device=device, dtype=torch.float32),
        "t_src": torch.from_numpy(t).to(device=device, dtype=torch.long),
        "t_tgt": torch.from_numpy(t).to(device=device, dtype=torch.long),
        "t_cam": torch.from_numpy(t_cam).to(device=device, dtype=torch.long),
    }
    pred = _run_model_for_queries(
        model=model,
        video_b=video_tensor,
        aspect_b=aspect_tensor,
        query=query,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    xyz = pred["xyz_3d"].numpy().astype(np.float32).reshape(num_frames, num_points, 3)
    visibility = np.isfinite(xyz).all(axis=-1)
    return xyz, visibility


def _error_color_rgb(error_m: np.ndarray, error_max_m: float) -> np.ndarray:
    err = np.asarray(error_m, dtype=np.float32)
    norm = np.clip(err / max(float(error_max_m), 1e-6), 0.0, 1.0)
    colors = np.zeros((err.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = np.rint(norm * 255.0).astype(np.uint8)
    colors[:, 1] = np.rint((1.0 - norm) * 255.0).astype(np.uint8)
    return colors


def _draw_sparse_error_overlay(
    *,
    rgb: np.ndarray,
    depth_pred_hw: np.ndarray,
    payload: dict[str, np.ndarray] | None,
    model_hw: tuple[int, int],
    error_max_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h, w = model_hw
    canvas = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).copy()
    empty = {
        "uv_norm": np.zeros((0, 2), dtype=np.float32),
        "gt_depth": np.zeros((0,), dtype=np.float32),
        "pred_depth": np.zeros((0,), dtype=np.float32),
        "abs_error": np.zeros((0,), dtype=np.float32),
    }
    if payload is None or payload.get("uv_norm", np.empty((0, 2))).shape[0] <= 0:
        return canvas, empty

    uv = np.asarray(payload["uv_norm"], dtype=np.float32)
    gt_depth = np.asarray(payload["xyz_cam"], dtype=np.float32)[:, 2]
    u = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    v = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    inside = (
        np.isfinite(uv).all(axis=-1)
        & np.isfinite(gt_depth)
        & (gt_depth > 0.0)
        & (gt_depth <= float(depth_max_m))
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )
    if not np.any(inside):
        return canvas, empty

    pred_depth = np.asarray(depth_pred_hw, dtype=np.float32)[v[inside], u[inside]]
    gt_valid = gt_depth[inside]
    uv_valid = uv[inside]
    finite = np.isfinite(pred_depth) & (pred_depth > 0.0)
    if not np.any(finite):
        return canvas, empty

    pred_valid = pred_depth[finite]
    gt_valid = gt_valid[finite]
    uv_valid = uv_valid[finite]
    u_valid = u[inside][finite]
    v_valid = v[inside][finite]
    abs_error = np.abs(pred_valid - gt_valid).astype(np.float32)
    colors = _error_color_rgb(abs_error, error_max_m=float(error_max_m))
    for uu, vv, color in zip(u_valid, v_valid, colors, strict=False):
        cv2.circle(canvas, (int(uu), int(vv)), 2, tuple(int(c) for c in color.tolist()), thickness=-1)

    payload_out = {
        "uv_norm": uv_valid.astype(np.float32),
        "gt_depth": gt_valid.astype(np.float32),
        "pred_depth": pred_valid.astype(np.float32),
        "abs_error": abs_error.astype(np.float32),
    }
    return canvas, payload_out


def _write_triplet_video(
    *,
    path: Path,
    video_model_rgb: np.ndarray,
    dense_xyz_cam: np.ndarray,
    dense_visibility: np.ndarray,
    sparse_payloads_by_frame: list[dict[str, np.ndarray] | None] | None,
    model_hw: tuple[int, int],
    grid_size: int,
    scale_global: float,
    error_max_m: float,
    depth_max_m: float,
    fps: float,
) -> None:
    num_frames = int(video_model_rgb.shape[0])
    h, w = model_hw
    xyz_cam = np.asarray(dense_xyz_cam, dtype=np.float32)
    point_vis = np.asarray(dense_visibility, dtype=bool)
    _, grid_rows, grid_cols = _make_normalized_uv_grid(int(grid_size))
    if xyz_cam.ndim != 3 or xyz_cam.shape[0] != num_frames:
        raise ValueError(f"dense_xyz_cam must have shape [T,N,3], got {xyz_cam.shape}")
    if int(grid_rows) * int(grid_cols) != int(xyz_cam.shape[1]):
        raise ValueError(
            f"depth grid {grid_rows}x{grid_cols} does not match point count {int(xyz_cam.shape[1])}"
        )

    depth_raw = np.full((num_frames, h, w), np.nan, dtype=np.float32)
    for frame_idx in range(num_frames):
        pts = xyz_cam[frame_idx]
        vis = point_vis[frame_idx] & np.isfinite(pts).all(axis=-1)
        z_all = pts[:, 2] * float(scale_global)
        valid_z = vis & np.isfinite(z_all) & (z_all > 1e-6)
        depth_grid = np.where(valid_z, z_all, np.nan).astype(np.float32)
        depth_grid_hw = depth_grid.reshape(int(grid_rows), int(grid_cols))
        mask = np.isfinite(depth_grid_hw).astype(np.float32)
        depth_up = cv2.resize(depth_grid_hw, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_up = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_raw[frame_idx] = np.where(mask_up > 1e-3, depth_up, np.nan).astype(np.float32)

    valid = np.isfinite(depth_raw) & (depth_raw > 0.0)
    if not np.any(valid):
        raise RuntimeError("Dense depth grid produced no valid positive depth values.")
    vals = depth_raw[valid]
    vmin = float(np.nanpercentile(vals, 5.0))
    vmax = float(np.nanpercentile(vals, 95.0))
    depth_rgb = np.stack([_colorize_depth_map_project_style(depth_raw[t], vmin=vmin, vmax=vmax) for t in range(num_frames)], axis=0)
    rgb_small = np.stack([cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in video_model_rgb], axis=0)
    sparse_payloads = sparse_payloads_by_frame or [None] * num_frames
    sparse_payloads = list(sparse_payloads)[:num_frames]
    if len(sparse_payloads) < num_frames:
        sparse_payloads.extend([None] * (num_frames - len(sparse_payloads)))
    error_frames = []
    compare_rows = []
    for frame_idx in range(num_frames):
        overlay, compare = _draw_sparse_error_overlay(
            rgb=video_model_rgb[frame_idx],
            depth_pred_hw=depth_raw[frame_idx],
            payload=sparse_payloads[frame_idx],
            model_hw=model_hw,
            error_max_m=float(error_max_m),
            depth_max_m=float(depth_max_m),
        )
        error_frames.append(overlay)
        if compare["abs_error"].shape[0] > 0:
            frame_ids = np.full((compare["abs_error"].shape[0], 1), int(frame_idx), dtype=np.float32)
            compare_rows.append(
                np.concatenate(
                    [
                        frame_ids,
                        compare["uv_norm"].astype(np.float32),
                        compare["gt_depth"][:, None].astype(np.float32),
                        compare["pred_depth"][:, None].astype(np.float32),
                        compare["abs_error"][:, None].astype(np.float32),
                    ],
                    axis=1,
                )
            )
    error_rgb = np.stack(error_frames, axis=0)
    triplet = np.concatenate([rgb_small, depth_rgb, error_rgb], axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    _export_video_from_frames(video_rgb=triplet, fps=float(fps), dst_video=path)
    compare_array = (
        np.concatenate(compare_rows, axis=0)
        if compare_rows
        else np.zeros((0, 6), dtype=np.float32)
    )
    np.savez_compressed(
        path.with_name(path.stem + "_raw.npz"),
        depth=depth_raw,
        vmin=np.asarray([vmin], dtype=np.float32),
        vmax=np.asarray([vmax], dtype=np.float32),
        sparse_compare=compare_array.astype(np.float32),
        sparse_compare_columns=np.asarray(["frame_idx", "u_norm", "v_norm", "gt_depth", "pred_depth", "abs_error"]),
    )


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    cols = None
    if rgb is not None:
        cols = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)[valid]
    if cols is None:
        cols = np.full((pts.shape[0], 3), 255, dtype=np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii") as handle:
        handle.write(header)
        for point, color in zip(pts, cols, strict=False):
            handle.write(
                f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _evaluate_scene(
    *,
    scene_dir: Path,
    model: torch.nn.Module,
    model_hw: tuple[int, int],
    args: argparse.Namespace,
    output_dir: Path,
    logger: Any,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    scene = _load_scene(scene_dir)
    valid_samples = [
        sample
        for sample in scene.samples
        if args.camera in sample.datums and "LIDAR" in sample.datums and sample.calibration_key in scene.calibrations
    ]
    if len(valid_samples) < int(args.num_frames):
        logger.warning("Skip scene=%s because only %d valid samples", scene.scene_id, len(valid_samples))
        return None
    valid_samples = valid_samples[: int(args.num_frames)]

    rgb_frames = [_load_rgb(scene.scene_dir / sample.datums[args.camera].filename) for sample in valid_samples]
    video_model_rgb = _resize_video(np.stack(rgb_frames, axis=0), image_hw=model_hw)
    device = next(model.parameters()).device
    video_tensor = (
        torch.from_numpy(video_model_rgb)
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        / 255.0
    )
    aspect = np.asarray(
        [[float(rgb_frames[0].shape[1]) / float(max(1, rgb_frames[0].shape[0]))]],
        dtype=np.float32,
    )
    aspect_tensor = torch.from_numpy(aspect).to(device=device, dtype=torch.float32)

    frame_metrics: list[dict[str, Any]] = []
    local_frame_indices: list[int] = []
    local_points: list[np.ndarray] = []
    local_gt_points: list[np.ndarray] = []
    local_colors: list[np.ndarray] = []
    sparse_payloads_by_frame: list[dict[str, np.ndarray] | None] = [None] * int(len(valid_samples))
    sparse_pred_ref0_points: list[np.ndarray] = []
    sparse_gt_ref0_points: list[np.ndarray] = []
    world_points: list[np.ndarray] = []
    world_colors: list[np.ndarray] = []

    memory = _encode_model_memory(model=model, video_b=video_tensor, aspect_b=aspect_tensor)
    t_ref0_world = np.linalg.inv(valid_samples[0].datums[str(args.camera)].sensor_to_world)
    for frame_idx, sample in enumerate(valid_samples):
        payload = _project_lidar_to_camera(
            scene=scene,
            sample=sample,
            camera_name=str(args.camera),
            model_hw=model_hw,
            max_queries=int(args.max_lidar_queries_per_frame),
            rng=rng,
        )
        if payload is None or payload["uv_norm"].shape[0] <= 0:
            continue
        sparse_payloads_by_frame[frame_idx] = payload
        query = _make_query(payload["uv_norm"], frame_idx=frame_idx, device=device)
        pred = _run_model_for_queries(
            model=model,
            video_b=video_tensor,
            aspect_b=aspect_tensor,
            query=query,
            chunk_size=int(args.query_chunk_size),
            memory_b=memory,
        )
        pred_xyz = pred["xyz_3d"].numpy().astype(np.float32)
        local_frame_indices.append(int(frame_idx))
        local_points.append(pred_xyz)
        local_gt_points.append(payload["xyz_cam"].astype(np.float32))
        rgb_small = cv2.resize(rgb_frames[frame_idx], (model_hw[1], model_hw[0]), interpolation=cv2.INTER_AREA)
        u_px = np.rint(payload["uv_norm"][:, 0] * float(max(model_hw[1] - 1, 1))).astype(np.int64)
        v_px = np.rint(payload["uv_norm"][:, 1] * float(max(model_hw[0] - 1, 1))).astype(np.int64)
        colors = rgb_small[np.clip(v_px, 0, model_hw[0] - 1), np.clip(u_px, 0, model_hw[1] - 1)]
        local_colors.append(colors)
        gt_mask = np.isfinite(payload["xyz_cam"]).all(axis=-1) & (payload["xyz_cam"][:, 2] > 0.0)
        if np.any(gt_mask):
            t_ref0_cam_gt = t_ref0_world @ payload["T_wc"].astype(np.float64)
            gt_h = np.concatenate(
                [payload["xyz_cam"][gt_mask].astype(np.float64), np.ones((int(np.count_nonzero(gt_mask)), 1), dtype=np.float64)],
                axis=1,
            )
            query_ref0 = _make_query_with_t_cam(payload["uv_norm"], frame_idx=frame_idx, t_cam_idx=0, device=device)
            pred_ref0_out = _run_model_for_queries(
                model=model,
                video_b=video_tensor,
                aspect_b=aspect_tensor,
                query=query_ref0,
                chunk_size=int(args.query_chunk_size),
                memory_b=memory,
            )
            pred_ref0 = pred_ref0_out["xyz_3d"].numpy().astype(np.float32)[gt_mask]
            gt_ref0 = (t_ref0_cam_gt @ gt_h.T).T[:, :3].astype(np.float32)
            sparse_pred_ref0_points.append(pred_ref0)
            sparse_gt_ref0_points.append(gt_ref0)
        if bool(args.save_world_ply):
            pred_h = np.concatenate([pred_xyz.astype(np.float64), np.ones((pred_xyz.shape[0], 1), dtype=np.float64)], axis=1)
            world = (payload["T_wc"].astype(np.float64) @ pred_h.T).T[:, :3].astype(np.float32)
            world_points.append(world)
            world_colors.append(colors)

    if not local_points:
        logger.warning("Skip scene=%s because no valid projected LiDAR queries survived", scene.scene_id)
        return None

    local_pred_all = np.concatenate(local_points, axis=0)
    local_gt_all = np.concatenate(local_gt_points, axis=0)
    scale_global = _compute_scale_factor_global(local_gt_all, local_pred_all)
    scene_summary = _local_metric_payload(local_pred_all, local_gt_all, scale_global=scale_global)
    if sparse_pred_ref0_points and sparse_gt_ref0_points:
        scene_summary.update(
            _ref0_metric_payload(
                np.concatenate(sparse_pred_ref0_points, axis=0),
                np.concatenate(sparse_gt_ref0_points, axis=0),
                scale_global=scale_global,
            )
        )
    for frame_idx, pred_frame, gt_frame in zip(local_frame_indices, local_points, local_gt_points, strict=False):
        metrics = _local_metric_payload(pred_frame, gt_frame, scale_global=scale_global)
        metrics.update({"frame_idx": int(frame_idx)})
        frame_metrics.append(metrics)
    scene_summary.update(
        {
            "scene_id": scene.scene_id,
            "split": scene.split,
            "camera": str(args.camera),
            "num_frames": int(len(valid_samples)),
            "model_image_size": [int(model_hw[0]), int(model_hw[1])],
            "frame_metrics": frame_metrics,
        }
    )

    vis_dir = output_dir / "vis"
    dense_xyz_cam = None
    dense_visibility = None
    dense_xyz_ref0_model = None
    dense_ref0_visibility = None
    point_rgb = None
    needs_dense_ref0 = bool(args.save_visualizations) or bool(args.save_local_ply)
    if needs_dense_ref0:
        try:
            uv_grid, _, _ = _make_normalized_uv_grid(int(args.depth_vis_grid))
            dense_xyz_cam, dense_visibility = _infer_dense_points(
                model=model,
                video_tensor=video_tensor,
                aspect_tensor=aspect_tensor,
                memory=memory,
                uv_grid=uv_grid,
                query_chunk_size=int(args.query_chunk_size),
                t_cam_mode="local",
            )
            point_uv_px = _normalized_grid_to_model_pixels(uv_grid, model_hw)
            point_uv_seq = np.tile(point_uv_px[None, :, :], (int(video_model_rgb.shape[0]), 1, 1)).astype(np.float32)
            point_rgb = _sample_rgb_from_uv_sequence(video_model_rgb, point_uv_seq)
            if bool(args.save_local_ply):
                dense_xyz_ref0_model, dense_ref0_visibility = _infer_dense_points(
                    model=model,
                    video_tensor=video_tensor,
                    aspect_tensor=aspect_tensor,
                    memory=memory,
                    uv_grid=uv_grid,
                    query_chunk_size=int(args.query_chunk_size),
                    t_cam_mode="ref0",
                )
        except Exception as exc:
            logger.warning("Failed to infer dense visualization point cloud for scene=%s: %s", scene.scene_id, exc)

    if bool(args.save_visualizations) and dense_xyz_cam is not None and dense_visibility is not None:
        try:
            _write_triplet_video(
                path=vis_dir / f"{scene.scene_id}_{args.camera}_triplet.mp4",
                video_model_rgb=video_model_rgb,
                dense_xyz_cam=dense_xyz_cam,
                dense_visibility=dense_visibility,
                sparse_payloads_by_frame=sparse_payloads_by_frame,
                model_hw=model_hw,
                grid_size=int(args.depth_vis_grid),
                scale_global=float(scene_summary.get("scale_global", 1.0)),
                error_max_m=float(args.error_vis_max_m),
                depth_max_m=float(args.depth_vis_max_m),
                fps=float(args.vis_fps),
            )
        except Exception as exc:
            logger.warning("Failed to write triplet video for scene=%s: %s", scene.scene_id, exc)
    if bool(args.save_per_frame_npz):
        try:
            vis_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                vis_dir / f"{scene.scene_id}_{args.camera}_local_pointcloud.npz",
                pred_xyz=np.asarray(local_points, dtype=object),
                rgb=np.asarray(local_colors, dtype=object),
            )
        except Exception as exc:
            logger.warning("Failed to write local npz for scene=%s: %s", scene.scene_id, exc)
    if bool(args.save_local_ply) and dense_xyz_ref0_model is not None and dense_ref0_visibility is not None and point_rgb is not None:
        try:
            scale = float(scene_summary.get("scale_global", 1.0))
            dense_scaled = np.asarray(dense_xyz_ref0_model, dtype=np.float32) * scale
            depth = dense_scaled[..., 2]
            valid = (
                np.asarray(dense_ref0_visibility, dtype=bool)
                & np.isfinite(dense_scaled).all(axis=-1)
                & np.isfinite(depth)
                & (depth > 0.0)
                & (depth <= float(args.depth_vis_max_m))
            )
            _write_ply(
                vis_dir / f"{scene.scene_id}_{args.camera}_pred_dense_ref0.ply",
                xyz=dense_scaled[valid],
                rgb=np.asarray(point_rgb, dtype=np.uint8)[valid],
            )
        except Exception as exc:
            logger.warning("Failed to write pred dense ref0 ply for scene=%s: %s", scene.scene_id, exc)
    if bool(args.save_local_ply) and sparse_gt_ref0_points and sparse_pred_ref0_points:
        try:
            scale = float(scene_summary.get("scale_global", 1.0))
            pred_sparse_xyz = np.concatenate(sparse_pred_ref0_points, axis=0) * scale
            gt_sparse_xyz = np.concatenate(sparse_gt_ref0_points, axis=0)
            pred_blue = np.tile(np.asarray([[60, 140, 255]], dtype=np.uint8), (pred_sparse_xyz.shape[0], 1))
            gt_orange = np.tile(np.asarray([[255, 140, 40]], dtype=np.uint8), (gt_sparse_xyz.shape[0], 1))
            _write_ply(
                vis_dir / f"{scene.scene_id}_{args.camera}_pred_gt_sparse_ref0_compare.ply",
                xyz=np.concatenate([pred_sparse_xyz, gt_sparse_xyz], axis=0),
                rgb=np.concatenate([pred_blue, gt_orange], axis=0),
            )
        except Exception as exc:
            logger.warning("Failed to write sparse pred/GT compare ref0 ply for scene=%s: %s", scene.scene_id, exc)
    if bool(args.save_world_ply) and world_points:
        try:
            _write_ply(
                vis_dir / f"{scene.scene_id}_{args.camera}_world_pointcloud_raw.ply",
                xyz=np.concatenate(world_points, axis=0),
                rgb=np.concatenate(world_colors, axis=0),
            )
        except Exception as exc:
            logger.warning("Failed to write world ply for scene=%s: %s", scene.scene_id, exc)
    return scene_summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("eval_reconstruction_in_ddad", output_dir)
    if bool(args.merge_shards_only):
        summary = _merge_shard_outputs(output_dir)
        logger.info("Merged shard outputs into %s (records=%d)", output_dir / "summary.json", int(summary["num_scene_records"]))
        return 0

    seed_everything(int(args.seed), deterministic=True)
    rng = np.random.default_rng(int(args.seed) + int(args.scene_shard_index))

    cfg = load_yaml_config(args.model_config)
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    model_hw = (int(image_size[0]), int(image_size[1]))
    device = _resolve_device(str(args.device))

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_model(cfg["model"]).eval()
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
    load_result = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded checkpoint %s", ckpt_path)
    logger.info("Missing keys: %d  Unexpected keys: %d", len(load_result.missing_keys), len(load_result.unexpected_keys))
    model.to(device).eval()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"DDAD root not found: {data_root}")
    scene_dirs = _list_scene_dirs(
        data_root=data_root,
        shard_index=int(args.scene_shard_index),
        shard_count=int(args.scene_shard_count),
        limit=int(args.limit_scenes),
        split=str(args.split),
    )
    logger.info(
        "Evaluating DDAD split=%s scenes=%d shard=%d/%d camera=%s frames=%d",
        str(args.split),
        len(scene_dirs),
        int(args.scene_shard_index),
        int(args.scene_shard_count),
        args.camera,
        int(args.num_frames),
    )

    per_scene_path = output_dir / f"per_scene_metrics_shard{int(args.scene_shard_index):02d}.jsonl"
    results: list[dict[str, Any]] = []
    with per_scene_path.open("w", encoding="utf-8") as handle:
        for scene_dir in scene_dirs:
            try:
                result = _evaluate_scene(
                    scene_dir=scene_dir,
                    model=model,
                    model_hw=model_hw,
                    args=args,
                    output_dir=output_dir,
                    logger=logger,
                    rng=rng,
                )
            except Exception as exc:
                logger.exception("Failed scene=%s: %s", scene_dir.name, exc)
                result = None
            if result is None:
                continue
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            logger.info(
                "scene=%s queries=%d local_depth_abs_rel=%.4f ref0_xyz_epe=%.4f",
                result["scene_id"],
                int(result.get("valid_queries", result.get("total_queries", 0))),
                float(result.get("local_depth_abs_rel_global", float("nan"))),
                float(result.get("ref0_xyz_epe_global_m", float("nan"))),
            )

    summary = {
        "inputs": {
            "model_config": str(args.model_config),
            "ckpt_path": str(ckpt_path),
            "data_root": str(data_root),
            "camera": str(args.camera),
            "num_frames": int(args.num_frames),
            "query_chunk_size": int(args.query_chunk_size),
            "max_lidar_queries_per_frame": int(args.max_lidar_queries_per_frame),
            "scene_shard_index": int(args.scene_shard_index),
            "scene_shard_count": int(args.scene_shard_count),
        },
        "summary": _weighted_summary(results),
    }
    summary_path = output_dir / f"summary_shard{int(args.scene_shard_index):02d}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
