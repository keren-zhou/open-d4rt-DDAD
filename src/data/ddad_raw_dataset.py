"""DDAD raw dataset adapter for D4RT reconstruction fine-tuning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bad_sample_registry import BadSampleRegistry, RetryableSampleError, failed_paths_from_exception, is_retryable_data_error
from .ddad_query_builder import build_queries_from_sparse_lidar
from .raw_augment import (
    RawAugmentConfig,
    apply_photometric_augment,
    build_augment_info,
    sample_frame_indices_with_stride,
)
from .seeding import SeededDatasetMixin


@dataclass(frozen=True)
class DdadRawConfig:
    root: Path
    split: str
    camera: str
    clip_frames: int
    image_size: tuple[int, int]
    queries_per_clip: int
    hard_query_ratio: float
    prob_t_tgt_equals_t_cam: float
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None
    t_src_tgt_delta_probs: tuple[float, ...] | None = None
    training: bool = True
    max_scenes: int | None = None
    max_lidar_points_per_frame: int = 20000
    depth_consistency_abs_m: float = 0.5
    depth_consistency_rel: float = 0.05
    depth_consistency_radius_px: float = 1.5
    force_tgt_cam_to_src: bool = False
    min_lidar_points_per_frame: int = 32
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("data/meta/bad_sample.json")
    max_sample_retries: int = 64


@dataclass
class _Datum:
    name: str
    filename: str
    sensor_to_world: np.ndarray
    width: int | None = None
    height: int | None = None


@dataclass
class _Sample:
    index: int
    calibration_key: str
    datums: dict[str, _Datum]


@dataclass
class _Calibration:
    intrinsics: dict[str, np.ndarray]


@dataclass
class _Scene:
    scene_id: str
    scene_dir: Path
    split: str
    samples: list[_Sample]
    calibrations: dict[str, _Calibration]
    src_h: int
    src_w: int


class _FilteredDdadSampleError(RetryableSampleError):
    """Sample-level quality filter; retry without treating the file as corrupt."""


def _quat_to_rot(q: dict[str, Any]) -> np.ndarray:
    qw = float(q["qw"])
    qx = float(q["qx"])
    qy = float(q["qy"])
    qz = float(q["qz"])
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _pose_to_mat(pose: dict[str, Any]) -> np.ndarray:
    rot = _quat_to_rot(pose["rotation"])
    tr = pose["translation"]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rot
    out[:3, 3] = [float(tr["x"]), float(tr["y"]), float(tr["z"])]
    return out


def _datum_payload(raw_datum: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "image" in raw_datum:
        return "image", raw_datum["image"]
    if "point_cloud" in raw_datum:
        return "point_cloud", raw_datum["point_cloud"]
    raise ValueError(f"Unsupported DDAD datum keys: {sorted(raw_datum.keys())}")


def _load_calibration(path: Path) -> _Calibration:
    raw = json.loads(path.read_text(encoding="utf-8"))
    names = [str(item) for item in raw["names"]]
    intrinsics: dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        intr = raw["intrinsics"][idx]
        intrinsics[name] = np.asarray(
            [
                [float(intr.get("fx", 0.0)), float(intr.get("skew", 0.0)), float(intr.get("cx", 0.0))],
                [0.0, float(intr.get("fy", 0.0)), float(intr.get("cy", 0.0))],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return _Calibration(intrinsics=intrinsics)


def _split_name(split: str) -> str:
    token = str(split).strip().lower()
    if token in {"train", "ddad_train"}:
        return "ddad_train"
    if token in {"val", "valid", "validation", "ddad_val", "test"}:
        return "ddad_val"
    if token in {"all", ""}:
        return "all"
    raise ValueError(f"Unsupported DDAD split: {split}")


def _resize_rgb(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = int(image_hw[0]), int(image_hw[1])
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return rgb


def _load_lidar_npz(path: Path) -> np.ndarray:
    payload = np.load(path)
    if "data" in payload:
        points = payload["data"]
    else:
        points = payload[payload.files[0]]
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Unsupported DDAD LiDAR npz shape at {path}: {points.shape}")
    return points[:, :3].astype(np.float32, copy=False)


def _transform_points(t_dst_src: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    hom = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    out = (np.asarray(t_dst_src, dtype=np.float32) @ hom.T).T
    return out[:, :3]


def _zbuffer_points(points_cam: np.ndarray, uv_px: np.ndarray, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = int(image_hw[0]), int(image_hw[1])
    z = points_cam[:, 2]
    valid = (
        np.isfinite(points_cam).all(axis=1)
        & np.isfinite(uv_px).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-6)
        & (uv_px[:, 0] >= 0.0)
        & (uv_px[:, 0] <= (w - 1))
        & (uv_px[:, 1] >= 0.0)
        & (uv_px[:, 1] <= (h - 1))
    )
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    pts = points_cam[valid]
    uv = uv_px[valid]
    u_int = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
    v_int = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
    best: dict[int, int] = {}
    for i, (uu, vv) in enumerate(zip(u_int, v_int, strict=False)):
        key = int(vv) * w + int(uu)
        old = best.get(key)
        if old is None or float(pts[i, 2]) < float(pts[old, 2]):
            best[key] = i
    keep = np.asarray(sorted(best.values()), dtype=np.int64)
    return pts[keep].astype(np.float32, copy=False), uv[keep].astype(np.float32, copy=False)


class DdadRawDataset(SeededDatasetMixin, Dataset):
    """Loads DDAD scene folders and builds sparse LiDAR D4RT supervision."""

    def __init__(self, config: DdadRawConfig) -> None:
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace="ddad_raw", default_seed=20260710)
        self.augment = config.augment or RawAugmentConfig()
        if not config.training:
            self.augment = RawAugmentConfig()
        self.bad_registry = BadSampleRegistry(path=config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))
        self.scenes = self._discover_scenes()
        if not self.scenes:
            raise ValueError(f"No valid DDAD scenes found for split={config.split} under {config.root}")

    def _discover_scenes(self) -> list[_Scene]:
        root = Path(self.cfg.root)
        if not root.exists():
            raise FileNotFoundError(f"DDAD root not found: {root}")
        wanted = _split_name(self.cfg.split)
        scenes: list[_Scene] = []
        for scene_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            scene = self._load_scene(scene_dir)
            if scene is None:
                continue
            if wanted != "all" and scene.split != wanted:
                continue
            scenes.append(scene)
            if self.cfg.max_scenes is not None and len(scenes) >= int(self.cfg.max_scenes):
                break
        return scenes

    def _load_scene(self, scene_dir: Path) -> _Scene | None:
        scene_files = sorted(scene_dir.glob("scene_*.json"))
        if not scene_files:
            return None
        raw = json.loads(scene_files[0].read_text(encoding="utf-8"))
        datum_by_key: dict[str, _Datum] = {}
        for item in raw.get("data", []):
            kind, payload = _datum_payload(item["datum"])
            datum_id = item["id"]
            name = str(datum_id["name"])
            datum_by_key[str(item["key"])] = _Datum(
                name=name,
                filename=str(payload["filename"]),
                sensor_to_world=_pose_to_mat(payload["pose"]),
                width=int(payload["width"]) if kind == "image" and "width" in payload else None,
                height=int(payload["height"]) if kind == "image" and "height" in payload else None,
            )

        samples: list[_Sample] = []
        for seq_idx, sample_raw in enumerate(raw.get("samples", [])):
            datums: dict[str, _Datum] = {}
            for key in sample_raw.get("datum_keys", []):
                datum = datum_by_key.get(str(key))
                if datum is not None:
                    datums[datum.name] = datum
            if self.cfg.camera in datums and "LIDAR" in datums:
                samples.append(
                    _Sample(
                        index=int(seq_idx),
                        calibration_key=str(sample_raw["calibration_key"]),
                        datums=datums,
                    )
                )
        samples.sort(key=lambda item: item.index)
        if len(samples) < int(self.cfg.clip_frames):
            return None

        calibrations: dict[str, _Calibration] = {}
        for p in sorted((scene_dir / "calibration").glob("*.json")):
            calibrations[p.stem] = _load_calibration(p)
        first_cam = samples[0].datums[self.cfg.camera]
        src_w = int(first_cam.width or 1936)
        src_h = int(first_cam.height or 1216)
        return _Scene(
            scene_id=scene_dir.name,
            scene_dir=scene_dir,
            split=str(raw.get("description", "")),
            samples=samples,
            calibrations=calibrations,
            src_h=src_h,
            src_w=src_w,
        )

    def __len__(self) -> int:
        base = len(self.scenes) * 30 if self.cfg.training else len(self.scenes)
        return max(base, len(self.scenes))

    def _scene(self, index: int) -> _Scene:
        if self.cfg.training:
            sid = int(self.rng.integers(0, len(self.scenes)))
            return self.scenes[sid]
        return self.scenes[index % len(self.scenes)]

    def _frame_indices(self, scene_len: int, index: int) -> list[int]:
        return sample_frame_indices_with_stride(
            rng=self.rng,
            scene_len=scene_len,
            clip_frames=int(self.cfg.clip_frames),
            cfg=self.augment,
            training=bool(self.cfg.training),
            index=index,
        )

    def _sample_key(self, scene: _Scene, idxs: list[int]) -> str:
        frame_token = ",".join(str(int(scene.samples[i].index)) for i in idxs)
        return f"ddad_raw::{scene.split}::{scene.scene_id}::{self.cfg.camera}::frames={frame_token}"

    def _sample_paths(self, scene: _Scene, idxs: list[int]) -> list[str]:
        out: list[str] = []
        for i in idxs:
            sample = scene.samples[i]
            out.append(str(scene.scene_dir / sample.datums[self.cfg.camera].filename))
            out.append(str(scene.scene_dir / sample.datums["LIDAR"].filename))
        return out

    def _project_lidar(self, scene: _Scene, sample: _Sample, k_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera = sample.datums[self.cfg.camera]
        lidar = sample.datums["LIDAR"]
        points_lidar = _load_lidar_npz(scene.scene_dir / lidar.filename)
        if points_lidar.shape[0] > int(self.cfg.max_lidar_points_per_frame):
            idx = self.rng.choice(points_lidar.shape[0], size=int(self.cfg.max_lidar_points_per_frame), replace=False)
            points_lidar = points_lidar[idx]
        t_cam_lidar = np.linalg.inv(camera.sensor_to_world).astype(np.float32) @ lidar.sensor_to_world.astype(np.float32)
        points_cam = _transform_points(t_cam_lidar, points_lidar)
        z = points_cam[:, 2]
        valid = np.isfinite(points_cam).all(axis=1) & np.isfinite(z) & (z > 1e-6)
        points_cam = points_cam[valid]
        if points_cam.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        proj = (k_model @ points_cam.T).T
        uv = np.stack([proj[:, 0] / points_cam[:, 2], proj[:, 1] / points_cam[:, 2]], axis=-1).astype(np.float32)
        return _zbuffer_points(points_cam.astype(np.float32, copy=False), uv, (self.h, self.w))

    def _build_sample(self, scene: _Scene, idxs: list[int], clip_start: int) -> dict[str, Any]:
        video_list: list[np.ndarray] = []
        k_list: list[np.ndarray] = []
        t_wc_list: list[np.ndarray] = []
        cam_valid = np.zeros((len(idxs),), dtype=np.bool_)
        lidar_points: list[np.ndarray] = []
        lidar_pixels: list[np.ndarray] = []

        sx = self.w / float(scene.src_w)
        sy = self.h / float(scene.src_h)
        for out_i, sample_idx in enumerate(idxs):
            sample = scene.samples[sample_idx]
            camera = sample.datums[self.cfg.camera]
            calib = scene.calibrations.get(sample.calibration_key)
            if calib is None or self.cfg.camera not in calib.intrinsics:
                raise _FilteredDdadSampleError(f"Missing calibration for {scene.scene_id}/{sample.index}")
            k = calib.intrinsics[self.cfg.camera].copy().astype(np.float32)
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy
            rgb = _resize_rgb(scene.scene_dir / camera.filename, (self.h, self.w))
            pts, pix = self._project_lidar(scene, sample, k)

            video_list.append(rgb)
            k_list.append(k)
            t_wc_list.append(camera.sensor_to_world.astype(np.float32))
            lidar_points.append(pts)
            lidar_pixels.append(pix)
            cam_valid[out_i] = True

        if min((pts.shape[0] for pts in lidar_points), default=0) < int(self.cfg.min_lidar_points_per_frame):
            raise _FilteredDdadSampleError(f"DDAD clip has too few projected LiDAR points: scene={scene.scene_id}")

        video_hwc = np.stack(video_list, axis=0).astype(np.float32) / 255.0
        video = np.transpose(video_hwc, (0, 3, 1, 2)).astype(np.float32, copy=False)
        if self.cfg.training:
            video = apply_photometric_augment(video, self.rng, self.augment)

        k_arr = np.stack(k_list, axis=0).astype(np.float32)
        t_wc_arr = np.stack(t_wc_list, axis=0).astype(np.float32)
        aspect_ratio = np.array([scene.src_w / max(1.0, scene.src_h)], dtype=np.float32)

        query, target, mask, query_stats, depth, depth_valid = build_queries_from_sparse_lidar(
            rng=self.rng,
            lidar_cam_points=lidar_points,
            lidar_pixels=lidar_pixels,
            k_seq=k_arr,
            t_wc_seq=t_wc_arr,
            camera_valid=cam_valid,
            image_hw=(self.h, self.w),
            queries_per_clip=int(self.cfg.queries_per_clip),
            hard_query_ratio=float(self.cfg.hard_query_ratio),
            prob_t_tgt_equals_t_cam=float(self.cfg.prob_t_tgt_equals_t_cam),
            t_src_tgt_delta_choices=self.cfg.t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=self.cfg.t_src_tgt_delta_probs,
            depth_consistency_abs_m=float(self.cfg.depth_consistency_abs_m),
            depth_consistency_rel=float(self.cfg.depth_consistency_rel),
            depth_consistency_radius_px=float(self.cfg.depth_consistency_radius_px),
            force_tgt_cam_to_src=bool(self.cfg.force_tgt_cam_to_src),
        )

        if int(mask["xyz_3d"].sum()) <= 0:
            raise _FilteredDdadSampleError(f"DDAD clip produced no valid xyz queries: scene={scene.scene_id}")

        return {
            "video": torch.from_numpy(video).float(),
            "aspect_ratio": torch.from_numpy(aspect_ratio.astype(np.float32)),
            "depth_m": torch.from_numpy(depth).float(),
            "depth_valid": torch.from_numpy(depth_valid).bool(),
            "query": {k: torch.from_numpy(v).to(torch.long if k.startswith("t_") else torch.float32) for k, v in query.items()},
            "query_stats": {k: torch.from_numpy(v).bool() for k, v in query_stats.items()},
            "target": {k: torch.from_numpy(v).float() for k, v in target.items()},
            "mask": {k: torch.from_numpy(v).bool() for k, v in mask.items()},
            "camera": {
                "K": torch.from_numpy(k_arr).float(),
                "T_wc": torch.from_numpy(t_wc_arr).float(),
                "camera_valid": torch.from_numpy(cam_valid).bool(),
            },
            "augment_info": {k: torch.from_numpy(v) for k, v in build_augment_info(None, image_hw=(self.h, self.w)).items()},
            "meta": {
                "dataset": "ddad_raw",
                "scene_id": scene.scene_id,
                "split": scene.split,
                "camera": self.cfg.camera,
                "clip_start": int(clip_start),
                "source_mode": "ddad_lidar_sparse_reconstruction",
                "native_hw": (scene.src_h, scene.src_w),
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        total = max(1, len(self))
        for attempt in range(self.max_sample_retries):
            query_index, _ = self._prepare_sample_rng(index=index, total=total, attempt=attempt)
            scene = self._scene(query_index)
            start = query_index % max(1, len(scene.samples) - int(self.cfg.clip_frames) + 1)
            idxs = self._frame_indices(len(scene.samples), start)
            clip_start = int(idxs[0]) if idxs else int(start)
            sample_key = self._sample_key(scene, idxs)
            sample_paths = self._sample_paths(scene, idxs)
            if self.bad_registry.is_bad_sample(sample_key):
                continue
            if self.bad_registry.has_any_bad_path(sample_paths):
                continue
            try:
                sample = self._build_sample(scene, idxs=idxs, clip_start=clip_start)
            except Exception as exc:
                if isinstance(exc, _FilteredDdadSampleError):
                    last_error = exc
                    continue
                if not is_retryable_data_error(exc):
                    raise
                last_error = exc
                self.bad_registry.mark_bad(
                    dataset="ddad_raw",
                    sample_key=sample_key,
                    sample_paths=sample_paths,
                    failed_paths=failed_paths_from_exception(exc),
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            sample["meta"]["sample_key"] = sample_key
            return sample

        raise RuntimeError(
            f"DdadRawDataset failed to produce a valid sample after {self.max_sample_retries} retries. "
            f"last_error={last_error}"
        )
