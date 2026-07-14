"""Lightweight DDAD dataloader smoke check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import apply_overrides, load_yaml_config
from src.data.builder import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default="configs/train_ddad_reconstruction.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--max-scenes", type=int, default=2)
    parser.add_argument("--queries-per-clip", type=int, default=512)
    parser.add_argument("--data-root", default="/data/jhc/ddad_train_val")
    parser.add_argument("--camera", default="CAMERA_01")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.train_config)
    overrides = [
        f"data.ddad.root={args.data_root}",
        f"data.ddad.camera={args.camera}",
        f"data.ddad.max_scenes={int(args.max_scenes)}",
        f"train_sampling.queries_per_clip={int(args.queries_per_clip)}",
        "runtime.train_num_workers=0",
        "runtime.val_num_workers=0",
    ]
    cfg = apply_overrides(cfg, overrides)
    dataset = build_dataset(split=str(args.split), cfg=cfg, manifest_arg=None)
    sample = dataset[0]

    mask_xyz = sample["mask"]["xyz_3d"]
    mask_uv = sample["mask"]["uv_2d"]
    finite_xyz = torch.isfinite(sample["target"]["xyz_3d"]).all(dim=-1)
    payload = {
        "split": str(args.split),
        "dataset_len": int(len(dataset)),
        "video_shape": list(sample["video"].shape),
        "query_shape": {k: list(v.shape) for k, v in sample["query"].items()},
        "target_shape": {k: list(v.shape) for k, v in sample["target"].items()},
        "xyz_valid": int(mask_xyz.sum().item()),
        "uv_valid": int(mask_uv.sum().item()),
        "finite_xyz_on_valid": bool(finite_xyz[mask_xyz].all().item()) if bool(mask_xyz.any()) else False,
        "depth_valid_pixels": int(sample["depth_valid"].sum().item()),
        "meta": sample["meta"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["xyz_valid"] <= 0 or not payload["finite_xyz_on_valid"]:
        raise RuntimeError(f"DDAD smoke failed: {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
