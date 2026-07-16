"""DDAD sparse LiDAR query builder for D4RT training."""

from __future__ import annotations

import numpy as np

from .raw_augment import depth_boundary_mask, sample_hard_query_flags, sample_t_tgt_t_cam


def _invert_poses(t_wc_seq: np.ndarray, camera_valid: np.ndarray) -> np.ndarray:
    t = int(t_wc_seq.shape[0])
    out = np.full((t, 4, 4), np.nan, dtype=np.float32)
    for i in range(t):
        if not bool(camera_valid[i]):
            continue
        try:
            out[i] = np.linalg.inv(t_wc_seq[i]).astype(np.float32)
        except np.linalg.LinAlgError:
            continue
    return out


def _project_points(k: np.ndarray, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    proj = (np.asarray(k, dtype=np.float32) @ points_cam[:, :3].T).T
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = proj[valid, 0] / z[valid]
    uv[valid, 1] = proj[valid, 1] / z[valid]
    return uv, valid


def _nearest_sparse_depth(
    uv_px: np.ndarray,
    target_pixels: np.ndarray,
    target_depth: np.ndarray,
    *,
    radius_px: float,
) -> np.ndarray:
    out = np.full((uv_px.shape[0],), np.nan, dtype=np.float32)
    if uv_px.shape[0] == 0 or target_pixels.shape[0] == 0:
        return out
    radius2 = float(radius_px) * float(radius_px)
    for i, uv in enumerate(uv_px):
        if not np.isfinite(uv).all():
            continue
        diff = target_pixels - uv[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        j = int(np.argmin(dist2))
        if float(dist2[j]) <= radius2:
            out[i] = float(target_depth[j])
    return out


def build_queries_from_sparse_lidar(
    *,
    rng: np.random.Generator,
    lidar_cam_points: list[np.ndarray],
    lidar_pixels: list[np.ndarray],
    k_seq: np.ndarray,
    t_wc_seq: np.ndarray,
    camera_valid: np.ndarray,
    image_hw: tuple[int, int],
    queries_per_clip: int,
    hard_query_ratio: float,
    prob_t_tgt_equals_t_cam: float,
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None,
    t_src_tgt_delta_probs: tuple[float, ...] | None = None,
    depth_consistency_abs_m: float = 0.5,
    depth_consistency_rel: float = 0.05,
    depth_consistency_radius_px: float = 1.5,
    force_tgt_cam_to_src: bool = False,
    reference0_ratio: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build D4RT query/target/mask/query_stats from projected DDAD LiDAR.

    ``lidar_cam_points[i]`` contains sparse 3D points in camera-i coordinates;
    ``lidar_pixels[i]`` contains their projected pixel coordinates in the model
    image. Both arrays must be z-buffered consistently by the caller.
    """

    t = int(len(lidar_cam_points))
    h, w = int(image_hw[0]), int(image_hw[1])
    m = int(queries_per_clip)
    if t <= 0:
        raise ValueError("DDAD query builder received an empty clip")

    q_t_src = rng.integers(0, t, size=(m,), dtype=np.int64)
    is_reference0_query = np.zeros((m,), dtype=np.bool_)
    if bool(force_tgt_cam_to_src) and reference0_ratio is not None:
        raise ValueError("force_tgt_cam_to_src and reference0_ratio cannot both be enabled")
    if bool(force_tgt_cam_to_src):
        q_t_tgt = q_t_src.copy()
        q_t_cam = q_t_src.copy()
    elif reference0_ratio is not None:
        ratio = float(reference0_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"reference0_ratio must be in [0, 1], got {ratio}")
        q_t_tgt = q_t_src.copy()
        q_t_cam = q_t_src.copy()
        is_reference0_query = sample_hard_query_flags(rng, m, ratio)
        q_t_cam[is_reference0_query] = 0
    else:
        q_t_tgt, q_t_cam, _ = sample_t_tgt_t_cam(
            rng=rng,
            queries_per_clip=m,
            clip_frames=t,
            prob_t_tgt_equals_t_cam=float(prob_t_tgt_equals_t_cam),
            q_t_src=q_t_src,
            t_src_tgt_delta_choices=t_src_tgt_delta_choices,
            t_src_tgt_delta_probs=t_src_tgt_delta_probs,
        )

    q_u = np.zeros((m,), dtype=np.float32)
    q_v = np.zeros((m,), dtype=np.float32)
    y_uv = np.zeros((m, 2), dtype=np.float32)
    y_xyz = np.zeros((m, 3), dtype=np.float32)
    y_disp = np.zeros((m, 3), dtype=np.float32)
    y_normal = np.zeros((m, 3), dtype=np.float32)
    y_vis = np.zeros((m,), dtype=np.float32)

    m_uv = np.zeros((m,), dtype=np.bool_)
    m_xyz = np.zeros((m,), dtype=np.bool_)
    m_disp = np.zeros((m,), dtype=np.bool_)
    m_vis = np.zeros((m,), dtype=np.bool_)
    m_normal = np.zeros((m,), dtype=np.bool_)
    is_hard_query = np.zeros((m,), dtype=np.bool_)

    t_wc = np.asarray(t_wc_seq, dtype=np.float32)
    t_cw = _invert_poses(t_wc, camera_valid)

    sparse_depth = np.full((t, h, w), np.nan, dtype=np.float32)
    sparse_valid = np.zeros((t, h, w), dtype=np.bool_)
    valid_slots: list[np.ndarray] = []
    hard_slots: list[np.ndarray] = []
    for fi in range(t):
        pts = np.asarray(lidar_cam_points[fi], dtype=np.float32)
        pix = np.asarray(lidar_pixels[fi], dtype=np.float32)
        valid = bool(camera_valid[fi]) and pts.ndim == 2 and pix.ndim == 2 and pts.shape[0] > 0 and pix.shape[0] == pts.shape[0]
        if not valid:
            valid_slots.append(np.zeros((0,), dtype=np.int64))
            hard_slots.append(np.zeros((0,), dtype=np.int64))
            continue
        z = pts[:, 2]
        finite = np.isfinite(pts).all(axis=1) & np.isfinite(pix).all(axis=1) & np.isfinite(z) & (z > 1e-6)
        valid_slots.append(np.flatnonzero(finite).astype(np.int64))
        u_int = np.clip(np.rint(pix[finite, 0]).astype(np.int64), 0, w - 1)
        v_int = np.clip(np.rint(pix[finite, 1]).astype(np.int64), 0, h - 1)
        idx_finite = np.flatnonzero(finite)
        for local_idx, uu, vv in zip(idx_finite, u_int, v_int, strict=False):
            zz = float(z[int(local_idx)])
            old = sparse_depth[fi, vv, uu]
            if not np.isfinite(old) or zz < float(old):
                sparse_depth[fi, vv, uu] = zz
                sparse_valid[fi, vv, uu] = True
        if float(hard_query_ratio) > 0.0 and sparse_valid[fi].any():
            bmask = depth_boundary_mask(sparse_depth[fi], sparse_valid[fi], q=0.8)
            if bmask.any():
                px = np.clip(np.rint(pix[:, 0]).astype(np.int64), 0, w - 1)
                py = np.clip(np.rint(pix[:, 1]).astype(np.int64), 0, h - 1)
                hard = finite & bmask[py, px]
                hard_slots.append(np.flatnonzero(hard).astype(np.int64))
            else:
                hard_slots.append(np.zeros((0,), dtype=np.int64))
        else:
            hard_slots.append(np.zeros((0,), dtype=np.int64))

    hard_target = int(sample_hard_query_flags(rng, m, float(hard_query_ratio)).sum())
    hard_eligible = np.array([i for i in range(m) if hard_slots[int(q_t_src[i])].size > 0], dtype=np.int64)
    use_hard = np.zeros((m,), dtype=np.bool_)
    if hard_target > 0 and hard_eligible.size > 0:
        picked = rng.choice(hard_eligible, size=min(hard_target, hard_eligible.size), replace=False)
        use_hard[picked.astype(np.int64)] = True

    w_norm = max(1.0, float(w - 1))
    h_norm = max(1.0, float(h - 1))

    for i in range(m):
        fs = int(q_t_src[i])
        ft = int(q_t_tgt[i])
        fc = int(q_t_cam[i])
        if not (bool(camera_valid[fs]) and bool(camera_valid[ft]) and bool(camera_valid[fc])):
            continue
        if not (np.isfinite(t_wc[fs]).all() and np.isfinite(t_cw[fc]).all() and np.isfinite(t_cw[ft]).all()):
            continue

        src_candidates = hard_slots[fs] if bool(use_hard[i]) and hard_slots[fs].size > 0 else valid_slots[fs]
        if src_candidates.size == 0:
            continue
        src_idx = int(src_candidates[int(rng.integers(0, src_candidates.size))])
        is_hard_query[i] = bool(use_hard[i]) and hard_slots[fs].size > 0

        src_point_cam = np.asarray(lidar_cam_points[fs][src_idx], dtype=np.float32)
        src_uv = np.asarray(lidar_pixels[fs][src_idx], dtype=np.float32)
        if not (np.isfinite(src_point_cam).all() and np.isfinite(src_uv).all() and float(src_point_cam[2]) > 1e-6):
            continue

        q_u[i] = float(np.clip(src_uv[0] / w_norm, 0.0, 1.0))
        q_v[i] = float(np.clip(src_uv[1] / h_norm, 0.0, 1.0))

        src_h = np.array([float(src_point_cam[0]), float(src_point_cam[1]), float(src_point_cam[2]), 1.0], dtype=np.float32)
        world_h = t_wc[fs] @ src_h
        xyz_cam = (t_cw[fc] @ world_h)[:3]
        if not np.isfinite(xyz_cam).all():
            continue
        y_xyz[i] = xyz_cam.astype(np.float32)
        m_xyz[i] = True

        # This is a static-world supervision proxy. It is trustworthy only when
        # the same world point is depth-consistent in the target frame.
        target_cam = (t_cw[ft] @ world_h)[:3]
        uv_tgt, valid_proj = _project_points(k_seq[ft], target_cam[None, :])
        target_visible = False
        if bool(valid_proj[0]):
            u_tgt = float(uv_tgt[0, 0])
            v_tgt = float(uv_tgt[0, 1])
            target_visible = 0.0 <= u_tgt <= (w - 1) and 0.0 <= v_tgt <= (h - 1)
            if target_visible:
                target_sparse_depth = _nearest_sparse_depth(
                    uv_tgt,
                    np.asarray(lidar_pixels[ft], dtype=np.float32),
                    np.asarray(lidar_cam_points[ft], dtype=np.float32)[:, 2],
                    radius_px=float(depth_consistency_radius_px),
                )[0]
                if np.isfinite(target_sparse_depth):
                    z_tgt = float(target_cam[2])
                    tol = max(float(depth_consistency_abs_m), float(depth_consistency_rel) * max(1.0, abs(target_sparse_depth)))
                    if abs(float(target_sparse_depth) - z_tgt) <= tol:
                        y_uv[i, 0] = float(np.clip(u_tgt / w_norm, 0.0, 1.0))
                        y_uv[i, 1] = float(np.clip(v_tgt / h_norm, 0.0, 1.0))
                        y_vis[i] = 1.0
                        m_uv[i] = True
                        m_vis[i] = True
                        m_disp[i] = True

        if fs == ft:
            y_uv[i, 0] = q_u[i]
            y_uv[i, 1] = q_v[i]
            y_vis[i] = 1.0
            m_uv[i] = True
            m_vis[i] = True
            m_disp[i] = True

    query = {"u": q_u, "v": q_v, "t_src": q_t_src, "t_tgt": q_t_tgt, "t_cam": q_t_cam}
    query_stats = {
        "is_hard_query": is_hard_query,
        "is_reference0_query": is_reference0_query,
    }
    target = {"xyz_3d": y_xyz, "uv_2d": y_uv, "visibility": y_vis, "displacement": y_disp, "normal": y_normal}
    mask = {"xyz_3d": m_xyz, "uv_2d": m_uv, "visibility": m_vis, "displacement": m_disp, "normal": m_normal}
    return query, target, mask, query_stats, sparse_depth, sparse_valid
