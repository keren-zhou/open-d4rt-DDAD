from __future__ import annotations

import unittest

import numpy as np

from eval_reconstruction_in_ddad import _compute_scale_factor_global, _local_metric_payload, _ref0_metric_payload
from src.data.ddad_query_builder import build_queries_from_sparse_lidar


class DdadLocalRef0Test(unittest.TestCase):
    def test_exact_half_reference0_queries_and_targets(self) -> None:
        frame_count = 4
        points = [np.asarray([[1.0, 1.0, 2.0]], dtype=np.float32) for _ in range(frame_count)]
        pixels = [np.asarray([[2.0, 2.0]], dtype=np.float32) for _ in range(frame_count)]
        k_seq = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
        t_wc_seq = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
        t_wc_seq[:, 0, 3] = np.arange(frame_count, dtype=np.float32)

        query, target, mask, stats, _, _ = build_queries_from_sparse_lidar(
            rng=np.random.default_rng(7),
            lidar_cam_points=points,
            lidar_pixels=pixels,
            k_seq=k_seq,
            t_wc_seq=t_wc_seq,
            camera_valid=np.ones((frame_count,), dtype=np.bool_),
            image_hw=(8, 8),
            queries_per_clip=10,
            hard_query_ratio=0.0,
            prob_t_tgt_equals_t_cam=1.0,
            reference0_ratio=0.5,
        )

        ref0 = stats["is_reference0_query"]
        self.assertEqual(int(ref0.sum()), 5)
        np.testing.assert_array_equal(query["t_tgt"], query["t_src"])
        np.testing.assert_array_equal(query["t_cam"][ref0], np.zeros((5,), dtype=np.int64))
        np.testing.assert_array_equal(query["t_cam"][~ref0], query["t_src"][~ref0])
        self.assertTrue(mask["xyz_3d"].all())

        expected_x = np.ones((10,), dtype=np.float32)
        expected_x[ref0] += query["t_src"][ref0].astype(np.float32)
        np.testing.assert_allclose(target["xyz_3d"][:, 0], expected_x, atol=1e-6)

    def test_local_and_ref0_metrics_share_one_scale(self) -> None:
        gt_local = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], dtype=np.float64)
        pred_local = gt_local / 2.0
        scale = _compute_scale_factor_global(gt_local, pred_local)
        self.assertAlmostEqual(scale, 2.0)

        local = _local_metric_payload(pred_local, gt_local, scale_global=scale)
        self.assertAlmostEqual(local["local_depth_abs_rel_global"], 0.0)
        self.assertAlmostEqual(local["local_xyz_epe_global_m"], 0.0)

        gt_ref0 = np.asarray([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]], dtype=np.float64)
        ref0 = _ref0_metric_payload(gt_ref0 / 2.0, gt_ref0, scale_global=scale)
        self.assertAlmostEqual(ref0["ref0_xyz_epe_global_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
