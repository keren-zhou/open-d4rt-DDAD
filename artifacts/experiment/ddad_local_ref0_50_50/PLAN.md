# DDAD Local/Ref0 50/50 Experiment Plan

## Contract

- Baseline: `output/ddad_reconstruction_train`, trained with 100% `(t_src, t_src, t_src)` queries.
- New supervision: exact 50% local `(t_src, t_src, t_src)` and 50% reference-0 `(t_src, t_src, 0)` queries per clip.
- Scale contract: keep OpenD4RT mean-depth-normalized XYZ training loss; use one GT-derived scale per DDAD scene for evaluation.
- Primary metrics: local scale-aligned depth AbsRel and reference-0 scale-aligned XYZ EPE.
- Raw metric-space errors are not primary metrics because the XYZ training objective is scale invariant.

## Implementation

- Add an explicit DDAD `reference0_ratio` sampling setting and query branch labels.
- Preserve the legacy `force_tgt_cam_to_src` behavior for old configs.
- Evaluate local and reference-0 queries on identical sparse LiDAR pixels.
- Estimate one scene scale from all local correspondences, then apply it unchanged to both metric branches.
- Add an isolated training config and output directory.

## Validation

- Synthetic tests for exact branch ratios and reference-frame GT transforms.
- DDAD dataloader smoke check for query counts and finite targets.
- Evaluation helper tests for one shared scale and aligned metrics.
- Training command dry run.

## Run

- Full run: 20,000 steps from the same OpenD4RT 48-frame checkpoint and optimizer settings as the baseline.
- Output: `output/ddad_reconstruction_local_ref0_50_50_train`.
- Stop if query ratios, ref0 GT transforms, or scene-global scale checks fail.

