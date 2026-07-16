# DDAD Local/Ref0 50/50 Run Status

## Status

Implementation and CPU/data validation are complete. The 20,000-step run has not started because the current environment reports `torch.cuda.is_available() == False` and zero CUDA devices.

## Verified

- Unit tests: `python -m unittest tests.test_ddad_local_ref0 -v`.
- Train data smoke: exact 256/256 split for 512 queries.
- Validation data smoke: exact 2048/2048 split for 4096 queries.
- Legacy config smoke: 100% local behavior remains unchanged.
- Training launcher dry run: resolves the new config, baseline checkpoint, 20,000 steps, and isolated output directory.

## Full Run Command

```bash
bash scripts/train_ddad_reconstruction_4gpu.sh \
  --train-config configs/train_ddad_reconstruction_local_ref0_50_50.yaml \
  --output-dir output/ddad_reconstruction_local_ref0_50_50_train \
  --init-model checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --data-root /data/jhc/ddad_train_val \
  --total-steps 20000 \
  --gpus 0,1,2,3
```

The pre-training checkpoint and trained checkpoint must both be evaluated with the revised evaluator before comparing the three primary metrics.

