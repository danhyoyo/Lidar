# Reproduced validation result

- Selected checkpoint: epoch 55 (minimum mean validation loss)
- Training precision: BF16
- Validation split: 1,497 KITTI frames from `splits/kitti/val.txt`
- Local KITTI-style rotated BEV AP R40:
  - mAP Moderate: 79.7872%
  - Mean AP over 3 classes x 3 difficulties: 80.3021%
  - Car/Pedestrian/Cyclist Moderate: 92.9813 / 65.7268 / 80.6534%
- PyTorch FP32 evaluation on RTX 5060 Ti:
  - model-only: 79.041 FPS
  - input-to-detections: 15.630 FPS
  - peak allocated VRAM: 355.1 MiB

The run became non-finite at epoch 81. Epoch 100 is therefore not a valid
evaluation checkpoint. The full machine-readable report is in
`evaluation_best_val.json`.
