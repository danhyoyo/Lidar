# KITTI LiDAR ablation: Gaussian propagation with ray visibility

## Method

All variants use RichBEV8, MobilePIXOR with Coordinate Attention, SG-FPN,
Center3D, and the same loss. For a1-a3, a fixed separable Gaussian propagates
each BEV feature channel into nearby empty cells:

    F_out[c,u] = F_base[c,u] + (1 - occupied[u]) * allowed[c,u]
                 * strength * (K * F_base[c])[u]

Measured cells keep their original features. The Gaussian uses sigma 1.5
input cells (0.15 m), radius 4 cells (0.4 m), and strength 1.0. This is
propagation of aggregated pillar features, not learned point Gaussians or 3DGS.

a2 and a3 form visibility masks by sampling the actual 3D segment immediately
before each LiDAR return. Samples cover 0.1-0.7 m before the return at 0.05 m
intervals. Each sample is placed into a BEV cell at its interpolated z height.
A cell crossed by a sampled ray is known free at that height; cells without
a ray sample stay unknown. Cells containing measured returns are never
masked. Only a local segment is sampled because the propagation kernel has
compact support. This is not a full scene occupancy map.

The dataset passes a temporary marker at the sensor origin through the same
augmentation as the points. It removes that marker before BEV encoding and
uses its transformed position as the origin of the visibility rays. Rotation,
scaling, and translation therefore share the same sensor geometry.

## Fixed configurations

| Variant | Config | Question tested |
| --- | --- | --- |
| a0 | `a0_rich8_baseline.json` | RichBEV8 baseline |
| a1 | `a1_gaussian_plain.json` | Does Gaussian propagation help? |
| a2 | `a2_gaussian_free.json` | Does one free-space mask help beyond a1? |
| a3 | `a3_gaussian_height_free.json` | Do three height-aware masks help beyond a2? |
| a4 | `a4_uniform_plain.json` | Is the Gaussian kernel shape useful beyond generic smoothing? |

a2 gates all eight propagated feature channels with one mask. a3 gates
occupancy channels 0-2 with masks aligned to the three RichBEV height bands
over z=[-2.5,1.0] m. Its summary channels 3-7 are gated only if all three
bands are known free. a4 uses a uniform 5x5 separable kernel. Its one-axis
second moment is 2.0 cells squared versus approximately 2.2 for the
truncated Gaussian, giving a similar spread scale.

All variants keep the same trainable detector parameters, geometry,
augmentation, targets, optimizer, learning-rate schedule, 50 epochs,
physical batch 2, accumulation 2, BF16, and checkpoint interval of 10
epochs. The propagation kernels have no trainable parameters.

## Data, metrics, and selection

- Train on the committed 5,984-frame `splits/kitti/train.txt`.
- Use 500 frames in `splits/kitti/gaussian_visibility/select.txt` for
  per-epoch validation loss and checkpoint selection.
- Choose the saved checkpoint with the highest local 3D Moderate mAP R40
  on the 500-frame selection split.
- Evaluate the chosen checkpoint once on the disjoint 997-frame
  `splits/kitti/gaussian_visibility/report.txt`.
- Start with seed 42 for a0-a4. Then repeat all five with seeds 43 and 44.
- Main comparison: 3D Moderate mAP R40, with each seed and mean ± sample
  standard deviation. Report per-class 3D/BEV AP by difficulty, distance
  bands for Pedestrian/Cyclist, precision, recall, false positives/frame
  at score threshold 0.05, preprocessing/model/total time, and parameters.
- The same 0.05 score threshold, 0.10 NMS threshold, and class IoU
  thresholds (Car 0.70, Pedestrian/Cyclist 0.50) apply to all variants.

The report split is a held-out subset of KITTI training labels. The local
evaluator reports KITTI-style metrics; this is not the official hidden KITTI
test-server score.

## Commands

Run from the repository root in WSL. The processed KITTI data path is
configured as `data/kitti/processed`. The raw KITTI root passed below must
contain `training/label_2` and `training/calib`. If the data is missing,
prepare it using the repository README first.

Check the implementation:

```bash
python3 tests/test_gaussian_visibility.py
python3 tests/test_mobile_bev.py
```

Train, select a checkpoint, and evaluate one variant:

```bash
python3 tools/kitti_training_pipeline/run_gaussian_visibility_ablation.py \
  --variant a3 --seed 42 \
  --kitti-root /home/ducanh/ros2_ws/data/KITTI/object
```

Run the full five-variant screen with seed 42:

```bash
for variant in a0 a1 a2 a3 a4; do
  python3 tools/kitti_training_pipeline/run_gaussian_visibility_ablation.py \
    --variant "$variant" --seed 42 \
    --kitti-root /home/ducanh/ros2_ws/data/KITTI/object
done
```

Repeat with seeds 43 and 44. Each run writes
`artifacts/kitti/gaussian_visibility_aN_seedS/metrics.jsonl` during training
and `report.json` after evaluation. To resume a phase, use
`--stage train`, `--stage select`, or `--stage report`. For an interrupted
training run, pass `--stage train --resume /path/to/checkpoint.pt`, then run
the select and report stages.

Aggregate completed full reports:

```bash
python3 tools/kitti_training_pipeline/summarize_gaussian_visibility.py
```

The command writes `artifacts/kitti/gaussian_visibility_summary.csv` and
prints mean ± standard deviation for 3D Moderate mAP R40. Use
`--allow-incomplete` to inspect the runs available so far.

## Interpretation

The primary causal comparisons are a1-a0, a2-a1, a3-a2, and a1-a4.
a4 has a similar spread variance rather than identical support; that
difference should be considered when interpreting a1-a4. The ray masks
represent measured free space near returns. They do not infer the full
unseen interior of an object. A3 may improve precision but reduce recall;
inspect both, especially for distant Pedestrian/Cyclist objects.

RadarGaussianDet3D already studies Gaussian splatting for radar BEV, and
visibility has previously been used for 3D detection. This experiment
tests a specific local ray constraint on fixed Gaussian pillar feature
propagation on KITTI.

- RadarGaussianDet3D: https://arxiv.org/abs/2509.16119
- WYSIWYG visibility for 3D detection: https://peiyunh.github.io/wysiwyg/index.html
- SPV-SSD visibility states: https://www.mdpi.com/2072-4292/15/1/161
