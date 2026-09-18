# KITTI LiDAR ablation: Gaussian propagation with ray visibility

## Method

All variants use RichBEV8, the inverted-residual MobilePIXOR backbone,
a center-based 3D head, and the same standard loss: heatmap focal loss plus
masked L1 for center offsets, box dimensions, and yaw. The classification
head starts with a low foreground prior. Coordinate Attention, gated FPN,
and UWAG are absent from these configs. The backbone uses inverted residual
blocks from the published IRBGHR-PIXOR line of work; this KITTI detector is
not a full reproduction of that indoor pedestrian system.

For a1-a3, a fixed separable Gaussian propagates
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
augmentation, targets, optimizer, learning-rate milestones at epochs 30/40,
and 50 total epochs,
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
  bands for Pedestrian/Cyclist, precision, recall, false positives/frame,
  and false positives whose predicted 3D centers fall in sampled observed
  free-space. Also report preprocessing/model/total time and parameters.
- The same 0.05 score threshold, 0.10 NMS threshold, and class IoU
  thresholds (Car 0.70, Pedestrian/Cyclist 0.50) apply to all variants.

The report split is a held-out subset of KITTI training labels. The local
evaluator reports KITTI-style metrics; this is not the official hidden KITTI
test-server score. The observed-free false-positive metric uses the same
three-band ray mask for every variant, counts only unmatched predictions
under the local Moderate 3D matching rules, and is excluded from latency.

## Commands

Run from the repository root. The raw KITTI root must contain
`training/velodyne`, `training/label_2`, and `training/calib`. The detector
reads a prepared directory containing `pointcloud/` and `label/`.

Check the implementation:

```bash
python3 tests/test_gaussian_visibility.py
python3 tests/test_mobile_bev.py
```

On Windows CMD, prepare the raw KITTI data once. Both directories in this
example are on drive D:, allowing hardlinks for the point clouds:

```bat
cd /d D:\project_lidar_BEV\Lidar
python tools\kitti_training_pipeline\prepare_kitti.py --kitti-root D:\project_lidar_BEV\datasets\KITTI\object --output-root D:\project_lidar_BEV\datasets\KITTI\processed --config-output D:\project_lidar_BEV\datasets\KITTI\processed\prep_config.json --pointcloud-mode hardlink
```

Train a0 first on Windows CMD, then select its checkpoint and report metrics:

```bat
python tools\kitti_training_pipeline\run_gaussian_visibility_ablation.py --variant a0 --seed 42 --epochs 50 --physical-batch-size 2 --accumulation-steps 2 --precision bf16 --num-workers 2 --device cuda --stage all --kitti-root D:\project_lidar_BEV\datasets\KITTI\object --processed-root D:\project_lidar_BEV\datasets\KITTI\processed --output-root D:\project_lidar_BEV\Lidar\artifacts\kitti_irb
```

Change only `--variant` to a1, a2, a3, and a4 for the first screen. With
`--stage train`, the runner stops after training; run the same command
with `--stage select` and then `--stage report` to finish that variant.
A run writes `metrics.jsonl` during training and `report.json` after the
held-out evaluation. For example, a0 seed 42 writes under
`artifacts/kitti_irb/gaussian_visibility_irb_a0_seed42/`.

On WSL, the equivalent single-run command is:

```bash
python3 tools/kitti_training_pipeline/run_gaussian_visibility_ablation.py \
  --variant a0 --seed 42 --epochs 50 \
  --physical-batch-size 2 --accumulation-steps 2 \
  --precision bf16 --num-workers 2 --device cuda --stage all \
  --kitti-root /home/ducanh/ros2_ws/data/KITTI/object \
  --processed-root /home/ducanh/datasets/data-kitti/training \
  --output-root artifacts/kitti_irb
```

Repeat all variants with seeds 43 and 44. For interrupted training, pass
`--stage train --resume PATH_TO_CHECKPOINT`, then run selection and reporting
as separate stages. To summarize completed reports under the custom output
root:

```bat
python tools\kitti_training_pipeline\summarize_gaussian_visibility.py --results-root D:\project_lidar_BEV\Lidar\artifacts\kitti_irb --output D:\project_lidar_BEV\Lidar\artifacts\kitti_irb\gaussian_visibility_irb_summary.csv
```

The summary requires a full 997-frame report for each variant and seed.
Use `--allow-incomplete` to inspect completed runs before all 15 finish.

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
- IRBGHR-PIXOR (published backbone lineage): https://doi.org/10.1109/ACCESS.2024.3351868
- CenterNet (center-based heatmap): https://arxiv.org/abs/1904.07850
- CenterPoint (center-based 3D detection): https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html
- WYSIWYG visibility for 3D detection: https://peiyunh.github.io/wysiwyg/index.html
- SPV-SSD visibility states: https://www.mdpi.com/2072-4292/15/1/161
