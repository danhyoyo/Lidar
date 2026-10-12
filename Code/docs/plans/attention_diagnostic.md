# First experiment: does global context help this LiDAR detector?

## Update after inspecting existing runs

See [the artifact review](attention_artifact_review.md) before starting new
training. Existing runs use 100 epochs and a 5,984/1,497 split; their original
manifests have been recovered in `splits/kitti/archived_5984_1497/`. The current
workspace manifests used below have 3,712/3,769 frames. Do not compare the
50-epoch runs below directly with the archived results.

The archive already contains a LiteMLA baseline, but its convolution baseline
also changes the neck and head. The most economical missing experiment is a
convolution model with the same SG-FPN and BN/SiLU head as that archived
LiteMLA run, matching its source, actual batch size 16, and 100-epoch recipe.
Its existing `last.pt` needs evaluation. Use the fresh pair below if reproducing
the archived setup is impractical.

## Objective

Train a controlled convolution versus LiteMLA pair before developing learned
context routing. Measure both aggregate detection quality and the distance bands
already supported by the evaluator. These runs are diagnostic experiments, not
evidence of a new method or state-of-the-art performance.

## Frozen experiment settings

| Setting | Value |
| --- | --- |
| Backbone | MobilePixorNeXt |
| Input | rich8, 800 x 704 |
| Neck | scale-gated FPN |
| Loss | baseline, as in the current master config |
| Seed | 42 |
| Training | 50 epochs, AdamW, cosine schedule with warmup |
| Training precision | BF16 |
| Physical batch / accumulation | 16 / 1 |
| Train / validation manifests | splits/kitti/train.txt / splits/kitti/val.txt |
| E0 | c4_attention = none |
| E1 | c4_attention = litemla, scales = [5], qk_norm = none |

The two config snapshots differ only in experiment.name and
model.c4_attention. If a different loss or batch size is needed, apply exactly
the same change to both snapshots before starting. Accumulation with a smaller
physical batch changes BatchNorm statistics; keep that setting equal across runs.

## Prerequisites

Run commands from the repository root in the CUDA/PyTorch environment used for
training. Both runs need CUDA with BF16 support, the project dependencies, and
prepared KITTI data at data/kitti/processed. The original KITTI object dataset is
also needed for evaluation labels and calibration.

At preparation time, this workspace had the split manifests but no installed
PyTorch and no data/kitti/processed directory. No training or GPU benchmark was
executed while preparing this protocol. Follow README.md to prepare the runtime
and dataset if needed. Keep the supplied manifests when preparing the data.

## Train the two models

Run E0 first, then E1 on the same GPU. Each command starts a fresh training run.
Use a new run name if these output directories already contain another run.

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/experiments/attention_diagnostic/e0_conv.json \
  --detector-root detector \
  --output-root artifacts/attention_diagnostic \
  --run-name e0_conv_s42 \
  --seed 42 --epochs 50 \
  --num-workers 6 --target-backend numba --precision bf16
```

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/experiments/attention_diagnostic/e1_litemla.json \
  --detector-root detector \
  --output-root artifacts/attention_diagnostic \
  --run-name e1_litemla_s42 \
  --seed 42 --epochs 50 \
  --num-workers 6 --target-backend numba --precision bf16
```

Short runs can validate the data/training pipeline, but their AP is not a final
architecture comparison. Do not remove attention from a trained LiteMLA
checkpoint and treat the result as the independently trained convolution model.

## Evaluate at the same training endpoint

Replace /path/to/KITTI/object with the directory containing training/label_2 and
training/calib. The primary comparison uses checkpoints/last.pt after both runs
finish 50 epochs. The trainer's selected/best.pt is selected by validation loss,
not by AP; evaluating it is an optional secondary comparison with the same
selection policy for both models.

Use each run's config.resolved.json, which records the actual training settings.
The current evaluator runs eager PyTorch FP32 at batch size 1, regardless of
training precision. Measure on the same otherwise idle GPU. These timings are
not BF16 inference benchmarks.

```bash
for variant in e0_conv e1_litemla; do
  python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
    --name "${variant}_s42_last" \
    --backend pytorch \
    --model "artifacts/attention_diagnostic/${variant}_s42/checkpoints/last.pt" \
    --config "artifacts/attention_diagnostic/${variant}_s42/config.resolved.json" \
    --detector-root detector \
    --kitti-root /path/to/KITTI/object \
    --split splits/kitti/val.txt \
    --output "artifacts/attention_diagnostic/evaluation/${variant}_s42.json" \
    --device cuda --warmup-frames 100
done
```

Do not pass both checkpoints to compare_models.py with a single config: the
architectures differ and checkpoint loading is strict.

## Record these measurements

| Measurement | Evaluation JSON location |
| --- | --- |
| BEV mAP Moderate | accuracy.map_moderate_percent |
| Per-class / difficulty AP, recall, FP counts | accuracy.per_class |
| Pedestrian and Cyclist distance bands | accuracy.per_class.CLASS.moderate_distance_bands |
| Model mean / p95 latency | latency.model.mean_ms / p95_ms |
| Offline input-to-detections latency | latency.input_to_detections |
| Peak allocated memory | runtime.torch_peak_memory_mb |
| Saved per-frame predictions | predictions.path |

The existing evaluator reports local KITTI-style BEV AP R40. It does not report
official hidden-test results or 3D AP. It currently provides distance bands for
Pedestrian and Cyclist; point-count and region-level benefit diagnostics require
additional analysis tooling.

## Decide the next experiment

1. If LiteMLA improves AP, check which classes and distance bands benefit. If the
   difference is small, repeat both models with seeds 43 and 44 before attributing
   it to architecture. Use unique run/output names for each seed.
2. If gains appear concentrated, build a paired region-level diagnostic of the
   loss/quality change with versus without context. Aggregate AP alone does not
   establish that a routing predictor can identify helpful regions.
3. Compare a learned router against random, confidence-only, and range/density
   routing under the same query budget. Account for selection, gather/scatter,
   memory construction, and attention time in the latency result.
4. If there is no stable benefit in aggregate or useful subsets, investigate the
   detector's actual failure modes before committing to context routing. Compare
   additional convolution capacity or another context operator as appropriate.
5. After a promising prototype, test a second backbone and second dataset with
   consistent protocols. Validate novelty against DynamicDet, Ada3D, BiFormer,
   and recent adaptive LiDAR detectors before framing a paper claim.
