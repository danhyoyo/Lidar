# AP validation runtime investigation

## Observed Colab log

Focal-only C3 run `...ctx_focal_atc3_local_none...-s42-F3-h32`:

- 3,712 training frames, batch 16: 232 batches, 122.0 seconds.
- 3,769 validation frames: validation loss completed in 62.0 seconds.
- Last output: `AP validation: epoch 1/50; full validation split; local_bev R40 Moderate`.
- User reports 50 minutes without further output. This places the waiting phase
  after training/loss validation. The exact live stack is not yet supplied;
  source inspection cannot distinguish hashing/I/O, initial inference and CPU
  decode/NMS. Requested an interruption traceback.

## Confirmed code findings

`evaluate_training_ap` saves a temporary checkpoint on the run filesystem. The
local evaluator then hashes every pointcloud, raw label and calibration file
before loading the runner and reporting frame progress. This split has 11,307
local input paths. The old interval was 50 frames, leaving preparation and
frames 1–49 silent.

Without torchvision rotated NMS, decoding uses Shapely CPU NMS. The previous
implementation computes polygon union/intersection for every remaining candidate
against each greedy pick, including disjoint boxes. The 500-detection cap applies
after NMS. Dense early-epoch predictions can be expensive; this is a plausible
bottleneck, not a confirmed Colab stack diagnosis.

## Changes

- Optional log callbacks route candidate saving, input hashing, model loading,
  first-frame phases, elapsed/ETA and AP computation into stdout and `train.log`.
  `progress_every=0` continues to suppress evaluator progress.
- Asset hashing still reads and hashes all original inputs; progress does not
  cache stale hashes or bypass provenance checks.
- CPU NMS precomputes AABBs and rejects disjoint pairs before the original GEOS
  IoU. Potentially overlapping pairs keep the original arithmetic and float32
  IoUs. Sorting/ties, classwise suppression, strict threshold comparison and
  post-NMS detection cap are unchanged.
- Model routing, loss, optimizer, AP schedule, selection, RNG and checkpoint
  behavior are unchanged. The completed epoch is still saved after AP.

## Verification

Focused suite: **222 passed, 2 skipped**, 38.22 seconds. Skipped cases require
reference CUDA. Coverage includes frozen flat decode, randomized/tied-score NMS,
touching/degenerate rectangles, 3D decode, grouped evaluation, hash provenance,
actual notebook training/AP, RNG parity, best-checkpoint selection and resume.
Independent review found no Blocker/Major; 16 tests and 40 extra parity checks
passed, including dtype and threshold-boundary cases.

Same-interpreter sparse benchmark, 300 disjoint boxes:

| CPU NMS | Kept | Polygon pairs | Seconds |
| --- | ---: | ---: | ---: |
| Previous | 300 | 44,850 | 0.3523 |
| AABB broad-phase | 300 | 0 | 0.0074 |

This demonstrates eliminated work, not the Colab full-split AP duration. Dense
overlapping scenes can still require many polygon operations; the new logs
locate remaining bottlenecks. No actual KITTI training was launched locally.

## Colab continuation

The running subprocess cannot load this patch. Stop the cell, update/reclone
source and restart training with the same saved config. Completed checkpoints
remain resumable. Without a completed epoch-1 checkpoint, epoch 1 runs again;
the observed train/validation time was about 3 minutes.

The existing `AP_EVERY` control can reduce scheduled full-split passes. Use the
same schedule across compared placements and a fresh run name when changing a
saved config; fewer AP checks change checkpoint selection opportunities and are
separate from this exact NMS optimization.
