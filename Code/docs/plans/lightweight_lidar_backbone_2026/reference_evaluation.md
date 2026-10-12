# KITTI reference AP3D and APBEV

Tasks55–57 implement the vertical objective, calibrated 3D decoding and an independent reference evaluator. Task58 adds main/ECA 3D runtime recipes and dedicated CUDA/AMP/worker training-resume-decode smoke; see [3D verification](release_b_verification.md). Synthetic perfect predictions verify evaluator behavior; they are not trained KITTI accuracy.

## Immutable reference and dependencies

The vendored OpenPCDet evaluator and rotated overlap are byte-identical to revision **233f849829b6ac19afb8af8837a0246890908755**:

- [Evaluator source](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/kitti/kitti_object_eval_python/eval.py).
- [CUDA overlap source](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/kitti/kitti_object_eval_python/rotate_iou.py).
- [Local provenance and SHA256 manifest](../../../tools/kitti_training_pipeline/kitti_reference/provenance.json). Repository Apache-2.0 and the overlap's MIT RRPN notices/licenses are retained. Every listed source/license hash is checked before importing the evaluator.

The reference requires actual compatible Numba CUDA, libNVVM and libdevice. Dependency failure raises an error; it never substitutes the historical ROI BEV evaluator or a CPU overlap implementation. AP calculations and metric-specific ignore rules are unchanged. The wrapper partitions overlap work using `num_parts=50`, avoiding a single quadratic overlap allocation for a full validation split.

On this machine, the training interpreter is Python3.14.4 with Torch2.11.0, NumPy2.4.6 and Numba0.67.0. The bundled older Numba CUDA backend failed CUDA compilation on Python3.14. The reference therefore uses **/tmp/lidar-kitti-reference-env/bin/python**, an isolated virtualenv inheriting those packages, with **numba-cuda0.30.4**, cuda-bindings13.4.3, cuda-core1.2.1 and cuda-pathfinder1.8.3. CUDA compiler/runtime wheels are isolated under `/tmp/lidar-kitti-cuda12`: nvidia-cuda-nvcc-cu12 **12.9.86**, nvidia-cuda-runtime-cu12 **12.9.79**. The training environment is unchanged.

`CUDA_HOME=/tmp/lidar-kitti-toolkit12` points `nvvm` to the compiler wheel's `nvidia/cuda_nvcc/nvvm` and `lib64` to the runtime wheel's `nvidia/cuda_runtime/lib`. A local `libnvvm.so.4` symlink points to the wheel's `libnvvm.so` for library discovery. These temporary paths must be reprovisioned if removed. Other machines may use their existing compatible CUDA toolkit/backend; they must run the parity gate and record their versions. Python3.10 remains unverified here.

```bash
CUDA_HOME=/tmp/lidar-kitti-toolkit12 NUMBA_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/tmp/lidar-kitti-reference-env/bin/python -m pytest \
  tests/test_kitti_3d_evaluation.py -q
```

The GPU is an RTX4050 Laptop, driver610.57.04. Both reference kernels and all reference parity tests must run; a skipped GPU test does not satisfy R13. Small synthetic workloads can emit Numba occupancy/parallelization warnings. Reports record Python/NumPy/Numba versions, CUDA backend path/runtime and toolkit path.

## Metrics and domain

`evaluate_kitti_3d.py` defaults to `--metric-mode 3d`; `--metric-mode bev` selects the same pinned reference's BEV overlap. **Both R11 and R40 are reported separately in either mode**, in percentages, for Car/Pedestrian/Cyclist × Easy/Moderate/Hard. IoU thresholds are 0.7/0.5/0.5. Moderate macro averages all three classes; AP-9 averages all nine class/difficulty cells. Missing classes remain in the macro and their valid GT counts are recorded; a partial synthetic split is not a benchmark score.

Raw `label_2` annotations retain the full benchmark GT domain, including Van, Person_sitting, DontCare and difficulty metadata. No processed-label ROI filter is applied to reference GT. Detection filtering uses actual image box height. GT and detection height boundaries differ in the pinned implementation; DontCare image-region suppression is specific to the image metric, and is not invented for the BEV/3D modes. Duplicate detections, ignored neighbors and vertical mismatch are covered explicitly.

The existing `evaluate_kitti_bev.py` remains a separate **local ROI BEV R40** diagnostic. Its metric is not interchangeable with reference APBEV. It rejects a 3D configuration. `compare_models.py` preserves its historical `local_bev` default; pass `--metric-mode 3d` or `bev` explicitly for reference comparisons. Task58 functional smoke/presets are complete; long comparison training and final protocol freeze remain task51, pending actual data/comparator audits.

## Decode and calibration

`filter_pred_3d` returns float32 `[N,9]`: **class, score, x, y, z_bottom, length, width, height, yaw**. It reuses group-local class mapping, BEV IQA ranking, classwise BEV-footprint NMS and one global detection cap. Vertical values stay attached to the same selected row. Height is `exp(log_height)` in FP32; non-finite or nonpositive 3D dimensions are rejected even with no candidates. BEV APIs still return `[N,7]` and reject vertical output unless 3D decoding is selected explicitly.

Bottom centers use the full LiDAR-to-rectified-camera transform. Camera yaw inverts the same 2D heading projection used by `prepare_kitti`, preserving roundtrip under nontrivial rectification rotation/translation. This is the detector's upright yaw-only representation; independent pitch/roll and front/back orientation are not predicted. Reference annotations use dimensions **[length,height,width]**, while KITTI text uses **height,width,length**.

Projected boxes use eight upright camera corners and `P2`, clipping the twelve edges to camera **z≥0.1m**, then clipping the resulting rectangle to actual PNG image bounds `[0,width−1] × [0,height−1]`. Fully hidden/off-image/degenerate projections produce no detection annotation. No dummy image bbox is supplied. Alpha is −10 and AOS is disabled because doubled yaw is pi-symmetric. External saved predictions retain their actual bbox and metadata; the wrapper still does not evaluate AOS.

## CLI usage

Original KITTI prediction text files can be evaluated independently of this detector. `--kitti-root` points directly to the original **training/** directory containing `label_2`; the split contains selected frame IDs. Each prediction file has the original sixteen KITTI fields, including score. Ground truth has fifteen fields.

```bash
CUDA_HOME=/tmp/lidar-kitti-toolkit12 /tmp/lidar-kitti-reference-env/bin/python \
  tools/kitti_training_pipeline/evaluate_kitti_3d.py \
  --predictions artifacts/my_run/kitti_predictions \
  --kitti-root data/KITTI/training --split splits/kitti/val.txt \
  --metric-mode 3d --output artifacts/my_run/ap3d_reference.json
```

Replace `--metric-mode 3d` with `bev` for reference APBEV. Replace `--predictions ...` with `--checkpoint ... --config ... --detector-root detector` for a complete **3D** PyTorch checkpoint; its resolved config must supply the processed data root, `data.box_mode=3d` and explicit positive `loss.vertical_loss_weight`. The actual checkpoint path uses Dataset, strict semantic checkpoint loading, full vertical decode, original calibration and image dimensions. A BEV checkpoint cannot invent missing z/height. TensorRT reference inference remains deferred.

Reference JSON records source hashes, split/frame/label/prediction or calibration hashes, model checkpoint SHA256, resolved config and semantic identity, head/quality settings and actual parameter counts. Comparison JSON/CSV/Markdown keep metric mode and both recall samplings separate, including failed-model rows. Reference inference currently makes no latency benchmark claim.

The [execution log](execution_progress.md) records actual verification counts, hardware/software conditions and preservation audit. Dedicated 3D AMP/worker smoke and final runtime presets are implemented in task58; trained AP and any SOTA claim require measured, matched benchmark experiments.
