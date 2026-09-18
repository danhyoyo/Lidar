# Tích hợp MobileBEV + ProbGeo-UQ thành C1/C2

### Goal

Port đầy đủ B0–B3 vào `integrated_P12` theo từng feature hunk, không merge nguyên branch đang xung đột. Thêm:

- **C0:** A1 architecture + B3 loss, control chuẩn hóa.
- **C1:** A3 architecture + B3 loss, accuracy profile.
- **C2:** A4 architecture + B3 loss, deployment profile.

Dữ liệu hiện tại ủng hộ lựa chọn A3/A4/B3, nhưng A-series và B-series dùng split khác nhau và mới có seed 42 nên không được so trực tiếp. :codex-file-citation{path="/home/duyennh/AI_projects/research_lidar/Lidar/outputs/2026-09-18-a0-b3-comparison/model_comparison_A0-A4_B0-B3.xlsx" purpose="source" artifact_kind="workbook" sheet="Overview" range="A6:R23"}

### Assumptions

- Round C dùng BF16, 50 epoch, physical batch 16, accumulation 1, seeds `{42,43,44}`.
- Chọn checkpoint trên `uq_calibration` 500 frame; chỉ sau đó đánh giá `uq_test` 997 frame và fit calibrator.
- Benchmark deployment trên cùng NVIDIA L4; khác phần cứng thì chỉ báo accuracy/UQ, chưa áp gate latency.
- Giữ nguyên B0–B3 lịch sử. C0–C2 có config riêng dưới `configs/kitti/probgeo_uq/`.
- B3 seed 42 chỉ được tái sử dụng nếu `run.json`, checkpoint hash và prediction hash xác nhận đúng 50 epoch/effective batch 16; thiếu hoặc lệch metadata thì train lại C0 seed 42.
- Không thêm uncertainty-aware NMS, epistemic UQ, distillation, quantization hay dependency mới.

### Plan

1. **Khóa regression contracts trước khi port**
   - Files: `tests/test_mobile_bev.py`, `tests/test_regressions.py`, `tests/test_standard_training_notebook.py`
   - Change:
     - Thêm checks cho GWD, heteroscedastic NLL, config validation, `log_var(6)`, decoder 15 cột, ONNX output thứ năm và legacy compatibility.
     - Kiểm tra C0–C2 chỉ khác các field architecture/note được phép.
   - Verify: các check ProbGeo mới phải fail trên code hiện tại; checks legacy vẫn pass.

2. **Port core B0–B3 vào code hiện tại**
   - Files: `detector/core/losses/loss_fn.py`, `detector/core/models/heads/cnn.py`, `detector/core/models/model.py`
   - Change:
     - Thêm loss modes `deterministic`, `gwd`, `heteroscedastic`, `probgeo_uq`.
     - Mở rộng `Header` bằng `predict_log_variance=False` và `initial_log_variance=-2.0`; khi bật, output thêm `log_var` sáu kênh.
     - Tính NLL/GWD ở FP32, clamp variance/size và fail rõ khi gặp NaN/Inf.
   - Verify: `python3 tests/test_mobile_bev.py --head --gwd --uncertainty-loss --loss`.

3. **Giữ contract inference/deployment tương thích**
   - Files: `detector/postprocess.py`, `tools/kitti_training_pipeline/export_onnx.py`, `tools/kitti_training_pipeline/build_tensorrt.py`
   - Change:
     - Legacy tiếp tục trả 7/9 cột; UQ append sáu raw log-variance thành 15 cột mà không đổi score/NMS.
     - ONNX/TensorRT chấp nhận bốn output legacy hoặc output thứ năm `log_var`, với validation shape từ config/metadata.
   - Verify: `python3 tests/test_mobile_bev.py --decode --export-contract --metrics`; chín cột đầu phải giữ nguyên.

4. **Port protocol và evaluator Proposal 2**
   - Files: `tools/kitti_training_pipeline/evaluate_uncertainty.py`, `tools/kitti_training_pipeline/common.py`, `splits/kitti/uq_*.txt`
   - Change:
     - Port calibrator, constant/range-point baselines, NLL, coverage, ENCE, AURC, AUROC và saturation metrics.
     - Port B0–B3 configs/docs/protocol nhưng giữ implementation hiện tại làm nguồn cho các bug fix dùng chung.
   - Verify: hash/union của hai split bằng đúng `val.txt`; `python3 tests/test_mobile_bev.py --probgeo-config --uncertainty-evaluator --probgeo-review`.

5. **Tạo ma trận Round C tối thiểu**
   - Files: `configs/kitti/probgeo_uq/c0_a1_probgeo_uq.json`, `c1_a3_probgeo_uq.json`, `c2_a4_probgeo_uq.json`
   - Change:
     - C0: `binary_slices`, sum-FPN.
     - C1: `binary_slices`, SG-FPN.
     - C2: `rich8`, SG-FPN.
     - Cả ba dùng cùng B3 loss/UQ, AdamW + MultiStepLR `[35, 45]`, augmentation, split, 50 epoch và effective batch 16.
   - Verify: expected parameters lần lượt khoảng `600,473`, `600,953`, `593,177`; C0/C1 có 35 input channels, C2 có 8.

6. **Sửa provenance ở điểm dùng chung**
   - Files: `tools/kitti_training_pipeline/train.py`, `3D_Lidar_Object_Detection_Notebook_standard.ipynb`
   - Change:
     - Ghi CLI overrides thực tế vào `config.resolved.json` và checkpoint config: epoch, batch, accumulation, precision, seed.
     - Notebook đăng ký C0–C2; chọn split/evaluator theo capability trong config thay vì prefix `B`.
   - Verify: smoke run với CLI override phải ghi đúng 50/16/1 trong resolved config; `python3 tests/test_standard_training_notebook.py`.

7. **Chạy verification đầy đủ**
   - Verify:
     - `python3 tests/test_mobile_bev.py`
     - `python3 tests/test_regressions.py`
     - `python3 tests/test_evaluation_table.py`
     - `python3 tests/test_standard_training_notebook.py`
     - `python3 -m compileall -q detector tools/kitti_training_pipeline`
   - Chạy trong environment PyTorch/CUDA của dự án; system Python hiện tại không có `torch`.

8. **Smoke và final runs**
   - Chạy C0/C1/C2 trước với 1 epoch, 8 train batches và 4 calibration batches.
   - Sau smoke, chạy `{C0,C1,C2} × {42,43,44}`; có thể bỏ C0-seed42 chỉ khi artifact cũ vượt provenance/parity gate.
   - Mỗi run phải có config/checkpoint/split hash, selection record, calibration/test predictions và uncertainty report; không ghi đè artifact A/B cũ.

9. **Tổng hợp thống kê và khóa hai profile**
   - Files: `tools/kitti_training_pipeline/summarize_round_c.py`, `results/kitti/probabilistic_geometry_uq/round_c/summary.{json,md}`
   - Change:
     - Báo mean ± SD qua ba seed.
     - Paired bootstrap 2.000 lần, RNG 42: cùng frame sample cho ba seed, tính AP delta từng seed rồi lấy trung bình.
   - Gates:
     - **C1 accuracy:** lower bound 95% CI của `C1 − C0` cho 3D mAP Moderate phải `> 0`.
     - **C2 deployment:** lower bound 95% CI của `C2 − C1` phải `> −1.0 pp`; E2E mean nhanh hơn ≥20% và input bytes giảm ≥50%.
     - C1/C2 chỉ giữ nhãn ProbGeo-UQ nếu calibrated NLL thắng constant và range+point-count, AURC thắng score/shuffled, và aggregate coverage gap không quá 5 điểm phần trăm.
   - Verify: chạy summarizer hai lần phải tạo kết quả byte-identical.

### Risks & mitigations

- Branches xung đột rộng: port UQ-specific hunks, không merge/cherry-pick nguyên branch.
- Batch 16 OOM: không hạ riêng từng model; nếu phải đổi batch thì khóa protocol mới và chạy lại toàn bộ C0–C2.
- Leakage: `uq_test` không được dùng để chọn checkpoint, weight, threshold hoặc calibrator.
- Artifact B3 cũ thiếu provenance: tự động loại khỏi reuse và chạy lại C0 seed 42.
- Profile không đạt gate: báo kết quả âm/trade-off, không tune lại trên test.

### Rollback plan

- Các field UQ mặc định tắt nên checkpoint/output legacy vẫn giữ nguyên.
- Tách commit thành core loss/head, runtime/evaluator, configs/notebook và result tooling để có thể revert độc lập.
- Artifact/result Round C dùng thư mục mới; không sửa hoặc xóa kết quả A0–A4/B0–B3.
