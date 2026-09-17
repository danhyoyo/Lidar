# Kế hoạch triển khai và thí nghiệm ProbGeo-UQ

### Goal

Triển khai `SPEC.md` như một extension backward-compatible của Center3D, chạy factorial B0–B3 trên A1 và đánh giá đồng thời detection accuracy, calibration, error ranking và runtime. Không đổi backbone/dataset hoặc thêm phương pháp epistemic trong vòng MVP.

### Assumptions

- KITTI đã được chuẩn bị theo `tools/kitti_training_pipeline/README.md`.
- Core experiment dùng A1 `Legacy35 + CoordAtt + Center3D + sum-FPN`; A4 chỉ được dùng sau khi vượt gate MobileBEV-Lite riêng.
- GPU benchmark chính là RTX 5060 Ti; TensorRT chạy trên cùng CUDA/TensorRT tương thích.
- `splits/kitti/train.txt` và `splits/kitti/val.txt` hiện tại là nguồn duy nhất để tạo train/calibration/test protocol.
- `RTK.md` được `AGENTS.md` tham chiếu nhưng không tồn tại trong checkout ngày 2026-09-14.
- Mọi config thiếu field mới phải giữ hành vi/output/checkpoint legacy.

### Plan

1. Khóa baseline, schema và split hashes
   - Files: `docs/probabilistic_geometry_uq/SPEC.md`, `results/kitti/probabilistic_geometry_uq/protocol.json`
   - Change:
     - Ghi commit, config A1, train/val split và Proposal hash.
     - Khóa thứ tự output, IoU thresholds, score/NMS thresholds và acceptance gates trước khi code.
   - Verify: `git rev-parse HEAD && sha256sum Proposal.xlsx configs/kitti/mobilebev/a1_legacy35_center3d.json splits/kitti/train.txt splits/kitti/val.txt`

2. Tạo calibration/test manifests bất biến
   - Files: `splits/kitti/uq_calibration.txt`, `splits/kitti/uq_test.txt`
   - Change:
     - Shuffle frame-level `val.txt` bằng `random.Random(42)`; 500 frame đầu vào calibration, 997 frame còn lại vào test.
     - Commit hai manifest và hash; không stratify bằng label để tránh logic chia split phụ.
   - Verify: kiểm tra không trùng, không thiếu và union bằng đúng 1,497 dòng của `val.txt` bằng một Python stdlib one-liner.

3. Viết regression checks thất bại cho head contract
   - Files: `tests/test_mobile_bev.py`
   - Change:
     - Xác nhận legacy head vẫn chỉ có bốn outputs với shape cũ.
     - Xác nhận UQ Center3D thêm `log_var(6,H,W)` và variance bias khởi tạo `-2`.
   - Verify: `python3 tests/test_mobile_bev.py --head` phải fail trước implementation vì option/output mới chưa tồn tại.

4. Thêm variance outputs vào head đang có
   - Files: `detector/core/models/heads/cnn.py`, `detector/core/models/model.py`
   - Change:
     - Khi bật UQ, final conv của `offset` và `size` xuất gấp đôi channels rồi split mean/log-variance.
     - Zero-init variance weights, set bias theo config; không thêm tower hoặc dependency.
   - Verify: `python3 tests/test_mobile_bev.py --head` và đếm chênh parameter đúng 102 cho Center3D/in_channels 16.

5. Viết GWD checks trước implementation
   - Files: `tests/test_mobile_bev.py`
   - Change:
     - Thêm cases identical, non-overlap, `θ+π`, width/length swap + `π/2`, square box và boundary-adjacent yaw.
     - Kiểm tra loss/gradient finite và GWD bằng 0 cho các biểu diễn hình học tương đương.
   - Verify: `python3 tests/test_mobile_bev.py --gwd` phải fail vì GWD chưa tồn tại.

6. Cài closed-form 2D GWD trong loss hiện có
   - Files: `detector/core/losses/loss_fn.py`
   - Change:
     - Decode `xy/wl/yaw` từ mean maps, dựng covariance footprint và dùng closed form 2x2.
     - Tính FP32, mask positive cells, clamp radicand số học và trả scalar diagnostic `gwd`.
   - Verify: `python3 tests/test_mobile_bev.py --gwd`

7. Viết heteroscedastic NLL checks trước implementation
   - Files: `tests/test_mobile_bev.py`
   - Change:
     - Kiểm tra công thức tại residual/log-variance biết trước, mask normalization và clamp bounds.
     - Kiểm tra large residual được attenuate nhưng variance inflation vẫn chịu log penalty; NaN/Inf phải raise.
   - Verify: `python3 tests/test_mobile_bev.py --uncertainty-loss` phải fail trước implementation.

8. Cài NLL và bốn loss modes
   - Files: `detector/core/losses/loss_fn.py`
   - Change:
     - Thêm `deterministic`, `gwd`, `heteroscedastic`, `probgeo_uq`; giữ nguyên `baseline`/`uwag`.
     - NLL chỉ áp lên concat `offset+size`; yaw L1 giữ nguyên; không kết hợp UWAG global scales với NLL.
   - Verify: `python3 tests/test_mobile_bev.py --uncertainty-loss && python3 tests/test_mobile_bev.py --loss`

9. Khóa config validation ở điểm dùng chung
   - Files: `detector/core/models/model.py`, `detector/core/losses/loss_fn.py`, `tools/kitti_training_pipeline/common.py`, `tests/test_mobile_bev.py`
   - Change:
     - Reject mismatch giữa loss mode và `predict_log_variance`, bounds sai, weight âm và dimension sai.
     - Config cũ không có field mới phải build/load như trước.
   - Verify: chạy các invalid-config cases và `python3 tests/test_mobile_bev.py --legacy-only`.

10. Giữ decode/AP prefix và append uncertainty
    - Files: `detector/postprocess.py`, `tests/test_mobile_bev.py`
    - Change:
      - Select cùng candidate/NMS indices cho `log_var`; append 6 cột sau 9 box columns.
      - Không dùng uncertainty để đổi score/NMS trong MVP.
    - Verify: `python3 tests/test_mobile_bev.py --decode`; first 9 columns phải byte-equivalent với decoder hiện tại trên cùng tensor.

11. Generalize ONNX raw-output contract
    - Files: `tools/kitti_training_pipeline/export_onnx.py`, `tests/test_mobile_bev.py`
    - Change:
      - Wrapper dùng 4 output names cho legacy và 5 cho UQ.
      - Metadata ghi shape, order, units và prediction schema version.
    - Verify: export một deterministic và một UQ smoke checkpoint; `onnx.checker.check_model` pass và output name/shape khớp config.

12. Generalize TensorRT contract từ metadata
    - Files: `tools/kitti_training_pipeline/build_tensorrt.py`, `tools/kitti_training_pipeline/evaluate_kitti_bev.py`
    - Change:
      - Bỏ set output khóa cứng; suy ra exact expected outputs/shapes từ config/ONNX metadata.
      - Reject engine thừa/thiếu output hoặc shape sai trước inference.
    - Verify: TensorRT smoke test pass cho engine legacy và UQ; engine/config chéo variant phải fail rõ ràng.

13. Giữ evaluator AP tương thích với prediction 15 cột
    - Files: `tools/kitti_training_pipeline/evaluate_kitti_bev.py`, `tests/test_mobile_bev.py`
    - Change:
      - `prediction_box()` đọc 9 cột đầu khi row có 15 cột.
      - Lưu schema 9/15 cột trong JSON/NPZ metadata; AP result của synthetic prefix không đổi.
    - Verify: `python3 tests/test_mobile_bev.py --metrics` với cùng boxes ở schema 9 và 15 phải cho AP giống nhau.

14. Viết uncertainty evaluator tối thiểu
    - Files: `tools/kitti_training_pipeline/evaluate_uncertainty.py`, `tests/test_mobile_bev.py`
    - Change:
      - Reuse ground-truth/IoU helpers; match Moderate TPs, tính six-dimensional residual và point count trong GT box.
      - Tính raw NLL, coverage, ENCE, sharpness, Spearman, risk-coverage/AURC, saturation và sample counts.
    - Verify: synthetic exact-calibration case đạt coverage/NLL mong đợi; inflated/shuffled cases xấu hơn.

15. Thêm calibrator và baseline rẻ
    - Files: `tools/kitti_training_pipeline/evaluate_uncertainty.py`, `tests/test_mobile_bev.py`
    - Change:
      - Fit per-dimension variance scale trên calibration set.
      - Fit constant variance và `range + point_count` bằng NumPy `lstsq`; thêm score-only, shuffled và oracle controls.
    - Verify: synthetic heteroscedastic data cho learned/calibrated model thắng constant; test split không được gọi trong hàm fit.

16. Tạo bốn config factorial
    - Files: `configs/kitti/probgeo_uq/b0_deterministic.json`, `b1_gwd.json`, `b2_heteroscedastic.json`, `b3_probgeo_uq.json`
    - Change:
      - Copy A1 và chỉ đổi `loss.name`, `gwd_weight`, `predict_log_variance`, variance init/bounds và note.
      - Giữ data, augmentation, optimizer, epochs, precision và architecture giống nhau.
    - Verify: `python3 -m json.tool` cho cả bốn file; canonical diff chỉ chứa allowlist fields.

17. Chạy full CPU checks trước GPU
    - Files: `tests/test_mobile_bev.py`, `detector/`, `tools/kitti_training_pipeline/`
    - Change:
      - Không thêm test framework; nối các check mới vào runner hiện có.
      - Compile toàn bộ Python và sửa regression legacy trước smoke training.
    - Verify: `python3 tests/test_mobile_bev.py && python3 -m compileall -q detector tools/kitti_training_pipeline`

18. Smoke train B0–B3
    - Files: `artifacts/kitti/probgeo_uq_smoke/`
    - Change:
      - Mỗi variant chạy 1 epoch, 8 train batches, 4 calibration batches, seed 42.
      - Kiểm tra finite loss/gradient, log-var saturation, checkpoint save/reload và decoded schema.
    - Verify: chạy `train.py` với `--epochs 1 --max-train-batches 8 --max-val-batches 4`; cả bốn run có checkpoint/config hash và không NaN/Inf.

19. Chạy screening không dùng làm kết luận
    - Files: `artifacts/kitti/probgeo_uq_screen/`
    - Change:
      - B0–B3, seed 42, 30 epochs; chọn checkpoint trên 500-frame calibration manifest.
      - Chỉ dùng để phát hiện divergence/saturation/config error; thay đổi method thì restart cả factorial bị ảnh hưởng.
    - Verify: evaluation calibration đủ 500 frame, test manifest chưa được dùng, mọi variant có cùng số update.

20. Chạy factorial xác nhận ba seed
    - Files: `artifacts/kitti/probgeo_uq_final/`
    - Change:
      - B0–B3 với seeds 42,43,44, 100 epochs.
      - Chọn checkpoint theo metric đã khóa; inference full val, calibration 500 và test 997.
    - Verify: đủ 12 checkpoint/evaluation bundles; mỗi bundle chứa config/checkpoint/split hash và finite outputs.

21. Tổng hợp calibration và statistical evidence
    - Files: `tools/kitti_training_pipeline/compare_models.py`, `results/kitti/probabilistic_geometry_uq/summary.json`, `summary.md`
    - Change:
      - Báo mean ± SD qua seed, paired frame bootstrap 2,000 lần seed 42 cho B1-B0, B2-B0 và B3-B0.
      - Tách AP, raw/calibrated UQ, baselines, range/point-count strata và overhead.
    - Verify: cùng inputs tạo summary byte-identical; totals khớp 4 variants × 3 seeds × 997 test frames.

22. Export và benchmark model đã khóa
    - Files: `artifacts/kitti/probgeo_uq_deploy/`, `results/kitti/probabilistic_geometry_uq/summary.md`
    - Change:
      - Export B0/B3 ONNX và TensorRT FP16; đo PyTorch FP32/TensorRT FP16 output parity, AP, latency, memory và bytes.
      - Không rescore/NMS theo uncertainty trong benchmark chính.
    - Verify: B3 đáp ứng parameter/latency gates; first 9 mean outputs trong tolerance và AP giảm không quá 0.2 điểm sau FP16.

23. Chạy replication có điều kiện trên A4
    - Files: `configs/kitti/probgeo_uq/a4_b0_deterministic.json`, `a4_b3_probgeo_uq.json`, `results/kitti/probabilistic_geometry_uq/summary.md`
    - Change:
      - Chỉ tạo/chạy hai config nếu A4 đã vượt MobileBEV-Lite gate.
      - Giữ nguyên loss/UQ protocol để kiểm tra kết quả có phụ thuộc A1 hay không.
    - Verify: ba seed A4-B0/B3 dùng cùng manifests, checkpoint rule và UQ evaluator với core experiment.

24. Khóa claim cuối
    - Files: `docs/probabilistic_geometry_uq/SPEC.md`, `results/kitti/probabilistic_geometry_uq/summary.md`, `tools/kitti_training_pipeline/README.md`
    - Change:
      - Chỉ gọi “aleatoric localization uncertainty” nếu thắng baseline và coverage gate.
      - Không gọi GWD covariance là predictive uncertainty, không claim epistemic/full-heading/full-3D Gaussian/SOTA/official KITTI test.
    - Verify: mọi claim số học liên kết được tới một result JSON, config, checkpoint và split hash.

### Risks & mitigations

- **Training non-finite:** loss FP32, bounded log-variance, finite loss/gradient checks; dừng và lưu epoch/batch thay vì `nan_to_num`.
- **Variance collapse/explosion:** khởi tạo `-2`, log penalty, saturation logging và raw/calibrated reports.
- **GWD numerical instability:** closed-form 2x2, positive-size decode, radicand clamp và boundary gradient tests.
- **Double weighting:** heteroscedastic modes không dùng UWAG global log-scales.
- **Calibration leakage:** manifests commit cố định; calibrator chỉ nhận calibration predictions; test chỉ dùng final report.
- **Weak novelty:** định vị contribution là lightweight Center3D integration + strict UQ evaluation; không nhận GWD/NLL là ý tưởng mới.
- **Geometry-only confounder:** constant, range+point-count, score-only và shuffled controls là bắt buộc.
- **Yaw limitation:** giữ claim orientation modulo `π`; chỉ thêm von Mises/direction head nếu nghiên cứu chuyển sang heading/motion.
- **Dense-map bias:** detection-level evaluation là endpoint chính; chỉ thêm object-balanced training khi diagnostic xác nhận.
- **Compute cost:** dừng sau screening nếu B2/B3 không học variance hữu ích; không chạy A4 replication trước gate.

### Rollback plan

- Field mới mặc định tắt; config/checkpoint legacy vẫn tạo bốn outputs và decode 7/9 cột.
- Nếu GWD thất bại, giữ B2; nếu NLL thất bại, giữ B1; nếu cả hai thất bại, quay về B0/UWAG mà không đổi data/backbone.
- Không ghi đè artifacts/results cũ. Mọi output mới nằm dưới `artifacts/kitti/probgeo_uq_*` và `results/kitti/probabilistic_geometry_uq/`.
- Revert theo bốn commit nhỏ: head/config, losses, decode/export/runtime, UQ evaluation. Split manifests và spec có thể giữ để audit.
