# Kế hoạch triển khai và thí nghiệm MobileBEV-Lite

### Goal

Triển khai đặc tả `SPEC.md` bằng thay đổi backward-compatible, xác minh hộp 3D và chạy ablation A1-A4 có thể tái lập. Không triển khai camera, temporal, sparse convolution, distillation hoặc quantization trong vòng này.

### Assumptions

- KITTI đã được chuẩn bị theo README và có thể truy cập ở một đường dẫn cục bộ.
- GPU dùng benchmark chính là RTX 5060 Ti; TensorRT chỉ chạy trong môi trường CUDA/TensorRT tương thích.
- Baseline tham chiếu là commit `24deca8`, config `configs/kitti/kitti_uwag_coordatt_aug.json` và split đã commit.
- `RTK.md` được AGENTS.md tham chiếu nhưng không tồn tại trong checkout ngày 2026-09-13.
- Mọi thay đổi giữ default cũ để config hiện tại vẫn load được.

### Plan

1. Khóa baseline và metric contract
   - Files: `docs/mobile_bev_lightweight/SPEC.md`, `results/kitti/uwag_coordatt_aug_bf16_seed42/evaluation_best_val.json`
   - Change:
     - Ghi lại commit/config/split hash và các metric baseline trong manifest thí nghiệm.
     - Khóa thứ tự output, IoU threshold, checkpoint selection và latency boundary trước khi code.
   - Verify: `git rev-parse HEAD && sha256sum configs/kitti/kitti_uwag_coordatt_aug.json splits/kitti/train.txt splits/kitti/val.txt`

2. Tạo self-check thất bại cho RichBEV và box 3D
   - Files: `tests/test_mobile_bev.py`
   - Change:
     - Thêm synthetic points kiểm tra 8 kênh, ô rỗng, band boundary, intensity scale và density clipping.
     - Thêm round-trip target/decode cho một hộp 3D và IoU3D identical/disjoint.
   - Verify: `python3 tests/test_mobile_bev.py` phải fail vì API mới chưa tồn tại.

3. Hợp nhất đường voxelization bị trùng
   - Files: `detector/core/datasets/utils_1/preprocess.py`, `detector/core/datasets/dataset.py`
   - Change:
     - Giữ một implementation legacy trong `preprocess.py` và để `Dataset` gọi helper này.
     - Xóa bản sao logic trong `Dataset.voxelize` sau khi output legacy khớp.
   - Verify: `python3 tests/test_mobile_bev.py --legacy-only` và so sánh byte-for-byte output cũ/mới trên synthetic input.

4. Cài RichBEV-8 bằng NumPy sẵn có
   - Files: `detector/core/datasets/utils_1/preprocess.py`, `detector/core/datasets/dataset.py`
   - Change:
     - Thêm `encode_bev(points, geometry, bev_encoding)` với `binary_slices` và `rich8`.
     - Dùng `bincount`, `add.at`/`maximum.at`; validate geometry, density norm và intensity scale tại trust boundary.
   - Verify: `python3 tests/test_mobile_bev.py --encoder`

5. Suy ra input channel từ encoder
   - Files: `tools/kitti_training_pipeline/common.py`, `detector/core/models/model.py`, `detector/core/models/backbones/mobilepixor_coordinate_attention.py`
   - Change:
     - `input_shape()` trả 35 hoặc 8 theo config.
     - Backbone nhận `input_channels=35` mặc định; config cũ không đổi hành vi.
   - Verify: `python3 tests/test_mobile_bev.py --shapes`

6. Thêm SG-FPN khởi tạo tương đương
   - Files: `detector/core/models/backbones/mobilepixor_coordinate_attention.py`
   - Change:
     - Thêm hai depthwise gate 32/16 channels sau lateral C4/C3.
     - Zero-init gate và chỉ bật bằng `scale_gated_fpn=true`.
   - Verify: `python3 tests/test_mobile_bev.py --gates` phải xác nhận output gate-on lúc init gần bằng sum-FPN với `atol=1e-6`.

7. Mở rộng head thành Center3D tối giản
   - Files: `detector/core/models/heads/cnn.py`, `detector/core/models/model.py`
   - Change:
     - Với `box_encoding=center3d`, đổi `offset` từ 2 lên 3 và `size` từ 2 lên 3.
     - Giữ bốn output names và legacy dimensions khi config không bật.
   - Verify: `python3 tests/test_mobile_bev.py --head`

8. Tạo target `z_center` và `log(h)`
   - Files: `detector/core/datasets/dataset.py`, `tools/kitti_training_pipeline/train.py`, `tools/kitti_training_pipeline/evaluate_kitti_bev.py`
   - Change:
     - Mở rộng map offset/size khi `box_encoding=center3d`.
     - Chuyển label KITTI bottom-center thành `z_center=z_bottom+h/2`.
     - Truyền một `box_encoding` từ model config vào mọi Dataset; mặc định `bev` cho config cũ.
   - Verify: `python3 tests/test_mobile_bev.py --targets`

9. Giữ UWAG và geometric loss tương thích 2D/3D
   - Files: `detector/core/losses/loss_fn.py`, `tests/test_mobile_bev.py`
   - Change:
     - L1 nhận 2 hoặc 3 channels; geometric term chỉ dùng `xy/wl` một cách tường minh.
     - Thêm finite check cho loss components và output dimensions.
   - Verify: `python3 tests/test_mobile_bev.py --loss`

10. Decode hộp 3D mà không đổi NMS
    - Files: `detector/postprocess.py`, `tests/test_mobile_bev.py`
    - Change:
      - Decode `z_center/h` khi output có 3 channels.
      - NMS tiếp tục nhận rotated BEV `(x,y,w,l,yaw)`; legacy vẫn trả 7 cột, Center3D trả 9 cột.
    - Verify: `python3 tests/test_mobile_bev.py --decode`

11. Mở evaluator cho BEV và 3D AP
    - Files: `tools/kitti_training_pipeline/evaluate_kitti_bev.py`, `tests/test_mobile_bev.py`
    - Change:
      - GroundTruth giữ `z_center/h`; generalize IoU sang `space=bev|3d`.
      - Ghi `accuracy.bev`, `accuracy.3d` và distance bands, không ghi đè schema cũ cho legacy result.
    - Verify: `python3 tests/test_mobile_bev.py --metrics`

12. Thêm checkpoint sweep theo metric đã khóa
    - Files: `tools/kitti_training_pipeline/select_checkpoint.py`, `tools/kitti_training_pipeline/evaluate_kitti_bev.py`
    - Change:
      - Evaluate các checkpoint được lưu mỗi 5 epoch trên cùng validation split.
      - Chọn duy nhất checkpoint có mAP 3D Moderate cao nhất; tie-break bằng mean 3D AP-9 rồi epoch sớm hơn.
    - Verify: chạy selector hai lần trên cùng checkpoint directory phải tạo cùng epoch, metric và checkpoint SHA-256.

13. Thêm bốn config ablation
    - Files: `configs/kitti/mobilebev/a1_legacy35_center3d.json`, `configs/kitti/mobilebev/a2_rich8_center3d.json`, `configs/kitti/mobilebev/a3_legacy35_sgfpn_center3d.json`, `configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json`
    - Change:
      - Chỉ khác ba field `bev_encoding`, `scale_gated_fpn`, `box_encoding`.
      - Giữ mọi training/data field giống config baseline.
    - Verify: `python3 -m json.tool configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json >/dev/null && diff -u configs/kitti/mobilebev/a1_legacy35_center3d.json configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json`

14. Cập nhật export và TensorRT shape contract
    - Files: `tools/kitti_training_pipeline/export_onnx.py`, `tools/kitti_training_pipeline/build_tensorrt.py`, `tools/kitti_training_pipeline/evaluate_kitti_bev.py`
    - Change:
      - Giữ output names `cls,offset,size,yaw`; metadata ghi output shapes và encoding.
      - Bắt lỗi engine cũ có shape không tương thích thay vì chạy sai.
    - Verify: `python3 tools/kitti_training_pipeline/export_onnx.py --config configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json --checkpoint /absolute/path/to/smoke.pt --detector-root detector --output /tmp/mobilebev_smoke.onnx --device cuda`

15. Chạy full CPU self-check và syntax check
    - Files: `tests/test_mobile_bev.py`, `detector/`, `tools/kitti_training_pipeline/`
    - Change:
      - Không thêm framework test hoặc dependency mới.
      - Sửa mọi regression legacy trước khi dùng GPU.
    - Verify: `python3 tests/test_mobile_bev.py && python3 -m compileall -q detector tools/kitti_training_pipeline`

16. Smoke train A1-A4
    - Files: `artifacts/kitti/mobilebev_smoke/`
    - Change:
      - Chạy mỗi variant trên subset cố định nhỏ, 1 epoch; kiểm tra finite loss, gradient và checkpoint reload.
      - Không dùng các số này làm kết quả nghiên cứu.
    - Verify: `python3 tools/kitti_training_pipeline/train.py --config configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json --detector-root detector --output-root artifacts/kitti --run-name mobilebev_a4_smoke_seed42 --epochs 1 --max-train-batches 8 --max-val-batches 4 --num-workers 2`

17. Benchmark encoder và model trước training dài
    - Files: `tools/kitti_training_pipeline/evaluate_kitti_bev.py`, `artifacts/kitti/mobilebev_bench/`
    - Change:
      - Đo input bytes, preprocess, H2D, model và peak VRAM trên cùng 100 frame cho A1/A4.
      - Dừng nếu RichBEV làm E2E chậm hơn A1; tối ưu aggregation trước khi training.
    - Verify: chạy evaluator A1 và A4 với `--max-frames 100 --warmup-frames 10`, rồi xác nhận cùng frame IDs và latency boundary trong hai JSON.

18. Chạy screening 30 epoch seed 42
    - Files: `artifacts/kitti/mobilebev_screen/`
    - Change:
      - Chạy A1-A4 để phát hiện failure mode, không chọn lại metric hoặc thay spec dựa trên variant thắng.
      - Ghi rõ mọi thay đổi bắt buộc nếu có và restart toàn bộ variant bị ảnh hưởng.
    - Verify: mỗi run có checkpoint, config hash, finite log và evaluation đủ 1,497 frame.

19. Chạy ablation xác nhận 100 epoch, 3 seed
    - Files: `artifacts/kitti/mobilebev_final/`
    - Change:
      - Chạy A1-A4 với seed 42, 43, 44.
      - Chọn checkpoint bằng mAP 3D Moderate đã khóa; không chọn theo loss nếu hai tiêu chí khác nhau.
    - Verify: đủ 12 evaluation JSON, mỗi file có 1,497 frame và hash checkpoint/config/split.

20. Tổng hợp thống kê và bootstrap
    - Files: `tools/kitti_training_pipeline/compare_models.py`, `results/kitti/mobilebev_lightweight/summary.json`, `results/kitti/mobilebev_lightweight/summary.md`
    - Change:
      - Tổng hợp mean ± standard deviation qua seed.
      - Recompute paired frame bootstrap 2,000 lần, seed 42, cho A4-A1; báo CI accuracy và bảng efficiency.
    - Verify: chạy cùng input hai lần phải tạo `summary.json` byte-identical; tổng frame/seed/variant phải khớp 1,497/3/4.

21. Kiểm tra novelty và claim
    - Files: `docs/mobile_bev_lightweight/SPEC.md`, `results/kitti/mobilebev_lightweight/summary.md`
    - Change:
      - So sánh trực tiếp với PIXOR, CenterPoint, EfficientDet/BiFPN và TriBand-BEV.
      - Chỉ giữ claim được dữ liệu A1-A4 hỗ trợ; bỏ “SOTA/first/edge-ready” nếu chưa có bằng chứng.
    - Verify: mọi con số trong phần kết luận truy ngược được tới một JSON kết quả và một config/hash.

22. Export và benchmark TensorRT FP16 cho model được khóa
    - Files: `tools/kitti_training_pipeline/export_onnx.py`, `tools/kitti_training_pipeline/build_tensorrt.py`, `tools/kitti_training_pipeline/compare_models.py`, `artifacts/kitti/mobilebev_deploy/`
    - Change:
      - Export A1/A4 cùng checkpoint rule; build engine trên cùng máy.
      - So PyTorch FP32 với TensorRT FP16 về output parity, AP và latency.
    - Verify: `python3 tools/kitti_training_pipeline/compare_models.py --model a4_pytorch=pytorch:/absolute/path/to/a4.pt --model a4_tensorrt=tensorrt:/absolute/path/to/a4.engine --config configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json --detector-root detector --kitti-root /absolute/path/to/KITTI/object --split splits/kitti/val.txt --output-dir artifacts/kitti/mobilebev_deploy/compare --device cuda --fail-on-model-error`

23. Final research gate
    - Files: `results/kitti/mobilebev_lightweight/summary.md`, `docs/mobile_bev_lightweight/SPEC.md`
    - Change:
      - Đánh dấu Go, Partial go hoặc No-go theo threshold đã khóa.
      - Liệt kê rõ `not_evaluated` cho official test hoặc edge hardware chưa chạy.
    - Verify: một người khác có thể dùng README/config/hash để lặp lại A1 và A4 mà không hỏi thêm tham số.

### Risks & mitigations

- **Non-finite training:** smoke và finite guard trước; không che lỗi bằng `nan_to_num`. Giảm precision hoặc thêm clipping chỉ sau khi xác định nguồn.
- **Preprocessing bottleneck:** benchmark 100 frame trước training dài; dùng NumPy reduction, không loop theo cell.
- **Mất vertical detail:** gate No-go RichBEV đã khóa; chỉ thử RichBEV-11 sau khi A2 thất bại.
- **Evaluator sai:** synthetic IoU/round-trip và đối chiếu official KITTI devkit trước claim.
- **Gate không học:** log mean/std/saturation của G3/G4; nếu bão hòa hoặc AP không tăng thì xóa gate.
- **Selection bias:** khóa metric, seed và threshold trước run; lưu tất cả checkpoint/evaluation.
- **Novelty overlap:** định vị contribution là tích hợp tối giản vào IRBGHR-MobilePIXOR và direct 3D regression, không nhận RichBEV/high-resolution fusion là ý tưởng đầu tiên.
- **Checkpoint/engine nhầm variant:** bắt buộc config/checkpoint/engine SHA-256 và shape metadata.

### Rollback plan

- Mọi field mới có default legacy, nên config hiện tại tiếp tục dùng 35-channel BEV, sum-FPN và BEV head.
- Nếu RichBEV thất bại, dùng A3; nếu SG-FPN thất bại, dùng A2; nếu cả hai thất bại, giữ A1 làm baseline 3D.
- Không ghi đè checkpoint/result cũ. Artifact mới nằm dưới `artifacts/kitti/mobilebev_*` và `results/kitti/mobilebev_lightweight/`.
- Revert theo từng commit nhỏ: encoder, gates, head/evaluator, configs. Không cần reset dữ liệu hoặc xóa baseline.
