# Q-OGA với quality target là IoU BEV thật

## Phiên bản đã triển khai

Ứng viên ưu tiên: Q-OGA, detached rotated BEV IoU tại positive peaks, không thêm IQA head. Regression, RDA, corner và EMA/softmax weighting giữ nguyên. Default `q_oga` vẫn dùng MGIoU proxy để bảo toàn baseline và checkpoint cũ.

Classification target với `qcfa_beta=1`:

\[
q=\operatorname{stopgrad}(IoU_{BEV}^{rotated}(B_{pred},B_{GT})),\qquad
L_{+}=|p-q|^2\,BCE(p,q).
\]

Gaussian shoulders vẫn là negatives có penalty discount như CQFL hiện tại. Chỉ positive peak cells có assignment mới cần tính IoU. Offset cùng grid cell có chung origin nên dùng hiệu offset là đủ; width, length và doubled yaw được decode theo quy ước hiện có/evaluator. Việc thay classification làm EMA và learned weights phản ứng theo loss mới; phép kiểm tra bảo toàn regression nói về giá trị các component, không hứa gradient của total loss giữ nguyên.

Tiền lệ quality/classification chung: [Generalized Focal Loss](https://arxiv.org/abs/2006.04388). Đây là cách thích nghi vào detector hiện tại, chưa phải chứng minh AP tăng hoặc đóng góp mới của một paper.

IoU dùng giao polygon thật, với FP32 và normalization theo kích thước mỗi cặp. Implementation PyTorch xử lý các corner nằm trong box đối diện và giao điểm cạnh theo batch trên device. Không dùng Shapely hoặc kéo tensor về CPU trong training. Shapely chỉ dùng làm reference cho tests. Không cần cài MMCV.

## Ba cấu hình, chạy từng lượt

| Config | Quality | Curriculum | Vai trò |
| --- | --- | --- | --- |
| [legacy_s42.json](../../configs/experiments/qoga_quality/legacy_s42.json) | MGIoU proxy | Không | Reference cùng source/recipe khi cần attribution sạch |
| [exact_iou_s42.json](../../configs/experiments/qoga_quality/exact_iou_s42.json) | Rotated IoU thật | Không | Candidate đầu tiên, chỉ đổi quality |
| [exact_iou_warmup8_s42.json](../../configs/experiments/qoga_quality/exact_iou_warmup8_s42.json) | Rotated IoU thật | 8 epoch | Candidate có điều kiện nếu target thấp/recall học yếu |

Các config là config đầy đủ, lấy recipe từ archive Q-OGA rich8/LiteMLA/SG-FPN seed 42: 100 epochs, LR .0007, AdamW, warmup LR 8 epochs, BF16, actual physical batch 16, accumulation 1. Batch 32 trong config archive được đổi về batch 16 thực tế ghi trong run.json. Các mặc định LiteMLA scales [5]/QK norm none được ghi rõ. Model, augmentation và các regression parameters giống nhau giữa ba config; tên experiment khác nhau để không ghi đè artifact. Không có script tự train cả ba.

Chạy từ root repo, trên máy có CUDA và KITTI processed ở `data/kitti/processed`, cùng manifest train/val:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/experiments/qoga_quality/exact_iou_s42.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name qoga_exact_iou_s42 \
  --seed 42 --device cuda --precision bf16 \
  --physical-batch-size 16 --accumulation-steps 1 \
  --num-workers 6 --target-backend numba --compile-model
```

CLI flags target backend, compile và workers phải ghi rõ vì trainer có defaults ghi đè các field tương ứng trong config. Nếu GPU không hỗ trợ BF16, recipe phải đổi và cần legacy control cùng recipe mới để quy gain riêng cho target. Không resume optimizer từ run đã train bằng quality khác rồi xem đó là target-only ablation; các candidate bắt đầu từ cùng seed.

Với notebook standard hiện có, dùng `PRESET="custom"`, `CONFIG_OVERRIDE="configs/experiments/qoga_quality/exact_iou_s42.json"`, đặt Cell 1 `SEED=42`, `EPOCHS=100`, `LEARNING_RATE=.0007`, `PHYSICAL_BATCH_SIZE=16`, `ACCUMULATION_STEPS=1`, `PRECISION="bf16"`, `NUM_WORKERS=6`, `TARGET_BACKEND="numba"`, `COMPILE_MODEL=True`, rồi tính lại `COMPILE_MODEL_ARGUMENT`/`RUNTIME_PROFILE`. Cell 1 ghi đè một số field sau khi đọc config, nên cần kiểm tra `config.resolved.json` của run. `header_use_iou` và `use_iou` phải false. Không đổi sang MS LiteMLA/reparam đồng thời với thí nghiệm quality.

## Curriculum tùy chọn

\[
\lambda_e=\operatorname{clamp}(e/8,0,1),\quad
q_e=(1-\lambda_e)+\lambda_e q_{IoU}.
\]

`e` là epoch zero-based. Epoch hiển thị 1 có lambda=0; epoch hiển thị 9 trở đi lambda=1. 8 epoch là giả thuyết ban đầu, không phải mốc tối ưu đã xác định.

Trainer gọi `set_loss_epoch(criterion, epoch-1)` trước train; validation dùng cùng epoch. Buffer `quality_epoch` được lưu chỉ với config bật curriculum. Manual loop phải gọi `criterion.set_epoch(e)`; thiếu epoch sẽ báo lỗi thay vì âm thầm giữ target 1. Resume cùng config khôi phục state, và trainer đặt lại epoch theo checkpoint để tiếp tục schedule.

Target-only không thêm state-dict key so với legacy. Các run curriculum có thêm buffer; không xem strict resume với config khác là cùng experiment.

## Log và đánh giá

Trong `metrics.jsonl`:

- `training_quality` và `validation.quality_iou_mean`: trung bình IoU thật tại các peak cells đã assigned.
- `quality_iou_zero_fraction`: tỷ lệ peak cells có IoU=0.
- `quality_target_mean`: trung bình target sau curriculum, trước exponent CQFL. Candidate dùng `qcfa_beta=1` để giữ ngữ nghĩa IoU.
- `quality_iou_mix`: lambda hiện tại; target-only luôn 1.
- `quality_peak_count`: tổng peak cells của lượt train/validation; đây không phải số GT object duy nhất khi xảy ra collision.

Means được gộp theo peak counts, không theo batch size; batch rỗng không làm lệch trung bình. Theo dõi cùng recall/AP theo lớp: chỉ tỷ lệ IoU=0 giảm chưa đủ chứng minh detection tốt hơn.

Đánh giá checkpoint cuối epoch 100 trước để đối chiếu archive last. Ví dụ (đổi KITTI_ROOT sang thư mục KITTI object training nguồn của bạn):

```bash
python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name qoga_exact_iou_s42_last --backend pytorch \
  --model artifacts/kitti/qoga_exact_iou_s42/checkpoints/last.pt \
  --config configs/experiments/qoga_quality/exact_iou_s42.json \
  --detector-root detector --kitti-root /path/to/KITTI/object \
  --split splits/kitti/val.txt \
  --score-threshold 0.05 --nms-threshold 0.1 \
  --output artifacts/kitti/qoga_exact_iou_s42/evaluation_validation_last.json \
  --device cuda
```

Đối chiếu mAP Moderate, Pedestrian tổng/30–50 m, Car/Cyclist, recall/FP và training time. Mốc archive: Q-OGA 90.2401% mAP, Pedestrian 82.8850%; Baseline + IQA 91.1235% mAP. Không xem các mốc này là gain đã đạt của candidate. Nếu source/recipe/split khác, dùng legacy config cùng source cho clean comparison.

`best.pt` vẫn chọn bằng minimum validation loss. Curriculum thay objective theo epoch, nên không dùng minimum loss xuyên schedule để gọi checkpoint tốt nhất cho detection. Sau khi so cùng epoch 100, có thể evaluate một số checkpoint đã lưu và chọn bằng AP; không cần thêm seed hoặc train lại để làm việc đó.

Giữ GW-QAL và checkpoint: Cyclist AP archive cao hơn Q-OGA khoảng 1.74 điểm. Chưa sửa GW-QAL trong thay đổi này; candidate Gaussian geometry với cùng exact IoU classification là đối chứng sau khi có kết quả Q-OGA.

## Kiểm tra và giới hạn

Chạy tests bằng môi trường có PyTorch, pytest, numpy và Shapely:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  tests/test_rotated_iou_targets.py tests/test_q_oga_exact_quality.py \
  tests/test_qoga_quality_configs.py tests/test_kitti_config_compatibility.py -q
```

Kiểm tra polygon reference, identity/non-overlap/tangency/containment/edge crossings, box xoay vuông, theta+pi và swapped dimensions, scale/translation, sparse/empty masks, detach, BF16, schedule/checkpoint, finite gradients và model/optimizer integration. Biến môi trường tắt auto-load pytest plugins ROS không liên quan trong máy hiện tại; không ảnh hưởng training.

Các kiểm tra này xác minh tính toán và tích hợp. Máy hiện tại không có CUDA hoạt động hoặc KITTI processed, nên chưa có training/AP/latency GPU mới. Lượt tiếp theo ưu tiên một target-only run seed 42; curriculum, corner removal và fixed task weights là ablation riêng, không bật tất cả cùng lúc.

### Kết quả xác minh ngày 2026-10-05

- Focused suite cho loss, IQA, GW-QAL, model/dataset integration, optimizer, trainer và notebook: **124 passed, 1 skipped, 7 subtests passed** bằng Conda AI_env/PyTorch 2.11.0. Backward/optimizer BF16 là kiểm tra CPU, không phải CUDA.
- 10.000 cặp box ngẫu nhiên, seed 42, so với float64 Shapely: max absolute error **4.1803e-7**, mean absolute error **3.7992e-9**. [Chi tiết](qoga_exact_quality_geometry_check.json). Đây là xác minh số học, không phải AP.
- CLI trainer thực chạy trên hai frame giả ở CPU: train hai epoch, lưu checkpoint, resume epoch 3. Với curriculum test hai epoch, lambda train và validation lần lượt **0, .5, 1**, mỗi epoch có một optimizer update. [Chi tiết](qoga_exact_quality_resume_check.json). Crop/precision/batch ở smoke khác recipe KITTI và không dùng làm kết quả benchmark.
- `train.py --help`, evaluator CLI flags, JSON configs, local document links và `git diff --check` đã xác minh.

Review: chưa thấy lỗi trong các trường hợp đã kiểm tra; legacy default và state keys được kiểm tra bảo toàn. AP thực, hiệu quả curriculum và overhead GPU còn chưa đo. Không sửa GW-QAL, IQA, corner formula hoặc learned weighting trong candidate này.
