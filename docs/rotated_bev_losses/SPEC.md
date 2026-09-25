# Đặc tả nghiên cứu loss hình học cho hộp xoay BEV

Trạng thái: **đã duyệt ngày 2026-09-22**.

Tài liệu này đặc tả chương trình **B-series** đánh giá loss hình học xoay độc
lập với ablation kiến trúc A-series. B0 đóng băng kiến trúc, data và training
hyperparameters từ A4 ban đầu, đồng thời giữ legacy UWAG; B1-B5 chỉ thay
objective huấn luyện trên cùng snapshot đó.
Không sửa config hoặc tài liệu MobileBEV cũ. Loss chỉ tồn tại lúc huấn luyện;
hợp đồng suy luận `cls`, `offset`, `size`, `yaw` không đổi.

## 1. Mục tiêu

- Giữ B0 làm mốc lịch sử A4 + legacy UWAG + auxiliary 0.2.
- So sánh controlled baseline B1 masked-L1 với B2 full KFIoU, B3
  ProbIoU-BD/Hellinger, B4 KLD và B5 MGIoU nếu vượt cổng khả thi.
- Đánh giá ổn định BF16, gradient, runtime và VRAM bên cạnh BEV AP.
- Giữ nguyên target encoding, model head, decoder, ONNX và TensorRT contract.
- Hoãn mọi hybrid/weighting B6+, QFL/VFL, GradNorm và PCGrad.

## 2. Hiện trạng repository

Target hiện tại gồm:

- `cls`: Gaussian center heatmap cho ba lớp.
- `offset`: `(dx, dy)` theo mét.
- `size`: `(log(width), log(length))`.
- `yaw`: `(cos(2 theta), sin(2 theta))`.

Regression mask phủ một đĩa quanh tâm object, không chỉ một center pixel.
Postprocess giải mã `length = exp(log_length)`, `width = exp(log_width)` và
`theta = 0.5 atan2(sin(2 theta), cos(2 theta))`.

Baseline hiện tại là modified focal loss cộng masked-L1 cho offset, size và
yaw. UWAG lịch sử tối ưu:

```text
sum_i(exp(-s_i) L_i + s_i), i in {cls, offset, size, yaw}
```

rồi cộng auxiliary yaw-aware axis-aligned footprint loss với trọng số 0.2.
Auxiliary này không phải rotated IoU thật và không có positional-overlap
gradient khi hai footprint không giao nhau.

README có ghi nhận một run non-finite ở epoch 81, nhưng repository không có đủ
artifact để xác minh độc lập. Đây chỉ là quan sát lịch sử cần tái lập.

Evaluator là KITTI-style rotated BEV AP R40 cục bộ, không phải official KITTI
devkit hoặc hidden test server. Đặc biệt, hành vi `DontCare` chưa được chứng
minh tương đương devkit chính thức.

## 3. Ranh giới nghiên cứu và kiến trúc triển khai

Tài liệu/config mới nằm riêng:

```text
docs/rotated_bev_losses/{SPEC.md,PLAN.md}
configs/kitti/loss_ablation/b0_...json ... b5_...json
```

Các B config là snapshot đầy đủ, không kế thừa động và không sửa
`configs/kitti/mobilebev/`. Trùng JSON nhỏ được chấp nhận để giữ artifact bất
biến; test phải xác nhận mọi trường ngoài `loss`, `note` và experiment ID giống
B0.

`loss_fn.py` chỉ điều phối classification, regression, reduction và weighting.
Mỗi objective hình học là một `torch.nn.Module` rõ ràng, để có constructor,
state và forward contract dễ kiểm thử:

- `KFIoULoss`, `ProbIoULoss` và `KLDLoss` nằm trong
  `gaussian_geometry_loss.py`;
- `MGIoULoss` nằm trong `mgiou_loss.py` và chỉ được expose sau khi B5 gate
  pass.

Các module có thể dùng helper thuần cho covariance, projection và reduction,
nhưng không tạo registry/factory/class hierarchy mới và không thêm dependency.
Docstring/comment trong module phải giải thích công thức, quy ước tensor và
edge case số học; mỗi module cũng ghi link bài báo và implementation/repo tham
khảo giống cách `focal_loss.py` ghi nguồn. Giữ `focal_loss.py` và `l1_loss.py`
hiện có.

## 4. Giải mã hộp và covariance

Tại mỗi pixel có `reg_mask = 1`:

```text
x = grid_x_metric + dx
y = grid_y_metric + dy
w = exp(log_w)
l = exp(log_l)
```

Vì prediction và target dùng cùng grid cell, center displacement trong loss là
`pred_offset - target_offset`; không cần truyền toàn bộ grid geometry vào
criterion.

Yaw dùng doubled angle. Covariance được dựng trực tiếp, không cần `atan2`:

```text
a = l^2 / k
b = w^2 / k
Sigma_xx = (a+b)/2 + (a-b) cos(2 theta)/2
Sigma_yy = (a+b)/2 - (a-b) cos(2 theta)/2
Sigma_xy =             (a-b) sin(2 theta)/2
```

Điểm bắt buộc: tại `theta = 0`, phương sai trục x tỷ lệ với `length^2`, không
phải `width^2`.

- KFIoU và KLD dùng `k = 4`.
- ProbIoU dùng `k = 12`.

Prediction yaw vector được chuẩn hóa với epsilon. Biểu diễn doubled angle bảo
đảm `theta` và `theta + pi` cho cùng covariance.

Trước mọi `solve`/`slogdet`, covariance được regularize theo scale của chính
hộp, không cộng một jitter tuyệt đối giống nhau cho mọi kích thước:

```text
s = max(a, b, 1)
Sigma_reg = Sigma + epsilon * s * I
```

Regularization tương đối này quan trọng trong FP32: với hộp có aspect ratio
lớn, phương sai nhỏ có thể bị mất do cancellation khi dựng các phần tử
`(a+b)/2 +/- (a-b) cos(2 theta)/2`; một jitter tuyệt đối `1e-6 I` không theo
kịp scale của covariance và không tạo được bảo đảm conditioning nhất quán.
Với eigenvalue nhỏ nhất tiến về zero, tỉ số điều kiện hiệu dụng được chặn xấp
xỉ `1/epsilon` (với epsilon nhỏ), thay vì phụ thuộc trực tiếp vào kích thước
hộp. `epsilon` vẫn là hyperparameter không thứ nguyên trong config.

Nguồn công thức: [KFIoU, ICLR 2023](https://arxiv.org/html/2201.12558v6),
[KLD, NeurIPS 2021](https://arxiv.org/html/2106.01883), và
[ProbIoU](https://arxiv.org/html/2106.06072).

## 5. Full KFIoU

Với `d = mu_p - mu_t`:

```text
L_center = log(1 + d^T Sigma_t^-1 d)
K = Sigma_p (Sigma_p + Sigma_t)^-1
Sigma_kf = Sigma_p - K Sigma_p
V(Sigma) = 4 sqrt(det(Sigma))
KFIoU = V(Sigma_kf) / (V(Sigma_p) + V(Sigma_t) - V(Sigma_kf))
L_shape = exp(1 - KFIoU) - 1
L_KFIoU = L_center + L_shape
```

Trong implementation FP32, không materialize phép trừ
`Sigma_kf = Sigma_p - K Sigma_p` để lấy thể tích vì phép trừ này có thể mất
chữ số có nghĩa. Dùng identity ổn định:

```text
logdet(Sigma_kf) = logdet(Sigma_p) + logdet(Sigma_t)
                    - logdet(Sigma_p + Sigma_t)
```

Các log-volume được suy ra từ `slogdet` của từng covariance và identity trên;
`Sigma_kf` chỉ là định nghĩa toán học của KFIoU, không phải tensor bắt buộc
phải dựng trong forward.

Đây là full loss có center term để giữ translation gradient khi hai hộp không
giao nhau. Với hai hộp giống hệt nhau, raw KFIoU 2D bằng `1/3`, nên minimum
của shape loss là `exp(2/3) - 1`, không phải 0. Test phải kiểm tra đúng đặc
tính này.

Shape term KFIoU tự nó không phụ thuộc khoảng cách center. Mốc “trong 9 pixel”
trong paper chỉ mô tả mức độ khớp xu hướng với SkewIoU, không phải bán kính
cutoff gradient. Translation gradient ở hộp không giao nhau đến từ center
term của full loss; nó hữu hạn và khác zero tại mọi khoảng cách hữu hạn được
kiểm thử, nhưng giảm tiệm cận xấp xỉ `1/r` khi khoảng cách rất lớn.

## 6. ProbIoU-BD/Hellinger

```text
Sigma_bar = (Sigma_p + Sigma_t) / 2
BD = 1/8 d^T Sigma_bar^-1 d
   + 1/2 log(det(Sigma_bar) / sqrt(det(Sigma_p) det(Sigma_t)))
L_H = sqrt(1 - exp(-BD))
```

Lịch loss cố định theo optimizer-update attempt đã đăng ký trước:

- Từ 0% đến trước 50%: dùng `BD`.
- Từ 50% trở đi: dùng `L_H`.

Global attempt và tổng planned attempts phải nằm trong checkpoint để resume
không làm lệch điểm chuyển. Không chọn switch point theo validation.

BD không bị chặn và giữ tín hiệu displacement hữu ích khi hộp còn xa;
Hellinger bị chặn và gradient bão hòa ở khoảng cách lớn. Đây là lý do dùng BD
trước, Hellinger sau. Paper ProbIoU gốc đánh giá DOTA/HRSC2016 và biểu diễn
Gaussian trên COCO; nó không báo cáo KITTI. Hiệu quả trên KITTI BEV vì vậy là
giả thuyết chuyển miền cần kiểm chứng trong chương trình này.

## 7. KLD comparator

Dùng hướng bất đối xứng cố định `prediction || target`:

```text
D_KL = 1/2 d^T Sigma_t^-1 d
     + 1/2 trace(Sigma_t^-1 Sigma_p)
     + 1/2 log(det(Sigma_t) / det(Sigma_p)) - 1

L_KLD = 1 - 1 / (tau + log(1 + D_KL)), tau = 1
```

Không tuyên bố hoặc kiểm thử tính đối xứng của KLD.

KLD nhạy với covariance suy biến khi width/length tiến về zero. Paper gốc
không đưa ra kết luận riêng về BF16 hoặc NaN; comparator này được giữ lại vì
triển khai bắt buộc cast FP32, scale-relative regularization, `solve`,
`slogdet` và finite checks.

## 8. Masking và reduction

- Classification loss giữ nguyên trên toàn heatmap.
- Geometry loss chỉ tính tại `reg_mask = 1`.
- Mỗi positive location sinh một scalar geometry loss.
- Reduction là mean trên số positive locations.
- Batch không có positive trả graph-connected zero.
- Không đổi dense regression disk hoặc thêm object/class reweighting.

## 9. Task weighting

Controlled baseline B1:

```text
L = L_cls + L_offset + L_size + L_yaw
```

Geometry arms B2-B5:

```text
L = L_cls + L_geometry
```

Regression coefficient cố định bằng 1.0 cho mọi primary arm.

B0 giữ nguyên objective lịch sử:

```text
sum_{i=1..4}(exp(-s_i) L_i + s_i) + 0.2 L_legacy_yaw_aware
```

B0 chỉ là historical anchor, không tham gia xếp hạng primary geometry vì đồng
thời khác B1-B5 ở weighting và legacy auxiliary. B6 `L1 + geometry` và B7
bounded UWAG đã được thiết kế nhưng hoãn cho tới khi B2-B5 xác định được loss
đơn tốt nhất; không tạo config placeholder trong vòng này.

## 10. Ổn định số

- Geometry math luôn chạy FP32 bên trong vùng tắt autocast.
- Input BF16 phải được cast sang FP32 trước `torch.linalg.solve`; vấn đề chính
  là độ chính xác/conditioning, không phải khẳng định BF16 có exponent range
  nhỏ hơn FP32.
- Không dùng explicit matrix inverse.
- Dùng solve/Cholesky solve và scale-relative covariance regularization
  `Sigma_reg = Sigma + epsilon * max(a, b, 1) * I`; không dùng fixed absolute
  jitter `1e-6 I`.
- Dùng `slogdet`; determinant sign không dương phải raise lỗi có ngữ cảnh.
- Không dùng `nan_to_num` để che lỗi.
- Giữ clamp log-size `[-10, 10]`, đồng thời log clamp count.
- Kiểm tra finite loss và toàn bộ gradient trước optimizer step.
- Update có gradient non-finite bị bỏ và được đếm; UWAG parameters cũng không
  được cập nhật ở step đó.
- Không clamp loss cuối.

## 11. Logging và checkpoint

Mỗi epoch ghi:

- Tổng loss và từng thành phần.
- B0 UWAG scales và `exp(-s_i)`; B1-B5 không có learned task scale.
- Global gradient L2 norm trước optimizer step.
- B0 UWAG parameter gradients.
- Non-finite loss count và skipped update count.
- Log-size clamp count.
- Learning rate, optimizer update attempts và successful updates.
- B3 ProbIoU form đang hoạt động.
- Training/validation time và peak CUDA allocated/reserved memory.

Mỗi run lưu resolved config, command, commit/dirty state, Python/PyTorch/CUDA
versions, GPU, seed và SHA256 của split/checkpoint. Các runtime controls (device,
precision, physical/effective batch size, accumulation, worker count và các
giới hạn smoke nếu có) phải nằm trong resolved config, không chỉ trong command
line.

Primary checkpoint là checkpoint cuối epoch 100 đã đăng ký trước. Không chọn
theo validation loss và không thay bằng last-finite checkpoint nếu run hỏng.
Trước khi load resume, phải verify immutable source/split hashes và SHA256
sidecar của checkpoint; mỗi resume thành công ghi git/runtime metadata hiện tại
vào manifest/metrics. Epoch có zero successful optimizer updates không được
publish final checkpoint hoặc `selected/final.pt`. Smoke run dùng final epoch
của smoke config nhưng không được đưa vào kết quả nghiên cứu.

## 12. Ma trận thí nghiệm

Điều kiện chung: một architecture/data snapshot từ A4, seeds `{42, 43, 44}`,
100 epochs, cùng split, optimizer, LR schedule, augmentation và effective batch
size. A-series không phải một chiều của ma trận B-series.

### Phase 0: cổng khả thi MGIoU

MGIoU dùng bốn pháp tuyến từ hai trục của prediction và target. Corner của hai
rectangle được chiếu lên từng pháp tuyến; mỗi trục tính 1D GIoU, sau đó:

```text
MGIoU = mean(GIoU_1, ..., GIoU_4)
L_MGIoU = (1 - MGIoU) / 2
```

Nguồn: [MGIoU, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/37505/41467)
và [official implementation](https://github.com/ldtho/MGIoU).

Vì head lưu doubled angle, prototype chuẩn hóa yaw vector rồi giải mã
`theta = 0.5 atan2(sin(2 theta), cos(2 theta))`. Nếu norm nhỏ hơn epsilon,
dùng fallback `theta = 0`, đếm fallback và không chia cho norm gần zero.

Cổng không dùng validation AP. MGIoU chỉ được research-qualified cho B5 nếu:

- identical/non-overlap/`theta + pi`/near-square/thin/BF16 tests đều finite;
- non-overlap có finite, nonzero position gradient;
- B5 smoke hai train batches không có non-finite hoặc skipped update;
- median forward+backward trên tensor representative không quá `2x` KFIoU.

Benchmark có warmup, đồng bộ CUDA và cùng positive count. Nếu không có CUDA,
runtime gate được ghi `pending` và chưa được phép chạy full MGIoU matrix. Nếu
gate fail, báo B5 là `infeasible`; không chạy full B5 matrix, và không điều
chỉnh ngưỡng hậu nghiệm.

Pre-gate đã chạy ngày **2026-09-22** trong conda environment `AI_env` với
PyTorch `2.11.0`, CUDA build `13.0`, RTX 4050 Laptop GPU (compute capability
`8.9`). Protocol là BF16, 20 warmup iterations, sau đó 100 iterations
forward+backward có đồng bộ CUDA. Các phép thử số học FP32/BF16/FP16 trên hộp
regular và extreme đều cho loss/gradient finite và gradient khác zero. Median
runtime (KFIoU/MGIoU, milliseconds) và tỉ số MGIoU/KFIoU:

| Positive count | KFIoU (ms) | MGIoU (ms) | Ratio |
|---:|---:|---:|---:|
| 32 | 3.1334 | 2.9647 | 0.9461x |
| 128 | 3.0146 | 3.0398 | 1.0084x |
| 512 | 2.7015 | 3.3149 | 1.2271x |

Kết luận gate: **pass**. Final B5 smoke đã được chạy và không ghi nhận
NaN hoặc skipped optimizer update. B5 được research-qualified cho báo cáo
single-seed hiện tại. Checkpoint `epoch_085` là checkpoint có mAP cao nhất
trong các checkpoint B5 đã đánh giá và là checkpoint dùng cho kết quả local:
`map_moderate_percent=56.1929` và `mean_ap_9_percent=56.1601`.

Việc research-qualify này chỉ xác nhận numerical/runtime gate và tính hợp lệ
của run B5; full B-series multi-seed matrix và thống kê paired theo seed vẫn
còn pending. Không dùng kết quả single-seed để tuyên bố statistical
significance.

### Phase B: B0-B5

| ID | Objective | Vai trò |
|---|---|---|
| B0 | A4 snapshot + legacy UWAG + auxiliary 0.2 | Historical anchor |
| B1 | Masked-L1 fixed | Controlled baseline |
| B2 | Full KFIoU fixed | Single-loss ablation |
| B3 | ProbIoU BD/Hellinger fixed | Single-loss ablation |
| B4 | KLD-log fixed | Single-loss ablation |
| B5 | MGIoU fixed | Research-qualified single-seed; full multi-seed matrix pending |

B0 chạy ba seeds để tái lập mốc lịch sử nhưng không tham gia primary ranking.
B1-B5 chạy ba seeds: 15 primary runs sau khi B5 gate đã pass. Toàn B-series
tương ứng 18 runs khi tính cả ba B0 anchor runs.

Xếp hạng B1-B5 theo thứ tự đăng ký trước:

1. Mean `map_moderate_percent`.
2. Nếu hòa, `mean_ap_9_percent`.
3. Nếu vẫn hòa, ít unstable/skipped steps hơn.
4. Nếu vẫn hòa, runtime thấp hơn.

Vì cùng dùng một validation split, kết quả là confirmatory nội bộ chứ không
phải xác nhận độc lập.

### Phase B6+: hoãn

B6 `L1 + geometry winner` fixed và B7 cùng hybrid với bounded UWAG đã được
phác thảo nhưng không nằm trong vòng này. Chỉ tạo chúng sau khi B2-B5 hoàn tất
và có phê duyệt mới; không dành config rỗng hoặc đổi ý nghĩa B0-B5.

QFL/VFL bị hoãn vì Gaussian heatmap hiện tại là spatial suppression target,
không phải IoU-quality target. Thay đổi này cần thiết kế lại target, score
calibration và ranking. Nguồn: [Generalized Focal Loss](https://papers.nips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html)
và [VarifocalNet](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html).

GradNorm và PCGrad cũng ngoài phạm vi vòng này vì cần thêm các lượt đánh giá
gradient theo task, tăng memory/runtime và tạo thêm biến gây nhiễu cho so sánh
loss. Không tuyên bố chúng chắc chắn OOM ở batch size 2. Nguồn:
[GradNorm](https://proceedings.mlr.press/v80/chen18a.html),
[PCGrad](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

## 13. Giả thuyết đăng ký trước

- Full KFIoU có thể cải thiện localization nhờ center term và covariance coupling;
  shape term riêng không cung cấp center gradient.
- ProbIoU-BD sớm tránh far-box saturation; Hellinger muộn hỗ trợ refinement.
- KLD là comparator mạnh cho hộp aspect-ratio lớn nhưng không mặc định thắng.
- MGIoU có thể giữ non-overlap gradient mà không cần covariance solve, nhưng
  chỉ là primary arm nếu vượt cổng số học/runtime.
- Ổn định tốt hơn không mặc nhiên đồng nghĩa AP cao hơn.
- Loss huấn luyện không thay inference output hoặc latency của cùng kiến trúc.

## 14. Metrics và thống kê

Primary: local KITTI-style BEV `map_moderate_percent` R40.

Secondary:

- `mean_ap_9_percent`.
- AP theo class/difficulty.
- Pedestrian/Cyclist distance bands 0-30 m, 30-50 m, 50-70.4 m.
- Runtime, peak VRAM, gradient norm, non-finite/skipped counts.
- MGIoU gate latency ratio và yaw fallback count.
- B0 UWAG trajectories và log-size clamp count.

Báo từng seed, mean +/- sample standard deviation và paired delta theo cùng
seed cho B1-B5. Không tuyên bố statistical significance với chỉ ba seeds. B0
được báo riêng như historical anchor, không gộp vào ranking.

Split cố định:

- Train: 5,984 samples; SHA256 `bda0e3fbce5cb31d07593d3997e52b2ae33fcbc57479e262ec86eedabfaf3db7`.
- Validation: 1,497 samples; SHA256 `cae7f7b5a72f28af771ebe878faa97b81bdafbcecd2be51205fd8cb5ecd2733a`.

## 15. Kiểm thử bắt buộc

- Identical boxes: ProbIoU/KLD/MGIoU bằng 0; KFIoU bằng minimum phi-zero.
- Non-overlap translation vẫn có finite, nonzero center gradient.
- `theta` và `theta + pi` cho cùng loss.
- MGIoU yaw vector gần zero dùng fallback hữu hạn và được đếm.
- Width/length ordering đúng tại `theta = 0`.
- Near-square và thin boxes finite.
- Empty mask trả graph-connected zero.
- Forward/backward finite, gồm BF16 autocast khi CUDA hỗ trợ.
- Resume giữ đúng ProbIoU switch schedule.
- L1 fixed giữ baseline số học cũ.
- Output contract vẫn là `cls`, `offset`, `size`, `yaw`.

## 16. Khả năng tái lập

Mỗi run directory là bất biến nếu không dùng `--resume`. Nó chứa resolved
config (gồm runtime controls), command, environment manifest, immutable source
và split hashes, metrics JSONL, diagnostics và checkpoint hash sidecar. Resume
phải dùng đúng run directory/checkpoint, verify các hashes trước khi load, ghi
git/runtime metadata của lần resume thành công và giữ global update-attempt
counters. Epoch không có successful update không được tạo/publish final
checkpoint.

## 17. Tương thích

Được thay đổi: loss calculation/configuration, training diagnostics,
checkpoint metadata và tests.

Không được thay đổi: model head, target encoding, decoded box semantics,
ONNX/TensorRT names/shapes và KITTI prediction format.

Config `name=baseline|uwag` cũ phải tiếp tục hoạt động. New schema chỉ có hiệu
lực khi có `regression` hoặc `weighting`; khi đó legacy auxiliary bị tắt.
Không sửa file dưới `configs/kitti/mobilebev/`; mọi B config nằm dưới
`configs/kitti/loss_ablation/` và codename không được tái sử dụng cho objective
khác.

## 18. Ngoài phạm vi và giới hạn

Ngoài phạm vi: B6/B7 hybrid, các task-weighting ablation mới, QFL/VFL,
GradNorm/PCGrad, Alpha-IoU, Inner-IoU, Smooth GIoU, polygon-IoU CUDA kernel,
z/height box, official KITTI hidden-test submission, per-loss hyperparameter
sweep và thay đổi inference/export. Inner-IoU paper
gốc chỉ thiết kế cho horizontal boxes; Smooth GIoU vẫn cần oriented polygon /
enclosing geometry. Cả hai cần một nghiên cứu riêng thay vì mở rộng vòng này.

Giới hạn:

- Local evaluator chưa được xác nhận tương đương official devkit.
- Một split và ba seeds không đủ cho kết luận thống kê mạnh.
- Dense regression disk tạo các quan sát tương quan và thiên lệch theo kích thước object.
- Yaw gần như không định danh khi hộp gần vuông.
- KFIoU có minimum phi-zero; shape term không phụ thuộc center và gradient của
  advanced center term giảm tiệm cận khi hộp rất xa.
- Hellinger ProbIoU bão hòa ở khoảng cách lớn; paper gốc không đánh giá KITTI.
- KLD bất đối xứng và nhạy với covariance suy biến dù triển khai có guard FP32.
- MGIoU cần half-angle decode/fallback cho head hiện tại; repository chính thức
  chưa hoàn thiện toàn bộ experiment subrepositories tại thời điểm đặc tả
  ([roadmap](https://github.com/ldtho/MGIoU)).
- KFIoU/KLD dùng factor 4, ProbIoU dùng factor 12 theo từng paper.
- Regression coefficient 1.0 là lựa chọn kiểm soát, không đảm bảo tối ưu riêng.
- B-series chưa phải xác nhận trên held-out test.
