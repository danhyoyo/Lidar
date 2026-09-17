# Đặc tả nghiên cứu: MobileBEV-Lite cho 3D LiDAR Object Detection

Trạng thái: Draft v1
Ngày khóa phạm vi: 2026-09-13
Baseline mã nguồn: commit `24deca8` (`mobileBEV-architecture`)

## Tóm tắt quyết định

Đề xuất xây dựng **MobileBEV-Lite** như một biến thể tiến hóa của `MobilePIXOR + Coordinate Attention tại C5 + UWAG`, không thay backbone bằng mạng mới. Ba thay đổi được kiểm soát độc lập:

1. **RichBEV-8** thay tensor occupancy 35 kênh bằng 8 kênh thống kê height, intensity, density và ba dải độ cao.
2. **Scale-Gated FPN (SG-FPN)** thay phép cộng cố định tại hai nhánh C4/C3 bằng cổng depthwise động, khởi tạo sao cho mô hình đúng bằng FPN hiện tại.
3. **Minimal 3D Center Head** mở rộng hai tensor regression hiện có để dự đoán thêm `z` và `h`, không tạo thêm tower hoặc output ONNX mới.

Mục tiêu khoa học là kiểm tra liệu tổ hợp này có cải thiện 3D AP cho Pedestrian/Cyclist và giảm chi phí truyền tensor mà vẫn giữ mô hình khoảng 0.6 triệu tham số hay không.

## Quyết định thiết kế

### Goal

- Tạo một detector LiDAR-only dự đoán hộp 3D đầy đủ trên KITTI, ưu tiên vật thể nhỏ, với chi phí inference thấp hơn baseline hiện tại.
- Tách được đóng góp của biểu diễn đầu vào và fusion bằng ablation có đối chứng.

### Constraints

- Giữ ROI KITTI, split 5,984/1,497, `out_size_factor=4`, augmentation, CoordAtt C5 và UWAG hiện tại để không thay nhiều biến cùng lúc.
- Không dùng 3D sparse convolution, Transformer, distillation hoặc camera fusion trong nghiên cứu này.
- Chỉ dùng toán tử thân thiện ONNX/TensorRT: convolution, transposed convolution, sigmoid, add và multiply.
- Baseline hiện tại từng phát sinh giá trị non-finite ở epoch 81; mọi run mới phải ghi nhận và dừng an toàn khi lỗi số học xuất hiện.
- Kết quả trên 1,497 frame chỉ được gọi là local validation, không gọi là official KITTI test result.

### Known context

- `Proposal.xlsx`, hàng 5 đề xuất Rich 2D BEV, Scale-gated FPN và 3D Center Head. Hàng 78 lặp lại hàng Cross-Representation Distillation và không ảnh hưởng phạm vi này.
- Encoder hiện tại tạo tensor `35 x 800 x 704` bằng occupancy nhị phân theo trục z và bỏ cột intensity của point cloud.
- Head hiện tại chỉ xuất `cls(3) + offset_xy(2) + size_wl(2) + yaw(2)`; decoder trả hộp BEV 7 trường và evaluator chỉ tính BEV AP R40.
- Backbone hiện tại dùng MobileNetV2 inverted residual, CoordAtt tại C5, hai phép cộng FPN C4/C3 và output stride 4.
- Kết quả đã lưu: mAP Moderate BEV 79.7872%, Pedestrian Moderate BEV AP 65.7268%, model-only 12.6516 ms, end-to-end 63.9807 ms, host-to-device 30.2022 ms và peak VRAM 355.1 MiB trên RTX 5060 Ti.
- Luận văn báo cáo biến thể CoordAtt C5 có 600,337 tham số và cho trade-off tốt nhất trong các vị trí attention đã thử.

### Risks

- Rich/high-resolution BEV đã là hướng nghiên cứu có trước; TriBand-BEV (2026) là công trình gần nhất và làm giảm độ mới nếu chỉ thay số kênh.
- Giảm 35 lát z xuống 8 kênh có thể giảm thông tin hình học theo chiều cao, đặc biệt với Cyclist.
- Phép tổng hợp RichBEV trên CPU có thể làm tăng preprocessing dù giảm H2D.
- Direct `z/h` regression có thể không đủ để tăng 3D AP nếu ground variation lớn.
- Gate có thể học bão hòa về 0/2 hoặc chỉ sao chép phép cộng cũ.
- Dùng một seed hoặc chọn checkpoint theo metric sau khi xem kết quả sẽ tạo bằng chứng yếu.

### Options (2–4)

1. **RichBEV-8 + SG-FPN + Minimal 3D Center Head**
   - Summary: thay input, hai điểm fusion và số chiều của head; giữ backbone/loss/runtime interface.
   - Pros / cons: giảm mạnh input và có ablation rõ; novelty mang tính tích hợp, cần đối chiếu TriBand-BEV.
   - Complexity / risk: trung bình; rủi ro chính ở encoder CPU và vertical regression.

2. **Pillar/sparse encoder + CenterPoint head**
   - Summary: thay toàn bộ front-end bằng PFN hoặc sparse convolution.
   - Pros / cons: baseline 3D mạnh hơn; tăng dependency, memory engineering và làm mất câu hỏi nghiên cứu về dense Mobile BEV.
   - Complexity / risk: cao.

3. **TriBand-BEV-style high-resolution bidirectional neck**
   - Summary: thêm nhiều mức feature và nhiều head scale.
   - Pros / cons: có bằng chứng tốt cho Pedestrian; gần công trình 2026, tăng FLOPs và khó chứng minh novelty.
   - Complexity / risk: cao.

### Recommendation

Chọn Option 1. Đây là thay đổi nhỏ nhất giải quyết đồng thời ba khoảng trống thật trong repo: intensity bị bỏ, fusion cố định và thiếu `z/h`. Điểm phân biệt với TriBand-BEV phải được viết rõ: RichBEV-8 dùng thống kê height/intensity/density, SG-FPN chỉ có hai gate residual trên FPN hiện tại, và mô hình hồi quy `z/h` trực tiếp thay vì phục hồi độ cao bằng IQR từ point cloud.

### Acceptance criteria

- Full model dự đoán hộp `(x, y, z, w, l, h, yaw)` và evaluator báo cả BEV AP R40 lẫn 3D AP R40.
- A4 cải thiện trung bình ít nhất **+2.0 điểm Pedestrian Moderate 3D AP** và **+1.0 điểm mean 3D AP-9** so với A1 qua 3 seed.
- BEV mAP Moderate của A4 không giảm quá 1.0 điểm so với A1.
- A4 có không quá 0.65 triệu tham số và không tăng model p95 latency quá 10% so với A1 trên cùng máy.
- RichBEV-8 giảm H2D mean latency ít nhất 50% và giảm end-to-end mean latency ít nhất 20% so với A1.
- Không có NaN/Inf trong loss, gradient, checkpoint đã chọn hoặc output evaluation.
- ONNX opset 17 hợp lệ; TensorRT FP16 giữ sai lệch output trong tolerance đã khóa và không giảm AP quá 0.2 điểm.
- Báo cáo 3 seed, mean ± standard deviation và paired frame bootstrap 95% CI cho A4 - A1.

## Research question và giả thuyết

**RQ1.** Một BEV 8 kênh giàu thông tin có thể thay 35 lát occupancy để giảm chi phí dữ liệu mà không làm mất accuracy không?

- H1a: RichBEV-8 giảm kích thước input lý thuyết từ 75.2 MiB xuống 17.2 MiB mỗi frame FP32, tương đương 77.1%.
- H1b: A2 có Pedestrian/Cyclist 3D AP không thấp hơn A1 và H2D latency thấp hơn.

**RQ2.** Hai gate tại C4/C3 có bảo toàn tín hiệu vật thể nhỏ tốt hơn phép cộng FPN cố định không?

- H2: A3 tăng Pedestrian Moderate 3D AP so với A1 mà thêm dưới 0.1% tham số.

**RQ3.** Tổ hợp RichBEV-8 và SG-FPN có tạo Pareto improvement không?

- H3: A4 thỏa đồng thời target accuracy, latency và memory trong Acceptance criteria.

Null hypothesis cho mỗi so sánh là chênh lệch AP bằng 0. Không kết luận từ một seed.

## Phạm vi

### Trong phạm vi

- KITTI Car, Pedestrian, Cyclist.
- LiDAR-only single-frame.
- Dense 2D BEV và MobilePIXOR backbone hiện có.
- PyTorch FP32/BF16 training, ONNX opset 17, TensorRT FP16 inference.
- Local BEV/3D AP R40, latency theo stage, peak VRAM và kích thước input.

### Ngoài phạm vi

- Camera-LiDAR fusion, temporal fusion, Mamba, DRL pruning và sparse convolution.
- Knowledge distillation và quantization trong vòng nghiên cứu đầu tiên.
- Tracking, ROS transport, rendering và HMI trong latency boundary.
- Tuning riêng trên official test set.

Chỉ mở rộng các mục ngoài phạm vi sau khi A4 vượt A1; nếu không, ablation hiện tại đã đủ chỉ ra thành phần thất bại.

## Đặc tả kiến trúc

### 1. RichBEV-8

Input raw KITTI có dạng `(x, y, z, reflectance)`. Với mỗi ô `(x, y)` trong lưới 0.1 m, encoder xuất theo thứ tự cố định:

| Kênh | Tên | Định nghĩa |
|---:|---|---|
| 0 | `occ_low` | 1 nếu có điểm ở 1/3 thấp của `[z_min, z_max]`, ngược lại 0 |
| 1 | `occ_mid` | 1 nếu có điểm ở 1/3 giữa |
| 2 | `occ_high` | 1 nếu có điểm ở 1/3 cao |
| 3 | `z_max` | max z chuẩn hóa về `[0,1]` |
| 4 | `z_mean` | mean z chuẩn hóa về `[0,1]` |
| 5 | `i_max` | max reflectance sau scale và clip `[0,1]` |
| 6 | `i_mean` | mean reflectance sau scale và clip `[0,1]` |
| 7 | `density` | `min(1, log(1+n) / log(1+density_norm))` |

Quy tắc:

- Ô rỗng có toàn bộ kênh bằng 0.
- Biên height band được suy ra từ `z_min/z_max`; không học từ validation.
- `density_norm=32` và `intensity_scale=1.0` là tham số calibration trong config. `intensity_scale` phải được đo lại cho sensor không dùng thang KITTI.
- Dùng NumPy `bincount`/scatter reduction trong một lần quét; không thêm dependency.
- Legacy encoder 35 kênh phải giữ nguyên dưới `bev_encoding.name="binary_slices"`.

Chi phí lý thuyết tại kích thước KITTI:

- Input FP32: `35*800*704*4 = 78,848,000 bytes` xuống `8*800*704*4 = 18,022,400 bytes`.
- Conv stem đầu tiên giảm khoảng 4.38 GMAC, từ 5.68 xuống 1.30 GMAC.
- Đây là ước lượng tĩnh; báo cáo cuối phải dùng profiler và latency đo thực tế.

### 2. Backbone

Giữ nguyên:

- Stem 32 channels.
- Inverted residual C2/C3/C4/C5: 24/32/64/96.
- CoordAtt macro-level tại C5.
- Lateral projection: C5->64, C4->32, C3->16.
- Hai transposed convolution và output stride 4.

Chỉ tham số hóa số kênh input từ encoder. Legacy mặc định là 35.

### 3. Scale-Gated FPN

Với mỗi mức `i` thuộc C4 và C3:

```text
U_i = upsample(P_{i+1})
L_i = lateral(C_i)
G_i = 2 * sigmoid(DWConv3x3_i(L_i + U_i))
P_i = U_i + G_i * L_i
```

- `DWConv3x3_i` có `groups=channels`, bias và không có BatchNorm.
- Khởi tạo toàn bộ weight/bias của gate bằng 0. Khi đó `G_i=1`, nên output ban đầu đúng bằng `U_i + L_i` của FPN hiện tại.
- Gate C4 có 320 tham số; gate C3 có 160; tổng 480 tham số.
- Không thêm bottom-up path, extra scale hoặc multi-head prediction.

### 4. Minimal 3D Center Head

Giữ bốn output name hiện tại để không mở rộng interface ONNX/TensorRT:

| Output | Shape | Ý nghĩa |
|---|---:|---|
| `cls` | 3 | Gaussian center heatmap cho Car/Pedestrian/Cyclist |
| `offset` | 3 | `[dx, dy, z_center]` |
| `size` | 3 | `[log(w), log(l), log(h)]` |
| `yaw` | 2 | `[cos(2*yaw), sin(2*yaw)]`, giữ tương thích baseline |

- `z_center = z_bottom + h/2` khi tạo target.
- Decoder trả `(class_id, score, x, y, z_center, w, l, h, yaw)`.
- Chỉ tăng output channel của hai convolution 1x1 cuối; không thêm tower mới.
- So với model 600,337 tham số, ước lượng A4 là khoảng **593,075 tham số**: giảm 7,776 ở input conv, thêm 34 ở head và 480 ở gates. Phải xác nhận bằng `sum(p.numel())` sau khi cài đặt.

### 5. Loss

- Giữ Modified Focal Loss cho `cls`.
- Giữ L1 cho `offset`, `size`, `yaw`; L1 tự bao phủ thêm `z_center` và `log(h)`.
- Giữ bốn UWAG task weights hiện tại, không tạo hai uncertainty parameter mới.
- Geometric term tiếp tục dùng hai chiều đầu của `offset/size` để tính yaw-aware BEV overlap.
- Chưa thêm 3D IoU loss. Chỉ mở mục này nếu A1 cho thấy `z/h` không hội tụ dù target và decode đúng.

### 6. Decode, NMS và 3D IoU

- NMS vẫn dùng rotated BEV box để giữ runtime và hành vi lựa chọn proposal.
- 3D IoU được tính bằng `BEV_intersection_area * vertical_overlap / union_volume`.
- `vertical_overlap = max(0, min(z_top_1,z_top_2) - max(z_bottom_1,z_bottom_2))`.
- Evaluator báo cả BEV và 3D AP R40 với IoU 0.7 cho Car, 0.5 cho Pedestrian/Cyclist và giữ difficulty rules hiện tại.
- Báo riêng các dải khoảng cách `[0,30)`, `[30,50)`, `[50,70.4]` m cho Pedestrian/Cyclist.

## Config contract

```json
{
  "data": {
    "bev_encoding": {
      "name": "rich8",
      "density_norm": 32.0,
      "intensity_scale": 1.0
    }
  },
  "model": {
    "backbone": "mobilepixor_coordatt",
    "scale_gated_fpn": true,
    "box_encoding": "center3d"
  }
}
```

Backward-compatible defaults:

- Thiếu `data.bev_encoding` -> `binary_slices`.
- Thiếu `model.scale_gated_fpn` -> `false`.
- Thiếu `model.box_encoding` -> `bev`.

## Ma trận ablation bắt buộc

| ID | BEV | Fusion | Head | Vai trò |
|---|---|---|---|---|
| A0 | Legacy35 | Sum | BEV | Reproduce kết quả hiện tại, không dùng để so 3D AP |
| A1 | Legacy35 | Sum | Center3D | Baseline 3D công bằng |
| A2 | RichBEV-8 | Sum | Center3D | Đóng góp của input |
| A3 | Legacy35 | SG-FPN | Center3D | Đóng góp của gate |
| A4 | RichBEV-8 | SG-FPN | Center3D | Full MobileBEV-Lite |

Mọi biến thể A1-A4 dùng cùng split, seed set `{42,43,44}`, optimizer, schedule, augmentation, loss, threshold và checkpoint rule. Không so 3D AP của A4 với A0 vì A0 không dự đoán `z/h`.

## Giao thức thí nghiệm

### Training

- Một smoke run nhỏ và một run 30 epoch/seed 42 để phát hiện lỗi triển khai; không dùng screening làm kết luận cuối.
- Final ablation: A1-A4, 100 epoch, 3 seed.
- Chọn checkpoint bằng `mAP 3D Moderate` trung bình 3 lớp, quy tắc khóa trước khi chạy.
- Giữ checkpoint mỗi 5 epoch để audit; không đổi selection metric sau khi xem kết quả.
- Dừng run nếu loss/gradient/output không finite và lưu batch/epoch gây lỗi.

### Accuracy

- Primary endpoint: Pedestrian Moderate 3D AP R40.
- Co-primary aggregate: mean 3D AP-9.
- Secondary: BEV AP R40 theo class/difficulty, mAP Moderate, max recall và AP theo khoảng cách.
- Report mean ± standard deviation qua 3 seed.
- Recompute AP trên 2,000 paired frame bootstrap samples, seed 42, cho A4 - A1.

### Efficiency

- Input bytes và channel count.
- Parameter count và GMAC/FLOPs bằng cùng profiler.
- Preprocess, H2D, model, decode/NMS và input-to-detections: mean, p50, p95, p99.
- Peak allocated VRAM.
- Mỗi benchmark dùng 10 warm-up frame, toàn bộ 1,497 frame, batch 1, cùng GPU/power mode/software.
- PyTorch FP32 là comparison baseline; TensorRT FP16 là deployment result riêng.

### Reproducibility

- Ghi commit, config hash, split hash, seed, checkpoint hash, CUDA, PyTorch, TensorRT và GPU.
- Không ghi zero cho bước chưa chạy; dùng `not_evaluated`.
- Lưu per-frame predictions để bootstrap và audit.
- Official KITTI test chỉ chạy sau khi khóa A4 và toàn bộ threshold.

## Tiêu chí go/no-go

- **Go:** A4 đạt toàn bộ acceptance criteria; viết contribution về Pareto improvement và ablation.
- **Partial go:** A2 giảm latency rõ nhưng accuracy không tăng; đóng góp được thu hẹp thành input compression/deployment và không nhận claim về small-object accuracy.
- **No-go SG-FPN:** A3/A4 không tăng Pedestrian 3D AP hoặc gate bão hòa; bỏ gates, giữ RichBEV + Center3D.
- **No-go RichBEV:** A2 giảm trên 1.0 điểm BEV mAP hoặc 3D AP; thử RichBEV-11 với thêm ba coarse occupancy bin chỉ như một follow-up đã định trước.
- **Stop:** không sửa được non-finite baseline hoặc evaluator 3D không khớp sanity tests/official devkit.

## Claim được phép dùng nếu đạt gate

> MobileBEV-Lite mở rộng IRBGHR-MobilePIXOR thành detector hộp 3D trực tiếp bằng một biểu diễn BEV 8 kênh và hai cổng fusion depthwise khởi tạo tương đương FPN gốc. Trên cùng giao thức KITTI, thiết kế được đánh giá bằng ablation đa seed theo accuracy, latency và memory.

Không dùng các claim “state of the art”, “first” hoặc “edge-ready” nếu chưa có đối chiếu đầy đủ và benchmark trên phần cứng edge thật.

## Tài liệu chính

1. Yang, Luo, Urtasun. [PIXOR: Real-time 3D Object Detection from Point Clouds](https://arxiv.org/abs/1902.06326).
2. Nguyen et al. [Enhancing Indoor Robot Pedestrian Detection Using Improved PIXOR Backbone and Gaussian Heatmap Regression](https://doi.org/10.1109/ACCESS.2024.3351868).
3. Lin et al. [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144).
4. Tan, Pang, Le. [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070).
5. Yin, Zhou, Krähenbühl. [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275).
6. Khoshkdahan, Vinel. [TriBand-BEV: Real-Time LiDAR-Only 3D Pedestrian Detection via Height-Aware BEV and High-Resolution Feature Fusion](https://arxiv.org/abs/2605.12220).
7. [KITTI 3D Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_3dobject.php).
