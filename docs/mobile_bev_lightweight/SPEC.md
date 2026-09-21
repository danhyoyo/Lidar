# MobileBEV-Lite: đặc tả BEV-only

## Mục tiêu

Đánh giá hai thay đổi nhẹ trên MobilePIXOR cho phát hiện vật thể trong không
gian nhìn từ trên xuống:

1. `RichBEV-8` nén đầu vào LiDAR từ 35 lát nhị phân xuống 8 kênh thống kê.
2. `SG-FPN` thay phép cộng FPN cố định bằng hai gate học được tại C4/C3.

Mô hình chỉ dự đoán hộp BEV `(x, y, w, l, yaw)`. Head gồm
`cls(3) + offset(2) + size(2) + yaw(2)`.

## Ma trận ablation

| ID | Input | Fusion | Head | Vai trò |
|---|---|---|---|---|
| A0 | Legacy35 | Sum | BEV | Baseline lịch sử |
| A1 | Legacy35 | Sum | BEV | Baseline kiểm soát cho A2–A4 |
| A2 | RichBEV-8 | Sum | BEV | Tác động của input |
| A3 | Legacy35 | SG-FPN | BEV | Tác động của gate |
| A4 | RichBEV-8 | SG-FPN | BEV | MobileBEV-Lite đầy đủ |

A1–A4 dùng cùng split, seed `{42, 43, 44}`, optimizer, schedule,
augmentation, loss, threshold và quy tắc chọn checkpoint.

## Chỉ số

- Primary: BEV mAP Moderate R40.
- Secondary: BEV AP R40 theo class/difficulty, mean AP-9 và AP Moderate theo
  khoảng cách cho Pedestrian/Cyclist.
- Deployment: số tham số, model latency, end-to-end latency, FPS, peak VRAM
  và kích thước tensor đầu vào.
- Báo mean ± standard deviation qua ba seed.

Evaluator là phép đo local khớp loader và không thay thế kết quả từ KITTI
official test server.

## Tiêu chí quyết định

- Chỉ claim cải thiện accuracy khi A4 tốt hơn A1 ổn định qua ba seed.
- BEV mAP Moderate của A4 không được giảm quá 1 điểm nếu đóng góp chính là
  giảm chi phí truyền dữ liệu.
- Nếu SG-FPN không cải thiện A3/A4, bỏ gate và giữ biến thể RichBEV đơn giản.
- Nếu RichBEV-8 giảm accuracy quá mức, giữ Legacy35 làm đầu vào chính.

## Hợp đồng cấu hình

- `data.bev_encoding` thiếu thì dùng `binary_slices`.
- `model.scale_gated_fpn` thiếu thì dùng `false`.
- Không có lựa chọn box encoding; toàn bộ pipeline dùng BEV.
