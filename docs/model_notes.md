# Model notes

Ngày cập nhật: 2026-09-21.

Pipeline hiện tại là BEV-only. Mọi model dùng head
`cls(3) + offset(2) + size(2) + yaw(2)` để dự đoán `(x, y, w, l, yaw)`.

| ID | Config | Input | Fusion | Vai trò |
|---|---|---|---|---|
| A0 | `configs/kitti/kitti_uwag_coordatt_aug.json` | Legacy35 | Sum | Baseline đã tái lập |
| A1 | `configs/kitti/mobilebev/a1_legacy35_bev.json` | Legacy35 | Sum | Baseline kiểm soát |
| A2 | `configs/kitti/mobilebev/a2_rich8_bev.json` | RichBEV-8 | Sum | Ablation input |
| A3 | `configs/kitti/mobilebev/a3_legacy35_sgfpn_bev.json` | Legacy35 | SG-FPN | Ablation fusion |
| A4 | `configs/kitti/mobilebev/a4_rich8_sgfpn_bev.json` | RichBEV-8 | SG-FPN | Full MobileBEV-Lite |

A1–A4 dùng 5.984 frame train, 1.497 frame validation, BF16, physical batch 2,
accumulation 2 và 100 epoch trong config seed 42. Protocol cuối dùng seed
`{42, 43, 44}` và báo BEV AP R40 cùng latency.

Repo chưa chứa kết quả huấn luyện đầy đủ cho A1–A4; không suy diễn claim
accuracy cho tới khi có artifact gắn với config, checkpoint và split cụ thể.
