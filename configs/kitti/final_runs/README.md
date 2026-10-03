# Danh Sách Cấu Hình Huấn Luyện (Chỉ Dùng SG-FPN Chuẩn)

Thư mục này chứa toàn bộ các cấu hình đã được tối ưu hóa, loại bỏ hoàn toàn RC-SGFPN/RC-BiSGFPN, và **chỉ sử dụng kiến trúc Bilinear Scale-Gated FPN (SG-FPN)** đã được chứng minh hiệu năng cao và ổn định.

---

## 1. Nhóm Các Run Đơn Lẻ & Đối Chiếu (01 -> 06)

| File Config | Kiến trúc / Cấu hình | Loss Function | Mục tiêu nghiên cứu |
| :--- | :--- | :---: | :--- |
| `01_mobilepixor_rich8_sumfpn_baseline.json` | MobilePixor cũ + Rich8 + SumFPN | Baseline | Lấp **Hàng 7** trống ở Bảng 1 (đối chiếu Rich8 vs SG-FPN) |
| `02_iqa_head_oga_loss.json` | MobilePixorNeXt + Single LiteMLA + SG-FPN + IQA Head | OGA | Cô lập đánh giá IQA Head với OGA Loss (không reparam) |
| `03_ms_litemla_q_oga_loss.json` | MobilePixorNeXt + MS-LiteMLA $(3, 5)$ + SG-FPN | Q-OGA | Cô lập đánh giá Attention đa kích thước với Q-OGA |
| `04_reparam_q_oga_loss.json` | Rep-MobilePixorNeXt + Single LiteMLA + SG-FPN | Q-OGA | Cô lập đánh giá Reparameterization với Q-OGA |
| `05_rich10_sgfpn_q_oga.json` | MobilePixorNeXt + Rich10 BEV + SG-FPN | Q-OGA | Khảo sát biểu diễn BEV 10 kênh |
| `06_rich12_sgfpn_q_oga.json` | MobilePixorNeXt + Rich12 BEV + SG-FPN | Q-OGA | Khảo sát biểu diễn BEV 12 kênh |

---

## 2. Trục A: Full Modular SOTA (Decoupled Quality: OGA Loss + IQA Head + SG-FPN)

Chuỗi bậc thang 3 bước chuẩn mực:

| Bậc | File Config | Cấu hình tích lũy | Tình trạng |
| :---: | :--- | :--- | :---: |
| **$M_0$** | *(Đã có)* | MobilePixorNeXt + Single LiteMLA + SG-FPN + OGA | **88.72% mAP** |
| **$M_1$** | `truc_A_M1_reparam_oga.json` | $M_0$ + Rep-MobilePixorNeXt Block | **87.44% mAP** *(đã có eval)* |
| **$M_2$** | `truc_A_M2_reparam_ms_litemla_oga.json` | $M_1$ + Multi-Scale LiteMLA $(3, 5)$ + QK-RMSNorm | Cần chạy |
| **$M_3$** | `truc_A_M3_sota_reparam_ms_litemla_iqa_oga.json` | $M_2$ + IQA Head + Joint NMS (**SOTA v2 TỐI HẬU**) | Cần chạy (Bản thử đã đạt 90.52%) |

---

## 3. Trục B: Ultra-Compact SOTA (Joint Quality: Q-OGA Loss + 4-Task Head + SG-FPN)

Chuỗi bậc thang siêu nhẹ giữ nguyên 4 nhánh phát hiện:

| Bậc | File Config | Cấu hình tích lũy | Tình trạng |
| :---: | :--- | :--- | :---: |
| **$M_0$** | *(Đã có)* | MobilePixorNeXt + Single LiteMLA + SG-FPN + Q-OGA | **90.24% mAP** |
| **$M_1$** | `truc_B_M1_reparam_q_oga.json` | $M_0$ + Rep-MobilePixorNeXt Block | Cần chạy |
| **$M_2$** | `truc_B_M2_sota_reparam_ms_litemla_q_oga.json` | $M_1$ + Multi-Scale LiteMLA $(3, 5)$ + QK-RMSNorm (**SOTA SIÊU NHẸ**) | Cần chạy |

---

## 4. Lệnh Chạy Huấn Luyện

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python tools/kitti_training_pipeline/train.py \
    --config configs/kitti/final_runs/<tên_config>.json \
    --detector-root detector \
    --output-root runs/final_runs
```
