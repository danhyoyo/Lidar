# KITTI Final Runs Configuration Directory

Thư mục này tập hợp đầy đủ các file cấu hình huấn luyện được chuẩn hóa, đã được kiểm thử tính tương thích (không lỗi cú pháp, đúng tham số mô hình và hàm loss).

---

## 1. Nhóm Các Run Đơn Lẻ & Đối Chiếu (01 -> 09)

| File Config | Kiến trúc / Cấu hình | Mục tiêu nghiên cứu |
| :--- | :--- | :--- |
| `01_rc_sgfpn_baseline.json` | MobilePixorNeXt + RC-SGFPN + Baseline Loss | Hoàn thiện **Dòng 32** trong Excel |
| `02_rc_bisgfpn_baseline.json` | MobilePixorNeXt + RC-BiSGFPN + Baseline Loss | Hoàn thiện **Dòng 33** trong Excel |
| `03_mobilepixor_rich8_sumfpn_baseline.json` | MobilePixor cũ + Rich8 + SumFPN + Baseline Loss | Lấp **Hàng 7** trống ở Bảng 1 |
| `04_iqa_head_oga_loss.json` | MobilePixorNeXt + IQA Head + OGA Loss (tắt reparam) | Chuẩn hóa IQA Head với OGA Loss |
| `05_rc_bisgfpn_q_oga_loss.json` | MobilePixorNeXt + RC-BiSGFPN + Q-OGA Loss | Ghép Neck mạnh nhất với Loss mạnh nhất (4-task) |
| `06_ms_litemla_q_oga_loss.json` | MobilePixorNeXt + MS-LiteMLA + Q-OGA Loss | Attention đa kích thước (3, 5) + Q-OGA |
| `07_reparam_q_oga_loss.json` | Rep-MobilePixorNeXt + Q-OGA Loss | Kiểm chứng lại Reparameterization với Q-OGA |
| `08_rich10_sgfpn_q_oga.json` | MobilePixorNeXt + Rich10 BEV + SG-FPN + Q-OGA | Khảo sát biểu diễn BEV 10 kênh |
| `09_rich12_sgfpn_q_oga.json` | MobilePixorNeXt + Rich12 BEV + SG-FPN + Q-OGA | Khảo sát biểu diễn BEV 12 kênh |

---

## 2. Trục A: Full Modular SOTA (Decoupled Quality: OGA Loss + IQA Head)

Chuỗi bậc thang chuẩn mực theo Roadmap v2:

| Bậc | File Config | Cấu hình tích lũy |
| :---: | :--- | :--- |
| **$M_0$** | *(Đã có)* | MobilePixorNeXt + Single LiteMLA + SG-FPN + OGA (**88.72% mAP**) |
| **$M_1$** | `truc_A_M1_reparam_oga.json` | $M_0$ + Rep-MobilePixorNeXt Block (**87.44% mAP** đã có eval) |
| **$M_2$** | `truc_A_M2_reparam_ms_litemla_oga.json` | $M_1$ + Multi-Scale LiteMLA $(3, 5)$ + QK-RMSNorm |
| **$M_3$** | `truc_A_M3_reparam_ms_litemla_rc_bisgfpn_oga.json` | $M_2$ + RC-BiSGFPN (Bidirectional Range-Conditioned Neck) |
| **$M_4$** | `truc_A_M4_sota_reparam_ms_litemla_rc_bisgfpn_iqa_oga.json` | $M_3$ + IQA Head + Joint NMS (**SOTA TỐI HẬU CỦA ĐỀ TÀI**) |

---

## 3. Trục B: Ultra-Compact SOTA (Joint Quality: Q-OGA Loss + Standard 4-Task Head)

Chuỗi bậc thang siêu nhẹ không cần thêm nhánh thứ 5:

| Bậc | File Config | Cấu hình tích lũy |
| :---: | :--- | :--- |
| **$M_0$** | *(Đã có)* | MobilePixorNeXt + Single LiteMLA + SG-FPN + Q-OGA (**90.24% mAP**) |
| **$M_1$** | `truc_B_M1_reparam_q_oga.json` | $M_0$ + Rep-MobilePixorNeXt Block |
| **$M_2$** | `truc_B_M2_reparam_ms_litemla_q_oga.json` | $M_1$ + Multi-Scale LiteMLA $(3, 5)$ + QK-RMSNorm |
| **$M_3$** | `truc_B_M3_sota_reparam_ms_litemla_rc_bisgfpn_q_oga.json` | $M_2$ + RC-BiSGFPN (**SOTA SIÊU NHẸ 4-TASK**) |

---

## 4. Hướng dẫn Chạy Huấn Luyện

Chạy bằng môi trường `AI_env` đã cài sẵn:
```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python tools/kitti_training_pipeline/train.py \
    --config configs/kitti/final_runs/<tên_config>.json \
    --detector-root detector \
    --output-root runs/final_runs
```
