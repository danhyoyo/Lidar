# Augmentation ablation A–E

Chạy **A và B trước**. C/D/E là ba thí nghiệm độc lập, mỗi run bật thêm một
physics flag so với B.

| Run | File | Augmentation |
|---|---|---|
| A | [a_standard.json](a_standard.json) | Standard OneOf, không sampling |
| B | [b_light_sampling.json](b_light_sampling.json) | A + sampling nhẹ |
| C | [c_sampling_density.json](c_sampling_density.json) | B + density subsampling |
| D | [d_sampling_shadow.json](d_sampling_shadow.json) | B + shadow masking |
| E | [e_sampling_intensity.json](e_sampling_intensity.json) | B + intensity calibration |

## Thiết lập chung

- Model: MobilePIXORNeXt, LiteMLA scale `[5]`, QK norm `none`, SG-FPN, Rich8.
- Loss: baseline focal + L1; không IQA/reparameterization.
- Seed 42; physical batch 16, accumulation 1; validation batch 32.
- AdamW: LR `0.0007`, weight decay `0.001`, warmup 4, gradient clipping 10.
- `train.epochs=50`: toàn bộ lịch train, gồm 4 epoch warmup và 46 epoch cosine.
- Global augmentation giống nhau: `mode="one_of"`, probability `0.5`; chọn một
  trong rotation ±20°, scaling 0.95–1.05 hoặc translation Gaussian std 0.4 m
  trên cả XYZ. Flip/dropout/intensity jitter không bật trong bộ đối chứng này.

## Sampling nhẹ B–E

- Probability `0.2`, thêm tối đa 1 Pedestrian và 1 Cyclist mỗi frame được chọn.
- `placement_mode="source_relative"`: range 0.8–1.2 lần range nguồn, azimuth ±5°.
  Heading xoay cùng azimuth, không chọn yaw độc lập từ toàn bộ [−π, π].
- Ground validation, static collision và line-of-sight bật ở cả B–E.
- B tắt density/shadow/intensity; C/D/E chỉ bật riêng flag tương ứng.
- `min_visible_points=5`, `min_visible_ratio=0.5` bảo vệ object đã có khi paste;
  chúng không phải bộ lọc chất lượng candidate sau BEV.
- Các physics flags là heuristic cần được kiểm chứng bằng AP và audit thực tế.

## Chạy và resume

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/augmentation/a_standard.json \
  --detector-root detector --output-root artifacts/kitti \
  --num-workers 6 --target-backend numba
```

Chạy từ repository root; các paths dữ liệu tương đối với root. Config hoàn thành
lịch LR ở epoch 50 và lưu `checkpoints/50epoch.pt`; evaluate cùng checkpoint này cho mọi run.
Mỗi config có `experiment.name` riêng để tạo tên artifacts khác nhau.

Để tiếp tục run bị gián đoạn, dùng **cùng config** với
`--resume <run>/checkpoints/last.pt`; giữ lịch 50 epoch.
Nếu muốn thử lịch 100 epoch, tạo run mới và train lại từ đầu với warmup 8.
Kết quả 50 epoch này dùng để so sánh A–E cùng ngân sách, không so trực tiếp với
baseline 100 epoch cũ để kết luận tác động của augmentation.
Smoke test `--epochs 1` được hỗ trợ; dùng thư mục/run-name riêng cho smoke.

Notebook đã đăng ký `VARIANT="A"` đến `"E"`; mặc định screening A ở epoch 50.
Xem [README gốc](../../../README.md) để chuẩn bị dữ liệu và evaluate.
