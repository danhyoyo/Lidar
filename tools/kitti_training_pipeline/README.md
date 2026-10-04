# KITTI augmentation training pipeline

Xem [hướng dẫn ở repository root](../../README.md) và [bộ cấu hình A–E](../../configs/kitti/augmentation/README.md).

- `train.py`: cùng model/loss, OneOf augmentation; bộ A–E dùng toàn bộ lịch 50 epoch, warmup 4. `--stop-after-epoch` vẫn là tùy chọn dừng sớm khi cần.
- `evaluate_kitti_bev.py`: evaluate checkpoint cùng epoch, mAP Moderate và AP từng lớp.
- `select_checkpoint.py`: hỗ trợ lựa chọn checkpoint theo AP sau khi có kết quả đánh giá.
- `export_onnx.py`, `build_tensorrt.py`, `compare_models.py`, `deploy_engine.py`: các công cụ export/deployment.

Physics flags là heuristic: shadow dùng solid bounding box, density có floor 5 điểm, intensity dùng gamma cố định.
B–E giữ ground/static collision/line-of-sight validation; box collision và volume clearing luôn áp dụng.
`sample_counts` là số object thêm tối đa, không phải tổng object mục tiêu trong scene.

`placement_mode="source_relative"` thay đổi range trong 0.8–1.2 lần và azimuth ±5°; heading xoay cùng azimuth để giữ hướng quan sát tương đối.
Mặc định legacy `placement_mode="random"` vẫn dùng các proposal corridor/fallback như trước.

Audit trước khi training:

```bash
python3 tools/visualization/audit_gt_sampler.py \
  --config configs/kitti/augmentation/b_light_sampling.json \
  --frames splits/kitti/train.txt --seed 42 \
  --output-dir artifacts/audit_b
```

Audit đo sampling trước global augmentation/BEV, báo acceptance, rejection, visibility và effective pose/physics settings.
