# Kế hoạch MobileBEV-Lite BEV-only

## Triển khai

1. Giữ encoder `binary_slices` để tái lập Legacy35.
2. Dùng encoder `rich8` cho A2/A4 và kiểm tra shape, dtype, giá trị biên.
3. Dùng SG-FPN cho A3/A4; gate khởi tạo tương đương sum-FPN.
4. Dùng chung BEV head, target, decoder, NMS và evaluator cho A1–A4.
5. Chọn checkpoint theo validation loss tại `<run>/selected/best.pt`.

## Cấu hình

- `a1_legacy35_bev.json`
- `a2_rich8_bev.json`
- `a3_legacy35_sgfpn_bev.json`
- `a4_rich8_sgfpn_bev.json`

Các config chỉ khác `data.bev_encoding` và `model.scale_gated_fpn`.

## Xác minh

```bash
python3 tests/test_mobile_bev.py
python3 tests/test_evaluation_table.py
python3 -m unittest tests/test_standard_training_notebook.py
python3 -m compileall detector tools tests
```

Smoke train A4 trước khi chạy đủ ba seed:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/mobilebev/a4_rich8_sgfpn_bev.json \
  --detector-root detector \
  --output-root artifacts/kitti \
  --run-name mobilebev_a4_smoke_seed42 \
  --epochs 1 --max-train-batches 8 --max-val-batches 4 --num-workers 2
```

## Báo cáo

Chạy A1–A4 với seed 42, 43 và 44. Lưu config resolved, checkpoint, hash split,
BEV AP R40, latency và peak memory cho từng run; không dùng số smoke test làm
kết quả nghiên cứu.
