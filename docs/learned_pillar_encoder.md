# So sánh rich8 với learned pillar32

`pillar32` là encoder học được bổ sung; `configs/config.json` vẫn mặc định rich8.
Đây là phép thử biểu diễn đầu vào, không phải triển khai toàn bộ detector PointPillars.

## Kiến trúc

```text
Points sau augmentation
  → lọc finite/ROI và nhóm theo x,y (CPU)
  → 10 feature cho mỗi điểm
  → Linear(10,32, bias=False) + BatchNorm + ReLU (model)
  → max-pooling các điểm trong từng pillar
  → scatter thành BEV [B,32,H,W]
  → MobilePixorNeXt → detection head hiện tại
```

Mười feature gồm xyz, intensity, ba offset xyz so với trung bình các điểm trong
pillar và ba offset xyz so với tâm pillar. Tọa độ/offset dùng mét; tâm z là trung
điểm khoảng ROI. Intensity được scale và clip [0,1] như rich8. Biên ROI mở với
epsilon 0,001 m như rich8. Mọi điểm hợp lệ được giữ; không cap số điểm/pillar,
không cap số pillar/frame và không padding điểm.
XY indexing dùng cùng phép floor-division float32 như rich8, kể cả tại biên ô.

Dataset chuẩn bị feature sau augmentation; Linear/BN nằm trong model nên nhận
gradient từ detection loss và được lưu cùng checkpoint. Pooling không phụ thuộc
thứ tự điểm. BN dùng các điểm hợp lệ trong batch; với toàn batch chỉ có một điểm,
dùng running statistics; với batch rỗng, BEV bằng zero và BN không cập nhật.

Input model là dictionary `features [N,10]`, `pillar_indices [N]`,
`coords [K,3]` theo thứ tự batch/y/x và `batch_size`. Khi dùng DataLoader trực tiếp,
chọn `collate_fn=collate_detector_batch` từ `core.datasets.dataset`. CLI train và
PyTorch evaluator tự xử lý đường input này. `common.input_shape(config)` vẫn
trả shape **BEV vào backbone**, không phải shape packed input của pillar32.

## Chạy hai cấu hình đối chứng

Hai file sau giữ cùng cấu hình ngoài encoder: backbone [3,4,2], attention/context
none, fusion 24, head 16, baseline loss, augmentation standard, seed 42,
AdamW 7e-4, 50 epoch, batch train 16/val 32, BF16. AP được đo trên toàn validation
mỗi epoch bằng cùng local BEV R40 protocol và cùng decode settings.

- `configs/experiments/encoders/rich8.json`: **648.313** parameters.
- `configs/experiments/encoders/pillar32.json`: **655.609** parameters; encoder có
  384 parameters, stem tăng 6.912. Tổng tăng **7.296 (1,125%)**.

Dùng dữ liệu processed ở `data/kitti/processed` và raw KITTI ở `data/kitti/raw`,
theo hướng dẫn prepare trong `tools/kitti_training_pipeline/README.md`. Chạy từ
repository root trên máy có PyTorch CUDA:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/experiments/encoders/rich8.json \
  --detector-root detector --output-root artifacts/kitti \
  --run-name encoder-rich8-s42 --seed 42

python3 tools/kitti_training_pipeline/train.py \
  --config configs/experiments/encoders/pillar32.json \
  --detector-root detector --output-root artifacts/kitti \
  --run-name encoder-pillar32-s42 --seed 42
```

Nếu dữ liệu ở nơi khác, truyền cùng `--override-json` cho cả hai run:

```json
{"data":{"kitti":{"location":"/path/to/processed"}},"evaluation":{"kitti_root":"/path/to/raw"}}
```

Checkpoint được chọn theo AP ở `selected/best_ap.pt`; dùng cùng seed khác
(ví dụ 43,44) và run name mới để kiểm tra độ lặp lại. Không so checkpoint best-loss
với best-AP. Giữ cùng physical batch, augmentation và AP interval; nếu phải giảm
batch do VRAM, thay đổi cả hai comparator tương ứng và ghi rõ điều kiện.

Đánh giá lại checkpoint pillar32 (rich8 dùng lệnh tương tự với run tương ứng):

```bash
python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name pillar32-s42 \
  --config artifacts/kitti/encoder-pillar32-s42/config.resolved.json \
  --model artifacts/kitti/encoder-pillar32-s42/selected/best_ap.pt \
  --backend pytorch --detector-root detector --kitti-root data/kitti/raw \
  --split splits/kitti/val.txt \
  --output artifacts/kitti/encoder-pillar32-s42/evaluation.json --device cuda
```

## Notebook

Trong notebook standard chọn `PRESET = "ENCODER_RICH8"` hoặc
`PRESET = "ENCODER_PILLAR32"`, dùng cùng `AUGMENTATION = "standard"` và các
runtime controls. Preset model/loss không lấy custom controls; BOX_MODE,
checkpoint selection, batch, seed và augmentation vẫn theo runtime controls.
Chọn cùng `BOX_MODE = "bev"`, `EVALUATION_MODES = ["local_bev"]`,
`CHECKPOINT_SELECTION = "ap"` để đối chiếu các run BEV hiện tại.

Nếu muốn dùng backbone/head custom đang nghiên cứu, giữ `PRESET = "custom"`,
chỉ đổi `BEV_ENCODING = "pillar32"`. Notebook tự đặt backend torch và 32 kênh.
Không dùng chung run name/checkpoint của rich8 để resume pillar32; resume yêu cầu
cùng semantic identity. Để so encoder từ đầu, không warm-start một bên.

## Chi phí và giới hạn

- Số tham số nhỏ không đảm bảo latency nhỏ. Linear chạy trên mọi điểm, pooling
  chạy trên occupied pillars, sau đó CNN xử lý BEV 32 kênh.
- Ở lưới 800×704, riêng BEV FP32/frame là 68,75 MiB, so với rich8 17,19 MiB.
  Đây chưa gồm point activations, backbone activations, gradients và optimizer.
- CPU feature preparation được tính trong preprocessing; learned MLP, pooling
  và scatter được tính trong model latency. Evaluator ghi `mean_input_bytes`
  thực tế của packed tensors; `input_bytes_fp32` là null vì input biến độ dài.
- Bắt đầu với eager (`compile_model=false`). Variable point/pillar counts và
  kiểm tra index có thể gây graph breaks/recompilation khi dùng torch.compile.
- ONNX/TensorRT hiện có chỉ nhận một dense tensor nên pillar32 được từ chối rõ
  ràng; train/resume/PyTorch evaluation là đường so sánh được hỗ trợ. Dense-shape
  profiler cũng từ chối packed input; đếm tham số bằng `model_parameter_report`.
- Đổi encoder không tự đổi BEV head thành full 3D head. AP sau huấn luyện và
  GPU latency cần thí nghiệm thực tế; synthetic smoke tests không chứng minh
  pillar32 tốt hơn rich8.

## Kiểm tra

```bash
python3 -m pytest tests/test_pillar_encoder.py -q
python3 -m pytest tests/ -q
```

Test gồm point features/ROI, giữ toàn bộ điểm, collation/permutation/batch
isolation, frame rỗng/một điểm, FP32/BF16 gradients, detector-loss parameter
updates, CLI train với AP, resume và DataLoader spawn workers.
