# So sánh rich8, pillar32, pillar_rich và pillar_rich_eca

`pillar32`, `pillar_rich` và `pillar_rich_eca` là encoder bổ sung; `configs/config.json` vẫn mặc định rich8.
Đây là phép thử biểu diễn đầu vào, không phải triển khai toàn bộ detector PointPillars.

## pillar_rich_eca: max–mean + ECA theo pillar

Hai biến thể có tên encoder riêng:

| Encoder | Learned pooling | Preset |
|---|---|---|
| `pillar_rich` | max-only | `ENCODER_PILLAR_RICH` |
| `pillar_rich_eca` | max–mean, gate ECA | `ENCODER_PILLAR_RICH_ECA` |

Chỉ cần chọn `name="pillar_rich_eca"` là bật gate, không cần thêm flag pooling.
`eca_kernel_size` mặc định 3; có thể khai báo một số nguyên dương lẻ khác.

Biến thể mới giữ nguyên rich8 8 kênh và learned24, cùng BEV **32 kênh**:

```text
Point features → Linear(10,24) → BN → ReLU
                       ├─ max theo pillar → m [K,24] ────────┐
                       └─ mean FP32       → μ [K,24] ────────┤
                                           └─ ECA → α ─────┤
                                       α*m + (1−α)*μ [K,24]
                                                 ↓ concat rich8
                                             scatter NCHW
```

`α = sigmoid(Conv1d_k(μ))`, với kernel mặc định 3, không bias. Gate thay đổi
theo từng occupied pillar và tương tác cục bộ theo chiều kênh. Đây là gate trộn
max–mean dùng cơ chế kiểu ECA, khác ECA gốc dùng global spatial average trên BEV.
Không dùng attention giữa các điểm. Gate chỉ tác động learned24, giữ nguyên rich8.

Nhánh learned có **291 tham số** (288 embedding/BN +3 kernel); toàn detector
comparison preset có **655.516**. Kernel khởi tạo zero nên α ban đầu là 0,5.
Mean, gate và phép trộn tính FP32 dưới autocast; output cast về dtype embedding.
Một điểm có max=mean; frame rỗng giữ backward connection tới mọi tham số.
Mọi điểm ROI hợp lệ vẫn được giữ, không sampling/cap/padding.

Để giảm chi phí, channel convolution ngắn được tính bằng shifted `addcmul`
(cùng cross-correlation và gradients như Conv1D), trộn bằng `lerp`, không tạo
dense mean BEV. Encoder scatter trực tiếp vào NCHW thay vì NHWC rồi copy toàn
BEV; bounds validation gộp thành một device predicate. Đây là tối ưu thực thi,
không phải bằng chứng tốc độ GPU gần rich8.

### Chọn trong notebook / CLI

```python
PRESET = "ENCODER_PILLAR_RICH_ECA"
AUGMENTATION = "standard"
```

Nếu giữ backbone/head/loss custom, chọn:

```python
PRESET = "custom"
BEV_ENCODING = "pillar_rich_eca"
NOTEBOOK_OVERRIDES = {
    "data": {"bev_encoding": {
        "eca_kernel_size": 3
    }}
}
```

Gộp phần trên vào `NOTEBOOK_OVERRIDES` hiện có nếu bạn đã đặt các override khác.
Resolver hỗ trợ notebook cũ trả `out_channels=None`, tự điền 32 và backend torch.
Biến thể mới có encoding identity và run-name chứa `pillar_rich_eca` / `eca_k3` riêng;
phải train run mới, không resume checkpoint max-only vào biến thể này.
`pillar_rich` luôn là **max-only**, giữ identity và state keys lịch sử.
Config cũ ghi `name="pillar_rich", pooling="max_mean_eca"` phải đổi name thành
`pillar_rich_eca`; không sửa metadata checkpoint để ép resume qua tên encoder.
Preset dài `ENCODER_PILLAR_RICH_MAX_MEAN_ECA` vẫn là alias của preset mới và
luôn resolve thành encoder `pillar_rich_eca`.
Không bật `local_attention="eca"` nếu mục tiêu là ablation encoder:
flag đó đặt attention ở backbone stride4.

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/experiments/encoders/pillar_rich_eca.json \
  --detector-root detector --output-root artifacts/kitti \
  --run-name encoder-pillar-rich-eca-s42 --seed 42
```

Config mới giữ cùng recipe với `pillar_rich.json`, gồm 50 epoch, AP mỗi epoch,
batch train16/val32. Khi so với archive đã chạy AP mỗi10 epoch/val16, cần override
cùng AP interval và val batch cho các comparator, hoặc chạy lại cùng recipe.

### Đo tốc độ thay vì suy ra từ số tham số

```bash
python3 tools/benchmarks/benchmark_pillar_encoders.py \
  --pointcloud /path/to/KITTI/training/velodyne/000000.bin \
  --device cuda --precision fp32 --warmup 10 --iterations 50 \
  --output artifacts/pillar-speed/fp32.json

python3 tools/benchmarks/benchmark_pillar_encoders.py \
  --pointcloud /path/to/KITTI/training/velodyne/000000.bin \
  --device cuda --precision bf16 --warmup 10 --iterations 50 \
  --output artifacts/pillar-speed/bf16.json
```

Benchmark giữ cùng backbone/head từ config, so `rich8`, `pillar_rich` và
`pillar_rich_eca` ở batch1. Ghi hardware/precision, input hash, mean/p50/p95,
preprocess/contiguous transfer/model, model-only và ratio so rich8. Dùng random
weights và lặp một point cloud: **không đo AP, không tính I/O/decode/NMS và
không thay thế đánh giá checkpoint**. Đo nhiều scene có mật độ điểm khác nhau,
chạy nhiều lần trong cùng runtime; so FP32 với FP32 và BF16 với BF16. Có thể dùng
`--synthetic --points 100000 --device cpu` để kiểm tra công cụ, nhưng điểm uniform
không đại diện mật độ/số điểm trên pillar của KITTI.

Dense BEV vẫn 32 kênh nên stem/activation đắt hơn rich8. BF16 có thể giúp trên
GPU hỗ trợ, nhưng cần đo ratio thực tế và AP với cùng precision. Nếu giữ yêu cầu
32 kênh/full point retention, không thể cam kết model-only latency bằng rich8;
pipeline tổng cũng phụ thuộc CPU prep, transfer và NMS. Gate thêm3 tham số không
có nghĩa segment mean hoặc bộ nhớ FP32 có chi phí bằng zero.

### Số đo ban đầu trên RTX 4050

Đo ngày 2026-10-09 trên **NVIDIA GeForce RTX 4050 Laptop GPU**, PyTorch 2.11.0,
batch1, FP32, một CPU thread, warmup10 và 50 lượt/frame. Cùng comparison
backbone/head, random weights, KITTI training frames 000000/000001/000010;
đo từng frame tuần tự. Đơn vị ms, giá trị mean:

| Frame | rich8 forward | pillar max forward | max–mean ECA forward | rich8 prep+transfer+forward | max–mean ECA prep+transfer+forward |
|---|---:|---:|---:|---:|---:|
| 000000 | 6.783 | 7.830 | 7.998 | 16.104 | 17.682 |
| 000001 | 6.801 | 7.852 | 8.046 | 16.235 | 17.884 |
| 000010 | 6.730 | 7.773 | 7.962 | 15.675 | 16.469 |

Trong các lượt này, variant mới có forward latency **cao hơn rich8 khoảng
18%**, prep+transfer+forward cao hơn **5–10%**. Phần mean/gate thêm khoảng
**0,17–0,19 ms** vào forward so với pillar max. BF16 cùng frame000000:
rich8 forward5,882 ms, variant6,777 ms; prep+transfer+forward14,748 và16,496 ms.
Đây là số đo ban đầu trên ba frame, không có I/O/decode/NMS; chưa chứng minh
latency toàn dataset, tốc độ train, AP hoặc tốc độ trên L4. Báo cáo JSON với
input/config hashes, p50/p95 và metadata nằm trong
`artifacts/pillar_max_mean_eca/rtx4050_kitti_*_*.json` (không đưa vào git).

```bash
python3 -m pytest tests/test_pillar_max_mean_eca.py \
  tests/test_pillar_encoder.py tests/test_pillar_rich_encoder.py -q
```

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

## pillar_rich: rich8 + learned24

`pillar_rich` dùng cùng mười point feature và cùng Linear/BN/ReLU/max-pooling,
nhưng nhánh học chỉ xuất 24 kênh. Tám thống kê rich8 được tính trên cùng điểm sau
augmentation/ROI, với cùng XY indexing, height bands, intensity_scale và density_norm:

```text
Điểm trong từng pillar
  ├─ rich8 stats [K,8]                         ─┐
  └─ Linear(10,24) + BN + ReLU → max [K,24]    ─┤
                                               ↓ concatenate [K,32]
                                               ↓ scatter một lần
                                          BEV [B,32,H,W] → backbone/head
```

Kênh 0–7 giữ đúng rich8; kênh 8–31 là feature học được. Packed input bổ sung
`rich_features [K,8]`, theo đúng thứ tự `coords`. Không tạo thêm dense rich8 BEV
trong preprocessing. Rich8 stats không có tham số học; nhánh học có **288 tham số**.
Ở FP32, tám kênh đầu khớp chính xác với rich8 độc lập trên cùng point cloud float32.
Khi autocast BF16, rich8 được cast theo dtype nhánh học trước concat, nên có rounding
BF16 như input convolution trong rich8. Occupied pillars và empty frames dùng cùng
quy tắc với pillar32. `density_norm` có tác dụng và nằm trong checkpoint identity của
pillar_rich; ở pillar32 nó không có tác dụng.

Nhánh rich8 giữ mật độ/occupancy/mean mà max pooling không bảo đảm giữ. Ví dụ nhân
đôi toàn bộ điểm không đổi learned max features khi eval, nhưng đổi kênh log_density
(trước saturation). Đây là lý do thử hybrid; không phải bằng chứng tăng AP.

## Chạy ba cấu hình đối chứng

Ba file sau giữ cùng cấu hình ngoài encoder: backbone [3,4,2], attention/context
none, fusion 24, head 16, baseline loss, augmentation standard, seed 42,
AdamW 7e-4, 50 epoch, batch train 16/val 32, BF16. AP được đo trên toàn validation
mỗi epoch bằng cùng local BEV R40 protocol và cùng decode settings.

- `configs/experiments/encoders/rich8.json`: **648.313** parameters.
- `configs/experiments/encoders/pillar32.json`: **655.609** parameters; encoder có
  384 parameters, stem tăng 6.912. Tổng tăng **7.296 (1,125%)**.
- `configs/experiments/encoders/pillar_rich.json`: **655.513** parameters; nhánh học có
  288 parameters. Tổng ít hơn pillar32 **96**, tăng **7.200 (1,111%)** so với rich8.

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

python3 tools/kitti_training_pipeline/train.py \
  --config configs/experiments/encoders/pillar_rich.json \
  --detector-root detector --output-root artifacts/kitti \
  --run-name encoder-pillar-rich-s42 --seed 42
```

Nếu dữ liệu ở nơi khác, truyền cùng `--override-json` cho cả ba run:

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

Trong notebook chọn `PRESET = "ENCODER_RICH8"`, `"ENCODER_PILLAR32"`,
`"ENCODER_PILLAR_RICH"` hoặc `"ENCODER_PILLAR_RICH_ECA"`, dùng cùng
`AUGMENTATION = "standard"` và các
runtime controls. Preset model/loss không lấy custom controls; BOX_MODE,
checkpoint selection, batch, seed và augmentation vẫn theo runtime controls.
Chọn cùng `BOX_MODE = "bev"`, `EVALUATION_MODES = ["local_bev"]`,
`CHECKPOINT_SELECTION = "ap"` để đối chiếu các run BEV hiện tại.

Nếu muốn dùng backbone/head custom đang nghiên cứu, giữ `PRESET = "custom"`,
chỉ đổi `BEV_ENCODING = "pillar32"`, `"pillar_rich"` hoặc `"pillar_rich_eca"`. Resolver đặt backend
torch và 32 kênh kể cả cell notebook cũ có bảng số kênh chưa chứa encoder mới;
không cần sửa notebook để dùng pillar_rich. Các giá trị sai khai báo rõ ràng vẫn
báo lỗi. Giữ cùng `NOTEBOOK_OVERRIDES` khi đối chiếu các encoder.
Không dùng chung run name/checkpoint của rich8 để resume pillar32; resume yêu cầu
cùng semantic identity. Để so encoder từ đầu, không warm-start một bên.
Không resume pillar32 vào pillar_rich: dù BEV đều 32 kênh, semantic identity và
trọng số nhánh học khác nhau. Chọn run name mới cho từng encoder.

Resolver hỗ trợ cell notebook cũ có bảng số kênh chưa chứa `pillar32`:
`out_channels=None` được điền thành 32 và backend thiếu được đặt thành torch
cho cả pillar32, pillar_rich và pillar_rich_eca.
Giá trị sai được khai báo rõ ràng (ví dụ 8 kênh hoặc backend numpy) vẫn báo lỗi.
Nếu dùng code resolver cũ, thêm đoạn sau trước `resolve_notebook_config(...)`:

```python
if BEV_ENCODING == "pillar32":
    custom_overrides["data"]["bev_encoding"].update(
        out_channels=32, version=1, backend="torch"
    )
```

## Chi phí và giới hạn

- Số tham số nhỏ không đảm bảo latency nhỏ. Linear chạy trên mọi điểm, pooling
  chạy trên occupied pillars, sau đó CNN xử lý BEV 32 kênh.
- Ở lưới 800×704, riêng BEV FP32/frame là 68,75 MiB, so với rich8 17,19 MiB.
  Đây chưa gồm point activations, backbone activations, gradients và optimizer.
- CPU feature preparation được tính trong preprocessing; learned MLP, pooling
  và scatter được tính trong model latency. Evaluator ghi `mean_input_bytes`
  thực tế của packed tensors; `input_bytes_fp32` là null vì input biến độ dài.
- pillar_rich có cùng bộ nhớ BEV 32 kênh với pillar32, giảm point activation từ
  32 xuống 24 kênh nhưng bổ sung chuẩn bị/chuyển dữ liệu rich8 stats. Phải đo latency
  thực tế; số tham số nhỏ hơn không chứng minh chạy nhanh hơn.
- Bắt đầu với eager (`compile_model=false`). Variable point/pillar counts và
  kiểm tra index có thể gây graph breaks/recompilation khi dùng torch.compile.
- ONNX/TensorRT hiện có chỉ nhận một dense tensor nên pillar32 được từ chối rõ
  ràng cho cả hai learned encoders; train/resume/PyTorch evaluation là đường so
  sánh được hỗ trợ. Dense-shape
  profiler cũng từ chối packed input; đếm tham số bằng `model_parameter_report`.
- Đổi encoder không tự đổi BEV head thành full 3D head. AP sau huấn luyện và
  GPU latency cần thí nghiệm thực tế; synthetic smoke tests không chứng minh
  pillar32 tốt hơn rich8.

## Kiểm tra

```bash
python3 -m pytest tests/test_pillar_encoder.py tests/test_pillar_rich_encoder.py -q
python3 -m pytest tests/ -q
```

Test gồm point features/ROI, giữ toàn bộ điểm, collation/permutation/batch
isolation, frame rỗng/một điểm, FP32/BF16 gradients, detector-loss parameter
updates, CLI train với AP, resume và DataLoader spawn workers.
