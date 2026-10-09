# So sánh rich8, pillar32, pillar_rich và pillar_rich_gate

`pillar32`, `pillar_rich` và `pillar_rich_gate` là các encoder learned bổ sung;
`configs/config.json` vẫn mặc định rich8. Đây là thử nghiệm biểu diễn đầu vào,
không phải triển khai toàn bộ detector PointPillars.

## pillar_rich_gate: rich8 điều khiển gate của learned max24

| Encoder | Learned pooling / refinement | Preset |
|---|---|---|
| `pillar_rich` | max-only | `ENCODER_PILLAR_RICH` |
| `pillar_rich_gate` | max, sau đó rich-conditioned channel gate | `ENCODER_PILLAR_RICH_GATE` |

Chọn `name="pillar_rich_gate"` là đủ. Gate có kiến trúc cố định, không cần thêm
flag; `pooling` nếu khai báo phải là `max`. Output vẫn **32 kênh**:

```text
Point features → Linear(10,24) → BN → ReLU → max m [K,24] ─────────┐
                                              │                 │
rich8 r [K,8] ─────────────────── concat [m,r] [K,32]             │
                                              ↓                 │
                                 Linear(32,4) → ReLU             │
                                              ↓                 │
                                    Linear(4,24)                 │
                                              ↓                 │
                                      2 * sigmoid → g ── m * g ─┘
                                                                  ↓
                                              concat [r, m*g] [K,32]
                                                                  ↓
                                                        scatter NCHW
```

Rich8 gồm ba height occupancy, z max/mean, intensity max/mean và log-density;
chúng đã được chuẩn bị trên cùng điểm sau augmentation, không cần thêm lượt
mean pooling trên embedding từng điểm. Gate chạy riêng trên mỗi occupied pillar,
không gom cả batch, không có attention theo không gian BEV hoặc trên từng điểm.
Rich8 giữ nguyên ở kênh 0–7; gate chỉ tác động learned24 ở kênh 8–31.

Hai Linear đều có bias: gate thêm **252 parameters**, tổng point encoder **540**
(288 point embedding/BN +252 gate). Toàn detector comparison preset có **655.765**.
Weight và bias Linear cuối khởi tạo zero; Linear đầu khởi tạo bình thường.
Do đó g ban đầu bằng 1, output bằng baseline max-only. Constructors của gate
nằm trong `torch.random.fork_rng(devices=[])` để không làm lệch khởi tạo
backbone/head dù dùng cùng seed. Khi gate học, hệ số nằm trong (0,2), có thể tăng
hoặc giảm feature. Đây là biến thể gating kiểu SE, không phải SE chuẩn trên BEV.

MLP gate theo autocast của model; sigmoid và scaling dùng FP32 rồi cast về dtype
embedding. Frame rỗng cho BEV zero và giữ backward connection tới mọi parameter;
batch một điểm dùng running statistics của BN. Mọi điểm ROI hợp lệ được giữ,
không sampling/cap/padding. Scatter vẫn trực tiếp vào NCHW.

### Chọn trong notebook / CLI

```python
PRESET = "ENCODER_PILLAR_RICH_GATE"
AUGMENTATION = "standard"
```

Nếu giữ model/loss custom:

```python
PRESET = "custom"
BEV_ENCODING = "pillar_rich_gate"
```

Resolver điền 32 kênh/backend torch kể cả notebook cũ trả `out_channels=None`.
Biến thể có checkpoint identity và run name riêng. Bắt đầu run mới; không resume
checkpoint max-only hoặc ECA vào gate. Encoder/preset/config ECA cũ đã được bỏ,
không được tự đổi nghĩa thành gate. Các kết quả ECA trước đây là thí nghiệm lịch sử.
Backbone `local_attention="eca"` là chức năng riêng và vẫn được hỗ trợ.

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/experiments/encoders/pillar_rich_gate.json \
  --detector-root detector --output-root artifacts/kitti \
  --run-name encoder-pillar-rich-gate-s42 --seed 42
```

Config giữ cùng recipe với `pillar_rich.json`: 50 epoch, AdamW 7e-4,
warmup4/cosine, BF16, augmentation standard, batch train16/val32, AP mỗi epoch.
Archive trước đây dùng val16 và AP mỗi10 epoch; nếu cần cùng lịch đo thì override
`val.physical_batch_size=16` và `train.checkpoint_selection.ap_every=10`.

### Đo tốc độ

```bash
python3 tools/benchmarks/benchmark_pillar_encoders.py \
  --pointcloud /path/to/KITTI/training/velodyne/000000.bin \
  --device cuda --precision bf16 --warmup 10 --iterations 50 \
  --output artifacts/pillar-speed/gate-bf16.json
```

Benchmark so rich8, pillar_rich và pillar_rich_gate với cùng backbone/head,
batch1, precision, input và seed. Báo cáo prep, transfer, forward, p50/p95,
input hash và parameter counts. Random weights, không đo AP/I/O/decode/NMS.
Có thể smoke-test bằng `--synthetic --points 1000 --device cpu`.
Gate không cần mean reduction mới nhưng MLP/scaling vẫn có chi phí. Dense BEV32
vẫn lớn hơn rich8. Chưa có AP hoặc số đo GPU của gate; số đo ECA cũ không phải
bằng chứng tốc độ của thiết kế này.

```bash
python3 -m pytest tests/test_pillar_rich_gate.py \
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
`"ENCODER_PILLAR_RICH"` hoặc `"ENCODER_PILLAR_RICH_GATE"`, dùng cùng
`AUGMENTATION = "standard"` và các
runtime controls. Preset model/loss không lấy custom controls; BOX_MODE,
checkpoint selection, batch, seed và augmentation vẫn theo runtime controls.
Chọn cùng `BOX_MODE = "bev"`, `EVALUATION_MODES = ["local_bev"]`,
`CHECKPOINT_SELECTION = "ap"` để đối chiếu các run BEV hiện tại.

Nếu muốn dùng backbone/head custom đang nghiên cứu, giữ `PRESET = "custom"`,
chỉ đổi `BEV_ENCODING = "pillar32"`, `"pillar_rich"` hoặc `"pillar_rich_gate"`. Resolver đặt backend
torch và 32 kênh kể cả cell notebook cũ có bảng số kênh chưa chứa encoder mới;
không cần sửa notebook để dùng pillar_rich. Các giá trị sai khai báo rõ ràng vẫn
báo lỗi. Giữ cùng `NOTEBOOK_OVERRIDES` khi đối chiếu các encoder.
Không dùng chung run name/checkpoint của rich8 để resume pillar32; resume yêu cầu
cùng semantic identity. Để so encoder từ đầu, không warm-start một bên.
Không resume pillar32 vào pillar_rich: dù BEV đều 32 kênh, semantic identity và
trọng số nhánh học khác nhau. Chọn run name mới cho từng encoder.

Resolver hỗ trợ cell notebook cũ có bảng số kênh chưa chứa `pillar32`:
`out_channels=None` được điền thành 32 và backend thiếu được đặt thành torch
cho cả pillar32, pillar_rich và pillar_rich_gate.
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
