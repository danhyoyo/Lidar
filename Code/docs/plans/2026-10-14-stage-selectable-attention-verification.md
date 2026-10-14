# Stage-selectable Focal / SimAM — implementation và verification

Ngày: 2026-10-14. Runtime và notebook đã triển khai theo
[spec](../superpowers/specs/2026-10-14-stage-selectable-attention-design.md) và
[plan](2026-10-14-stage-selectable-attention-implementation-plan.md).

## Hành vi đã triển khai

Một instance Focal và một instance local attention có thể chọn vị trí độc lập:

| Control | Giá trị | Ý nghĩa |
| --- | --- | --- |
| `FOCAL_STAGE` | `"c3"`, `"c4"`, `"c5"` | Vị trí Focal khi `C4_CONTEXT="focal"` |
| `LOCAL_ATTENTION_STAGE` | `"c3"`, `"c4"`, `"c5"` | Vị trí SimAM khi `LOCAL_ATTENTION="simam"`; cũng hỗ trợ ECA |
| `ATTENTION_ORDER` | `"focal_then_local"`, `"local_then_focal"` | Thứ tự khi hai module cùng stage |

| Tên stage | Module trunk | Stride | Channels |
| --- | --- | --- | --- |
| C3 / Stage 2 | `stage2` | 4 | 48 |
| C4 / Stage 3 | `stage3` | 8 | 96 |
| C5 / Stage 4 | `stage4` | 16 | 128 |

Attention được áp dụng sau khi stage hoàn thành. Tensor sau attention được dùng
cho cả stage tiếp theo và lateral tương ứng; C5 cấp cho lateral cuối hoặc RC neck.
Đầu ra backbone vẫn stride 4. Taps `attention_c3_output`, `attention_c4_output`,
`attention_c5_output` chỉ capture tensor, không có tham số.

Registry/resolver, constructor, checkpoint identity, run names, profile và các
công cụ capture/diagram đều sử dụng vị trí thực tế. Prefix lịch sử `c4_context`
và `c3_light_attention` giữ nguyên để load checkpoint cũ; prefix không còn là
nguồn xác định vị trí. Công thức Focal, SimAM, neck, heads và loss giữ nguyên.

## Sử dụng trên notebook

Chọn `BACKBONE="mobilepixornext"` và `PRESET="custom"` trong cell cấu hình của
Standard hoặc Robust. Ví dụ Focal C5 và SimAM C3:

```python
PRESET = "custom"
C4_CONTEXT = "focal"
FOCAL_STAGE = "c5"
LOCAL_ATTENTION = "simam"
LOCAL_ATTENTION_STAGE = "c3"
ATTENTION_ORDER = "focal_then_local"
```

Để cùng C4, đặt cả hai stage thành `"c4"`. Đổi order thành `"local_then_focal"`
để chạy SimAM trước Focal. Tắt riêng bằng `C4_CONTEXT="none"` hoặc
`LOCAL_ATTENTION="none"`; tắt cả hai cho baseline. Order chỉ được ghi vào resolved
config khi cả hai module đang bật và cùng stage. Preset/config-file mode tiếp tục
dùng cấu hình đã chọn; stage controls chỉ áp dụng trong custom mode.

Notebook in summary stage/stride/channels/order và semantic hash. Custom merge
dọn field kế thừa của module đã tắt hoặc order không còn áp dụng; raw config chứa
field không hợp lệ vẫn bị từ chối. Config snapshot đã train có placement khác
bị từ chối ghi đè; dùng run mới khi đổi topology.

Không tạo experiment JSON, manifest hoặc preset mới, không sửa file dưới
`Code/configs/`. Snapshot của run vẫn được lưu theo pipeline hiện có.

## Checkpoint và tương thích

Topology mặc định (Focal C4, local attention C3, kể cả các module bị tắt) giữ
semantic metadata version 1. Topology di chuyển/chung stage dùng version 2.
Tests xác nhận config cũ và explicit legacy placement có cùng hash, state keys,
khởi tạo và đầu ra. Topology khác stage/order có identity khác; resume/evaluation
reject mismatch trước khi mutate model/optimizer, kể cả khi shapes giống nhau.

Full checkpoint roundtrip được kiểm tra cùng optimizer, scheduler và criterion
sau một bước backward/update. TorchScript vẫn qua regression tests.

## Bằng chứng kiểm tra

Interpreter: `/home/duyennh/miniconda3/envs/AI_env/bin/python`, Python 3.14.4,
PyTorch 2.11.0. Bộ test cuối chạy ngoài sandbox để DataLoader worker có thể tạo
socket và PyTorch truy cập GPU. Không khởi chạy training trên dataset thật.

- Focused suite: **433 passed, 0 skipped**, 21.27 giây. Gồm resolver, legacy
  compatibility, routing, training, checkpoint, notebook, presets, capture,
  visualization, diagram và TorchScript regression.
- Coverage **19 topology**: baseline; Focal-only ở 3 stage; SimAM-only ở 3 stage;
  hai module khác stage ở 6 tổ hợp; cùng stage ở 3 stage × 2 order.
- Từng topology qua actual forward, grouped loss/backward, optimizer uniqueness,
  checkpoint identity, và config cell thật của cả hai notebook.
- Meta tensors kiểm tra shape KITTI đầy đủ và số tham số Focal 15.472 / 25.120 /
  31.552 tại C3/C4/C5; SimAM có 1 tham số. RC neck, ECA và shared 3D header có
  regression coverage riêng.
- **8 CUDA AMP checks passed**: FP16/BF16 cho các placement đại diện, loss và
  gradient hữu hạn. Điều này xác nhận smoke correctness, không phải parity AP.
- Full suite: **2.146 passed, 19 skipped, 4 failed**, cùng 15 subtests passed,
  233.86 giây. Cả 4 failure ở đường KITTI reference runtime bên dưới.
- Code review độc lập không còn Blocker/Major. `git diff --check` qua; không có
  diff trong `Code/configs/`, `focal_context.py` hoặc `light_attention.py`.

Focused verification từ `Code/`:

```bash
export ATTENTION_PY=/home/duyennh/miniconda3/envs/AI_env/bin/python
export PYTHONPATH=.:detector:tools/kitti_training_pipeline:tests
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MPLCONFIGDIR=/tmp/lidar-attention-matplotlib
export IPYTHONDIR=/tmp/lidar-attention-ipython
export NUMBA_CACHE_DIR=/tmp/lidar-stage-numba
"$ATTENTION_PY" -m pytest \
  tests/test_backbone_feature_config.py \
  tests/test_attention_stage_compatibility.py \
  tests/test_attention_stage_routing.py tests/test_attention_stage_training.py \
  tests/test_attention_stage_checkpoint.py tests/test_attention_stage_notebook.py \
  tests/test_backbone_attention_integration.py tests/test_grouped_checkpoint.py \
  tests/test_under1m_notebook_controls.py tests/test_under1m_notebook_config.py \
  tests/test_attention_presets.py tests/test_robust_notebook_setup.py \
  tests/test_kitti_pipeline_visualization.py \
  tests/test_attention_placement_visualization.py tests/test_backbone_diagram.py \
  tests/test_mobilepixornext_backbone.py -q --tb=short
"$ATTENTION_PY" -m pytest tests/ -q --tb=short
```

## Actual P16 replay và visualization

Nguồn là checkpoint `selected/best_ap.pt` của P16 đã dùng cho visualization trước,
SHA256 `5224f786b661e494fec5d585119a8951aaa6dcc0933492e3985d9c34d1eec081`.
Frame `000134` được encode lại và chạy model production trên CPU:

- Embedded config/identity được validate; strict state-dict load qua.
- Final features, logits và probabilities khớp frozen reference NPZ với
  `rtol=1e-5, atol=1e-6`.
- Model state và checkpoint nguồn không bị thay đổi.
- Receipt: `artifacts/attention_stage_verification/p16_replay.json` từ repo root.
- Gallery actual checkpoint:
  `artifacts/attention_stage_verification/visualization/000134/trained_5224f786b661/index.html`.

`compare_attention_placements.py --mode checkpoint` chạy đúng topology trong
checkpoint, capture refined taps và lưu heatmaps/NPZ/metadata. Intervention mode
cũ vẫn ghi rõ các placement chưa train; không dùng heatmap để kết luận stage
tốt nhất về AP. Người dùng sẽ train lại để so sánh chất lượng các placement.

## Giới hạn full suite và chỉnh sửa phát hiện khi kiểm tra

Probe ngoài sandbox xác nhận `torch.cuda.is_available() == True` nhưng
`numba.cuda.is_available() == False`. Vì vậy 4 tests vốn yêu cầu KITTI reference
CUDA thất bại; parity subprocess báo 16 skipped. Không đổi evaluator sang CPU
fallback hay bỏ kiểm tra reference để làm full suite xanh:

1. `test_benchmark_evidence.py::test_actual_reference_checkpoint_evidence_retains_mode_both_samplings_and_pin[bev]`
2. Cùng test trên với `[3d]`.
3. `test_unified_notebook_workflow.py::test_actual_notebook_cells_complete_a_small_run_with_direct_commands[3d]`
4. `test_unified_notebook_workflow.py::test_reference_runtime_receipt_requires_actual_unskipped_pinned_cuda_parity`

Các đường production evaluator/reference không có diff. AP evaluation cần môi
trường Numba CUDA/reference tương thích; verification này chưa đo KITTI AP.

Follow-up lỗi import trên Colab: config cell trước đây reload top-level
`notebook_config` nhưng import hai placement helpers qua package alias riêng,
có thể còn cache bản cũ. Cả hai notebook và generator nay dùng cùng top-level
module đã reload. Regression tái hiện đúng ImportError ở cả hai notebook trước
sửa; sau sửa bộ notebook/import/setup có **123 passed**. Đồng bộ notebook phải
đi cùng source Python mới; sửa import không bổ sung helper vào file cũ trên Drive.

Follow-up setup Robust: source checkout được kiểm tra bằng AST trước import
(helpers, resolver fields, constructor arguments). Checkout cũ trên Drive hoặc
`/content/Lidar` được giữ nguyên; Colab clone source mới vào thư mục runtime
trống và kiểm tra lại. Setup ưu tiên source đã chọn trên `sys.path` và dọn cache
pipeline/detector để rerun không dùng module từ checkout trước. Regression bao
phủ source thiếu từng thành phần, layout `Code/Code`, clone chưa đủ source và
kernel đã nạp module cũ. Các thư mục dataset/artifact vẫn giữ cấu hình hiện có.
Bộ kiểm tra notebook/setup/import/resolver/routing/checkpoint sau bản sửa này:
**214 passed**, 4.11 giây; chỉ cell setup Robust và template tương ứng thay đổi.

Một lỗi có sẵn trong Standard được sửa khi kiểm tra workflow: cell tạo GT
database truyền CLI flags không tồn tại. Nay truyền `--processed-root`,
`--train-split`, `--val-split` đúng parser hiện tại; thuật toán sampling giữ nguyên.
Standard thay cell 3 và 5; Robust thay cell 3; generator đồng bộ config cell.

Một số tests cũ so recipe archived với master mới, hoặc chạy synthetic smoke
với AP checkpoint selection, không khớp mục đích test hiện tại. Tests được cập
nhật dùng base archived/canonical identity và loss selection cho partial smoke;
không sửa master config hay pipeline checkpoint selection mặc định.
