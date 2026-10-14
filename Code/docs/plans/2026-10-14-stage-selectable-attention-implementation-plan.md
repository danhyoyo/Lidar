# Stage-selectable Focal / SimAM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cho phép người dùng tự chọn stage cho Focal/SimAM và thứ tự khi cùng
stage trên notebook, rồi train mới bằng model production tương ứng.

**Architecture:** Mở rộng resolver bằng ba field placement/order; giữ một
instance của mỗi module với state prefix lịch sử. Một helper định tuyến module
sau từng stage và cấp cùng refined tensor cho downsample lẫn lateral. Semantic
metadata giữ version 1 cho topology cũ và dùng version 2 cho topology mới.

**Tech Stack:** Python, PyTorch, pytest, JSON notebook; không thêm dependency.

---

Trạng thái: **đã thực hiện Tasks 1–8, 2026-10-14**. Xem
[verification report](2026-10-14-stage-selectable-attention-verification.md)
cho kết quả tests, replay checkpoint và giới hạn KITTI reference runtime.
Các đoạn code bên dưới là hướng dẫn thiết kế; helper routing thực tế dùng lời gọi
module tường minh để giữ tương thích TorchScript, thay vì danh sách module động.

Phạm vi theo yêu cầu người dùng: **không tạo experiment configs, manifest,
presets hoặc config generator**.
Người dùng thay controls trên notebook. Vẫn giữ việc notebook lưu config resolved
của run, vì train/eval/checkpoint cần snapshot authoritative.

Spec nguồn: [stage-selectable attention design](../superpowers/specs/2026-10-14-stage-selectable-attention-design.md).
Nếu danh sách công việc cũ trong spec chưa đủ chi tiết, dùng plan này để thực thi.
Skill tương ứng trong phiên Codex là `executing-plans`; đây không phải yêu cầu
phải dùng Claude hoặc phải tạo subagents.

### Goal

Hoàn thành đường đi notebook controls → resolved config → model construction →
forward/backward → checkpoint identity/reload, với C3/C4/C5 và hai shared-stage orders.

### Assumptions

- Mỗi module tối đa một instance và một stage; không thêm stem/multi-stage mode.
- Focal defaults C4, local attention defaults C3. Module bật/tắt bằng field cũ.
- Công thức module, D=64/dilation defaults, neck, heads và loss giữ nguyên.
- `local_attention_stage` áp dụng cả adapter ECA hiện có; không thêm ECA recipes.
- Default notebook vẫn giữ lựa chọn loại module hiện tại; không tự bật SimAM.
- Người dùng sẽ train sau; bước implementation chỉ test/smoke, không train AP.
- Không chỉnh config files hiện có hoặc artifacts/paper figures đã lưu.

### Plan

1. Freeze compatibility baseline trước khi thay resolver.
2. Thêm stage schema, validation và legacy/v2 canonical metadata.
3. Routing production + capture taps, giữ đăng ký module/init cũ.
4. Kiểm chứng 19 topologies về routing, backward, counts và shape.
5. Stage-aware checkpoint identity, resume rejection, run names và reports.
6. Notebook merge/control flow và kiểm chứng 19 tổ hợp ngay từ config cell.
7. Capture/diagram consumers đọc vị trí thực tế, không đoán theo state prefix.
8. Docs và final verification gồm replay checkpoint P16 thực.

Mỗi task bên dưới có vòng test thất bại → triển khai nhỏ → test qua. Với phần
docs/capture có thể tái sử dụng test hiện có nếu nó đã kiểm tra đúng hành vi.
Các commit message là mốc đề xuất khi thực hiện commit, không phải lệnh phải chạy
trong bước lập plan này. Không tạo worktree hay commit chỉ để viết tài liệu.

## Environment và lệnh dùng chung

Paths trong task tương đối với `Code/`, trừ khi ghi rõ `Demo/`. Chạy từ `Code/`:

```bash
export ATTENTION_PY=/home/duyennh/miniconda3/envs/AI_env/bin/python
export PYTHONPATH=.:detector:tools/kitti_training_pipeline:tests
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MPLCONFIGDIR=/tmp/lidar-stage-attention-matplotlib
```

Máy hiện tại có PyTorch 2.11.0 trong interpreter này; CUDA không khả dụng trong
phiên kiểm tra trước. Khi thực thi, kiểm tra lại khả năng CUDA và ghi skip nếu cần.

Notebook Standard hiện có 18 cells; Robust có 22 cells. Generator
`tools/generate_robust_notebook.py` chỉ sinh Robust; không copy output đó sang Standard.

## Task 1: Freeze hành vi và identity lịch sử

**Files:**
- Create: `tests/fixtures/attention_placement_legacy_v1.json`.
- Create: `tests/test_attention_stage_compatibility.py`.
- Read: `detector/core/backbone_config.py`, `tools/kitti_training_pipeline/common.py`.
- Read: `tests/test_context_grouped_compatibility.py`, `tests/test_grouped_checkpoint.py`.

**Step 1 — Lưu metadata/hash trước khi thay code.**

Fixture chỉ là dữ liệu regression, không phải experiment config cho training.
Lưu metadata và semantic hash hiện tại cho bốn loại sau:

```python
LEGACY_CASES = {
    "none": {"backbone": "mobilepixornext"},
    "focal": {"backbone": "mobilepixornext", "c4_context": "focal"},
    "focal_simam": {
        "backbone": "mobilepixornext", "c4_context": "focal",
        "local_attention": "simam",
    },
    "focal_eca": {
        "backbone": "mobilepixornext", "c4_context": "focal",
        "local_attention": "eca",
    },
}
```

Với mỗi case, gọi resolver cũ và lưu `semantic_metadata()`/`semantic_hash` vào
fixture. Với một full config từ fixture grouped hiện có, lưu thêm
`checkpoint_identity(config)` và `generate_run_name(config)` để phát hiện việc
to_dict mới vô tình làm thay checkpoint identity/run name cũ.

**Step 2 — Viết test baseline và chạy trước khi sửa resolver.**

```python
def test_legacy_feature_metadata_is_unchanged():
    import json
    from pathlib import Path
    from core.backbone_config import resolve_backbone_features

    records = json.loads((Path(__file__).parent / "fixtures" /
                         "attention_placement_legacy_v1.json").read_text())
    for record in records["feature_cases"]:
        resolved = resolve_backbone_features(record["input"])
        assert resolved.semantic_metadata() == record["metadata"]
        assert resolved.semantic_hash == record["sha256"]
```

Run:

```bash
"$ATTENTION_PY" -m pytest tests/test_attention_stage_compatibility.py tests/test_backbone_feature_config.py tests/test_backbone_attention_integration.py tests/test_grouped_checkpoint.py -q
```

Expected: baseline qua với code cũ. Đây là fixture freeze, chưa phải test của tính
năng mới. Không regenerate golden fixture để làm test qua sau khi sửa code.

**Step 3 — Lưu reference P16 ngoài source nếu chưa có.**

Reference sẵn có ở repository root:
`Demo/outputs/attention_placements/000134/tensors/F_C4_S_C3_FS.npz`, cùng
`metadata.json`; chứa final features, logits, probabilities từ checkpoint P16 cũ.
Ghi lại hashes của checkpoint, cloud và config. Không đưa checkpoint/NPZ vào git.
Nếu artifact không còn, chạy capture bằng code cũ trước khi thay backbone.

**Deliverable:** golden identity fixture và baseline tests; mốc commit đề xuất
`test(attention): freeze legacy placement compatibility`.

## Task 2: Resolver, active fields và semantic metadata

**Files:**
- Modify: `detector/core/backbone_config.py` — `BackboneFeatureConfig`, field sets,
  typo checks, `resolve_backbone_features`.
- Modify: `tests/test_backbone_feature_config.py`.
- Modify: `tests/test_attention_stage_compatibility.py`.

**Step 1 — Viết test mới rồi chạy RED.**

```python
def test_same_stage_defaults_to_focal_then_local():
    from core.backbone_config import resolve_backbone_features
    resolved = resolve_backbone_features({
        "backbone": "mobilepixornext", "c4_context": "focal",
        "local_attention": "simam", "focal_stage": "c4",
        "local_attention_stage": "c4",
    })
    assert resolved.focal_stage == "c4"
    assert resolved.local_attention_stage == "c4"
    assert resolved.attention_order == "focal_then_local"
    assert resolved.semantic_metadata()["resolver_version"] == 2
```

Thêm parameterized tests cho:

- Active module missing stage → default C4/C3.
- Stage uppercase normalize; stage/order type sai, null, typo hoặc unsupported → lỗi.
- Field stage khi module tắt, order khi khác stage hoặc thiếu module → lỗi.
- ECA dùng stage chung, giữ kernel validation; ECA/SimAM fields xung đột vẫn lỗi.
- Non-MobilePixorNeXt không được bỏ qua field stage/order.
- Omitted shared-stage order và explicit default có cùng identity.
- Legacy omitted stages và explicit C4/C3 có cùng golden identity.
- Resolver không mutate raw config; metadata/to_dict không alias list nội bộ.

Run: `"$ATTENTION_PY" -m pytest tests/test_backbone_feature_config.py -q`.
Expected RED: new stage fields bị resolver hiện tại từ chối, không phải lỗi import môi trường.

**Step 2 — Thêm bảng stage thuần, bất biến.**

```python
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class AttentionStage:
    name: str
    backbone_module: str
    stride: int
    channels: int


ATTENTION_STAGES = MappingProxyType({
    "c3": AttentionStage("c3", "stage2", 4, 48),
    "c4": AttentionStage("c4", "stage3", 8, 96),
    "c5": AttentionStage("c5", "stage4", 16, 128),
})
```

Thêm ba fields optional vào feature dataclass: `focal_stage`,
`local_attention_stage`, `attention_order`; inactive values là None và không
được ghi trong `to_dict()`. Active values được materialize khi resolve.

Không import torch/model/backend vào resolver. Thêm fields vào feature-option
sets và typo detection; đặc biệt `focal_stage` không có prefix `c4_context_` nên
phải được thêm explicit vào nhóm inactive/other-backbone rejection.

**Step 3 — Resolve placement sau khi resolve loại module.**

- Focal active: `_choice(cfg, "focal_stage", "c4", set(ATTENTION_STAGES))`.
- Local active: `_choice(cfg, "local_attention_stage", "c3", set(ATTENTION_STAGES))`.
- Inactive: reject field raw, giữ resolved None.
- Nếu cả hai active và cùng stage, resolve order với default `focal_then_local`.
- Trường hợp khác: reject raw order, giữ resolved None.
- Không thay numerical validation/init defaults hiện tại.

**Step 4 — Tách effective placement report khỏi legacy identity.**

Thêm `effective_placement()` trả cấu trúc sau để notebook/report/diagram dùng:

```json
{
  "focal": {"stage": "c4", "stride": 8, "channels": 96},
  "local_attention": {"stage": "c3", "stride": 4, "channels": 48},
  "attention_order": null
}
```

Disabled entry là null. Metadata v2 dùng trực tiếp report này. Metadata v1 giữ
nguyên placement constant và config cũ; loại ba fields mới khỏi config **chỉ khi
tạo legacy semantic metadata**, không loại khỏi resolved config.

Legacy-equivalent predicate là:

```python
def has_legacy_placement(features):
    return (
        (features.c4_context == "none" or features.focal_stage == "c4")
        and (features.local_attention == "none"
             or features.local_attention_stage == "c3")
    )
```

Giữ `semantic_hash` từ canonical JSON metadata như hiện tại. Không bump outer
checkpoint version vì topology mới đã được thể hiện trong feature metadata.

**Step 5 — GREEN và compatibility gate.**

```bash
"$ATTENTION_PY" -m pytest tests/test_backbone_feature_config.py tests/test_attention_stage_compatibility.py -q
```

Expected: new validation tests qua và golden v1 không đổi. Mốc commit:
`feat(attention): resolve stage selection and canonical placement metadata`.

## Task 3: Constructor, registry và stage routing production

**Files:**
- Modify: `detector/core/models/backbones/mobilepixornext.py` — constructor và forward.
- Modify: `detector/core/models/backbones/registry.py` — thread resolved options.
- Create: `tests/test_attention_stage_routing.py`.
- Modify: `tests/test_backbone_attention_integration.py`.

**Step 1 — Test mới RED.**

Viết tests construction Focal C3/C4/C5 → channels 48/96/128; direct constructor
và registry cùng resolved options/cùng seed → cùng state dict. Test stage taps,
shared-stage order, call count và downstream tensor identity.

Run: `"$ATTENTION_PY" -m pytest tests/test_attention_stage_routing.py -q`.
Expected RED: constructor chưa nhận `focal_stage` hoặc runtime vẫn dùng C4.

**Step 2 — Thread fields qua constructor.**

- Thêm ba keyword arguments. Dùng sentinel `_UNSET` cho default omission để
  explicit None ở API mới vẫn bị resolver từ chối như null trong JSON.
- Chỉ thêm field vào raw options khi argument không phải sentinel.
- Lưu immutable resolved feature options và effective stages vào backbone.
- Giữ prefix `c4_context.*` và `c3_light_attention.*` ở mọi stage.
- Focal constructor lấy channels từ `ATTENTION_STAGES[focal_stage].channels`.
- Giữ thứ tự initialization: trunk/neck → detail → local → Focal. Không đăng ký
  module vào một container thứ hai; không tạo context/local instance cho mỗi stage.
- Sau initialization hiện tại, thêm ba Identity taps không có state:
  `attention_c3_output`, `attention_c4_output`, `attention_c5_output`.

**Step 3 — Helper routing tham chiếu.**

Code sau là toàn bộ logic composition cần thực hiện; tên resolved attrs phải được
gán ở constructor từ feature resolver:

```python
def _apply_stage_attention(self, feature, stage):
    focal_home = self.focal_stage or "c4"
    local_home = self.local_attention_stage or "c3"
    operations = []
    if stage == focal_home:
        operations.append(self.c4_context)
    if stage == local_home:
        operations.append(self.c3_light_attention)
    if self.attention_order == "local_then_focal":
        operations.reverse()
    for operation in operations:
        feature = operation(feature)
    return getattr(self, f"attention_{stage}_output")(feature)
```

Fallback homes chỉ dành cho disabled Identity để giữ hooks lịch sử. Metadata
disabled placement vẫn null. Shared-stage order chỉ active khi cả hai module bật.

Đổi đầu forward, giữ toàn bộ neck phía dưới:

```python
c1 = self.stem(x)
c2 = self._apply_stage_attention(self.stage2(self.down2(c1)), "c3")
c3 = self.stage3(self.down3(c2))
c4 = self._apply_stage_attention(self.c4_attention(c3), "c4")
c5 = self._apply_stage_attention(self.stage4(self.down4(c4)), "c5")
```

Giữ tên biến `c2/c3/c4` nếu nó giảm diff; comments phải ghi rõ feature level thật.
Không duplicate neck hoặc đổi RC-neck arguments. Mỗi attention/tap chạy một lần.

**Step 4 — Test composition bằng module không giao hoán.**

```python
class Affine(torch.nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.scale = scale
        self.offset = offset
        self.calls = 0

    def forward(self, feature):
        self.calls += 1
        return feature * self.scale + self.offset
```

Trong model đã resolve hai module cùng C3, thay Focal bằng `Affine(2,1)` và local
bằng `Affine(3,-4)`. Capture stage2 output và `attention_c3_output`. FS phải bằng
`3*(2*X+1)-4`; SF bằng `2*(3*X-4)+1`. Mỗi module có calls=1. Đây là test định
tuyến có ý nghĩa, không dùng hai Identity có thể làm test thứ tự qua giả.

Kiểm tra `attention_c3_output` là cùng object mà `down3` và `lat_c3` nhận;
tương tự C4/down4/lat_c4 và C5/lat_c5. RC neck nhận refined C3/C4/C5.

**Step 5 — GREEN và legacy routing.**

```bash
"$ATTENTION_PY" -m pytest tests/test_attention_stage_routing.py tests/test_backbone_attention_integration.py tests/test_mobilepixornext_backbone.py tests/test_attention_stage_compatibility.py -q
```

Expected: existing Identity-hook tests vẫn qua, keys không thêm tensors từ taps.
Mốc commit: `feat(backbone): route focal and local attention after selected stages`.

## Task 4: Coverage 19 tổ hợp, training gradients và full-grid shapes

**Files:**
- Modify: `tests/test_attention_stage_routing.py`.
- Create: `tests/test_attention_stage_training.py`.
- Read/reuse: `tests/test_context_grouped_compatibility.py`,
  `tests/test_grouped_training.py`, `tests/test_grouped_losses.py`.

**Step 1 — Tạo topology tuples trong bộ nhớ.**

```python
from itertools import product

PLACEMENTS = (None, "c3", "c4", "c5")
ATTENTION_CASES = [
    (focal, local, "focal_then_local" if focal and focal == local else None)
    for focal, local in product(PLACEMENTS, repeat=2)
] + [(stage, stage, "local_then_focal") for stage in PLACEMENTS[1:]]
assert len(ATTENTION_CASES) == 19


def placement_patch(focal, local, order):
    patch = {
        "c4_context": "focal" if focal else "none",
        "local_attention": "simam" if local else "none",
    }
    if focal:
        patch["focal_stage"] = focal
    if local:
        patch["local_attention_stage"] = local
    if order:
        patch["attention_order"] = order
    return patch
```

Khi apply vào full fixture config, xóa feature fields thuộc các module đã tắt
trước khi update để test không tạo invalid stale override. Helper này chỉ trong
tests; không viết files/presets hoặc sửa default notebook.

**Step 2 — Forward/backward FP32 cho cả 19 cases.**

- Dùng actual model và actual grouped OGA/IQA criterion từ fixtures hiện có.
- Input grid nhỏ [B,14,32,48], B=2; repeat targets đúng batch/spatial contract.
- Loss finite; backward; active gamma/beta/core parameters có finite gradient
  khi được nối tới loss. Không yêu cầu mọi gradient nonzero trên mọi random sample.
- Kiểm tra optimizer chứa mỗi trainable parameter đúng một lần.
- Baseline none/none phải không có gamma/beta của hai attention modules.
- Có ít nhất một representative smoke cho BEV/shared-3D headers để xác nhận
  placement không làm thay head contract. Không thay loss formulas.
- ECA: thêm construction/routing smoke ở C4 và C5, không mở bộ thí nghiệm ECA.

**Step 3 — Meta audit trong bộ nhớ.**

Build đủ 19 configs trên meta device với input [1,14,800,704]. Assert output
backbone [1,32,200,176] và head shapes. Đếm Focal params:
15,472 tại C3; 25,120 tại C4; 31,552 tại C5; SimAM adapter=1.
Kiểm tra state dict không có duplicate aliases của cùng parameter storage.

**Step 4 — AMP checks theo hardware.**

CPU FP32 luôn chạy. Test CUDA FP16/BF16 forward/backward cho ba colocated stages
và native C4/C3, dùng `skipif` khi CUDA không có; BF16 check thêm hardware support.
Không dùng CPU autocast kết quả để tuyên bố CUDA ổn.

**Step 5 — Verify.**

```bash
"$ATTENTION_PY" -m pytest tests/test_attention_stage_routing.py tests/test_attention_stage_training.py tests/test_context_grouped_compatibility.py tests/test_light_attention.py tests/test_focal_context.py -q
```

Expected: CPU cases qua; CUDA cases qua hoặc skip có lý do. Mốc commit:
`test(attention): cover placement training gradients and full-grid shapes`.

## Task 5: Checkpoint identity, run names và profile report

**Files:**
- Modify: `tools/kitti_training_pipeline/common.py` — identity/name/report consumers.
- Modify: `tools/benchmarks/profile_detector.py` — placement trong report.
- Create: `tests/test_attention_stage_checkpoint.py`.
- Modify: `tests/test_grouped_checkpoint.py`, `tests/test_attention_stage_compatibility.py`.
- Read: `tools/kitti_training_pipeline/train.py` — `restore_checkpoint` preflight.

**Step 1 — RED trên shape-equivalent semantic mismatches.**

- Cùng Focal C4, chuyển SimAM C3→C5: state shapes tương đương nhưng resume/eval
  phải từ chối. Destination model/criterion/optimizer không bị mutate.
- Focal/SimAM cùng C4: FS→SF phải bị từ chối dù mọi state keys/shapes giống nhau.
- Cùng topology với omitted/explicit order default được chấp nhận.
- Round-trip save/load cho native, Focal C3 + SimAM C5 và colocated SF; giữ
  gamma/beta, BN buffers và optimizer state khi full resume.
- 19 cases có 19 canonical identities và 19 tên run khác nhau.
- Explicit legacy stages có identity/tên run như golden trước thay code.

Run: `"$ATTENTION_PY" -m pytest tests/test_attention_stage_checkpoint.py -q`.
Expected RED nếu run names/report chưa chứa effective placement; không bỏ qua
test chỉ vì resolver v2 đã làm một phần identity test qua.

**Step 2 — Canonicalize model extras và format tên run.**

`checkpoint_identity` hiện pop dataclass fields rồi đặt `model.features`.
Đảm bảo field additions vẫn được pop, không để lại `focal_stage`/order ở outer
model cạnh metadata đã canonicalize. Golden legacy identity phải byte-compatible.

Nếu feature metadata version 1, giữ tag format cũ. Nếu version 2, tên run chứa
actual stage và order, ví dụ:
`ctx_focal_atc4_local_simam_atc4_orderfs_f32_detail0_<digest>`.
Omit order tag khi separate/disabled. Giữ semantic digest cuối như hiện tại.

Không đổi checkpoint loader sang `strict=False`, không sửa config nhúng để ép load.
Không tự warm-start/resume khi người dùng chọn stage khác. Helper warm-start
shape-matching đã có vẫn là tính năng explicit cũ; không thêm stage-mapping logic.

**Step 3 — Profile/report.**

Thêm `effective_attention_placement` từ resolver vào report ngoài
`feature_metadata`; legacy metadata v1 vẫn có constant cũ nên report cần actual
active/off entries. Counts tiếp tục đọc registered instances và model thật.
Không suy vị trí từ prefix. Không thêm options cấu hình mới để chỉ phục vụ report.

**Step 4 — Verify.**

```bash
"$ATTENTION_PY" -m pytest tests/test_attention_stage_checkpoint.py tests/test_grouped_checkpoint.py tests/test_attention_stage_compatibility.py -q
```

Expected: reject trước mutation, same-topology round trips qua, golden legacy qua.
Mốc commit: `feat(checkpoint): identify attention placement and order`.

## Task 6: Notebook controls và cleanup có nguồn gốc rõ ràng

**Files:**
- Modify: `tools/kitti_training_pipeline/notebook_config.py` — custom override merge.
- Modify: `3D_Lidar_Object_Detection_Notebook_standard.ipynb` — config cell chứa
  `config_dict = resolve_notebook_config(`; markdown hướng dẫn gần cell nếu cần.
- Modify: `tools/generate_robust_notebook.py` — control/config cell template.
- Regenerate: `3D_Lidar_Detection_Pipeline_Robust.ipynb`.
- Create: `tests/test_attention_stage_notebook.py`.
- Modify: `tests/test_under1m_notebook_controls.py`,
  `tests/test_under1m_notebook_config.py`, `tests/test_robust_notebook_setup.py`.

**Step 1 — RED: execute actual notebook config cell.**

Tái sử dụng `execute_cell` pattern trong `test_under1m_notebook_controls.py`:
test chạy cell với control edits, không chỉ search text trong JSON notebook.

- Parameterize 19 cases bằng `C4_CONTEXT`, `LOCAL_ATTENTION`, stages và order.
- Check resolved model, snapshot config đã lưu, identity và run name.
- Reexecute cùng namespace: shared FS→shared SF→separate→off→shared.
- Switch SimAM↔ECA, custom↔preset/config; không giữ stage/order/kernel/lambda sai.
- Config file có explicit invalid inactive fields vẫn bị từ chối.
- Defaults, old preset identities và file/runtime precedence giữ nguyên.

Run: `"$ATTENTION_PY" -m pytest tests/test_attention_stage_notebook.py -q`.
Expected RED: control chưa có hoặc chưa tới resolved config.

**Step 2 — Thêm controls, giữ defaults hiện có.**

```python
FOCAL_STAGE = "c4"                  # c3 | c4 | c5
LOCAL_ATTENTION_STAGE = "c3"        # c3 | c4 | c5
ATTENTION_ORDER = "focal_then_local"  # focal_then_local | local_then_focal
```

Tạo patch placement trong custom mode, chỉ emit active fields:

```python
if C4_CONTEXT == "focal":
    cm["focal_stage"] = FOCAL_STAGE
if LOCAL_ATTENTION != "none":
    cm["local_attention_stage"] = LOCAL_ATTENTION_STAGE
if (C4_CONTEXT == "focal" and LOCAL_ATTENTION != "none"
        and FOCAL_STAGE.lower() == LOCAL_ATTENTION_STAGE.lower()):
    cm["attention_order"] = ATTENTION_ORDER
```

Notebook controls phải validate type trước `.lower()` hoặc dùng shared parser để
tránh lỗi AttributeError cho list/bool. Dòng patch phía trên là sau validation
controls. External raw config luôn giữ resolver strict.

**Step 3 — Cleanup inherited fields trong custom override merge.**

`resolve_notebook_config` đọc base fresh mỗi lần nhưng deep merge có thể giữ
field cũ. Thêm helper nhận cả **base model và raw custom model patch** để phân
biệt inherited fields với fields người dùng explicit đưa vào:

1. Merge bản copy, không mutate base/patch.
2. Resolve selector/effective stage để xác định field nào còn áp dụng.
3. Loại **inherited** focal fields/stage khi Focal tắt; local-specific fields/stage
   khi local tắt hoặc đổi loại; inherited order khi hai module không còn cùng stage.
4. Không xóa các fields lỗi được explicit đưa trong custom patch. Để resolver
   từ chối chúng và báo đúng tên field.
5. Không áp dụng sanitation này lên `file_override`, runtime config hoặc checkpoint
   embedded config; không âm thầm sửa external config sai.

Ví dụ base có colocated FS; patch chỉ chuyển local stage thành C3 → xóa order
inherited. Nếu patch cũng explicit order FS dù stage khác nhau → lỗi.
Ví dụ base SimAM có lambda; patch đổi sang ECA → xóa lambda inherited; nếu patch
explicit đưa lambda vào ECA → lỗi.

Không thêm preset hoặc materialize experiment JSON. `save_notebook_config`
tiếp tục lưu current run snapshot, gồm resolved stage/order active.

**Step 4 — Summary trên notebook.**

In bảng effective placement từ shared resolver: kind, C-level, Stage 2/3/4,
stride, channels, order và semantic hash. Với stage khác nhau in thứ tự thực tế
theo backbone depth; không nói global FS/SF nếu không co-located. Disabled hiện off.

**Step 5 — Cập nhật hai notebook đúng cách.**

- Standard: dùng Python JSON editor tìm cell bằng marker unique; sửa control,
  patch emission và summary của đúng cell; giữ các cell khác/metadata.
- Robust: sửa generator, sinh ra directory tạm rồi đối chiếu trước khi cập nhật
  Robust tracked file. Không overwrite Standard từ generator.
- Không strip toàn bộ outputs, reorder cells hoặc đổi setup/train/eval cells
  ngoài nội dung cần thiết của stage controls.

Lệnh regenerate cô lập từ repository root:

```bash
mkdir -p /tmp/lidar-stage-notebook-generation
cd /tmp/lidar-stage-notebook-generation
"$ATTENTION_PY" /home/duyennh/AI_projects/research_lidar/Lidar/Code/tools/generate_robust_notebook.py
```

Sau đó về `Code/` để chạy tests; PYTHONPATH tương đối trong setup phải được set
lại khi đổi cwd. So sánh generated Robust với tracked Robust, cập nhật có kiểm soát.

**Step 6 — Verify.**

```bash
"$ATTENTION_PY" -m pytest tests/test_attention_stage_notebook.py tests/test_under1m_notebook_controls.py tests/test_under1m_notebook_config.py tests/test_attention_presets.py tests/test_robust_notebook_setup.py tests/test_simple_notebook_workflow.py -q
```

Expected: 19 control cases tạo model configs hợp lệ; preset/config modes không
bị custom controls ghi đè. Mốc commit:
`feat(notebook): expose attention stages and shared-stage order`.

## Task 7: Capture và diagram consumers

**Files:**
- Modify: `tools/visualization/export_kitti_pipeline.py`.
- Modify: `tools/visualization/compare_attention_placements.py`.
- Modify: `tools/visualization/export_backbone_diagram.py`.
- Modify if needed: `tools/visualization/draw_mobilepixornext_paper.py`
  (`inspect_config`), `tools/visualization/draw_bev_backbone_heads.py`.
- Mirror necessary capture changes: repository-root `Demo/code/export_kitti_pipeline.py`.
- Modify: `tests/test_kitti_pipeline_visualization.py`,
  `tests/test_attention_placement_visualization.py`, `tests/test_backbone_diagram.py`.

**Step 1 — Tests cho capture đúng sau attention.**

- Native placement replay unchanged; both shared orders capture stage sau module
  cuối cùng, không chỉ raw `stage2/stage3/stage4` output.
- Capture taps không đổi model state; hooks được tháo cả khi forward lỗi.
- Các báo cáo không gọi moved `c4_context` là C4 chỉ vì prefix.
- Intervention tool hiện có phải tiếp tục được ghi là intervention.

**Step 2 — Capture real trained checkpoint.**

Pipeline chọn ba `backbone.attention_c*_output` khi model có taps; legacy
fallback vẫn giữ cho model khác. Ghi placement/order trong metadata và tensor keys.
Không coi hook trên module `stage2` là output sau Focal/SimAM.

Trong compare tool, thêm mode trained checkpoint chạy model đúng config đã train
và capture taps; không tự tạo Focal mới trong mode này. Mode intervention hiện tại
chỉ nhận checkpoint native C4/C3 phù hợp assumptions, hoặc phải có guard rõ ràng
trước khi sử dụng placement/sizing cố định. Nhãn untrained/intervention không được
gỡ chỉ vì tool chạy qua sau khi production đã hỗ trợ chọn stage.

**Step 3 — Generic NN-SVG graph theo effective placement.**

- `_context_view` dùng channels/stride thực, không hard-code 96/stride8.
- `_local_view` dùng channels/stride thực, không hard-code 48/stride4.
- Trong loop stages, lấy danh sách module theo actual stage và shared order.
- Lưu `refined_stage_outputs` để cả next downsample và lateral dùng node cuối
  sau attention. C5 attention phải xuất hiện trước lateral C5.
- `verify_model` capture các output tap tương ứng và đối chiếu graph shapes.

Renderer curated cho hình lịch sử có thể giữ layout cũ nhưng phải reject
nonlegacy placement bằng lỗi rõ ràng, hướng người dùng sang generic exporter.
Không redraw/publish toàn bộ paper figures cũ trong task này.

**Step 4 — Verify.**

```bash
"$ATTENTION_PY" -m pytest tests/test_kitti_pipeline_visualization.py tests/test_attention_placement_visualization.py tests/test_backbone_diagram.py -q
```

Expected: captures finite/correct stage/order, hooks cleanup, graph đúng mọi stage.
Mốc commit: `fix(visualization): report actual attention placement`.

## Task 8: Documentation và final verification

**Files:**
- Modify: `docs/mobilepixornext_architecture.md`.
- Modify: `docs/plans/lightweight_lidar_backbone_2026/attention_specification.md`.
- Modify: `tools/kitti_training_pipeline/README.md`.
- Update status only after implementation: source spec và plan này.

**Step 1 — Hướng dẫn đổi controls trên notebook.**

Giải thích C3=Stage2/stride4, C4=Stage3/stride8, C5=Stage4/stride16. Ví dụ:

```python
# Khác stage: Focal C4 + SimAM C3.
C4_CONTEXT = "focal"
FOCAL_STAGE = "c4"
LOCAL_ATTENTION = "simam"
LOCAL_ATTENTION_STAGE = "c3"
ATTENTION_ORDER = "focal_then_local"  # không emit khi stage khác nhau
```

Ví dụ cùng C5 chỉ đổi hai stage thành `c5`; đảo thứ tự bằng
`ATTENTION_ORDER="local_then_focal"`. Tắt bằng selectors none, không dùng stage off.
Nhắc rerun config cell, xem resolved summary và dùng run mới khi topology thay đổi.
Không tạo example training config files; code snippets trong docs là đủ.

**Step 2 — Focused suites rồi full suite.**

```bash
"$ATTENTION_PY" -m pytest tests/test_backbone_feature_config.py tests/test_attention_stage_compatibility.py tests/test_attention_stage_routing.py tests/test_attention_stage_training.py tests/test_attention_stage_checkpoint.py tests/test_attention_stage_notebook.py tests/test_backbone_attention_integration.py tests/test_grouped_checkpoint.py tests/test_under1m_notebook_controls.py tests/test_under1m_notebook_config.py tests/test_attention_presets.py tests/test_robust_notebook_setup.py tests/test_kitti_pipeline_visualization.py tests/test_attention_placement_visualization.py tests/test_backbone_diagram.py -q
"$ATTENTION_PY" -m pytest tests/ -q
git diff --check
```

Focused suite qua rồi mới chạy full suite. Nếu full suite có lỗi environment hoặc
hardware/optional tool, phân biệt rõ với regression và xử lý lỗi thuộc change.

**Step 3 — P16 actual-checkpoint replay, không training.**

Checkpoint nguồn đã dùng trong phiên này:

```text
/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-oga_loss-hist14-iqa-sgfpn-ctx_focal_local_simam_f32_detail0_8f04a8fa523a-grouped_e02c0e499671-s42-P16-h32/selected/best_ap.pt
```

Tạo test/smoke opt-in đọc embedded config, gọi `validate_evaluation_checkpoint`,
build model, load strict, encode đúng cloud `000134.bin` và capture taps/final/head.
So sánh với frozen reference NPZ task 1: features, logits và probabilities,
tolerance FP32 `rtol=1e-5, atol=1e-6` trong cùng environment. Hash nguồn phải khớp
reference metadata; model eval state không bị mutate. CPU comparison không đo AP.

Không coi missing local external checkpoint là blocker của unit tests portable;
trong workspace hiện có nguồn nên thực hiện smoke trước khi giao implementation.

**Step 4 — Kiểm tra phạm vi diff.**

- Không có changes dưới `configs/experiments/`, không đổi `configs/config.json`.
- Không có script materialize attention-placement configs hoặc preset mới.
- Không đổi formula trong `focal_context.py`/`light_attention.py` nếu không có bug
  phát hiện độc lập và được nêu rõ.
- Standard chỉ cell config/hướng dẫn cần thiết; Robust khớp generator.
- Existing P16 artifacts/Downloads checkpoint/data vẫn nguyên.
- Không có tự động training, deployment hoặc publication.

**Step 5 — Completion evidence.**

Ghi số tests pass/skip, 19-topology coverage, actual replay result và bất kỳ CUDA
limitation nào. Spec chuyển trạng thái implemented chỉ khi acceptance criteria
đã được kiểm chứng, không chỉ vì file code đã sửa.
Mốc commit: `docs(attention): document notebook stage controls and verification`.

### Risks & mitigations

- **Legacy hash thay đổi vì to_dict thêm fields:** golden fixture + riêng hóa
  semantic v1 config serialization.
- **Same-shape checkpoint load sai:** identity preflight trước mutation; negative
  tests relocation SimAM và shared-order swap.
- **Identity taps làm double registration:** module instances chỉ một prefix;
  taps state-free; count optimizer/state keys kiểm chứng.
- **Notebook cleanup che lỗi explicit:** chỉ remove inherited invalid fields;
  raw explicit invalid config luôn bị resolver từ chối.
- **Standard bị overwrite bằng Robust:** sửa cell unique; generator output chạy
  ở `/tmp`, kiểm tra diff hai notebook riêng.
- **Feature captures trước module cuối:** capture dedicated taps, test composition.
- **Chi phí C3 Focal:** actual counts/meta shapes trong report; không tự đổi D
  hay hứa latency tương đương khi chưa đo.
- **False optimality claim:** implementation tests xác nhận correctness; người
  dùng train lại để đo AP theo stage. Heatmap/intervention không thay trained ablation.

### Rollback plan

Revert riêng commits/hunks stage-selection mới sau khi giữ user edits. Không dùng
`git reset --hard`, không xóa dữ liệu/checkpoint/output hoặc work từ các lượt trước.
Config cũ/state prefixes/identity v1 được giữ nên old checkpoints vẫn chạy với code
cũ. Checkpoint topology v2 cần code hỗ trợ stage selection để replay; giữ artifacts
riêng. Vì không tạo presets/config files mới, không có config matrix để rollback.

## Checklist giao implementation

- [x] C3/C4/C5 và hai shared-stage orders tới được production forward.
- [x] Refined tensor dùng chung cho downstream/lateral đúng mức.
- [x] 19 trường hợp trong tests; gradient/optimizer/shape checks qua.
- [x] Config/hash/state/output legacy replay qua.
- [x] Relocation/order khác bị resume/eval mismatch từ chối trước mutation.
- [x] Notebook Standard và Robust có controls, summary, active-only snapshot.
- [x] Chuyển mode/module/stage không giữ stale fields hoặc che lỗi explicit.
- [x] Reports/captures/diagrams đọc actual placement.
- [x] Không tạo experiment configs/presets, không tự chạy training.
- [x] Focused/full suite, hardware skips, actual P16 replay được ghi lại.
