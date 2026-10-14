# Spec: chọn stage cho Focal Context và SimAM trong MobilePixorNeXt

Ngày: 2026-10-14.

Trạng thái: **đã triển khai và kiểm chứng runtime, 2026-10-14**.
Chi tiết bằng chứng, hướng dẫn notebook và giới hạn môi trường:
[verification report](../../plans/2026-10-14-stage-selectable-attention-verification.md).
Các phần Known context bên dưới mô tả baseline trước khi triển khai.

Cập nhật phạm vi theo yêu cầu tiếp theo: **không tạo config thí nghiệm, preset
hay script sinh matrix**. Người dùng tự đổi controls trên notebook. Bộ 19 tổ hợp
chỉ là coverage cho tests tạo config trong bộ nhớ. Plan chi tiết hiện hành:
[implementation plan](../../plans/2026-10-14-stage-selectable-attention-implementation-plan.md).

### Goal

Cho phép train mới MobilePixorNeXt với Focal Context và SimAM ở C3, C4 hoặc C5,
độc lập nhau. Khi cùng stage, hỗ trợ cả Focal → SimAM và SimAM → Focal.
Mọi cấu hình phải đi qua model production và được ghi nhận đúng trong checkpoint.

### Constraints

- Một Focal và một local-attention adapter tối đa trong một model; mỗi module
  chọn một stage. Chưa hỗ trợ cùng một module xuất hiện đồng thời ở nhiều stage.
- Stage hỗ trợ: C3/C4/C5, tương ứng Stage 2/3/4 và stride 4/8/16.
- Không thay công thức Focal, SimAM, neck, head, loss, encoding hoặc augmentation.
- Mặc định của cấu hình cũ: Focal C4, local attention C3.
- Config cũ, state-dict keys, initialization và semantic identity của kiến trúc
  cũ phải được bảo toàn. Không cần chuyển đổi checkpoint đã có.
- Cấu hình mới khác topology phải có identity khác, kể cả chỉ đổi thứ tự khi
  cùng stage và state-dict shapes không đổi.
- Triển khai production không dùng forward hook để chèn attention.
- Không tự chạy 19 lượt training. Người dùng sẽ train lại sau khi implementation
  được hoàn thành và kiểm chứng.

### Known context

- `detector/core/models/backbones/mobilepixornext.py` hiện đặt
  `c3_light_attention` sau `stage2`; `c4_context` sau `stage3`.
- `detector/core/backbone_config.py` là resolver thuần, dùng chung cho constructor,
  registry, validation, semantic metadata và hash.
- Metadata resolver version 1 hiện cố định
  `placement={context_stride:8, local_attention_stride:4}`.
- `tools/kitti_training_pipeline/common.py` dùng metadata này trong checkpoint
  identity, resume/evaluation validation và tên run.
- `tools/generate_robust_notebook.py` sinh notebook Robust (22 cells). Notebook
  standard hiện có 18 cells, là file riêng; chỉ sửa cell cấu hình của nó, không
  ghi đè bằng output của generator Robust.
- `tools/visualization/compare_attention_placements.py` vừa tạo 19 visualization
  interventions. Focal C3/C5 của bộ hình đó mới khởi tạo; chúng không phải kết quả
  19 model đã được train. Công cụ này không thay thế runtime production.

### Risks

- C3/C4/C5 trong tài liệu không trùng hoàn toàn tên biến `c2/c3/c4` ở forward cũ.
- Chỉ đổi forward mà quên identity có thể làm resume sai kiến trúc mà không lỗi.
- Đổi prefix module hoặc thứ tự khởi tạo có thể làm hỏng checkpoint/fixture cũ.
- Notebook merge có thể giữ stage/order của preset trước sau khi tắt module.
- Focal C3 tốn compute/bộ nhớ hơn; đổi stage cũng thay số tham số Focal.
- Attention chỉ áp dụng ở lateral hoặc chỉ ở downsample sẽ làm phép ablation
  không đúng kiến trúc đã mô tả.

### Options

1. **Thêm stage/order vào schema hiện tại — khuyến nghị.**
   Giữ `c4_context` và `local_attention` để chọn loại module, thêm ba field vị trí.
   Ít migration; giữ checkpoint cũ. Tên `c4_context` là tên lịch sử, cần giải thích.
   Độ phức tạp vừa; phần khó chính là canonical metadata và notebook cleanup.
2. **Đổi sang schema `stage_attention` dạng danh sách cho từng stage.**
   Mở rộng được nhiều module/multi-stage, nhưng cần alias, migration, thứ tự và
   nhiều quy tắc conflict; vượt nhu cầu 19 biến thể hiện tại. Độ phức tạp cao.
3. **Tạo backbone riêng cho từng tổ hợp.**
   Dễ làm một prototype, nhưng lặp code và dễ lệch neck/head/identity giữa run.
   Không phù hợp để train và duy trì 19 cấu hình.

### Recommendation

Chọn option 1: thêm `focal_stage`, `local_attention_stage`, `attention_order`.
Giữ topology mặc định SimAM C3 + Focal C4 khi hai module đang bật. Bật/tắt module
tiếp tục dùng các field cũ. Mọi stage/order mới được resolve trước khi build model.

### Acceptance criteria

- [x] Có thể build, forward và backward đủ 19 tổ hợp Focal/SimAM bên dưới.
- [x] Attention giữ shape tại stage và đầu ra backbone vẫn stride 4.
- [x] Feature sau attention đi vào cả stage tiếp theo và lateral tương ứng;
  riêng C5 đi vào lateral C5/RC-neck.
- [x] Hai thứ tự cùng stage được thực thi đúng, chỉ gọi mỗi module một lần.
- [x] Config cũ và explicit legacy-equivalent config có cùng semantic identity;
  prefix, weights và predictions cũ vẫn tương thích.
- [x] Stage khác hoặc order khác có identity và run name khác; resume/evaluation
  từ checkpoint không tương thích bị từ chối.
- [x] Notebook tạo được các lựa chọn này; inactive fields không bị giữ lại khi
  chuyển loại module, tắt module hoặc đổi stage.
- [x] Tests tạo đủ 19 config trong bộ nhớ từ cùng base để kiểm chứng routing,
  identity và gradient; không thêm experiment JSON, manifest hay preset.
- [x] Report và visualization ghi vị trí thực tế; không gắn nhãn mọi
  `c4_context` là C4 nếu module đã chuyển stage.
- [x] Không còn blocker/major trong kiểm tra resolver, routing, identity và
  tương thích cũ. GPU/AMP checks được ghi rõ nếu môi trường chỉ có CPU.

## 1. Stage contract và phạm vi

| Config stage | Backbone stage | Feature level | Channels | Stride | KITTI H×W |
|---|---|---|---:|---:|---|
| `c3` | `stage2` | C3 | 48 | 4 | 200×176 |
| `c4` | `stage3` | C4 | 96 | 8 | 100×88 |
| `c5` | `stage4` | C5 | 128 | 16 | 50×44 |

Stage luôn chỉ output sau toàn bộ block của stage, không phải sau một block con
hay sau projection của neck. Không mở hook tại stem, từng block, FPN hoặc logits
trong release này. Không thay output stride hay số kênh của head.

Schema local attention hiện có `none|eca|simam` được giữ. `local_attention_stage`
là field chung cho adapter này, nên ECA cũng resolve và route theo cùng contract;
không tạo ECA presets mới trong bộ 19 thí nghiệm Focal/SimAM.

## 2. Public config

Các field mới đặt trong `model`:

| Field | Giá trị | Default khi active | Quy tắc |
|---|---|---|---|
| `focal_stage` | `c3|c4|c5` | `c4` | Chỉ có khi `c4_context=focal` |
| `local_attention_stage` | `c3|c4|c5` | `c3` | Chỉ có khi `local_attention=eca|simam` |
| `attention_order` | `focal_then_local|local_then_focal` | `focal_then_local` | Chỉ có khi cả hai active và cùng stage |

Giữ nguyên `c4_context=none|focal` và các field `c4_context_*` về bottleneck,
dilation, version, residual init. Tên lịch sử này không còn quyết định placement;
`focal_stage` quyết định placement. Không thêm alias `focal`, `simam_stage`, số
stage, hoặc danh sách stage trong cùng release.

Ví dụ placement mặc định, tương đương model P16 về attention:

```json
{
  "model": {
    "backbone": "mobilepixornext",
    "c4_context": "focal",
    "focal_stage": "c4",
    "local_attention": "simam",
    "local_attention_stage": "c3"
  }
}
```

Ví dụ cùng C4, Focal rồi SimAM:

```json
{
  "model": {
    "backbone": "mobilepixornext",
    "c4_context": "focal",
    "focal_stage": "c4",
    "local_attention": "simam",
    "local_attention_stage": "c4",
    "attention_order": "focal_then_local"
  }
}
```

Đổi `attention_order` thành `local_then_focal` để đảo thứ tự. Nếu không khai báo
order và hai module cùng stage, resolver materialize `focal_then_local`.

Ví dụ tắt cả hai:

```json
{"model": {"backbone": "mobilepixornext", "c4_context": "none", "local_attention": "none"}}
```

Các ví dụ là model-section patches, không phải config dataset/train hoàn chỉnh.

### 2.1. Validation và canonicalization

- Normalize stage/order strings về lowercase như `_choice` hiện tại.
- Reject booleans, integers, lists, null, stage ngoài danh sách và field typo.
- Explicit `focal_stage` khi Focal tắt là lỗi, kể cả giá trị `c4` hoặc null.
- Explicit `local_attention_stage` khi local attention tắt là lỗi.
- Explicit `attention_order` khi thiếu một module hoặc hai stage khác nhau là
  lỗi, kể cả giá trị default. Không hash một option không được thực thi.
- Không dùng `focal_stage=off` để tắt Focal; dùng `c4_context=none`.
- Các field mới trên backbone khác MobilePixorNeXt phải bị từ chối.
- Constructor trực tiếp, registry, notebook validation và checkpoint identity
  phải dùng chung resolver; không tự viết bảng stage/default riêng.
- Resolver không mutate config của caller. `to_dict()` chỉ ghi các field active;
  stage được materialize cho module active, order chỉ khi co-located.
- Quy tắc grid chia hết cho 16, module-specific inactive overrides, neck/detail
  incompatibility và validation scalar hiện tại được giữ nguyên.

## 3. Runtime routing

Giữ một instance `self.c4_context` và một instance
`self.c3_light_attention` để bảo toàn state prefixes. Đây là **tên đăng ký lịch
sử**, không phải vị trí thực thi. Không đăng ký cùng instance thêm lần nữa bằng
ModuleDict/Sequential/attribute alias. Không tạo module mới trong `forward`.

Khởi tạo Focal bằng channels của stage đã resolve: 48/96/128. Dilation, D và
LayerScale lấy từ config, không tự thay để bù chi phí hoặc làm visualization rõ.
SimAM/ECA giữ factory và wrapper residual hiện có.

Thêm helper production `_apply_stage_attention(feature, stage)`; stage table
được dùng chung từ backbone config. Luồng khái niệm:

```text
stem = stem(input)
c3 = apply_stage_attention(stage2(down2(stem)), "c3")
c4 = apply_stage_attention(c4_attention(stage3(down3(c3))), "c4")
c5 = apply_stage_attention(stage4(down4(c4)), "c5")
output = existing_neck(c3, c4, c5, optional_stem_detail=stem)
```

`c4_attention` hiện là Identity của option đã retired; không khôi phục LiteMLA.
Không copy neck vào helper và không thay công thức scale gate.

Với stage chứa hai module:

```text
focal_then_local: Y = local_attention(focal_context(X))
local_then_focal: Y = focal_context(local_attention(X))
```

Module thứ hai phải nhận tensor **sau module thứ nhất**, không nhận lại X. Không
chạy hai nhánh song song rồi cộng kết quả. Không dùng chung gamma/beta.

C3 sau xử lý cấp cho cả `down3` và `lat_c3`; C4 sau xử lý cấp cho cả `down4` và
`lat_c4`; C5 sau xử lý cấp cho `lat_c5`. RC neck nhận đúng ba refined tensors.
Helper không resize hay project bổ sung ngoài chính Focal.

### 3.1. Disabled modules và compatibility

Disabled adapter là `nn.Identity`, không có persistent state. Với cấu hình legacy,
giữ thực thi Identity local tại C3 và Identity context tại C4 để không phá các
capture hooks/tests hiện có; inactive Identity không được ghi thành active module
trong metadata. Mỗi instance Identity này cũng chỉ được gọi một lần.

Giữ thứ tự tạo module hiện tại: trunk/neck, optional detail, local adapter, Focal.
Legacy-equivalent config cùng seed phải giữ initialization stream hiện tại.
Thay stage Focal làm đổi tensor shape và lượng RNG tiêu thụ; không tuyên bố rằng
cùng seed đảm bảo mọi shared head weight giống nhau giữa các kiến trúc khác shape.

## 4. Identity, checkpoint và run name

### 4.1. Preserve version 1 cho topology tương đương legacy

Một config là legacy-equivalent nếu mọi module active ở vị trí lịch sử:
Focal ở C4 và local attention ở C3. Module tắt không có effective placement.

- Giữ nguyên hoàn toàn `semantic_metadata()` version 1 và hash cũ cho topology
  này, dù raw config có khai báo explicit `focal_stage=c4`/`local_attention_stage=c3`.
- Khi tạo phần `config` trong semantic metadata version 1, lọc ba field mới để
  bytes canonical không thay. Điều này khác với `to_dict()` dùng để lưu resolved
  config, vốn materialize stage active.
- Giữ outer checkpoint identity version và tên run cũ cho legacy-equivalent
  topology. Không tự nâng version của mọi checkpoint.
- Checkpoint cũ build và load strict như trước; không cần mapping tên tensor.
- Raw config digest có thể thay khi viết explicit defaults, nhưng semantic hash
  phải không thay; không sửa metadata/hash được lưu trong checkpoint cũ.

### 4.2. Metadata version 2 cho topology mới

Nếu bất kỳ module active chuyển khỏi vị trí lịch sử, resolver metadata version 2
ghi placement thực tế, ví dụ:

```json
{
  "resolver_version": 2,
  "placement": {
    "focal": {"stage": "c4", "stride": 8, "channels": 96},
    "local_attention": {"stage": "c4", "stride": 8, "channels": 96},
    "attention_order": "focal_then_local"
  },
  "config": {"...": "canonical active feature fields"}
}
```

Trong metadata v2, module tắt có placement null; order null nếu không co-located.
`config` là resolved feature config đầy đủ, không dùng literal `...` ở runtime.
Stage/order, module settings và các feature options khác tham gia semantic hash.

- Explicit cùng-stage default order và omitted order phải cùng identity.
- Đảo order phải đổi identity dù số tham số và shapes không đổi.
- Đổi stage của SimAM cũng phải đổi identity dù core parameter-free và beta chỉ
  một scalar; strict `load_state_dict` đơn thuần không đủ phát hiện đổi stage.
- `validate_evaluation_checkpoint` và resume validation so sánh identity trước
  khi load. Không bypass bằng `strict=False` hay sửa config nhúng.
- Full resume chỉ cho đúng topology/objective/training contract hiện có. Warm
  start/mapping weights sang topology khác không nằm trong release này.

Tên run mới cần có tag dễ đọc, thí dụ
`ctx_focal_atc4_local_simam_atc4_orderfs_f32_detail0_<digest>`.
Tag `orderfs/ordersf` chỉ có khi co-located. Hash vẫn là nguồn identity chính;
thay stage/order phải cho tên khác, và tên phải tái lập được.

## 5. Notebook và coverage các tổ hợp

Thêm controls vào generator và notebook chuẩn được sinh:

```python
FOCAL_STAGE = "c4"                 # c3 | c4 | c5
LOCAL_ATTENTION_STAGE = "c3"       # c3 | c4 | c5
ATTENTION_ORDER = "focal_then_local"  # khi hai module cùng stage
```

Giữ `C4_CONTEXT`, `LOCAL_ATTENTION` và các controls module hiện tại. Không tự bật
SimAM trong những preset trước đây để local attention none.

Trong custom mode, merge model patch rồi loại field của module đã tắt/thay loại,
và loại `attention_order` nếu không co-located. Emit `focal_stage` chỉ khi Focal
active; emit local stage chỉ khi local attention active. Không giữ stage/order
từ preset cũ sau khi người dùng chuyển option. Final resolved summary phải in:
module, stage, stride, channels, order và semantic hash.

Preset/config mode tiếp tục tôn trọng preset hoặc file đã chọn; custom controls
không vô tình ghi đè các chế độ này. Không tạo preset mới hoặc sửa định nghĩa
preset lịch sử. Việc lưu `config.resolved.json` của mỗi run vẫn hoạt động như
hiện tại để checkpoint có config authoritative.

Bộ test coverage gồm chính xác 19 biến thể; người dùng có thể chọn từng biến thể
bằng controls trên notebook:

| Nhóm | Focal | SimAM | Order | Số run |
|---|---|---|---|---:|
| Baseline | off | off | — | 1 |
| SimAM-only | off | từng C3/C4/C5 | — | 3 |
| Focal-only | từng C3/C4/C5 | off | — | 3 |
| Hai stage khác nhau | từng C3/C4/C5 | từng C3/C4/C5, khác Focal | — | 6 |
| Cùng stage, Focal trước | cùng từng C3/C4/C5 | cùng Focal | FS | 3 |
| Cùng stage, SimAM trước | cùng từng C3/C4/C5 | cùng Focal | SF | 3 |

Test IDs dùng `f_off_s_off`, `f_off_s_c3`, `f_c4_s_off`, `f_c4_s_c3`,
`f_c4_s_c4_fs`, `f_c4_s_c4_sf` theo quy tắc trên; không materialize thành file config.
Các fixture train dùng cùng một base trong bộ nhớ, ví dụ P16 recipe: Hist14,
[3,4,2], SG-FPN fusion32/output32, grouped independent BEV heads và OGA + IQA.

Khi người dùng train để so sánh vị trí, nên giữ loss, encoding, split, augmentation, batch size,
epochs, precision, seed và checkpoint-selection policy. Chỉ khác attention
kind/settings cần thiết cho bật/tắt, stage/order và experiment name. Vị trí Focal
làm đổi channels, parameter count và compute tự nhiên; không che sự khác biệt
bằng tự đổi D/dilation. Khi so sánh AP, đo theo từng lớp và đo latency/memory;
chạy thêm seed cho ứng viên tốt nhất trước khi kết luận tối ưu.

## 6. Report, visualization và parameter budget

- Lưu nguyên tensor/checkpoint của model đã train theo stage thực; không dùng
  newly initialized moved Focal để đại diện cho model đã train ở vị trí đó.
- Profile report thêm danh sách placement và order bên cạnh counts/shapes.
- Không suy ra vị trí từ tên `c4_context` hoặc `c3_light_attention`.
- Visualization hiện tại có mode intervention phải giữ nhãn intervention; bổ
  sung mode đọc checkpoint/config thực và capture stage **sau cả hai module**.
- Hook của `stage2/stage3/stage4` đơn thuần sẽ capture trước attention sau khi
  routing vào helper; cần tap Identity không-state hoặc API capture có tên rõ
  như `attention_c3_output`, `attention_c4_output`, `attention_c5_output`.
- Generic NN-SVG diagram phải đọc stage/order và channel thực. Renderer curated
  cho sơ đồ lịch sử có thể từ chối topology mới với lỗi rõ ràng và chỉ dẫn dùng
  generic renderer, thay vì xuất hình gắn sai vị trí module.
- Nếu thêm tap module, không tạo parameters/buffers, không đổi RNG hay state
  dict; capture code và schematic phải đọc cùng placement metadata.

Với Focal D=64, L=3, convolution không bias và BN affine như hiện tại:

```text
Parameters = C*(3*D + L + 6) + D*D + 9*D*L
C3 (C=48):  15,472
C4 (C=96):  25,120
C5 (C=128): 31,552
```

SimAM core vẫn không có tham số; wrapper thêm một beta. ECA core k=3 thêm ba
weights, wrapper thêm một beta. H×W khác nhau nên không suy latency chỉ từ counts.
Report tổng detector phải đếm từ model thực; không gán mọi biến thể nhãn under1m
trước khi kiểm tra actual count.

## 7. Verification contract

1. **Resolver:** defaults cũ, active-only serialization, mọi giá trị stage/order
   hợp lệ, typo/type/null rejection, inactive rejection, other-backbone rejection,
   normalization và input không bị mutate.
2. **Routing:** parameterize 19 Focal/SimAM topologies; shapes, số lần gọi và
   tensor identity tới downsample/lateral. Cùng stage kiểm tra chính xác composition
   cả hai order bằng module không giao hoán; không chỉ kiểm tra shape.
3. **Training:** batch ≥2 và grid nhỏ chia hết cho 16; forward/backward FP32,
   gradient finite tới active modules/trunk/heads. Không đòi mọi parameter có
   gradient nonzero trên mọi random input hoặc mọi beta/gamma identity setting.
4. **Compatibility:** frozen old config/hash/state fixture; omitted/default
   stages giữ state keys, initialization và eval outputs. Checkpoint P16 thực
   hiện có load strict và replay trong cùng environment với tolerance đã ghi.
5. **Identity:** 19 canonical topology identities khác nhau; explicit default
   order không tạo topology thứ 20; reorder/relocate có cùng shape vẫn bị resume
   validation từ chối. Same-topology checkpoint evaluation được chấp nhận.
6. **Notebook/tổ hợp:** switching none↔simam↔eca, stage riêng↔chung, custom↔preset
   không giữ stale fields; generated notebook chạy được config-building cells;
   tests có 19 tổ hợp duy nhất tạo trong bộ nhớ; không thêm config files/presets.
7. **Shape/budget:** meta-device full KITTI input [1,14,800,704], output
   [1,32,200,176], counts Focal đúng cả ba stage và không có duplicate registration.
8. **Runtime hardware:** CPU real-scan smoke cho representative placements và
   save/reload; CUDA FP16/BF16 forward/backward khi có GPU. Ghi skip rõ nếu không
   có CUDA. Không tuyên bố GPU latency/AP từ CPU hoặc meta-device tests.

## 8. Implementation plan

### Goal

Đưa contract stage-selectable vào đường train/eval production, bảo toàn checkpoint
cũ, rồi thêm controls notebook để người dùng tự chọn vị trí và train lại.

### Assumptions

- V1 cần một placement cho mỗi module, ba mức C3/C4/C5 và cả hai shared-stage order.
- Model/core module formulas không thay. D giữ 64 trong bộ ablation mặc định.
- Dùng môi trường PyTorch sẵn có; chưa train AP hoặc deploy ONNX/TensorRT.

### Plan

1. Resolver và backward-compatible metadata
   - Files: `detector/core/backbone_config.py`,
     `tests/test_backbone_feature_config.py`.
   - Change: stage table, three fields, validation, active serialization,
     legacy-v1/new-topology-v2 metadata.
   - Verify: `python -m pytest tests/test_backbone_feature_config.py -q`.
2. Constructor/registry và routing production
   - Files: `detector/core/models/backbones/mobilepixornext.py`,
     `detector/core/models/backbones/registry.py`,
     `tests/test_backbone_attention_integration.py`,
     `tests/test_attention_stage_routing.py` (new).
   - Change: channel-aware Focal construction, helper composition và optional
     state-free output taps; giữ historical registration/init.
   - Verify: `python -m pytest tests/test_backbone_attention_integration.py tests/test_attention_stage_routing.py tests/test_light_attention.py tests/test_focal_context.py -q`.
3. Identity, run names và reports
   - Files: `tools/kitti_training_pipeline/common.py`,
     `tools/benchmarks/profile_detector.py`, `tests/test_grouped_checkpoint.py`,
     `tests/test_attention_stage_checkpoint.py` (new).
   - Change: stage-aware names/report, resume/evaluation tests và fixture replay.
   - Verify: `python -m pytest tests/test_grouped_checkpoint.py tests/test_attention_stage_checkpoint.py -q`.
4. Notebook config và controls
   - Files: `tools/kitti_training_pipeline/notebook_config.py`,
     `tools/generate_robust_notebook.py`,
     `3D_Lidar_Object_Detection_Notebook_standard.ipynb` (chỉ cell cấu hình),
     `3D_Lidar_Detection_Pipeline_Robust.ipynb` (generated),
     `tests/test_under1m_notebook_controls.py`,
     `tests/test_under1m_notebook_config.py`.
   - Change: conditional controls emission, stale-field cleanup và placement summary.
   - Verify: `python -m pytest tests/test_under1m_notebook_controls.py tests/test_under1m_notebook_config.py tests/test_attention_presets.py -q`; regenerate bằng entry point hiện có của generator.
5. Notebook-driven coverage và training instructions
   - Files: `tests/test_attention_stage_notebook.py` (new),
     `tools/kitti_training_pipeline/README.md`.
   - Change: chạy config cell với 19 tổ hợp controls, kiểm tra config trong bộ nhớ,
     identity/run name và hướng dẫn đổi stage trên notebook. Không tạo recipe files.
   - Verify: `python -m pytest tests/test_attention_stage_notebook.py -q`; profile
     19 model configs trong bộ nhớ bằng meta device.
6. Consumers của tên stage cố định
   - Files: `tools/visualization/export_kitti_pipeline.py`,
     `tools/visualization/compare_attention_placements.py`,
     `tools/visualization/draw_bev_backbone_heads.py`, các consumer tương ứng
     trong `Demo/code/` nếu cần; `tests/test_kitti_pipeline_visualization.py`,
     `tests/test_attention_placement_visualization.py`.
   - Change: capture/report đúng effective placement; giữ mode intervention
     riêng với trained-checkpoint visualization. Schematic đọc metadata, không
     hard-code module name → stage mapping.
   - Verify: `python -m pytest tests/test_kitti_pipeline_visualization.py tests/test_attention_placement_visualization.py -q`; CPU actual-checkpoint smoke và kiểm tra hình/capture metadata.
7. Final verification và docs
   - Files: `docs/mobilepixornext_architecture.md`,
     `docs/plans/lightweight_lidar_backbone_2026/attention_specification.md`,
     `tools/kitti_training_pipeline/README.md`.
   - Change: cập nhật current stage contract, giữ rõ các kết quả cũ chỉ thuộc
     topology cũ; ghi checks/hardware skips.
   - Verify: toàn bộ focused suites phía trên, sau đó
     `python -m pytest tests/ -q`, `git diff --check`, real P16 strict replay.

Các path ở plan tương đối với `Code/`. Lệnh `python` dùng interpreter có PyTorch;
trong máy hiện tại là `/home/duyennh/miniconda3/envs/AI_env/bin/python`.
Khi chạy tests từ `Code/`, set
`PYTHONPATH=.:detector:tools/kitti_training_pipeline` nếu suite yêu cầu import trực tiếp.

### Risks & mitigations

- Identity migration: test golden version-1 metadata trước khi thay resolver.
- Shape-equivalent reorder: test explicit mismatch rejection, không dựa shape.
- Stale notebook options: validate raw external config; chỉ cleanup ở bước xây
  custom override có ngữ nghĩa rõ, không silently sanitize file config sai.
- New initialization: train mới mỗi topology, không lấy intervention figure làm AP.
- Scope tăng: không thêm multi-stage instances, warm-start mapping hay loss/head
  changes trong release này.

### Rollback plan

Chỉ revert các thay đổi source/config mới thuộc stage selection. Không đổi hoặc
xóa dataset, checkpoint, run cũ, visualization đã tạo hoặc notebook người dùng.
Vì historical prefixes và identity v1 được giữ, checkpoint cũ vẫn dùng được sau
rollback. Run topology mới giữ artifacts riêng và cần code hỗ trợ v2 để replay.

## 9. Quyết định chốt cho spec

- Mỗi module một stage; C3/C4/C5.
- Hai thứ tự khi cùng stage; mặc định Focal rồi local attention.
- Ba public fields mới; không rename field/model prefix lịch sử.
- Default legacy giữ semantic version 1; nonlegacy topology dùng resolver version 2.
- Người dùng tự chọn tổ hợp trên notebook; tests kiểm chứng 19 topology trong
  bộ nhớ. Không tạo config files/presets và không tự chạy training.
- Nhiều instance của cùng module tại nhiều stage, stem attention và transfer
  checkpoint sang topology khác là follow-up riêng nếu được yêu cầu.
