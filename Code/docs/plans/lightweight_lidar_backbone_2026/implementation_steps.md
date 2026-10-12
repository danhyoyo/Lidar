# Lộ trình encoding → backbone, cấu hình mới không dùng LiteMLA

> Historical outline. The authoritative detailed English documents are [specification.md](specification.md) and [implementation_plan.md](implementation_plan.md), including grouped-head compatibility and the explicit BEV/3D milestones.

### Goal

Xây dựng một biến thể MobilePixorNeXt dùng histogram hình học, processing gọn và CNN BEV không LiteMLA, backbone gồm neck dưới 1M. Thực hiện trên `research/mobilepixornext-under1m`, đã cập nhật hybrid GT sampler tại `3654a0b`. Đây là kế hoạch triển khai, chưa phải mã đã thay đổi hoặc kết quả accuracy.

### Assumptions

- KITTI, grid 0,1 m; output stride 4; channels stage 32/48/96/128; expansion 2,5. Cấu hình mới ưu tiên neck/head input 32 channels; ngân sách mô phỏng 16 channels ở cuối tài liệu là mốc tham chiếu trước khi tăng width.
- Ưu tiên GPU để so mAP với thuật toán khác. Bước 6 và tối ưu Jetson/TensorRT ở bước 8 được hoãn sang giai đoạn sau.
- Nếu bật grouped heads, OGA/IQA và mọi strategy/pipeline hiện được hỗ trợ phải đạt [acceptance tương thích](grouped_heads_compatibility.md). Các tổ hợp config hiện bị cấm không được bypass.
- Không dùng distillation. Cấu hình mới đặt `c4_attention=none`; tùy chọn LiteMLA cũ vẫn giúp đọc cấu hình/checkpoint lịch sử.
- Giữ recipe hybrid GT sampler và loss cố định khi chọn kiến trúc; không thay encoding, augmentation và loss một cách ngầm định.
- Cần huấn luyện một model cuối khi đổi input hoặc thêm block. Functional tests không chứng minh tăng AP.
- Có các thay đổi Q-OGA/notebook chưa commit của người dùng; không gộp chúng vào commit encoding/backbone một cách vô tình.
- Các file test/module mới ghi dưới đây là deliverable dự kiến. Môi trường hiện có: `export LIDAR_PYTHON=/tmp/lidar-fixes-env/bin/python`; dùng môi trường tương đương nếu chạy trên máy khác.
- Tách quy mô: bước 1–5 tạo phiên bản đầu; bước 6 là tối ưu stem tùy chọn; bước 7 là research context tùy chọn; bước 8–9 là xác nhận integration/deployment/3D task.

### Plan

1. Định nghĩa schema encoding và thống nhất cách tính channels
   - Files: `detector/core/bev_encoding.py` (mới), `detector/core/datasets/utils_1/preprocess.py`, `detector/core/models/model.py`, `tools/kitti_training_pipeline/common.py`, `tools/kitti_training_pipeline/notebook_config.py`, `tests/test_bev_encoding_spec.py` (mới).
   - Change: thêm encoding có tên riêng `hist14`; metadata gồm version, thứ tự channels, height-bin edges, geometry và normalization. Dùng chung schema để resolve input channels cho dataset/model/export/notebook/run naming. Giữ ngữ nghĩa rich8/10/11/12 hiện có.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_bev_encoding_spec.py tests/test_rich_encodings.py tests/test_run_name_and_logging.py`; encoding lạ/config không hợp lệ phải báo lỗi rõ; hist14 trả đúng input channels 14.

2. Viết encoder NumPy reference cho hist14
   - Files: `detector/core/datasets/utils_1/preprocess.py`, `tests/test_hist14_encoding.py` (mới).
   - Change: 4 bins height count, 4 stats max/mean/span/std z, 2 stats max/mean intensity, log-density, occupancy, mean point offsets x/y trong cell. Bắt đầu với 4 equal-width bins trong z bounds; chưa coi số bin này là tối ưu.
   - Change: normalize histogram counts và density bằng log-count cố định; z theo geometry; offsets `(mean_x-cell_center_x)/x_res` và tương tự y, có dấu. Empty cells bằng zero, variance được clamp ở zero trước sqrt. Mean point offset là input feature, không thay target object center.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_hist14_encoding.py`; kiểm tra cell rỗng/một point, histogram có count biết trước, permutation invariance, đúng trục x/y, NaN/out-of-range và biên bins; output HWC float32 finite.

3. Tối ưu processing với backend compiled có reference đối chiếu
   - Files: `detector/core/datasets/utils_1/bev_backend.py` (mới), `detector/core/datasets/utils_1/preprocess.py`, `tests/test_hist14_backend.py` (mới), `tools/benchmarks/benchmark_bev_encodings.py` (mới).
   - Change: một lượt tích lũy count/bin counts/sums/extrema/squared sums và offsets; chuẩn hóa dense output sau. Có lựa chọn backend rõ ràng, NumPy fallback, không đổi số học/ngữ nghĩa giữa train và deploy.
   - Change: chỉ cache dữ liệu hình học tĩnh hoặc mẫu không augmentation. Encoder train tiếp tục chạy sau hybrid/global augmentation. Ghi thời gian compile/warmup riêng, không đưa vào steady-state throughput.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_hist14_backend.py tests/test_hybrid_augmentation.py`; compiled output khớp reference trong tolerance. Benchmark bằng CLI mới với point count, shape, dtype, hardware và warmup được ghi vào JSON; tốc độ chỉ được claim khi có phép đo.

4. Cấu hình CNN không LiteMLA trước khi thêm capacity
   - Files: `tools/kitti_training_pipeline/notebook_config.py`, `3D_Lidar_Object_Detection_Notebook_standard.ipynb`, `configs/experiments/under1m/hist14_conv.json` (mới), `tests/test_notebook_experiment_config.py`, `tests/test_mobilepixornext_backbone.py`.
   - Change: biến thể mới chọn hist14, `c4_attention=none`, `c4_attention_scales=[]`, `c4_attention_qk_norm=none`; SG-FPN, expansion 2,5, depth 2/4/2 và head input 32 ở mốc này. Code đã hỗ trợ `nn.Identity` tại vị trí attention.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_mobilepixornext_backbone.py tests/test_notebook_experiment_config.py tests/test_standard_training_notebook.py`; xác nhận biến thể mới không khởi tạo LiteMLA, forward 14 channels và backward có gradient finite.

5. Tăng xử lý local geometry tại stride 4
   - Files: `detector/core/models/backbones/mobilepixornext.py`, `detector/core/models/backbones/registry.py`, `tools/kitti_training_pipeline/notebook_config.py`, `configs/experiments/under1m/hist14_local3_conv.json` (mới), `tests/test_mobilepixornext_backbone.py`, `tests/test_model_registry.py`.
   - Change: thêm `stage_depths` có validation và mặc định cũ `(2,4,2)`. Preset mới chọn `(3,4,2)`: thêm một block 48ch ở stage stride 4, expansion 2,5. Depth/channel và output shape được ghi trong config/run name.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_mobilepixornext_backbone.py tests/test_model_registry.py`; preset local3 có output `(B,32,200,176)` từ `(B,14,800,704)`; backbone gồm neck 635.408 với topology chuẩn đã mô phỏng (16-channel reference là 631.920). Đếm lại từ implementation cuối.

6. Tùy chọn giảm compute stem để phục vụ Jetson
   - Files: `detector/core/models/backbones/mobilepixornext.py`, `detector/core/models/backbones/registry.py`, `tools/kitti_training_pipeline/notebook_config.py`, `tests/test_mobilepixornext_backbone.py`.
   - Change: thêm `stem_type=legacy|depthwise`. Bản depthwise giữ Conv3×3 stride 2 đầu tiên, thay Conv3×3 thứ hai bằng DW3×3 + BN + SiLU + PW1×1 + BN + SiLU. Không đặt đây là cải thiện accuracy đã biết.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_mobilepixornext_backbone.py`; shape/backward đúng, backbone gồm neck dự kiến 627.568 ở output 32. Profile operator trên TensorRT/Jetson: ít MAC hơn không tự bảo đảm ít latency hơn. Bước này đang hoãn; dùng legacy stem cho model đầu.

7. Tùy chọn context convolution/gating sau phiên bản CNN đầu
   - Files: `detector/core/models/backbones/mobilepixornext_blocks.py`, `detector/core/models/backbones/mobilepixornext.py`, `detector/core/models/backbones/registry.py`, `tests/test_focal_context.py` (mới).
   - Change: thử context module dense tại stride 8: bottleneck 64, DW convolution nhiều mức, pooled context và Hadamard gating/residual. Thêm field riêng `c4_context`; biến thể CNN đầu chọn `none`. Focal module không cần LiteMLA.
   - Change: chỉ thêm support-aware pooling khi đã định nghĩa đúng occupancy/count pyramid và epsilon. Không hard-mask center prediction ở empty cells. Range không được dùng thay số points/occlusion.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_focal_context.py tests/test_mobilepixornext_backbone.py`; kiểm tra empty-support, finite gradients, residual initialization, output shape và ONNX support. Khoảng 25.120 tham số là estimate của generic prototype, chưa gồm mọi conditioning adapter.

8. Hoàn thiện pipeline, deployment và dữ liệu sau sampler
   - Files: `tools/kitti_training_pipeline/common.py`, `tools/kitti_training_pipeline/export_onnx.py`, `tools/kitti_training_pipeline/build_tensorrt.py`, `tools/kitti_training_pipeline/compare_models.py`, `tools/kitti_training_pipeline/train.py`, `tests/test_kitti_config_compatibility.py`, `tests/test_hybrid_augmentation.py`, `tests/test_standard_training_notebook.py`.
   - Change: lưu schema encoding/architecture vào resolved config và checkpoint metadata; báo lỗi mismatch thay vì nạp checkpoint cũ âm thầm. Notebook có preset mới, dùng `AUGMENTATION=hybrid_gt`, geometry/config được dùng thống nhất khi prepare/train/eval/export.
   - Change: train smoke trên dữ liệu tổng hợp hoặc vài batch trước full training. Đánh giá GPU/mAP trước; ONNX agreement, TensorRT FP16 và INT8 là phần deployment được hoãn.
   - Verify: `$LIDAR_PYTHON -m pytest -q tests/test_kitti_config_compatibility.py tests/test_hybrid_augmentation.py tests/test_notebook_experiment_config.py tests/test_standard_training_notebook.py`; trên GPU/Jetson đo riêng encoding, transfer, model và decode/NMS ở batch 1; ghi power mode/versions và mean/p50/p95.

9. Hoàn thành task 3D trước khi dùng nhãn SOTA 3D
   - Files: `detector/core/models/heads/cnn.py`, `detector/core/datasets/utils_1/target_backend.py`, `detector/core/datasets/dataset.py`, `detector/core/losses/loss_fn.py`, các loss strategies hỗ trợ, `detector/postprocess.py`, `tools/kitti_training_pipeline/evaluate_kitti_3d.py` (mới), tests target/decode/evaluation 3D (mới).
   - Change: head/target/loss/decode dự đoán đủ `(x,y,z,l,w,h,yaw)`; định nghĩa z center/bottom rõ ràng. Chuyển LiDAR/camera đúng calibration, evaluator KITTI chuẩn xử lý ignored/DontCare/difficulty. Việc này là một gói riêng ngoài backbone.
   - Verify: kiểm tra roundtrip GT → targets → decoded boxes trên synthetic scenes với calibration biết trước; reference IoU và evaluator parity; đánh giá AP3D R40 sau training. Nếu chưa làm bước này, chỉ báo cáo BEV detection.

### Risks & mitigations

- Histogram14 chưa được đánh giá AP; kiểm tra đơn vị chỉ xác nhận đúng phép tính. Mean offsets có thể chịu bias của bề mặt nhìn thấy; giữ dưới dạng feature.
- Bỏ LiteMLA có thể giảm accuracy dù giảm params/MAC. Không lấy count làm bằng chứng chất lượng.
- Thêm block ở stride 4 tăng compute rõ rệt. Bản hist14 local3 legacy stem có Conv MAC cao hơn phần convolution baseline cũ; cần báo cáo đúng, không gọi nó nhanh hơn trước khi đo.
- Depthwise stem có thể giảm capacity hoặc bị giới hạn memory bandwidth; tách thành option. Không tăng thêm detail skip/RC-BiSGFPN cùng lúc.
- Functional smoke/test/ONNX validation có thể làm trước khi train một model cuối; không thay thế accuracy benchmark. Không hứa SOTA tuyệt đối khi thiếu thực nghiệm.
- Backbone parameter budget được báo cáo cả main body và gồm neck; encoder/head/total tách riêng. Không dùng INT8 để tuyên bố giảm parameter count.

### Rollback plan

Mỗi bước có config hoặc option riêng và commit chỉ chứa files của bước đó. Giữ legacy encodings/stem/depth defaults và cấu hình lịch sử để tái lập checkpoint. Nếu backend compiled lỗi, chọn NumPy reference; nếu stem/context chưa được xác nhận, dùng legacy stem và context none. Rollback bằng cấu hình hoặc revert commit của bước tương ứng, bảo toàn các thay đổi Q-OGA chưa commit của người dùng. Không reset toàn workspace.

## Ngân sách mô phỏng cấu trúc

Đã khởi tạo prototype tạm trên meta device từ code hiện tại; không sửa implementation repo. Số MAC bên dưới chỉ tính Conv2d, không phải latency hoặc tổng FLOPs. [Bằng chứng](implementation_budget.json)

| Prototype | Backbone gồm neck | Conv GMAC toàn detector BEV |
| --- | ---: | ---: |
| rich8, bỏ LiteMLA | 616.080 | 7,1194 |
| input14, bỏ LiteMLA | 617.808 | 7,3627 |
| input14, stage depth 3/4/2 | 631.920 | 7,8510 |
| input14, stage depth 3/4/2, depthwise stem | 624.080 | 6,7381 |

Phiên bản đầu ưu tiên bước 1–5: **hist14 → legacy stem → stages 3/4/2 → SG-FPN output 32 → head**, không LiteMLA và không bắt buộc context mới. Grouped heads là lựa chọn có acceptance riêng cho loss/IQA/targets/decode/resume. Backbone gồm neck là 635.408 trong prototype 32-channel; chất lượng vẫn cần xác nhận trên model được huấn luyện. Jetson optimization được hoãn.
