# Yêu cầu tương thích khi chia nhóm detection heads

> Historical compatibility outline. See [specification.md](specification.md), sections 3 and 6–10, for the detailed English contracts, and [implementation_plan.md](implementation_plan.md) for implementation and verification tasks.

Trạng thái: yêu cầu thiết kế/acceptance cho implementation sau này, chưa phải tính năng đã có. User yêu cầu OGA, IQA và pipeline hiện tại tiếp tục hoạt động khi chia heads. Ưu tiên hiện tại là GPU/mAP; Jetson/TensorRT optimization được hoãn.

## Thiết kế dự kiến

- Neck output 32 channels; head groups configurable, dự kiến `[Car]` và `[Pedestrian, Cyclist]`.
- Mỗi group có classification, offset, size, yaw và IoU branch riêng khi strategy hỗ trợ IQA.
- Schema prediction/target theo group phải giữ class mapping rõ ràng. Group definitions là partition đủ các lớp, không trùng, không thiếu.
- Single-head mode giữ contract/state-dict hiện tại để tái lập và đọc checkpoint cũ. Không coi checkpoint single-head là resume tương đương của model grouped-head.

## Target assignment

Mỗi group có heatmap, regression maps và `reg_mask` độc lập. Nếu Car và Pedestrian có center ở cùng cell, mỗi group phải giữ target của mình. Trong cùng group vẫn áp dụng quy tắc ownership/collision được định nghĩa; hai heads không loại bỏ collision Pedestrian/Cyclist. Giữ conventions metric offsets, log width/length và doubled yaw đúng mã hiện có.

Gaussian classification giữ C_group channels; binary classification cần local background và remapping labels riêng. Không chỉ slice global `reg_mask` hoặc lặp một target regression dùng chung vào cả hai groups.

## Loss compatibility

| Strategy / option | Yêu cầu |
| --- | --- |
| baseline | Classification/regression đúng group; IQA có supervision khi `use_iou=true`. |
| OGA | Geometry và uncertainty weighting dùng targets/masks đúng group; gradient qua regression và parameters của criterion. |
| OGA + IQA | IoU target lấy từ box của đúng group; IoU branch của mỗi group được giám sát, target detached; giữ selectable quality target methods. |
| Q-OGA legacy | Quality classification và range weighting theo group; giữ behavior hiện có, không tự suy ra có IQA supervision. |
| Q-OGA exact + warmup | Quality tại peak đúng group; `set_epoch` và quality curriculum được truyền tới criterion tương ứng. |
| GW-QAL | Quality/geometric terms và positive masks đúng group; không đồng nhất quality-coupled classification với IQA branch. |
| UWAG | Regression/geometry và learned weighting hoạt động đúng; không tạo IQA branch không có supervision. |

Loss composition phải được định nghĩa rõ: group weights explicit và có normalization để không vô tình tăng loss scale chỉ vì số groups tăng. Chọn và lưu weighting state theo thiết kế; không dùng cùng một stateful OGA strategy để gọi tuần tự nhiều groups rồi coi đó là một cập nhật EMA tương đương single-head. Nếu dùng một strategy instance cho mỗi group thì mọi parameters/buffers đều được optimizer/checkpoint quản lý.

Những tổ hợp đang bị cấm vẫn bị cấm: Q-OGA exact hiện yêu cầu Gaussian/BEV, reject `use_iou` và prediction `iou`. Grouping không được bypass các validation này. Nếu muốn hỗ trợ thêm exact Q-OGA + IQA hoặc 3D quality trong tương lai, đó là feature riêng cần loss/supervision/test riêng.

## Decode, quality ranking và NMS

Decode box bằng regression branch của đúng group. Quality score lấy từ IoU branch cùng group; không lấy score Car để rank box Pedestrian/Cyclist. Giữ cách kết hợp classification/quality và `nms_alpha` như cấu hình đã chọn. Remap local classes về global Car/Pedestrian/Cyclist trước evaluation. Giữ chính sách NMS/class filtering hiện có để so sánh công bằng; mọi thay đổi chính sách NMS là option riêng, không xuất hiện ngầm khi chia heads.

## Training, diagnostics và resume

- Một group không có GT vẫn có classification negatives, regression/IoU loss zero hữu hạn và backward hợp lệ.
- Quality means aggregate theo positive counts, counts cộng đúng; không trung bình mean của group rỗng với group có positive.
- Log aggregate và per-group classification/regression/geometry/quality diagnostics; không làm trainer hiểu diagnostic là loss component.
- AMP/gradient clipping/optimizer bao gồm toàn bộ head và criterion parameters. Curriculum epoch truyền tới mọi group.
- Checkpoint ghi groups/order, channel width, encoding schema, model/criterion/optimizer/scheduler và EMA/curriculum state.
- Resume cùng grouped architecture phải khôi phục objective/weights/buffers; model cũ đọc qua single-head mode. Warm-start sang architecture khác là thao tác explicit, không silent non-strict resume.

## Acceptance checks trước khi claim tương thích

1. Test mỗi supported strategy trong single và grouped modes, Gaussian/binary khi strategy hỗ trợ; IQA on/off ở các tổ hợp hợp lệ.
2. Single-head parity: cùng weights/input/targets cho output, loss, gradients và state-dict legacy đúng như trước.
3. Group isolation: đổi target Car không đổi raw loss/IoU targets của Pedestrian/Cyclist; kiểm tra collision cross-group cùng cell.
4. Empty-group và empty-scene: losses finite, gradients có đường đi đúng và quality telemetry không NaN.
5. IQA ranking test: hai groups có IoU logits khác nhau; decode dùng đúng quality của từng group và global labels không tráo.
6. Resume test: optimizer chứa criterion params, OGA/GW-QAL EMA và Q-OGA quality epoch được khôi phục.
7. Integration: hist14 + hybrid sampler + grouped heads + OGA/IQA chạy một training step và evaluator trả đủ ba lớp.

Chỉ khi các checks này pass mới gọi grouped-head implementation tương thích. Chưa có implementation hoặc test grouped-head ở thời điểm ghi tài liệu.
