# Encoding và processing trước khi tăng độ phức tạp backbone

### Goal

Cải thiện đầu vào hình học cho MobilePixorNeXt phục vụ KITTI, giữ backbone dưới 1M và khả năng triển khai TensorRT/Jetson. Ưu tiên một encoder thống kê gọn, chưa có claim tăng AP khi chưa huấn luyện.

### Constraints

- Giữ grid 0,1 m và interface HWC hiện tại của dataset; model nhận BCHW.
- Không có ablation dài; vẫn cần huấn luyện khi thay định nghĩa input.
- Tách thời gian encoding khỏi inference; không gọi encoder parameter-free là compute-free.
- Encoding mới phải có tên/config riêng, thống nhất train/eval/export/notebook và metadata checkpoint.

### Known context

`rich8` có ba binary height bands, max/mean z, max/mean intensity và log-density. `rich12` đã có height span/std và density bù range. Augmentation diễn ra trước encoding trong dataset hiện tại. Backbone gồm neck là 674.256 tham số với input 8 channels.

Height histogram là prior art của [PillarHist](https://arxiv.org/html/2405.18734v1). Encoder raster dưới đây là adaptation đề xuất, không phải implementation nguyên bản của paper.

### Risks

- Nhiều channels tăng activation, input transfer và stem MACs.
- Histogram có thể ít thông tin trong cell một point; centroid là vị trí point quan sát được, không phải object center.
- Range compensation có thể khuếch đại noise; hard ground removal có thể xóa point của vật thể nhỏ.
- Thay encoding làm checkpoint cũ không tương thích hoàn toàn; không đổi ngầm ý nghĩa rich8/rich12.
- Cache BEV trước augmentation sẽ không đúng dữ liệu train đã biến đổi.

### Options (2–4)

1. Giữ rich12 và tối ưu implementation: ít thay đổi, phù hợp ưu tiên latency, nhưng không bổ sung phân bố z chi tiết.
2. Histogram raster + geometric statistics: thêm tín hiệu chiều cao và vị trí trong cell, không learned encoder; chi phí và tích hợp vừa phải.
3. Learned pillar encoder với pooling: biểu diễn linh hoạt hơn, nhưng thêm xử lý per-point/scatter và rủi ro deployment.
4. Ground-relative encoding hoặc grid đa độ phân giải: tiềm năng hình học cao, nhưng phụ thuộc ground estimator hoặc fusion nhiều grid; chưa ưu tiên nếu thiếu thời gian.

### Recommendation

Chọn phương án 2 trước, giữ backbone/neck/loss hiện tại ở vòng đầu. Bản dự kiến 14 channels: 4 histogram count theo z, 4 height statistics (max, mean, span, std), 2 intensity statistics (max, mean), log-density, occupancy và 2 mean point offsets x/y so với cell center. Histogram/count dùng normalization cố định; offsets chuẩn hóa theo cell size, giữ dấu và có occupancy để phân biệt empty. Không lấy observed centroid thay target center.

Giữ density thô và range riêng nếu thử gating sau này; không mặc định quadratic range compensation là confidence. Bắt đầu với 4 height bins; số bin tối ưu chưa được xác minh. Bản 12 channels bỏ hai offsets là cấu hình đơn giản hơn, không phải baseline đã chạy.

Processing: dùng một pass tích lũy count, histogram, sums, extrema và sums-of-squares bằng kernel compiled khi backend có lợi; normalize và tạo dense output sau. Dùng float32, clamp variance âm do roundoff, xử lý empty/single-point cells rõ ràng. Có reference NumPy để đối chiếu và fallback. Cache index grid/range tĩnh; cache encoding chỉ cho dữ liệu không augmentation, khóa theo geometry/config.

Nếu giữ stem hiện tại, input 8→14 thêm 1.728 weights, backbone gồm neck dự kiến 675.984; Conv MAC tăng 0,2433024G cho input batch 1. Đây là tính toán shape/weight, không phải phép đo latency. Sau encoding mới, mới tăng stage stride 4 từ 2→3 blocks nếu tiếp tục thay backbone.

### Acceptance criteria

- Đúng histogram/offset/statistics trên point cloud biết trước; permutation invariant và đúng trục x/y.
- Empty và single-point cells finite; out-of-range/NaN được xử lý theo quy tắc xác định.
- Backend compiled khớp reference trong tolerance; augmentation và labels giữ đúng hệ tọa độ.
- Config/model/export/notebook dùng cùng channel specification; checkpoint lưu schema encoding.
- Đếm backbone <1M; đo thời gian encoding/transfer/model riêng khi có phần cứng.
- Không coi unit tests hay parameter counts là bằng chứng tăng accuracy hoặc SOTA.
