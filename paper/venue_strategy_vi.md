# Đề xuất nơi nộp

Ngày đối chiếu thông tin: 13/10/2026. Đây là đánh giá mức phù hợp của bài; chưa khẳng định Q1/Q2, CORE rank hoặc xác suất nhận. Các nhãn đó cần đối chiếu đúng cơ sở dữ liệu và năm mà trường/đơn vị của bạn sử dụng.

## Lựa chọn đề xuất

**Mục tiêu journal: IEEE Transactions on Intelligent Vehicles (T-IV).** Scope có vehicle environment perception và xử lý LiDAR; regular paper được gợi ý khoảng 10 trang IEEE hai cột. Mình chọn đây làm hướng định vị bản thảo. Để cạnh tranh, project cần thêm baseline cùng protocol và bằng chứng về tính ổn định, novelty và ứng dụng. Đánh giá mức cạnh tranh này là nhận định của mình. [Scope và hướng dẫn T-IV](https://ieee-itss.org/pub/t-iv/).

**Nếu ưu tiên hội nghị: IEEE Intelligent Vehicles Symposium (IV).** Hướng perception cho xe của bài phù hợp. IEEE IV và ITSC được ITSS liệt kê là hai hội nghị flagship; đây là thông tin của society, không tự quy đổi thành CORE A/A*. [IEEE IV](https://ieee-itss.org/conf/iv/), [IEEE ITSC](https://ieee-itss.org/conf/itsc/).

| Nơi nộp | Mức ưu tiên của mình | Điều kiện để bản thảo phù hợp hơn |
|---|---|---|
| IEEE T-IV | Journal mục tiêu đầu tiên, tham vọng | Thực nghiệm cùng evaluator, baseline mạnh, nhiều seed; làm rõ geometry–quality và bottleneck vertical |
| IEEE IV | Hội nghị mục tiêu đầu tiên | Cô đọng vào một thông điệp và một bộ ablation; bổ sung baseline và benchmark theo yêu cầu kỳ tổ chức |
| IEEE ITSC | Hội nghị thay thế đáng cân nhắc | Nhấn mạnh perception trong hệ thống giao thông; kiểm tra CFP và template của kỳ dự định nộp |
| IEEE Transactions on Intelligent Transportation Systems | Journal tham vọng khác | Cần câu chuyện gắn với ITS và đánh giá rộng hơn; scope bao gồm nghiên cứu công nghệ cho giao thông thông minh |
| IEEE Robotics and Automation Letters | Có điều kiện | Phù hợp hơn nếu bổ sung ứng dụng robot/xe tự hành thực và đóng góp rõ; 6 trang với tối đa 2 trang thêm, mọi appendix phải nằm trong giới hạn 8 trang; cần rút gọn gói bản thảo hiện tại |
| IEEE Sensors Journal | Phương án journal thiên về sensing | Định vị xử lý dữ liệu LiDAR, hiệu quả tính toán và giới hạn đo; vẫn cần novelty và thực nghiệm thuyết phục |

Nguồn cho các scope/định dạng còn lại: [T-ITS](https://ieee-itss.org/pub/t-its/), [RA-L](https://www.ieee-ras.org/publications/ra-l/), [RA-L author information](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/), [IEEE Sensors Journal](https://ieee-sensors.org/ieee-sensors-journal/). Phần ưu tiên và điều kiện trong bảng là đánh giá riêng cho project này.

## Vì sao chưa chọn CVPR/ICCV/NeurIPS làm hướng chính?

Với bằng chứng đã cung cấp, phần lớn thành phần kiến trúc và objective dựa trên ý tưởng đã công bố: large-kernel convolution, FPN, focal modulation, SimAM, projection geometry và quality estimation. Điểm mạnh hiện nay là hệ thống nhỏ, ablation tương tác và diagnostic vertical. Để nhắm các hội nghị thị giác/machine learning tổng quát hàng đầu, cần một đóng góp phương pháp đủ rộng cùng thực nghiệm vượt ra ngoài một dataset và một seed. Đây là nhận định nghiên cứu của mình, không phải quy định chính thức hay dự đoán quyết định reviewer.

## Thứ tự nâng cấp bài để nhắm T-IV hoặc IV

1. **Chốt protocol chung.** Dùng reference R40 cho Car/Pedestrian/Cyclist; cùng split, input ROI, NMS, precision và hardware khi so sánh. Nếu giữ local evaluator, trình bày thêm như ablation nội bộ.
2. **Baseline tối thiểu.** So với PIXOR/PointPillars và ít nhất một detector pillar hiện đại có code công khai, trong điều kiện có thể đối chiếu công bằng. Không lấy FPS và AP từ các máy/protocol khác rồi coi là cùng benchmark.
3. **Lặp các thí nghiệm quyết định.** Ít nhất ba seed cho P1–P4 và P4/P16; báo cáo mean/std và chi phí chạy. Run shared/independent 3D cần cùng 100 epoch và cùng tiêu chí chọn checkpoint.
4. **Tách đóng góp OGA.** Coordinate-only, thêm projection, thêm corner, thêm weighting; nếu nghiên cứu quality 3D thì so surrogate BEV, exact BEV và quality cuboid trong cùng điều kiện.
5. **Củng cố vertical.** Kiểm tra lỗi theo range, occlusion và ground geometry. Oracle hiện tại chỉ là phân tích có điều kiện; mọi cải tiến phải được đánh giá lại bằng reference AP.
6. **Bằng chứng tổng quát hoặc triển khai.** Dataset thứ hai cần cấu hình/adaptation rõ; nếu chủ đề chính là edge, đo trên thiết bị đích với batch một, warmup, precision, p50/p95 và pipeline hoàn chỉnh.

Các bước này là đề xuất công việc tiếp theo, chưa được chạy trong quá trình viết bài. Bản thảo hiện tại đã dùng đúng số liệu đã có, nên có thể xem xét và chỉnh sửa ngay.

## Chuẩn bị trước khi nộp

Thêm tên tác giả, affiliation, email, corresponding author, funding và thông tin công bố code/data. T-IV có yêu cầu tác giả/biography theo hướng dẫn chính thức; kiểm tra lại template và quy trình tại ngày nộp. Với IV/ITSC, chọn kỳ tổ chức còn nhận bài rồi kiểm tra CFP, deadline và page limit của chính kỳ đó. Tài liệu này không đưa deadline chưa xác minh và chưa thực hiện nộp bài.
