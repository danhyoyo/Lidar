# Phân tích bài tham khảo và định vị MobilePixorNeXt

## 1. Bài tham khảo đã làm gì?

Bài **Enhancing Indoor Robot Pedestrian Detection Using Improved PIXOR Backbone and Gaussian Heatmap Regression in 3D LiDAR Point Clouds**, IEEE Access 2024, phát triển IRBGHR-PIXOR: backbone dùng inverted residual, dự đoán tâm bằng Gaussian heatmap và focal loss sửa đổi. Bài báo báo cáo AP 97,17% tại IoU 0,5 trên JRDB cho pedestrian. Đây là kết quả của tác giả trong bài gốc, chưa được tái lập trong lần chuẩn bị bản thảo này. [Bài gốc, DOI](https://doi.org/10.1109/ACCESS.2024.3351868).

Điểm có thể học là cách tổ chức chuỗi lập luận từ biểu diễn BEV, backbone gọn, supervision tâm, đến ablation. Khi tiếp nối bài này, cần nêu rõ kiến trúc, dữ liệu và objective mới; thay dataset hoặc ghép thêm attention đơn thuần khó tạo một đóng góp mạnh. JRDB có nhiều môi trường trong và ngoài nhà; bài tham khảo định vị ứng dụng robot trong không gian có người, còn project hiện tại đánh giá ba lớp trên cảnh giao thông KITTI.

Không dùng 97,17% của JRDB để đối chiếu trực tiếp với mAP KITTI. Dataset, lớp đối tượng, ROI, IoU, cách tính AP và evaluator khác nhau. Bản thảo trích dẫn bài gốc như tiền đề nghiên cứu.

## 2. Những điểm khác của project hiện tại

| Thành phần | IRBGHR-PIXOR | Project hiện tại |
|---|---|---|
| Bài toán | Pedestrian với JRDB | Car, Pedestrian, Cyclist với KITTI |
| Biểu diễn | BEV với các lát chiều cao/reflectance | Hist14: histogram chiều cao, thống kê Z/intensity, density, occupancy, offset |
| Backbone | Inverted residual | Depthwise 7×7, expansion 2,5, SiLU, residual scale |
| Fusion/context | Kiến trúc của bài gốc | Scale-gated FPN, focal context tại stride 8, SimAM tại stride 4 |
| Objective | Gaussian heatmap và focal loss sửa đổi | Gaussian focal, OGA, adaptive task weighting, IQA hoặc Q-OGA |
| Head | Theo bài gốc | Hai nhóm Car và Pedestrian/Cyclist; run 3D cuối dùng convolution chia sẻ |
| Vertical | Không phải baseline 3D tương ứng của KITTI hiện tại | Run Final3D đã huấn luyện branch z-bottom/log-height |

MGIoU, SimAM, ECA, focal loss và focal modulation đều có công trình gốc, được trích dẫn trong `references.bib`. OGA là gói objective triển khai dựa trên projection geometry, corner distance và bounded task weighting; không mô tả MGIoU như công thức mới của project. IQA dùng clamped mean projection GIoU của footprint, chưa phải exact rotated IoU hay 3D IoU. Tên biến chứa “uncertainty” không đủ để khẳng định một mô hình xác suất hoặc bảo toàn tổng gradient.

## 3. Câu chuyện khoa học nên đặt ở trung tâm

**Geometry và quality có lợi ích chồng lấp trong một detector gọn; khi mở rộng lên 3D, độ chính xác vertical trở thành giới hạn rõ rệt.** Đây là câu chuyện có bằng chứng cụ thể hơn việc liệt kê các module đã thêm.

Trong factorial P1–P4, OGA tăng 1,59 điểm phần trăm khi không có IQA, nhưng chỉ tăng 0,27 khi có IQA. IQA tăng 2,93 điểm với baseline và 1,61 với OGA. Interaction quan sát là −1,32 điểm. Điều này gợi ý hai cơ chế chia sẻ một phần lợi ích; chỉ có seed 42 nên chưa kết luận về ý nghĩa thống kê hoặc cơ chế nhân quả.

SimAM giúp P4 tăng từ 70,58% lên 70,99% mAP BEV local Moderate, chủ yếu qua Pedestrian (+1,03 điểm). ECA và detail path không nâng macro mAP trong các run tương ứng. Focal context cũng phụ thuộc cấu hình: tốt hơn trong OGA+IQA, thấp hơn trong baseline không IQA. Bài không nên khẳng định mọi module luôn cải thiện.

Cặp retrain O100/Q100 mới hoàn thành 100 epoch với SimAM và không có IQA; hai config chỉ khác tên objective. Q-OGA đạt 70,52% so với OGA 69,69% (+0,82 điểm), cả hai chọn checkpoint epoch 70. Pedestrian tăng 4,21 điểm, Car giảm 0,11 và Cyclist giảm 1,62. Khi chọn theo minimum validation loss, lợi thế chỉ còn 0,11 điểm. Đây là so sánh loss chính được cập nhật, dùng cùng seed 42; không thay các số factorial 50 epoch. Xem [phân tích cặp retrain](loss_comparison_vi.md).

## 4. Những số liệu có thể đưa vào bài

| Mô hình | Protocol | Moderate mAP | Tham số detector | Thông tin runtime |
|---|---|---:|---:|---|
| P4, BEV, 50 epoch | Local loader-aligned BEV R40 | 70,58% | 852.929 | Model 8,19 ms trên L4 FP32 |
| P16, BEV, 50 epoch | Local loader-aligned BEV R40 | 70,99% | 852.930 | Model 8,55 ms; pipeline offline 56,83 ms |
| O100, OGA, BEV, 100 epoch | Local loader-aligned BEV R40 | 69,69% | 815.744 | Model 7,88 ms; pipeline offline 47,32 ms |
| Q100, Q-OGA, BEV, 100 epoch | Local loader-aligned BEV R40 | 70,52% | 815.744 | Model 7,93 ms; pipeline offline 42,10 ms |
| Final3D, shared head, 100 epoch | Reference BEV R40 | 71,09% | 797.382 | Chưa có stage latency trong report |
| Cùng Final3D, checkpoint epoch 70 | Reference 3D R40 | 57,75% | 797.382 | Không gán runtime của P16 BEV cho run này |

Final3D có Car/Pedestrian/Cyclist 3D Moderate AP lần lượt 61,02/52,52/59,72%. Backbone gồm neck có 666.769 tham số, hai head có tổng 130.613; 12 tham số criterion dùng lúc train không được tính vào detector.

Giữa P16 BEV và Final3D, head, vertical branch, lịch train, tiêu chí chọn checkpoint và evaluator đều khác. Chênh lệch AP không chứng minh hiệu quả riêng của shared convolution. Số giảm 55.548 tham số là phép đối chiếu cấu trúc đã lưu, chưa thay thế ablation 3D có cùng lịch train.

## 5. Phân tích lỗi 3D có giá trị gì?

Diagnostic CPU giữ nguyên association theo BEV, scores, footprints và image boxes; oracle thay camera bottom vertical location/height. Trong 7.130 Car đã match BEV, số pass IoU 3D tăng từ 5.743 lên 6.791 nếu thay vertical location. Oracle này cứu 1.048/1.387 cặp đang fail, tương đương 75,6%. MAE LiDAR z-bottom 8,66 cm, bias chỉ −0,82 cm: lỗi theo từng đối tượng đáng chú ý hơn một dịch chuyển hằng toàn cục.

Đây là tỷ lệ pass có điều kiện, không phải oracle AP. Diagnostic không mô phỏng đầy đủ ignores/DontCare hoặc cách matching của reference evaluator; output CPU cũng khác GPU một lượng nhỏ. Bản thảo ghi rõ những giới hạn này. Hướng nghiên cứu tiếp theo hợp lý là quality cho cuboid 3D hoặc vertical-aware supervision, nhưng chưa có số liệu để khẳng định mức cải thiện.

## 6. Những điểm đã sửa khi tập hợp bằng chứng

- Dùng split thực tế 3.712 train/3.769 validation, thay số 5.984/1.497 trong tài liệu cũ. Hai manifest hiện tại không trùng frame.
- Đối chiếu 21 run và 41 evaluation với JSON gốc. Ô Excel bị diễn giải thành ngày không được dùng làm số đo.
- Bổ sung hai run O100/Q100 và bốn raw report mới: tổng 23 run BEV/45 report. Config mới đã khớp training hash; không cần phục hồi field head như 21 run cũ.
- Cả 21 config BEV nhận được có thêm `header_architecture=shared` sau run. Bỏ đúng field này phục hồi SHA-256 training ban đầu; checkpoint chứa branch convolution độc lập. Lưu hai bản trong `research/configs/` và audit, không sửa dữ liệu gốc.
- Dùng architecture vector tương ứng slide 21 với kích thước 200×176 và fusion 32. Một số slide cũ ghi Rich14, fusion 24 hoặc gán nhầm tên P3/P4, ECA/SimAM.
- Demo P4 dùng vertical extent ước lượng từ LiDAR; không gọi ảnh cuboid cũ là dự đoán của Final3D. Video playback 10 FPS không phải benchmark inference.
- AP-selected và loss-selected được ghi riêng. Với RC24, loss checkpoint cao hơn AP checkpoint; không đổi tiêu chí tùy run để chọn số đẹp hơn.

## 7. Những phần cần bổ sung trước khi nhắm nơi nộp mạnh

Ưu tiên baseline hiện đại cùng reference evaluator, seed lặp lại cho factorial chính, và ablation shared/independent 3D cùng budget. Tiếp theo là dataset thứ hai hoặc stress test có mục tiêu, cùng benchmark thiết bị đích nếu muốn tuyên bố ứng dụng edge. Hiện chưa có hidden-test result, chưa chứng minh vượt SOTA, và chưa có camera fusion. Chiến lược nộp cụ thể ở `venue_strategy_vi.md`; bản thảo đã viết hoàn chỉnh bằng bằng chứng hiện có.
