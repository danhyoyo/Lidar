# So sánh grouped head hiện tại và đề xuất

[Hình so sánh PNG](comparison.png) · [SVG](comparison.svg) · [PDF vector](comparison.pdf).

Các bản riêng để chèn vào tài liệu:

- Hiện tại: [PNG](current_heads.png), [SVG](current_heads.svg), [PDF](current_heads.pdf).
- Đề xuất: [PNG](proposed_heads.png), [SVG](proposed_heads.svg), [PDF](proposed_heads.pdf).

Cả hai hình có hai group Car và Pedestrian/Cyclist, cùng nhận toàn bộ feature
`32×200×176`. Mỗi group có 5 nhánh: heatmap, offset, size, yaw và IoU quality.
Shape là `C×H×W`, bỏ batch. Tất cả Conv3×3 trong head dùng stride 1, padding 1,
bias=False, BN và SiLU. Conv1×1 cuối có bias, không BN/activation.

Hình trái lấy cấu trúc từ `detector/core/models/heads/grouped.py` và `cnn.py`:
mỗi nhánh sở hữu hai Conv3×3 riêng rồi một Conv1×1 ra prediction.

Hình phải là proposed đã triển khai: mỗi group sở hữu một Conv3×3 chung, rồi mỗi nhánh sở hữu
một Conv3×3 riêng và một Conv1×1. Conv chung thay thế Conv3×3 đầu tiên của 5
nhánh trong group. Hai group vẫn có trọng số riêng; mỗi đường dự đoán vẫn đi
qua hai Conv3×3. Chọn `model.header_architecture="shared"` để dùng hình phải;
`"independent"` hoặc không khai báo dùng hình trái. Hai notebook có control
`HEADER_ARCHITECTURE` tương ứng. Config và checkpoint cũ tiếp tục dùng hình trái.

| Chi phí toàn bộ hai group head | Hiện tại | Đề xuất |
| --- | ---: | ---: |
| Conv3×3 | 20 | 12 |
| Conv1×1 | 10 | 10 |
| Tham số, gồm BN affine và bias | 186,161 | 111,921 |
| MAC convolution tại 200×176, batch 1 | 6,507,212,800 | 3,911,987,200 |

MAC không bao gồm backbone, BN, activation hoặc postprocess. Những số này là
chi phí kiến trúc, không phải phép đo latency. Chưa có kết quả train/AP cho
phương án đề xuất.

[comparison.json](comparison.json) ghi nguồn cấu hình, hash code, số tham số
và shape output. Generator chạy hai kiến trúc production GroupedHeader
trên cùng một tensor ngẫu nhiên `1×32×200×176` ở CPU eval, kiểm tra output hữu
hạn và shape khớp cho cả 10 nhánh. Renderer kiểm tra chữ không chồng nhau hoặc
bị cắt trước khi xuất.

Chạy từ thư mục `Code` để tạo lại:

```bash
python tools/visualization/draw_head_comparison.py
```

Mỗi hình có PNG 300 DPI, SVG chỉnh sửa được, PDF vector và `.tex` dùng cuboid
PlotNeuralNet từ `third_party/PlotNeuralNet`. Khi dùng Overleaf, tải `.tex` cùng
`layers/`. TikZ chưa được biên dịch vì máy thiếu LaTeX compiler; PDF xuất trực
tiếp bằng Matplotlib được kiểm tra riêng.
