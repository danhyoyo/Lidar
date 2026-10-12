# MobilePixorNeXt + SimAM, với hai group head

[PNG 300 DPI](architecture_bev_backbone_heads.png) ·
[SVG chỉnh sửa được](architecture_bev_backbone_heads.svg) ·
[PDF vector](architecture_bev_backbone_heads.pdf) ·
[PlotNeuralNet/TikZ](architecture_bev_backbone_heads.tex) ·
[Trang xem hình](index.html).

Hình vẽ hai Header thực sự tách biệt, cùng nhận feature `32×200×176` từ
backbone. Group Car có heatmap 1 kênh; group Pedestrian/Cyclist có heatmap 2
kênh theo thứ tự Pedestrian rồi Cyclist. Mỗi group sở hữu các nhánh offset,
size, yaw (mỗi nhánh 2 kênh) và IoU quality (1 kênh), độc lập với group còn lại.
Mỗi nhánh là Conv3×3 → BN → SiLU → Conv3×3 → BN → SiLU → Conv1×1.

Đây là biến thể theo yêu cầu thêm SimAM, được suy ra từ cấu hình artifact gốc
`hist14_focal_none_f32_s42`. [config.resolved.json](config.resolved.json) của biến
thể chỉ thay đổi ba giá trị model:

```json
{
  "local_attention": "simam",
  "local_attention_simam_lambda": 0.0001,
  "local_attention_layer_scale_init": 0.001
}
```

Residual SimAM đặt **sau 3 NeXt block của Stage 2 / C3**, trước cả Downsample 3
và lateral L3. Vị trí và toán tử lấy từ
`detector/core/models/backbones/mobilepixornext.py` và `light_attention.py`.
Core SimAM không có tham số học; wrapper có một scalar beta, khởi tạo 0.001:
`Y = X + beta * (SimAM(X) - X)`. SimAM tính thống kê FP32 riêng cho từng batch
và channel, với `d=(X-meanHW(X))²`, `v=sumHW(d)/max(HW-1,1)`,
`e=d/[4(v+lambda)]+0.5`, rồi `SimAM(X)=X*sigmoid(e)`.

Backbone gồm neck có **666,769** tham số; toàn detector có **852,930** tham số.
SimAM giữ shape C3 là `48×200×176`; đầu ra backbone vẫn `32×200×176`.
Focal Context vẫn ở C4, stage depths `[3,4,2]`, SG-FPN có fusion width 32.

[architecture_bev_backbone_heads.json](architecture_bev_backbone_heads.json)
ghi cấu hình nguồn, SHA-256, ba override, tham số Conv2d, số tham số và tensor
shape qua một forward CPU full resolution. Đây là kiểm tra kiến trúc với model
khởi tạo mới; không phải kết quả đánh giá checkpoint đã train với SimAM.

Chạy từ thư mục `Code` để tạo lại:

```bash
python tools/visualization/draw_bev_backbone_heads.py --simam
```

Mã `.tex` dùng `third_party/PlotNeuralNet/PlotNeuralNet/PyCore/TikzGen.py::ToConv`.
Upload `.tex` cùng `layers/` khi dùng Overleaf. Máy hiện tại thiếu LaTeX compiler,
nên TikZ chưa được biên dịch; PDF được xuất trực tiếp bằng Matplotlib và kiểm
tra là vector. Nhãn shape dùng `C×H×W`, bỏ batch.
