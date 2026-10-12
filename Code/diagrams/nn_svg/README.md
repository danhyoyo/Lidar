# Backbone SVG (NN-SVG)

Các hình vector này được tạo từ `configs/config.json` bằng
[export_backbone_diagram.py](../../tools/visualization/export_backbone_diagram.py).
Chúng phản ánh MobilePixorNeXt + SG-FPN hiện tại, mặc định không có C4 attention.

| Hình | SVG |
|---|---|
| Backbone và FPN tổng thể | [overview.svg](overview.svg) |
| Stem | [stem.svg](stem.svg) |
| Mỗi block trong stage 2 / 3 / 4 | [stage2.svg](stage2.svg), [stage3.svg](stage3.svg), [stage4.svg](stage4.svg) |
| Bên trong MobilePixorNeXtBlock ở 48 / 96 / 128 kênh | [block2.svg](block2.svg), [block3.svg](block3.svg), [block4.svg](block4.svg) |
| Scale gate ở P4 / P3 | [gate4.svg](gate4.svg), [gate3.svg](gate3.svg) |
| Upsampling và refinement U4 / U3 | [u4.svg](u4.svg), [u3.svg](u3.svg) |
| Projection tới detection header | [output_projection.svg](output_projection.svg) |

Mở [Backbone.html](../../tools/visualization/nn_svg/Backbone.html) để chỉnh màu,
khoảng cách, kích thước chữ và xuất hình. [Hướng dẫn](../../tools/visualization/nn_svg/README.md)
có cách tạo hình cho experiment bật FocalContext/ECA/SimAM/detail/RepConv.

Kích thước nhãn là `C × H × W` suy từ config và code; B được lược bỏ.
Kích thước cuboid theo log để hình dễ đọc, không theo tỉ lệ mét.

```bash
python3 tools/visualization/export_backbone_diagram.py --svg-dir diagrams/nn_svg
```

Bản 2D dùng cho bài báo nằm ở [paper_2d/](../paper_2d/README.md), gồm PDF vector,
SVG chỉnh sửa được, PNG 600 dpi và caption LaTeX. Bản 3D ở đây được giữ riêng.
