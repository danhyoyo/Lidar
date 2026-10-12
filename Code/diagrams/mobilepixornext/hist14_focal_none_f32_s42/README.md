# MobilePixorNeXt — hist14 / Focal Context / SG-FPN f32

Hình tổng thể **BEV → backbone → heads** theo bố cục `other_paper.png`:
[PNG](architecture_bev_backbone_heads.png),
[SVG](architecture_bev_backbone_heads.svg),
[PDF](architecture_bev_backbone_heads.pdf),
[PlotNeuralNet/TikZ](architecture_bev_backbone_heads.tex).
Encoder gồm các block đứng, ba hàng stage đi xuống, SG-FPN đi lên và inset NeXt
block. Hai panel head riêng cho Car (K=1) và Pedestrian/Cyclist (K=2), mỗi group
có đủ 5 nhánh độc lập. Bản thêm SimAM tại C3 nằm ở
[hist14_focal_simam_f32_s42](../hist14_focal_simam_f32_s42/README.md).
Tạo lại hình tổng thể từ thư mục `Code` bằng:

```bash
python tools/visualization/draw_bev_backbone_heads.py
```

Bộ 17 sơ đồ theo phong cách hình mẫu trong thư mục cha: màu pastel, font serif,
khung nét đứt, khối tensor và đường residual. Mở [index.html](index.html) để xem
toàn bộ hình và tải từng định dạng. Mỗi hình có SVG chỉnh sửa được, PNG 300 DPI,
PDF vector và mã PlotNeuralNet/TikZ `.tex`.

Nguồn cấu hình là `config.resolved.json` trong artifact
`mobilepixornext-hybrid_gt_aug-oga_loss-hist14-iqa-sgfpn-ctx_focal_local_none_f32_detail0_e45d83922c52-grouped_e02c0e499671-s42-P4-h32`.
Bản sao [config.resolved.json](config.resolved.json) giữ nguyên nội dung nguồn;
[architecture.json](architecture.json) ghi SHA-256, kích thước tensor, số tham số,
thông số từng Conv2d và kết quả kiểm tra forward.

| Hình | Nội dung |
| --- | --- |
| [overview.svg](overview.svg) | Tổng quan backbone, neck, template head và hai inset |
| [backbone.svg](backbone.svg) | Backbone gồm encoder và SG-FPN |
| [stem.svg](stem.svg) | Hai Conv3×3 thường, 14→32→32 |
| [stage2.svg](stage2.svg) | Downsample 32→48 và 3 NeXt block |
| [stage3.svg](stage3.svg) | Downsample 48→96, 4 NeXt block và Focal Context |
| [stage4.svg](stage4.svg) | Downsample 96→128 và 2 NeXt block |
| [next_block_c48.svg](next_block_c48.svg) | NeXt block 48→120→48 |
| [next_block_c96.svg](next_block_c96.svg) | NeXt block 96→240→96 |
| [next_block_c128.svg](next_block_c128.svg) | NeXt block 128→320→128 |
| [focal_context.svg](focal_context.svg) | BN, projection 96→132, split Q/S/G=64/64/4, context, modulation, residual |
| [sg_fpn.svg](sg_fpn.svg) | Ba lateral projection và hai bước fusion |
| [gate_p4.svg](gate_p4.svg) | Gate P4, 48 kênh |
| [gate_p3.svg](gate_p3.svg) | Gate P3, 32 kênh |
| [upsample_u4.svg](upsample_u4.svg) | Bilinear ×2, DWConv3×3 48→48, BN, SiLU |
| [upsample_u3.svg](upsample_u3.svg) | Bilinear ×2, Conv3×3 thường 48→32, BN, SiLU |
| [output_projection.svg](output_projection.svg) | Conv3×3 32→32, BN, SiLU |
| [grouped_heads.svg](grouped_heads.svg) | Hai Header độc lập: Car và Pedestrian/Cyclist, mỗi nhóm có 5 nhánh |

Kích thước theo `C × H × W`, bỏ batch:
`14×800×704 → 32×400×352 → 48×200×176 → 96×100×88 → 128×50×44`.
Đầu ra backbone là `32×200×176`, stride 4. Stage depths là `[3,4,2]`,
expansion 2.5, Focal bottleneck 64 và dilation `[1,2,3]`.

Backbone gồm neck có **666,768** tham số. Hai head có 93,064 và 93,097 tham số;
toàn detector có **852,929** tham số. Các số này được đếm từ model khởi tạo
theo cấu hình, không phải kết quả benchmark tốc độ hoặc độ chính xác.

Luồng Focal local là **tuần tự**. Global mean lấy từ mức local cuối Z3.
Softmax có 4 trọng số theo từng pixel, trộn 3 mức local và 1 global; context
được chiếu 64→64, nhân với Q, chiếu 64→96, BN và LayerScale trước residual.
SG-FPN dùng `P = U + 2 sigmoid(DWConv(L+U)) ⊙ L`; gate zero-init cho multiplier
ban đầu bằng 1. Các lateral Conv1×1 không có BN/activation. U4 dùng depthwise,
U3 dùng convolution thường. Không có local attention, RepConv hoặc detail path.

`overview` dùng template head chung với chú thích hai nhóm; `grouped_heads`
vẽ riêng từng nhóm và số kênh output thực tế. Mọi output head là raw prediction,
với `box_mode=bev`. Augmentation và OGA loss thuộc phần huấn luyện, không phải
block của backbone nên không được đưa vào luồng feature inference.

## Tạo lại

Chạy từ thư mục `Code`, với Python có PyTorch và Matplotlib:

```bash
python tools/visualization/draw_artifact_architecture.py \
  --config diagrams/mobilepixornext/hist14_focal_none_f32_s42/config.resolved.json \
  --output-dir diagrams/mobilepixornext/hist14_focal_none_f32_s42 \
  --dpi 300
```

Trên máy hiện tại có thể dùng `/home/duyennh/miniconda3/envs/AI_env/bin/python`.
Generator nhắm vào preset này và từ chối cấu hình khác các giá trị ghi trên.
Generator chạy backbone và toàn detector trên CPU ở full resolution, kiểm tra
tensor hữu hạn, shape head, cấu trúc block, gate initialization và số tham số.
Renderer kiểm tra text không chồng nhau hoặc nằm ngoài canvas trước khi xuất.

Mã `.tex` gọi trực tiếp `third_party/PlotNeuralNet/PlotNeuralNet/PyCore/TikzGen.py::ToConv`
cho khối tensor. `layers/` chứa tài nguyên gốc và giấy phép MIT. Khi dùng
Overleaf, tải `.tex` cùng `layers/`. Máy hiện tại không có LaTeX compiler nên
**TikZ chưa được biên dịch**; PDF được xuất trực tiếp bằng Matplotlib và đã kiểm
tra text/vector. Kích thước cuboid mang tính minh họa; nhãn tensor mới là kích
thước dữ liệu thực tế.
