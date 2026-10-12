# PaperNN Studio · Publication Neural Network Studio

Mở [Backbone.html](Backbone.html) trong trình duyệt để thiết kế sơ đồ mạng nơ-ron chuyên dụng cho bài báo khoa học (CVPR, ICCV, ECCV, NeurIPS, IEEE Trans).
Hoạt động **hoàn toàn offline**, không cần server, npm, GPU, hoặc PyTorch.

Preset mặc định được cập nhật ngày 2026-10-12 thành **P16 — Ours / Proposed
(P4 + residual SimAM, λ = 0.0001)**, lấy từ `config.resolved.json` của run
`P16-h32` trong archive kết quả. Bản sao nguyên vẹn của config nằm ở
[configs/p16.config.resolved.json](configs/p16.config.resolved.json); đường dẫn
archive, tên run và SHA256 config nằm trong `provenance` của JSON sơ đồ.

Mở `Backbone.html`, chọn **Khôi phục P16 · output 32 / fusion 32** hoặc nạp
[mobilepixornext.json](mobilepixornext.json) để dùng P16. Bản nháp được lưu riêng
theo SHA256 config của preset: cập nhật từ preset 24/16 cũ sang P16 sẽ mở P16
32/32, còn các chỉnh sửa đã lưu với P16 tiếp tục được khôi phục khi mở lại.
Trong tab **Lưu**, nút **Tải bản nháp trước cập nhật P16** cho phép lấy lại bản
nháp cũ nếu trình duyệt còn giữ nó. Thao tác khôi phục có thể Undo; lưu JSON trước nếu cần
giữ bản vẽ đang chỉnh. Để vẽ đối chứng P4, nạp [presets/p4.json](presets/p4.json).
Preset rich8 trước khi cập nhật được lưu ở
[presets/legacy_rich8.json](presets/legacy_rich8.json).

Bố cục giữ nguyên mẫu bottom-up/top-down ban đầu: encoder đi xuống ở bên
trái, lateral ở giữa, FPN đi lên ở bên phải. Các khối có sẵn giữ nguyên vị
trí, kích thước và màu; dữ liệu cấu hình được cập nhật theo P16. Tên nội bộ
`MobilePixorNeXt` được giữ để renderer tiếp tục dùng bố cục gốc. Bản nháp P16
với bố cục ngang từ lần cập nhật trước không tự nạp vào bố cục đã khôi phục.

## Triết lý thiết kế: Hộp LEGO Deep Learning nguyên tử (Atomic Primitives)

Thay vì giới hạn người dùng trong các khuôn mẫu hình chữ nhật đồng nhất hoặc các template dựng sẵn cố định, **PaperNN Studio** cung cấp toàn bộ các viên gạch nguyên tử cấu tạo nên các mạng nơ-ron hiện đại. Mỗi nhóm nguyên tử sở hữu **hình học trực quan đặc thù (Visual Semantics)** theo chuẩn quy ước của các hội nghị AI hàng đầu:

| Nhóm thành phần | Khối nguyên tử tiêu biểu | Hình dạng hình học (SVG Geometry) | Ý nghĩa trực quan |
| :--- | :--- | :--- | :--- |
| **Tensors & Data** | Input, Output, Feature, Embedding | Thẻ chữ nhật bo góc (`standard`) | Thể hiện dữ liệu đầu vào / đầu ra / không gian đặc trưng |
| **Convolutions & Linear** | Conv 2D, Conv 1D, Depthwise (DW), Pointwise (1×1), Linear / Dense, Deconv | Thẻ chữ nhật tiêu chuẩn (`standard`) | Biến đổi tích chập hoặc phép chiếu ma trận trọng số |
| **Normalization** | BatchNorm, LayerNorm, RMSNorm, GroupNorm | Thanh dọc siêu mảnh (`node-slim`, 44×76 px) | Chuẩn hóa kênh/layer; thanh hẹp giúp sơ đồ gọn gàng, tinh tế |
| **Activations** | ReLU, GELU, SiLU / Swish, Sigmoid, Tanh | Thanh dọc bo tròn (`node-slim`, 44×76 px) | Hàm phi tuyến nối tiếp sau các tầng tính toán |
| **Pooling & Scaling** | MaxPool 2×2, AvgPool 2×2, Upsample ↑2, PixelShuffle | Hình thang đa giác (`node-trapezoid`, `<polygon>`) | Thu hẹp kích thước (trapezoid $\blacktriangleright$) hoặc mở rộng phân giải ($\blacktriangleleft$) |
| **Math Operations** | Add (+), Multiply (⊙), Concat (©), Split (⑂), Matmul (⊗), LayerScale (γ), Softmax (σ), Gate (G) | Vòng tròn ký hiệu toán học (`circle`, `<ellipse>`) | Các phép cộng residual, nhân phần tử, ghép nối, tích ma trận |
| **Attention & Transformers** | MHSA, QKV Proj, FFN / MLP, RoPE, MoE Router | Thẻ khối chức năng cao cấp (`standard`) | Các module phức hợp trong kiến trúc nền tảng hiện đại |
| **Tầng lặp lại (Repeated Layers)** | Conv ×N, Transformer Block ×L | Đổ bóng xếp lớp viền ngoài (`stacked-shadow`) | Trực quan hóa số lượng tầng lặp lại (`repeat > 1`) |

## Tính năng nổi bật cho bài báo khoa học

1. **Thư viện Accordion Drawer đa năng**:
   - Tab **Block** phân loại trực quan thành 7 danh mục: Tensors, Convolutions & Linear, Normalization, Activations, Pooling, Math Ops, Attention.
   - Nhấn 1-click vào bất kỳ thành phần nào để tự động thêm vào canvas với kích thước và tỷ lệ tối ưu cho hình dạng đó.

2. **Tự do lắp ghép mọi cấu trúc kiến trúc (Custom Topology)**:
   - Tự do vẽ và kết nối mọi mạng: CNN (ResNet, ConvNeXt, MobileNet), Transformers (ViT, Swin, DiT), U-Net (FPN, Skip connections), LLM blocks (RMSNorm + RoPE + QKV + Gated MLP), MoE (Mixture of Experts).
   - Nối dây trực quan bằng chế độ **Nối dây (Connect)** hoặc form kết nối với cổng ra/vào tùy chọn (Trên, Dưới, Trái, Phải, Tự động).

3. **Mẫu kiến trúc tham khảo sẵn có (Presets)**:
   - **ResNet Residual Block (CVPR 2016)**: Khối tích chập residual kinh điển.
   - **Vision Transformer Layer (ICLR 2021)**: Khối Encoder Transformer gồm LayerNorm + MHSA + FFN + Residual.
   - **MobilePixorNeXt (LiDAR BEV)**: Backbone LiDAR xe tự hành với Scale-Gated FPN.

4. **Bộ công cụ Căn gióng & Dàn trang (Alignment & Distribution)**:
   - Khi chọn từ 2 block trở lên: Căn lề Trái, Giữa ngang, Phải, Trên, Giữa dọc, Dưới.
   - Khi chọn từ 3 block trở lên: Tự động dãn đều khoảng cách theo chiều ngang hoặc dọc.

5. **Bảng chú thích màu tự động (Publication Legend)**:
   - Tích chọn **"Hiện bảng chú thích màu (Legend)"** để sinh tự động bảng chú thích phản ánh đúng màu sắc và hình dạng hình học (thanh mảnh, hình thang, vòng tròn, thẻ).

6. **Đường gióng khổ bài báo & Kích thước in thực tế**:
   - Hỗ trợ chuẩn IEEE/CVPR: **Khổ 1 cột (85 mm)** và **Khổ 2 cột (182 mm)**.
   - Phản hồi trực tiếp kích thước in theo milimet trên màn hình.

7. **Xuất ảnh chất lượng cao trực tiếp trên trình duyệt**:
   - **↓ Xuất SVG**: Vector gốc độ nét vô hạn, mở và tinh chỉnh trong Inkscape hoặc Illustrator.
   - **↓ PNG (600 DPI)**: Raster hóa canvas siêu nét ở độ phân giải 600 DPI, sẵn sàng nộp bài báo.
   - **↓ Lưu JSON chỉnh sửa**: Lưu toàn bộ kiến trúc để chỉnh sửa hoặc chia sẻ.


Khi **Chọn vùng** đang bật, kéo chuột giữa để pan; hoặc tắt chế độ này và kéo
nền bằng chuột trái. Cuộn chuột hoặc **＋ / −** vẫn zoom. Trên điện thoại, bật
**Chọn vùng**, kéo ngón tay từ nền để chọn, rồi kéo block trong cụm để di chuyển;
**Thuộc tính** mở các thao tác cho cụm. Chọn nhiều block áp dụng cho hai kiểu 2D
chỉnh sửa. Đổi view sẽ bỏ vùng chọn.

Trình duyệt lưu bản nháp local nếu hỗ trợ localStorage; mở lại cùng URL và cùng
preset nguồn để tiếp tục. Dùng JSON khi chuyển máy/trình duyệt/URL. **Khôi phục backbone từ config**
trở về preset nguồn; thao tác này có thể Undo trong phiên hiện tại.

Các kiểu khác: **2D · trắng đen** dùng cùng dữ liệu và bố cục chỉnh sửa;
**2D · ba panel cũ** giữ hình composite từ config; **Tensor 3D** giữ các thanh
khoảng cách, chữ, chiều sâu, màu và nền. Kéo/resize và cổng nối tùy chỉnh áp dụng
cho hai kiểu 2D chỉnh sửa. Nhấp đúp block hoặc **Mở sơ đồ chi tiết** để xem từng
stage, block, gate, upsampling và module tùy chọn. Mỗi view chỉnh độc lập.

## Bản vẽ và model

Preset P16 từ `configs/p16.config.resolved.json` có BEV hist14
`14 × 800 × 704`, stage depths 3/4/2, expansion 2.5, SG-FPN fusion32 và output
`32 × 200 × 176` (C/H/W, batch omitted). Source graph mô tả:

- Stem hai Conv 3×3 (stride 2 rồi 1), mỗi Conv có BN + SiLU.
- Stage 2/3/4 ở stride 4/8/16; mỗi stage gồm downsample + từng NeXt block.
  Channels lần lượt là 48/96/128, stem có 32 channels.
- Residual SimAM sau Stage 2 ở stride 4: `X + β × (SimAM(X) − X)`,
  λ = 0.0001, β khởi tạo 0.001. Feature sau SimAM cấp cả Stage 3 và lateral L3.
- FocalContext sau Stage 3 ở stride 8: bottleneck64, dilation 1/2/3,
  residual scale khởi tạo 0.001; cấp cả Stage 4 và lateral L4.
- Block: DW 7×7, BN, PW expand, SiLU, PW project, BN, LayerScale, residual.
  Hidden channels dùng `int(round(C * expansion))`.
- FPN: các lateral 1×1, bilinear resize ×2 (`align_corners=False`), refinement
  và `P = U + 2σ(DW(L + U)) ⊙ L`. Tắt gate thì cộng L/U trực tiếp.
- L5/L4 có 48 channels; L3/U3/P3 có 32 channels. Output projection 3×3
  giữ 32 channels. Detail path và reparameterization đều tắt.
- FocalContext, residual ECA/SimAM, stem detail, RepConv training/deploy xuất
  đúng các view khi config bật. Detection heads nằm ngoài phạm vi preset.

P4 có cùng backbone/FocalContext/fusion/output và không có local attention;
config gốc nằm ở [configs/p4.config.resolved.json](configs/p4.config.resolved.json).
Grouped Car/Pedestrian-Cyclist heads, IQA và OGA được ghi trong runtime config;
heads và training loss không thuộc sơ đồ backbone. Không dùng preset
`hist14_local3_focal_simam_grouped_oga_iqa.json` trong `configs/experiments` để
thay thế config P16 này: preset thí nghiệm đó dùng fusion24, còn run P16 dùng
fusion32.

SVG và JSON có bố cục chỉnh sửa của P16 nằm ở
[exports/p16/backbone.svg](exports/p16/backbone.svg) và
[exports/p16/backbone.editor.json](exports/p16/backbone.editor.json).
Tái tạo chúng từ preset bằng renderer của Studio (chạy từ thư mục `Code`):

```bash
node tools/visualization/papernn_studio/export-reference.cjs \
  tools/visualization/papernn_studio/mobilepixornext.json \
  tools/visualization/papernn_studio/exports/p16
```

Custom sửa dữ liệu bản vẽ; shape là số người dùng nhập. Renderer không suy shape
các block sau, không dựng hay huấn luyện một mạng mới từ bản vẽ. Chỉnh nội dung
đánh dấu `manual drawing` thay cho một tuyên bố shape đã được model xác minh.
Sửa vị trí/màu hoặc khung nhóm giữ topology/shape nguồn. Nhãn Cin→Cout của block
bật tùy chọn này đọc C của block nguồn và block đích; repeat đọc giá trị hiện tại.

## Xuất SVG/PDF/PNG và tái tạo

Cần Node.js và `rsvg-convert` (librsvg) cho PDF/PNG; SVG giữ chữ và hình vector.

```bash
# Bố cục theo hai hình mẫu + source JSON chỉnh sửa được
python3 tools/visualization/export_reference_figure.py

# Từ JSON tải xuống sau khi bạn custom trên giao diện
python3 tools/visualization/export_reference_figure.py \
  --document backbone-custom.json --output-dir artifacts/diagrams/custom

# Config của một run/experiment, giữ preset mặc định
python3 tools/visualization/export_reference_figure.py \
  --config path/to/config.resolved.json --output-dir artifacts/diagrams/run_name
```

Các file: `backbone.svg`, `backbone.pdf`, `backbone.png` (600 dpi),
`backbone.editor.json`, `provenance.json` và SVG chi tiết trong `details/`.
PDF/SVG không phụ thuộc `--dpi`; font PDF được nhúng bởi librsvg, font thực tế
phụ thuộc máy xuất. Thu nhỏ hình theo template bài báo cũng làm chữ nhỏ theo.

Lệnh cũ vẫn dùng được:

```bash
# Tạo lại preset config và các cuboid 3D
python3 tools/visualization/export_backbone_diagram.py --svg-dir diagrams/nn_svg

# SVG theo bố cục mẫu hoặc trắng đen (không cần librsvg)
python3 tools/visualization/export_backbone_diagram.py \
  --output-dir artifacts/nn_svg/preset --svg-dir artifacts/nn_svg/reference --svg-style reference

# Composite ba panel của bản 2D trước
python3 tools/visualization/export_paper_figure.py --details
```

`--svg-style` hỗ trợ `3d`, `paper` (composite cũ), `reference`, `mono`.
Node trực tiếp: `node tools/visualization/nn_svg/export-reference.cjs document.json out`.

## Dữ liệu chỉnh sửa

JSON giữ `schema_version: 1` và `views` của adapter cũ. Mỗi node có `layout`:
`x`, `y` (tâm), `width`, `height`, `fill`, `stroke`, `font_size`, `rotation`
(0/±90), `show_shape`. `caption` là nhãn vẽ, `subtitle` là các dòng mô tả;
`repeat_label` đặt tên block lặp, `channel_projection` bật nhãn Cin→Cout.
Mỗi edge có `id`, `source_port`, `target_port`, `waypoints` (tùy chọn).
`graph.groups` chứa `id`, `label`, `node_ids`, `fill`, `stroke`, `padding`.

Layout của MobilePixorNeXt là preset có thể sửa. `Reference.js` đọc graph và
layout, không tra ID như `stage2`/`p3` để gắn tọa độ cố định khi render. JSON cũ
chưa có layout được khởi tạo bố cục; JSON đã chỉnh giữ vị trí và các nhánh lưu.
Mọi import/command được kiểm tra endpoint, shape, geometry, color và group.

Adapter config dùng resolver BEV/backbone chung, hỗ trợ MobilePixorNeXt với
`neck_type=scale_gated_fpn`/`sgfpn` và grid chia hết cho 16. Không trace một model
Python bất kỳ; thay topology model cần cập nhật adapter. Pillar encoding chỉ
xuất BEV input, không mở point encoder. Có thể audit shape qua model thật với
`python3 tools/visualization/export_backbone_diagram.py --verify-model` (cần PyTorch,
meta-device forward, không cấp phát feature BEV lớn).

## Kiểm tra

```bash
python3 -m pytest tests/test_backbone_diagram.py tests/test_paper_figure.py tests/test_reference_figure.py -q
node tools/visualization/nn_svg/tests/renderer.test.cjs
node tools/visualization/nn_svg/tests/paper.test.cjs
node tools/visualization/nn_svg/tests/editor.test.cjs
```

Browser tests cần `playwright-core` và Chromium của Playwright:

```bash
node tools/visualization/nn_svg/tests/browser-smoke.cjs
node tools/visualization/nn_svg/tests/editor-browser.cjs
node tools/visualization/nn_svg/tests/ux-browser.cjs
node tools/visualization/nn_svg/tests/selection-browser.cjs
```
