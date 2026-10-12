# NN-SVG backbone editor

Mở [Backbone.html](Backbone.html) trong trình duyệt. Bản mở rộng từ
[alexlenail/NN-SVG](https://github.com/alexlenail/NN-SVG) hoạt động offline, không
cần npm, GPU, server hoặc PyTorch. Nguồn và commit upstream ở [UPSTREAM.md](UPSTREAM.md);
giữ giấy phép MIT. Các tab FCNN/LeNet/AlexNet gốc vẫn dùng CDN của upstream.

## Điều khiển editor

Bảng bên trái có bốn tab **Kiến trúc**, **Block**, **Nhóm**, **Lưu**. Tab có
thể chuyển bằng phím mũi tên khi đang focus. Trên điện thoại, canvas hiện trước;
**Điều khiển** mở bảng, **✕** hoặc Escape đóng bảng. Dùng **＋ / −** dưới canvas
để zoom bằng cảm ứng hoặc bàn phím; **Vừa khung** đặt lại toàn hình.

Trong form, **Chưa áp dụng** báo phần đang sửa. **Áp dụng** lưu; **Hủy sửa** bỏ
phần sửa chưa áp dụng. Khi chuyển block, view, kiểu hình hoặc thực hiện thao tác
thêm/xóa/xuất, các form đang sửa được kiểm tra và áp dụng cùng nhau. Giá trị sai
giữ nguyên để sửa, không ghi dở các form khác. Chuyển tab hoặc đóng bảng trên
điện thoại giữ phần đang nhập. Lưu JSON trước khi đóng/reload trang để giữ các
thay đổi; localStorage giữ tài liệu đã áp dụng.

Các slider 3D có ô nhập số, đơn vị, phím mũi tên và **Đặt lại trình bày 3D**.
Chỉnh slider giữ pan/zoom và các trường chưa áp dụng. Cỡ chữ 2D có slider riêng
trong thuộc tính block và được lưu bằng **Áp dụng block**.

## Kiểu hình theo mẫu và custom block

Mặc định **2D · màu theo hình mẫu**: encoder dọc, decoder đi lên, pastel, khung
nhóm nét đứt và vòng tròn cho phép toán. Bố cục tham khảo
`diagrams/other_paper.png` và `other_paper_backbone.png`; model vẫn dùng bilinear
resize + refinement và scale gate đúng theo config. Xem [hình xuất sẵn](../../../diagrams/reference_2d/README.md).

- Chọn block: sửa nhãn, phụ đề, C/H/W, màu, cỡ chữ, vị trí, kích thước, hướng chữ,
  hiển thị shape/kênh vào→ra, kernel/stride/repeat/groups, rồi **Áp dụng block**.
- **Chọn vùng**: giữ chuột trái trên nền rồi kéo một hình chữ nhật để chọn nhiều
  block. Kéo theo hướng nào cũng được; block giao với vùng sẽ được chọn. Có viền
  sáng và số block đang chọn. Trong chế độ xem thường, **Shift/Ctrl/Cmd + kéo nền**
  cũng chọn vùng; giữ phím này khi kéo thêm vùng để cộng vào lựa chọn hiện tại.
  **Shift/Ctrl/Cmd + nhấp block** thêm/bỏ riêng block đó; **Ctrl/Cmd+A** chọn tất cả
  trong view khi không nhập form. **Escape** bỏ chọn hoặc hủy thao tác đang kéo.
- Kéo một block trong cụm đang chọn để **di chuyển cả cụm**, giữ khoảng cách giữa
  các block; giữ **Shift** để căn theo lưới 10 đơn vị. Tab **Block** có **Tạo nhóm
  từ vùng chọn**, **Xóa các block đã chọn** và **Bỏ chọn tất cả**. Phím **Delete**
  cũng xóa cụm; các đường nối và thành viên nhóm liên quan được cập nhật cùng
  nhau. Mỗi lần di chuyển/xóa/tạo nhóm là một bước Undo. Bản vẽ cần giữ ít nhất
  một block. Vùng chọn không lưu vào tài liệu JSON hoặc SVG xuất ra.
- **Kéo / resize**: kéo thân block; kéo góc dưới bên phải để đổi kích thước.
  Giữ Shift khi kéo để căn vị trí theo lưới 10 đơn vị.
- Bấm **＋ Block**, chọn loại trong tab **Block** → **＋ Thêm**.
  Block mới/nhân bản tìm vị trí trống và được chọn để sửa ngay. Có thể nhân bản/xóa
  block, bắt đầu từ một bản vẽ mới, hoặc sửa backbone đã nạp.
- **Nối block**: nhấp block nguồn rồi block đích; banner hiển thị bước đang làm.
  Hoặc dùng **Thêm kết nối bằng danh sách** trong tab Block, chọn nguồn/đích
  rồi **Tạo kết nối** bằng bàn phím hoặc cảm ứng. Chọn đường nối để sửa nguồn,
  đích, nhãn, nét liền/đứt, cổng ra/vào và một điểm rẽ X/Y tùy chọn.
- Tab **Nhóm**: thêm nhóm từ các block đang chọn, đặt tên/màu và tích checkbox
  cho các block trong nhóm; không cần Ctrl/Cmd. Có số block, Chọn tất cả/Bỏ chọn
  và hướng dẫn khi nhóm còn trống.
  Khung tự theo vị trí và kích thước của các block. Xóa nhóm giữ các block.
- **↶ / ↷** hoặc Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z: Undo/Redo.
- **Lưu → Lưu JSON chỉnh sửa**: lưu toàn bộ kiến trúc, màu, kích
  thước, vị trí, đường nối và nhóm. Dùng **Nạp kiến trúc JSON** để mở lại.
- **Xuất SVG** tải toàn bộ hình đang xem, độc lập với pan/zoom, không kèm
  highlight, vùng chọn hoặc resize handle. **Vừa khung** đưa canvas về toàn hình.

Khi **Chọn vùng** đang bật, kéo chuột giữa để pan; hoặc tắt chế độ này và kéo
nền bằng chuột trái. Cuộn chuột hoặc **＋ / −** vẫn zoom. Trên điện thoại, bật
**Chọn vùng**, kéo ngón tay từ nền để chọn, rồi kéo block trong cụm để di chuyển;
**Thuộc tính** mở các thao tác cho cụm. Chọn nhiều block áp dụng cho hai kiểu 2D
chỉnh sửa. Đổi view sẽ bỏ vùng chọn.

Trình duyệt lưu bản nháp local nếu hỗ trợ localStorage; mở lại cùng URL để tiếp
tục. Dùng JSON khi chuyển máy/trình duyệt/URL. **Khôi phục backbone từ config**
trở về preset nguồn; thao tác này có thể Undo trong phiên hiện tại.

Các kiểu khác: **2D · trắng đen** dùng cùng dữ liệu và bố cục chỉnh sửa;
**2D · ba panel cũ** giữ hình composite từ config; **Tensor 3D** giữ các thanh
khoảng cách, chữ, chiều sâu, màu và nền. Kéo/resize và cổng nối tùy chỉnh áp dụng
cho hai kiểu 2D chỉnh sửa. Nhấp đúp block hoặc **Mở sơ đồ chi tiết** để xem từng
stage, block, gate, upsampling và module tùy chọn. Mỗi view chỉnh độc lập.

## Bản vẽ và model

Preset từ `configs/config.json` có BEV rich8, stage depths 2/4/2 và output
`16 × 200 × 176` (C/H/W, batch omitted). Source graph mô tả:

- Stem hai Conv 3×3 (stride 2 rồi 1), mỗi Conv có BN + SiLU.
- Stage 2/3/4 ở stride 4/8/16; mỗi stage gồm downsample + từng NeXt block.
- Block: DW 7×7, BN, PW expand, SiLU, PW project, BN, LayerScale, residual.
  Hidden channels dùng `int(round(C * expansion))`.
- FPN: các lateral 1×1, bilinear resize ×2 (`align_corners=False`), refinement
  và `P = U + 2σ(DW(L + U)) ⊙ L`. Tắt gate thì cộng L/U trực tiếp.
- FocalContext, residual ECA/SimAM, stem detail, RepConv training/deploy xuất
  đúng các view khi config bật. Detection heads nằm ngoài phạm vi preset.

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
