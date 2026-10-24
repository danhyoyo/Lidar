# Camera + Hist14 BEV

Ảnh: `camera_hist14_bev_5x2.png`, 2488 × 4072 pixel.

Bố cục 5 hàng × 2 cột dựa theo `Pasted image.png`: camera RGB có box bên
trái, bản đồ Hist14 nền đen và box BEV bên phải. Các frame theo thứ tự:
`007038`, `005571`, `007016`, `006761`, `002425`.

Ảnh được dựng trực tiếp bằng NumPy/Matplotlib từ dữ liệu KITTI và encoder
`hist14` hiện có. Không sử dụng ảnh do mô hình sinh ảnh tạo ra.
Camera, scan và calibration trong mỗi hàng thuộc cùng frame.
Detection lấy từ cache inference Hist14 có IQA, checkpoint epoch 130.
Không dùng ground truth. Chiều cao box camera được ước lượng từ LiDAR;
checkpoint dự đoán footprint BEV.

Tensor đầu vào có shape `[14, 800, 704]`. Để hiển thị bằng một ảnh màu,
chuẩn hóa các kênh và dùng cùng một phép chiếu PCA ba thành phần cho cả
năm frame. Occupancy (C11) xác định nền đen; kênh này có giá trị hằng trên
các ô có điểm nên không đóng góp phương sai PCA. Các kênh còn lại tham gia
phép chiếu. Ảnh RGB là biểu diễn giảm chiều, không thể hiện riêng hoặc giữ
nguyên toàn bộ thông tin của 14 kênh. Màu không phải nhãn lớp hay màu TriBand.

ROI hiển thị: X=[0,70.4], Y=[−30,30] m. X hướng sang phải, Y hướng lên.
Dùng max pooling 2×2 và gamma 0.65 chỉ khi hiển thị. Ảnh camera đầy đủ được
co giãn theo tỷ lệ panel của ảnh mẫu; box camera dùng cùng phép biến đổi.

Tái tạo từ thư mục gốc dự án:

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python Result/encoderBevviz/render_hist14.py
```

`hist14_render_metadata.json` ghi hash dữ liệu, tensor, cache detection,
tham số PCA, checkpoint và các box camera. Script kiểm tra kích thước ảnh,
giá trị hữu hạn trong tensor và hash dữ liệu nguồn sau khi xuất ảnh.

## Bản 4 × 2: camera + footprint BEV, tên Strata-BEV14

Ảnh bổ sung: `camera_bev_footprints_strata_bev14_4x2.png`, 2488 × 3286 pixel.
Giữ bốn frame đầu của bản 5 × 2. Cột trái vẫn là camera nhưng mỗi box chỉ có
bốn cạnh footprint trên mặt đường, bỏ mặt trên và cạnh đứng của cuboid.
Cột phải có tiêu đề `Encoded Strata-BEV14 Map` theo yêu cầu; tensor nguồn
vẫn dùng encoder `hist14` trong code và phép chiếu PCA của bản trước.

Tạo lại bằng:

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python Result/encoderBevviz/render_hist14_bev_4x2.py --left camera
```

Metadata đi kèm: `camera_bev_footprints_strata_bev14_4x2.json`.

## Bản 3 × 2

`camera_bev_footprints_strata_bev14_3x2.png` giữ hàng 1, 2 và 4 của bản
4 × 2; bỏ hàng 3 (frame `007016`). Thứ tự còn lại: `007038`, `005571`,
`006761`. Giữ nguyên footprint BEV, tên Strata-BEV14 và phép chiếu màu.

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python Result/encoderBevviz/render_hist14_bev_4x2.py --left camera --drop-row 3
```

Bản chỉ có ảnh camera ở cột trái: `camera_rgb_strata_bev14_3x2.png`.
Bỏ toàn bộ box và nhãn detection ở ảnh camera, tiêu đề trái là `RGB Image`;
giữ nguyên cột Strata-BEV14 và các detection BEV ở cột phải.

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python Result/encoderBevviz/render_hist14_bev_4x2.py --left camera --drop-row 3 --clean-camera
```
