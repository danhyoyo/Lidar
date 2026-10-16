# BEV Hist14 + lân cận + hình học — 16/10/2026

Đã tạo tensor BEV thật và visualization trên branch `feature/canupo-inspired-bev22`. Kết quả thử cho thấy thông tin lân cận bổ sung một phần cho Hist14, còn dimensionality đa tỉ lệ tạo khác biệt lớn hơn trong phép phân loại điểm theo nhãn hộp. Encoder NumPy 22 kênh hiện còn tốn khoảng 102 ms/frame trên CPU; chưa được tích hợp hoặc huấn luyện trong detector.

**Mở [viewer so sánh tương tác](../../artifacts/bev_geometry_maps_20261016/index.html)** để chọn frame, encoder và từng channel; hai map đồng bộ zoom/pan, có bật/tắt hộp GT. Viewer là file HTML local, không dùng dịch vụ ngoài.

## “Rich14” được hiểu như thế nào?

Code hiện tại có `rich8` và `hist14`, không có encoder runtime tên `rich14`. Tài liệu `paper/analysis_vi.md` ghi nhận Rich14 là tên cũ trên một số slide. Vì vậy baseline 14 kênh ở đây là **Hist14 / Rich14 theo tên cũ**. Rich8 thực tế cũng được dựng để đối chiếu, không tạo một baseline Rich14 giả định.

## Các tensor đã dựng

| Biểu diễn | Số kênh | Nội dung |
|---|---:|---|
| Rich8 | 8 | Encoder Rich8 hiện có |
| Hist14 | 14 | Encoder Hist14 hiện có, không sửa |
| Hist14 + lân cận | 18 | 14 kênh cũ + log số lân cận và cờ geometry hợp lệ ở hai scale |
| Hist14 + lân cận + hình học | 22 | 18 kênh trên + hai hệ số dimensionality ở mỗi scale |

Tensor lưu ở dạng HWC `[800,704,C]`; khi đưa vào model tương ứng `[C,800,704]`. Mười bốn kênh đầu của bản 18/22 đã được kiểm tra khớp chính xác Hist14 gốc. Bổ sung feature không làm đổi tọa độ, thêm/bớt điểm, hoặc lấp các ô trống. Chỉ các pillar vốn có điểm nhận feature.

| Index | Kênh mới | Ý nghĩa |
|---:|---|---|
| 14 | `neighbor_log_count_1m` | log(1+N) trong cửa sổ XY 11×11 ô |
| 15 | `geometry_valid_1m` | Có ít nhất 6 điểm và tổng phương sai khác 0 |
| 16 | `neighbor_log_count_2m` | log(1+N) trong cửa sổ XY 21×21 ô |
| 17 | `geometry_valid_2m` | Cờ hợp lệ ở cửa sổ lớn |
| 18–19 | `dimensionality_1d_1m`, `dimensionality_2d_1m` | Hệ số a,b của cấu trúc dạng đường/mặt ở cửa sổ nhỏ |
| 20–21 | `dimensionality_1d_2m`, `dimensionality_2d_2m` | Hệ số tương tự ở cửa sổ lớn |

Tại độ phân giải 0,1 m, “1 m / 2 m” là tên scale danh nghĩa; cửa sổ có tâm tại ô dùng 11/21 ô nên bề rộng thực là **1,1 / 2,1 m**. Lân cận được cắt tại biên ROI, giữ toàn chiều cao ROI −2,5 đến 1 m. Đây là lân cận cột theo XY, **khác quả cầu 3D quanh từng điểm của CANUPO gốc**.

Encoder tích lũy số điểm, tổng xyz và các tích tọa độ; gộp bằng summed-area tables, tính covariance 3×3 rồi lấy eigenvalues. Dùng float64 và tọa độ đã trừ một mốc giữa ROI để giảm mất chính xác số học. Với eigenvalues giảm dần λ1≥λ2≥λ3 và T=λ1+λ2+λ3:

```text
a = (λ1 − λ2) / T        # hệ số dạng đường 1D
b = 2(λ2 − λ3) / T       # hệ số dạng mặt 2D
c = 3λ3 / T = 1−a−b     # hệ số dạng khối 3D
```

Map RGB hình học dùng R=a, G=b, B=c. Giá trị này biểu diễn hình thái các điểm quan sát được, không phải xác suất Car/Pedestrian/Cyclist. Nơi thiếu điểm hoặc suy biến có a=b=c=0 và cờ valid=0; không gán mặc định thành cấu trúc 3D.

## Xem trực quan

![Cùng vùng pedestrian qua các lớp thông tin](../../artifacts/bev_geometry_maps_20261016/comparison_pedestrian.png)

Trong ví dụ pedestrian của frame 004425, kênh chiều cao thể hiện cụm điểm nổi lên; kênh lân cận thể hiện mật độ cao quanh cụm đó. Map hình học phân biệt cách các điểm phân bố quanh cụm với các dải điểm mặt đường. Ở scale lớn, cùng vùng có thể chuyển sang màu xanh dương hơn do phân bố chiếm thể tích trong cột lân cận; điều này không có nghĩa “xanh dương = người”.

Ở các vùng mặt đường xa trong ảnh toàn cảnh, cửa sổ nhỏ có thể chỉ chứa một dải quét nên thiên đỏ/1D. Cửa sổ lớn chứa thêm dải quét và có thể chuyển sang xanh lá/2D. Đây là lý do cần đọc cả scale và số lân cận; ngưỡng 6 điểm không tự bảo đảm hình học đại diện đúng bề mặt thật. Scale lớn cũng trộn đối tượng với mặt đường hoặc vật thể lân cận, làm mờ ranh giới.

| Frame | Mục đích xem | Tệp |
|---|---|---|
| 000804 | Frame validation ngẫu nhiên có cả ba lớp, gần và xa | [Toàn cảnh](../../artifacts/bev_geometry_maps_20261016/000804/overview.png), [phóng to](../../artifacts/bev_geometry_maps_20261016/000804/object_zooms.png) |
| 004425 | Pedestrian/Cyclist gần, Car xa | [Toàn cảnh](../../artifacts/bev_geometry_maps_20261016/004425/overview.png), [phóng to](../../artifacts/bev_geometry_maps_20261016/004425/object_zooms.png) |
| 004567 | Nhiều đối tượng khoảng 45–53 m, ít điểm | [Toàn cảnh](../../artifacts/bev_geometry_maps_20261016/004567/overview.png), [phóng to](../../artifacts/bev_geometry_maps_20261016/004567/object_zooms.png) |
| 001801 | Nhiều xe gần và Cyclist xa khoảng 61 m | [Toàn cảnh](../../artifacts/bev_geometry_maps_20261016/001801/overview.png), [phóng to](../../artifacts/bev_geometry_maps_20261016/001801/object_zooms.png) |

Mỗi frame có `channels_hist14.png` cho toàn bộ 14 kênh gốc, `channels_added8.png` cho tám kênh mới, `channels_rich8.png`, `bev_tensors.npz` và map score của bốn probe. Hộp vẽ trên ảnh là ground truth để định vị; GT không tham gia tạo feature.

Một RGB composite chỉ thể hiện vài kênh được chọn, không thể chứa trọn tensor 14/18/22 kênh. Rich8 occupancy và Hist14 count có cách mã hóa khác nhau nên không suy luận thông tin tốt hơn từ độ sáng của hai RGB này. Từng scalar channel có thang hiển thị cố định; giới hạn màu log-count lấy từ percentile 99,5 của mẫu fit, không lấy từ validation. Các kênh cũ giống nhau giữa ba bản là hành vi đúng: thông tin mới nằm ở channel 14–21.

## Đánh giá định lượng trên cùng dữ liệu

Dùng lại 144 frame của thử nghiệm trước: 64 fit, 16 tune thuộc train split và 64 validation. 48 frame validation chọn ngẫu nhiên với 196.608 core ngẫu nhiên được dùng cho số chính; frame/điểm bổ sung để nhìn đối tượng nhỏ không đưa vào AP chính. Nhãn yếu, FOV/ignore masks và seed giữ nguyên.

Khác thử nghiệm CANUPO trước, probe lần này **chỉ nhận đúng các kênh của tensor BEV tại pillar của điểm**, không cộng z/intensity/range riêng của từng điểm. Cùng HistGradientBoostingClassifier 100 vòng/15 lá, cùng trọng số lớp, không early stopping. Threshold chọn riêng trên tune để đạt recall điểm toàn cục 95%.

| Đầu vào probe | Point AP (%) | AUROC (%) | Recall trên validation (%) | Loại điểm ngoài hộp (%) |
|---|---:|---:|---:|---:|
| Rich8 | 65,68 | 94,70 | 96,06 | 70,33 |
| Hist14 / Rich14 cũ | 66,96 | 94,86 | 96,39 | 70,81 |
| Hist14 + lân cận | 68,81 | 95,52 | 94,85 | 77,37 |
| **Hist14 + lân cận + hình học** | **76,84** | **97,56** | **95,76** | **87,05** |

Thêm lân cận tăng **1,85 điểm phần trăm** point AP. Sau đó thêm hình học tăng tiếp **8,02 điểm**, tổng tăng **9,88 điểm** so với Hist14. Bootstrap ghép cặp theo frame, 200 lần/seed 43, cho khoảng percentile 95%:

- Hist18 − Hist14: **+0,25 đến +3,54 điểm**.
- Hist22 − Hist18: **+4,79 đến +12,05 điểm**.
- Hist22 − Hist14: **+6,67 đến +13,83 điểm**.

Trong các mẫu fit, kênh log số lân cận có tương quan tuyến tính mạnh nhất với density sẵn có, Pearson khoảng 0,73/0,64. Các hệ số hình học có tương quan mạnh nhất về trị tuyệt đối khoảng 0,28–0,43 với kênh Hist14. Kết hợp với ablation, điều này gợi ý hình học bổ sung nhiều hơn việc chỉ đếm điểm quanh pillar. Pearson thấp không chứng minh độc lập thông tin hay quan hệ nhân quả.

Trong mẫu validation xa trên 40 m, tỷ lệ geometry hợp lệ tại cửa sổ nhỏ/lớn là Car **83,54/94,43%**, Pedestrian **96,88/100%**, Cyclist **98,94/100%**. Pedestrian xa chỉ có 32 điểm mẫu và Cyclist 94 điểm, nên chưa đủ kết luận chắc về mọi đối tượng xa. Cột XY gom thêm điểm ở các độ cao khác nhau; tỷ lệ valid cao hơn không chứng minh tốt hơn quả cầu CANUPO.

Các số trên là **point AP với nhãn trong/ngoài hộp**, không phải KITTI detection AP. Điểm mặt đường và đối tượng có thể cùng pillar nhưng có nhãn điểm khác nhau; probe này không tương đương supervision heatmap của detector. Nó cũng không có receptive field CNN. Những điều này giới hạn cách suy ra lợi ích khi huấn luyện MobilePixorNeXt. Không so trực tiếp 66,96/76,84 ở đây với 69,62/77,62 của protocol feature khác trong lần trước.

Đã sửa và tính lại toàn bộ probe sau khi phát hiện float32 có thể gán sai pillar cho một số điểm sát biên ô. Lookup Hist14 hiện dùng float64/floor đúng như rasterizer; Rich8 dùng chính float32 floor-division của encoder gốc. Regression test chứa các điểm biên từng bị lấy nhầm vào ô trống. Tất cả core hiện khớp ô có điểm của encoder tương ứng. Các tensor/ảnh Hist14 ban đầu không bị lỗi này; lỗi nằm ở cách lấy feature của core để đánh giá. Số trong bảng là lần tính lại, thay cho số tạm thời trước sửa.

## Chi phí và khuyến nghị

Đo một lần/frame trong 144 frame trên CPU Ryzen 5 7640HS, NumPy, OPENBLAS_NUM_THREADS=1; các số p50/p95 dưới đây là phân vị giữa các frame, không phải benchmark lặp trên một frame:

| Bước | p50 | p95 |
|---|---:|---:|
| Rich8 | 7,93 ms | 9,98 ms |
| Hist14 | 7,83 ms | 9,41 ms |
| Thêm lân cận và hình học | 92,40 ms | 108,50 ms |
| Đường tính tensor 22 kênh | **101,71 ms** | **117,15 ms** |

Chưa đo đường 18 kênh tối giản riêng: công cụ hiện tại tạo đồng thời các tensor và tính cả covariance/eigenvalues. Thời gian trên không gồm đọc file, tạo hình, transfer GPU hoặc chạy detector; không gọi đây là latency inference toàn pipeline.

Dense input FP32 của Rich8/Hist14/Hist18/Hist22 lần lượt khoảng **17,19 / 30,08 / 38,67 / 47,27 MiB/frame**. Tăng từ 14 lên 22 kênh làm bộ nhớ input tăng 57,1%; các bản NPZ nén không phản ánh bộ nhớ tensor lúc chạy.

**Nên giữ hướng 22 kênh để làm ablation detector**, vì phép thử tensor thực vẫn có lợi ích sau control density-only. Trước khi dùng trong pipeline nhanh, cần tối ưu/batched GPU hoặc giảm chi phí tính geometry; prototype NumPy hiện chưa phù hợp với mục tiêu realtime khi cộng thêm model. Phải huấn luyện lại với đúng encoder, tính feature sau augmentation làm đổi điểm/lân cận và giữ các điều kiện khác cố định. Ưu tiên theo dõi AP từng lớp, độ xa, lỗi vertical và latency tổng. Chưa có lý do lọc cứng point cloud dựa trên các map này.

## Tái lập và kiểm tra

- [Encoder thử nghiệm](../tools/analysis/bev_geometry_maps.py), [script phân tích/xuất hình](../tools/visualization/compare_bev_geometry_maps.py), [tests](../tests/test_bev_geometry_maps.py).
- [Metrics](../../artifacts/bev_geometry_maps_20261016/metrics.json), [thống kê kênh](../../artifacts/bev_geometry_maps_20261016/feature_statistics.json), [runtime từng frame](../../artifacts/bev_geometry_maps_20261016/encoding_timing.json), [coverage](../../artifacts/bev_geometry_maps_20261016/geometry_coverage.json).
- [Protocol và hash](../../artifacts/bev_geometry_maps_20261016/manifest.json). Detector/config hiện tại chưa thay đổi.
- [Kết quả kiểm chứng](../../artifacts/bev_geometry_maps_20261016/verification.json): 13 tests qua, tính lại AP/AUROC từ prediction đã lưu, kiểm tra 615.834 core được lấy đúng ô, tensor giữ nguyên Hist14 và occupancy/geometry masks khớp.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/canupo-matplotlib \
/home/duyennh/miniconda3/envs/AI_env/bin/python \
  Code/tools/visualization/compare_bev_geometry_maps.py \
  --output artifacts/bev_geometry_maps_20261016
```

Thêm `--render-only` để dùng lại feature đã lưu và xuất lại hình; công cụ refit các probe, kiểm tra AP khớp bản lưu và giữ lại bootstrap đã kiểm chứng. Kiểm tra geometry gồm đối chiếu cửa sổ với covariance tính trực tiếp, đường/mặt/khối, điểm rỗng/NaN/thiếu điểm, bất biến khi tịnh tiến cả điểm và ROI, và giữ nguyên Hist14. Đã kiểm tra cú pháp JavaScript và chạy DOM-mock smoke test cho mọi lựa chọn frame/variant/channel, GT toggle, zoom/pan đồng bộ và reset; chưa chạy bằng browser engine thật.
