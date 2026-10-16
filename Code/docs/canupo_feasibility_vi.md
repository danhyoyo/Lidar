# Thử nghiệm CANUPO trên KITTI có sẵn — 16/10/2026

**Cập nhật khi kiểm tra visualization:** bước lấy Hist14 tại core trong phiên thử ban đầu tính chỉ số ô bằng float32, trong khi rasterizer Hist14 dùng float64. Một số điểm sát biên có thể lấy feature từ ô kế bên. Các số probe có Hist14 dưới đây được giữ cùng snapshot gốc để truy vết và không xem là số chuẩn sau sửa lỗi; descriptor/point classifier CANUPO gốc không bị lỗi này. [Phép so sánh BEV mới](bev_geometry_visualization_vi.md) đã quy điểm về ô đúng theo từng encoder và tính lại các probe. Script phân tích cũng đã sửa và từ chối cache cũ khi chạy lại.

**Kết quả:** đặc trưng hình học CANUPO bổ sung thông tin hữu ích cho phép phân loại điểm dựa trên Hist14. Tuy nhiên, binary CANUPO gốc được thử quá chậm khi tính cho toàn ROI, và lọc điểm theo classifier có thể làm mất nhiều điểm của một số đối tượng. Hướng đáng thử tiếp là bổ sung thông tin hình học vào BEV hoặc dùng làm supervision offline; chưa nên đưa nguyên binary này vào đường chạy detector.

Đây là thử nghiệm thực tế bằng bộ CANUPO Linux 64-bit của tác giả, gồm `canupo`, `suggest_classifier_lda`, `validate_classifier` và `classify`. Không huấn luyện lại MobilePixorNeXt trong thử nghiệm này. Các số AP dưới đây là **average precision của phân loại điểm theo nhãn yếu từ bounding box**, không phải KITTI BEV/3D detection AP.

## Dữ liệu và cách thử

Dữ liệu tại `Code/Kitti_Raw` có đủ 7.481 frame training kèm scan, calibration và nhãn hộp. Dùng split hiện tại `Code/splits/kitti/train.txt` / `val.txt`: 3.712 / 3.769 frame, không trùng ID. Chọn mẫu bằng seed 42:

| Phần | Frame | Điểm core đã tính descriptor | Cách dùng |
|---|---:|---:|---|
| Fit | 64 | 274.700 | Học classifier; 48 frame ngẫu nhiên + 16 frame bổ sung pedestrian/cyclist |
| Tune, lấy trong train split | 16 | 67.870 | Chọn threshold theo recall toàn cục 95% |
| Validation | 64 | 273.264 | 48 frame ngẫu nhiên + 16 frame bổ sung pedestrian/cyclist |
| Tổng | 144 | 615.834 | Các scan gốc có tổng 17.240.000 điểm |

Mỗi frame lấy 4.096 điểm ngẫu nhiên trong phần ROI có nhãn sử dụng được, rồi bổ sung tối đa 64 điểm/hộp để chẩn đoán đối tượng nhỏ. **Chỉ các điểm thuộc mẫu ngẫu nhiên trong 48 frame validation ngẫu nhiên được dùng cho chỉ số chính**: 196.608 điểm, trong đó 12.534 điểm dương tính, chiếm 6,38%. Các điểm bổ sung không được đưa vào precision/AP chính.

Nhãn dương tính là điểm nằm trong hộp 3D đã hiệu chuẩn của Car/Pedestrian/Cyclist. Nhãn âm là điểm ngoài các hộp được gán nhãn. Chỉ đánh giá phần nhìn thấy qua camera image_2 bằng P2 và kích thước ảnh thực; loại vùng DontCare, các lớp ngoài mục tiêu, hộp chồng lấp và vùng đệm ngoài hộp rộng 0,15 m. Ngữ cảnh để tính descriptor vẫn lấy từ point cloud gốc quanh ROI, với padding hơn 2 m, không giới hạn theo nhãn.

Đây là **nhãn yếu**: hộp có thể chứa điểm nền; điểm ngoài hộp không phải ground truth semantic background hoàn chỉnh. Không có nhãn pointwise cây/mặt đường nên thử nghiệm không đo accuracy tách vegetation/ground.

CANUPO chạy với **đường kính 0,5 / 1 / 2 / 4 m**, tương ứng bán kính 0,25 / 0,5 / 1 / 2 m. Mỗi scale xuất hai tọa độ dimensionality a,b; thành phần thứ ba là 1−a−b. Kiểm tra bằng binary gốc trên đường thẳng, mặt phẳng và khối cho kết quả đúng loại hình học mong đợi. Với dưới 3 lân cận, mã nguồn gốc có thể giữ lại descriptor của scale lớn hơn; vì thế phân tích luôn lưu số lân cận. Mốc 6 lân cận trong báo cáo là chỉ báo hỗ trợ hình học thận trọng, không phải điều kiện bắt buộc của CANUPO.

Các probe dùng cùng HistGradientBoostingClassifier: 100 vòng, 15 lá, learning rate 0,08, L2=1, không early stopping, cân bằng trọng số foreground/background, seed 42. Không đưa vị trí hộp, kích thước hộp hoặc nhãn đối tượng vào feature. Hist14 cục bộ lấy đúng 14 kênh từ encoder có sẵn tại pillar của điểm, cộng z/intensity/range của điểm, tổng 17 feature. Nhánh support có log số lân cận và cờ đủ 6 lân cận ở bốn scale, thêm 8 feature. Nhánh kết hợp có thêm 8 feature a,b CANUPO, tổng 33.

Classifier LDA gốc dùng tối đa 8.000 điểm/lớp từ phần fit, qua luồng `suggest_classifier_lda → SVG → validate_classifier → PRM → classify`, không chỉnh sửa đường phân chia SVG, không dùng dữ liệu validation để học. Nó dùng dimensionality thuần và có năng lực mô hình khác với probe boosted-tree.

## Kết quả trên 48 frame validation ngẫu nhiên

| Feature / classifier | Point AP (%) | AUROC (%) | Recall điểm trong hộp (%) | Loại điểm ngoài hộp (%) |
|---|---:|---:|---:|---:|
| CANUPO LDA gốc | 20,26 | 86,11 | 93,94 | 64,67 |
| CANUPO geometry + probe | 38,01 | 90,88 | 96,09 | 69,72 |
| CANUPO geometry + support + probe | 44,51 | 92,07 | 94,53 | 72,91 |
| Hist14 cục bộ + probe | 69,62 | 95,18 | 95,76 | 71,92 |
| Hist14 + support + probe | 72,22 | 96,04 | 95,41 | 78,59 |
| **Hist14 + support + CANUPO + probe** | **77,62** | **97,68** | **95,51** | **87,71** |

Các threshold ở hai cột cuối được chọn riêng trên 16 frame tune thuộc train split để đạt recall toàn cục ít nhất 95%. Recall validation có thể khác 95%; không chọn threshold bằng validation.

Tổ hợp đầy đủ tăng **8,00 điểm phần trăm point AP** so với Hist14 cục bộ. Khoảng bootstrap 95% theo frame, 200 lần lấy mẫu với seed 43, là **+5,22 đến +11,93 điểm**. Sau khi đã thêm thông tin số lân cận, CANUPO dimensionality vẫn tăng **5,40 điểm**, với khoảng **+3,37 đến +8,33 điểm**. Điều này cho thấy tín hiệu bổ sung không chỉ đến từ việc đếm điểm lân cận.

Geometry đa tỉ lệ cũng tốt hơn các probe geometry một scale trong cấu hình thử: point AP của đường kính 0,5 / 1 / 2 / 4 m lần lượt là 15,38 / 16,39 / 15,51 / 18,18%, so với 38,01% khi kết hợp bốn scale. Các mô hình một scale dùng cùng thiết lập probe, không được tối ưu hyperparameter riêng.

![So sánh PR và số lân cận](../../artifacts/canupo_feasibility_20261016/main/feature_comparison.png)

Phép thử này chứng minh tín hiệu bổ sung cho **bộ phân loại điểm có ngữ cảnh cục bộ hạn chế**. CNN của detector đã kết hợp thông tin giữa nhiều ô BEV; vì vậy mức tăng ở đây không dự báo trực tiếp mức tăng khi thêm kênh vào MobilePixorNeXt. Bootstrap chỉ phản ánh bất định của mẫu frame validation, không phản ánh biến động giữa nhiều seed huấn luyện, cảm biến hoặc dataset.

## Đối tượng xa và rủi ro lọc điểm

Trên toàn bộ 64 frame validation có 213 Car, 62 Pedestrian và 16 Cyclist có ít nhất một điểm mẫu hợp lệ. Ở vùng xa từ 40 m trở lên, đường kính 0,5 m chỉ có đủ 6 lân cận ở **0,74% điểm Car, 0% điểm Pedestrian và 1,06% điểm Cyclist**. Với đường kính 1 m, các tỷ lệ tương ứng tăng lên 63,49 / 71,88 / 73,40%. Scale nhỏ có thể thiếu thông tin ở xa; cần sử dụng số lân cận hoặc cờ invalid, thay vì xem mọi descriptor đều đáng tin như nhau.

Mẫu xa còn ít: Car 808 điểm/51 đối tượng, Pedestrian 32 điểm/3 đối tượng, Cyclist 94 điểm/7 đối tượng. Các tỷ lệ này là chẩn đoán trong mẫu thử, chưa đủ để chọn bộ scale tối ưu cho mọi cảnh. Lân cận được tính trên cả scene nên có đủ điểm không có nghĩa tất cả đều thuộc đối tượng; scale lớn có thể bao gồm mặt đường và đối tượng gần đó.

![Độ hỗ trợ hình học theo khoảng cách](../../artifacts/canupo_feasibility_20261016/main/neighbor_support_by_distance.png)

Tại threshold chọn theo recall toàn cục 95% trên train tuning, classifier CANUPO LDA gốc giữ 91,83 / 98,25 / 95,03% điểm mẫu Car/Pedestrian/Cyclist trên toàn bộ 64 frame validation. Tuy vậy, nó loại **hơn một nửa số điểm mẫu ở 19/213 Car và 1/16 Cyclist**. Trong nhóm chỉ tính các đối tượng có ít nhất 10 điểm mẫu, vẫn có 18 Car và 1 Cyclist bị mất hơn nửa điểm.

Ngay cả tổ hợp feature tốt nhất cũng có 11 Car, 1 Pedestrian và 2 Cyclist bị loại hơn nửa điểm mẫu tại threshold của nó; hai Car mất toàn bộ điểm mẫu. Đây là tỷ lệ trên các điểm đã lấy mẫu, không phải phép đếm mọi điểm của hộp, và chưa phải số đối tượng detector bỏ sót. Kết quả đủ để thấy recall điểm trung bình cao không bảo đảm giữ hình học tốt cho từng đối tượng. Lần tích hợp đầu nên giữ nguyên điểm và thêm feature/soft weight.

## Chi phí đo thực tế

Máy thử: AMD Ryzen 5 7640HS, x86_64, CANUPO gốc chạy với OMP_NUM_THREADS=4. Với 4.096 core ngẫu nhiên/frame cộng các core bổ sung cho đối tượng, median thời gian descriptor là **282,34 ms**, p95 **489,57 ms**, trên 144 frame.

Đo riêng ba frame ngẫu nhiên ở các tứ phân vị số điểm ROI; mỗi frame chạy ba lần, dùng **mọi điểm trong ROI** làm core:

| Frame | Điểm core toàn ROI | CANUPO descriptor, median | CANUPO classify, median | Hist14 NumPy, median |
|---|---:|---:|---:|---:|
| 003736 | 57.089 | 8.931 ms | 215 ms | 8,40 ms |
| 002277 | 59.544 | 18.044 ms | 248 ms | 6,85 ms |
| 005857 | 61.062 | 8.519 ms | 209 ms | 8,09 ms |

Median gộp của chín lần đo là **8.930,72 ms cho descriptor**, **215,72 ms cho classify**, và **8,09 ms cho Hist14 NumPy**. Descriptor time gồm khởi động CLI, đọc ASCII, tìm lân cận/PCA và ghi MSC; chưa gồm chuẩn bị các file ASCII nguồn/core. Chi phí ghi core ASCII riêng khoảng 66–89 ms/frame. Đây là số đo của binary legacy, chưa phải benchmark CloudCompare hiện đại hoặc một triển khai covariance/GPU đã tối ưu. Không quy đổi các số này thành latency của detector trên GPU.

## Đề xuất cho project

1. **Ưu tiên một ablation thêm feature hình học vào BEV**, giữ nguyên point cloud. Thử 1–2 scale có cờ đủ lân cận; chọn scale trên train/tuning rồi đánh giá validation. So sánh riêng density-only với density + dimensionality để kiểm chứng phần đóng góp hình học.
2. **Dùng CANUPO như công cụ offline** để phân tích scene hoặc tạo supervision hình học cho một nhánh học nhẹ. Nếu dùng teacher, phải tính feature sau các phép augmentation làm đổi lân cận; cache từ scene gốc có thể không còn đúng sau GT sampling, point removal hoặc scaling.
3. **Chưa đưa binary gốc vào runtime và chưa lọc cứng điểm.** Chi phí toàn ROI và mất điểm tập trung ở một số đối tượng không phù hợp với mục tiêu detector nhẹ hiện tại.
4. Thí nghiệm detector tiếp theo phải giữ cùng split, seed, augmentation, head/loss và budget; báo cáo BEV/3D AP từng lớp, lỗi z-bottom/height, recall theo khoảng cách và latency toàn pipeline. Đó mới là bằng chứng về lợi ích đối với project chính.

## Tệp và khả năng tái lập

- Script: [analyze_canupo_kitti.py](../tools/analysis/analyze_canupo_kitti.py).
- [Manifest, các frame và hash](../../artifacts/canupo_feasibility_20261016/main/manifest.json).
- [Metrics đầy đủ](../../artifacts/canupo_feasibility_20261016/main/metrics.json), [bảng CSV](../../artifacts/canupo_feasibility_20261016/main/metrics_summary.csv), [runtime](../../artifacts/canupo_feasibility_20261016/main/runtime.json).
- [Kết quả kiểm chứng](../../artifacts/canupo_feasibility_20261016/main/verification.json): kiểm tra lại 432 hash input, căn chỉnh đúng MSC/feature của 144 frame, tính lại AP/AUROC của 10 classifier từ prediction đã lưu, và xác nhận cache từ chối tham số sampling khác.
- [Minh họa một scene thực](../../artifacts/canupo_feasibility_20261016/main/scene_comparison.png).
- [Snapshot source đúng SHA-256 của lần chạy](../../artifacts/canupo_feasibility_20261016/main/analysis_source_at_run.py). Script hiện tại bổ sung kiểm tra cache và rút gọn log; phần tính toán của lần chạy được lưu trong snapshot.
- `provenance.json` ở thư mục artifact cấp trên lưu hash hai archive tải về và thông tin lỗi tương thích đã xử lý. Dữ liệu gốc, checkpoint và code detector không bị sửa.

Ví dụ chạy lại với một thư mục output mới sau khi tải và giải nén bộ CANUPO Linux của tác giả vào `/tmp`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/canupo-matplotlib \
/home/duyennh/miniconda3/envs/AI_env/bin/python \
  Code/tools/analysis/analyze_canupo_kitti.py \
  --binary-dir /tmp/canupo_linux_static_64bits \
  --output artifacts/canupo_feasibility_rerun
```

Binary 2013 bị lỗi khi Cairo/fontconfig tải module gconv của hệ thống mới. Backtrace xác định lỗi ở đường vẽ chữ SVG; cấu hình font riêng không nạp font hệ thống giúp hoàn tất classifier gốc mà không sửa binary/phần học số học. Sáu kiểm tra nhỏ đã qua, bao gồm đọc split, hộp quay và z-bottom, đọc/ghép đúng dòng MSC khi OpenMP đổi thứ tự, FOV/DontCare/lớp ignore/halo, đối chiếu bản xuất MSC của chương trình gốc, và đường/mặt/khối với binary thật. Pipeline smoke 10 frame và lần chạy chính 144 frame đều hoàn tất.

Nguồn thuật toán và công cụ: [trang CANUPO của tác giả](https://nicolas.brodu.net/en/recherche/canupo/), [bài gốc](https://arxiv.org/abs/1107.0550), [tài liệu plugin CloudCompare](https://www.cloudcompare.org/doc/wiki/index.php/CANUPO_(plugin)).
