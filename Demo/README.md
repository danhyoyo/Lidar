# Demo KITTI: BEV, 3D, heatmap và video

Thư mục này gom code và toàn bộ kết quả visualization đã tạo trong phiên làm
việc. Các ảnh, video, tensor và metadata được sao chép nguyên vẹn; `manifest.json`
ghi nguồn và SHA256 từng file.

## Mở kết quả

**Từng ảnh riêng: 4 thư mục hàng_cột, mỗi thư mục 2 PNG 600 DPI:**
[thư mục ảnh](outputs/heatmap/individual_panels_003736_005663_red/),
[tải toàn bộ ZIP](outputs/heatmap/individual_panels_003736_005663_red.zip).
`1_1`=MobilePIXOR/003736, `1_2`=MobilePIXOR/005663,
`2_1`=Ours/003736, `2_2`=Ours/005663. Mỗi thư mục chỉ có
`detect.png` và `heatmap.png`, kích thước 1631×1170, không chữ/viền ngoài.
Chạy lại bằng `Demo/code/export_kitti_individual_panels.py`.

**Bản hai hàng, chỉ MobilePIXOR và Ours, không có Ground truth:**
[PNG 600 DPI](outputs/heatmap/models_only_003736_005663_red/mobilepixor_ours_bev_003736_005663_2x2.png),
[SVG](outputs/heatmap/models_only_003736_005663_red/mobilepixor_ours_bev_003736_005663_2x2.svg).
Giữ trái 003736, phải 005663, box đỏ và nguyên vẹn detection/heatmap đã kiểm tra.
Chạy lại bằng `Demo/code/export_kitti_models_only.py`. Bản ba hàng phía dưới
được giữ nguyên; verification.json xác nhận các tensor/box và kích thước panel.

**Bản hiện tại: trái 003736, phải 005663; box đỏ, Ground truth 48 mm.**
[PNG 600 DPI](outputs/heatmap/reference_bev_003736_005663_red/gt_mobilepixor_ours_bev_003736_005663_3x2.png),
[SVG](outputs/heatmap/reference_bev_003736_005663_red/gt_mobilepixor_ours_bev_003736_005663_3x2.svg).
Hai checkpoint đã chạy lại trên đúng hai scan; tensor khớp với các lần suy luận
gốc. Script xuất bản này theo mặc định.

**Bản mới nhất: khôi phục trái 003736, đổi phải sang 001225; box đỏ, GT 48 mm.**
[PNG 600 DPI](outputs/heatmap/reference_bev_003736_001225_red/gt_mobilepixor_ours_bev_003736_001225_3x2.png),
[SVG](outputs/heatmap/reference_bev_003736_001225_red/gt_mobilepixor_ours_bev_003736_001225_3x2.svg).
Hai mô hình chạy lại trên đúng hai scan; prediction và heatmap khớp nguyên vẹn
với các bản gốc trước khi đổi vị trí.

**Bản trước: trái 001225, phải 003034, box đỏ và Ground truth lớn hơn.**
[PNG 600 DPI](outputs/heatmap/reference_bev_001225_003034_red/gt_mobilepixor_ours_bev_001225_003034_3x2.png),
[SVG](outputs/heatmap/reference_bev_001225_003034_red/gt_mobilepixor_ours_bev_001225_003034_3x2.svg).
Chọn trái 001225 sau khi xem sáu cảnh có ít lỗi Ours: pointcloud phân bố rộng hơn.
Trong ROI cố định, Ours đúng 5/5, FP=FN=0; MobilePIXOR TP=4, FP=1, FN=1.
Viền GT/prediction đều đỏ `#ff3030`; chiều cao hàng GT tăng từ 42 lên 48 mm.
Cột phải giữ nguyên scan và tensor detection/heatmap của cả hai mô hình.
Ảnh cũ vẫn giữ nguyên.

**Bản trước: trái 003736, phải 003034, chọn theo cả detection và heatmap.**
[PNG 600 DPI](outputs/heatmap/reference_bev_selected_003736_003034/gt_mobilepixor_ours_bev_003736_003034_3x2.png),
[SVG](outputs/heatmap/reference_bev_selected_003736_003034/gt_mobilepixor_ours_bev_003736_003034_3x2.svg),
[scan phải](data/kitti_groundtruth/training/velodyne/003034.bin).
Trong ROI đang vẽ, Ours detect TP=6, FP=0, FN=0; MobilePIXOR TP=5, FP=3, FN=1.
Heatmap max-class: Ours có 6 đỉnh đúng và 0 đỉnh thừa; MobilePIXOR có 6 đỉnh
đúng và 2 đỉnh thừa, với threshold 0.30 và ghép tâm GT trong 1 m.
Thang xám [0,1], grid gốc và crop giữ nguyên. Cột trái giữ nguyên từng pixel.
[Bảng 31 cảnh](outputs/heatmap/comparison_scene_search/ranking.csv) và
[hồ sơ chọn cảnh](outputs/heatmap/reference_bev_selected_003736_003034/selection.json)
lưu tiêu chí, tensor và kiểm tra độc lập. Đây là ví dụ được chọn để minh họa ưu
thế thật trên một cảnh, không phải benchmark ngẫu nhiên; điểm heatmap là tiêu
chí minh họa riêng.

**Cảnh được chọn theo lỗi Ours thấp nhất trong 32 scan validation: 003736.**
[PNG 600 DPI](outputs/heatmap/reference_bev_ours_selected_003736/gt_mobilepixor_ours_bev_003736_005663_3x2.png),
[SVG](outputs/heatmap/reference_bev_ours_selected_003736/gt_mobilepixor_ours_bev_003736_005663_3x2.svg),
[scan gốc](data/kitti_groundtruth/training/velodyne/003736.bin).
Trong ROI cố định, Ours có TP=7, FP=0, FN=0; MobilePIXOR có TP=6, FP=0, FN=1.
Score vẫn 0.30, NMS 0.10; IoU BEV Car=0.70, Pedestrian/Cyclist=0.50.
[Bảng 32 scan](outputs/heatmap/ours_scene_search/ranking.csv) và
[hồ sơ chọn cảnh](outputs/heatmap/reference_bev_ours_selected_003736/selection.json)
lưu tiêu chí và kết quả thật. Đây là ví dụ minh họa được lựa chọn theo hiệu quả
Ours trong ROI, không phải kết quả benchmark ngẫu nhiên. Hai box Ours ngoài ROI
vẫn được giữ trong prediction NPY; ROI và ngưỡng không thay đổi khi chọn cảnh.
GT giữ chiều cao 42 mm, cột phải trong bản này là 005663.

**Bản mới nhất: cột 1 là scan 000008, Ground truth tăng 20%:**
[PNG 600 DPI](outputs/heatmap/reference_bev_000008_005663/gt_mobilepixor_ours_bev_000008_005663_3x2.png),
[SVG](outputs/heatmap/reference_bev_000008_005663/gt_mobilepixor_ours_bev_000008_005663_3x2.svg).
GT cao 42 mm; hai hàng model giữ nguyên. Hai mô hình chạy mới trên scan KITTI gốc;
Ours dùng `checkpoints/130epoch.pt`.
Có thể đổi bằng `--frames 000008 005663 --ground-truth-height-mm 42`.

**Bản mới: cột 1 dùng scan 004338, Ground truth nhỏ, hai hàng mô hình lớn hơn:**
[PNG 600 DPI](outputs/heatmap/reference_bev_004338_005663/gt_mobilepixor_ours_bev_004338_005663_3x2.png),
[SVG](outputs/heatmap/reference_bev_004338_005663/gt_mobilepixor_ours_bev_004338_005663_3x2.svg).
Scan mới tải từ kho KITTI chính thức, khớp CRC32 và SHA256 đã ghi trước đó.
Hai checkpoint chạy inference mới; cột 2 giữ nguyên dữ liệu và kết quả dự đoán.
Hàng GT cao 35 mm, hai hàng model cao khoảng 50 mm; toàn bộ box chỉ là BEV.
Script `Demo/code/export_kitti_reference_comparison.py` tạo bản so sánh.

**Bản BEV zoom sát pointcloud, Ours dùng trực tiếp `checkpoints/130epoch.pt`:**
[PNG 600 DPI](outputs/heatmap/reference_gt_comparison_bev_zoom/gt_mobilepixor_ours_bev_zoom_3x2.png),
[SVG](outputs/heatmap/reference_gt_comparison_bev_zoom/gt_mobilepixor_ours_bev_zoom_3x2.svg).
Crop chung X=[5,38], Y=[-26,20] m cho cả GT, detection và heatmap; không dựng chiều cao box.
Chạy lại bằng `Demo/code/export_kitti_reference_comparison.py` với manifest mặc định
`Demo/configs/reference_bev_zoom.json`. Dữ liệu inference mới và hash nằm trong thư mục kết quả.

Ảnh theo `ref.png`, **3 hàng × 2 cảnh**, chỉ có nhãn Ground truth / MobilePIXOR / Ours:
[PNG 600 DPI](outputs/heatmap/reference_gt_comparison/gt_mobilepixor_ours_reference_3x2.png),
[SVG](outputs/heatmap/reference_gt_comparison/gt_mobilepixor_ours_reference_3x2.svg).
Chạy mới cả hai checkpoint bằng `Demo/code/export_kitti_reference_comparison.py`;
`outputs/heatmap/reference_gt_comparison/inference/` lưu scan gốc, detection,
logits/probabilities, cấu hình và hash. Cảnh 000134 thuộc validation, 005663 thuộc train.

So sánh mới sáu checkpoint, **3 hàng × 2 cặp pointcloud detection + heatmap**:
[frame validation 000134](outputs/heatmap/model_comparison/000134/comparison_000134_3x2_perspective.png),
[frame train 005663](outputs/heatmap/model_comparison/005663/comparison_005663_3x2_perspective.png).
[README và bằng chứng inference](outputs/heatmap/model_comparison/README.md) ghi
checkpoint, scan nguồn đã khớp hash, tensor thật, bản BEV/SVG và lệnh chạy lại.

| Nội dung | File chính |
|---|---|
| Ảnh camera stitching, frame 005663 | [stitched.png](outputs/camera/005663/stitched.png) |
| Raw pointcloud BEV, màu theo độ cao Z | [pointcloud_height_bev.png](outputs/pointcloud_height/005663/pointcloud_height_bev.png) |
| Raw pointcloud 3D, cùng thang màu độ cao Z | [pointcloud_height_3d.png](outputs/pointcloud_height/005663/pointcloud_height_3d.png) |
| BEV detection đơn giản, chỉ score | [detection_bev.png](outputs/detection_simple/005663/detection_bev.png) |
| 3D detection đơn giản, chỉ score | [detection_3d.png](outputs/detection_simple/005663/detection_3d.png) |
| Zoom riêng từng object, một ảnh 3D mỗi object | [zooms/](outputs/detection_simple/005663/zooms/) |
| Pipeline raw → augmentation → hist14 → backbone → detection | [pipeline_overview.png](outputs/pipeline/005663/pipeline_overview.png) |
| Toàn bộ 14 kênh hist14 | [04_hist14_channels.png](outputs/pipeline/005663/04_hist14_channels.png) |
| Các lớp backbone và neck thật | [05_backbone_features.png](outputs/pipeline/005663/05_backbone_features.png) |
| Video hai camera + LiDAR + BEV + heatmap | [visualization.mp4](outputs/video/camera_lidar_bev_heatmap/visualization.mp4) |
| Preview video camera | [preview.png](outputs/video/camera_lidar_bev_heatmap/preview.png) |
| Sáu mốc trong video camera | [contact_sheet.png](outputs/video/camera_lidar_bev_heatmap/contact_sheet.png) |
| Video LiDAR + BEV + heatmap | [visualization.mp4](outputs/video/lidar_bev_heatmap/visualization.mp4) |
| BEV frame 005663 | [overview_preview.png](outputs/bev/005663/overview_preview.png) |
| BEV frame validation 004338 | [overview_preview.png](outputs/bev/004338_validation/overview_preview.png) |
| Box 3D từ BEV và LiDAR, frame 005663 | [overview_preview.png](outputs/3d_lidar_lift/005663/overview_preview.png) |
| Box 3D frame validation 004338 | [overview_preview.png](outputs/3d_lidar_lift/004338_validation/overview_preview.png) |
| Heatmap model và Gaussian target, frame 005663 | [heatmap_preview.png](outputs/heatmap/005663/heatmap_preview.png) |
| Overview dùng ground truth, frame 005663 | [overview_preview.png](outputs/ground_truth/005663/overview_preview.png) |

Hai video đều có **174 frame liên tiếp, 10 FPS, 17.4 giây**, drive
`2011_09_26_drive_0014_sync`, raw frame 77–250. Video camera là 1920 × 1080;
video ba panel là 1920 × 800.

## Cấu trúc

```text
Demo/
├── README.md
├── manifest.json
├── code/                         # Các script visualization và Colab
├── tests/                        # Kiểm tra visualization
├── docs/                         # Hướng dẫn chi tiết và các kế hoạch
├── data/
│   ├── images/training/
│   │   ├── image_2/               # 174 ảnh camera trái
│   │   └── image_3/               # 174 ảnh camera phải
│   └── temporal_mapping/         # Mapping chính thức + frame list
└── outputs/
    ├── bev/                      # 005663 và 004338_validation
    ├── 3d_lidar_lift/             # 005663 và 004338_validation
    ├── heatmap/005663/
    ├── ground_truth/005663/
    ├── pipeline/005663/           # Raw, augmentation, cells, hist14, features, heads, detections
    └── video/
        ├── lidar_bev_heatmap/
        └── camera_lidar_bev_heatmap/
```

Các thư mục output giữ cả PNG/SVG, zoom Car/Cyclist/Pedestrian, prediction
NPY/NPZ, logits/probabilities, ảnh từng frame và metadata tương ứng.
`data/images/extraction_metadata.json` ghi ZIP ảnh người dùng cung cấp.

## Nguồn dự đoán

- **BEV:** checkpoint đã chọn, cấu hình nhúng, strict state dict; model nhận LiDAR.
- **3D:** XY, kích thước mặt bằng, yaw, class và score lấy từ BEV. Z và chiều cao
  được ước lượng từ điểm LiDAR thật; checkpoint hiện tại chưa có head dự đoán Z/h.
  Score giữ nguyên từ BEV.
- **Heatmap:** `gaussian_heatmap*` là sigmoid classification logits của model;
  `gaussian_target*` là target Gaussian từ nhãn KITTI. Hai loại được lưu riêng.
- **Video camera:** ảnh RGB chỉ dùng hiển thị đồng bộ. Prediction và heatmap
  phát lại từ cache inference của đúng scan, không có inference bằng RGB.
- **Ground truth:** output trong `outputs/ground_truth` dùng nhãn thật để tham chiếu.

Frame 005663 thuộc train split; frame 004338 và toàn bộ clip thuộc validation
split của thí nghiệm. Các ảnh validation vẫn nằm trong KITTI `training`.

## Chạy lại

Giữ `Demo` cạnh `Code` ở thư mục gốc dự án. Script trong `Demo/code` dùng chung
thư viện detector và dataset tại `Code`; chỉ đường dẫn bootstrap được điều chỉnh
để chạy từ vị trí mới. Checkpoint và dataset LiDAR vẫn dùng bản đang có trong
dự án/Downloads. Metadata kết quả giữ đường dẫn nguồn ban đầu để đối chiếu.

Chạy từ thư mục gốc `Lidar`:

```bash
DEMO_PY=/home/duyennh/miniconda3/envs/AI_env/bin/python
DEMO_CHECKPOINT=/home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-oga_loss-hist14-iqa-sgfpn-ctx_focal_local_none_f32_detail0_e45d83922c52-grouped_e02c0e499671-s42-P4-h32/selected/best.pt

# BEV từ checkpoint trên frame 005663
"$DEMO_PY" Demo/code/predict_kitti_overview.py \
  --checkpoint "$DEMO_CHECKPOINT" --frame 005663 --device cpu \
  --output-dir Demo/outputs/bev/005663

# Toàn bộ pipeline trên cloud sau augmentation, lấy tensor thật từ model
"$DEMO_PY" Demo/code/export_kitti_pipeline.py \
  --checkpoint "$DEMO_CHECKPOINT" --frame 005663 --seed 2 --device cpu \
  --output-dir Demo/outputs/pipeline/005663

# Raw pointcloud BEV và 3D, màu turbo theo Z thật; cùng thang màu cho cả hai
"$DEMO_PY" Demo/code/export_kitti_height_pointcloud.py --frame 005663 \
  --output-dir Demo/outputs/pointcloud_height/005663

# Detection trên đúng raw scan: nền theo Z, Car đỏ / Pedestrian xanh / Cyclist vàng
"$DEMO_PY" Demo/code/export_kitti_simple_detections.py \
  --prediction-dir Result/artifacts/kitti_overview_model \
  --height-metadata Demo/outputs/pointcloud_height/005663/metadata.json \
  --output-dir Demo/outputs/detection_simple/005663

# Ước lượng Z/h từ LiDAR và dựng box 3D
"$DEMO_PY" Demo/code/lift_kitti_bev_to_3d.py \
  --prediction-dir Demo/outputs/bev/005663 \
  --output-dir Demo/outputs/3d_lidar_lift/005663

# Heatmap prediction + Gaussian target trên cùng frame
"$DEMO_PY" Demo/code/export_kitti_gaussian_heatmap.py \
  --prediction-dir Demo/outputs/bev/005663 --source both --device cpu \
  --output-dir Demo/outputs/heatmap/005663

# Inference và video cho 174 frame
"$DEMO_PY" Demo/code/export_kitti_sequence_video.py \
  --checkpoint "$DEMO_CHECKPOINT" --device cpu \
  --mapping-dir Demo/data/temporal_mapping \
  --frame-list Demo/data/temporal_mapping/longest_consecutive_object_ids.txt \
  --output-dir Demo/outputs/video/lidar_bev_heatmap

# Ghép hai camera vào cache inference trên
"$DEMO_PY" Demo/code/export_kitti_camera_video.py \
  --cache-dir Demo/outputs/video/lidar_bev_heatmap \
  --image-dir Demo/data/images --camera-mode stereo \
  --output-dir Demo/outputs/video/camera_lidar_bev_heatmap
```

Với frame validation, dùng `--frame 004338` và output
`Demo/outputs/bev/004338_validation`, rồi truyền thư mục đó vào các bước 3D/heatmap.
Dùng `--output-dir` khác để tạo một lần chạy mới nếu cần giữ bản demo đã gom.

Code lấy ảnh từ Colab:
[colab_export_matching_kitti_images.py](code/colab_export_matching_kitti_images.py).
Hướng dẫn đầy đủ từng script:
[README_kitti_overview.md](docs/README_kitti_overview.md).

## Kiểm tra

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$DEMO_PY" -m pytest Demo/tests -q
```

`outputs/` và `data/` được bỏ qua khi thêm vào Git bằng `.gitignore` trong Demo.

Pipeline mới lưu đầy đủ trong `outputs/pipeline/005663`: ảnh từng bước PNG/SVG,
14 kênh hist14, tensor backbone/head, cloud raw/augmented và metadata. Cloud
augmentation dùng crop vật thể KITTI thật từ 128 donor frame thuộc train,
không lấy từ validation hoặc chính frame đang xem. Đây là minh họa đường dữ
liệu training rồi chạy checkpoint ở chế độ eval, không phải kết quả AP.
Feature màu là activation của model; chỉ heatmap classification là xác suất.
Với checkpoint BEV hiện tại, hình cuboid cuối dùng Z/h ước lượng từ LiDAR.

## Giải nén kitti_images.zip từ MyDrive trong Colab

[extract_kitti_images.py](code/extract_kitti_images.py) chạy độc lập bằng Python
standard library. Upload script vào `/content` của Colab, mount Drive, rồi chạy:

```python
!python /content/extract_kitti_images.py --mydrive-root /content/drive/MyDrive --output-dir /content/kitti_raw/training/image_2
```

Script tìm đúng tên `kitti_images.zip` kể cả trong thư mục con, chỉ lấy ảnh
training camera trái và giữ tên sáu chữ số theo Object ID. Có thể truyền
`--archive /content/drive/MyDrive/folder/kitti_images.zip` để chọn file cụ thể.
Kích thước từng ảnh được lưu vào `image_2_extraction.json`.
