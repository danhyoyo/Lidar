# Demo KITTI: BEV, 3D, heatmap và video

Thư mục này gom code và toàn bộ kết quả visualization đã tạo trong phiên làm
việc. Các ảnh, video, tensor và metadata được sao chép nguyên vẹn; `manifest.json`
ghi nguồn và SHA256 từng file.

## Mở kết quả

### Bộ input Hist14 để ghép diagram bài báo

Frame **005663**, scan gốc không augmentation; vùng focus X=0–45 m,
Y=−20–20 m có 4 Car, 13 Pedestrian, 4 Cyclist theo nhãn KITTI.
Mở [gallery](outputs/hist14_inputs/005663/index.html) hoặc
[bảng tổng hợp 14 kênh](outputs/hist14_inputs/005663/hist14_overview.png).
Ảnh từng kênh có tên/thang màu nằm trong `outputs/hist14_inputs/005663/channels/`;
PNG nền trong suốt để ghép diagram nằm trong `transparent/`.
Có cloud 3D/BEV, toàn bộ ROI, tensor gốc và metadata nguồn;
[README kết quả](outputs/hist14_inputs/005663/README.md) giải thích công thức từng kênh.
Chạy lại bằng `Demo/code/export_hist14_inputs.py`.

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
