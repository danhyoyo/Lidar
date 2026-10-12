# Visualization KITTI cho hình overview

Script: `Code/tools/visualization/export_kitti_overview.py`.

Đọc trực tiếp `.bin` float32 XYZI, label KITTI và calibration của cùng frame.
Mọi điểm hiển thị đều lấy từ dữ liệu gốc; không sinh điểm, nội suy, tăng mật độ hay
chèn vật thể. Box được dựng trong hệ camera rồi chuyển cả 8 đỉnh sang LiDAR bằng
`inverse(R0_rect @ Tr_velo_to_cam)`, giữ nguyên ảnh hưởng pitch/roll của calibration.
Hình chiếu 3D orthographic giữ cả chiều cao, với cùng góc nhìn cho hai panel cảnh.

## Chạy từ thư mục gốc workspace

Phụ thuộc: `numpy`, `matplotlib`, `Pillow`. Không cần GPU, Open3D hoặc màn hình GUI.

```bash
python3 Code/tools/visualization/export_kitti_overview.py \
  --kitti-root Code/Kitti_Raw \
  --frame auto \
  --zoom-padding 0 \
  --output-dir Result/artifacts/kitti_overview
```

Script hỗ trợ cấu trúc chính thức `training/{velodyne,label_2,calib}`, thư mục
training phẳng, và cấu trúc giải nén riêng từng ZIP hiện có trong workspace.
Nó quét label của tất cả frame có đủ `.bin` và calibration, ưu tiên có đủ ba lớp,
số lượng đối tượng dùng được tối thiểu ở mỗi lớp, rồi tổng số đối tượng. Trong
30 ứng viên đầu, script kiểm tra số điểm thực trong box và yêu cầu mỗi lớp có ít
nhất một đối tượng với 20 điểm. Có thể tăng `--scan-top-k` để xét thêm ứng viên.
Đối tượng dùng được nằm phía trước LiDAR, trong 45 m, occlusion 0/1 và truncation
không quá 0.3. Ảnh zoom của mỗi lớp chọn box có nhiều điểm nhất trong nhóm này.

Frame đã chọn: **005663**, 123,465 điểm gốc, **4 Car + 4 Cyclist + 13 Pedestrian**.
Các zoom cùng frame:

| Lớp | Chỉ số dòng label, bắt đầu từ 0 | Điểm bên trong box |
| --- | ---: | ---: |
| Car | 1 | 160 |
| Cyclist | 9 | 530 |
| Pedestrian | 17 | 606 |

Để dựng lại đúng frame mà không quét dataset:

```bash
python3 Code/tools/visualization/export_kitti_overview.py \
  --frame 005663 --zoom-padding 0 \
  --output-dir Result/artifacts/kitti_overview
```

## File xuất

| File | Dùng cho |
| --- | --- |
| `lidar_pointcloud.png`, `.svg` | Panel input 3D LiDAR, màu theo độ cao |
| `detection_3d.png`, `.svg` | Cùng cảnh với điểm cyan và box đỏ |
| `zoom_car.png`, `.svg` | Zoom một Car |
| `zoom_cyclist.png`, `.svg` | Zoom một Cyclist |
| `zoom_pedestrian.png`, `.svg` | Zoom một Pedestrian |
| `object_zooms.png` | Một ảnh ghép ba zoom |
| `overview_preview.png` | Ảnh ghép hai cảnh và ba zoom để xem nhanh |
| `metadata.json` | Frame, loại box, số điểm, góc nhìn, ROI, đường dẫn và SHA256 input |
| `frame_candidates.json` | Các ứng viên và số đối tượng / điểm; được xuất khi chọn auto |
| `zoom_*_points.npz` | Điểm XYZI nguyên gốc, index trong `.bin` và tọa độ 8 đỉnh box |

Panel cảnh mặc định 1800 × 1200 px, zoom 1000 × 1000 px, 300 DPI, không trục
hoặc tiêu đề để ghép vào overview. SVG giữ box dạng vector và nhúng layer điểm
dạng raster nhằm tránh file vector chứa hàng chục nghìn marker. Dữ liệu và ảnh
xuất ở `Result/artifacts/` được gitignore.

## Tùy chỉnh bố cục

```bash
python3 Code/tools/visualization/export_kitti_overview.py \
  --frame 005663 --zoom-padding 0 \
  --elevation 28 --azimuth -145 \
  --zoom-elevation 14 --zoom-azimuth -150 \
  --scene-size 2400 1600 --zoom-size 1400 --dpi 300 \
  --output-dir Result/artifacts/kitti_overview_large
```

- `--roi XMIN XMAX YMIN YMAX ZMIN ZMAX`: vùng điểm LiDAR, đơn vị mét; mặc định
  `-2 45 -24 24 -2.8 4`. Hai panel cảnh dùng cùng ROI, camera và toàn bộ điểm
  trong ROI, không lấy mẫu thưa. Đây là vùng phía trước có annotation, không phải
  toàn bộ scan 360 độ. Có thể dùng `--roi -45 45 -45 45 -3 6` cho cảnh 360 độ.
- `--point-size`, `--zoom-point-size`: diện tích marker (points²), không thay đổi
  số điểm đo.
- `--zoom-padding 0`: chỉ hiển thị điểm trong box. Giá trị `0.15` thêm vùng ngữ
  cảnh thực xung quanh box; số điểm bên trong và số điểm hiển thị được ghi riêng.
- `--zoom-azimuth`: góc nhìn tương đối so với trục chiều dài của từng vật thể;
  `--azimuth`: góc nhìn toàn cảnh trong hệ LiDAR.
- `--class-colors`: Car đỏ, Cyclist vàng, Pedestrian xanh lá.
- `--labels`: hiện chỉ số đối tượng ở ground truth, hoặc score thật ở prediction.
  Ba đối tượng được chọn để zoom có thêm tên lớp trên panel cảnh.

## Ground truth và prediction

**Bộ ảnh mặc định dùng KITTI ground truth để minh họa box 3D, không phải kết quả
inference của mô hình.** Không có checkpoint hoặc prediction thật trong workspace
khi tạo bộ ảnh này. Khi đưa vào bài báo, caption cần ghi rõ ground truth nếu
chưa thay bằng prediction.

Script cũng nhận kết quả detection thật theo định dạng KITTI 16 trường:
`TYPE TRUNC OCC ALPHA X1 Y1 X2 Y2 H W L X Y Z ROTATION_Y SCORE`.
Tọa độ `X Y Z` là bottom center trong rectified camera, không phải LiDAR.
Không truyền trực tiếp label đã chuyển sang định dạng detector 8 trường.

```bash
python3 Code/tools/visualization/export_kitti_overview.py \
  --frame 005663 \
  --predictions-dir /path/to/real/kitti_predictions \
  --score-threshold 0.3 --labels --zoom-padding 0 \
  --output-dir Result/artifacts/kitti_overview_predictions
```

File cần có là `005663.txt`. Prediction được lọc theo score thật; không dùng
occlusion/truncation để loại box vì các trường này có thể là `-1` trong kết quả
inference. Chế độ này cần prediction đủ cả ba lớp với điểm LiDAR trong box để
xuất ba zoom. Nếu thiếu lớp hoặc box không đủ điểm, script báo lỗi thay vì thay
thế bằng box GT. `metadata.json` ghi loại box và SHA256 prediction để truy vết.

## Kiểm chứng

```bash
python3 -m pytest Code/tests/test_kitti_overview_visualization.py -q
python3 -m py_compile Code/tools/visualization/export_kitti_overview.py
```

Tests kiểm tra phép biến đổi đủ pitch/roll, point-in-box, binary thiếu byte,
đọc score prediction, chọn đủ ba lớp, layout ZIP và xuất end-to-end giữ nguyên
điểm / index. Fixture nhỏ trong tests chỉ dùng kiểm tra code; ảnh xuất cho bài
báo dùng dữ liệu KITTI thật trong workspace.

Nguồn dataset / format: [KITTI Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php).

## Chạy checkpoint thật đã cung cấp

Entry point mới: `Code/tools/visualization/predict_kitti_overview.py`.
Checkpoint `selected/best.pt` của run
`mobilepixornext-hybrid_gt_aug-oga_loss-hist14-iqa-sgfpn-ctx_focal_local_none_f32_detail0_e45d83922c52-grouped_e02c0e499671-s42-P4-h32`
là model **BEV**, epoch 50. Không có vertical head trong state dict. Do đó script
xuất box XY xoay và score thật; không gán chiều cao / Z từ ground truth để tạo
box 3D. Trên cảnh 3D, hình chữ nhật chỉ là footprint BEV đặt trên mặt phẳng hiển
thị `z=-1.7 m`, ghi rõ trong metadata. Điểm XYZ vẫn giữ nguyên giá trị đo gốc.

Môi trường đã chạy thành công trên CPU:
`/home/duyennh/miniconda3/envs/AI_env/bin/python`, PyTorch 2.11.0, NumPy, Numba,
Shapely, Matplotlib và Pillow. Python hệ thống hiện không có PyTorch.

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python \
  Code/tools/visualization/predict_kitti_overview.py \
  --checkpoint /home/duyennh/Downloads/artifacts-20261011T082304Z-1-001/artifacts/mobilepixornext-hybrid_gt_aug-oga_loss-hist14-iqa-sgfpn-ctx_focal_local_none_f32_detail0_e45d83922c52-grouped_e02c0e499671-s42-P4-h32/selected \
  --frame 005663 --device cpu \
  --score-threshold 0.3 --nms-threshold 0.1 \
  --output-dir Result/artifacts/kitti_overview_model
```

Mặc định script dùng **config được lưu trong checkpoint**, xác thực semantic
identity và load toàn bộ weights với `strict=True`. File config ngoài của run
hiện khai báo `header_architecture: shared`, không khớp checkpoint gốc; nó không
được tự động dùng. Nếu truyền `--config` rõ ràng, config này vẫn phải qua kiểm tra
tương thích đầy đủ. Không sửa checkpoint hoặc file config bên ngoài.

Score sử dụng IQA ranking của decoder hiện có, với `nms_alpha=0.5`:
`sqrt(class_probability * sigmoid(iou_logit))`. Ngưỡng truyền vào decoder áp dụng
cả ranking score và class probability; nó không chỉ lọc cột score của một file
prediction đã lưu. NMS classwise BEV có threshold 0.1.

Kết quả chạy mới, threshold 0.3:

| Frame | Split | Car | Cyclist | Pedestrian | Thư mục kết quả |
| --- | --- | ---: | ---: | ---: | --- |
| 005663 | train | 3 | 5 | 12 | `Result/artifacts/kitti_overview_model` |
| 004338 | validation | 4 | 5 | 7 | `Result/artifacts/kitti_overview_model_validation` |

Split validation cục bộ đã được đối chiếu bằng tập frame ID với file
`evaluation_ap_local_bev.predictions.npz` của run và trùng toàn bộ 3,769 frame.
Các kết quả trên là số detection sau decode/NMS; không phải số true positive
hoặc điểm AP trên một frame.

Chạy lại bộ validation bằng cách thay `--frame 004338` và
`--output-dir Result/artifacts/kitti_overview_model_validation` trong lệnh trên.

Các file mới trong mỗi thư mục:

- `detection_bev.png/.svg`: góc nhìn từ trên xuống, box BEV và score thật.
- `prediction_footprints_3d.png/.svg`: cùng footprint hiển thị trên cảnh LiDAR 3D.
- `lidar_pointcloud.png/.svg`, `zoom_car.png/.svg`, `zoom_cyclist.png/.svg`,
  `zoom_pedestrian.png/.svg`, `overview_preview.png`, `object_zooms.png`.
- `bev_predictions.npz`, `<frame>_bev_predictions.npy`, `predictions.csv`:
  output thật `[class_id, score, x, y, length, width, yaw]` trong hệ LiDAR.
- `config.inference.json`: config chính xác đã dùng.
- `metadata.json`: checkpoint/config/bin hashes, epoch, decoder settings,
  split manifests, các prediction, loại hình học và lệnh chạy lại.
- `zoom_*_points.npz`: điểm XYZ/intensity gốc, source indices và row BEV tương ứng.

Zoom chỉ crop theo footprint XY dự đoán và ROI Z hiển thị; có thể chứa điểm nền
và vật thể lân cận. Nó không phải instance segmentation hoặc box 3D dự đoán.
Không dùng label GT để chọn prediction, crop điểm hoặc dựng hình.

```bash
python3 -m pytest Code/tests/test_kitti_overview_prediction.py \
  Code/tests/test_kitti_overview_visualization.py -q
```

## Cuboid 3D từ prediction BEV và điểm LiDAR

Script `Code/tools/visualization/lift_kitti_bev_to_3d.py` bổ sung bước hậu xử lý để
vẽ cuboid 3D cho yêu cầu tiếp theo. **X/Y, dài/rộng, yaw, lớp và score lấy nguyên
từ model; Z đáy và chiều cao được ước lượng từ LiDAR**, không phải output của
vertical head. Không dùng nhãn GT, chiều cao cố định theo lớp hoặc điểm sinh thêm.

Phương pháp:

1. Lấy điểm quanh mỗi box, loại các vùng XY đã có prediction để tránh dùng điểm
   vật thể làm mẫu mặt đường. Thử vòng lân cận rộng 2, 4 rồi 7 m khi cần.
2. Chọn điểm đo thấp đại diện cho từng ô XY 1 m, khớp mặt phẳng mặt đường bằng
   least squares với loại outlier theo MAD. Cần ít nhất 6 ô hỗ trợ và slope hợp lệ.
3. Trong footprint dự đoán, lấy điểm cao hơn mặt đường 0.2 m. Tách nhóm theo khoảng
   cách Z 0.45 m và giữ nhóm có nhiều điểm nhất để giảm ảnh hưởng điểm cao rời rạc.
4. Dùng Z mặt đường tại tâm box làm đáy, quantile 98% của Z nhóm điểm hỗ trợ làm
   đỉnh. Cần tối thiểu 5 điểm hỗ trợ và khoảng cao 0.3 m. Không đủ dữ liệu thì ghi
   lý do trong metadata và không gán chiều cao thay thế.

```bash
python3 Code/tools/visualization/lift_kitti_bev_to_3d.py \
  --prediction-dir Result/artifacts/kitti_overview_model \
  --output-dir Result/artifacts/kitti_overview_model_3d

python3 Code/tools/visualization/lift_kitti_bev_to_3d.py \
  --prediction-dir Result/artifacts/kitti_overview_model_validation \
  --output-dir Result/artifacts/kitti_overview_model_validation_3d
```

Đã xuất cuboid cho 20/20 prediction ở frame 005663 và 16/16 ở frame 004338.
Box nằm ngoài ROI hiển thị vẫn được lưu trong NPZ; chỉ box giao với ROI mới
được vẽ trên panel cảnh, với số lượng hiển thị được ghi riêng trong metadata.
Các output gồm `detection_3d.png/.svg`, ba zoom riêng `.png/.svg`,
`lidar_pointcloud.png/.svg`, `object_zooms.png`, `overview_preview.png`, các subset
điểm gốc `.npz`, `metadata.json` và `lifted_detections_3d.npz`.

Trong file detection NPZ, `rows` có thứ tự
`[class, score, x, y, z_bottom, length, width, height, yaw]`.
`source_prediction_indices` liên kết từng cuboid với row BEV gốc. Metadata lưu
plane, source indices dùng khớp ground / chiều cao, estimator settings, hashes của
input và nguồn inference. Class/score/geometry BEV được giữ nguyên cả dtype float32.

Zoom chỉ dùng các điểm đo gốc bên trong cuboid đã ước lượng; không nội suy.
Chiều cao có thể thấp hơn vật thể thực khi điểm thưa hoặc bị che khuất. Score
hiển thị vẫn là score BEV của model, không phải confidence mới cho chiều cao.
Khi dùng hình trong bài báo, có thể ghi caption:
“3D visualization of predicted BEV boxes with LiDAR-estimated vertical extents.”

```bash
python3 -m pytest Code/tests/test_kitti_3d_lifting.py \
  Code/tests/test_kitti_overview_prediction.py \
  Code/tests/test_kitti_overview_visualization.py -q
```

## Gaussian heatmap cùng frame 005663

Script `Code/tools/visualization/export_kitti_gaussian_heatmap.py` xuất hai bộ riêng:

- `gaussian_heatmap*.png/.svg`: **prediction thật** từ `sigmoid(cls_logits)` của
  checkpoint đã dùng cho detection. Lấy tensor trước peak filtering, threshold,
  IQA reranking và NMS; không blur hoặc tự dựng Gaussian quanh box dự đoán.
- `gaussian_target*.png/.svg`: **Gaussian target** từ label và calibration KITTI,
  qua `convert_labels`, `Dataset.filter_boxes` và `Dataset.update_cls_map` của
  pipeline. Không augmentation. Target dùng label; inference của model chỉ dùng LiDAR.

Lệnh chạy từ gốc workspace, dùng môi trường PyTorch hiện có:

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python \
  Code/tools/visualization/export_kitti_gaussian_heatmap.py \
  --prediction-dir Result/artifacts/kitti_overview_model \
  --device cpu \
  --output-dir Result/artifacts/kitti_overview_heatmap
```

Mặc định script đọc frame/checkpoint từ metadata của detection trước đó, kiểm tra
SHA256 checkpoint và `.bin`, dùng config nhúng trong checkpoint, chạy lại inference
và đối chiếu các row detection trước khi render. Có thể dùng `--checkpoint /path/to/selected`
và `--frame 005663` để chạy độc lập. `--source model`, `--source target` hoặc
`--source both` chọn loại heatmap; mặc định `both`.

Các panel tổng hợp lấy **max qua ba lớp**, với panel riêng có hậu tố `_car`,
`_pedestrian`, `_cyclist`. Ảnh 1200 × 1200 px, 300 DPI, không trục/tiêu đề để ghép
overview. `heatmap_preview.png` có tên từng loại, từng lớp, trục mét và colorbar.
Tất cả dùng `viridis`, thang cố định **0–1**, hiển thị từng ô gốc bằng `nearest`.
`--cmap`, `--size` và `--dpi` chỉ thay đổi trình bày.

Tensor đầy đủ có shape **[3, 200, 176] = [class, y, x]**, thứ tự kênh
**Car, Pedestrian, Cyclist**, ô **0.4 × 0.4 m**. Grouped heads được ghép theo global
class ID của checkpoint. ROI mặc định `0 45 -24 24` bám vùng cảnh trước đó trong
phạm vi model, làm tròn ra biên ô thực thành `0 45.2 -24 24`. Dùng `--full-grid`
để hiển thị toàn bộ XY của model. Target có **4 Car, 4 Cyclist, 13 Pedestrian**;
`gaussian_overlap=0.1`, `min_radius=4`, sigma `(2 * radius + 1) / 6` đúng pipeline.

`model_heatmaps.npz` lưu logits và probabilities đầy đủ, `gaussian_targets.npz`
lưu target và box thực sau lọc, `*_crop.npz` lưu tensor ROI cùng tọa độ mét.
`metadata.json` ghi nguồn dữ liệu, hashes, cấu hình inference, ROI và cách gộp lớp.
Frame 005663 nằm trong train split, như metadata detection đã ghi.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest \
  Code/tests/test_kitti_gaussian_heatmap.py \
  Code/tests/test_kitti_overview_visualization.py \
  Code/tests/test_kitti_overview_prediction.py \
  Code/tests/test_kitti_3d_lifting.py -q
```
