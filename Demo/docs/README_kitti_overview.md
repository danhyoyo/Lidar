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

## Video visualization theo thứ tự thời gian

Script `Code/tools/visualization/export_kitti_sequence_video.py` dùng một instance
model cho toàn bộ clip, lấy prediction và heatmap classification từ cùng forward
của từng scan. Bố cục ba panel: LiDAR XYZ chiếu 3D orthographic, box BEV có score,
và heatmap sigmoid gộp max qua lớp. Giữ cố định camera, ROI và thang màu; không
sinh điểm, thêm chiều cao vào box hoặc làm mượt prediction theo thời gian.

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python \
  Code/tools/visualization/export_kitti_sequence_video.py \
  --device cpu \
  --output-dir Result/artifacts/kitti_sequence_video
```

Default frame list là
`Result/artifacts/kitti_temporal_mapping/longest_consecutive_object_ids.txt`.
Mỗi Object ID được tra lại bằng `train_mapping[train_rand[object_id] - 1]` trong
official KITTI devkit. Script yêu cầu raw frame liên tiếp trong cùng drive,
reject ID trùng, đảo thứ tự, gap hoặc thiếu `.bin`; không sort theo Object ID.
Chuỗi đã xuất gồm **174 frame**, raw **77–250**, drive
**2011_09_26_drive_0014_sync**, **10 FPS, 17.4 giây, 1920 × 800 px, H.264 MP4**.
Mapping chỉ cho raw frame index; timestamp tuyệt đối không có trong bản Object.
Thời gian nguồn tương đối lấy theo tần số danh định KITTI 10 Hz. `--fps` đổi tốc
độ phát, không tạo frame mới. ROI XY được snap ra biên ô heatmap và dùng chung
cho các panel. Counts dưới BEV là box giao ROI hiển thị; NPZ giữ toàn bộ prediction.

Checkpoint mặc định lấy từ metadata của bộ single-frame trước đó, kiểm tra SHA256
và dùng config nhúng trong checkpoint. Có thể truyền `--checkpoint /path/to/selected`
để dùng model khác cùng encoding hist14/BEV/Gaussian. `--frame-list` và
`--mapping-dir` chọn chuỗi/mapping khác. Smoke nhanh bằng `--limit 4` và output-dir riêng.
Phụ thuộc thêm: PyTorch, Numba (theo config), FFmpeg và ffprobe.

Output:

- `visualization.mp4`: clip ghép ba panel, số frame sau decode được kiểm tra bằng ffprobe.
- `preview.png`, `contact_sheet.png`: một frame và montage sáu mốc của clip.
- `frames/frame_000000.png`…: ảnh gốc trước nén video.
- `bev_predictions.npz`: key là Object ID, row `[class,score,x,y,length,width,yaw]`.
- `heatmaps/<object_id>.npz`: logits/probabilities đầy đủ, global class order,
  `scene_source_indices` của các điểm thật hiển thị trong `.bin` gốc.
- `frame_manifest.json`: thứ tự video/raw frame, đường dẫn/hashes LiDAR, số điểm,
  counts toàn bộ/hiển thị và hash heatmap từng frame.
- `metadata.json`, `config.inference.json`: nguồn checkpoint, config, mapping,
  camera/ROI/FPS, thời gian, split membership và lệnh tái tạo.

Ground truth không tham gia inference/render clip này. Việc phát detection theo
thời gian không tự tạo tracking ID; box/score vẫn là inference độc lập từng frame.

## Ghép ảnh camera trái/phải vào video 174 frame

`Code/tools/visualization/export_kitti_camera_video.py` đọc ZIP ảnh đã tải từ Colab
và ghép camera thật vào kết quả inference của clip trên. Mặc định dùng
`~/Downloads/kitti_training_174frames_images.zip`, kiểm tra thứ tự trong
`frame_ids.txt`, CRC, đủ **348 ảnh RGB 1242 × 375** và trích đúng các member
`training/image_2/<ObjectID>.png`, `training/image_3/<ObjectID>.png` vào
`Result/artifacts/kitti_sequence_images/`. Không dùng video index hoặc raw frame
để tìm ảnh. Các frame validation vẫn thuộc thư mục KITTI `training`.

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python \
  Code/tools/visualization/export_kitti_camera_video.py \
  --images-zip /home/duyennh/Downloads/kitti_training_174frames_images.zip \
  --output-dir Result/artifacts/kitti_sequence_video_camera
```

Video **1920 × 1080, 10 FPS, 174 frame, 17.4 giây** gồm hai camera ở hàng trên
và LiDAR XYZ, box BEV, heatmap classification ở hàng dưới. Ảnh giữ tỷ lệ và màu
RGB gốc. `--camera-mode left` chỉ hiển thị camera trái; `--image-dir` chọn root
ảnh đã giải nén, có cấu trúc `training/image_2`, `training/image_3`. Dùng `--limit 1`
và output-dir riêng để kiểm tra bố cục nhanh.

RGB chỉ dùng để hiển thị đồng bộ. Model vẫn nhận LiDAR; script phát lại các
prediction/heatmap đã lưu, không forward lại và không vẽ box suy đoán lên ảnh.
Script đối chiếu config, official mapping, `.bin`, heatmap và chỉ số điểm bằng
SHA256 hoặc dữ liệu gốc trước khi render; thang heatmap cố định 0–1.

Output mới: `visualization.mp4`, `preview.png`, `contact_sheet.png`, `frames/`,
`frame_manifest.json` (thêm đường dẫn/hash ảnh trái/phải) và `metadata.json`
(ZIP nguồn, camera mode, hashes cache, checkpoint, FPS và lệnh tái tạo).
Video ba panel và các tensor inference gốc vẫn ở output directory trước đó.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest \
  Code/tests/test_kitti_camera_video.py \
  Code/tests/test_kitti_sequence_video.py -q
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest \
  Code/tests/test_kitti_sequence_video.py \
  Code/tests/test_kitti_gaussian_heatmap.py \
  Code/tests/test_kitti_overview_prediction.py \
  Code/tests/test_kitti_overview_visualization.py \
  Code/tests/test_kitti_3d_lifting.py -q
```
# Raw points through the actual checkpoint

`export_kitti_pipeline.py` exports a complete training-data demonstration on
one actual KITTI train frame: raw XYZI → Hybrid GT + world transforms → occupied
XY/four-height-band cells → hist14 → backbone/neck tensors → classification
heads → decoded detections. Its output includes individual PNG/SVG panels,
`pipeline_overview.png`, all 14 hist14 channels, eight actual feature channels
per captured layer, full tensors in NPZ and source/checkpoint/augmentation
provenance in `metadata.json`.

```bash
python Code/tools/visualization/export_kitti_pipeline.py \
  --checkpoint /path/to/selected/best.pt \
  --kitti-root Code/Kitti_Raw --frame 005663 --seed 2 --device cpu \
  --output-dir Demo/outputs/pipeline/005663
```

The default donor DB uses 128 real frames from `Code/splits/kitti/train.txt`,
excluding the scene itself and all validation frames. Scans are linked, and
object crops use the repository's existing calibrated label conversion and
GT database builder. It is a documented preview subset, not the original full
training database. The saved checkpoint's sampling probability is respected;
`--force-sampling` explicitly requests a forced preview. No artificial points
are generated. Augmented scenes are fed into `model.eval()` for reproducible
checkpoint activations; this run does not report validation AP.

hist14 is the actual 14-channel encoding name; rich8 is another encoding. Its
encoder directly aggregates points into XY pillars and four height bands.
The voxel picture shows those occupied groups rather than adding a separate
sparse 3D convolution stage. Feature images show the RMS of actual channels
and individual channels, with 99th-percentile display scales; full values are
saved without this normalization. Classification heatmaps retain sigmoid
probabilities in [0,1]. For a BEV checkpoint, the final cuboid view uses the
existing measured-LiDAR Z/height postprocess; a native 3D checkpoint supplies
its own Z/height. The 3D panels use the fixed ROI recorded in metadata.

### Raw pointcloud colored by height

```bash
python Code/tools/visualization/export_kitti_height_pointcloud.py --frame 005663
```

Writes separate BEV and orthographic 3D PNG/SVG files under
`Demo/outputs/pointcloud_height/005663`. Both use `turbo` colors from measured
LiDAR Z in metres, with a shared height scale. Both images fill the canvas on a
black background without axes, titles, colorbars or explanatory text. Default
height display limits are the 1st
and 99th Z percentiles within the ROI, recorded in `metadata.json`; extremes
are color-clipped without changing or subsampling points. Use
`--height-range -2.3 1.0` to set a fixed physical scale across multiple frames.

### Height background with instance detections and zooms

```bash
python Code/tools/visualization/export_kitti_height_detections.py \
  --prediction-dir Result/artifacts/kitti_overview_model \
  --height-metadata Demo/outputs/pointcloud_height/005663/metadata.json \
  --output-dir Demo/outputs/height_detections/005663
```

Uses the verified raw-frame checkpoint cache from `predict_kitti_overview.py`.
Rejects mismatched scan hashes, config hashes or prediction rows. Each prediction
has its own color and frame-local ID, retained in BEV, 3D and zoom views; these
are not tracking IDs. Background points retain the raw render's measured-Z
turbo scale. Box colors also identify actual above-ground foreground support
points; the highest-score box owns any overlapping support.

Outputs both scene views as PNG/SVG, a zoom pair for every visible object under
`zooms/<ID>_<class>/`, measured context/foreground crops with source-point
indices, `object_zooms.png`, and `class_zoom_examples.png`. Vertical geometry
uses the existing LiDAR lifting without GT boxes or class height priors;
unsupported predictions keep their BEV box without inventing a 3D cuboid.

### Simple detection images, matching the earlier detection folders

```bash
python Code/tools/visualization/export_kitti_simple_detections.py \
  --prediction-dir Result/artifacts/kitti_overview_model \
  --height-metadata Demo/outputs/pointcloud_height/005663/metadata.json
```

Writes `detection_bev.png`, `detection_3d.png` and exactly one isolated 3D PNG
per supported visible object under `Demo/outputs/detection_simple/005663/zooms/`.
Images have a black background and shared colors by class (Car coral red,
Pedestrian green, Cyclist yellow), applied to boxes and measured foreground
points consistently across scenes and zooms. They show numeric scores only: no
IDs, class labels, axes, titles, height legends or combined previews. Zoom
points are cropped from the original scan inside that object's cuboid, with
the same yaw-aligned view used by the earlier 3D detection exporter.
Provenance and original scan indices are stored separately in
`Demo/data/simple_detections/005663.json`, so the output folder contains only
images. BEV geometry/scores and the existing LiDAR vertical estimator are reused.

### Camera stitching

```bash
python Code/tools/visualization/stitch_kitti_stereo.py \
  --left Demo/data/images/training/image_2/005663.png \
  --right Demo/data/images/training/image_3/005663.png \
  --output Demo/outputs/camera/005663/stitched.png \
  --metadata Demo/data/images/005663_stitch_metadata.json
```

Aligns camera 3 into camera 2 coordinates using SIFT/RANSAC, selects a graph-cut
seam and feathers a narrow boundary. The image uses real source pixels without
synthetic content or captions. This stereo pair overlaps almost completely;
the resulting field-of-view increase is small, and foreground depth parallax
limits a single homography. The uneven outer edge is cropped to a filled rectangle.
