# PIXOR KITTI: train 50 epoch

Mở [notebook độc lập](../PIXOR_KITTI_Train_50Epochs.ipynb) trên Colab GPU hoặc
Jupyter Linux có CUDA PyTorch, sửa `KITTI_ROOT` và `OUTPUT_ROOT`. Chạy từ trên xuống.
Notebook nhúng runner, helper, hai manifest và reference evaluator; không cần
upload repo MobilePixorNeXt, clone implementation PIXOR hoặc tải weights ban đầu.

Sau mount Drive, chạy cell mã/split thu gọn, cell giải nén dữ liệu rồi cell cài đặt
ngắn. Mã nhúng hiển thị xuống dòng bình thường khi mở ra.

## Dữ liệu và split

Cần KITTI object detection gốc cho toàn bộ 7481 frame. `KITTI_ROOT` mặc định nhận
thư mục `/content/drive/MyDrive/KITTI_DATASET_ZIP` như Robust: có thể chứa RAR/ZIP/TAR
riêng cho velodyne/calib/label_2, archive tổng hoặc các thư mục đã giải nén.
Cell dữ liệu chuẩn hóa vào `/content/kitti_raw/training`, hỗ trợ thư mục lồng nhau,
RAR solid/multipart (giữ đủ các volume cạnh nhau), bỏ qua hidden test và kiểm tra
dung lượng theo số byte giải nén thực. RAR tự cài công cụ trên Colab khi thiếu.
Archive gốc giữ trên Drive, chạy lại tái sử dụng file đã xác minh.

Danh sách ID 3712 train / 3769 val giống manifests trong repo
và OpenPCDet; hash bắt buộc khớp, không chia lại ngẫu nhiên. Validation IDs không
được dùng tạo training target, augmentation hoặc thống kê projection.

Mặc định nhúng cùng metadata `configs/datasets/kitti_image_sizes.json` với Robust,
width/height theo từng frame và camera image_2; không cần giải nén pixel ảnh.
Có thể đặt `IMAGE_SIZES_JSON` riêng hoặc tắt `USE_BUNDLED_IMAGE_SIZES` để dùng PNG.
Không giả định một kích thước ảnh chung. Bin/calib/labels đã giải nén ở local sẽ
không bị copy thêm lần nữa. Nếu đọc raw trên Drive, tùy chọn copy local còn hiệu lực.
Point cloud được đọc lazy theo frame; không đưa pixel ảnh vào model.

## Mô hình và recipe tái lập

Đây là **PIXOR reproduction, Car/BEV**, không phải weights chính thức hoặc một
bản tái tạo số AP của tác giả. Tham khảo [PIXOR CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.html)
và [implementation độc lập](https://github.com/philip-huang/PIXOR).

- ROI LiDAR x=[0,70), y=[-40,40), z=[-2.5,1), resolution 0.1 m.
- Input 35 occupancy height slices + max reflectance: 36×800×700.
- Bottleneck stages [3,6,6,3], planes [24,48,64,96], top-down deconvolution tới
  stride 4, header bốn Conv/BN/ReLU, output 1 classification logit + 6 geometry.
- Geometry: cos/sin yaw, dx/dy, log width/length. Fixed normalization không học từ
  validation: mean=[0,0,0,0,log(1.6),log(3.9)], scale=[1,1,3,3,1,1].
- Positive footprint 0.7×box, ignore border tới 1.2×box; Van footprints được ignore.
  FOV và DontCare được mask bằng projection ground plane z=-1.73 m, có inverse
  augmentation trước khi project. Đây là approximation của recipe, ghi rõ khi báo cáo.
- Focal loss alpha=.25/gamma=2 trên logits, Smooth-L1 trên positive geometry.
- Augmentation chỉ train: flip y, rotation ±5°, XY scale .95–1.05, XY translation ±.25 m.
- Tổng 50 epoch; SGD lr=.01, momentum=.9, weight decay=1e-4, lr×.1 tại 30/40.
- Batch 2, accumulation 4 (effective batch 8), AMP train; seed 42. Nếu OOM, đổi
  batch 1/accumulation 8 trước khi bắt đầu. Inference/profiling luôn FP32 batch 1.

Smoke dùng cùng precision với train. FP16 có thể overflow gradient khi GradScaler
khởi tạo ở loss scale 65536 dù forward/loss hữu hạn. Smoke bỏ qua optimizer step
đó, giảm scale và chạy lại cùng batch tối đa 16 lần; chỉ báo `Smoke OK` sau khi
gradient hữu hạn và optimizer thực sự cập nhật. Không clip gradient NaN/Inf để
tiếp tục hoặc bỏ qua lỗi FP32. Loss NaN/Inf dừng ngay; overflow kéo dài hết số lần
thử cũng dừng. Có thể đặt `AMP=False` rồi chạy lại cell cấu hình để chẩn đoán FP32.
Cảnh báo Numba grid size 1 khi kiểm tra reference IoU chỉ báo ít việc trên GPU.

Recipe có khác biệt với thí nghiệm gốc (số epoch, normalization, reflectance,
header activations, assignment và augmentation); tên báo cáo nên là
**PIXOR (our reproduction, 50 epochs)**. Giữ `config.resolved.json` cùng kết quả.

## Checkpoint, validation và resume

`last.pth` lưu sau mỗi training epoch, gồm model/optimizer/scheduler/AMP scaler,
RNG, lịch sử và identity của data/config/code. Trước validation dài có checkpoint
đánh dấu pending; resume hoàn thành validation này trước khi sang epoch kế tiếp.
`best.pth` chọn bằng **full val Car Moderate APBEV R40** mỗi 5 epoch và epoch cuối.
`epoch_005.pth` ... `epoch_050.pth` giữ các mốc mặc định.

Để resume sau khi Colab ngắt: giữ output, `RUN_NAME`, recipe và `RESUME='auto'`,
chạy lại từ đầu. `EPOCHS=50` là tổng số epoch. Thay đổi model, batch, data hoặc
training recipe sẽ bị từ chối resume để tránh trộn thí nghiệm. Số workers/ổ local
có thể đổi; reproducibility tuyệt đối vẫn phụ thuộc runtime và GPU.

Nếu bị ngắt giữa epoch, runner tiếp tục từ checkpoint cuối epoch trước, không
resume giữa batch. Nếu ngắt trong validation, runner chạy lại validation epoch đó.

## Metric và projection

Chỉ báo **Car APBEV@0.7, R11/R40, Easy/Moderate/Hard**, GT nguyên bản đủ 3769 val.
Evaluator `eval.py`/`rotate_iou.py` nhúng nguyên bản từ OpenPCDet revision
`233f849829b6ac19afb8af8837a0246890908755`, SHA-256 kiểm tra trước mỗi lần chấm.
Giấy phép Apache 2.0 và nguồn đi kèm bundle.

PIXOR không có z/height head. Để xuất KITTI text và bbox camera, runner dùng
median height và bottom-z của Car **train only**. Prior này ảnh hưởng projection
và filter difficulty của prediction; cần giữ ghi chú này khi so APBEV. Không gọi
đó là AP3D hoặc macro ba lớp. `bev_predictions_lidar.json` giữ footprint trước
projection, `kitti_predictions/*.txt` giữ output thực sự dùng để chấm.

Validation đã dùng chọn checkpoint; đây không phải KITTI hidden-test result.

## Latency và output

FP32 batch 1, tắt TF32, bỏ 10 frame timing đầu nhưng vẫn tính đủ AP. Preprocess
gồm bin/calib/occupancy/FOV/tensor; image shapes đã cache. H2D/network bằng CUDA
events, decode+rotated NMS CPU OpenCV gồm D2H, tổng bằng wall clock đồng bộ.
Projection, file export và AP ở ngoài timing. FPS=1000/mean_ms, không phải
throughput DataLoader nhiều worker. So sánh GPU/ổ đĩa/precision và đọc khác biệt
preprocessing trước khi ghép với baseline OpenPCDet.

Output: config, data audit/manifests, environment/pip freeze, history CSV/JSON,
logs, checkpoints, validation theo epoch, final evaluation R40/R11, per-frame
timing/mean/P50/P95/P99/FPS và comparison Car/BEV CSV. ZIP báo cáo bỏ checkpoints;
giữ `last.pth`/`best.pth` riêng trên Drive.

## Phát triển và verification

```bash
python3 Code/tools/generate_openpcdet_benchmark_notebook.py
python3 Code/tools/generate_pixor_training_notebook.py
python3 -m pytest Code/tests/test_openpcdet_benchmark.py Code/tests/test_pixor_training.py -q
python3 -m pytest Code/tests/test_kitti_notebook_data.py -q
```

Test synthetic kiểm tra occupancy, calibration/yaw/dimension, target/decode,
reflection/NMS, split/bundle và resume identity. Khi có Torch, chạy thêm full
backbone forward/backward trên input nhỏ, smoke FP16 overflow/retry với GradScaler
thật trên CPU, gradient FP32 không hữu hạn và vòng train/resume synthetic trên CPU.
Smoke GPU/full dataset/50-epoch training cần chạy trên máy người dùng; tạo notebook
không tạo ra số AP/FPS đo thực tế.
