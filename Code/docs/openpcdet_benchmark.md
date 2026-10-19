# OpenPCDet KITTI validation benchmark

Mở [notebook độc lập](../OpenPCDet_KITTI_Validation_Benchmark.ipynb) trên Colab,
bật GPU rồi sửa `KITTI_ROOT` và `OUTPUT_ROOT`. Chạy các cell từ trên xuống.
Không cần upload helper hoặc clone project MobilePixorNeXt. Notebook tự tạo
runner, clone OpenPCDet tại revision đã ghim và tải năm checkpoint chính thức:
PointPillars, SECOND, PointRCNN, Part-A²-Anchor và PV-RCNN.

Sau cell mount Drive, chạy cell mã thu gọn, cell chuẩn bị dữ liệu rồi cell cài đặt
ngắn. Mở cell thu gọn nếu muốn đọc mã nguồn bình thường.

Metadata đã đọc trực tiếp từ header weights: PointPillars 80/9280, SECOND
80/9280, PointRCNN 77/17864, Part-A²-Anchor 80/12400, PV-RCNN 77/17864
(epoch/iteration). Không giả định iteration của model mới giống model khác.
Đây là checkpoint trong bảng KITTI validation của model zoo; không có manifest
ID train được nhúng trong weights để chứng minh độc lập nguồn gốc từng frame.

`KITTI_ROOT` mặc định là `/content/drive/MyDrive/KITTI_DATASET_ZIP`, cùng Notebook
Robust. Nhận thư mục chứa `velodyne.rar`, `calib.rar`, `label_2.rar`, ZIP/TAR hoặc
một archive tổng, cũng nhận dữ liệu đã giải nén và thư mục lồng nhau.
RAR tự cài `rarfile`/`unrar` trên Colab khi cần; trên workstation cần cài hai công
cụ này. RAR solid/multipart được giải nén theo batch; giữ các volume cạnh nhau.

Cell dữ liệu chỉ chọn đúng 3.769 ID val, bỏ qua `testing/`, kiểm tra dung lượng
theo kích thước giải nén và tạo `/content/kitti_raw/training/{velodyne,calib,label_2}`.
Giữ archive nguồn trên Drive, ghi file tạm rồi đổi tên, xác minh size/CRC khi có;
chạy lại tái sử dụng file đã kiểm tra. Tắt lần copy local thứ hai khi đã giải nén.
Nếu thiếu component hoặc ID, báo rõ trước khi cài môi trường GPU.

Không cần giải nén pixel ảnh: notebook nhúng cùng `configs/datasets/kitti_image_sizes.json`
với Robust, chứa width/height đúng từng frame, camera image_2, không dùng kích thước
cố định. Có thể đặt `IMAGE_SIZES_JSON` riêng (ưu tiên, kiểm tra đủ ID) hoặc tắt
`USE_BUNDLED_IMAGE_SIZES` để giải nén/đọc PNG. Raw inputs có `image_2` vẫn được hỗ trợ.
Hai split được kiểm tra canonical hash, không chia ngẫu nhiên lại. Không cần train
lại hoặc tạo GT database.

OpenPCDet hướng dẫn tải KITTI riêng rồi chuẩn bị dữ liệu; không cần tải lại nếu
đã có đủ archive. Nguồn: [official dataset preparation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#kitti-dataset).

## Môi trường

Linux, Python 3.10–3.13, CUDA PyTorch và nvcc cùng CUDA major. Notebook tạo venv
`--system-site-packages`, giữ Torch của máy chạy và cài dependencies trong venv.
Không cần restart kernel vì model chạy ở process riêng. Các API C++ cũ trong
CUDA extensions được đổi cơ học sang API tương ứng; hash trước/sau lưu trong
`extension_compatibility.json`. Evaluator và model Python upstream giữ nguyên.

Registry `pcdet/datasets/__init__.py` của checkout benchmark chỉ nạp
`DatasetTemplate` và `KittiDataset`. Upstream nạp cả Argoverse 2 và các dataset
khác ngay khi import KITTI, có thể lỗi `No module named av2` dù chỉ chạy KITTI.
Runner bỏ các import/entry registry không dùng, giữ nguyên class KITTI,
sampler/dataloader và evaluator. Hash trước/sau cùng danh sách dataset được bỏ
lưu trong `dataset_import_compatibility.json`; chạy lại giữ nguyên manifest.
Nguồn: [registry upstream tại revision đã ghim](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/__init__.py).

Với phiên Colab đã build bằng notebook cũ, chạy cell mã nhúng của bản mới để
cập nhật `openpcdet_benchmark.py`, rồi chạy lại GPU check bằng `PYTHON` và
`SETTINGS_PATH` hiện tại. Runtime activation tự sửa registry trước khi import
evaluator, không cần build CUDA extensions hay giải nén dữ liệu lại.

Notebook in stdout/stderr của process con trực tiếp trong cell. `setup.log` và
`gpu_check.log` nằm trong thư mục kết quả; các lệnh khác ghi vào
`WORK_DIR/logs/commands.log`. Exception giữ 40 dòng cuối. Vì vậy khi một lệnh
thất bại, đọc traceback bên trong trước `CalledProcessError` để xác định nguyên
nhân, thay vì thay dependencies dựa trên exit code.

Profile mặc định chọn spconv 2.3.8 theo CUDA Torch, từ cu118 đến cu126. CUDA/GPU
mới hơn các wheel này cần kiểm tra thực tế; có thể đặt `SPCONV_PACKAGE` cho wheel
phù hợp. Không có cam kết mọi CUDA/GPU đều được hỗ trợ. Cell GPU check chạy kernel
IoU của cả Torch extension và reference Numba; smoke chạy từng model trước full val.

## Đầu ra

- `comparison.csv`, `comparison.json`, `comparison.md`: điểm R11/R40 BEV/3D Moderate
  theo lớp và macro, tham số, network/final NMS/pipeline timing.
- `full/<model>/report.json`: Easy/Moderate/Hard, mean/P50/P95/P99/FPS tất cả stage,
  cấu hình NMS/ROI/FOV, hash checkpoint/evaluator/split và thông tin GPU/dependencies.
- `full/<model>/kitti_predictions/*.txt`: prediction theo format KITTI, có thể
  chấm lại bằng `tools/kitti_training_pipeline/evaluate_kitti_3d.py --predictions`.
- `latency_per_frame.csv`, config, môi trường, pip freeze và lỗi nếu có.
- `dataset_import_compatibility.json`, `extension_compatibility.json`,
  `setup.log`, `gpu_check.log`: thay đổi tương thích và log chuẩn bị môi trường.
- ZIP chỉ chứa kết quả, không chứa dataset hay trọng số.

GT gốc giữ đủ Van, Person_sitting, DontCare và difficulty; không lọc theo ROI.
AP dùng prediction text đã xuất, độ chính xác 4 số thập phân của exporter upstream.
Full benchmark phải có đủ 3.769 frame. Smoke/partial/error không có điểm trong
bảng chính. Không diễn giải macro AP trên ba lớp thành Car AP.

## Phạm vi timing

Batch 1, FP32, tắt TF32, bỏ 10 frame đầu khỏi timing nhưng giữ chúng trong AP.
Chạy tuần tự, không DataLoader workers/prefetch. Preprocess gồm đọc bin/calib,
FOV/voxelization và collation; H2D/network dùng CUDA events có synchronization;
final NMS và tổng pipeline dùng wall clock với synchronization.

`network` là module_list, bao gồm decode/proposal NMS nội bộ tùy kiến trúc.
`decode_nms` là final post_processing. Không đưa GT vào inference nên không tính
GT recall trong timing. Projection/ghi KITTI/tính AP nằm ngoài tổng pipeline.
FPS là 1000/mean latency batch 1, không phải throughput của pipeline chạy song song.
Tùy chọn chép validation sang ổ local tránh Drive chi phối kết quả; khi so với
Final3D cần dùng cùng loại ổ, GPU, precision và ranh giới đo tương ứng.

Notebook này chưa tích hợp chạy checkpoint Final3D. Các timing BEV cũ không được
tự gán cho Final3D. Giữ nguyên preprocessing/postprocessing riêng của checkpoint;
đây là so sánh các hệ thống pretrained, không phải ablation cùng recipe training.

## Phát triển và kiểm tra

Nguồn runner: `tools/openpcdet_benchmark.py`; dữ liệu: `tools/kitti_notebook_data.py`.
Sau khi sửa runner hoặc shared helper, tái tạo cả hai notebook:

```bash
python3 Code/tools/generate_openpcdet_benchmark_notebook.py
python3 Code/tools/generate_pixor_training_notebook.py
python3 -m pytest Code/tests/test_openpcdet_benchmark.py -q
```

Các kiểm tra CPU xác minh manifest, label/dimension order, thống kê timing,
fingerprint, summary và bản runner được nhúng. Chạy GPU end-to-end cần thực hiện
trên máy chứa dataset và CUDA; việc tạo notebook không tạo số AP/FPS mới.
