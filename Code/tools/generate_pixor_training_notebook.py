#!/usr/bin/env python3
"""Generate a standalone PIXOR notebook with code, exact splits and evaluator."""
import json
from pathlib import Path

from generate_openpcdet_benchmark_notebook import cell, embedded_source_cell, data_bundle, bundle_literal, data_preparation_cell

ROOT = Path(__file__).resolve().parents[1]


def build():
    reference = ROOT / 'tools/kitti_training_pipeline/kitti_reference'
    bundle = {**data_bundle(), **{name: (ROOT / 'tools' / name).read_text() for name in ('pixor_training.py', 'openpcdet_benchmark.py')}}
    for name in ('eval.py', 'rotate_iou.py', '__init__.py', 'LICENSE.OpenPCDet'):
        bundle['kitti_reference/' + name] = (reference / name).read_text()
    bundle['kitti_reference/NOTICE.txt'] = '''Reference evaluator from OpenPCDet,
revision 233f849829b6ac19afb8af8837a0246890908755.
https://github.com/open-mmlab/OpenPCDet
OpenPCDet is distributed under Apache License 2.0:
https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/LICENSE
The two evaluator files are embedded unchanged; their hashes are checked before scoring.
'''
    for name in ('train', 'val'):
        rows = (ROOT / 'splits/kitti' / (name + '.txt')).read_text().splitlines()
        bundle['splits/' + name + '.txt'] = '\n'.join(row.split(';')[0].strip() for row in rows if row.strip()) + '\n'
    cells = [cell('markdown', '''# PIXOR — train KITTI 50 epoch, split 3.712 / 3.769

Notebook độc lập: nhận KITTI raw hoặc RAR/ZIP/TAR như Notebook Robust.
Bật GPU, sửa hai đường dẫn rồi chạy từ trên xuống. Train **từ đầu, tổng 50 epoch**,
giữ nguyên ID train/val đã kiểm tra với OpenPCDet; không chia ngẫu nhiên lại.

Đây là **PIXOR reproduction — Car / BEV**: input occupancy + reflectance 36 kênh,
ResNet bottleneck `[3,6,6,3]`, top-down fusion và dense classification/regression.
Không phải checkpoint chính thức của tác giả; recipe được ghi đầy đủ trong output.
Không có nhánh z/height, không báo AP3D hoặc macro ba lớp.

- Lưu `last.pth` mỗi epoch, `best.pth` theo Car Moderate R40 và checkpoint mỗi 5 epoch.
- Resume giữ optimizer/scheduler/scaler; `EPOCHS=50` là tổng số epoch, không cộng thêm.
- Chấm Car APBEV@0.7 R40/R11 trên toàn bộ 3.769 val bằng reference evaluator đã ghim.
- Đo FP32 batch 1 trên GPU hiện tại: preprocessing, H2D, network, decode/NMS, pipeline.
- Code, split và evaluator được nhúng sẵn; không cần upload project hoặc cài OpenPCDet.
'''), cell('markdown', '''## 1. Cấu hình

KITTI cần đủ bin/calib/labels của 7.481 frame, có thể nằm trong archive riêng.
Metadata kích thước ảnh theo từng ID được nhúng từ Robust; không cần pixel ảnh.
Đặt output trên Drive để giữ checkpoint khi phiên Colab kết thúc.

Mặc định batch 2 × tích lũy 4 = effective batch 8, AMP khi train. Nếu smoke bị OOM,
đổi batch 1 × tích lũy 8 **trước khi train**. Đổi recipe sau khi có checkpoint cần
`RUN_NAME` mới. Giữ `RUN_NAME` cũ và `RESUME='auto'` để tiếp tục một lượt train.
'''), cell('code', '''from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import sys

ON_COLAB = 'COLAB_RELEASE_TAG' in os.environ or Path('/content').is_dir()
MOUNT_DRIVE = ON_COLAB
KITTI_ROOT = '/content/drive/MyDrive/KITTI_DATASET_ZIP'  # Thư mục RAR/ZIP/TAR hoặc KITTI raw
OUTPUT_ROOT = '/content/drive/MyDrive/lidar_artifacts/pixor'  # SỬA nếu cần
WORK_DIR = '/content/pixor_kitti_training' if ON_COLAB else str(Path.cwd() / 'artifacts/pixor_work')
EXTRACT_DIR = '/content/kitti_raw' if ON_COLAB else str(Path(WORK_DIR).parent / 'kitti_raw')
IMAGE_SIZES_JSON = None  # Có thể dùng metadata riêng; None => tự tìm như Robust
USE_BUNDLED_IMAGE_SIZES = True  # Cùng metadata image_2 theo từng ID của Notebook Robust
RUN_NAME = 'pixor_car_bev_50ep_seed42'
GPU_INDEX = '0'
EPOCHS = 50
BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
WORKERS = 2  # Dùng 0 nếu runtime không cho tạo workers
AMP = True
COPY_DATA_TO_LOCAL = True  # Cần khoảng 30 GB cho bin/calib/labels; tự kiểm tra dung lượng
RESUME = 'auto'  # auto => checkpoint last.pth trong RUN_NAME; '' => train mới
EVAL_EVERY = 5
EVAL_CHECKPOINT = 'best'  # Có thể đổi 'last' để đánh giá epoch 50
RUN_TRAIN = True
RUN_EVALUATE = True

WORK_DIR = Path(WORK_DIR).expanduser().resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = WORK_DIR / 'pixor_settings.json'
SETTINGS = dict(kitti_root=KITTI_ROOT, output_root=OUTPUT_ROOT, work_dir=str(WORK_DIR),
    run_name=RUN_NAME, epochs=EPOCHS, batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS,
    workers=WORKERS, amp=AMP, copy_data_to_local=COPY_DATA_TO_LOCAL, resume=RESUME,
    eval_every=EVAL_EVERY, eval_checkpoint=EVAL_CHECKPOINT, image_sizes_json=IMAGE_SIZES_JSON)
SETTINGS_PATH.write_text(json.dumps(SETTINGS, ensure_ascii=False, indent=2))
PROCESS_ENV = os.environ.copy()
PROCESS_ENV['CUDA_VISIBLE_DEVICES'] = GPU_INDEX
PROCESS_ENV.pop('CUDA_LAUNCH_BLOCKING', None)
PROCESS_ENV['PYTHONUNBUFFERED'] = '1'
RUN_OUTPUT = Path(OUTPUT_ROOT).expanduser().resolve() / RUN_NAME

def execute(command, log=None):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    if log is None:
        return subprocess.run(list(map(str, command)), env=PROCESS_ENV, check=True)
    with Path(log).open('a') as stream:
        process = subprocess.Popen(list(map(str, command)), env=PROCESS_ENV, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        result = process.wait()
        if result:
            raise subprocess.CalledProcessError(result, command)

print('Run:', RUN_NAME, '| tổng epoch:', EPOCHS)
'''), cell('code', '''if MOUNT_DRIVE:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    else:
        print('Drive đã mount; dùng kết nối hiện tại.')
RUN_OUTPUT.mkdir(parents=True, exist_ok=True)
print('Nguồn KITTI:', KITTI_ROOT)
'''), cell('markdown', '''## 2. Chuẩn bị dữ liệu và cài môi trường

Nhận RAR/ZIP/TAR hoặc thư mục raw, kể cả cấu trúc lồng nhau như Notebook Robust.
Tự cài bộ đọc RAR trên Colab khi cần; giải nén đủ 7.481 ID train/val vào SSD local,
kiểm tra dung lượng và tái sử dụng file đã xác minh khi chạy lại. Archive Drive
được giữ nguyên; không chép thêm một bộ sau khi đã giải nén. Metadata image_2
nhúng từ Robust thay cho pixel ảnh. Có thể đặt `IMAGE_SIZES_JSON` riêng hoặc tắt
`USE_BUNDLED_IMAGE_SIZES` để dùng PNG.

Cần Linux, Python 3.10–3.13 và PyTorch CUDA có sẵn. Notebook tạo virtualenv kế thừa
Torch hiện tại, cài NumPy/OpenCV/Numba riêng. Không build spconv/C++ extensions.
Numba CUDA được dùng cho rotated IoU của evaluator; smoke kiểm tra kernel trước train.
Chạy cell mã thu gọn, cell dữ liệu, rồi cell cài môi trường. Mã nguồn được xuống
dòng bình thường khi mở cell để đọc.
'''), embedded_source_cell('Chuẩn bị mã PIXOR và split (chạy một lần, có thể thu gọn)',
    bundle_literal(bundle) + '''
for relative, source in BUNDLE.items():
    destination = WORK_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
RUNNER = WORK_DIR / 'pixor_training.py'
print('Đã chuẩn bị runner và split:', WORK_DIR)
'''), data_preparation_cell(['train', 'val'], 'copy_data_to_local'),
cell('code', '''# Cài NumPy/OpenCV/Numba sau khi KITTI sẵn sàng.
execute([sys.executable, RUNNER, 'setup', '--settings', SETTINGS_PATH])
PYTHON = WORK_DIR / 'env/bin/python'
(RUN_OUTPUT / 'notebook_settings.json').write_text(SETTINGS_PATH.read_text())
for name in ('pixor_training.py', 'openpcdet_benchmark.py'):
    (RUN_OUTPUT / name).write_text(BUNDLE[name])
shutil.copyfile(WORK_DIR / 'dataset_preparation.json', RUN_OUTPUT / 'dataset_preparation.json')
'''), cell('markdown', '''## 3. Audit dữ liệu, giữ nguyên split

Kiểm tra hash danh sách ID, trùng lặp/giao nhau, file XYZI/PNG/calibration/labels.
Chỉ training IDs tạo target và thống kê. Validation labels chỉ dùng khi tính AP.
Nếu chép sang ổ local, notebook giữ nguyên dữ liệu nguồn; không cần copy pixel ảnh.

PIXOR không dự đoán chiều cao và z. Khi xuất định dạng KITTI, notebook dùng **median
height/bottom-z của Car trong train** để chiếu hộp lên camera và tạo bbox 2D.
Đây là giả định xuất BEV được ghi trong report; ảnh hưởng projection/difficulty của
prediction, không tạo ra một model 3D. GT validation được giữ nguyên, không cắt ROI.
'''), cell('code', '''execute([PYTHON, RUNNER, 'prepare', '--settings', SETTINGS_PATH], log=RUN_OUTPUT / 'prepare.log')
audit = json.loads((RUN_OUTPUT / 'data_audit.json').read_text())
print({name: len(ids) for name, ids in audit['splits'].items()})
print('Priors chỉ từ train:', audit['projection_prior'])
'''), cell('markdown', '''## 4. Smoke forward/backward trước khi train

Chạy một batch training qua model đầy đủ, loss và optimizer step trong process riêng;
kiểm tra reference IoU CUDA. Smoke không ghi checkpoint và không tính vào 50 epoch.
AMP có thể overflow gradient ở loss scale ban đầu dù loss hữu hạn. Smoke cho
GradScaler giảm scale và thử lại cùng batch tối đa 16 lần; chỉ báo `Smoke OK` sau
một optimizer step thực sự. Loss NaN/Inf hoặc gradient FP32 không hữu hạn vẫn dừng.
'''), cell('code', '''execute([PYTHON, RUNNER, 'smoke', '--settings', SETTINGS_PATH], log=RUN_OUTPUT / 'smoke.log')
'''), cell('markdown', '''## 5. Train tổng 50 epoch

Mặc định SGD lr 0.01, momentum 0.9, weight decay 1e-4; giảm lr tại epoch 30/40.
Focal loss trên logits + Smooth-L1 trên pixel positive; normalized geometry target
cố định. Augmentation chỉ train: phản chiếu y, xoay ±5°, scale 0.95–1.05, tịnh tiến
BEV ±0.25 m. Không dùng GT database hoặc validation để huấn luyện.

Đây là recipe tái lập 50 epoch của notebook; không khẳng định trùng toàn bộ recipe
hoặc tái tạo score công bố của PIXOR gốc. `config.resolved.json` ghi mọi khác biệt.
Mỗi 5 epoch và tại epoch cuối chấm toàn bộ val để chọn `best.pth`; `last.pth` lưu
optimizer/scheduler/scaler để resume. Model tốt nhất có thể ở trước epoch 50.
'''), cell('code', '''if RUN_TRAIN:
    execute([PYTHON, RUNNER, 'train', '--settings', SETTINGS_PATH], log=RUN_OUTPUT / 'train.log')
else:
    print('RUN_TRAIN=False: chưa train; đánh giá cần có checkpoint tương ứng.')
'''), cell('markdown', '''## 6. Đánh giá checkpoint best hoặc last, đo latency/FPS

Chấm đủ 3.769 frame, Car IoU 0.7, Easy/Moderate/Hard, R11/R40. Đo FP32 batch 1,
tắt TF32, bỏ 10 frame đầu khỏi timing nhưng vẫn giữ trong AP. Decode/NMS dùng
CPU OpenCV và bao gồm chuyển prediction về CPU. Preprocess gồm bin/calib, occupancy,
FOV và tạo tensor; kích thước ảnh đã cache. Ghi rõ khác biệt preprocessing khi so
với runner OpenPCDet. Projection/ghi file/tính AP nằm ngoài timing.

Đối chiếu với cột **Car APBEV R40** của năm model OpenPCDet. Không đưa PIXOR vào
cột AP3D hoặc macro Car/Pedestrian/Cyclist. Đây vẫn là kết quả validation được dùng
chọn checkpoint, không phải điểm trên KITTI hidden test.
'''), cell('code', '''if RUN_EVALUATE:
    execute([PYTHON, RUNNER, 'evaluate', '--settings', SETTINGS_PATH], log=RUN_OUTPUT / 'evaluate.log')
    report_path = RUN_OUTPUT / 'evaluation' / Path(EVAL_CHECKPOINT).stem / 'report.json'
    report = json.loads(report_path.read_text())
    print('Checkpoint epoch:', report['epoch'])
    print('Car APBEV@0.7 R40:', report['accuracy']['R40'])
    print('FP32 pipeline:', report['latency']['input_to_detections'])
    print('Report:', report_path)
'''), cell('markdown', '''## 7. Xuất kết quả gọn

Checkpoint đã nằm tại output trên Drive; file ZIP dưới đây chỉ chứa config, manifest,
history, prediction, report và log. `last.pth` / `best.pth` cần giữ riêng để resume.
'''), cell('code', '''# Không đóng gói dataset hoặc checkpoint lớn vào ZIP báo cáo.
import zipfile
archive = RUN_OUTPUT.parent / (RUN_NAME + '_results.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(RUN_OUTPUT.rglob('*')):
        if path.is_file() and 'checkpoints' not in path.relative_to(RUN_OUTPUT).parts:
            zipped.write(path, arcname=path.relative_to(RUN_OUTPUT))
print('ZIP:', archive)
print('Checkpoint:', RUN_OUTPUT / 'checkpoints')
'''), cell('markdown', '''Nguồn kiến trúc: [PIXOR — CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.html).
Tham khảo implementation: [philip-huang/PIXOR](https://github.com/philip-huang/PIXOR).
Split/evaluator: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), revision
`233f849829b6ac19afb8af8837a0246890908755`, nhúng nguyên bản và kiểm tra SHA-256.

Nếu hết bộ nhớ: giảm batch trước khi train; nếu hết ổ local, tắt copy hoặc đổi ổ.
Nếu phiên bị ngắt: mở lại notebook, giữ cùng output/run/recipe và chạy từ đầu;
`RESUME='auto'` sẽ tiếp tục checkpoint epoch gần nhất, không train lại từ epoch 1.
''')]
    return dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[]),
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python')), nbformat=4, nbformat_minor=5)


if __name__ == '__main__':
    destination = ROOT / 'PIXOR_KITTI_Train_50Epochs.ipynb'
    destination.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + '\n')
    print(destination)
