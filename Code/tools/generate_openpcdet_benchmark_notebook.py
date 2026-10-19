#!/usr/bin/env python3
"""Generate the standalone notebook, embedding the tested runner verbatim."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {},
              'id': hashlib.sha256(source.encode()).hexdigest()[:12],
              'source': source.strip('\n').splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def multiline_literal(source):
    """Preserve file bytes while showing actual source lines in the notebook."""
    for quote in ("'''", '"""'):
        if quote not in source:
            # Sentinel newlines keep quotes/backslashes at file boundaries safe.
            return 'r' + quote + '\n' + source + '\n' + quote + '[1:-1]'
    raise ValueError('Embedded source contains both triple-quote delimiters')


def embedded_source_cell(title, source):
    item = cell('code', '#@title ' + title + '\n' + source)
    item['metadata'].update(collapsed=True, cellView='form',
                            jupyter={'source_hidden': True}, tags=['hide-input', 'embedded-source'])
    return item


def data_bundle():
    bundle = {'kitti_notebook_data.py': (ROOT / 'tools/kitti_notebook_data.py').read_text(),
              'kitti_training_pipeline/__init__.py': '',
              'kitti_training_pipeline/image_metadata.py': (ROOT / 'tools/kitti_training_pipeline/image_metadata.py').read_text(),
              'metadata/kitti_image_sizes.json': (ROOT / 'configs/datasets/kitti_image_sizes.json').read_text()}
    for name in ('train', 'val'):
        rows = (ROOT / 'splits/kitti' / (name + '.txt')).read_text().splitlines()
        bundle['splits/' + name + '.txt'] = '\n'.join(row.split(';')[0].strip() for row in rows if row.strip()) + '\n'
    return bundle


def bundle_literal(bundle):
    return 'BUNDLE = {\n' + ''.join('    ' + repr(name) + ': ' + multiline_literal(source) + ',\n'
                                  for name, source in bundle.items()) + '}\n'


def data_preparation_cell(split_names, copy_setting):
    source = '''# Nhận thư mục raw hoặc RAR, ZIP, TAR như Notebook Robust.
sys.path.insert(0, str(WORK_DIR))
from kitti_notebook_data import prepare_dataset
FRAME_IDS = [line.strip() for split_name in SPLIT_NAMES
             for line in (WORK_DIR / 'splits' / (split_name + '.txt')).read_text().splitlines() if line.strip()]
DATA_RESULT = prepare_dataset(KITTI_ROOT, EXTRACT_DIR, FRAME_IDS,
    metadata_path=IMAGE_SIZES_JSON,
    fallback_metadata=WORK_DIR / 'metadata/kitti_image_sizes.json' if USE_BUNDLED_IMAGE_SIZES else None,
    allow_install=ON_COLAB)
KITTI_ROOT = DATA_RESULT['training_root']
SETTINGS.update(kitti_root=KITTI_ROOT, image_sizes_json=DATA_RESULT['image_sizes_json'])
if DATA_RESULT['staged'] or Path(EXTRACT_DIR).expanduser().resolve() in Path(KITTI_ROOT).parents:
    SETTINGS[COPY_SETTING] = False  # Đã ở local SSD; tránh chép thêm một bộ.
SETTINGS_PATH.write_text(json.dumps(SETTINGS, ensure_ascii=False, indent=2))
(WORK_DIR / 'dataset_preparation.json').write_text(json.dumps(DATA_RESULT, ensure_ascii=False, indent=2))
print('KITTI sẵn sàng:', KITTI_ROOT, '| số frame:', len(FRAME_IDS))
'''
    return cell('code', source.replace('SPLIT_NAMES', repr(split_names)).replace('COPY_SETTING', repr(copy_setting)))


def build():
    runner = (ROOT / 'tools/openpcdet_benchmark.py').read_text()
    bundle = {'openpcdet_benchmark.py': runner, **data_bundle()}
    cells = [cell('markdown', '''# KITTI validation — 5 model OpenPCDet

Notebook độc lập: nhận KITTI raw hoặc RAR/ZIP/TAR như Notebook Robust. Bật **Runtime →
Change runtime type → GPU**, sửa đường dẫn ở cell 1 rồi chạy từ trên xuống.

- PointPillars, SECOND, PointRCNN, Part-A²-Anchor và PV-RCNN.
- Tải năm checkpoint chính thức đã kiểm tra metadata; không huấn luyện.
- Xác minh split 3.712 train / 3.769 validation bằng hash của từng danh sách ID.
- Chấm **APBEV và AP3D, R11 và R40**, ba lớp × Easy/Moderate/Hard, GT nguyên bản.
- Đo preprocessing, H2D, network, final NMS và tổng pipeline; xuất mean/P50/P95/P99/FPS.
- Kết quả và lỗi được lưu riêng từng model. Smoke hoặc chạy thiếu frame không được
  đưa vào bảng kết quả benchmark đầy đủ.

OpenPCDet được ghim tại `233f849829b6ac19afb8af8837a0246890908755`. Mỗi checkpoint
giữ nguyên ROI, voxel size, FOV, anchors và cấu hình NMS của nó.
'''), cell('markdown', '''## 1. Cấu hình

`KITTI_ROOT` nhận thư mục RAR/ZIP/TAR, một file archive hoặc KITTI đã giải nén.
Mặc định là `/content/drive/MyDrive/KITTI_DATASET_ZIP`, giống Notebook Robust.
Không cần KITTI test ẩn hoặc GT database. Metadata kích thước ảnh từng frame
được nhúng từ cùng file của Robust; không cần pixel ảnh cho LiDAR inference.
`OUTPUT_ROOT` nên nằm trên Drive để giữ kết quả khi Colab ngắt kết nối.

Tùy chọn chép validation sang ổ local giúp giảm ảnh hưởng I/O Drive. Khi so với
model của bạn, hãy dùng cùng loại ổ lưu trữ, GPU, precision và phạm vi đo.
'''), cell('code', '''from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import shlex
from collections import deque

ON_COLAB = 'COLAB_RELEASE_TAG' in os.environ or Path('/content').is_dir()
MOUNT_DRIVE = ON_COLAB
KITTI_ROOT = '/content/drive/MyDrive/KITTI_DATASET_ZIP'  # Thư mục RAR/ZIP/TAR hoặc KITTI raw
OUTPUT_ROOT = '/content/drive/MyDrive/lidar_artifacts/openpcdet'  # SỬA nếu cần
WORK_DIR = '/content/openpcdet_kitti_benchmark' if ON_COLAB else str(Path.cwd() / 'artifacts/openpcdet_work')
EXTRACT_DIR = '/content/kitti_raw' if ON_COLAB else str(Path(WORK_DIR).parent / 'kitti_raw')
IMAGE_SIZES_JSON = None  # Có thể dùng metadata riêng; None => tự tìm như Robust
USE_BUNDLED_IMAGE_SIZES = True  # Cùng metadata image_2 theo từng ID của Notebook Robust
MODELS = ['pointpillars', 'second', 'pointrcnn', 'parta2_anchor', 'pv_rcnn']
GPU_INDEX = '0'
COPY_VAL_TO_LOCAL = True
WARMUP_FRAMES = 10
SMOKE_FRAMES = 3
MAX_FRAMES = None  # None = đầy đủ 3.769; số nhỏ chỉ dùng debug, không có điểm trong bảng chính
RUN_FULL = True
RUN_NAME = 'kitti_val_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
BUILD_JOBS = 2  # Giảm còn 1 nếu build thiếu RAM
SPCONV_PACKAGE = None  # Tự chọn wheel; có thể đặt ví dụ 'spconv-cu126==2.3.8'
CHECKPOINT_OVERRIDES = {}  # Tải tay đúng file chính thức nếu Drive hết quota: {'second': '/path/second_7862.pth'}

WORK_DIR = Path(WORK_DIR).expanduser().resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = WORK_DIR / 'notebook_settings.json'
SETTINGS = dict(kitti_root=KITTI_ROOT, output_root=OUTPUT_ROOT, work_dir=str(WORK_DIR),
    models=MODELS, run_name=RUN_NAME, warmup_frames=WARMUP_FRAMES,
    smoke_frames=SMOKE_FRAMES, max_frames=MAX_FRAMES, build_jobs=BUILD_JOBS,
    spconv_package=SPCONV_PACKAGE, checkpoint_overrides=CHECKPOINT_OVERRIDES,
    copy_val_to_local=COPY_VAL_TO_LOCAL, image_sizes_json=IMAGE_SIZES_JSON)
SETTINGS_PATH.write_text(json.dumps(SETTINGS, ensure_ascii=False, indent=2))
PROCESS_ENV = os.environ.copy()
PROCESS_ENV['CUDA_VISIBLE_DEVICES'] = GPU_INDEX
PROCESS_ENV.pop('CUDA_LAUNCH_BLOCKING', None)
PROCESS_ENV['PYTHONUNBUFFERED'] = '1'

def execute(command, *, log=None, stdout=None, **kwargs):
    command = list(map(str, command))
    heading = '+ ' + shlex.join(command)
    print(heading, flush=True)
    log_path = Path(log) if log is not None else WORK_DIR / 'logs/commands.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=40)
    with log_path.open('a') as transcript:
        transcript.write(heading + '\\n')
        transcript.flush()
        process = subprocess.Popen(command, env=PROCESS_ENV, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, **kwargs)
        try:
            for line in process.stdout:
                tail.append(line)
                transcript.write(line)
                transcript.flush()
                if stdout is None:
                    print(line, end='', flush=True)
                else:
                    stdout.write(line)
            returncode = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
        transcript.write(f'[exit code: {returncode}]\\n')
    if returncode:
        error = subprocess.CalledProcessError(returncode, command, output=''.join(tail))
        if hasattr(error, 'add_note'):
            error.add_note(f'Log: {log_path}\\n' + ''.join(tail))
        raise error
    return subprocess.CompletedProcess(command, returncode)

print('Run:', RUN_NAME)
print('Settings:', SETTINGS_PATH)
'''), cell('code', '''if MOUNT_DRIVE:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    else:
        print('Drive đã mount; dùng kết nối hiện tại.')
print('Nguồn KITTI:', KITTI_ROOT)
'''), cell('markdown', '''## 2. Chuẩn bị dữ liệu và cài môi trường

Nhận `velodyne.rar`, `calib.rar`, `label_2.rar` hoặc ZIP/TAR, cả thư mục lồng nhau.
Trên Colab tự cài rarfile/unrar nếu thiếu. Giải nén đúng 3.769 ID val vào SSD local,
bỏ qua hidden test, kiểm tra dung lượng và tái sử dụng file đã xác minh khi chạy lại.
Archive nguồn được giữ nguyên trên Drive. Không chép thêm một bộ sau khi đã giải nén.

Metadata image_2 theo từng ID được nhúng từ Notebook Robust thay cho pixel ảnh.
Đặt `IMAGE_SIZES_JSON` riêng hoặc tắt `USE_BUNDLED_IMAGE_SIZES` để dùng PNG.

Cell này tạo một virtualenv kế thừa PyTorch CUDA hiện có, cài dependencies vào
virtualenv và build CUDA extensions. Kernel notebook không cần restart.
Cần Linux, Python 3.10–3.13, `nvcc`, CUDA PyTorch và GPU. CUDA toolkit và Torch
phải cùng major version. Lần đầu có thể mất vài phút.

Runner sửa cơ học các API C++ cũ trong CUDA extensions để tương thích Torch
mới; ghi từng file/hash vào `extension_compatibility.json`. Registry dataset chỉ
nạp KITTI để tránh phụ thuộc SDK Argoverse/Waymo khác; lưu hash trước/sau trong
`dataset_import_compatibility.json`. Mã KITTI, evaluator và model Python giữ nguyên.
Các wheel spconv dùng bản 2.3.8; GPU/CUDA rất mới có thể cần
wheel tương thích do bạn cài riêng. Cell kiểm tra CUDA và smoke sẽ dừng rõ nếu
môi trường chưa tương thích, trước khi chạy toàn bộ validation.
Chạy cell mã thu gọn, cell dữ liệu, rồi cell cài môi trường. Bạn chỉ cần sửa đường
dẫn ở phần 1; mã nhúng vẫn có thể mở ra để đọc.
'''), embedded_source_cell('Chuẩn bị mã OpenPCDet, split và bộ đọc archive (chạy một lần)',
    bundle_literal(bundle) + '''
for relative, source in BUNDLE.items():
    destination = WORK_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
RUNNER_SOURCE = BUNDLE['openpcdet_benchmark.py']
RUNNER = WORK_DIR / 'openpcdet_benchmark.py'
print('Đã chuẩn bị runner:', RUNNER)
'''), data_preparation_cell(['val'], 'copy_val_to_local'),
cell('code', '''# Cài dependencies và build CUDA extensions sau khi KITTI sẵn sàng.
RUN_OUTPUT = Path(OUTPUT_ROOT).expanduser().resolve() / RUN_NAME
RUN_OUTPUT.mkdir(parents=True, exist_ok=True)
execute([sys.executable, RUNNER, 'setup', '--settings', SETTINGS_PATH], log=RUN_OUTPUT / 'setup.log')
PYTHON = WORK_DIR / 'env/bin/python'
(RUN_OUTPUT / 'settings.json').write_text(SETTINGS_PATH.read_text())
(RUN_OUTPUT / 'runner.py').write_text(RUNNER_SOURCE)
import shutil
shutil.copyfile(WORK_DIR / 'extension_compatibility.json', RUN_OUTPUT / 'extension_compatibility.json')
shutil.copyfile(WORK_DIR / 'dataset_import_compatibility.json', RUN_OUTPUT / 'dataset_import_compatibility.json')
shutil.copyfile(WORK_DIR / 'setup.json', RUN_OUTPUT / 'setup.json')
shutil.copyfile(WORK_DIR / 'dataset_preparation.json', RUN_OUTPUT / 'dataset_preparation.json')
'''), cell('code', '''# Kiểm tra cả CUDA extensions và kernel rotated IoU của reference evaluator.
execute([PYTHON, RUNNER, 'gpu-check', '--settings', SETTINGS_PATH], log=RUN_OUTPUT / 'gpu_check.log')
# Lưu dependencies đầy đủ để tái lập môi trường.
with (RUN_OUTPUT / 'pip_freeze.txt').open('w') as stream:
    execute([PYTHON, '-m', 'pip', 'freeze'], stdout=stream)
'''), cell('markdown', '''## 3. Kiểm tra dữ liệu và tải checkpoint

Chỉ đọc/tùy chọn chép validation; không thay đổi dataset gốc. GT dùng nguyên bản,
bao gồm Van, Person_sitting và DontCare. Không lọc GT theo ROI của từng model.
Metadata checkpoint được kiểm tra: PointPillars 80/9280, SECOND 80/9280,
PointRCNN 77/17864, Part-A²-Anchor 80/12400, PV-RCNN 77/17864 (epoch / iteration).
Đây là các checkpoint của bảng KITTI validation. Không có danh sách ID huấn luyện
nhúng trong weights; metadata không chứng minh độc lập từng frame đã dùng để train.
'''), cell('code', '''execute([PYTHON, RUNNER, 'prepare', '--settings', SETTINGS_PATH])
downloaded = []
for model_name in MODELS:
    try:
        execute([PYTHON, RUNNER, 'download', '--settings', SETTINGS_PATH, '--model', model_name])
        downloaded.append(model_name)
    except subprocess.CalledProcessError:
        print(f'{model_name}: tải/kiểm tra checkpoint thất bại. Xem lỗi phía trên; các model khác vẫn tiếp tục.')
print('Checkpoint sẵn sàng:', downloaded)
'''), cell('markdown', '''## 4. Smoke trước khi chạy đầy đủ

Mỗi model chạy trong một process riêng để giải phóng GPU khi kết thúc.
Smoke kiểm tra load toàn bộ trọng số, preprocessing, inference, xuất KITTI và
AP reference trên vài frame. Kết quả smoke được đánh dấu riêng, không dùng trong bài.
'''), cell('code', '''smoke_passed = []
for model_name in downloaded:
    try:
        execute([PYTHON, RUNNER, 'run', '--settings', SETTINGS_PATH,
                 '--model', model_name, '--phase', 'smoke'])
        smoke_passed.append(model_name)
    except subprocess.CalledProcessError:
        print(f'{model_name}: smoke thất bại; xem {RUN_OUTPUT / "smoke" / model_name / "failure.json"}')
print('Smoke đạt:', smoke_passed)
'''), cell('markdown', '''## 5. Chạy KITTI validation và đo tốc độ

Default: 3.769 frame, batch 1 FP32, tắt TF32, loại 10 frame đầu **chỉ khỏi timing**.
AP vẫn dùng toàn bộ frame. Mỗi model giữ config gốc, bao gồm score/NMS.

`network` đo `module_list`: có thể chứa decode nội bộ và proposal NMS của PV-RCNN.
`decode_nms` đo final postprocessing. Tổng pipeline từ đọc bin/calib và FOV/
voxelization đến detection, chạy tuần tự. Không tính GT recall, projection sang
KITTI, ghi file, hoặc tính AP. FPS = 1000 / mean latency khi batch=1.

AP tính từ prediction `.txt` đã xuất (format gốc 4 chữ số thập phân), bằng chính
reference evaluator được ghim. Bảng chính dùng R40, vẫn lưu đầy đủ R11.
'''), cell('code', '''if RUN_FULL:
    for model_name in smoke_passed:
        try:
            execute([PYTHON, RUNNER, 'run', '--settings', SETTINGS_PATH,
                     '--model', model_name, '--phase', 'full'])
        except subprocess.CalledProcessError:
            print(f'{model_name}: run thất bại; xem {RUN_OUTPUT / "full" / model_name / "failure.json"}')
else:
    print('RUN_FULL=False: chỉ chạy smoke, chưa có benchmark đầy đủ.')
'''), cell('markdown', '''## 6. Bảng kết quả và file để tải về

`comparison.csv`: tham số, R11/R40 BEV/3D Moderate theo từng lớp và macro,
network/final NMS/pipeline latency và FPS. Mỗi `report.json` có đủ Easy/Moderate/Hard,
mọi stage, phân vị latency, môi trường và hash config/checkpoint/split/evaluator.
`latency_per_frame.csv` lưu mẫu timing; `kitti_predictions/` dùng được với evaluator
reference trong project của bạn. Run thiếu frame, smoke hoặc lỗi có ô điểm trống.

Nếu so trực tiếp với Final3D, hãy đo lại Final3D với cùng protocol timing; không
gán latency của một run BEV cũ cho model 3D. Giữ riêng khác biệt về ROI/FOV/recipe
huấn luyện trong phần mô tả thí nghiệm.
'''), cell('code', '''execute([PYTHON, RUNNER, 'summarize', '--settings', SETTINGS_PATH])
from IPython.display import Markdown, display
display(Markdown((RUN_OUTPUT / 'comparison.md').read_text()))
print('Tất cả kết quả:', RUN_OUTPUT)
# Chỉ nén kết quả của lượt này, không đóng gói dataset/checkpoints.
archive = shutil.make_archive(str(RUN_OUTPUT.parent / (RUN_NAME + '_results')), 'zip', RUN_OUTPUT)
print('ZIP:', archive)
'''), cell('markdown', '''### Xử lý lỗi thường gặp

- `CalledProcessError`: lỗi gốc được in ngay phía trên; xem thêm `setup.log` hoặc
  `gpu_check.log` trong thư mục kết quả, các lệnh khác ở `WORK_DIR/logs/commands.log`.
- `No module named av2` từ bản notebook cũ: chạy cell mã nhúng của bản mới để
  cập nhật runner, rồi chạy lại GPU check bằng môi trường đã build.

- Không thấy GPU: bật GPU runtime trước khi chạy cell 2.
- Thiếu `nvcc` hoặc CUDA major không khớp: dùng CUDA toolkit/PyTorch tương thích;
  notebook giữ nguyên Torch đang có, không tự thay CUDA driver.
- Build bị kill: đặt `BUILD_JOBS=1`, chạy lại từ cell cấu hình.
- Google Drive quota: tải đúng checkpoint từ Model Zoo rồi điền `CHECKPOINT_OVERRIDES`.
- Thiếu PNG/calib/labels: `KITTI_ROOT` phải là KITTI object detection gốc, không
  phải thư mục point cloud đã mã hóa/processed của model khác.
- Hết ổ local: tắt `COPY_VAL_TO_LOCAL` hoặc chọn ổ lớn hơn; ghi rõ loại ổ khi báo FPS.
- Đổi dữ liệu/config: dùng `RUN_NAME` mới và chạy lại cell kiểm tra dữ liệu.

Nguồn: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet),
[Model Zoo](https://github.com/open-mmlab/OpenPCDet#model-zoo),
[spconv](https://github.com/traveller59/spconv).
''')]
    return dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[]),
                kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                language_info=dict(name='python')), nbformat=4, nbformat_minor=5)


if __name__ == '__main__':
    destination = ROOT / 'OpenPCDet_KITTI_Validation_Benchmark.ipynb'
    destination.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + '\n')
    print(destination)
