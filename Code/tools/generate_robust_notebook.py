#!/usr/bin/env python3
"""Generate Robust_3D_Lidar_Detection_Pipeline.ipynb."""

import json
from pathlib import Path

def make_markdown_cell(source_text):
    lines = [line + "\n" for line in source_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def make_code_cell(source_code):
    lines = [line + "\n" for line in source_code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def build_notebook():
    cells = []

    # Cell 0: Header Markdown
    cells.append(make_markdown_cell("""# 🚗 Robust 3D LiDAR Object Detection Pipeline (Colab & Local)

Pipeline nghiên cứu và huấn luyện mô hình phát hiện vật thể 3D LiDAR siêu nhẹ (<1M parameters), hỗ trợ kiến trúc **MobilePixorNeXt**, **BEVNeXt**, cơ chế mã hóa **hist14**, **Focal Context**, **Grouped Heads**, các hàm mất mát tiên tiến (**OGA**, **Q-OGA**) và kỹ thuật tăng cường **Hybrid-GT**.

### 📌 Các tính năng chính trong Notebook này:
1. **Thiết lập môi trường linh hoạt**: Tự động nhận diện Google Colab (mount Drive) hoặc Local máy trạm, kiểm tra VRAM và hỗ trợ BF16.
2. **Cấu hình nhanh bằng Cell lệnh**: Thay đổi trực tiếp các siêu tham số, kiến trúc mô hình, hàm mất mát và augmentation ngay trong cell mà không cần sửa file JSON. Tự động kiểm tra ngân sách tham số (<1M).
3. **Quản lý & Tải Dataset KITTI**: Kiểm tra và hỗ trợ giải nén/tải dữ liệu thô (`velodyne`, `label_2`, `calib`, `image_2`).
4. **Tiền xử lý dữ liệu chuẩn (`prepare_kitti.py`)**: Tự động chuyển đổi nhãn sang toạ độ Velodyne và phân chia train/val manifests.
5. **Xây dựng Database cho GT Sampler (`build_gt_database.py`)**: Cắt object crops từ tập train độc quyền cho `hybrid_gt` & `openpcdet_gt`.
6. **Trực quan hóa GT Sampling**: Xem trực quan 3 giai đoạn chèn vật thể vào đám mây điểm trước khi train.
7. **Sanity Smoke Train**: Chạy thử 1-2 batch để kiểm tra forward, backward, loss và optimizer trong <30 giây.
8. **Huấn luyện chính thức & Auto-Resume**: Stream log thời gian thực, tích hợp TensorBoard, lưu trữ song song 2 checkpoint: `best_ap` và `best_loss`.
9. **Đánh giá toàn diện Checkpoint Kép**: Đánh giá độc lập cả 2 model chiến thắng (`best_ap` và `best_loss`) theo chuẩn BEV & 3D, xuất bảng so sánh chi tiết."""))

    # Cell 1: Environment Setup
    cells.append(make_code_cell("""# ==============================================================================
# Cell 1: Environment Setup, Mounting & Dependency Verification
# ==============================================================================
import os
import sys
import math
import json
import shlex
import shutil
import hashlib
import tempfile
from pathlib import Path
import torch

q = shlex.quote

# 1. Detect if running inside Google Colab
IN_COLAB = "google.colab" in sys.modules
GIT_REPO_URL = "https://github.com/danhyoyo/Lidar.git"
GIT_BRANCH = "research/mobilepixornext-under1m"

if IN_COLAB:
    print("Detected Google Colab environment. Mounting Google Drive...")
    from google.colab import drive
    drive.mount("/content/drive")
    DEFAULT_REPO_DIR = Path("/content/Lidar").resolve()
    DRIVE_BASE = Path("/content/drive/MyDrive/lidar_project").resolve()

    # Auto-clone repository if not present on Colab
    if not (DEFAULT_REPO_DIR / "configs/config.json").is_file():
        if (Path.cwd() / "configs/config.json").is_file():
            DEFAULT_REPO_DIR = Path.cwd().resolve()
            print(f"Using repository in current working directory: {DEFAULT_REPO_DIR}")
        elif (Path("/content/drive/MyDrive/Lidar/configs/config.json")).is_file():
            DEFAULT_REPO_DIR = Path("/content/drive/MyDrive/Lidar").resolve()
            print(f"Found repository on Google Drive: {DEFAULT_REPO_DIR}")
        else:
            print(f"Repository not found at {DEFAULT_REPO_DIR}. Cloning branch '{GIT_BRANCH}' from GitHub...")
            if DEFAULT_REPO_DIR.exists():
                shutil.rmtree(DEFAULT_REPO_DIR, ignore_errors=True)
            !git clone -b {q(GIT_BRANCH)} {q(GIT_REPO_URL)} {q(str(DEFAULT_REPO_DIR))}
            if not (DEFAULT_REPO_DIR / "configs/config.json").is_file():
                raise FileNotFoundError(f"Failed to clone repository from {GIT_REPO_URL} into {DEFAULT_REPO_DIR}")
            print(f"✓ Cloned branch '{GIT_BRANCH}' successfully into {DEFAULT_REPO_DIR}")

    print("Verifying requirements on Colab...")
    !pip install -q -r {q(str(DEFAULT_REPO_DIR / "requirements-kitti.txt"))}

    # Thư mục lưu trữ vĩnh viễn trên Drive: Checkpoints, Configs & Logs
    DEFAULT_ARTIFACT_ROOT = DRIVE_BASE / "artifacts"
    # Thư mục chứa dataset trên Google Drive (chứa các file rar/zip: velodyne.rar, calib.rar,...)
    DEFAULT_DRIVE_RAW_DIR = Path("/content/drive/MyDrive/KITTI_DATASET_ZIP").resolve()
    # Thư mục giải nén cục bộ trên Colab (/content/) để đạt tốc độ đọc SSD nhanh nhất
    DEFAULT_RAW_ROOT = Path("/content/kitti_raw").resolve()
    DEFAULT_PROCESSED_ROOT = Path("/content/kitti_processed").resolve()
else:
    print("Running in local workstation environment.")
    DEFAULT_REPO_DIR = Path.cwd().resolve()
    DEFAULT_ARTIFACT_ROOT = (DEFAULT_REPO_DIR / "artifacts").resolve()
    DEFAULT_DRIVE_RAW_DIR = (DEFAULT_REPO_DIR / "data/kitti/raw").resolve()
    DEFAULT_RAW_ROOT = (DEFAULT_REPO_DIR / "data/kitti/raw").resolve()
    DEFAULT_PROCESSED_ROOT = (DEFAULT_REPO_DIR / "data/kitti/processed").resolve()

# 2. Base directory configuration
REPO_DIR = DEFAULT_REPO_DIR
ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT
DRIVE_RAW_DIR = DEFAULT_DRIVE_RAW_DIR
RAW_KITTI_ROOT = DEFAULT_RAW_ROOT
PROCESSED_DATASET_DIR = DEFAULT_PROCESSED_ROOT

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
RAW_KITTI_ROOT.mkdir(parents=True, exist_ok=True)
PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 3. Path & Module Configuration
os.chdir(REPO_DIR)
pipeline_dir = REPO_DIR / "tools/kitti_training_pipeline"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))
os.environ["PYTHONUNBUFFERED"] = "1"

from common import configure_detector_imports
configure_detector_imports(REPO_DIR / "detector")

# 4. Device & Hardware Inspection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working Directory       : {REPO_DIR}")
print(f"Artifacts (Drive)       : {ARTIFACT_ROOT}")
print(f"Dataset Archives (Drive): {DRIVE_RAW_DIR}")
print(f"Local Fast Raw Dir      : {RAW_KITTI_ROOT}")
print(f"Local Processed Dir     : {PROCESSED_DATASET_DIR}")
print(f"Execution Device        : {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"GPU Hardware            : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"BF16 Precision Support  : {'YES' if bf16_supported else 'NO'}")
print("✓ Setup complete.")"""))

    # Cell 2: Markdown Configuration Guide
    cells.append(make_markdown_cell("""## ⚙️ 2. Quick Configuration Control Panel (Bảng điều khiển cấu hình nhanh)

Chỉnh sửa trực tiếp các tham số bên dưới. Cell này sẽ:
- Phân giải cấu hình qua `resolve_notebook_config` từ `configs/config.json`.
- Tự động kiểm tra tính tương thích giữa Model, Attention, Loss và Head mode.
- **Bật/tắt BatchNorm & SiLU ở Detection Head**: Tùy chỉnh trực tiếp qua `HEADER_USE_BN = True/False` và `HEADER_USE_SILU = True/False` (hoặc `"silu"`, `"relu"`, `"none"`).
- Khởi tạo kiến trúc mô hình và thực hiện **Audit Parameter Budget** để đảm bảo mô hình nằm trong ngân sách **< 1M tham số**!
- Lưu cấu hình giải quyết thực tế vào `config.resolved.json` trong thư mục run artifact."""))

    # Cell 3: Code Configuration Panel
    cells.append(make_code_cell("""# ==============================================================================
# Cell 2: Quick Configuration Control Panel
# ==============================================================================
import importlib
import notebook_config
import common
importlib.reload(notebook_config)
importlib.reload(common)
from common import (create_experiment_config, generate_run_name, input_shape,
                    build_model, model_parameter_report, read_json, write_json)
from notebook_config import resolve_notebook_config, save_notebook_config
from tools.kitti_training_pipeline.notebook_workflow import validate_modes

# --- 1. RUN & COMPUTE CONTROLS ---
SEED = 42
PRECISION = "bf16"            # L4/A100: "bf16"; RTX 30/40: "bf16" or "fp16"; GTX: "fp32"
PHYSICAL_BATCH_SIZE = 16      # Giảm xuống 8 hoặc 4 nếu thiếu VRAM
ACCUMULATION_STEPS = 1        # Gradient accumulation (tăng nếu muốn batch size hiệu dụng lớn hơn)
VAL_BATCH_SIZE = None         # None -> 2 * PHYSICAL_BATCH_SIZE
EPOCHS = 50
WARMUP_EPOCHS = 4
LEARNING_RATE = 0.0007        # Chuẩn cho AdamW: 7e-4
NUM_WORKERS = 4               # Số tiến trình nạp dữ liệu
TARGET_BACKEND = "numba"      # "numba" (nhanh hơn) hoặc "python"
COMPILE_MODEL = False         # torch.compile (tùy chọn)

# --- 2. MODEL ARCHITECTURE (<1M BUDGET) ---
PRESET = "custom"             # "custom" | "config" | "UNDER1M_FOCAL_OGA_IQA" | "UNDER1M_FOCAL_ECA_OGA_IQA"
BACKBONE = "mobilepixornext"
BACKBONE_OUT_DIM = 32         # 32 (Focal main) hoặc 16 (legacy)
BEV_ENCODING = "hist14"       # "hist14" (14 kênh v1) | "rich8" | "binary_slices"
BEV_BACKEND = "numpy"         # "numpy" | "numba"
STAGE_DEPTHS = [3, 4, 2]

# Head & Task Grouping
HEAD_MODE = "grouped"         # "grouped" (Car riêng, Ped/Cyc riêng) | "legacy_single"
HEAD_GROUPS = [
    {"name": "car", "classes": ["Car"]},
    {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}
]
GROUP_WEIGHTS = None          # None -> trọng số bằng nhau; hoặc {"car": 1.0, "ped_cyc": 1.0}
BOX_MODE = "bev"              # "bev" (5 params: x, y, w, l, yaw) | "3d" (7 params: thêm z, h)
SCALE_GATED_FPN = True
NECK_TYPE = "scale_gated_fpn" # "scale_gated_fpn" | "rc_sgfpn" | "rc_bisgfpn"
RC_GATE_MODE = "mul"        # mul: range scale; add: range bias (RC necks only)
NECK_FUSION_CHANNELS = 24
DETAIL_PATH = False
USE_REPARAM = False
HEADER_USE_IOU = True         # Bật nhánh dự đoán chất lượng IoU (IQA)

# Head Normalization & Activation Controls
HEADER_USE_BN = True          # True: Bật BatchNorm2d trong Detection Head | False: Tắt (nn.Identity)
HEADER_USE_SILU = True        # True: Bật SiLU activation trong Detection Head | False: Tắt activation ("none")
                              # (Có thể truyền chuỗi: "silu", "relu", "none")

# Attention & Focal Context
C4_CONTEXT = "focal"          # "focal" (main under-1m) | "none"
C4_CONTEXT_BOTTLENECK = 64
C4_CONTEXT_DILATIONS = [1, 2, 3]
C4_CONTEXT_LAYER_SCALE_INIT = 0.001
LOCAL_ATTENTION = "none"      # "none" | "eca" | "simam"
LOCAL_ATTENTION_ECA_KERNEL_SIZE = 3
LOCAL_ATTENTION_SIMAM_LAMBDA = 0.0001
LOCAL_ATTENTION_LAYER_SCALE_INIT = 0.001

# --- 3. LOSS & CRITERION ---
LOSS_NAME = "oga"             # "oga" (Oriented Geometry-Aware) | "q_oga" | "baseline"
IOU_TARGET_TYPE = "mgiou"     # "mgiou" | "yaw_footprint"
IOU_LOSS_WEIGHT = 1.0
VERTICAL_LOSS_WEIGHT = 1.0    # Trọng số nhánh z/height khi BOX_MODE="3d"
OPTIMIZER = "adamw"
WEIGHT_DECAY = 0.001
GRAD_CLIP_NORM = 10.0

# --- 4. AUGMENTATION ---
AUGMENTATION = "hybrid_gt"    # "hybrid_gt" | "openpcdet_gt" | "standard" | "none"
HYBRID_OPTIONS = {
    "PROBABILITY": 0.5,
    "SAMPLE_GROUPS": ["Car:8", "Pedestrian:6", "Cyclist:6"],
    "CACHE_SIZE_MB": 256,
    "GEOMETRY_BACKEND": "auto",
    "PLACEMENT_MODE": "source_relative",
    "ENABLE_GROUND_VALIDATION": True,
    "ENABLE_STATIC_COLLISION": True,
    "ENABLE_LINE_OF_SIGHT": False,
    "ENABLE_SHADOW_MASKING": False,
    "ENABLE_DENSITY_SUBSAMPLE": True,
    "ENABLE_RADIOMETRIC_CALIBRATION": False,
    "ENABLE_VISIBILITY_PROTECTION": True,
}

# --- 5. CHECKPOINT & EVALUATION CONTROLS ---
CHECKPOINT_SELECTION = "ap"   # "ap" (lưu theo max val AP) | "loss" (lưu theo min val loss)
AP_EVERY = 1                  # Tính AP mỗi N epoch
AP_METRIC_MODE = "auto"
EVALUATION_MODES = ["local_bev"] if BOX_MODE == "bev" else ["3d", "bev"]
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.10
MAX_DETECTIONS = 500
NMS_ALPHA = 0.5
PEAK_MODE = "per_class"
TRAIN_SPLIT = "splits/kitti/train.txt"
VAL_SPLIT = "splits/kitti/val.txt"
CUSTOM_RUN_NAME = None
EXPERIMENT_TAG = None

# ==============================================================================
# RESOLVE CONFIGURATION DICTIONARY
# ==============================================================================
CONFIG_BASE = "configs/config.json"

if isinstance(HEADER_USE_SILU, bool):
    resolved_header_act = "silu" if HEADER_USE_SILU else "none"
elif HEADER_USE_SILU is None or str(HEADER_USE_SILU).lower() in ("none", "false", "no", "0"):
    resolved_header_act = "none"
else:
    resolved_header_act = str(HEADER_USE_SILU).lower()

custom_overrides = {
    "model": {
        "backbone": BACKBONE, "backbone_out_dim": BACKBONE_OUT_DIM, "head_mode": HEAD_MODE,
        "cls_encoding": "gaussian", "scale_gated_fpn": SCALE_GATED_FPN, "neck_type": NECK_TYPE,
        "rc_gate_mode": RC_GATE_MODE,
        "use_reparam": USE_REPARAM, "header_use_iou": HEADER_USE_IOU,
        "header_use_bn": bool(HEADER_USE_BN),
        "header_act": resolved_header_act,
        "c4_attention": "none", "c4_attention_scales": [], "c4_attention_qk_norm": "none",
        "deploy": False,
    },
    "loss": {
        "name": LOSS_NAME, "use_iou": HEADER_USE_IOU,
        "iou_target_type": IOU_TARGET_TYPE, "iou_loss_weight": IOU_LOSS_WEIGHT
    },
    "data": {
        "bev_encoding": {
            "name": BEV_ENCODING, "density_norm": 32, "intensity_scale": 1,
            "out_channels": {"rich8": 8, "hist14": 14}.get(BEV_ENCODING),
        }
    },
}

if BACKBONE == "mobilepixornext":
    cm = custom_overrides["model"]
    cm.update(stage_depths=STAGE_DEPTHS, c4_context=C4_CONTEXT,
              local_attention=LOCAL_ATTENTION, neck_fusion_channels=NECK_FUSION_CHANNELS, detail_path=DETAIL_PATH)
    if C4_CONTEXT == "focal":
        cm.update(c4_context_version=1, c4_context_bottleneck=C4_CONTEXT_BOTTLENECK,
                  c4_context_dilations=C4_CONTEXT_DILATIONS, c4_context_layer_scale_init=C4_CONTEXT_LAYER_SCALE_INIT)
    if LOCAL_ATTENTION != "none":
        cm["local_attention_layer_scale_init"] = LOCAL_ATTENTION_LAYER_SCALE_INIT
        if LOCAL_ATTENTION == "eca": cm["local_attention_eca_kernel_size"] = LOCAL_ATTENTION_ECA_KERNEL_SIZE
        if LOCAL_ATTENTION == "simam": cm["local_attention_simam_lambda"] = LOCAL_ATTENTION_SIMAM_LAMBDA

if BEV_ENCODING == "hist14":
    custom_overrides["data"]["bev_encoding"].update(version=1, backend=BEV_BACKEND)

if HEAD_MODE == "grouped":
    custom_overrides["data"]["head_groups"] = HEAD_GROUPS
    if GROUP_WEIGHTS is not None:
        custom_overrides["loss"]["group_weights"] = GROUP_WEIGHTS

effective_val_batch = PHYSICAL_BATCH_SIZE * 2 if VAL_BATCH_SIZE is None else VAL_BATCH_SIZE
runtime_loss = {"vertical_loss_weight": VERTICAL_LOSS_WEIGHT} if BOX_MODE == "3d" else {}

config_dict = resolve_notebook_config(
    REPO_DIR,
    base_path=CONFIG_BASE,
    preset=PRESET,
    custom_overrides=custom_overrides,
    augmentation=AUGMENTATION,
    hybrid_options=HYBRID_OPTIONS,
    runtime_overrides={
        "seed": SEED,
        "data": {
            "box_mode": BOX_MODE,
            "kitti": {"location": str(PROCESSED_DATASET_DIR), "nms_alpha": NMS_ALPHA, "peak_mode": PEAK_MODE}
        },
        "model": {
            "header_use_bn": bool(HEADER_USE_BN),
            "header_act": resolved_header_act,
        },
        "loss": runtime_loss,
        "train": {
            "data": str(TRAIN_SPLIT), "epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
            "learning_rate": LEARNING_RATE, "optimizer": OPTIMIZER, "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM, "num_workers": NUM_WORKERS,
            "physical_batch_size": PHYSICAL_BATCH_SIZE, "accumulation_steps": ACCUMULATION_STEPS,
            "precision": PRECISION, "target_backend": TARGET_BACKEND, "compile_model": COMPILE_MODEL,
            "checkpoint_selection": {"primary": CHECKPOINT_SELECTION, "ap_every": AP_EVERY, "metric_mode": AP_METRIC_MODE}
        },
        "val": {"data": str(VAL_SPLIT), "physical_batch_size": effective_val_batch, "val_every": 1},
        "evaluation": {
            "kitti_root": str(RAW_KITTI_ROOT), "modes": list(EVALUATION_MODES),
            "score_threshold": SCORE_THRESHOLD, "nms_threshold": NMS_THRESHOLD, "max_detections": MAX_DETECTIONS
        },
    },
)

validate_modes(config_dict, EVALUATION_MODES)

extra_tags = [EXPERIMENT_TAG] if EXPERIMENT_TAG else []
if config_dict["model"].get("backbone_out_dim", 16) != 16:
    extra_tags.append(f"h{config_dict['model']['backbone_out_dim']}")
RUN_NAME = CUSTOM_RUN_NAME or generate_run_name(config_dict, seed=SEED, extra_tags=extra_tags)
VARIANT = PRESET if PRESET not in ("custom", "config") else RUN_NAME
RUN_DIR = ARTIFACT_ROOT / RUN_NAME
CONFIG = save_notebook_config(RUN_DIR, config_dict)

print("=" * 80)
print(f"🎯 EXPERIMENT RUN: {RUN_NAME}")
print(f"Config Snapshot : {CONFIG}")
print(f"BEV Encoding    : {config_dict['data']['bev_encoding']['name']} ({BEV_BACKEND}) | Input shape: {input_shape(config_dict)}")
print(f"Box Mode        : {BOX_MODE} | Loss: {LOSS_NAME} | Augmentation: {AUGMENTATION}")
print(f"Head Config     : BatchNorm={'ON' if HEADER_USE_BN else 'OFF'} | Activation={resolved_header_act}")
print(f"Precision       : {PRECISION} | Batch Size: {PHYSICAL_BATCH_SIZE} x {ACCUMULATION_STEPS} | Epochs: {EPOCHS}")
print("=" * 80)

# Model Parameter Budget Audit
model_inst = build_model(config_dict)
param_rep = model_parameter_report(model_inst)
counts = param_rep.get("parameter_counts", {})
total_p = counts.get("total_detector", 0)
backbone_p = counts.get("backbone_including_neck", 0)
heads_p = counts.get("heads", 0)

print(f"📊 PARAMETER AUDIT:")
print(f"   - Total Parameters   : {total_p:,}")
print(f"   - Backbone + Neck    : {backbone_p:,}")
print(f"   - Detection Heads    : {heads_p:,}")
if total_p <= 1_000_000:
    print(f"   👉 Budget Check: PASS (< 1M parameters)")
else:
    print(f"   ⚠️ WARNING: Exceeds 1M budget by {total_p - 1_000_000:,} parameters!")
print("=" * 80)"""))

    # Cell 4: Markdown Data Preparation
    cells.append(make_markdown_cell("""## 📦 3. Giải nén Dataset KITTI từ Google Drive vào Colab Local Disk

Để huấn luyện đạt tốc độ đọc I/O cao nhất (tránh hoàn toàn nghẽn mạng do FUSE mount của Google Drive), chúng ta lưu giữ an toàn các file nén (`.rar`, `.zip` hoặc `.tar`) trên Google Drive và **giải nén trực tiếp vào ổ đĩa SSD cục bộ của Colab** (`/content/kitti_raw`).

- Tự động cài đặt tiện ích `unrar` nếu Colab chưa có.
- Tự động quét file `.rar`/`.zip` trong thư mục: `DRIVE_DATASET_PATH` (mặc định: `/content/drive/MyDrive/KITTI_DATASET_ZIP`).
- Giải nén siêu tốc vào `RAW_KITTI_ROOT` (`/content/kitti_raw/training`) trên SSD nội bộ Colab.
- Tự động nhận diện và sắp xếp cấu trúc thư mục chuẩn `training/velodyne`, `training/label_2`, `training/calib`.
- Bỏ qua tự động nếu thư mục cục bộ đã có đủ 7,481 files."""))

    # Cell 5: Code Download & Extract
    cells.append(make_code_cell("""# ==============================================================================
# Cell 3: Extract KITTI Dataset (.rar / .zip) from Google Drive to Local Colab SSD
# ==============================================================================
import os
import sys
import shutil
from pathlib import Path
from shlex import quote as q

# 1. Đảm bảo tiện ích unrar đã sẵn sàng
if shutil.which("unrar") is None:
    print("Installing unrar utility for extracting .rar files...")
    !apt-get update -qq && apt-get install -y -qq unrar

# Đường dẫn thư mục chứa dataset / file nén (.rar, .zip) trên Google Drive
# Mặc định lấy từ KITTI_DATASET_ZIP, hoặc có thể chỉnh sửa trực tiếp tại đây:
DRIVE_DATASET_PATH = Path("/content/drive/MyDrive/KITTI_DATASET_ZIP").resolve() if IN_COLAB else DRIVE_RAW_DIR

# Tự động quét và phát hiện thư mục chứa file rar trên Google Drive nếu đường dẫn trên trống
def auto_detect_kitti_drive_path(current_path: Path) -> Path:
    def has_kitti_archives(folder: Path) -> bool:
        if not folder.is_dir():
            return False
        files = [p.name.lower() for p in folder.iterdir() if p.is_file()]
        return any(
            (ext in f) and any(k in f for k in ("velodyne", "label", "calib", "kitti"))
            for f in files for ext in (".rar", ".zip", ".tar")
        )

    if has_kitti_archives(current_path):
        return current_path

    search_dirs = [
        Path("/content/drive/MyDrive/KITTI_DATASET_ZIP"),
        Path("/content/drive/MyDrive/KITTI_ZIP"),
        Path("/content/drive/MyDrive/KITTI_RAW"),
        Path("/content/drive/MyDrive/kitti_raw"),
        Path("/content/drive/MyDrive/kitti"),
        Path("/content/drive/MyDrive/KITTI"),
        Path("/content/drive/MyDrive/KITTI_DATASET"),
        Path("/content/drive/MyDrive/data_object"),
        Path("/content/drive/MyDrive/lidar_project/raw"),
        Path("/content/drive/MyDrive"),
    ]
    for d in search_dirs:
        if has_kitti_archives(d):
            print(f"💡 Tự động phát hiện các file nén KITTI tại: {d}")
            return d.resolve()
    return current_path

DRIVE_DATASET_PATH = auto_detect_kitti_drive_path(DRIVE_DATASET_PATH)

RAW_TRAINING = RAW_KITTI_ROOT if RAW_KITTI_ROOT.name == "training" else RAW_KITTI_ROOT / "training"
EXTRACTION_ROOT = RAW_TRAINING.parent
RAW_TRAINING.mkdir(parents=True, exist_ok=True)
EXTRACTION_ROOT.mkdir(parents=True, exist_ok=True)

archives_spec = {
    "velodyne": ("*.bin", 7481, ["velodyne"]),
    "label_2": ("*.txt", 7481, ["label_2", "label", "labels"]),
    "calib": ("*.txt", 7481, ["calib"]),
}
if BOX_MODE == "3d" or "3d" in EVALUATION_MODES or "bev" in EVALUATION_MODES:
    archives_spec["image_2"] = ("*.png", 7481, ["image_2", "image", "images"])

print("=" * 80)
print(f"📂 Google Drive Source Path : {DRIVE_DATASET_PATH}")
print(f"⚡ Local Colab SSD Target  : {RAW_TRAINING}")
if DRIVE_DATASET_PATH.is_dir():
    found_items = [p.name for p in DRIVE_DATASET_PATH.iterdir()]
    print(f"📄 Files detected on Drive : {found_items}")
else:
    print(f"⚠️ Drive directory not found: {DRIVE_DATASET_PATH}")
print("=" * 80)

for name, (pattern, expected_count, keywords) in archives_spec.items():
    target_folder = RAW_TRAINING / name
    current_count = len(list(target_folder.glob(pattern))) if target_folder.is_dir() else 0
    if current_count >= expected_count:
        print(f"[{name.upper()}] Local SSD folder ready: {target_folder} ({current_count}/{expected_count} files)")
        continue

    print(f"[{name.upper()}] Incomplete or missing ({current_count}/{expected_count}). Searching on Google Drive...")

    # 1. Tìm kiếm file nén (.rar, .zip, .tar)
    archive_candidates = [
        DRIVE_DATASET_PATH / f"{name}.rar",
        DRIVE_DATASET_PATH / f"data_object_{name}.rar",
        DRIVE_DATASET_PATH / f"kitti_{name}.rar",
        DRIVE_DATASET_PATH / f"{name}.zip",
        DRIVE_DATASET_PATH / f"data_object_{name}.zip",
        DRIVE_DATASET_PATH / f"kitti_{name}.zip",
        DRIVE_DATASET_PATH / f"{name}.tar",
        DRIVE_DATASET_PATH / "training" / f"{name}.rar",
        DRIVE_DATASET_PATH / "training" / f"data_object_{name}.rar",
        DRIVE_DATASET_PATH / "training" / f"{name}.zip",
        DRIVE_DATASET_PATH / "training" / f"data_object_{name}.zip",
        EXTRACTION_ROOT / f"{name}.rar",
        EXTRACTION_ROOT / f"data_object_{name}.rar",
        EXTRACTION_ROOT / f"{name}.zip",
        EXTRACTION_ROOT / f"data_object_{name}.zip",
    ]
    for kw in keywords:
        archive_candidates.extend([
            DRIVE_DATASET_PATH / f"{kw}.rar",
            DRIVE_DATASET_PATH / f"data_object_{kw}.rar",
            DRIVE_DATASET_PATH / f"{kw}.zip",
            DRIVE_DATASET_PATH / f"data_object_{kw}.zip",
        ])

    archive_path = next((p for p in archive_candidates if p.is_file()), None)

    # Nếu chưa tìm thấy theo tên chuẩn, quét đệ quy tìm bất kỳ file nén nào chứa từ khóa trong thư mục Drive
    if archive_path is None and DRIVE_DATASET_PATH.is_dir():
        for ext in ("*.rar", "*.zip", "*.tar", "*.tar.gz"):
            for candidate in sorted(DRIVE_DATASET_PATH.rglob(ext)):
                cand_lower = candidate.name.lower()
                if any(kw in cand_lower for kw in keywords):
                    if "part" in cand_lower and not any(cand_lower.endswith(f"{p}.rar") or f"{p}." in cand_lower for p in ("part1", "part01", "part001", "part.1")):
                        continue
                    archive_path = candidate
                    break
            if archive_path is not None:
                break

    if archive_path is not None:
        print(f"Found archive on Drive: {archive_path}")
        print(f"Extracting {archive_path.name} -> {EXTRACTION_ROOT} (fast local SSD)...")
        ext = archive_path.suffix.lower()
        if ext == ".rar":
            !unrar x -o+ {q(str(archive_path))} {q(str(EXTRACTION_ROOT))}/
        elif ext in (".tar", ".gz"):
            !tar -xf {q(str(archive_path))} -C {q(str(EXTRACTION_ROOT))}
        else:
            !unzip -q -o {q(str(archive_path))} -d {q(str(EXTRACTION_ROOT))}

        # Tự động tổ chức lại nếu giải nén vào thư mục lồng nhau hoặc cùng cấp
        if len(list(target_folder.glob(pattern))) < expected_count:
            for found_dir in EXTRACTION_ROOT.rglob(name):
                if found_dir.is_dir() and found_dir.resolve() != target_folder.resolve():
                    found_files = list(found_dir.glob(pattern))
                    if len(found_files) >= expected_count:
                        print(f"Relocating extracted {name} folder from {found_dir} to {target_folder}...")
                        shutil.rmtree(target_folder, ignore_errors=True)
                        shutil.move(str(found_dir), str(target_folder))
                        break

        # Nếu các file được giải nén phẳng (flat) trực tiếp vào EXTRACTION_ROOT
        if len(list(target_folder.glob(pattern))) < expected_count:
            flat_files = list(EXTRACTION_ROOT.glob(pattern))
            if len(flat_files) >= expected_count:
                print(f"Relocating {len(flat_files)} flat {pattern} files to {target_folder}...")
                target_folder.mkdir(parents=True, exist_ok=True)
                for f in flat_files:
                    f.rename(target_folder / f.name)
    else:
        # 2. Nếu đã có sẵn thư mục chưa nén trên Drive, copy nhanh sang local SSD
        drive_folder_candidates = [
            DRIVE_DATASET_PATH / name,
            DRIVE_DATASET_PATH / "training" / name,
        ]
        for kw in keywords:
            drive_folder_candidates.extend([
                DRIVE_DATASET_PATH / kw,
                DRIVE_DATASET_PATH / "training" / kw,
            ])
        drive_folder = next((p for p in drive_folder_candidates if p.is_dir()), None)
        if drive_folder is not None:
            drive_files = len(list(drive_folder.glob(pattern)))
            print(f"Found uncompressed folder on Drive: {drive_folder} ({drive_files} files).")
            print(f"Copying to local Colab SSD for maximum DataLoader I/O speed...")
            !cp -r {q(str(drive_folder))} {q(str(target_folder))}
        else:
            drive_contents = [p.name for p in DRIVE_DATASET_PATH.iterdir()] if DRIVE_DATASET_PATH.is_dir() else []
            raise FileNotFoundError(
                f"Could not find '{name}' archive (.rar/.zip) on Google Drive!\\n"
                f"Searched in: {DRIVE_DATASET_PATH}\\n"
                f"Files found in Drive folder: {drive_contents}\\n"
                f"Please verify your Drive folder path in DRIVE_DATASET_PATH."
            )

    final_count = len(list(target_folder.glob(pattern))) if target_folder.is_dir() else 0
    print(f"[{name.upper()}] Verified {final_count} files in {target_folder}.")
    if final_count < expected_count:
        print(f"⚠️ Warning: Found {final_count} files for {name}, expected {expected_count}.")

print("=" * 80)
print("✓ All raw KITTI data extracted to local SSD and verified.")"""))

    # Cell 6: Markdown Prepare KITTI
    cells.append(make_markdown_cell("""## 🔄 4. Preprocess KITTI & Create Manifests (`prepare_kitti.py`)

Chuyển đổi nhãn camera KITTI sang toạ độ Velodyne của detector và tạo các file manifest phân chia tập train/val (`train.txt`, `val.txt`).
Nếu tập dữ liệu đã được tiền xử lý trước đó, bước này sẽ tự động bỏ qua để tiết kiệm thời gian (trừ khi đặt `FORCE_REPREPARE = True`)."""))

    # Cell 7: Code Prepare KITTI
    cells.append(make_code_cell("""# ==============================================================================
# Cell 4: Preprocess KITTI Dataset
# ==============================================================================
FORCE_REPREPARE = False
train_manifest = PROCESSED_DATASET_DIR / "train.txt"
val_manifest = PROCESSED_DATASET_DIR / "val.txt"
processed_pc = PROCESSED_DATASET_DIR / "pointcloud"
processed_lbl = PROCESSED_DATASET_DIR / "label"

is_prepared = (
    train_manifest.is_file() and val_manifest.is_file() and
    processed_pc.is_dir() and processed_lbl.is_dir() and
    len(list(processed_pc.glob("*.bin"))) >= 7481 and
    len(list(processed_lbl.glob("*.txt"))) >= 7481
)

if is_prepared and not FORCE_REPREPARE:
    print(f"Processed dataset already exists at: {PROCESSED_DATASET_DIR}")
    train_n = len(train_manifest.read_text().splitlines())
    val_n = len(val_manifest.read_text().splitlines())
    print(f"Train split: {train_n:,} frames | Val split: {val_n:,} frames")
else:
    print(f"Processing KITTI into detector format...")
    train_ids_arg = f"--train-ids {q(str(REPO_DIR / TRAIN_SPLIT))}" if (REPO_DIR / TRAIN_SPLIT).is_file() else ""
    val_ids_arg = f"--val-ids {q(str(REPO_DIR / VAL_SPLIT))}" if (REPO_DIR / VAL_SPLIT).is_file() else ""
    kitti_base = RAW_TRAINING.parent

    !{q(sys.executable)} -u tools/kitti_training_pipeline/prepare_kitti.py \
      --kitti-root {q(str(kitti_base))} \
      --output-root {q(str(PROCESSED_DATASET_DIR))} \
      --config-output {q(str(PROCESSED_DATASET_DIR / "generated_baseline.json"))} \
      {train_ids_arg} {val_ids_arg}

    if _exit_code != 0:
        raise RuntimeError("prepare_kitti.py failed. Check logs above.")
    print("✓ KITTI preprocessing completed.")"""))

    # Cell 8: Markdown GT Database
    cells.append(make_markdown_cell("""## 🗄️ 5. Build Ground Truth Database (`build_gt_database.py`)

Cắt đám mây điểm đối tượng (`Car`, `Pedestrian`, `Cyclist`) thành các file `.bin` mẫu và tạo file chỉ mục `gt_database/dbinfos_train.json`.
* **Quan trọng**: Database được tạo **nghiêm ngặt chỉ từ các frame của tập train**, đảm bảo không rò rỉ (leak) dữ liệu từ tập validation!
* Bắt buộc phải có để sử dụng chế độ tăng cường `hybrid_gt` hoặc `openpcdet_gt`."""))

    # Cell 9: Code Build GT Database
    cells.append(make_code_cell("""# ==============================================================================
# Cell 5: Build Ground Truth Database for Augmentation
# ==============================================================================
FORCE_REBUILD_GT_DATABASE = False
dbinfo_path = PROCESSED_DATASET_DIR / "gt_database" / "dbinfos_train.json"

if dbinfo_path.is_file() and not FORCE_REBUILD_GT_DATABASE:
    try:
        db_data = json.loads(dbinfo_path.read_text())
        car_count = len(db_data.get("Car", []))
        ped_count = len(db_data.get("Pedestrian", []))
        cyc_count = len(db_data.get("Cyclist", []))
        print(f"Existing GT Database verified: {dbinfo_path}")
        print(f"Sample bank -> Car: {car_count:,} | Pedestrian: {ped_count:,} | Cyclist: {cyc_count:,}")
    except Exception as e:
        print(f"Existing database corrupted ({e}). Rebuilding...")
        FORCE_REBUILD_GT_DATABASE = True

if not dbinfo_path.is_file() or FORCE_REBUILD_GT_DATABASE:
    print("Building Ground Truth database from training frames...")
    train_split_file = PROCESSED_DATASET_DIR / "train.txt"
    val_split_file = PROCESSED_DATASET_DIR / "val.txt"
    if not train_split_file.is_file():
        train_split_file = REPO_DIR / TRAIN_SPLIT
    if not val_split_file.is_file():
        val_split_file = REPO_DIR / VAL_SPLIT

    !{q(sys.executable)} -u tools/kitti_training_pipeline/build_gt_database.py \
      --processed-root {q(str(PROCESSED_DATASET_DIR))} \
      --train-split {q(str(train_split_file))} \
      --val-split {q(str(val_split_file))} \
      --output-dir {q(str(PROCESSED_DATASET_DIR / "gt_database"))}

    if _exit_code != 0:
        raise RuntimeError("build_gt_database.py failed!")
    print(f"✓ GT Database built successfully at: {PROCESSED_DATASET_DIR / 'gt_database'}")"""))

    # Cell 10: Markdown Visualize GT Sampling
    cells.append(make_markdown_cell("""## 👁️ 6. Trực quan hóa GT Sampler (Visualize GT Sampling)

Kiểm tra trực quan quá trình chèn vật thể ảo (`GT sampling`) vào đám mây điểm.
Hiển thị 3 bảng so sánh:
1. **Original**: Đám mây điểm và Bounding box ban đầu.
2. **GT Sampling**: Các hộp và điểm được chèn thêm (màu xanh neon nổi bật).
3. **GT + World Transforms**: Sau khi áp dụng biến đổi toàn cục (xoay, co giãn, lật)."""))

    # Cell 11: Code Visualize GT Sampling
    cells.append(make_code_cell("""# ==============================================================================
# Cell 6: Visualize GT Sampling (Original vs Sampled vs Transformed)
# ==============================================================================
ENABLE_GT_VISUALIZATION = True
FRAME_INDEX = 0             # Chỉ số frame trong train.txt để xem
VISUALIZATION_SEED = 42
FORCE_SAMPLING = True       # Ép buộc chèn mẫu ngay cả khi xác suất ngẫu nhiên < 1

if ENABLE_GT_VISUALIZATION and AUGMENTATION in ("hybrid_gt", "openpcdet_gt"):
    from tools.visualization.visualize_gt_sampling import preview_sampling, render_preview, print_summary
    import matplotlib.pyplot as plt

    print(f"Generating preview for frame index {FRAME_INDEX} with {AUGMENTATION}...")
    try:
        preview = preview_sampling(
            config_dict,
            repo_dir=REPO_DIR,
            frame_index=FRAME_INDEX,
            seed=VISUALIZATION_SEED,
            force_sampling=FORCE_SAMPLING
        )
        print_summary(preview)

        fig = render_preview(preview, config_dict)
        plt.show()

        preview_save_path = RUN_DIR / "gt_sampling_preview.png"
        fig.savefig(preview_save_path, dpi=140)
        print(f"✓ Saved preview figure to: {preview_save_path}")
    except Exception as e:
        print(f"Could not render preview: {e}")
else:
    print("GT Visualization is disabled or current augmentation mode does not use GT database.")"""))

    # Cell 12: Markdown Smoke Test
    cells.append(make_markdown_cell("""## 🧪 7. Sanity Smoke Train (Kiểm tra nhanh tính toàn vẹn)

Chạy thử 1 epoch cực ngắn (chỉ 2 batch train và 2 batch val) trên thư mục tạm.
Giúp kiểm tra:
- Khả năng forward và backward của mô hình trên GPU.
- Tính hữu hạn của các thành phần loss (không bị NaN / Inf).
- Cập nhật trọng số của optimizer.
👉 Tiết kiệm thời gian, đảm bảo cấu hình chạy trơn tru trước khi bắt đầu Full Training."""))

    # Cell 13: Code Smoke Test
    cells.append(make_code_cell("""# ==============================================================================
# Cell 7: Fast Sanity Smoke Train
# ==============================================================================
RUN_SMOKE_TEST = True
SMOKE_TRAIN_BATCHES = 2
SMOKE_VAL_BATCHES = 2

if RUN_SMOKE_TEST:
    print("Initiating Sanity Smoke Test...")
    smoke_dir = Path(tempfile.mkdtemp(prefix="smoke_lidar_"))
    smoke_config = create_experiment_config(config_dict, {
        "train": {
            "epochs": 1,
            "warmup_epochs": 0,
            "checkpoint_selection": {"primary": "loss"}
        }
    })
    smoke_cfg_path = smoke_dir / "config.json"
    write_json(smoke_cfg_path, smoke_config)

    !{q(sys.executable)} -u tools/kitti_training_pipeline/train.py \
      --config {q(str(smoke_cfg_path))} \
      --detector-root detector \
      --output-root {q(str(smoke_dir))} \
      --run-name smoke \
      --device {q(DEVICE)} \
      --max-train-batches {SMOKE_TRAIN_BATCHES} \
      --max-val-batches {SMOKE_VAL_BATCHES}

    if _exit_code != 0:
        raise RuntimeError("Smoke test failed! Review trace above.")

    smoke_metric_file = smoke_dir / "smoke/metrics.jsonl"
    smoke_ckpt = smoke_dir / "smoke/checkpoints/last.pt"
    if smoke_metric_file.is_file() and smoke_ckpt.is_file():
        metrics = [json.loads(line) for line in smoke_metric_file.read_text().splitlines() if line.strip()]
        if metrics and math.isfinite(metrics[0]["train_objective"]) and math.isfinite(metrics[0]["validation"]["loss"]):
            print(f"✅ Smoke Test PASSED! Train Objective: {metrics[0]['train_objective']:.4f} | Val Loss: {metrics[0]['validation']['loss']:.4f}")
        else:
            raise RuntimeError("Smoke test finished with non-finite loss.")
    shutil.rmtree(smoke_dir, ignore_errors=True)
else:
    print("Smoke test skipped.")"""))

    # Cell 14: Markdown TensorBoard
    cells.append(make_markdown_cell("""## 📈 8. TensorBoard Telemetry

Theo dõi đường cong loss thành phần, learning rate, peak VRAM và các chỉ số validation AP theo thời gian thực."""))

    # Cell 15: Code TensorBoard
    cells.append(make_code_cell("""# ==============================================================================
# Cell 8: Launch TensorBoard inline
# ==============================================================================
%load_ext tensorboard
%tensorboard --logdir {q(str(ARTIFACT_ROOT))}"""))

    # Cell 16: Markdown Full Train
    cells.append(make_markdown_cell("""## 🚀 9. Full Training & Smart Auto-Resume (Huấn luyện chính thức)

Bắt đầu huấn luyện mô hình.
- **Tự động Resume**: Nếu bị ngắt kết nối hoặc Colab timeout, chỉ cần chạy lại cell này để tự động tìm checkpoint mới nhất (`last.pt` hoặc `{N}epoch.pt`) và tiếp tục huấn luyện.
- **Dual Checkpoint Retention**: Tự động lưu độc lập **`best_ap_checkpoint.pth`** và **`best_loss_checkpoint.pth`**.
- In tóm tắt Loss & LR trước mỗi vòng tính AP để loại bỏ hiện tượng treo màn hình."""))

    # Cell 17: Code Full Train
    cells.append(make_code_cell("""# ==============================================================================
# Cell 9: Full Training & Smart Auto-Resume
# ==============================================================================
import re

CHECKPOINT_DIR = RUN_DIR / "checkpoints"
RUN_METADATA_PATH = RUN_DIR / "run.json"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RUN_METADATA = {
    "variant": VARIANT,
    "run_name": RUN_NAME,
    "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    "seed": SEED,
    "precision": PRECISION,
    "physical_batch_size": PHYSICAL_BATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "target_backend": TARGET_BACKEND,
    "epochs": EPOCHS,
}

if RUN_METADATA_PATH.is_file():
    saved_meta = read_json(RUN_METADATA_PATH)
    critical = ["config_sha256", "seed", "precision", "physical_batch_size"]
    diffs = [k for k in critical if saved_meta.get(k) != RUN_METADATA.get(k)]
    if diffs:
        print(f"⚠️ Warning: Config differs from previous run on keys: {diffs}")
write_json(RUN_METADATA_PATH, RUN_METADATA)

def get_checkpoint_epoch(path):
    st = torch.load(path, map_location="cpu", weights_only=True)
    return int(st["epoch"])

last_ckpt = CHECKPOINT_DIR / "last.pt"
epoch_ckpts = [p for p in CHECKPOINT_DIR.glob("*epoch.pt") if re.fullmatch(r"\\d+epoch\\.pt", p.name)]
resume_ckpt = last_ckpt if last_ckpt.is_file() else max(epoch_ckpts, key=get_checkpoint_epoch, default=None)
start_epoch = get_checkpoint_epoch(resume_ckpt) if resume_ckpt else 0

if start_epoch >= EPOCHS:
    print(f"Training already completed: {start_epoch}/{EPOCHS} epochs.")
else:
    resume_arg = f"--resume {q(str(resume_ckpt))}" if resume_ckpt else ""
    print(f"🚀 Starting training run: {RUN_NAME}")
    print(f"Epoch progress: {start_epoch + 1} -> {EPOCHS}")
    print(f"Logs: {RUN_DIR / 'train.log'}")
    print(f"Checkpoints dir: {CHECKPOINT_DIR}")

    !{q(sys.executable)} -u tools/kitti_training_pipeline/train.py \
      --config {q(str(CONFIG))} \
      --detector-root detector \
      --output-root {q(str(ARTIFACT_ROOT))} \
      --run-name {q(RUN_NAME)} \
      --device {q(DEVICE)} \
      {resume_arg}

    if _exit_code != 0:
        raise RuntimeError("Training failed. Review train.log and outputs above.")
    print(f"✓ Training finished! Artifacts stored in {RUN_DIR}")"""))

    # Cell 18: Markdown Dual Evaluation
    cells.append(make_markdown_cell("""## 🏆 10. Đánh giá Checkpoint Kép (`best_ap` & `best_loss`)

Đánh giá và so sánh độc lập cả 2 model tốt nhất:
1. **Best AP Checkpoint**: Model có điểm AP cao nhất trên tập validation.
2. **Best Loss Checkpoint**: Model có tổng validation loss nhỏ nhất.

Bảng kết quả sẽ hiển thị chi tiết AP từng lớp (`Car`, `Pedestrian`, `Cyclist`), Mean Moderate AP và Validation Loss."""))

    # Cell 19: Code Dual Evaluation
    cells.append(make_code_cell("""# ==============================================================================
# Cell 10: Dual Checkpoint Evaluation (Best AP & Best Loss)
# ==============================================================================
import pandas as pd
from IPython.display import display
from tools.kitti_training_pipeline import checkpoint_selection, notebook_workflow
from tools.kitti_training_pipeline.notebook_workflow import selected_runs, evaluation_rows

RUN_EVALUATION = True

if RUN_EVALUATION:
    print(f"Evaluating best winners for run: {RUN_NAME}...")
    evaluated_runs = selected_runs(RUN_DIR)
    comparison_rows = []

    for kind, evaluated_run in evaluated_runs.items():
        print(f"\\n=======================================================")
        print(f"🔍 Evaluating '{kind.upper()}' checkpoint (Epoch {evaluated_run['epoch']})")
        print(f"=======================================================")
        for mode in EVALUATION_MODES:
            output_json = RUN_DIR / f"evaluation_{kind}_{mode}.json"
            raw_root = globals().get("RAW_KITTI_ROOT", RAW_TRAINING.parent)
            cmd_args = notebook_workflow.evaluation_command(
                evaluated_run,
                mode,
                raw_root,
                output_json,
                repo_root=REPO_DIR,
                device=DEVICE
            )
            cmd_args.insert(1, "-u")
            eval_cmd = " ".join(q(str(a)) for a in cmd_args)

            !{eval_cmd}
            if _exit_code != 0:
                raise RuntimeError(f"Evaluation failed for {kind} in mode {mode}")

            report = read_json(output_json)
            rows = evaluation_rows(report, selection_kind=kind)
            for row in rows:
                row["checkpoint_winner"] = kind
                row["best_epoch"] = evaluated_run["epoch"]
                row["val_loss"] = evaluated_run["validation_loss"]
            comparison_rows.extend(rows)

    if comparison_rows:
        df_comparison = pd.DataFrame(comparison_rows)
        print("\\n" + "=" * 80)
        print("🏆 EVALUATION SUMMARY (BEST AP vs BEST LOSS)")
        print("=" * 80)
        display(df_comparison)

        csv_path = RUN_DIR / "comparison_best_ap_vs_loss.csv"
        df_comparison.to_csv(csv_path, index=False)
        write_json(RUN_DIR / "comparison_best_ap_vs_loss.json", {"rows": comparison_rows})
        print(f"✓ Summary saved to: {csv_path}")"""))

    # Cell 20: Markdown External Evaluation
    cells.append(make_markdown_cell("""## 🔍 11. Đánh giá Checkpoint của Run khác (Tùy chọn)

Nếu bạn có một run huấn luyện khác trong `artifacts` muốn đánh giá lại hoặc so sánh, chỉ cần đặt đường dẫn thư mục vào biến `EXTERNAL_RUN_DIR`."""))

    # Cell 21: Code External Evaluation
    cells.append(make_code_cell("""# ==============================================================================
# Cell 11: Evaluate External Run (Optional)
# ==============================================================================
EXTERNAL_RUN_DIR = None  # Ví dụ: ARTIFACT_ROOT / "ten_run_khac"

if EXTERNAL_RUN_DIR and Path(EXTERNAL_RUN_DIR).is_dir():
    ext_path = Path(EXTERNAL_RUN_DIR)
    print(f"Evaluating external run: {ext_path}")
    ext_runs = selected_runs(ext_path)
    ext_rows = []
    for kind, ext_run in ext_runs.items():
        for mode in ext_run["config"].get("evaluation", {}).get("modes", ["local_bev"]):
            out_file = ext_path / f"standalone_eval_{kind}_{mode}.json"
            raw_root = ext_run["config"].get("evaluation", {}).get("kitti_root", RAW_KITTI_ROOT)
            args = notebook_workflow.evaluation_command(ext_run, mode, raw_root, out_file, repo_root=REPO_DIR, device=DEVICE)
            args.insert(1, "-u")
            !{" ".join(q(str(a)) for a in args)}
            if _exit_code == 0:
                ext_rows.extend(evaluation_rows(read_json(out_file), selection_kind=kind))
    if ext_rows:
        display(pd.DataFrame(ext_rows))
else:
    print("No external run specified. Set EXTERNAL_RUN_DIR to evaluate an existing run.")"""))

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_file = Path("3D_Lidar_Detection_Pipeline_Robust.ipynb")
    out_file.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {out_file} ({len(cells)} cells)")

if __name__ == "__main__":
    build_notebook()
