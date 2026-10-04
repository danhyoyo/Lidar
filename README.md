# KITTI augmentation ablation

Branch này chỉ giữ cấu hình augmentation trong [`configs/kitti/augmentation`](configs/kitti/augmentation/README.md).
Model cố định: MobilePIXORNeXt + LiteMLA + SG-FPN, Rich8, baseline loss.

| Run | Config | Thay đổi |
|---|---|---|
| A | `a_standard.json` | Standard OneOf, không sampling |
| B | `b_light_sampling.json` | A + sampling nhẹ |
| C | `c_sampling_density.json` | B + density subsampling |
| D | `d_sampling_shadow.json` | B + shadow masking |
| E | `e_sampling_intensity.json` | B + intensity calibration |

Chạy A/B trước, evaluate cùng epoch 50, rồi mới quyết định thử C/D/E.
Các run có tên riêng theo config để không ghi đè nhau.

## Môi trường

Cài PyTorch phù hợp GPU/CUDA, sau đó:

```bash
python3 -m pip install -r requirements-kitti.txt
```

Notebook Colab: [`3D_Lidar_Object_Detection_Notebook_standard.ipynb`](3D_Lidar_Object_Detection_Notebook_standard.ipynb).
Chọn `VARIANT = "A"` hoặc `"B"`, giữ `EPOCHS = 50`.

## Chuẩn bị KITTI

Các manifests trong `splits/kitti/` dùng 5.984 frame train và 1.497 frame validation.
Chạy từ thư mục gốc repository:

```bash
python3 tools/kitti_training_pipeline/prepare_kitti.py \
  --kitti-root /path/to/KITTI/object \
  --output-root data/kitti/processed \
  --config-output data/kitti/generated_kitti.json \
  --train-ids splits/kitti/train.txt \
  --val-ids splits/kitti/val.txt
```

B–E cần GT database tạo từ đúng tập train. Khi chưa chắc nguồn hoặc phiên bản cache, tạo lại:

```bash
python3 tools/dataset_converter/create_gt_database.py \
  --processed-dir data/kitti/processed \
  --train-ids splits/kitti/train.txt \
  --output-file data/kitti/kitti_gt_database.pkl
```

## Screening 50 epoch

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/augmentation/a_standard.json \
  --detector-root detector --output-root artifacts/kitti \
  --num-workers 6 --target-backend numba
```

Config đặt toàn bộ lịch train 50 epoch: warmup 4 epoch, cosine 46 epoch;
LR cuối lịch đạt `1e-6`. Đổi config để chạy B–E.
Cùng batch 16, seed 42, AdamW LR 0.0007, weight decay 0.001 và clipping 10.

## Evaluate cùng checkpoint

```bash
python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name a_standard_epoch50 --backend pytorch \
  --model artifacts/kitti/mobilepixornext-standard_aug-baseline_loss-rich8-baseline_iou-sgfpn-a_standard-s42/checkpoints/50epoch.pt \
  --config configs/kitti/augmentation/a_standard.json \
  --detector-root detector --kitti-root /path/to/KITTI/object \
  --split splits/kitti/val.txt \
  --output artifacts/kitti/a_standard_epoch50.json --device cuda
```

So sánh mAP Moderate và AP từng lớp của A–E tại cùng epoch 50.
Metric là local KITTI-style rotated BEV AP R40, không phải AP 3D/hidden test chính thức.
`selected/best.pt` chọn theo minimum validation loss; notebook evaluate checkpoint cuối epoch 50 để giữ cùng ngân sách.

## Resume run bị gián đoạn

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/augmentation/a_standard.json \
  --detector-root detector --output-root artifacts/kitti \
  --num-workers 6 --target-backend numba \
  --resume artifacts/kitti/mobilepixornext-standard_aug-baseline_loss-rich8-baseline_iou-sgfpn-a_standard-s42/checkpoints/last.pt
```

Optimizer và scheduler được khôi phục; lịch 50 epoch giữ nguyên.
Trong notebook chạy lại cell training/evaluation để tiếp tục run bị gián đoạn.
Muốn thử lịch 100 epoch, tạo run mới và train lại từ đầu với warmup 8.
So sánh A–E tại 50 epoch; baseline 100 epoch cũ có ngân sách và lịch LR khác.

## Kiểm chứng

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/lidar-mpl python3 -m pytest -q \
  tests/test_augmentation_configs.py tests/test_gt_sampler_config.py \
  tests/test_gt_sampler.py tests/test_optimizer_scheduler.py \
  tests/test_training_screening.py tests/test_standard_training_notebook.py
```

Các model/loss khác vẫn có unit tests nhưng không còn config thực nghiệm trên branch này.
Xem kiến trúc tại [`docs/mobilepixornext_architecture.md`](docs/mobilepixornext_architecture.md).
