# B0-B5 Rotated BEV Loss Ablation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Xây dựng B0-B5 như một nghiên cứu loss độc lập trên một snapshot kiến trúc duy nhất, không sửa hoặc đổi nghĩa A0-A4.

**Architecture:** B0-B5 dùng full JSON configs riêng dưới `configs/kitti/loss_ablation/`; mọi trường ngoài loss/codename giống nhau. `LossFunction` chỉ điều phối. Mỗi geometry objective là một `torch.nn.Module` rõ ràng (`KFIoULoss`, `ProbIoULoss`, `KLDLoss`, `MGIoULoss`) trong module chuyên trách; shared helpers được phép dùng cho công thức chung. Old configs tiếp tục đi qua legacy path nguyên trạng.

**Tech Stack:** Python 3, PyTorch, NumPy, `unittest`; không thêm dependency.

**Constraints:** Không commit, không push và không chạy job 100 epochs khi chưa có phê duyệt riêng. Không sửa docs/config MobileBEV, model/head, target, decoder, evaluator hoặc export contract.

**Source specification:** `docs/rotated_bev_losses/SPEC.md`.

---

## Hợp đồng codename và thư mục

```text
configs/kitti/loss_ablation/
├── b0_a4_legacy_uwag.json
├── b1_l1_fixed.json
├── b2_kfiou_fixed.json
├── b3_probiou_fixed.json
├── b4_kld_fixed.json
└── b5_mgiou_fixed.json       # candidate, expose có chủ đích cho L4 smoke
```

| ID | Objective | Vai trò |
|---|---|---|
| B0 | A4 snapshot + legacy UWAG + auxiliary 0.2 | Historical anchor |
| B1 | Masked-L1 fixed | Controlled baseline |
| B2 | Full KFIoU fixed | Single-loss ablation |
| B3 | ProbIoU BD/Hellinger fixed | Single-loss ablation |
| B4 | KLD-log fixed | Single-loss ablation |
| B5 | MGIoU fixed | Single-loss ablation, gated |

Codename là bất biến: không tái sử dụng B0-B5 cho objective khác. B6/B7 hybrid
không thuộc plan này.

New loss schema cho B1-B5:

```json
{
  "regression": "l1|kfiou|probiou|kld|mgiou",
  "weighting": "fixed",
  "regression_weight": 1.0,
  "covariance_epsilon": 0.000001,
  "max_abs_log_size": 10.0,
  "probiou_switch_fraction": 0.5,
  "kld_tau": 1.0
}
```

B0 giữ nguyên old schema `name=uwag`. Các config B là full snapshots, không
dùng inheritance hoặc CLI-generated config.

---

### Task 1: Đóng băng B0-B5 configs và invariant tests

**Files:**

- Create: `configs/kitti/loss_ablation/b0_a4_legacy_uwag.json`
- Create: `configs/kitti/loss_ablation/b1_l1_fixed.json`
- Create: `configs/kitti/loss_ablation/b2_kfiou_fixed.json`
- Create: `configs/kitti/loss_ablation/b3_probiou_fixed.json`
- Create: `configs/kitti/loss_ablation/b4_kld_fixed.json`
- Create: `tests/test_loss_ablation.py`
- Reference only: `configs/kitti/mobilebev/a4_rich8_sgfpn_bev.json`

**Step 1: Ghi RED config tests**

Load sáu JSON và assert:

1. File/codename/objective đúng bảng trên.
2. Sau khi bỏ `loss` và `note`, B0-B5 giống hệt nhau.
3. `data`, `model`, augmentation, split, optimizer, schedule, precision và
   effective batch size khớp snapshot A4 đã duyệt.
4. B0 giữ exact legacy loss mapping; B1-B5 dùng fixed new schema.
5. Không file nào dưới `configs/kitti/mobilebev/` thay đổi.

Run:

```bash
python3 -m unittest tests/test_loss_ablation.py
```

Expected: FAIL vì B configs chưa tồn tại.

**Step 2: Tạo full config snapshots tối thiểu**

Copy giá trị A4 vào B0-B5; chỉ đổi `note`, `loss` và thêm cùng một
`train.selection_policy="final_epoch"` cho toàn B-series. B0 giữ:

```json
{
  "epsilon": 0.0001,
  "geometric_weight": 0.2,
  "initial_log_scales": [0, 0, 0, 0],
  "max_abs_log_size": 10,
  "name": "uwag"
}
```

Không symlink và không thêm config loader.

**Step 3: Chạy GREEN**

Run `python3 -m unittest tests/test_loss_ablation.py`.

Expected: config invariants PASS.

---

### Task 2: Tách geometry math khỏi `LossFunction`

**Files:**

- Create: `detector/core/losses/gaussian_geometry_loss.py`
- Modify: `detector/core/losses/loss_fn.py`
- Modify: `tests/test_loss_ablation.py`

**Step 1: RED tests cho covariance**

- `theta=0`, `(width,length)=(2,4)`, divisor 4 cho `diag(4,1)`.
- `theta` và `theta+pi` cho cùng covariance.
- Near-square/thin box finite và positive definite.
- Input BF16 được cast sang FP32 trước linear algebra.

**Step 2: Implement helper dùng chung**

```python
def box_covariance(log_size, doubled_yaw, divisor, epsilon, max_abs_log_size):
    log_size = log_size.float()
    doubled_yaw = doubled_yaw.float()
    clamped = log_size.clamp(-max_abs_log_size, max_abs_log_size)
    width, length = torch.exp(clamped).unbind(dim=-1)
    cos2, sin2 = F.normalize(doubled_yaw, dim=-1, eps=epsilon).unbind(dim=-1)
    parallel = length.square() / divisor
    perpendicular = width.square() / divisor
    mean = 0.5 * (parallel + perpendicular)
    delta = 0.5 * (parallel - perpendicular)
    covariance = torch.stack(
        (mean + delta * cos2, delta * sin2,
         delta * sin2, mean - delta * cos2), dim=-1
    ).reshape(-1, 2, 2)
    eye = torch.eye(2, dtype=torch.float32, device=covariance.device)
    scale = torch.maximum(
        torch.maximum(parallel, perpendicular),
        torch.ones_like(parallel),
    )
    covariance = covariance + epsilon * scale.reshape(-1, 1, 1) * eye
    return covariance, (clamped != log_size).sum()
```

Regularization phải là scale-relative `epsilon * max(a, b, 1) * I`, không
phải fixed absolute `epsilon * I`. Trong FP32, covariance của hộp aspect-ratio
lớn có thể mất phương sai nhỏ do cancellation trong biểu thức
`(a+b)/2 +/- (a-b) cos(2 theta)/2`; jitter tuyệt đối không bảo đảm cùng mức
conditioning trên mọi kích thước. Scale-relative regularization giữ
condition number hiệu dụng xấp xỉ không quá `1/epsilon` khi eigenvalue nhỏ
tiến về zero. `max_abs_log_size` vẫn clamp log-size trong `[-10, 10]` và phải
tiếp tục đếm số phần tử bị clamp.

Không dùng `atan2`, explicit inverse hoặc `nan_to_num` cho Gaussian losses.

**Step 3: GREEN**

Run `python3 -m unittest tests/test_loss_ablation.py`.

Expected: covariance tests PASS; legacy `LossFunction` tests vẫn PASS.

---

### Task 3: B2 Full KFIoU bằng TDD

**Files:**

- Modify: `detector/core/losses/gaussian_geometry_loss.py`
- Modify: `tests/test_loss_ablation.py`

**Step 1: RED tests**

- Identical boxes trả `exp(2/3)-1`.
- Offset `(10,0)` so với `(0,0)` cho finite, nonzero position gradient.
- Empty mask trả graph-connected zero.

**Step 2: Implement full loss**

```text
center = log(1 + d^T Sigma_t^-1 d)
K = Sigma_p (Sigma_p + Sigma_t)^-1
Sigma_kf = Sigma_p - K Sigma_p
KFIoU = V(Sigma_kf) / (V(Sigma_p) + V(Sigma_t) - V(Sigma_kf))
loss = center + exp(1 - KFIoU) - 1
```

Dùng `torch.linalg.solve`, `slogdet`, divisor 4 và positive-mask mean. Để tránh
FP32 cancellation, không materialize `Sigma_kf` bằng phép trừ trong forward;
fused volume dùng identity ổn định:

```text
logdet(Sigma_kf) = logdet(Sigma_p) + logdet(Sigma_t)
                    - logdet(Sigma_p + Sigma_t)
```

Shape term không được mô tả là center-aware.

**Step 3: GREEN**

Run `python3 -m unittest tests.test_loss_ablation.KFIoUTests`.

Expected: PASS.

---

### Task 4: B3 ProbIoU và B4 KLD bằng TDD

**Files:**

- Modify: `detector/core/losses/gaussian_geometry_loss.py`
- Modify: `tests/test_loss_ablation.py`

**Step 1: RED tests**

- Identical ProbIoU/KLD bằng 0.
- B3 dùng BD tại progress `<0.5`, Hellinger tại `>=0.5`.
- Cả hai có finite, nonzero translation gradient.
- Near-square/thin và BF16-autocast forward/backward finite.

**Step 2: Implement B3**

Dùng divisor 12, Bhattacharyya distance và:

```python
values = bd if progress < switch_fraction else torch.sqrt(
    (-torch.expm1(-bd)).clamp_min(0.0)
)
```

Chỉ clamp roundoff âm của BD. Progress dựa trên optimizer-update attempts và
phải resume được.

**Step 3: Implement B4**

Dùng hướng `prediction || target`, divisor 4:

```text
D_KL = 0.5 d^T Sigma_t^-1 d
     + 0.5 trace(Sigma_t^-1 Sigma_p)
     + 0.5 log(det(Sigma_t)/det(Sigma_p)) - 1
loss = 1 - 1 / (1 + log(1 + D_KL))
```

Geometry math luôn FP32; reject `kld_tau < 1`.

**Step 4: GREEN**

Run `python3 -m unittest tests/test_loss_ablation.py`.

Expected: B2-B4 unit tests PASS; CUDA BF16 có thể SKIP rõ ràng.

---

### Task 5: B5 MGIoU numerical/runtime pre-gate

**Files:**

- Create: `detector/core/losses/mgiou_loss.py`
- Modify: `tests/test_loss_ablation.py`

**Step 1: Implement research prototype sau RED tests**

Tests bắt buộc: identical zero, finite non-overlap gradient, `theta+pi`,
near-square/thin, BF16 và doubled-yaw `(0,0)` fallback count.

Decode yaw an toàn:

```python
norm = torch.linalg.vector_norm(doubled_yaw.float(), dim=-1, keepdim=True)
valid = norm >= epsilon
unit = doubled_yaw.float() / norm.clamp_min(epsilon)
cos2 = torch.where(valid, unit[..., :1], torch.ones_like(unit[..., :1]))
sin2 = torch.where(valid, unit[..., 1:], torch.zeros_like(unit[..., 1:]))
theta = 0.5 * torch.atan2(sin2, cos2)
```

Tạo four corners, project lên bốn axes của prediction/target, tính 1D GIoU và
`loss=(1-mean_giou)/2`.

**Step 2: Chạy pre-gate số học và runtime**

- Unit suite PASS.
- Cùng representative positive count: 20 warmups, 100 synchronized CUDA
  forward+backward iterations.
- Median MGIoU không quá `2x` KFIoU.

Nếu không có CUDA, status `pending`; nếu fail, status `infeasible`. Cả hai
trường hợp không tạo B5 config và không đổi ngưỡng. Nếu pass, đánh dấu
`candidate`; smoke gate cuối diễn ra sau khi integration/trainer hoàn tất.

Pre-gate đã chạy ngày **2026-09-22** trong conda environment `AI_env` với
PyTorch `2.11.0`, CUDA build `13.0`, RTX 4050 Laptop GPU (compute capability
`8.9`), đúng protocol BF16 gồm 20 warmup và 100 synchronized
forward+backward iterations. FP32/BF16/FP16 trên input regular và extreme đều
finite với gradient khác zero. Median runtime (KFIoU/MGIoU, milliseconds):

| Positive count | KFIoU (ms) | MGIoU (ms) | Ratio MGIoU/KFIoU |
|---:|---:|---:|---:|
| 32 | 3.1334 | 2.9647 | 0.9461x |
| 128 | 3.0146 | 3.0398 | 1.0084x |
| 512 | 2.7015 | 3.3149 | 1.2271x |

Kết luận pre-gate: **candidate**. Final B5 smoke vẫn **pending/blocked** vì
`data/kitti/processed` không tồn tại trong workspace. Theo ủy quyền rõ ràng
của user, `configs/kitti/loss_ablation/b5_mgiou_fixed.json` và selector
`mgiou` được expose riêng để chạy final two-train-batch smoke trên Colab
NVIDIA L4. Đây chỉ là candidate smoke artifact, **chưa research-qualified**
và không cho phép full B5 matrix/ranking cho tới khi gate final pass.

---

### Task 6: Tích hợp B0-B5 vào `LossFunction`

**Files:**

- Modify: `detector/core/losses/loss_fn.py`
- Modify: `tests/test_loss_ablation.py`

**Step 1: RED compatibility tests**

1. Old `name=baseline` giữ đúng focal + ba L1.
2. B0/old `name=uwag` giữ four-task `log_scales` và auxiliary 0.2.
3. B1 trả fixed `cls+offset+size+yaw`.
4. B2-B4 và B5 candidate trả fixed `cls+geometry`; B5 thêm integer
   `yaw_fallback_count` diagnostic.
5. Old UWAG state dict load strict.
6. Output dictionary giữ keys `loss/cls/offset/size/yaw/geo`.

**Step 2: Giữ orchestrator mỏng**

Resolve legacy/new schema trực tiếp trong constructor, validate bằng sets và
khởi tạo module geometry tương ứng. Mỗi class giữ công thức và numerical guard
của đúng objective; `LossFunction` chỉ chuẩn bị masked tensors, gọi `forward`
và cộng component losses. Không registry, factory hoặc class hierarchy mới.
`forward(pred,target,progress=0.0)` giữ default để caller cũ không vỡ. Các
docstring/comment của module phải nêu công thức toán, quy ước `[log_w,
log_l]`, doubled-yaw, FP32 solve/scale-relative regularization và link
paper/repo tham khảo.

**Step 3: GREEN**

```bash
python3 -m unittest tests/test_loss_ablation.py
python3 tests/test_mobile_bev.py --loss
```

Expected: PASS; old config behavior không đổi.

---

### Task 7: Progress, finite-gradient guard và opt-in final checkpoint

**Files:**

- Modify: `tools/kitti_training_pipeline/train.py`
- Modify: `tests/test_loss_ablation.py`
- Reuse: `tools/kitti_training_pipeline/common.py`

**Step 1: RED helper/resume tests**

- Planned attempts dùng `epochs * ceil(batches/accumulation)`.
- Resume B3 giữ `global_update_attempts` và switch point.
- Gradient `inf` bỏ toàn accumulation group; finite gradient step bình thường.
- Config thiếu `selection_policy` giữ legacy checkpoint behavior.
- B config `selection_policy=final_epoch` ghi `selected/final.pt`.

**Step 2: Implement tối thiểu**

- Truyền normalized progress vào criterion.
- Lưu global attempts/successful/skipped trong checkpoint.
- Sau `scaler.unscale_`, kiểm tra toàn bộ gradients finite trước step.
- Log component loss, gradient norm, skipped count, clamp count, active B3 form,
  B0 scales và B5 fallback count khi có.
- Đưa toàn bộ runtime controls (device, precision, physical/effective batch,
  accumulation, workers và smoke limits nếu có) vào resolved config.
- Khi resume, verify immutable source/split hashes và checkpoint SHA256 sidecar
  trước khi load; mỗi resume thành công ghi git/runtime metadata hiện tại.
- Epoch có zero successful optimizer updates không được publish final checkpoint
  hoặc `selected/final.pt`.
- `final_epoch` là opt-in; không đổi hành vi configs cũ.

**Step 3: GREEN**

```bash
python3 -m unittest tests/test_loss_ablation.py
python3 tests/test_mobile_bev.py --training-guard
```

Expected: PASS.

---

### Task 8: Full verification và smoke B0-B5

**Files:** Không thêm tệp.

**Step 1: Test suite**

```bash
python3 -m compileall detector tools tests
python3 -m unittest tests/test_loss_ablation.py
python3 tests/test_mobile_bev.py
python3 tests/test_regressions.py
python3 tests/test_evaluation_table.py
```

Expected: PASS; CUDA-only tests có thể SKIP rõ ràng.

**Step 2: Smoke B0-B4**

Chạy B0-B4 với:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/loss_ablation/CONFIG.json \
  --detector-root detector --output-root artifacts/kitti \
  --run-name CODENAME_smoke_seed42 --seed 42 \
  --epochs 1 --max-train-batches 2 --max-val-batches 1 --num-workers 0
```

Expected: finite metrics/gradients, zero skipped bình thường, immutable manifest,
split/source hashes và loadable final checkpoint. Resume phải verify checkpoint
sidecar trước khi load; epoch không có successful update không được publish
final checkpoint.

**Step 3: Hoàn tất B5 smoke gate trên Colab NVIDIA L4**

Candidate config đã được expose theo ủy quyền rõ ràng của user, chỉ để final
two-train-batch smoke. Trên Colab L4, dùng đúng CLI hiện có:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config configs/kitti/loss_ablation/b5_mgiou_fixed.json \
  --detector-root detector --output-root artifacts/kitti \
  --run-name B5_candidate_L4_smoke_seed42 --seed 42 \
  --device cuda --precision bf16 --epochs 1 \
  --max-train-batches 2 --max-val-batches 1 --num-workers 0
```

Yêu cầu gate vẫn là zero non-finite và zero skipped update. Chỉ kết quả smoke
thành công mới cho phép B5 được research-qualified và xem xét full matrix.
Không có KITTI smoke cục bộ nào được tuyên bố ở đây.

**Step 4: Chuẩn bị nhưng không chạy full matrix**

```text
B0-B4 x seed 42/43/44 = 15 runs total, gồm 3 historical-anchor runs
B5 x seed 42/43/44    = +3 runs chỉ khi gate pass
```

Primary ranking chỉ dùng B1-B5. Không chạy job 100 epochs trong implementation
turn; báo test/smoke/gate result và xin phép compute-heavy runs riêng.

## Tiêu chí hoàn thành

- A0-A4 docs/configs và old runtime behavior không đổi.
- B0-B4 configs tồn tại riêng, invariant tests PASS; B5 candidate được expose
  theo ủy quyền cho L4 smoke nhưng không research-qualified trước gate pass.
- Mỗi codename map tới đúng một frozen objective.
- `loss_fn.py` chỉ điều phối; Gaussian và MGIoU math nằm ở module riêng dưới
  dạng các class `nn.Module` (`KFIoULoss`, `ProbIoULoss`, `KLDLoss`, và gated
  `MGIoULoss`), không nhân bản công thức trong orchestrator.
- Unit/regression tests và smoke configs finite.
- B3 schedule resume đúng; non-finite gradients không update optimizer.
- Model/head/target/decoder/evaluator/export implementation không đổi.
- Không dependency, config inheritance, registry, commit hoặc push mới.
