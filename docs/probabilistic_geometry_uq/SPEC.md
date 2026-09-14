# Đặc tả nghiên cứu: ProbGeo-UQ cho LiDAR 3D Object Detection

Trạng thái: Draft v1

Ngày khóa phạm vi: 2026-09-14

Baseline mã nguồn: commit `f9c31e6`

Nguồn đề xuất: `Proposal.xlsx`, hàng 3, hướng “Xác suất Hình học & Định lượng Bất định”

## Tóm tắt quyết định

Xây dựng **ProbGeo-UQ** như một thay đổi trực giao trên đường `Center3D` hiện có:

1. Biểu diễn footprint hộp xoay bằng Gaussian 2D và thêm **Gaussian Wasserstein Distance (GWD)** làm geometric loss.
2. Dự đoán sáu log-variance phụ thuộc đầu vào cho `[x, y, z, log(w), log(l), log(h)]` và huấn luyện bằng Gaussian negative log-likelihood (NLL).
3. Giữ `yaw=[cos(2θ), sin(2θ)]` và loss yaw hiện tại. GWD xử lý tính chu kỳ modulo `π`; MVP không tuyên bố định lượng bất định hướng đầu/đuôi.
4. Giữ classification, backbone, voxelization, target và BEV NMS hiện tại. Bất định chỉ được xuất và đánh giá; không đổi score hoặc NMS trước khi chứng minh calibration tốt.

Đây là thay đổi nhỏ nhất kiểm tra được hai giả thuyết độc lập: geometric loss liên tục có cải thiện localization hay không, và variance học được có phản ánh sai số/sự thưa điểm tốt hơn các baseline rẻ hay không.

## Hai phân phối phải tách biệt

| Ký hiệu | Ý nghĩa | Nguồn |
|---|---|---|
| `Σ_box` | Hình dạng footprint của hộp, suy ra tất định từ `w,l,yaw` | Dùng trong GWD |
| `Σ_pred` | Bất định aleatoric học được của tham số hồi quy | Dùng trong NLL/calibration |

`Σ_box` không phải predictive uncertainty. Không dùng độ lớn của hộp làm “độ tin cậy”, không gọi GWD là uncertainty estimator, và không cộng hai covariance này như thể cùng một đại lượng.

## Kiểm tra hiện trạng repo

- `Header` hiện xuất bốn tensor: `cls`, `offset`, `size`, `yaw`.
- Với `box_encoding="center3d"`, mean regression có tám kênh: `offset=[dx,dy,z_center]`, `size=[log(w),log(l),log(h)]`, `yaw=[cos(2θ),sin(2θ)]`.
- `LossFunction(name="uwag")` hiện chỉ học bốn scale toàn cục theo task. Đây là **homoscedastic task weighting**, không phải per-object heteroscedastic uncertainty.
- Geometric term hiện tại là một xấp xỉ axis-aligned BEV IoU nhân yaw agreement, chưa phải rotated IoU/GWD.
- Decoder và evaluator đã hỗ trợ hộp Center3D 9 cột, BEV/3D AP R40 và distance bands.
- Test dataset đã trả raw `points`, nên có thể đếm điểm trong GT box ở evaluator mà không sửa dataset.
- ONNX/TensorRT đang khóa cứng đúng bốn output name ở ba nơi.
- Run UWAG đã lưu đạt local BEV mAP Moderate `79.7872%`, nhưng trở thành non-finite tại epoch 81. Loss mới phải tính ở FP32 và có giới hạn variance.
- MobileBEV-Lite A1–A4 đã code xong nhưng chưa được huấn luyện/đánh giá trong checkout này. Để không trộn hai đóng góp, thí nghiệm ProbGeo-UQ chính dùng A1 (`Legacy35 + CoordAtt + Center3D + sum-FPN`); chỉ xác nhận lại trên A4 nếu A4 đã vượt gate riêng.

## Sửa lỗi tài liệu tham khảo trong Proposal.xlsx

Link `https://arxiv.org/abs/2405.09942` ở hàng 3 là **FPDIoU Loss**, không phải GWD. Nguồn đúng cho GWD là Yang et al., ICML 2021: <https://proceedings.mlr.press/v139/yang21l.html>.

Không sửa workbook trong task đặc tả này. Khi cập nhật Proposal.xlsx, đổi nhãn/link hoặc ghi FPDIoU thành một phương án so sánh riêng.

## Goal

- Cải thiện tính liên tục và độ phù hợp hình học của regression loss cho hộp xoay.
- Ước lượng aleatoric localization uncertainty theo từng detection với chi phí inference rất nhỏ.
- Chứng minh uncertainty mang thông tin ngoài score, khoảng cách và số điểm LiDAR.
- Giữ đường legacy, PyTorch, ONNX và TensorRT tương thích theo config.

## Research questions và giả thuyết

**RQ1.** GWD có cải thiện localization của hộp xoay so với L1 + geometric loss hiện tại không?

- H1: thêm GWD làm giảm GWD error và không làm giảm 3D mAP Moderate quá 1.0 điểm; hiệu quả được báo riêng gần biên yaw và với hộp có aspect ratio lớn.

**RQ2.** Heteroscedastic NLL có học được localization uncertainty hữu ích không?

- H2a: NLL đã calibration của head thấp hơn constant-variance và `range + point_count` baseline trên test split.
- H2b: predicted uncertainty tương quan dương với absolute localization error và tăng theo range/độ thưa điểm, nhưng vẫn thắng baseline chỉ dùng hai biến đó.

**RQ3.** Kết hợp GWD + heteroscedastic NLL có tạo trade-off tốt nhất không?

- H3: model đầy đủ giữ AP trong non-inferiority margin, cải thiện NLL/risk-coverage, thêm không quá 1% tham số và không quá 5% model p95 latency.

Không kết luận từ một seed hoặc chỉ từ correlation plot.

## Phạm vi

### Trong phạm vi

- KITTI Car, Pedestrian, Cyclist; LiDAR-only, single-frame.
- Detector A1 Center3D hiện có; xác nhận A4 là thí nghiệm mở rộng có điều kiện.
- Aleatoric uncertainty cho center và size.
- Gaussian 2D footprint + GWD cho rotated BEV geometry.
- PyTorch BF16 training với loss nội bộ FP32; ONNX opset 17; TensorRT FP16 inference.
- Local KITTI-style BEV/3D AP R40, NLL, coverage, ENCE, sharpness, error ranking, range và point-count strata.

### Ngoài phạm vi MVP

- Epistemic uncertainty, MC dropout, ensemble và LiDAR-MIMO.
- Classification/object-existence calibration.
- Uncertainty của heading đầu/đuôi hoặc directed yaw; encoding hiện tại chỉ xác định orientation modulo `π`.
- Full 3D Gaussian GWD, correlated covariance, mixture density, Student-t/Laplace likelihood.
- Uncertainty-aware NMS, variance voting, tracking, motion classification và conformal prediction.
- Camera/temporal/sensor-level uncertainty và corruption benchmark đầy đủ.

Chỉ mở một mục ngoài phạm vi sau khi diagnostic của MVP chỉ ra đúng failure mode cần nó.

## Đặc tả toán học

### 1. Mean box hiện có

Tại mỗi positive cell, giữ mean target/prediction:

```text
r = [dx, dy, z_center, log(w), log(l), log(h)]
q = [cos(2*yaw), sin(2*yaw)]
```

- `dx,dy,z_center` có đơn vị mét.
- `log(w),log(l),log(h)` không có đơn vị.
- `q` biểu diễn orientation modulo `π` và được normalize với `epsilon` khi decode/GWD.
- Không đổi target map hoặc `reg_mask`.

### 2. Gaussian footprint cho GWD

Với box BEV `b=(x,y,l,w,θ)`:

```text
mu_box = [x, y]
R(theta) = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
Sigma_box = R(theta) diag([l^2/4, w^2/4]) R(theta)^T
```

Khoảng cách Wasserstein bậc hai:

```text
D_W^2 = ||mu_p - mu_t||_2^2
        + tr(Sigma_p + Sigma_t)
        - 2 * sqrt(tr(Sigma_p Sigma_t)
                   + 2 * sqrt(det(Sigma_p) det(Sigma_t)))
```

Loss chuẩn hóa:

```text
L_gwd = 1 - 1 / (1 + log(1 + max(D_W^2, 0)))
```

- Dùng closed form 2x2, không gọi eigendecomposition/matrix square root.
- Tính toàn bộ bằng FP32, clamp determinant/radicand ở `epsilon` chỉ để chống sai số số học.
- Average trên `reg_mask` như các regression loss hiện tại.
- Test bắt buộc: identical boxes, `θ` và `θ+π`, `(w,l,θ)` và `(l,w,θ+π/2)`, square box, non-overlap và finite gradient.

### 3. Heteroscedastic Gaussian NLL

Head phụ dự đoán `s=log(variance)` cho đúng sáu thành phần của `r`:

```text
L_nll = mean_mask,dim 0.5 * (exp(-s) * (r_hat-r)^2 + s)
s_used = clamp(s, log_var_min, log_var_max)
```

Quy tắc:

- `log_var_min=-7`, `log_var_max=4` là mặc định ban đầu; khóa trước full run và ghi saturation rate ở hai biên.
- Khởi tạo weight của nửa variance-output bằng 0 và bias bằng `initial_log_variance=-2`.
- NLL chạy FP32 dưới autocast; không dùng `nan_to_num` để che lỗi.
- Không áp NLL lên `yaw` trong MVP. `yaw` tiếp tục dùng masked L1; GWD cung cấp coupling hình học với `w/l`.
- NLL và UWAG global task weights không chạy đồng thời để tránh hai cơ chế cùng học cách giảm trọng số residual.

### 4. Total loss và ma trận 2x2

```text
L_total = L_cls + L_reg + L_yaw + lambda_gwd * L_gwd
```

| Variant | `L_reg` | `L_gwd` | Variance head |
|---|---|---:|---:|
| B0 | masked L1 | 0 | No |
| B1 | masked L1 | On | No |
| B2 | heteroscedastic NLL | 0 | Yes |
| B3 | heteroscedastic NLL | On | Yes |

- `L_cls`: modified focal loss hiện tại.
- `L_yaw`: masked L1 hiện tại trên doubled-angle vector.
- `lambda_gwd=0.2`, kế thừa scale của geometric term hiện tại; không tune sau khi xem test.
- UWAG chỉ là comparator lịch sử, không nằm trong factorial B0–B3.

## Head và output contract

### PyTorch output

Khi `predict_log_variance=false`, output giữ nguyên:

```text
cls, offset, size, yaw
```

Khi `predict_log_variance=true`, thêm:

```text
log_var = [log_var_dx, log_var_dy, log_var_z,
           log_var_logw, log_var_logl, log_var_logh]
```

Không tạo tower mới. Hai final convolution của `offset` và `size` xuất gấp đôi channels rồi split thành mean và log-variance. Với `in_channels=16`, Center3D chỉ thêm khoảng 102 tham số.

### Decoded prediction

- Deterministic Center3D giữ 9 cột: `class,score,x,y,z,w,l,h,yaw`.
- ProbGeo-UQ dùng 15 cột: 9 cột trên + 6 raw log-variance theo thứ tự cố định.
- AP/NMS chỉ đọc 9 cột đầu.
- File metadata phải ghi schema/version và nghĩa/đơn vị từng cột.
- Derived physical size uncertainty dùng delta approximation khi cần: `std(w) ≈ w*std(logw)`; luôn lưu raw log-space value để tái lập.

## Config contract

```json
{
  "model": {
    "box_encoding": "center3d",
    "predict_log_variance": true,
    "initial_log_variance": -2.0
  },
  "loss": {
    "name": "probgeo_uq",
    "gwd_weight": 0.2,
    "log_var_min": -7.0,
    "log_var_max": 4.0,
    "epsilon": 0.0001
  }
}
```

Tên loss mới:

- `deterministic`: B0.
- `gwd`: B1.
- `heteroscedastic`: B2.
- `probgeo_uq`: B3.
- `baseline` và `uwag` cũ giữ nguyên hành vi.

Validation tại config boundary:

- `predict_log_variance=true` bắt buộc cho `heteroscedastic` và `probgeo_uq`.
- `predict_log_variance=false` bắt buộc cho B0/B1 để không thêm output không được train.
- `log_var_min < initial_log_variance < log_var_max`.
- `gwd_weight >= 0`; Center3D regression dimension phải là 6 cho uncertainty.

## Protocol dữ liệu, model selection và calibration

- Training: giữ `splits/kitti/train.txt` (5,984 frame).
- Từ `splits/kitti/val.txt` (1,497 frame), tạo manifest cố định seed 42:
  - `uq_calibration.txt`: 500 frame, dùng chọn checkpoint và fit variance scale/baseline.
  - `uq_test.txt`: 997 frame, chỉ mở sau khi khóa checkpoint và calibrator.
- Báo thêm AP trên full 1,497-frame val để nối với lịch sử, nhưng primary UQ metrics chỉ lấy từ 997-frame test.
- Calibrator chỉ nhân variance theo từng dimension: `variance'_d = alpha_d * variance_d`.
- Fit analytic `alpha_d = mean(error_d^2 / variance_d)` trên matched calibration detections, clamp dương, rồi freeze.
- Báo cả raw và calibrated uncertainty. Không tune threshold/loss weight trên `uq_test.txt`.

## Matching population

Calibration localization dùng detection sau NMS:

1. Tách theo class và frame.
2. Xếp prediction giảm dần theo score.
3. Greedy one-to-one match như AP hiện có.
4. Chỉ đưa true positives ở IoU threshold KITTI (`0.7` Car, `0.5` Pedestrian/Cyclist) và difficulty Moderate vào regression-calibration metrics.

Điều này đo calibration có điều kiện trên các detection đã được định vị đủ đúng. Báo riêng:

- số matched samples theo class/stratum;
- failure-detection AUROC/AURC trên toàn bộ prediction, với failure là không match được theo cùng protocol;
- AP/recall để tránh một model “calibrated” bằng cách bỏ các detection khó.

## Uncertainty metrics

### Bắt buộc

- Gaussian NLL tổng và theo 6 dimensions.
- Empirical coverage tại `1σ` và `1.96σ`, kèm absolute coverage gap so với `68.27%` và `95%`.
- ENCE với 10 equal-count bins; không báo bin có dưới 30 samples.
- Sharpness: mean/median predicted standard deviation theo dimension.
- Spearman correlation giữa predicted std và absolute error.
- Risk-coverage/AURC khi loại dần detection theo uncertainty.
- AP R40, recall, latency, peak memory, model bytes và parameter count.
- Saturation rate của `s` tại `log_var_min/max`.

### Baseline/negative controls

- Constant variance theo dimension, fit trên calibration set.
- `range + point_count`: fit `log variance` bằng `numpy.linalg.lstsq` từ `1, log(1+range), log(1+points)`.
- Score-only ranking.
- Shuffled learned uncertainty, seed cố định.
- Oracle absolute error chỉ làm upper bound, không trình bày như phương pháp.

Learned uncertainty chỉ được xem là đóng góp nếu thắng constant và geometry-only baseline trên test, không chỉ có correlation dương.

### Conditional analysis

- Class: Car, Pedestrian, Cyclist.
- Range: `[0,30)`, `[30,50)`, `[50,70.4]` m.
- LiDAR points trong GT 3D box: `0–5`, `6–20`, `>20`.
- KITTI difficulty, occlusion, truncation.
- Aspect ratio và khoảng cách tới yaw decoding boundary.

Chỉ báo metric cho stratum đủ `n>=100`; nếu ít hơn, báo `n` và không diễn giải mạnh.

## Ma trận thí nghiệm

### Core factorial

| ID | Architecture | Regression | GWD | Seeds |
|---|---|---|---:|---|
| B0 | A1 Center3D | L1 | No | 42,43,44 |
| B1 | A1 Center3D | L1 | Yes | 42,43,44 |
| B2 | A1 Center3D | Heteroscedastic NLL | No | 42,43,44 |
| B3 | A1 Center3D | Heteroscedastic NLL | Yes | 42,43,44 |

Mọi field data/train/model khác phải byte-equivalent sau khi bỏ các field loss/UQ được phép khác.

### Xác nhận kiến trúc có điều kiện

Chỉ khi MobileBEV-Lite A4 đã vượt acceptance gate riêng, chạy thêm A4-B0 và A4-B3 với ba seed. Không huấn luyện đủ 4 biến thể trên cả A1/A4 trước khi biết A4 có đáng giữ hay không.

## Acceptance criteria

### Implementation gate

- Config legacy tạo đúng bốn output và load checkpoint cũ strict.
- UQ config tạo `log_var(6)`; mọi loss/output/gradient finite trong smoke run.
- GWD invariance và boundary tests pass; không có dependency mới.
- ONNX/TensorRT chấp nhận contract 4 hoặc 5 outputs và kiểm tra đúng shape.
- 9 cột đầu của decoded B0/B1 không đổi so với code hiện tại trên cùng synthetic tensor.

### Research gate

- B3 so với B0: lower bound của paired bootstrap 95% CI cho chênh lệch 3D mAP Moderate lớn hơn `-1.0` điểm.
- B1/B3: GWD error giảm so với counterpart không GWD; chỉ claim AP improvement khi 95% CI không cắt 0.
- B3 calibrated NLL tốt hơn constant và `range + point_count` baseline trên test theo paired bootstrap.
- Aggregate 68%/95% coverage gap không quá 5 điểm phần trăm; conditional gap không quá 10 điểm khi đủ mẫu.
- AURC tốt hơn score-only và shuffled uncertainty.
- Thêm không quá 1% parameters và không quá 5% model p95 latency trên cùng phần cứng.
- Ba seed được báo bằng mean ± standard deviation; mọi số truy ngược tới config/checkpoint/split hash.

Nếu AP tốt nhưng uncertainty không thắng baseline rẻ, kết luận chỉ là geometric-loss improvement. Nếu uncertainty tốt nhưng AP giảm ngoài margin, kết luận là trade-off, không gọi là detector tốt hơn toàn diện.

## Phải sửa những gì

| File | Sửa tối thiểu |
|---|---|
| `detector/core/models/heads/cnn.py` | Split mean/log-variance từ final conv của `offset` và `size`; legacy mặc định không đổi |
| `detector/core/models/model.py` | Truyền `predict_log_variance` và `initial_log_variance` vào `Header` |
| `detector/core/losses/loss_fn.py` | Thêm masked heteroscedastic NLL, closed-form 2D GWD, loss modes, finite/saturation diagnostics |
| `detector/postprocess.py` | Validate/select `log_var` cùng candidate/NMS và append 6 cột; 9 cột đầu giữ nguyên |
| `tools/kitti_training_pipeline/export_onnx.py` | Output names động: 4 legacy hoặc thêm `log_var`; metadata ghi schema |
| `tools/kitti_training_pipeline/build_tensorrt.py` | Smoke test chấp nhận đúng contract đọc từ ONNX metadata thay vì set khóa cứng |
| `tools/kitti_training_pipeline/evaluate_kitti_bev.py` | TensorRT shape contract động; AP đọc prefix 9 cột; lưu prediction schema 9/15 cột |
| `tools/kitti_training_pipeline/evaluate_uncertainty.py` | File mới: matching, calibration scale, baselines, coverage/ENCE/sharpness/AURC và point-count strata |
| `tools/kitti_training_pipeline/compare_models.py` | Thêm các cột UQ/overhead cần cho bảng B0–B3, không đổi cột legacy |
| `tests/test_mobile_bev.py` | Head/loss/GWD/decode/metric/config regression checks |
| `configs/kitti/probgeo_uq/b0_*.json` … `b3_*.json` | Bốn config factorial chỉ khác field được khóa |
| `splits/kitti/uq_calibration.txt`, `splits/kitti/uq_test.txt` | Hai manifest cố định, không overlap, hợp lại đúng val 1,497 frame |
| `tools/kitti_training_pipeline/README.md` | Lệnh train/evaluate/UQ, output schema và claim boundary |

Không cần sửa `detector/core/datasets/dataset.py`, BEV encoder, backbone hoặc `train.py` cho MVP. Target sáu mean đã có; training loop tự log mọi scalar trong loss dictionary.

## Risks và validity threats

- **Variance inflation:** log term, clamp, saturation logging và constant baseline ngăn diễn giải một variance lớn vô dụng là tốt.
- **Dense positive-map weighting:** training NLL vẫn theo `reg_mask` hiện có và có thể overweight box lớn. Detection-level test metrics là kết luận chính; chỉ thêm object-balanced target nếu diagnostic xác nhận bias.
- **Selection bias khi chỉ đo TP:** luôn báo AP/recall và failure-detection metrics cùng calibration.
- **Yaw identifiability:** doubled-angle/Gaussian footprint không phân biệt front/rear. Không dùng claim heading uncertainty hoặc downstream motion trong MVP.
- **Distribution assumption:** kiểm tra histogram/QQ của standardized residual. Chỉ thử Laplace/Student-t nếu residual test cho thấy Gaussian sai rõ.
- **Calibration leakage:** calibration/test manifests bất biến; test không dùng chọn checkpoint, scale hoặc threshold.
- **Annotation noise:** audit mẫu residual cao; không tự động coi mọi outlier là sensor uncertainty.
- **Current A4 not evaluated:** core result phải đứng độc lập trên A1; A4 chỉ là replication có điều kiện.
- **Preprint risk:** các paper 2026 chỉ là bối cảnh mới, không phải nền tảng duy nhất của phương pháp.

## Nguồn chính

- Kendall & Gal, “What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?”, NeurIPS 2017: <https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html>
- Feng et al., “Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection”, IV 2019: <https://arxiv.org/abs/1809.05590>
- Yang et al., “Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss”, ICML 2021: <https://proceedings.mlr.press/v139/yang21l.html>
- Yang et al., “Learning High-Precision Bounding Box for Rotated Object Detection via KLD”, NeurIPS 2021: <https://proceedings.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html>
- Yang et al., “Detecting Rotated Objects as Gaussian Distributions and Its 3-D Generalization”, 2022: <https://arxiv.org/abs/2209.10839>
- Meyer et al., “LaserNet: An Efficient Probabilistic 3D Object Detector”, CVPR 2019: <https://openaccess.thecvf.com/content_CVPR_2019/html/Meyer_LaserNet_An_Efficient_Probabilistic_3D_Object_Detector_for_Autonomous_Driving_CVPR_2019_paper.html>
- Hall et al., “Probabilistic Object Detection: Definition and Evaluation”, WACV 2020: <https://openaccess.thecvf.com/content_WACV_2020/html/Hall_Probabilistic_Object_Detection_Definition_and_Evaluation_WACV_2020_paper.html>
- Küppers et al., “Parametric and Multivariate Uncertainty Calibration for Regression and Object Detection”, 2022: <https://arxiv.org/abs/2207.01242>
- Schinagl et al., “GACE: Geometry Aware Confidence Enhancement”, ICCV 2023: <https://openaccess.thecvf.com/content/ICCV2023/html/Schinagl_GACE_Geometry_Aware_Confidence_Enhancement_for_Black-Box_3D_Object_Detectors_ICCV_2023_paper.html>
- Xu et al., “Rethinking Boundary Discontinuity Problem for Oriented Object Detection”, CVPR 2024: <https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Rethinking_Boundary_Discontinuity_Problem_for_Oriented_Object_Detection_CVPR_2024_paper.html>
- Pitropov et al., “LiDAR-MIMO”, IV 2022: <https://arxiv.org/abs/2206.00214>
- Schröder et al., “Taming Perception Jitter”, preprint 2026: <https://arxiv.org/abs/2606.09350>
- Beemelmanns et al., “Query2Uncertainty”, preprint 2026: <https://arxiv.org/abs/2605.05328>
