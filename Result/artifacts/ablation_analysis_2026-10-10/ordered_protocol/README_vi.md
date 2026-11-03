# Thứ tự ablation BEV từ MobilePIXOR đến full model

### Goal

Xây dựng chuỗi thí nghiệm có đối chứng: **MobilePIXOR → MobilePixorNeXt/depth → augmentation → encoder → neck → head → C4 Focal → loss → IQA → xác nhận final**. Bảng kiến trúc dùng baseline loss và IQA tắt; loss/IQA có bảng riêng. Proposed là thiết kế được chọn sau kiểm chứng, chưa mặc định mọi module đều có lợi.

### Assumptions

- Mục tiêu BEV. Screening dùng local ROI BEV R40; chưa ghép số liệu này với reference APBEV của các bài khác.
- Vòng đầu: seed42; 50 epoch/warmup4; AdamW lr0,0007, min_lr1e-6, weight_decay0,001; batch train16/accumulation1, val16; BF16 train, FP32 eval; AP mỗi10 epoch và final; chọn Moderate macro AP của Car/Pedestrian/Cyclist; score0,05, NMS0,1, maxdet500; cùng grid0,1m và ROI.
- Nếu dùng 100 epoch cho final, các đối chứng chính phải cùng lịch mới. Không resume lịch50 thành100 rồi coi tương đương train100 từ đầu. Best AP hiện là tốt nhất trong các epoch đã đánh giá.
- Mỗi ablation train từ đầu; không warm-start riêng proposed. Cùng seed, manifests, processed data, GT database train-only và source code. Đổi run name khi đổi cấu hình/seed.
- Tám full config đầu nằm trong `configs/` cạnh báo cáo. Đã kiểm tra pure detection/encoding/selection schemas; chưa dựng model hoặc train vì Python hiện không có PyTorch. Cần profile và smoke trên môi trường train.
- Các template augmentation dùng depth3/4/2 như reference hiện có. Nếu chọn depth2/4/2, sửa đồng nhất tất cả template; không tái dùng kết quả depth3 làm đối chứng depth2.

### Plan

1. **Backbone và head interface: B0/B1/B2/B3.**
   - Files: `configs/B0_mobilepixor_legacy35_original_head.json`, `B1_mobilepixor_legacy35_bn_silu_head.json`, `B2_mobilepixornext_legacy35_depth242.json`, `B3_mobilepixornext_legacy35_depth342.json`.
   - Change: B0 MobilePIXOR, legacy35, single H16, head không BN/activation. B1 chỉ thêm BN+SiLU vào head. B2 thay bộ trích xuất bằng NeXt depth2/4/2, cùng head H16. B3 chỉ tăng stage stride4 từ2 lên3 block. Giữ standard OneOf, baseline loss, IQA/gate/context tắt.
   - Verify: profile NeXt bằng `python3 tools/benchmarks/profile_detector.py --config artifacts/ablation_analysis_2026-10-10/ordered_protocol/configs/B2_mobilepixornext_legacy35_depth242.json`. MobilePIXOR dùng parameter report của evaluator vì profiler này chỉ hỗ trợ NeXt. Train/evaluate full split; đối chiếu resolved configs và AP/params/latency.
   - Decision: B1−B0 đo gói BN/activation của head; B2−B1 đo bộ trích xuất NeXt; B3−B2 đo depth. MobilePIXOR có deconv neck, NeXt có bilinear/FPN tích hợp: B2−B1 không cô lập riêng block backbone. Muốn claim riêng block cần đối chứng dùng chung neck/adapters.
   - Existing: R0 cũ log38/100, R1 cũ100epoch; chọn loss và head khác nhau. Không thay cho cặp matched50/AP-policy này.

2. **Augmentation: bốn preset notebook U0/U1/U2/U3, trên một rich8 reference cố định.**
   - Files: `configs/U0_rich8_none.json`, `R0_reference_rich8_standard.json` (U1), `U2_rich8_openpcdet_gt_preset.json`, `U3_rich8_hybrid_gt_preset.json`.
   - Change: U0 none; U1 standard; U2 openpcdet_gt; U3 hybrid_gt. Giữ NeXt depth3/4/2, rich8, single H16/F24, baseline loss, IQA/gate/context tắt; cùng split, seed và lịch train. Reference legacy35→rich8 là bước nối, chưa kết luận rich8 tối ưu.
   - Verify: GT DB chỉ từ training IDs. Preset hiện tại: openpcdet_gt Car15/Ped10/Cyc10, mặc định p=1; hybrid_gt Car8/Ped6/Cyc6, p=0,5. Saved hybrid runs khác preset hiện tại. Bảng này so sánh toàn bộ recipe, chưa cô lập cơ chế hybrid.
   - Decision: chọn U* theo BEV AP và chi phí sau khi so cả bốn preset; không mặc định hybrid. Xác nhận runner-up trên final model nếu kết quả gần nhau.
   - Đối chứng bổ sung G1/G2/G3 nếu claim GT sampling/hybrid: giữ ba file cũ `U1_rich8_global_no_gt.json`, `U2_rich8_standard_gt_matched.json`, `U3_rich8_hybrid_gt_matched.json`. G1 dùng backend `openpcdet_global` không GT; G2/G3 cùng global, DB, quotas Car6/Ped8/Cyc10, p=0,5 và PREPARE. G2−G1 đo thêm GT sampling; G3−G2 đo thay sampler với các thiết lập chung được giữ cố định. Các đối chứng này không thay bốn preset chính.

3. **Encoder: E0/E1/E2/E3/E4.**
   - Files: full configs sinh từ U* resolved config; thay toàn bộ `data.bev_encoding` bằng schema từng variant. Saved schemas nằm trong `../audit.json`.
   - Change: E0 rich8 (tái dùng U*); E1 binary_slices/legacy35; E2 rich10; E3 rich12; E4 hist14. Khóa depth*, single H16/F24, augmentation U*, gate/context tắt, baseline loss và IQA tắt.
   - Verify: `bev_encoding_spec` và `resolve_detection_config`; profile đúng channels/input shape; báo AP từng lớp và thời gian encode/H2D/model/pipeline. Thay encoding cần training mới.
   - Decision: chọn E* theo accuracy/latency budget. Có thể giữ rich8/hist14 như hai phương án khi tradeoff chưa rõ. R2/R3/R4/R8 là screening standard đã có; chỉ tái dùng nếu U*=standard và source/depth/head/schedule matched. Nếu U* khác standard, phải chạy encoder comparisons dưới U*.
   - Mechanism: nếu claim histogram hoặc xy-offsets riêng, thêm schemas bỏ histogram/bỏ offsets; các toggle này chưa có sẵn, cần triển khai/kiểm chứng.

4. **Neck: gate trước, width sau.**
   - Files: full configs từ E*; `detector/core/backbone_config.py`, `detector/core/models/backbones/mobilepixornext.py`.
   - Change: N0 additive FPN F24/H16 (tái dùng E*); N1 chỉ bật scale gate ở F24/H16. Chọn gate G*. Với cùng G*: N2 F32/H16, N3 F24/H32, N4 F32/H32, so với F24/H16.
   - Verify: profile actual params/output shapes, AP và pipeline latency. N2 đo fusion width; N3 thay cả output projection và head input capacity, không phải neck-only; N4 kiểm tra phối hợp hai widths.
   - Decision: chọn N*, không chốt32/32 trước. Width là capacity tuning, không tự thành contribution mới.
   - Optional: nếu bài claim RC/Bi neck, so SG-FPN gated / RC-SGFPN mul / RC-SGFPN add / RC-BiSGFPN tại cùng F24/H và input. Hiện RC không hỗ trợ `neck_fusion_channels=32`; không gán chênh lệch giữa RC F24 và SG F32 cho riêng range conditioning.

5. **Head: H0/H1.**
   - Files: configs từ N*; `detector/core/detection_config.py`, `detector/core/task_groups.py`.
   - Change: H0 single (tái dùng N*); H1 grouped Car/(Ped,Cyc), cùng BN+SiLU/H và các phần đã khóa. Khai báo head groups; weights bằng nhau, resolver chuẩn hóa0,5/0,5.
   - Verify: actual head params, AP từng lớp/macro, decode/NMS latency. So H1−H0 là gói targets/head/loss reduction; capacity cũng thay đổi. Nếu cần chứng minh cơ chế ownership, phân tích collision hoặc thêm capacity-matched single.
   - Decision: chọn H*. Nếu claim partition này tối ưu, thêm ba nhóm riêng Car/Ped/Cyc.
   - Existing: R5 grouped49,01 dùng lịch100/warmup8 và log70epoch, không thay H1 matched50/warmup4.

6. **C4 Focal: F0/F1, trước loss.**
   - Files: configs từ H*; `detector/core/models/backbones/focal_context.py`.
   - Change: F0 context none (tái dùng H*); F1 focal ở stride8, D64, dilations[1,2,3], LayerScale1e-3. Giữ baseline loss/IQA tắt; không bật ECA cùng run F1.
   - Verify: feature config validation, full shape/budget, AP/latency và distance diagnostics với số GT. Nếu claim multiscale, so [1], [1,2], [1,2,3] tại cùng D/hook; nếu claim global hoặc learned fusion cần controls bỏ global/uniform fusion trong code.
   - Decision: chọn F*. Kiến trúc A lúc này đã cố định: backbone/depth + E* + N* + H* + F*. Focal không giúp thì loại khỏi final, cập nhật schematic.

7. **Loss riêng: L0/L1/L2/L3.**
   - Files: configs từ A; `detector/core/losses/strategies/{baseline,uwag,oga,q_oga}.py`.
   - Change: L0 baseline (tái dùng F*); L1 UWAG; L2 OGA; L3 Q-OGA. Giữ Gaussian classification, `header_use_iou=false`, `loss.use_iou=false`; mọi phần khác khóa. Train từng strategy từ đầu.
   - Verify: cùng AP selection; AP từng lớp, localization errors, criterion state/weights. Không so raw validation loss giữa objectives.
   - Mechanism nếu OGA là contribution: thêm SmoothL1 không geometry; SmoothL1+corner fixed weights; SmoothL1+MGIoU fixed weights; SmoothL1+cả hai fixed weights. So với baseline L1 và full OGA. OGA hiện đổi cả L1→SmoothL1, geometry và weighting; controls chưa exposed đủ, cần triển khai trước.
   - Decision: chọn loss theo accuracy/ổn định, không theo tên proposed. Q-OGA quality supervision trong classification khác nhánh IQA riêng.

8. **IQA riêng: Q0/Q1/Q2/Q3.**
   - Files: configs từ A; `detector/core/losses/iou_targets.py`, `detector/postprocess.py`, `tools/kitti_training_pipeline/evaluate_kitti_bev.py`.
   - Change: Q0 baseline IQA-off (L0); Q1 OGA IQA-off (L2); Q2 baseline IQA-on; Q3 OGA IQA-on. Bật cùng lúc header_use_iou/use_iou; khóa target và weight. `rotated_iou` là BEV IoU thật; `mgiou` là proxy, phải ghi đúng. Baseline/OGA đều hỗ trợ IQA, UWAG/Q-OGA hiện không hỗ trợ.
   - Verify: chọn checkpoint với alpha0 cho đối chứng auxiliary supervision; trên cùng checkpoint evaluate alpha[0,.25,.5,.75,1] bằng `--nms-alpha`. Sweep không cần retrain. Alpha0 không tắt quality training. Alpha>0 đổi gói quality scoring, có thể ảnh hưởng peaks/score cutoff ngoài NMS ordering; không gọi ranking-only nếu chưa kiểm soát các bước đó.
   - Decision: Q2−Q0/Q3−Q1 tại alpha0 đo auxiliary branch; alpha>0−alpha0 đo scoring. Nếu đổi quality target cần train lại. Khóa alpha cuối theo validation, xác nhận nhiều seed.

9. **Xác nhận final, leave-one-out và paper comparison.**
   - Files: final `config.resolved.json`, manifests, checkpoint hashes, evaluation CSV/JSON.
   - Change: giữ các module đã chọn. Paired seeds42/43/44 cho baseline/final và ablations hỗ trợ claim chính. Trên final architecture/objective, so U* với đối thủ augmentation mạnh nhất để kiểm tra interaction. Final bỏ từng contribution, giữ mọi phần khác; early-stage ablation không tự thay leave-one-out trên full model.
   - Verify: full Easy/Moderate/Hard ba lớp, Moderate macro mean±sample std; params backbone/neck/heads/total/criterion, MACs đúng input; latency encode/H2D/model/NMS/pipeline cùng hardware/precision. Kiểm tra source/split/train-only database.
   - Protocol: reference wrapper hiện yêu cầu checkpoint3D, nên cần bổ sung/kiểm chứng BEV path trước khi so trực tiếp với APBEV paper khác. Internal local ROI BEV dùng được cho screening, ghi đúng nhãn. Main comparison với external methods là bảng riêng; dataset/evaluator/sampling/class domain phải tương thích.
   - Decision: ours = A + U* + objective/quality settings đã xác nhận, với resolved config và checkpoint cụ thể.

### Risks & mitigations

- Chọn từng bước theo1seed có thể bỏ module có interaction. Giữ runner-up và xác nhận trên full model, đặc biệt augmentation×encoder/context và loss×IQA.
- Tên neck SG-FPN không chứng minh gate đang bật; đọc flag. Thay H16→H32 tăng head capacity. Đặt tên so sánh đúng gói thay đổi.
- Hai anchor hybrid cũ và các preset hiện có khác settings; không ghi đè checkpoint cũ hoặc ghép AP từ validation1497 với validation3769.
- Các config đầu chỉ qua pure validation, chưa CUDA/model smoke. Controls chi tiết của loss/context/encoder chưa có đủ, không tự thêm JSON flag không được hỗ trợ.
- Vòng cơ bản có tối đa khoảng22 run mới ở seed đầu nếu chạy đủ bốn backbone controls, ba augmentation, bốn encoder mới, bốn neck variants, một grouped, một focal, ba loss và hai IQA. Reference R4 cần chạy lại nếu không matched. Old encoder runs có thể giảm số run khi U*=standard. Thêm4 OGA mechanism controls nếu claim OGA; optional neck/context/head, final leave-one-out/seeds/external comparison có budget riêng.
- CSV có cả aliases, optional và evaluate-only: không đếm40 dòng CSV thành40 run training.

### Rollback plan

Mọi đầu ra mới nằm trong ignored `artifacts/ablation_analysis_2026-10-10/ordered_protocol`; không sửa source/master config/Downloads checkpoints. Module thất bại được bỏ bằng full config mới; giữ negative results và source/config identity, không xóa run không tăng AP.

## Lệnh cho batch đầu

Chạy từ repository root trên môi trường đã có PyTorch và dữ liệu. Full configs dùng `data/kitti/processed` và `data/kitti/raw`; nếu Colab, sửa hai đường dẫn này theo location thực tế.

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config artifacts/ablation_analysis_2026-10-10/ordered_protocol/configs/B0_mobilepixor_legacy35_original_head.json \
  --detector-root detector --output-root artifacts/paper_bev_ordered \
  --run-name paper_B0_mobilepixor_legacy35_original_s42 --seed 42

python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name paper_B0_mobilepixor_legacy35_original_s42 --backend pytorch \
  --model artifacts/paper_bev_ordered/paper_B0_mobilepixor_legacy35_original_s42/selected/best_ap.pt \
  --config artifacts/paper_bev_ordered/paper_B0_mobilepixor_legacy35_original_s42/config.resolved.json \
  --detector-root detector --kitti-root data/kitti/raw --split splits/kitti/val.txt \
  --nms-alpha 0 \
  --output artifacts/paper_bev_ordered/paper_B0_mobilepixor_legacy35_original_s42/evaluation_ap_local_bev.json
```

Thay config/run name cho B1/B2/B3. Sau khi quyết định depth, kiểm tra reference/standard U1 rồi chạy none U0, openpcdet_gt U2 và hybrid_gt U3. Các config giai đoạn sau được sinh từ resolved winner, không merge nguyên preset làm thay đổi các trường khác.
