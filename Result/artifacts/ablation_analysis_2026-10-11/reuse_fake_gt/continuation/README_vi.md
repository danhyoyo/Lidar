# Bảng chạy tiếp ablation BEV — tái sử dụng Fake_opengt

## Goal

Hoàn thành các đối chứng augmentation, neck, encoder, head, C4 Focal, loss và IQA, tận dụng run hiện có. Chạy ngay P01/P02/P03; các bước sau sinh config từ winner thực tế. Không train lại GT baseline chỉ để đổi backend/tên.

## Assumptions

- N5 là GT sampling baseline/proxy đã thống nhất: NeXt depth3/4/2, hist14, SG F24/H32, grouped Car/Pedestrian+Cyclist, baseline loss, IQA off; AP 67,2803%. Implementation là hybrid_gt_sampling tắt ENABLE_*, cần mô tả đúng trong paper.
- Batch đầu khóa 50epoch, warmup4, AdamW lr0,0007, weight_decay0,001, train batch16, accumulation1, BF16, seed42, cùng train/val split; chọn best AP Moderate ba lớp, local BEV R40, AP every10. Val batch16. Các batch sau kế thừa lịch train và batch của anchor; giữ nhất quán trong từng cặp.
- GT runs dùng cùng DB từ training IDs, quota Car15/Ped10/Cyc10, p0,5, source_relative, attempts6, candidate multiplier3, margin0,1, cùng global queue. Không dùng N1 với p=1/attempts1 làm đối chứng matched cho N5/P01.
- Model giữ local_attention=none, detail_path=false, reparam=false trong bảng chính. Không đồng thời bật ECA khi kiểm tra focal.
- File config trỏ tới data/kitti/processed và data/kitti/raw. Chỉnh đường dẫn theo môi trường thật trước khi chạy. Script batch1 nhận output-root và raw-KITTI-root tùy chọn; raw root phải khớp config.
- Những run mới train từ đầu. Không warm-start từ model baseline vì sẽ đổi bài toán ablation. Resume chỉ cho đúng run bị gián đoạn, cùng config.

## Plan

1. Batch augmentation: chạy ba run sẵn sàng.
   - Files: `../configs/P01_hybrid_f24_h32.json`, `P02_standard_f24_h32.json`, `P03_none_f24_h32.json`, `run_batch1.sh`.
   - Change: P01 bật ground/static/density so với N5; P02 standard; P03 none. Model, encoder, head, loss và lịch train giữ nguyên.
   - Verify: kiểm tra `config.resolved.json`, `selected/selection.json`, `evaluation_ap_local_bev.json` và AP ba lớp. Chọn U* giữa N5/P01/P02/P03 theo cùng metric và kiểm tra trade-off từng lớp.
2. Batch neck: so width/topology rồi gate.
   - Files: dùng `generate_stage.py --stage neck_candidates` từ U* trên SG F24/H32.
   - Change: X02 SG F32/H32, X03 RC-SGFPN mul F24/H32. Nếu U*=hybrid, N6/N3 đã có các cấu hình này; tái dùng khi train DB/split/source tương thích và thống nhất evaluation. Nếu U* khác, train X02/X03 mới, không ghép kết quả hybrid vào bảng augmentation khác.
   - Verify: P01↔N3 so neck cùng F24/H32; P01↔N6 so fusion width cùng H32. Giữ SG gated mạnh nhất làm anchor cho `--stage gate`, chạy P04 gateoff cùng width. P04 không nhận RC anchor vì boolean scale_gated_fpn không cô lập RC gate. Chọn A* giữa gated SG, ungated SG và RC candidate.
3. Batch encoder: hist14/rich8 trên A*.
   - Files: `generate_stage.py --stage encoder` sinh P05.
   - Change: chỉ đổi encoding/channels; hist14 A* tái dùng. Chọn E* là encoder thắng; không mặc định hist14 thắng.
   - Verify: đối chiếu augmentation/model/loss/train/val không đổi; lưu AP/latency và config identity.
4. Batch head: grouped/single trên E*.
   - Files: `generate_stage.py --stage head` sinh P06.
   - Change: single bỏ head_groups/group_weights, giữ width và các module khác. So với E* grouped, chọn H*.
   - Verify: số lớp, mapping, target và group aggregation; báo đây là grouped-head package, không chỉ riêng một conv layer.
5. Batch C4: none/focal trên H*.
   - Files: `generate_stage.py --stage context` sinh P07.
   - Change: focal v1, bottleneck64, dilations[1,2,3], LayerScale0,001; baseline loss/IQAoff. Chọn F*: architecture/augmentation cuối cho bảng loss/IQA.
   - Verify: context fields, tensor shapes/parameter budget trong môi trường PyTorch và AP/latency; không bật local attention hoặc đổi neck cùng run.
6. Batch loss: ba strategies, IQAoff.
   - Files: `generate_stage.py --stage loss` sinh P08/P09 từ chính F* baseline-loss resolved config.
   - Change: L0 tái dùng F* baseline; P08 UWAG; P09 OGA. Cùng architecture, augmentation, target assignment và schedule.
   - Verify: flags IQAoff, loss registry name, criterion parameters, seed/split/config. Đây là so sánh gói loss; OGA term controls nằm ở bảng conditional nếu claim từng cơ chế.
7. Batch IQA và scoring.
   - Files: `generate_stage.py --stage iqa` sinh P10/P11 từ chính F* baseline-loss resolved config, không lấy P09 làm source generator.
   - Change: P10 baseline+IQA, P11 OGA+IQA; cùng rotated_iou target BEV và weight1. Hai flag header_use_iou/loss.use_iou cùngtrue, alpha0 khi chọn checkpoint.
   - Verify: bảng2×2 gồm L0 baselineoff, P09 OGAoff, P10 baselineon, P11 OGAon. Sweep alpha[0,.25,.5,.75,1] trên cùng checkpoint P10/P11 là evaluation, không train mới; alpha ảnh hưởng scoring/peaks/threshold, không chỉ NMS.
8. Mốc baseline và xác nhận paper.
   - Files: `configs/B01_mobilepixor_50ep.json`, `B02_next_legacy35_50ep.json`; configs final sinh sau khi freeze.
   - Change: hai baseline có 50epoch/warmup4/AP-selection để thay mốc lịch100epoch/loss-selection cũ. Đây là package comparison; nếu claim extractor thuần cần matched-head/depth controls.
   - Verify: paired seeds42/43/44 cho baseline/final và claim chính; báo mean/std, leave-one-out các contributions trên full model và kiểm tra augmentation runner-up trên final architecture. Mục tiêu metric hiện tại là local ROI BEV; không ghi thành official KITTI test AP.

## Bảng chạy chính

| Batch | ID | Cấu hình / thay đổi | Đối chứng | Trạng thái |
|---|---|---|---|---|
| Aug | N5 | GT baseline/proxy, hist14, SG F24/H32, grouped | P01/P02/P03 | Tái dùng, AP 67,28% |
| Aug | P01 | Hybrid ground/static/density, cùng model N5 | N5 | Chạy ngay |
| Aug | P02 | Standard, cùng model N5 | N5/P01/P03 | Chạy ngay |
| Aug | P03 | None, cùng model N5 | P02/N5/P01 | Chạy ngay |
| Neck | X02 | U*, SG gated F32/H32 | U* SG F24/H32 | Tái dùng N6 nếu matched hybrid; nếu không, chạy mới |
| Neck | X03 | U*, RC mul F24/H32 | U* SG F24/H32 | Tái dùng N3 nếu matched hybrid; nếu không, chạy mới |
| Gate | P04 | Gateoff trên SG gated mạnh nhất, cùng F/H | SG gated cùng recipe | Sau batch Aug/Neck |
| Encoder | P05 | Rich8 trên A* | Hist14 A* | Sau chốt neck |
| Head | P06 | Single trên E* | Grouped E* | Sau chốt encoder |
| C4 | P07 | Focal trên H*, baseline/IQAoff | Contextnone H* | Sau chốt head |
| Loss | L0 | Baseline/IQAoff trên F* | P08/P09 | Tái dùng F* |
| Loss | P08 | UWAG/IQAoff trên F* | L0/P09 | Sau chốt architecture |
| Loss | P09 | OGA/IQAoff trên F* | L0/P08 | Cùng anchor P08 |
| IQA | P10 | Baseline/IQAon, rotated_iou, alpha0 | L0 | Cùng architecture F* |
| IQA | P11 | OGA/IQAon, rotated_iou, alpha0 | P09 | Cùng architecture F* |
| Scoring | Qeval | Alpha0/.25/.5/.75/1 | Cùng checkpoint P10/P11 | Chỉ evaluate |
| Baseline | B01 | MobilePIXOR legacy35 standard,50ep/AP | B02/final | Config sẵn, batch paper |
| Baseline | B02 | NeXt legacy35 standard,50ep/AP | B01/final | Config sẵn, batch paper |
| Confirmation | FINAL | Paired seeds và final leave-one-out | Baseline/final | Sau freeze |

N* là mã audit của run gốc. File `run_table.csv` có các cột điền AP trung bình/từng lớp, best epoch và notes. `conditional_controls.csv` liệt kê các controls phụ và điều kiện cần chạy.

## Lệnh batch đầu

Chạy từ repository root sau khi kiểm tra đường dẫn dữ liệu và môi trường PyTorch:

```bash
bash artifacts/ablation_analysis_2026-10-11/reuse_fake_gt/continuation/run_batch1.sh
```

Script train rồi evaluate P01, P02, P03 lần lượt, seed42, output `artifacts/paper_bev_continuation`; không chạy nhiều training job cùngGPU. Không tự ghi đè run đã có.

Nếu chỉ chạy P01 trước:

```bash
python3 tools/kitti_training_pipeline/train.py \
  --config artifacts/ablation_analysis_2026-10-11/reuse_fake_gt/configs/P01_hybrid_f24_h32.json \
  --detector-root detector --output-root artifacts/paper_bev_continuation \
  --run-name paper_P01_hybrid_f24_h32_s42 --seed 42
```

## Sinh config sau khi có winner

Ví dụ P01 thắng augmentation, sinh neck candidates từ resolved config thật:

```bash
python3 artifacts/ablation_analysis_2026-10-11/reuse_fake_gt/continuation/generate_stage.py \
  --stage neck_candidates \
  --anchor artifacts/paper_bev_continuation/paper_P01_hybrid_f24_h32_s42/config.resolved.json \
  --output-dir artifacts/paper_bev_continuation/configs_after_aug
```

Nếu P02/P03 thắng, thay anchor bằng resolved config của run đó. Nếu N5 thắng, dùng config.resolved.json N5. X02/X03 được sinh để kiểm tra config; chỉ train khi chưa có run matched tương ứng.

Các stage tiếp theo: `gate`, `encoder`, `head`, `context`, `loss`, `iqa`. Với mỗi stage, thay anchor bằng winner đúng bước và dùng output directory mới. Loss/IQA cùng dùng F* baseline/IQAoff. Generator giữ nguyên augmentation và training settings của source, kiểm tra schema/flag compatibility và từ chối ghi đè config khác trong cùng output path. Không chọn winner tự động và không launch training.

## Ngân sách

- Batch đầu: 3 run mới.
- Nếu hybrid thắng và tái dùng được N3/N6: chuỗi P01–P11 cần 11 run mới ở seed 42. P04/05/06/07/08/09/10/11 là 8 run sau batch đầu.
- Nếu augmentation khác thắng: thường thêm 2 neck candidates X02/X03, tổng 13 run cho chuỗi khảo sát chính.
- Hai baseline paper B01/B02 thêm 2 run. Confirmation seeds/final leave-one-out có budget riêng.
- Đây là ngân sách khảo sát, không phải mọi claim của bài báo đều được xác nhận chỉ bằng 11/13 run.

## Risks & mitigations

- Chọn module theo 1 seed có thể bỏ qua interaction; giữ runner-up và kiểm tra trên final, đặc biệt augmentation×encoder/context, loss×IQA.
- Visibility flag bật ở P01 nhưng nhánh kiểm tra không hoạt động với shadowoff, extra_width0 và margin0,1. Không ghi nhận đây là ablation chứng minh visibility. Nếu dùng shadow, visibility cần kiểm tra theo cơ chế thực sự.
- Nếu claim từng thành phần hybrid: thêm groundoff/staticoff/densityoff từ winner. LOS/shadow/radiometric riêng và combined chỉ cần nếu dùng/claim chúng; không mặc định bật full tốt nhất.
- Nếu claim multiscale C4 Focal: thêm dilations[1] và[1,2] để so với[1,2,3], giữ D64/hook/architecture.
- Nếu claim cơ chế OGA mới: LG1 SmoothL1only, LG2 +corner fixed, LG3 +MGIoU fixed, LG4 +both fixed; P09 adaptive làm mốc cuối. Code hiện chưa cung cấp đủ term switches để chạy các đối chứng này chỉ bằng đổi loss.name; phải bổ sung implementation trước.
- N5/N3/N6 cùng đường dẫn DB trong config chưa đủ xác nhận train assets bằnghash. Kiểm tra DB chỉ từ training IDs và manifest disjoint; record audit trong các run mới.
- Các run có GT hiện đạt best AP ở epoch 50. Đây là budget50, chưa chứng minh hội tụ. Nếu tăng budget, tăng đồng nhất các cặp và giữ cùng LR schedule.
- Chuẩn hóa evaluation source/settings khi tái dùng checkpoint cũ. Không dùng chênh latency e2e chứa I/O/encoding để quy riêng cho neck/head.

## Rollback plan

Chỉ thêm plan/config/script trong ignored artifacts. Không sửa source/master config, không đổi checkpoint Downloads. Module không tăng AP có thể được bỏ khỏi recipe final; vẫn lưu negative result và resolved configs. Nếu thay anchor, dùng output directory/run name mới; không merge tùy tiện các preset toàn phần làm đổi những module ngoài đối chứng.
