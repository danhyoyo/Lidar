# Kế hoạch tái sử dụng Fake_opengt, không train lại GT baseline

Dùng N5 (SG F24/H32, hist14, grouped, baseline loss, IQA off) làm mốc bảng augmentation. N5 đạt 67,2803% local BEV Moderate AP. Đây là hybrid_gt_sampling tắt các kiểm tra bổ sung, được dùng như GT-sampling baseline/proxy; không mô tả là backend gt_sampling nguyên bản.

## Batch đầu: ba run mới

| ID | Recipe | Đối chứng |
|---|---|---|
| P01 | Hybrid bật ground/static/density/visibility, giống N6 nhưng F24/H32 | N5: tác dụng gói kiểm tra; N3: RC vs SG cùng F24/H32; N6: fusion F24 vs F32 trên hybrid |
| P02 | Standard trên cùng model N5 | Bảng augmentation |
| P03 | None trên cùng model N5 | Bảng augmentation |

Giữ 50epoch, warmup4, train/val batch16, seed42, AP every10/local_bev; cùng encoder, model, loss, quota15/10/10, probability0,5 ở các run sampling, global transforms và nguồn DB. P01 đổi đúng các cờ kiểm tra sampler so với N5; P02/P03 thay recipe augmentation. Full configs chỉ thêm thay đường dẫn local cho môi trường chạy; chưa training.

P01 với N3/N6 khác val batch16/32. Khi chốt so sánh, đánh giá cùng settings và cùng source; không cần train lại chỉ để thống nhất batch đánh giá. N5/P01 cùng val batch16 và policy chọn AP.

Không dùng N1 (probability1, attempts1, margin0) làm đối chứng trực tiếp cho N5/P01. Các run N4/N5/N7 là nhóm proxy cùng sampler settings.

## Sau batch đầu

1. Chọn augmentation dựa trên bốn recipe P03/P02/N5/P01. Không mặc định hybrid thắng.
2. Nếu dùng hybrid: P01 là bridge với N3 và N6. Chỉ chọn neck sau khi so cùng width/sampler. Chạy gate off từ neck được chọn; N2 chưa đủ vì quota khác N6.
3. Nếu cần full F24/F32 × H16/H32 bảng proxy: thêm C01_hybrid_constraints_off từ thư mục configs cấp trên. P01 không thay ô F32/H32 proxy này vì bật các checks khác.
4. Encoder rich8/hist14 và single/grouped dưới winner augmentation/neck: mỗi phần thêm một run, tái dùng hist14/grouped winner. Config C08/C09 cũ chỉ đúng nếu chọn N6 làm anchor; nếu winner khác phải sinh lại.
5. C4 context none/focal thêm một run trên anchor đã chọn; loss baseline, IQA off.
6. Loss baseline/UWAG/OGA thêm hai run; Q-OGA thêm một nếu cần. Các loss-term controls chỉ bắt buộc nếu claim cơ chế OGA.
7. IQA baseline/OGA off/on thêm hai run; off tái dùng loss table. Alpha sweep cùng checkpoint chỉ evaluation.
8. Hai baseline paper MobilePIXOR/NeXt35 cần lịch50epoch/AP-selection công bằng nếu đưa vào bảng incremental. Xác nhận nhiều seed và leave-one-out contributions trên final có ngân sách riêng.

LOS/shadow/radiometric chưa chạy. Nếu không đưa chúng vào proposed, không bắt buộc bật full. Nếu claim chúng, thêm individual controls và kết hợp.

## Số run

Batch đầu cần 3 run mới. Lộ trình từ batch augmentation qua gate, encoder, head, focal, loss baseline/UWAG/OGA và IQA cần khoảng 11 run mới nếu giữ các module và không thêm Q-OGA. Full proxy-width factorial, global-no-GT, baseline retraining, term controls, additional hybrid flags và confirmation seeds tính riêng. Số 11 là budget kế hoạch, không phải điều kiện đủ cho mọi claim của bài báo.
