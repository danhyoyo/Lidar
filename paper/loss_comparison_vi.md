# Cập nhật so sánh OGA và Q-OGA — 13/10/2026

Nguồn mới: `/home/duyennh/Downloads/drive-download-20261013T095105Z-1-001`. Hai run được ký hiệu **O100** (OGA) và **Q100** (Q-OGA) trong bản thảo.

## So sánh chính: checkpoint chọn theo AP

| Chỉ số | OGA | Q-OGA | Q-OGA − OGA |
|---|---:|---:|---:|
| Car Moderate AP | 84,88% | 84,77% | −0,11 điểm |
| Pedestrian Moderate AP | 57,06% | 61,27% | +4,21 điểm |
| Cyclist Moderate AP | 67,13% | 65,51% | −1,62 điểm |
| Moderate mAP | 69,69% | 70,52% | +0,82 điểm |
| Epoch được chọn | 70 | 70 | — |
| Tham số detector | 815.744 | 815.744 | 0 |
| Model forward trên L4 FP32 | 7,885 ms | 7,929 ms | +0,045 ms |
| Pipeline offline | 47,32 ms | 42,10 ms | −5,22 ms |

Hai config chỉ khác `loss.name`. Cả hai hoàn thành 100 epoch, seed 42, batch 16, BF16, AdamW, warmup learning rate 8 epoch, Hist14, focal context, SimAM và grouped head độc lập. Không có IQA branch; exponent nhân quality ở decoder bằng 0. Bốn report AP/loss có cùng split validation 3.769 frame, encoding semantic hash, input asset hashes và clean source commit. SHA-256 config và checkpoint khớp dữ liệu đã lưu.

Q-OGA tăng mAP trung bình chủ yếu nhờ Pedestrian; Car và Cyclist giảm nhẹ. Nó cao hơn OGA ở 9/10 lần đánh giá AP theo lịch, trừ epoch 20. Tại epoch 100, mAP là 70,32% so với 68,79% (+1,53 điểm). Các đường nối trong hình chỉ nối mười lần đo AP đã lưu.

Các chênh lệch được tính từ AP chưa làm tròn: 70,515097779% − 69,690111393% = 0,824986386 điểm, làm tròn thành +0,82 điểm.

Ở score threshold 0,05, FP giảm từ 118.795 xuống 77.704 (34,59%), tương ứng 31,52 xuống 20,62 FP/frame; micro recall là 89,49% và 89,67%. Đây là hai operating point cùng threshold, chưa phải so sánh cùng recall hoặc đánh giá calibration. Model forward gần tương đương; khác biệt pipeline chủ yếu đi cùng decode/NMS giảm từ 21,05 xuống 15,53 ms và số candidate output thấp hơn.

## Độ nhạy với tiêu chí chọn checkpoint

| Tiêu chí | OGA: epoch / Moderate mAP | Q-OGA: epoch / Moderate mAP | Chênh lệch |
|---|---|---|---:|
| Maximum AP trong lịch mỗi 10 epoch | 70 / 69,690111% | 70 / 70,515098% | +0,824986 điểm |
| Minimum validation loss từng epoch | 45 / 69,690913% | 44 / 69,796116% | +0,105203 điểm |

Checkpoint OGA chọn theo loss cao hơn checkpoint chọn theo AP chỉ 0,000801 điểm, đều làm tròn thành 69,69%. Epoch 45 nằm ngoài lịch AP mỗi 10 epoch. Bản thảo giữ cùng tiêu chí AP cho so sánh chính và đưa checkpoint loss vào phân tích độ nhạy.

Minimum validation objective là 0,988347 với OGA và 0,993990 với Q-OGA. Hai objective có định nghĩa và weighting khác nhau, vì vậy giá trị loss thô không dùng để kết luận objective nào phát hiện tốt hơn. Phụ lục có đường cong train/validation objective trên trục log, bao gồm đầy đủ epoch 1–100; AP là thước đo chung để so sánh.

## Các tệp đã cập nhật

Nội dung Abstract, Experimental Setup, Results, Discussion, Conclusion và Supplementary đã tích hợp cặp mới. Bảng factorial P1–P4 tiếp tục dùng các run 50 epoch tương ứng; P11 cũ chỉ còn trong bảng lịch sử của phụ lục. Không dùng chênh lệch giữa cặp 100 epoch mới và P4/P16 50 epoch để suy ra tác dụng riêng của IQA hoặc SimAM.

- [PDF chính](main.pdf), [PDF phụ lục](supplementary.pdf), [nguồn Overleaf](manuscript_source.zip).
- [Hình AP theo epoch](figures/loss_retrain.pdf), [hình train/validation objective](figures/loss_retrain_objectives.pdf).
- [So sánh số liệu chính xác và audit](research/loss_retrain_comparison.json); raw configs, metrics, selection và evaluations trong `research/loss_retrain/`.
- Bản trước cập nhật: `revisions/v1_before_loss_retrain_20261013.zip`.

Cặp mới vẫn dùng một seed 42. Chênh lệch trên là quan sát trong các run được cung cấp, chưa có ước lượng biến thiên giữa seed hoặc kết luận về ý nghĩa thống kê. Việc sửa bản thảo chỉ đọc kết quả có sẵn, vẽ lại và biên dịch tài liệu.
