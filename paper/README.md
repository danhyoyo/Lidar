# MobilePixorNeXt manuscript

Bản thảo tiếng Anh được chuẩn bị theo cấu trúc IMRaD và định dạng IEEE journal. Hướng journal đề xuất là **IEEE Transactions on Intelligent Vehicles**; IEEE IV/ITSC là lựa chọn hội nghị. Xem [chiến lược nơi nộp](venue_strategy_vi.md) và [phân tích tiếng Việt](analysis_vi.md).

**Cập nhật 13/10/2026:** đã thay so sánh OGA–Q-OGA chính bằng hai run retrain 100 epoch mới. Q-OGA đạt 70,52% so với OGA 69,69% (+0,82 điểm); cả hai chọn epoch 70. Chi tiết nằm trong [phân tích loss mới](loss_comparison_vi.md).

## Tệp chính

| Tệp | Nội dung |
|---|---|
| `main.pdf`, `main.tex` | Bản thảo chính 9 trang và nguồn LaTeX |
| `supplementary.pdf`, `supplementary.tex` | Phụ lục 13 trang: full ablation, loss retrain, provenance, encoder, demo và diagnostic |
| `manuscript_source.zip` | Nguồn LaTeX, BibTeX và các hình thực sự dùng trong hai tài liệu, sẵn để upload Overleaf |
| `references.bib` | 23 nguồn thực, dùng BibTeX IEEE |
| `research/citation_audit.md` | Nguồn xác minh và claim tương ứng từng citation |
| `figures/` | Diagram vector có sẵn, hình Demo và plot từ kết quả đã lưu |
| `figures/source_diagrams.pptx` | Bản sao nguyên vẹn PowerPoint 25 slide của bạn |
| `tables/` | Các bảng LaTeX lấy từ JSON kết quả |
| `research/` | Config, raw/summarized evidence, split, nguồn và hash audit |
| `scripts/prepare_assets.py` | Tập hợp bằng chứng và vẽ plot từ kết quả hiện có |
| `scripts/loss_retrain_assets.py` | Đọc và đối chiếu cặp O100/Q100, tạo bảng và đường cong loss/AP |
| `scripts/build.py` | Biên dịch cả hai PDF bằng latexmk hoặc Tectonic |
| `research/document_checks.json` | Kết quả kiểm tra PDF, citation, tham chiếu, font và digest bản render |

## Build

Từ thư mục `paper/`, nếu có TeX Live và latexmk:

```sh
python3 scripts/build.py --engine latexmk
```

Hoặc dùng [Tectonic](https://github.com/tectonic-typesetting/tectonic):

```sh
python3 scripts/build.py --engine tectonic
```

Tectonic tải/caches các gói TeX khi build lần đầu. Có thể chỉ định executable qua `--tectonic /path/to/tectonic`. Để sửa trên Overleaf, tải `main.tex`, `supplementary.tex`, `references.bib`, `sections/`, `tables/` và `figures/`; dùng XeLaTeX hoặc pdfLaTeX với BibTeX. Các PDF/vector phục vụ LaTeX đã có sẵn, không cần render lại PPTX.

## Nguồn và giới hạn kết quả

- Main BEV study: 21 archived run, 41 evaluation đã lưu; 19 run hoàn thành 50 epoch với AP selection. B0/B1 là historical run có budget khác.
- Cặp loss retrain mới: O100/Q100, 100 epoch, SimAM, không IQA, 815.744 tham số mỗi run. Cộng thêm hai run/bốn report này, inventory BEV có 23 run/45 report. Bảng factorial 50 epoch giữ nguyên các run P1–P4 tương ứng.
- Split thực tế: 3.712 train/3.769 validation, không trùng frame. Local BEV và reference AP được ghi riêng.
- P16 BEV: 70,99% local R40 Moderate mAP, 852.930 tham số; model 8,55 ms và pipeline offline 56,83 ms trên L4 FP32.
- Final3D: shared head, 100 epoch, checkpoint epoch 70; 797.382 tham số, 71,09% BEV và 57,75% 3D reference R40 Moderate mAP. Report chưa có stage latency cho run này.
- Không có training, inference hay detector test mới trong quá trình chuẩn bị paper. Các phép kiểm tra là đối chiếu nguồn, hash, số liệu, citation và bản render tài liệu.
- Config BEV có một field head chia sẻ được thêm sau run; bản phục hồi khớp training hash nằm trong `research/configs/`, bản nhận được nằm trong `as_provided/`. Audit ghi cả hai.
- Hai config O100/Q100 mới đã khớp training digest ngay từ bản nhận được; dữ liệu nguyên bản và hash audit nằm trong `research/loss_retrain/` và `loss_retrain_comparison.json`.
- Demo P4 cuboid dùng vertical ước lượng từ LiDAR. Bản thảo phân biệt chúng với vertical branch đã train của Final3D.

`prepare_assets.py` cần matplotlib, NumPy và python-pptx; nếu chạy lại, cần các archive gốc tại đường dẫn lưu trong inventory, `final_3d_run_location.txt`, và cặp mới tại `/home/duyennh/Downloads/drive-download-20261013T095105Z-1-001`. Việc chỉnh sửa/biên dịch LaTeX dùng các asset đã đóng gói và không cần archive checkpoint trong Downloads. Các CSV ablation nhận được vẫn mô tả 21 run ban đầu; inventory/summary JSON và bảng LaTeX đã bổ sung cặp mới. Bản trước cập nhật nằm trong `revisions/v1_before_loss_retrain_20261013.zip`.

## Trạng thái bản thảo

Đã có toàn bộ phần abstract, introduction, related work, method, experiments, results, discussion, conclusion và supplementary. Tên tác giả đang để `Anonymous manuscript draft` do chưa có thông tin tác giả. Cần thay thông tin này và bổ sung các mục tác giả/funding/biography theo venue trước khi nộp. Tài liệu `venue_strategy_vi.md` nêu các thí nghiệm đề xuất để tăng sức cạnh tranh; bản thảo không giả định các thí nghiệm ấy đã hoàn thành.
