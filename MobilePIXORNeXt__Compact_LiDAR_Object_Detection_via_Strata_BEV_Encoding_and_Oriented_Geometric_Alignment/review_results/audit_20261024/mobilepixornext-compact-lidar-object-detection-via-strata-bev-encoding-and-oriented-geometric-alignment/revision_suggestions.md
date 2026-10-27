# Việc đã sửa và việc còn thiếu

Đã chỉnh source đúng folder được yêu cầu; thư mục `paper/` không được đồng bộ. Bản gốc của phiên này và diff được giữ trong artifacts.

| Nhóm | Đã sửa | Còn cần bằng chứng |
|---|---|---|
| Entry/abstract | Một entry source, một abstract; số deployable/train-only nhất quán | Không |
| Protocol | Local/reference theo từng nhóm, bỏ winner claim | Evaluator equivalence hoặc common reference reevaluation |
| Run identity | Sửa sampler/head/context/epoch/selection labels | Original report/config/checkpoint/prediction hashes |
| Bảng loss | Sửa per-class P1/P11 và params; phân biệt legacy focal | Đối chứng loss trên final no-focal / shared heads |
| Ablation | Inventory có runID; tách P1/S32; bỏ causal gain không đủ control | Matched component runs |
| Seed | Giữ mean/std mô tả, ghi encoding/budget khác | Fixed-recipe seed study |
| Timing | L4/T4 riêng; sửa percentile có nguồn, phân biệt offline boundary | Common hardware/CPU/pinning/postprocess setup |
| Methods | Modulo-pi, surrogate quality, shared gradient, tensor index | Soft-min D/units/normalization clarification |
| Figures | Caption giới hạn, không dùng ảnh vehicle chứng minh VRU AP | Regenerate Pareto/loss legends from run manifests |

Các sửa đổi câu chi tiết ở `artifacts/data/source_changes.patch`; metadata bảo vệ nội dung ở `artifacts/data/rewrite_comments.tex`. Không có số liệu thực nghiệm mới được tạo.
