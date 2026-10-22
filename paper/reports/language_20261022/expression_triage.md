# Đối chiếu ứng viên expression sau sửa

`improve_expression.py` nêu hai ứng viên theo từ `make`, mỗi ứng viên xuất hiện ở lượt chạy file chính và file con. Đây là cảnh báo phụ thuộc ngữ cảnh, không phải lỗi ngữ pháp hoặc lệnh sửa tự động.

## Methodology, dòng 291 — U085

> Zero-initialized gating weights and biases make $\mathbf{G}_r$ initially equal to $1.0$, corresponding to standard residual addition.

**Giữ nguyên:** `make ... equal to ...` mô tả quan hệ xác định giữa khởi tạo và giá trị gate. Cụm này có vị ngữ rõ, gắn với công thức trong cùng phần; không phải cấu trúc vòng `make use of` mà rule đề phòng. Các token toán học thực tế được bảo vệ đầy đủ trong [U085](rewrites.md#u085).

```text
[LLM]
Changed: none after contextual review
Protected: original math, notation, and technical predicate; see the source block in U085
Meaning-Check: PRESERVED
Risk-Flags: none
```

## Results, dòng 30 — U125

> Sparse returns and aspect-ratio differences make small, non-rigid objects such as pedestrians difficult to detect.

**Giữ nguyên:** `make ... difficult to detect` là cấu trúc gây trạng thái trực tiếp, diễn đạt khó khăn chung của bài toán. Dùng hiện tại ở câu nền tảng này hợp lệ; các câu thuật lại kết quả ngay sau dùng quá khứ. Không thay bằng một động từ chỉ để xóa cảnh báo từ khóa. Xem [U125](rewrites.md#u125).

```text
[LLM]
Changed: none after contextual review
Protected: problem scope, object category, and the technical meaning
Meaning-Check: PRESERVED
Risk-Flags: none
```
