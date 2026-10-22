# Cảnh báo deai và đối chiếu thì

Bảng dưới lấy từ lượt duyệt toàn tài liệu của file chính. Các file con không có heading cấp section có thể bị deai_batch bỏ qua; kết quả không có hit của file con không phải chứng nhận không lỗi.

| Lượt | Vị trí | Loại | Quyết định | Lý do |
|---|---|---|---|---|
| before | sections/methodology.tex:142 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| before | sections/methodology.tex:144 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| before | sections/methodology.tex:8 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| before | sections/methodology.tex:192 | tense | present-tense-exception | Định nghĩa kênh hoặc tính chất hình học chung; thì hiện tại hợp lệ. |
| before | sections/methodology.tex:207 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| before | sections/methodology.tex:261 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| before | sections/methodology.tex:261 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| before | sections/results.tex:89 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| before | sections/results.tex:93 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| before | sections/results.tex:98 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| before | sections/results.tex:102 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| before | tables/comparison.tex:23 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| before | sections/results.tex:12 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| before | sections/results.tex:12 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| before | sections/results.tex:30 | tense | general-truth-exception | presents mô tả khó khăn chung của bài toán; không phải lỗi thì. Đoạn được sửa vì từ ngữ và cấu trúc. |
| before | sections/results.tex:33 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| before | sections/results.tex:45 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| before | sections/results.tex:48 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| before | sections/results.tex:123 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| before | sections/results.tex:125 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| before | sections/results.tex:183 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| before | sections/results.tex:183 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| before | sections/results.tex:190 | tense | edited-operation-or-finding | Phần thuật lại thao tác/kết quả được đổi sang quá khứ; câu nền tảng chung được giữ hiện tại. |
| after | sections/methodology.tex:142 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| after | sections/methodology.tex:144 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| after | sections/methodology.tex:8 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| after | sections/methodology.tex:8 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| after | sections/methodology.tex:50 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| after | sections/methodology.tex:84 | tense | present-tense-exception | Định nghĩa kênh hoặc tính chất hình học chung; thì hiện tại hợp lệ. |
| after | sections/results.tex:89 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| after | sections/results.tex:93 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| after | sections/results.tex:98 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| after | sections/results.tex:102 | burstiness | false-positive-markup | LaTeX \item/\textit hoặc \par/\vspace lặp; không phải các đoạn văn lặp cách mở đầu. |
| after | tables/comparison.tex:23 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| after | sections/results.tex:12 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| after | sections/results.tex:183 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
| after | sections/results.tex:183 | tense | present-tense-exception | Caption, chú giải bảng hoặc mô tả trực tiếp nội dung hình dùng thì hiện tại hợp lệ. |
