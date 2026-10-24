# Báo cáo tối ưu Hist14 — 18/10/2026

Hist14 Numba revision 3 nhanh hơn implementation Binary35 hiện tại trên cùng CPU,
cùng frame và cùng hình học. Mô hình giữ nguyên 14 kênh và dùng được checkpoint cũ.

| Phạm vi đo CPU | Binary35 hiện tại | Hist14 revision 3 |
| --- | ---: | ---: |
| Rasterize | 6.662 ms | 2.693 ms |
| Rasterize + tensor CHW liên tục | 102.296 ms | 3.013 ms |

Rasterize Hist14 nhanh hơn **2.47 lần** trong phép đo này.
Binary35 cần đổi bố trí tensor dày 78.85 MB thành CHW liên tục; Hist14 ghi CHW trực tiếp
và trả 31.54 MB. Trong evaluator cũ, bước contiguous của Binary35 có thể nằm trong
timer H2D. Không dùng hàng CHW để thay trực tiếp cột preprocess trong bảng T4/L4.

Bản Hist14 trước sửa đã là revision 2 và đã tối ưu từ trước. Lượt sửa hiện tại giảm
thời gian CHW từ **3.085 → 3.013 ms**,
tương đương **2.34%**.
Chỉ mức giảm thêm này được quy cho lượt sửa hiện tại. Chưa có bằng chứng đây là giới
hạn tốc độ tối đa trên mọi máy hoặc so với mọi implementation Binary35 có thể có.

## Điều kiện và thay đổi

- AMD Ryzen 5 7640HS, Python 3.14.4, NumPy 2.4.6, Numba 0.67.0; kernel CPU tuần tự.
- 12 frame cách đều trên danh sách KITTI validation 3769 frame; 5 warmup,
  30 lần đo/frame/phương pháp; thứ tự phương pháp ngẫu nhiên ở từng vòng.
- Loại sớm điểm ngoài ROI theo từng trục, trước khi đọc đặc trưng tiếp theo;
  điều kiện ROI loại cả NaN/Inf tọa độ, còn intensity giữ kiểm tra finite riêng.
- Giữ nguyên công thức, thứ tự tính float64, đầu ra float32, bin/ROI, 14 kênh,
  semantic hash và vùng nhớ độc lập giữa các lần gọi.
- Phương án đổi thứ tự scatter, padding accumulator và literal dispatch được thử
  rồi bỏ vì không cho lợi ích ổn định. Raw thử nghiệm nằm ở artifacts/hist14_speed_v3.

## Kiểm chứng

- So sánh bit uint32, gồm signed zero: **0 sai khác** trên **100 frame validation
  và 8 trường hợp biên**, cho cả HWC và CHW, đối chiếu bản trước sửa.
- **177 tests đạt**: encoding, benchmark, schema, dataset/augmentation integration
  và checkpoint. Không chạy training hoặc đánh giá AP mới.
- Snapshot trước sửa trùng Git HEAD byte-for-byte, SHA-256:
  `89ac8b999dbeddf8c49da1347d84606c4834e64c4b1e10a9c1ede7caac6b7542`.

## Áp dụng

File thực thi đã cập nhật: `Code/detector/core/datasets/utils_1/bev_backend.py`.
Config/checkpoint T4 đang dùng `name: hist14`, `backend: numba`, nên chỉ cần dùng
source mới, khởi động lại Python để bỏ module cũ và warmup trước khi đo.
Không cần train lại cho thay đổi implementation này. Các phép đo AP/speed đã lưu
trong Excel T4/L4 vẫn giữ nguyên; muốn công bố speed GPU mới cần chạy lại profiling
trên cùng GPU, CPU, dữ liệu và phạm vi timer.

CLI đo lại từ thư mục Code:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python3 tools/benchmarks/benchmark_hist14_vs_binary.py \
  --pointcloud Kitti_Raw/velodyne/training/velodyne/000001.bin \
               Kitti_Raw/velodyne/training/velodyne/007480.bin \
  --warmup 5 --iterations 30 --output artifacts/hist14_vs_binary.json
```

**Giới hạn:** thời gian CPU gồm rasterization/bố trí tensor theo từng hàng;
không gồm đọc file, augmentation, H2D, forward, decode/NMS, AP hoặc JIT trong số
steady-state. Đây là phép đo local, không phải kết quả E2E T4/L4.
