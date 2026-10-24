# So sánh IQA trên ảnh camera KITTI

Ảnh chính: `iqa_comparison_007016.png` (3040 × 2185).

- Hàng trên: điểm phân loại, `nms_alpha=0`.
- Hàng dưới: điểm kết hợp IQA, `nms_alpha=0.5`.
- Cùng một forward của checkpoint `Final-h32/selected/best_ap.pt`, epoch 130;
  cùng dữ liệu LiDAR, ảnh camera trái và calibration của frame 007016.
- Score threshold 0.3; classwise rotated BEV NMS threshold 0.1.
- Đây là so sánh tác động của IQA trong hậu xử lý. Checkpoint được train có IQA;
  hàng trên không phải một mô hình được huấn luyện riêng không có nhánh IQA.
- Box từ LiDAR được chiếu qua `P2 @ R0_rect @ Tr_velo_to_cam`. Z và chiều cao
  ước lượng từ mặt đất cục bộ và các điểm LiDAR trong footprint; không phải
  giá trị được dự đoán bởi head 3D. Cả hai hàng dùng chung phép ước lượng cho
  các candidate giống nhau. Không sử dụng ground truth.

Frame được chọn trong 174 frame người dùng cung cấp vì có khác biệt candidate
nhìn thấy được. Đây là ví dụ minh họa, không phải kết luận về độ chính xác.
Trước khi chiếu có 12/13 detection; ảnh hiển thị 10/11 box có hỗ trợ chiều cao
từ LiDAR và nằm trong trường nhìn camera. Hàng dưới có thêm một candidate
Pedestrian (được đánh dấu bằng mũi tên); chưa gán TP/FP cho candidate này.

`007016_metadata.json` chứa hash ảnh, LiDAR, calibration, checkpoint và các box
được vẽ/bỏ qua. `007016_detections.npz` lưu detection của frame. Kết quả suy luận
cả 174 frame nằm trong `detections_all_frames.npz` và `scan_summary.json`.

Chạy lại từ thư mục gốc dự án:

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python Result/iqa_comparison_20261019/scan_frames.py
/home/duyennh/miniconda3/envs/AI_env/bin/python Result/iqa_comparison_20261019/render_comparison.py --frame 007016
```
