# Camera detection + Hist14 BEV, 3 hàng × 2 cột

Ảnh chính: `camera_detection_hist14_3x2.png`, 5280 × 3520 pixel.

Ba hàng là frame KITTI `002425`, `005571`, `007016`, lấy ảnh camera trái từ
`/home/duyennh/Downloads/kitti_training_174frames_images/training/image_2`.

- Trái: detection thật có IQA từ checkpoint Final-h32 `best_ap.pt`, epoch 130;
  score threshold 0.3, rotated BEV NMS threshold 0.1, nms_alpha 0.5.
- Phải: đủ 14 map input encoder **Hist14 v1** của đúng scan từng hàng, bố trí
  hai dòng × bảy kênh. Encoder trong code tên `hist14`, không có tên `rich14`.
- Mỗi kênh dùng một màu riêng; bảng màu và miền chuẩn hóa giữ nguyên giữa
  ba frame. Tên kênh đi kèm trên từng map; thứ tự C00–C13.
- Tất cả map dùng toàn ROI X=[0,70.4], Y=[−40,40] m; X hướng lên, +Y sang trái.
  Đây là encoder input, chưa qua backbone; không phải activation hoặc attention.
- Tensor float32 gốc có shape `[14,800,704]`, lưu trong ba file
  `<frame>_hist14_tensor.npz`. Không augmentation, không dùng ground truth.
- Khi vẽ thumbnail, gộp max 2×2 trên đáp ứng đã chuẩn hóa và dùng gamma 0.5
  để giữ các ô thưa nhìn thấy được. Chỉ bước hiển thị áp dụng phép biến đổi này;
  tensor gốc giữ nguyên. Ô trống nền tối, giá trị 0 ở ô có điểm dùng sắc độ nhạt.
- Detection nhận pointcloud, không nhận ảnh RGB. Box chiếu lên camera bằng
  calibration của chính frame; Z và chiều cao box ước lượng từ điểm LiDAR.

`metadata.json` ghi hash ảnh, LiDAR, calibration, checkpoint, cache prediction,
tensor, màu kênh, thông số hiển thị và các box được vẽ/bỏ qua.

Tạo lại từ thư mục gốc dự án:

```bash
/home/duyennh/miniconda3/envs/AI_env/bin/python Result/camera_hist14_3x2_20261019/create_figure.py
```
