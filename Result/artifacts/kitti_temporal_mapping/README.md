# Thứ tự thời gian của KITTI Object hiện có

Nguồn: official KITTI devkit_object.zip, không suy đoán từ tên file.
URL: https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip

Cách tra Object ID: train_mapping.splitlines()[train_rand[int(object_id)] - 1].
train_rand là hoán vị 1-based; Object ID và raw frame là 0-based.
Đã đối chiếu ví dụ trong README chính thức: Object 000000 -> drive 0106, raw frame 48.

object_to_raw_order.csv: toàn bộ 7.481 Object ID, date, drive, raw_frame; sắp theo
(date, drive, raw_frame). Label/calibration vẫn dùng Object ID cũ, không đổi tên.

longest_consecutive_object_ids.txt: 174 Object ID theo thứ tự thời gian từ raw
frame 77 đến 250 của 2011_09_26_drive_0014_sync. Đã kiểm tra các raw frame liên
tiếp và đủ .bin, label_2, calib tương ứng trong Code/Kitti_Raw. Có thể dùng làm
video 10 FPS khoảng 17.4 giây; chưa render hoặc chạy detection cho chuỗi này.
Đây là clip demo; chưa kiểm tra độ phủ lớp hoặc phân tách train/validation của chuỗi.

selected_drive_object_ids.txt: 20 Object ID có sẵn của drive chứa 005663.
005663 -> 2011_09_28_drive_0021_sync, raw frame 127. Chuỗi này cách quãng 7–12
raw frame; muốn đủ 10 Hz phải bổ sung dữ liệu raw của đúng drive.

Mapping chỉ phục hồi thứ tự/frame gốc; timestamp tuyệt đối lấy từ timestamps.txt
của drive raw synced. Không tạo ra những frame bị thiếu.
summary.json lưu hash devkit, kết quả tra và thống kê các chuỗi liên tiếp.
