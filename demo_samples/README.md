# Thư mục ảnh mẫu cho Demo (`demo_samples/`)

Mỗi thư mục con đại diện cho một nhóm đối tượng (tách từ tập dữ liệu thật **Lung** và **SUIM**):
- `lung_xray/`: Ảnh X-quang phổi (4 cặp).
- `suim_diver_HD/`: Thợ lặn dưới nước (3 cặp).
- `suim_fish_FV/`: Cá & sinh vật biển (3 cặp).
- `suim_robot_RO/`: Robot & thiết bị dưới nước (3 cặp).
- `suim_wreck_WR/`: Xác tàu đắm (3 cặp).

## Quy ước đặt tên file
Trong cùng một thư mục, mỗi mẫu gồm 2 file:
- `*_img.jpg` (hoặc `.png`): Ảnh gốc.
- `*_mask.png`: Mask nhị phân (vùng trắng = đối tượng cần tách, nền đen = background).

## Cách dùng khi tự upload ảnh trên Demo
1. Chọn **1 cặp** làm **Support** (ví dụ: `diver_1_img.jpg` + `diver_1_mask.png`).
2. Chọn **1 ảnh khác cùng loại** làm **Query** (ví dụ: `diver_2_img.jpg`, có thể chọn thêm `diver_2_mask.png` để tính IoU).
