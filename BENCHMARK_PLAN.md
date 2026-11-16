# Kịch bản chạy thực nghiệm Table 4 (3 máy song song trên Modal T4)

Mục tiêu: Chạy đầy đủ **ABCDFSS-Plus** ở chế độ **`every-episode` (`no-pp`, `num_epochs=25`, `seed=42`)** trên 5 bộ dữ liệu (`Deepglobe`, `ISIC`, `Lung`, `FSS-1000`, `SUIM`) ở cả **1-shot** và **5-shot** để so sánh trực tiếp với dòng `Ours (no-pp)` trong **Table 4** của [paper.pdf](paper.pdf).

---

## 1. Phân chia 3 phần (Thời lượng & Chi phí gần tương đương)

Đơn giá tham chiếu trên Modal **NVIDIA T4**: ~**\$0.59 / giờ GPU**.

| Phần | Máy thực hiện | Các đơn vị được giao (`benchmark:shot`) | Tổng GPU-giờ ước tính | Thời gian chờ (chạy song song) | Chi phí ước tính |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **Part 1** | Máy 1 | `lung:1` (0.9h), `suim:1` (2.1h), `fss:1` (3.1h), `fss:5` (8.3h) | **~14.4 giờ** | **~8.3 giờ** | **~\$8.50** |
| **Part 2** | Máy 2 | `deepglobe:1` (2.4h), `lung:5` (2.4h), `isic:5` (9.0h) | **~13.8 giờ** | **~9.0 giờ** | **~\$8.14** |
| **Part 3** | Máy 3 | `isic:1` (3.4h), `suim:5` (5.7h), `deepglobe:5` (6.4h) | **~15.5 giờ** | **~6.4 giờ** | **~\$9.15** |
| **Tổng** | **3 máy** | **Đủ 10/10 đơn vị (Table 4)** | **~43.7 giờ** | **~6.4 – 9.0 giờ** | **~\$25.79** |

> Mỗi máy chỉ cần chạy 1 tài khoản Modal (gói Starter \$30 credit miễn phí dư sức chi trả phần của mình). Chỉ các dataset thuộc phần của máy nào thì Modal của máy đó mới tự tải về.

---

## 2. Hướng dẫn cài đặt trên từng máy (1 lần duy nhất)

1. **Clone repo & cài môi trường:**
   ```powershell
   git clone <URL_REPO>
   cd BTL_CV
   python -m venv .venv
   .\.venv\Scripts\pip.exe install modal
   ```
2. **Đăng nhập Modal** (mỗi máy dùng 1 tài khoản Modal có sẵn billing + credit):
   ```powershell
   .\.venv\Scripts\modal.exe setup
   ```
3. **Cung cấp Kaggle API Token** (để Modal tự tải dataset từ Kaggle vào Volume `abcdfss-datasets`):
   * Lưu token `KGAT_...` vào `%USERPROFILE%\.kaggle\access_token` (hoặc đặt biến môi trường `KAGGLE_API_TOKEN`, hoặc đặt file `kaggle.json` tại `%USERPROFILE%\.kaggle\kaggle.json`).

---

## 3. Lệnh chạy thực nghiệm trên từng máy (`--detach` — tắt máy vẫn chạy)

Chạy đúng lệnh tương ứng với phần được giao (có thể tắt laptop sau khi thấy dòng `[detached] Launching ... GPU units`):

* **Máy 1 (Part 1):**
  ```powershell
  .\.venv\Scripts\modal.exe run --detach modal_benchmark.py --part 1
  ```
* **Máy 2 (Part 2):**
  ```powershell
  .\.venv\Scripts\modal.exe run --detach modal_benchmark.py --part 2
  ```
* **Máy 3 (Part 3):**
  ```powershell
  .\.venv\Scripts\modal.exe run --detach modal_benchmark.py --part 3
  ```

*(Nếu trên máy đó đã từng tải dataset vào `abcdfss-datasets` trước đó rồi, có thể thêm `--skip-download` để bỏ qua bước kiểm tra tải dataset.)*

---

## 4. Lấy kết quả về & Tổng hợp thành Table 4

1. **Trên từng máy**, sau khi xem trên web `https://modal.com` thấy App `abcdfss-plus-table4` đã hoàn tất (`Stopped` / `Completed`), tải file kết quả của phần mình về thư mục `results/parts/`:
   * **Máy 1:**
     ```powershell
     .\.venv\Scripts\modal.exe volume get abcdfss-results table4_part1.json results/parts/
     ```
   * **Máy 2:**
     ```powershell
     .\.venv\Scripts\modal.exe volume get abcdfss-results table4_part2.json results/parts/
     ```
   * **Máy 3:**
     ```powershell
     .\.venv\Scripts\modal.exe volume get abcdfss-results table4_part3.json results/parts/
     ```
   Sau đó push 3 file `results/parts/table4_part{1,2,3}.json` lên repo (hoặc gửi về máy tổng hợp).

2. **Tại máy tổng hợp**, khi đã có đủ 3 file trong `results/parts/`, chạy:
   ```powershell
   .\.venv\Scripts\python.exe merge_table4_results.py
   ```
   Script sẽ tự động:
   * Kiểm tra đủ 10/10 đơn vị (báo rõ nếu thiếu đơn vị nào).
   * Tính trung bình **CD-FSS Avg.** (`deepglobe`, `isic`, `lung`, `fss`) cho cả 1-shot và 5-shot.
   * Xuất ra [results/table4_results.csv](results/table4_results.csv) và [results/table4_results.json](results/table4_results.json).
   * In sẵn dòng HTML `<tr>...</tr>` theo đúng chuẩn **Table 4** để dán trực tiếp vào [REPORT.md](REPORT.md).
