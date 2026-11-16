# ABCDFSS-Plus: Tái tạo & Cải tiến "Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation" (CVPR 2024)

## 1. Cấu trúc thư mục
- `paper.pdf`: Bài báo gốc (*Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation*, Jonas Herzog, CVPR 2024).
- `ABCDFSS/`: Mã nguồn chính thức của tác giả (đã sửa tương thích đường dẫn Windows trong `data/suim.py`).
- `datasets/`: Dữ liệu benchmark thực tế gồm:
  - `lung/`: Chest X-ray Lung Segmentation (Montgomery Set, 138 ảnh + mask).
  - `suim/`: Semantic Segmentation of Underwater Imagery (SUIM, 1.598 ảnh, 7 lớp: `FV, HD, PF, RI, RO, SR, WR`).
- `improved_abcdfss.py`: Cài đặt 3 hướng cải tiến đề xuất (`SpatialContextAdapter`, `ImprovedDAMatComparison` với DWLA & Cosine Affinity, `compute_support_self_loss`).
- `run_experiments.py`: Script chạy thực nghiệm tái tạo (baseline + ablation theo Table 6 & Table 7/8) và đánh giá các cải tiến trên cùng tập episodes cố định.
- `generate_qualitative.py`: Sinh ảnh trực quan so sánh Baseline vs. Improved.
- `app.py`: Ứng dụng web demo tương tác bằng **Streamlit**.
- `results/`: Lưu trữ kết quả thực nghiệm (`metrics.json`, `comparison_charts.png`, `qualitative_comparison.png`).

## 2. Hướng dẫn chạy
Kích hoạt môi trường ảo `.venv`:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Chạy thực nghiệm tái tạo & cải tiến
```powershell
python run_experiments.py --lung-episodes 50 --suim-episodes 70
python generate_qualitative.py
```

### Chạy Demo tương tác
```powershell
streamlit run app.py
```
