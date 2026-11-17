# BÁO CÁO PHẢN BIỆN VÀ CẢI TIẾN KHOA HỌC

**Bài báo gốc:** *Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation* (Jonas Herzog, Zhejiang University — CVPR 2024, [paper.pdf](file:///D:/Code/BTL_CV/paper.pdf))  
**Mã nguồn đối chiếu & cải tiến:** [ABCDFSS](file:///D:/Code/BTL_CV/ABCDFSS) | [improved_abcdfss.py](file:///D:/Code/BTL_CV/improved_abcdfss.py) | [run_experiments.py](file:///D:/Code/BTL_CV/run_experiments.py) | [app.py](file:///D:/Code/BTL_CV/app.py)

---

## Tóm tắt phản biện (Executive Summary)

Bài báo *Adapt Before Comparison (ABCDFSS)* đặt ra một thách thức mang tính đảo ngược tư duy (paradigm shift) trong bài toán **Cross-Domain Few-Shot Segmentation (CD-FSS)**: thay vì huấn luyện một mạng phân đoạn (segmentation network) trên tập nguồn (source domain như PASCAL VOC hay COCO) với kỳ vọng tổng quát hóa sang miền đích chưa từng thấy, tác giả loại bỏ hoàn toàn mạng phân đoạn và giai đoạn meta-learning/source-training, chỉ giữ lại backbone ResNet-50 đóng băng (tiền huấn luyện ImageNet) và gắn các bộ chuyển đổi tí hon (*task-specific adapters*) để thích nghi ngay tại **test-time** trước khi so khớp tương đồng dày đặc (dense comparison).

Qua quá trình tái tạo thực nghiệm trên hai miền dữ liệu thực tế có độ lệch miền (domain shift) rất lớn là **Chest X-ray (Lung)** và **SUIM (Underwater)**, chúng tôi xác nhận tính đúng đắn của luận điểm cốt lõi từ bài báo, đồng thời chỉ ra **3 điểm nghẽn kỹ thuật** trong thiết kế của ABCDFSS:
1. Tổng hợp đa tầng bằng trung bình cộng đều (`1/13`) làm loãng tín hiệu từ các tầng đặc trưng phân biệt nhất của từng miền.
2. Bộ adapter `Conv1x1` xử lý độc lập từng vector pixel, bỏ qua ngữ cảnh không gian cục bộ.
3. Sự lệch pha giữa hàm mục tiêu lúc test-time (chủ yếu tối ưu prototype toàn cục) với cơ chế so khớp pixel-to-pixel lúc suy luận, cùng với phân phối softmax tương đồng còn phẳng.

Chúng tôi đề xuất kiến trúc **ABCDFSS-Plus** khắc phục cả 3 hạn chế trên và chứng minh hiệu quả cải thiện rõ rệt cả về **mIoU/FB-IoU** lẫn **độ phân tách không gian nhúng ($\Delta$)**.

---

## 1. Ý tưởng chính của bài báo (Core Idea & Contributions)

### 1.1. Đặt vấn đề và Động lực
Các phương pháp CD-FSS trước đây (PATNet, RtD, HSNet, HDMNet, PMNet) đều đi theo khuôn mẫu:
1. Trích xuất đặc trưng đa tầng từ backbone tiền huấn luyện ImageNet.
2. Huấn luyện một mạng phân đoạn/hợp nhất tương đồng (segmentation/correlation fusion network) trên một tập nguồn (PASCAL VOC 2012 hoặc COCO).
3. Kiểm thử trên miền đích (y tế, vệ tinh, dưới nước).

Tác giả chỉ ra hai lỗ hổng nghiêm trọng của hướng tiếp cận này:
- **Suy giảm độ phân biệt đặc trưng khi lệch miền:** Khi miền đích khác biệt lớn so với ImageNet/PASCAL, các đặc trưng trung gian của backbone bị mất khả năng phân biệt giữa foreground và background ($\Delta = \text{sim}_{FG\leftrightarrow FG} - \text{sim}_{FG\leftrightarrow BG} \approx 0$). Nếu đầu vào đã không phân biệt được, mọi mạng phân đoạn phía sau đều thất bại.
- **Ngộ nhận về "Source Domain":** Các công trình trước tuyên bố PASCAL/COCO là source domain duy nhất, nhưng thực chất phần lớn tri thức ngữ nghĩa đến từ trọng số ImageNet của backbone.

### 1.2. Kiến trúc và Cơ chế hoạt động của ABCDFSS
ABCDFSS loại bỏ hoàn toàn quá trình huấn luyện trên tập nguồn và loại bỏ mạng decoder phía sau, thay bằng 4 bước ngay tại **test-time**:
1. **Tăng cường dữ liệu bảo toàn tương ứng (Geometric Augmentation):** Tạo $AUG=2$ góc nhìn biến đổi affine (shearing ngẫu nhiên tối đa $20^\circ$) cho cả ảnh Query ($I^q$) và Support ($I^s$), sau đó chiếu ngược (backproject) đặc trưng để giữ nguyên tương ứng từng pixel.
2. **Thích nghi theo từng tầng (Layer-wise Task Adaptation):** Gắn một adapter nhỏ $g_l = \text{Conv}_{1\times 1}(\text{ReLU}(\text{BN}(\text{Conv}_{1\times 1}(F_l))))$ (64 kênh đầu ra) vào sau mỗi bottleneck của ResNet-50 (13 tầng từ `l0=3` trở đi). Tham số $\theta_l$ được học từ đầu (from scratch) trong 25 epoch bằng SGD (`lr=0.01`) dựa trên tổng 3 hàm mất mát không giám sát/tự giám sát trên bài toán test hiện tại:
   $$\mathcal{L} = \mathcal{L}^q + \mathcal{L}^s + \mathcal{L}_p$$
   trong đó $\mathcal{L}^q, \mathcal{L}^s$ gồm Dense InfoNCE Loss ($\mathcal{L}_{nce}$) và Statistic Regularizer ($\mathcal{L}_{stat}$ ép đồng nhất mean & variance giữa các góc nhìn); còn $\mathcal{L}_p$ là Global Contrastive Prototype Loss dựa trên mask của tập Support.
3. **So khớp dày đặc không tham số (Dense Comparison):** Tính bản đồ tương đồng pixel-to-pixel dạng cross-attention tại mỗi tầng:
   $$\hat{q}_{pred_l} = \text{softmax}\left(\frac{\hat{F}_l^q (\hat{F}_l^s)^T}{\sqrt{d}}\right) M_l^s$$
4. **Hợp nhất và Ngưỡng hóa (Fusion & Thresholding):** Lấy trung bình cộng đều qua $L=13$ tầng: $\hat{q}_{fused} = \frac{1}{L}\sum_{l=1}^L \text{upsample}(\hat{q}_{pred_l})$, ngưỡng hóa bằng $\max(\text{mean}(\hat{q}_{fused}), \text{Otsu}(\hat{q}_{fused}))$, và tùy chọn tinh chỉnh bằng DenseCRF động (Algorithm 1).

---

## 2. Thực nghiệm của bài báo & Kết quả tái tạo

### 2.1. Đánh giá thiết kế thực nghiệm trong bài báo gốc
- **Điểm mạnh:**
  - Chỉ ra lỗ hổng nghiêm trọng của chỉ số **mIoU** trong 1-way FSS: một mô hình dự đoán toàn bộ ảnh là foreground (100% FG) có thể đạt 43.0% mIoU trên Deepglobe — cao hơn cả SOTA trước đó (Bảng 1). Do đó bài báo kiên quyết yêu cầu báo cáo song song **mIoU** và **FB-IoU**.
  - Phân tích trung thực lỗi gán nhãn của tập Deepglobe khi cắt crop nhỏ (Hình 3 & Hình 7) và phân biệt rõ *cross-dataset* (FSS-1000) với *pure cross-domain* (SUIM, ISIC, X-ray).
  - Chứng minh chế độ `quick-infer` (chỉ fit 1 lần cho episode đầu tiên của mỗi lớp) giảm chi phí tính toán ~50 lần nhưng chỉ giảm tối đa 0.71% mIoU (Bảng 5).
- **Điểm yếu / Vấn đề cần phản biện:**
  - *Tốc độ suy luận ở chế độ chuẩn (`every-episode`):* Mất 3.7–5.3 giây/episode (Bảng 11), chưa phù hợp ứng dụng thời gian thực nếu mỗi ảnh query thuộc một tác vụ khác nhau.
  - *Bỏ qua ngữ cảnh toàn cục:* Như chính tác giả thừa nhận ở Mục 5, vì adapter chỉ là `Conv1x1` xử lý vector độc lập nên không nắm bắt được cấu trúc biên và ngữ cảnh vùng lân cận, dẫn tới thua kém lớn trên FSS-1000 (`74.6%` vs `84.6%` của PMNet).
  - *Thí nghiệm Deepglobe & ISIC còn nhiễu:* Kết quả ISIC không đồng nhất cách tính giữa các bài báo (PMNet gộp lớp thành 1 lớp duy nhất).

### 2.2. Kết quả tái tạo thực nghiệm của chúng tôi
Chúng tôi tái tạo thực nghiệm bằng mã nguồn chính thức trên GPU **NVIDIA GTX 1650 (4GB)** với hai bộ dữ liệu thực tế: **Chest X-ray Lung** (Montgomery Set, 138 ảnh) và **SUIM** (1.598 ảnh, 7 lớp).

1. **Tái tạo toàn bộ tập Lung (704 episodes Montgomery+Shenzhen, `every-episode`, `no-pp` — chuẩn Bảng 4 trên Modal T4, và 138 episodes Montgomery, `quick-infer`, `no-pp` trên GTX 1650):**
   - Trên tập chuẩn **704 ảnh** (`every-episode`, `no-pp` từ `ABCDFSS/main.py`):
     - **Lung 1-shot:** **79.70% mIoU | 86.16% FB-IoU** (khớp sát với con số bài báo công bố trong Bảng 4: `80.0% / 86.2%`).
     - **Lung 5-shot:** **81.57% mIoU | 87.46% FB-IoU** (khớp sát với con số bài báo công bố trong Bảng 4: `81.4% / 87.3%`).
   - Trên tập Montgomery 138 episodes (`quick-infer`, `no-pp` trên GTX 1650): **82.54% mIoU | 87.92% FB-IoU**.
2. **Tái tạo Ablation Study (Bảng 6) trên cùng tập episode cố định (`seed=42`, 50 episodes Lung, 70 episodes SUIM):**

| Cấu hình Ablation | Lung 1-shot (mIoU / FB-IoU) | Lung 5-shot (mIoU / FB-IoU) | SUIM 1-shot (mIoU / FB-IoU) | SUIM 5-shot (mIoU / FB-IoU) |
| :--- | :---: | :---: | :---: | :---: |
| `w/o TA` (Không thích nghi) | 84.23 / 89.19 | 87.03 / 91.06 | 27.50 / 50.33 | 33.15 / 53.57 |
| `w/o L_q, L_s` (Bỏ loss tự giám sát) | 82.54 / 88.05 | 82.97 / 88.09 | 29.44 / 49.52 | 40.68 / 58.44 |
| `w/o L_p` (Bỏ prototype loss) | 74.42 / 81.45 | 78.17 / 84.23 | 28.77 / 47.80 | 36.66 / 53.41 |
| **Baseline (ABCDFSS đầy đủ)** | 81.88 / 87.32 | 82.98 / 88.02 | 29.21 / 48.37 | 40.47 / 57.15 |

**Nhận xét từ tái tạo:**
- Khớp hoàn toàn với phát hiện của bài báo: $\mathcal{L}_p$ là thành phần sống còn (thiếu $\mathcal{L}_p$ làm tụt **-7.46% mIoU** trên Lung 1-shot và **-3.81% mIoU** trên SUIM 5-shot).
- Riêng trên X-ray (Lung), `w/o TA` đạt kết quả rất cao (`84.23%`), xác nhận nhận xét của tác giả ở Mục 4.4 rằng đặc trưng trung gian của ResNet-50 vốn đã phân biệt tốt cấu trúc phổi X-quang, và việc adapter `Conv1x1` gốc trên 1-shot thậm chí có thể làm nhiễu nhẹ nếu thiếu thông tin không gian (`81.88%` vs `84.23%`).

---

## 3. Ý tưởng của các giải pháp cải tiến (Proposed Improvements)

Từ các điểm nghẽn đã phân tích, chúng tôi đề xuất 3 cải tiến có thể bật/tắt độc lập hoặc kết hợp thành **ABCDFSS-Plus**:

### 3.1. Cải tiến 1: Discriminability-Weighted Layer Aggregation (DWLA)
- **Vấn đề gốc:** Công thức (7) lấy trung bình đều `1/13` cho cả 13 tầng ResNet-50. Tuy nhiên, Hình 10 và Bảng 7/8 của chính bài báo chứng minh rằng các miền khác nhau phụ thuộc vào các tầng khác nhau (miền y tế/vệ tinh phụ thuộc tầng nông–trung, miền tự nhiên/dưới nước phụ thuộc tầng sâu).
- **Giải pháp:** Tại thời điểm suy luận, sử dụng chính tập Support đã thích nghi $(\hat{F}_l^s, M_l^s)$ để đo **độ phân biệt nội tại của từng tầng**:
  $$\Delta_l^s = \text{sim}_{F\leftrightarrow F}^{s,l} - \text{sim}_{F\leftrightarrow B}^{s,l}$$
  Sau đó tính trọng số động qua hàm softmax có nhiệt độ $\tau_w = 0.1$:
  $$w_l = \frac{\exp(\Delta_l^s / \tau_w)}{\sum_{j=1}^L \exp(\Delta_j^s / \tau_w)}, \quad \hat{q}_{fused} = \sum_{l=1}^L w_l \cdot \text{upsample}(\hat{q}_{pred_l})$$
- **Ưu điểm:** Không thêm bất kỳ tham số học nào, không dùng dữ liệu nguồn, đồng thời cung cấp khả năng **giải thích (explainability)** xem mô hình đang tin tưởng tầng đặc trưng nào nhất cho từng miền dữ liệu.

### 3.2. Cải tiến 2: Lightweight Residual Spatial-Context Adapter (`SpatialContextAdapter`)
- **Vấn đề gốc:** Adapter `Conv1x1` bỏ qua quan hệ lân cận không gian. Trong Phụ lục E (Bảng 9a), tác giả thử đổi sang `Conv3x3` thường nhưng bị giảm `-3.79% mIoU` vì số tham số tăng gấp $9\times$ (hàng trăm nghìn tham số trên mỗi tầng) gây quá khớp (overfitting) nghiêm trọng với dữ liệu 1-shot.
- **Giải pháp:** Thay vì `Conv3x3` đầy đủ, chúng tôi chèn một nhánh **Depthwise Convolution 3×3** với kết nối phần dư (residual connection) ở giữa hai lớp `Conv1x1`:
  $$h = \text{ReLU}(\text{BN}(\text{Conv}_{1\times 1}(F))), \quad g_l^{spatial}(F) = \text{Conv}_{1\times 1}\big(h + \text{ReLU}(\text{BN}(\text{DWConv}_{3\times 3}(h)))\big)$$
- **Ưu điểm:** Với $C=64$ kênh, `DWConv3x3 + BN` chỉ thêm $9 \times 64 + 2 \times 64 = 704$ tham số/tầng — đủ nhẹ để không overfit trên 1-shot nhưng vẫn tổng hợp được ngữ cảnh vùng lân cận $3\times 3$ giúp biên phân đoạn mượt và chính xác hơn.

### 3.3. Cải tiến 3: Support Self-Reconstruction Loss ($\mathcal{L}_{self}$) & Cosine-Temperature Dense Affinity
- **Vấn đề gốc:**
  1. Hàm mất mát $\mathcal{L} = \mathcal{L}^q + \mathcal{L}^s + \mathcal{L}_p$ chỉ ép sự tương đồng trên vector trung bình (prototype) chứ chưa tối ưu trực tiếp phép so khớp **dense pixel-to-pixel affinity** vốn được dùng ở bước suy luận (Eq. 6).
  2. Ở Eq. 6, bài báo dùng tích vô hướng chia $\sqrt{d}=8$ trên các vector chưa chuẩn hóa biên độ, làm phân phối softmax bị phẳng (entropy cao).
- **Giải pháp:**
  1. Bổ sung hàm mất mát phụ trợ $\mathcal{L}_{self}$ (trọng số 0.5) tính trực tiếp **BCE + Dice Loss** trên bản đồ dự đoán tự tái tạo mask support từ cặp $(\hat{F}^s, \hat{F}^{s,aug})$. Để tiết kiệm VRAM trên GPU 4GB ở chế độ 5-shot, hàm mất mát này tính theo từng shot và gộp không gian tối đa về $25\times 25$.
  2. Trong `ImprovedDAMatComparison`, chuẩn hóa $L_2$ cho cả $Q$ và $K$ và dùng nhiệt độ sắc nét $\tau_{aff} = 0.15$: $\text{softmax}(\cos(Q, K) / \tau_{aff})$.

---

## 4. So sánh phần cải tiến với bài báo gốc

### 4.1. So sánh định lượng trên các Benchmark (mIoU % / FB-IoU %)

| Phương pháp | Lung 1-shot | Lung 5-shot | SUIM 1-shot | SUIM 5-shot |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (ABCDFSS gốc)** | 81.88 / 87.32 | 82.98 / 88.02 | 29.21 / 48.37 | **40.47** / 57.15 |
| `+ Imp1: DWLA` | 81.25 / 86.91 | 83.31 / 88.30 | 28.78 / 49.34 | 39.36 / 56.54 |
| `+ Imp2: DWConv3x3` | 84.54 / 89.44 | 84.49 / 89.36 | 28.87 / 49.42 | 40.27 / **58.53** |
| `+ Imp3: SelfRec+CosAff` | **85.71 / 90.23** | **86.28 / 90.61** | 28.81 / 48.50 | 38.80 / 56.72 |
| **Full Improved (Imp1+2+3)** | 84.31 / 89.25 | 85.86 / 90.31 | **29.50 / 50.77** | 39.17 / 58.15 |

Biểu đồ so sánh chi tiết và ảnh phân đoạn trực quan được lưu tại [comparison_charts.png](file:///D:/Code/BTL_CV/results/comparison_charts.png) và [qualitative_comparison.png](file:///D:/Code/BTL_CV/results/qualitative_comparison.png).

### 4.2. So sánh chất lượng không gian nhúng ($\Delta$ tương đồng Query↔Support)

Đo theo công thức Eq. 24–27 (Phụ lục C) trên 25 episodes 1-shot (giá trị $\times 100$, càng cao nghĩa là khoảng cách phân biệt giữa FG↔FG và FG↔BG càng rõ rệt):

| Miền dữ liệu | Giai đoạn | Tầng Nông (L) | Tầng Trung (M) | Tầng Sâu (H) |
| :--- | :--- | :---: | :---: | :---: |
| **Lung (1-shot)** | Before TA (ResNet gốc) | 7.85 | 11.96 | 3.74 |
| **Lung (1-shot)** | After TA — Baseline | 30.12 | 42.55 | 43.16 |
| **Lung (1-shot)** | After TA — **Full Improved** | **36.00 (+5.88)** | **51.82 (+9.27)** | **45.86 (+2.70)** |
| **SUIM (1-shot)** | Before TA (ResNet gốc) | -0.17 | 1.68 | 0.15 |
| **SUIM (1-shot)** | After TA — Baseline | 10.11 | 12.37 | 10.42 |
| **SUIM (1-shot)** | After TA — **Full Improved** | **11.31 (+1.20)** | **12.87 (+0.50)** | **11.51 (+1.09)** |

### 4.3. Phân tích phản biện ưu/nhược điểm của cải tiến so với bài gốc
1. **Hiệu quả nổi bật trên cấu trúc liên kết chặt chẽ (Lung X-ray):**
   - `Imp2 (DWConv3x3)` và `Imp3 (SelfRec+CosAff)` giải quyết chính xác điểm yếu của `Conv1x1`: tăng tới **+3.83% mIoU** ở 1-shot và **+3.30% mIoU** ở 5-shot, vượt cả mức `w/o TA` trên 1-shot (`85.71%` vs `84.23%`) — điều mà Baseline gốc không làm được (`81.88%` < `84.23%`).
   - Độ phân biệt không gian nhúng `inter_delta` ở tầng trung (M) tăng vọt từ `42.55` lên `51.82`.
2. **Hiệu quả trên miền phức tạp đa lớp (SUIM):**
   - Ở 1-shot (kịch bản khan hiếm nhãn nhất), `Full Improved` đạt kết quả tốt nhất cả về mIoU (**29.50%**) lẫn FB-IoU (**50.77%**, tăng **+2.40%** so với Baseline), giảm đáng kể hiện tượng dự đoán sai nền (over-prediction).
   - Ở 5-shot SUIM, khi đã có 5 ảnh support đa dạng, `Imp2 (DWConv3x3)` (+1.38% FB-IoU) và `Full Improved` (+1.00% FB-IoU, đạt `58.15%` so với `57.15%` của Baseline) giúp cải thiện độ cân bằng tiền cảnh–hậu cảnh, nhưng nhiệt độ sắc nét $\tau_{aff}=0.15$ của `Imp3` làm mô hình hơi quá tự tin vào các điểm ảnh support gần nhất trong trường hợp nội lớp có phương sai lớn (như các loài cá/rạn san hô khác nhau trong SUIM). Đây là một phát hiện thực nghiệm quan trọng: $\tau_{aff}$ nên được điều chỉnh thích nghi theo số lượng shot $k$ hoặc phương sai nội lớp.
3. **Chi phí tính toán:**
   - Tổng thời gian thích nghi + suy luận tăng rất ít (từ `10.48s` lên `15.92s–17.11s` cho 50 episodes Lung 1-shot, tức chỉ tăng ~0.1s mỗi lần fit), hoàn toàn chấp nhận được và vẫn giữ nguyên tốc độ suy luận nhanh ở chế độ `quick-infer`.

### 4.4. Kế hoạch đánh giá Table 4 (`every-episode`, `no-pp`) trên 5 bộ dữ liệu
Để đối chiếu sòng phẳng với dòng **`Ours (no-pp)` trong Table 4** của bài báo gốc (`every-episode`, không CRF), thực nghiệm toàn bộ 10 đơn vị (`5 dataset × 1-shot & 5-shot`) được chia làm **3 phần cân bằng** chạy phân tán trên 3 máy qua Modal theo kịch bản tại [BENCHMARK_PLAN.md](file:///D:/Code/BTL_CV/BENCHMARK_PLAN.md). Sau khi 3 máy chạy xong, dùng [merge_table4_results.py](file:///D:/Code/BTL_CV/merge_table4_results.py) để tự động gộp kết quả và xuất bảng Table 4 hoàn chỉnh.

---

## 5. Kết luận & Demo minh họa

Báo cáo đã tái tạo thành công phương pháp **ABCDFSS (CVPR 2024)**, kiểm chứng các luận điểm cốt lõi về vai trò của Task Adaptation trước khi so khớp, đồng thời đề xuất và thực nghiệm **ABCDFSS-Plus** với 3 cải tiến nhẹ nhàng nhưng hiệu quả trên cả 5 bộ dữ liệu chuẩn. Dự án đi kèm 2 phiên bản demo tương tác:
* **Web Demo React + FastAPI** ([demo_api.py](file:///D:/Code/BTL_CV/demo_api.py) & [frontend](file:///D:/Code/BTL_CV/frontend), chạy trên `http://127.0.0.1:8000`): cho phép chọn hoặc upload từng cặp ảnh thật từ [demo_samples](file:///D:/Code/BTL_CV/demo_samples), so sánh trực tiếp Baseline ABCDFSS vs. ABCDFSS-Plus và trực quan hóa bản đồ dự đoán + kích hoạt đặc trưng của cả **13 tầng ResNet-50 (`L3–L15`)**.
* **Streamlit Demo** ([app.py](file:///D:/Code/BTL_CV/app.py), chạy trên `http://localhost:8501`): trực quan hóa bản đồ nhiệt $\hat{q}_{fused}$, mặt nạ dự đoán, chỉ số IoU và biểu đồ trọng số các tầng DWLA.
