# PROJECT MASTER SPEC: Sequence-to-Sequence Weather Forecasting (Topic 5) 

## 1. DEFINE (Định nghĩa Bài toán, Dữ liệu & Quy tắc Chống Gian Lận)

### 1.1. Purpose (Mục đích dự án - WHY & WHO)
- **Mục đích (Why):** Xây dựng hệ thống AI dự báo nhiệt độ liên tiếp trong 3 ngày tới (72 giờ) dựa vào dữ liệu thời tiết của 7 ngày trước đó (168 giờ). Qua đó, chứng minh việc thêm cơ chế "Chú ý" (Attention) và kiến trúc Transformer giúp dự báo dài hạn tốt hơn mô hình LSTM truyền thống.
- **Vai trò của AI (Who):** Bạn đóng vai trò là Kỹ sư Học sâu (Senior Deep Learning Engineer), viết code bằng PyTorch rõ ràng, dễ hiểu và tuân thủ tuyệt đối các bài kiểm tra tự động (`assert`) ở cuối mỗi file.

### 1.2. Input / Output Contract (Đầu vào & Đầu ra cố định - WHAT)
- **Dữ liệu gốc:** Bộ dữ liệu thời tiết Jena Climate (~420.551 dòng, đo 10 phút/lần với 14 thông số khí tượng). Sau khi gom trung bình theo từng giờ (`1h`), ta có khoảng ~70.091 giờ dữ liệu.
- **Đầu vào (Input `X`):** Khối dữ liệu kích thước `(Batch_size, 168, 14)` — tức là 168 giờ (7 ngày) quá khứ của đầy đủ 14 thông số thời tiết.
- **Đầu ra chính (Output `Y`):** Khối dữ liệu kích thước `(Batch_size, 72, 1)` — tức là 72 giờ (3 ngày) nhiệt độ `T (degC)` trong tương lai.
- **Đầu ra phụ (Giải thích mô hình):** Bảng trọng số `attention_weights` kích thước `(Batch_size, 72, 168)` cho biết mô hình đã "nhìn" vào giờ nào trong quá khứ để đoán tương lai (dành cho Model 2 & 3; Model 1 trả về `None`).

### 1.3. Strict Anti-Leakage Protocol (Quy trình 4 pha chống "Lộ đề" - WHEN)
> **Căn cứ khoa học (Verifiable References):**
> Quy trình này tuân thủ nghiêm ngặt tiêu chuẩn chống rò rỉ dữ liệu từ:
> 1. *Scikit-Learn Official Documentation (Section 12.2: Data Leakage)*: Tuyệt đối không gọi hàm `.fit()` trên dữ liệu kiểm thử (Val/Test); mọi bước tiền xử lý có tính toán thống kê phải học riêng từ tập Train.
> 2. *An Introduction to Statistical Learning (ISLR - James et al., Section 5.1.5)* & *Feature Engineering and Selection (Max Kuhn, Chapter 3.4)*: Mọi bước ước lượng tham số (trung bình, độ lệch chuẩn, điền khuyết) phải diễn ra **SAU KHI** đã chia tách hoàn toàn cả 3 tập Train / Validation / Test.
> 3. *Nghiên cứu của Đại học Princeton (Kapoor & Narayanan, 2023 - Tạp chí Patterns)*: Xử lý dữ liệu chung trên toàn bộ tập dữ liệu trước khi chia tách là nguyên nhân số 1 khiến mô hình học máy đạt điểm ảo khi train nhưng thất bại ngoài thực tế.

Để hiểu đơn giản nhất, hãy tưởng tượng **Tập Train** là *Sách bài tập ở nhà*, **Tập Validation (Val)** là *Đề thi thử*, và **Tập Test** là *Đề thi tốt nghiệp cuối cùng*. Để mô hình học thật, thi thật, AI bắt buộc phải xử lý dữ liệu theo đúng **4 pha tuần tự** sau:

- **Pha 1 – Chia xong cả 3 phần TRƯỚC khi chạm vào số liệu (Split First):**
  - Đọc file CSV, đổi các số đo bị hỏng của máy đo gió (`-9999.0`) thành ô trống (`NaN`), và gom dữ liệu từ 10 phút về khung **1 giờ (`1h`)** (chỉ tính trung bình trong phạm vi từng giờ, tuyệt đối chưa nối dữ liệu giữa các giờ).
  - Cắt ngay dữ liệu theo đúng thứ tự thời gian từ quá khứ đến tương lai thành 3 phần riêng biệt: **70% đầu làm `train_df`**, **15% tiếp theo làm `val_df`**, và **15% cuối cùng làm `test_df`**. Cất kỹ `val_df` và `test_df` sang một bên.
- **Pha 2 – Chỉ khám phá và vẽ biểu đồ trên Tập Train (EDA on Train Only):**
  - Chỉ dùng 70% `train_df` để vẽ biểu đồ xem quy luật mùa, tìm điểm bất thường và xem mối liên hệ giữa các biến.
  - *Lý do dễ hiểu:* Nếu bạn nhìn vào `val_df` hoặc `test_df` để vẽ biểu đồ, mắt bạn đã "nhìn trộm đề thi" (gọi là *Rò rỉ nhận thức - Cognitive Leakage*) rồi mới quay lại sửa cách làm bài.
- **Pha 3 – Điền ô trống và Chuẩn hóa chỉ lấy công thức từ Tập Train (Fit on Train, Apply to All):**
  - **Điền ô trống (Missing values):** Ô trống ở tập nào thì tự lấy giờ liền trước trong chính tập đó để lấp vào (`ffill`), không lấy số cuối tập Train đắp sang đầu tập Val/Test.
  - **Đưa dữ liệu về cùng thang đo (StandardScaler):** Chỉ dùng hàm `.fit()` trên `train_df` để tính số Trung bình (Mean) và Độ lệch chuẩn (Std). Sau đó, cầm đúng 2 con số của tập Train đó áp vào (`.transform()`) để tính cho `val_df` và `test_df`.
- **Pha 4 – Cắt cửa sổ 7 ngày -> 3 ngày riêng trong từng tập (Isolated Sliding Windows):**
  - Tạo các cặp dữ liệu (7 ngày quá khứ -> 3 ngày tương lai) độc lập bên trong từng tập `train_df`, `val_df`, `test_df`. Tuyệt đối không ghép chung lại rồi mới cắt, để tránh việc một mẫu dữ liệu nằm vắt ngang qua ranh giới giữa hai tập.

### 1.4. Success Criteria (Tiêu chí nghiệm thu kết quả)
1. **Chứng minh không "lộ đề" (Zero Leakage):** Vượt qua lệnh kiểm tra tự động: thời gian của `train < val < test` và bộ `StandardScaler` chỉ đếm đúng số dòng của `train_df`.
2. **Độ chính xác (Accuracy):** Đổi kết quả dự đoán về lại độ C thực tế và tính sai số **MAE, MSE, RMSE, $R^2$**, cùng tỷ lệ % số giờ dự đoán lệch không quá $\pm 1.5^\circ\text{C}$.
3. **Tốc độ & Tài nguyên (Speed & Hardware):** Đo rõ mất bao nhiêu mili-giây (`ms`) để mô hình tính từ `Input -> Output`, và tốn bao nhiêu MB bộ nhớ **RAM** cũng như **VRAM (GPU)**.

---

## 2. DESIGN (Thiết kế Kiến trúc, Sơ đồ Luồng & Cách Thực hiện)

### 2.1 Cấu trúc file

```text
course_work_topic_5_LSTM_regression_jena_climate_dataset/
│
├── configs/                        # Thư mục chứa các file cấu hình hệ thống
│   └── config.yaml                 # Lưu toàn bộ siêu tham số (seq_len=168, pred_len=72, batch_size, lr, d_model...)
│
├── docs/                           # Tài liệu đặc tả kỹ thuật phục vụ quy trình Vibe Coding (Spec-Driven)
│   ├── data.md                     # Đặc tả luồng xử lý dữ liệu, EDA và quy tắc chống rò rỉ (Anti-Data Leakage)
│   ├── demo.md                     # Đặc tả thiết kế giao diện ứng dụng Demo tương tác (Streamlit / FastAPI)
│   ├── evalution.md                # Đặc tả kịch bản đánh giá (MAE, RMSE, R2, Latency ms, RAM/VRAM & Error Analysis)
│   ├── course_work_topic_5_decription.md # Tài liệu gốc (Master Spec) định nghĩa tổng thể dự án (Define & Design)
│   ├── model.md                    # Đặc tả kiến trúc 3 mô hình (Baseline Seq2Seq LSTM, Attention LSTM, Transformer)
│   └── training.md                 # Đặc tả quy trình huấn luyện, Teacher Forcing, Early Stopping & Gradient Clipping
│
├── data/                           # Thư mục lưu bộ dữ liệu gốc Jena Climate (jena_climate_2009_2016.csv)
│
├── src/                            # Mã nguồn chính của dự án (Được thiết kế theo chuẩn mô-đun hóa / OOP)
│   ├── __init__.py                 # Khởi tạo biến thư mục src thành một Python Package
│   ├── data_module.py              # Class JenaDataModule (Resample 1h, chia Train/Val/Test, StandardScaler, Sliding Window DataLoader)
│   ├── model_builder.py            # Class khởi tạo và quản lý 3 mô hình (Seq2SeqLSTM, Seq2SeqAttention, TimeSeriesTransformer)
│   ├── trainer.py                  # Class ModelTrainer (Vòng lặp huấn luyện, Validation, lưu Checkpoint & đo tài nguyên RAM/VRAM)
│   ├── logger.py                   # Class TensorBoardLogger (Ghi nhận lịch sử Train/Val Loss, Learning Rate và biểu đồ dự báo)
│   ├── eda.py                      # Script phân tích khám phá dữ liệu trên tập Train (Xu hướng nhiệt độ, phân phối, Correlation Heatmap)
│   └── evaluation.py               # Script chấm điểm trên tập Test, giải chuẩn hóa về độ C, đo tốc độ (Input -> Output) & vẽ Attention Heatmap
│
├── evalution_results/              # Thư mục lưu trữ toàn bộ kết quả đầu ra sau khi chạy đánh giá trên tập Test
│   └── evaluation_YYYYMMDD/        # Thư mục con lưu bảng chỉ số (CSV/JSON) và hình ảnh biểu đồ theo từng ngày chạy thực nghiệm
│
├── runs/                           # Thư mục lưu log của TensorBoard (SummaryWriter) để theo dõi quá trình hội tụ trực quan
│
├── checkpoints/                    # Thư mục lưu file trọng số mô hình tốt nhất sau huấn luyện (.pth / .pt) và bộ Scaler (.pkl)
│
├── notebooks/                      # Thư mục chứa Jupyter Notebooks phục vụ chạy thử nghiệm nhanh trên Colab / Local
│   ├── experiment_demo.ipynb       # Notebook chạy thử nghiệm mô hình, kiểm tra dự báo và đo nhanh hiệu năng
│   └── jena_climate_eda.ipynb      # Notebook trực quan hóa dữ liệu khám phá (EDA) và kiểm thử quy trình tiền xử lý
│
├── main.py                         # Điểm khởi chạy chính (Entry point) điều phối toàn bộ Pipeline từ tiền xử lý đến huấn luyện & đánh giá
├── requirements.txt                # Danh sách các thư viện cần cài đặt (torch, pandas, numpy, scikit-learn, matplotlib, streamlit, psutil...)
└── README.md                       # Tài liệu hướng dẫn cài đặt môi trường, cấu trúc dự án và các câu lệnh chạy chương trình
```
# PROJECT MASTER SPEC: Sequence-to-Sequence Weather Forecasting (Topic 5) 

## 1. DEFINE (Định nghĩa Bài toán, Dữ liệu & Quy tắc Chống Gian Lận)

### 1.1. Purpose (Mục đích dự án - WHY & WHO)
- **Mục đích (Why):** Xây dựng hệ thống AI dự báo nhiệt độ liên tiếp trong 3 ngày tới (72 giờ) dựa vào dữ liệu thời tiết của 7 ngày trước đó (168 giờ). Qua đó, chứng minh việc thêm cơ chế "Chú ý" (Attention) và kiến trúc Transformer giúp dự báo dài hạn tốt hơn mô hình LSTM truyền thống.
- **Vai trò của AI (Who):** Bạn đóng vai trò là Kỹ sư Học sâu (Senior Deep Learning Engineer), viết code bằng PyTorch rõ ràng, dễ hiểu và tuân thủ tuyệt đối các bài kiểm tra tự động (`assert`) ở cuối mỗi file.

### 1.2. Input / Output Contract (Đầu vào & Đầu ra cố định - WHAT)
- **Dữ liệu gốc:** Bộ dữ liệu thời tiết Jena Climate (~420.551 dòng, đo 10 phút/lần với 14 thông số khí tượng). Sau khi gom trung bình theo từng giờ (`1h`), ta có khoảng ~70.091 giờ dữ liệu.
- **Đầu vào (Input `X`):** Khối dữ liệu kích thước `(Batch_size, 168, 14)` — tức là 168 giờ (7 ngày) quá khứ của đầy đủ 14 thông số thời tiết.
- **Đầu ra chính (Output `Y`):** Khối dữ liệu kích thước `(Batch_size, 72, 1)` — tức là 72 giờ (3 ngày) nhiệt độ `T (degC)` trong tương lai.
- **Đầu ra phụ (Giải thích mô hình):** Bảng trọng số `attention_weights` kích thước `(Batch_size, 72, 168)` cho biết mô hình đã "nhìn" vào giờ nào trong quá khứ để đoán tương lai (dành cho Model 2 & 3; Model 1 trả về `None`).

### 1.3. Strict Anti-Leakage Protocol (Quy trình 4 pha chống "Lộ đề" - WHEN)
> **Căn cứ khoa học (Verifiable References):**
> Quy trình này tuân thủ nghiêm ngặt tiêu chuẩn chống rò rỉ dữ liệu từ:
> 1. *Scikit-Learn Official Documentation (Section 12.2: Data Leakage)*: Tuyệt đối không gọi hàm `.fit()` trên dữ liệu kiểm thử (Val/Test); mọi bước tiền xử lý có tính toán thống kê phải học riêng từ tập Train.
> 2. *An Introduction to Statistical Learning (ISLR - James et al., Section 5.1.5)* & *Feature Engineering and Selection (Max Kuhn, Chapter 3.4)*: Mọi bước ước lượng tham số (trung bình, độ lệch chuẩn, điền khuyết) phải diễn ra **SAU KHI** đã chia tách hoàn toàn cả 3 tập Train / Validation / Test.
> 3. *Nghiên cứu của Đại học Princeton (Kapoor & Narayanan, 2023 - Tạp chí Patterns)*: Xử lý dữ liệu chung trên toàn bộ tập dữ liệu trước khi chia tách là nguyên nhân số 1 khiến mô hình học máy đạt điểm ảo khi train nhưng thất bại ngoài thực tế.

Để hiểu đơn giản nhất, hãy tưởng tượng **Tập Train** là *Sách bài tập ở nhà*, **Tập Validation (Val)** là *Đề thi thử*, và **Tập Test** là *Đề thi tốt nghiệp cuối cùng*. Để mô hình học thật, thi thật, AI bắt buộc phải xử lý dữ liệu theo đúng **4 pha tuần tự** sau:

- **Pha 1 – Chia xong cả 3 phần TRƯỚC khi chạm vào số liệu (Split First):**
  - Đọc file CSV, đổi các số đo bị hỏng của máy đo gió (`-9999.0`) thành ô trống (`NaN`), và gom dữ liệu từ 10 phút về khung **1 giờ (`1h`)** (chỉ tính trung bình trong phạm vi từng giờ, tuyệt đối chưa nối dữ liệu giữa các giờ).
  - Cắt ngay dữ liệu theo đúng thứ tự thời gian từ quá khứ đến tương lai thành 3 phần riêng biệt: **70% đầu làm `train_df`**, **15% tiếp theo làm `val_df`**, và **15% cuối cùng làm `test_df`**. Cất kỹ `val_df` và `test_df` sang một bên.
- **Pha 2 – Chỉ khám phá và vẽ biểu đồ trên Tập Train (EDA on Train Only):**
  - Chỉ dùng 70% `train_df` để vẽ biểu đồ xem quy luật mùa, tìm điểm bất thường và xem mối liên hệ giữa các biến.
  - *Lý do dễ hiểu:* Nếu bạn nhìn vào `val_df` hoặc `test_df` để vẽ biểu đồ, mắt bạn đã "nhìn trộm đề thi" (gọi là *Rò rỉ nhận thức - Cognitive Leakage*) rồi mới quay lại sửa cách làm bài.
- **Pha 3 – Điền ô trống và Chuẩn hóa chỉ lấy công thức từ Tập Train (Fit on Train, Apply to All):**
  - **Điền ô trống (Missing values):** Ô trống ở tập nào thì tự lấy giờ liền trước trong chính tập đó để lấp vào (`ffill`), không lấy số cuối tập Train đắp sang đầu tập Val/Test.
  - **Đưa dữ liệu về cùng thang đo (StandardScaler):** Chỉ dùng hàm `.fit()` trên `train_df` để tính số Trung bình (Mean) và Độ lệch chuẩn (Std). Sau đó, cầm đúng 2 con số của tập Train đó áp vào (`.transform()`) để tính cho `val_df` và `test_df`.
- **Pha 4 – Cắt cửa sổ 7 ngày -> 3 ngày riêng trong từng tập (Isolated Sliding Windows):**
  - Tạo các cặp dữ liệu (7 ngày quá khứ -> 3 ngày tương lai) độc lập bên trong từng tập `train_df`, `val_df`, `test_df`. Tuyệt đối không ghép chung lại rồi mới cắt, để tránh việc một mẫu dữ liệu nằm vắt ngang qua ranh giới giữa hai tập.

### 1.4. Success Criteria (Tiêu chí nghiệm thu kết quả)
1. **Chứng minh không "lộ đề" (Zero Leakage):** Vượt qua lệnh kiểm tra tự động: thời gian của `train < val < test` và bộ `StandardScaler` chỉ đếm đúng số dòng của `train_df`.
2. **Độ chính xác (Accuracy):** Đổi kết quả dự đoán về lại độ C thực tế và tính sai số **MAE, MSE, RMSE, $R^2$**, cùng tỷ lệ % số giờ dự đoán lệch không quá $\pm 1.5^\circ\text{C}$.
3. **Tốc độ & Tài nguyên (Speed & Hardware):** Đo rõ mất bao nhiêu mili-giây (`ms`) để mô hình tính từ `Input -> Output`, và tốn bao nhiêu MB bộ nhớ **RAM** cũng như **VRAM (GPU)**.

---

## 2. DESIGN (Cấu trúc file, Thiết kế Kiến trúc,  & Cách Thực hiện)

### 2.1 Cấu trúc file

```text
course_work_topic_5_LSTM_regression_jena_climate_dataset/
│
├── configs/                        # Thư mục chứa các file cấu hình hệ thống
│   └── config.yaml                 # Lưu toàn bộ siêu tham số (seq_len=168, pred_len=72, batch_size, lr, d_model...)
│
├── docs/                           # Tài liệu đặc tả kỹ thuật phục vụ quy trình Vibe Coding (Spec-Driven)
│   ├── data.md                     # Đặc tả luồng xử lý dữ liệu, EDA và quy tắc chống rò rỉ (Anti-Data Leakage)
│   ├── demo.md                     # Đặc tả thiết kế giao diện ứng dụng Demo tương tác (Streamlit / FastAPI)
│   ├── evalution.md                # Đặc tả kịch bản đánh giá (MAE, RMSE, R2, Latency ms, RAM/VRAM & Error Analysis)
│   ├── course_work_topic_5_decription.md # Tài liệu gốc (Master Spec) định nghĩa tổng thể dự án (Define & Design)
│   ├── model.md                    # Đặc tả kiến trúc 3 mô hình (Baseline Seq2Seq LSTM, Attention LSTM, Transformer)
│   └── training.md                 # Đặc tả quy trình huấn luyện, Teacher Forcing, Early Stopping & Gradient Clipping
│
├── data/                           # Thư mục lưu bộ dữ liệu gốc Jena Climate (jena_climate_2009_2016.csv)
│
├── src/                            # Mã nguồn chính của dự án (Được thiết kế theo chuẩn mô-đun hóa / OOP)
│   ├── __init__.py                 # Khởi tạo biến thư mục src thành một Python Package
│   ├── data_module.py              # Class JenaDataModule (Resample 1h, chia Train/Val/Test, StandardScaler, Sliding Window DataLoader)
│   ├── model_builder.py            # Class khởi tạo và quản lý 3 mô hình (Seq2SeqLSTM, Seq2SeqAttention, TimeSeriesTransformer)
│   ├── trainer.py                  # Class ModelTrainer (Vòng lặp huấn luyện, Validation, lưu Checkpoint & đo tài nguyên RAM/VRAM)
│   ├── logger.py                   # Class TensorBoardLogger (Ghi nhận lịch sử Train/Val Loss, Learning Rate và biểu đồ dự báo)
│   ├── eda.py                      # Script phân tích khám phá dữ liệu trên tập Train (Xu hướng nhiệt độ, phân phối, Correlation Heatmap)
│   └── evaluation.py               # Script chấm điểm trên tập Test, giải chuẩn hóa về độ C, đo tốc độ (Input -> Output) & vẽ Attention Heatmap
│
├── evalution_results/              # Thư mục lưu trữ toàn bộ kết quả đầu ra sau khi chạy đánh giá trên tập Test
│   └── evaluation_YYYYMMDD/        # Thư mục con lưu bảng chỉ số (CSV/JSON) và hình ảnh biểu đồ theo từng ngày chạy thực nghiệm
│
├── runs/                           # Thư mục lưu log của TensorBoard (SummaryWriter) để theo dõi quá trình hội tụ trực quan
│
├── checkpoints/                    # Thư mục lưu file trọng số mô hình tốt nhất sau huấn luyện (.pth / .pt) và bộ Scaler (.pkl)
│
├── notebooks/                      # Thư mục chứa Jupyter Notebooks phục vụ chạy thử nghiệm nhanh trên Colab / Local
│   ├── experiment_demo.ipynb       # Notebook chạy thử nghiệm mô hình, kiểm tra dự báo và đo nhanh hiệu năng
│   └── jena_climate_eda.ipynb      # Notebook trực quan hóa dữ liệu khám phá (EDA) và kiểm thử quy trình tiền xử lý
│
├── main.py                         # Điểm khởi chạy chính (Entry point) điều phối toàn bộ Pipeline từ tiền xử lý đến huấn luyện & đánh giá
├── requirements.txt                # Danh sách các thư viện cần cài đặt (torch, pandas, numpy, scikit-learn, matplotlib, streamlit, psutil...)
└── README.md                       # Tài liệu hướng dẫn cài đặt môi trường, cấu trúc dự án và các câu lệnh chạy chương trình
```
### 2.2 Quy tắc viết code 
#### Quy tắc kỹ thuật thống nhất
- **Giao diện hàm chung (Unified Interface):** Cả 3 mô hình bắt buộc viết hàm `forward` giống hệt nhau về đầu vào và đầu ra để dùng chung 1 file huấn luyện:
  `forward(self, src, trg=None, teacher_forcing_ratio=0.0) -> Tuple[Tensor, Optional[Tensor]]`
- **Cách dạy mô hình (Teacher Forcing):** Khi đang học (Train), thỉnh thoảng cho mô hình LSTM biết nhiệt độ thật của giờ trước để đoán giờ sau (giảm dần tỷ lệ nhắc bài theo từng vòng). Khi thi thử (Val) và thi thật (Test), bắt buộc tắt nhắc bài (`teacher_forcing_ratio = 0.0`).
- **Giữ mô hình ổn định:** Luôn dùng `clip_grad_norm_(max_norm=1.0)` để số không bị vọt lên vô cực, và dùng `EarlyStopping` để tự động dừng lại khi mô hình bắt đầu học vẹt.

#### Quy chuẩn viết Docstring `"""..."""` bắt buộc cho mọi Class và Hàm
Khi sinh mã nguồn Python, AI **BẮT BUỘC** phải viết chú thích bằng tiếng Việt đặt trong cặp dấu `"""..."""` ngay dưới tên mỗi Class và mỗi Hàm để người đọc hiểu rõ luồng xử lý:

1. **Đối với Class (Lớp):** Phải ghi rõ 3 gạch đầu dòng:
   - Mục đích của Class đó trong bài toán Jena Climate là gì?
   - Dữ liệu hoặc kiến trúc bên trong bao gồm những thành phần nào (ví dụ: 14 thuộc tính khí tượng, các tầng Encoder/Decoder/Attention)?
   - Bản chất bài toán đang giải quyết (Hồi quy đa bước dự báo 72 giờ nhiệt độ `T (degC)` tiếp theo, hoặc quy tắc chống rò rỉ dữ liệu).

   *Mẫu chuẩn bắt buộc cho Class:*
   ```python
   class ModelBuilder_Seq2SeqAttention(nn.Module):
       """Xây dựng mô hình Encoder-Decoder LSTM kết hợp Attention cho bộ dữ liệu Jena Climate.
       - Dữ liệu: Nhận chuỗi 168 giờ quá khứ (bên trong bao gồm: 14 thuộc tính khí tượng đã gộp mẫu 1h và chuẩn hóa từ tập Train).
       - Kiến trúc bên trong bao gồm: EncoderLSTM trích xuất đặc trưng, BahdanauAttention tính trọng số chú ý, và DecoderLSTM giải mã từng bước.
       - Đây là bài toán Hồi quy đa bước (Multi-step Regression) dự báo chuỗi nhiệt độ T (degC) trong 72 giờ (3 ngày) tiếp theo.
       """

2. Đối với Hàm / Phương thức (def): Phải ghi rõ 3 gạch đầu dòng:

  - Hàm này thực hiện hành động gì?

  - Input: Tên biến đầu vào, ý nghĩa và kích thước Tensor (Shape) cụ thể.

  - Output: Kết quả trả về và kích thước Tensor (Shape) đầu ra.

  *Mẫu chuẩn bắt buộc cho Hàm:*
  ```python
  def forward(self, src, trg=None, teacher_forcing_ratio=0.0):
      """Thực hiện lan truyền tiến (Forward pass) dự báo nhiệt độ 3 ngày tới.
      - Input:
          + src: Chuỗi thời tiết quá khứ, shape (Batch_size, 168, 14).
          + trg: Chuỗi nhiệt độ thật dùng để nhắc bài khi train, shape (Batch_size, 72, 1).
          + teacher_forcing_ratio: Tỷ lệ nhắc bài (bằng 0.0 khi chạy Val/Test).
      - Output:
          + predictions: Chuỗi nhiệt độ dự báo, shape (Batch_size, 72, 1).
          + attn_weights: Bảng trọng số Attention giải thích mô hình, shape (Batch_size, 72, 168) hoặc None.
      """
  ```