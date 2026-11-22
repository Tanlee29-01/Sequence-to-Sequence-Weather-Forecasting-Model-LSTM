# Sequence-to-Sequence Weather Forecasting (Topic 5)
## Dự Báo Nhiệt Độ Chuỗi Thời Gian Đa Bước (Jena Climate Dataset)

Dự án này triển khai hệ thống học sâu dự báo nhiệt độ liên tiếp trong 3 ngày tới (72 giờ) dựa vào dữ liệu khí tượng của 7 ngày trước đó (168 giờ) từ bộ dữ liệu Jena Climate, so sánh 3 kiến trúc:
1. **Baseline Seq2Seq LSTM**: Mô hình Encoder-Decoder LSTM cơ sở không có Attention.
2. **Seq2Seq Attention LSTM**: Encoder-Decoder LSTM kết hợp cơ chế Bahdanau Additive Attention.
3. **TimeSeries Transformer**: Kiến trúc Transformer hiện đại với Multi-Head Attention và Causal Masking.

---

## 1. Cấu Trúc Dự Án (Project Structure)

```text
course_work_topic_5/
│
├── configs/                        # Cấu hình hệ thống và siêu tham số
│   └── config.yaml                 # Lưu toàn bộ tham số (seq_len=168, pred_len=72, lr, batch_size...)
│
├── docs/                           # Tài liệu đặc tả kỹ thuật dự án
│   ├── course_work_topic_5_description.md # Master Spec định nghĩa tổng thể dự án
│   ├── data.md                     # Đặc tả luồng xử lý dữ liệu và quy tắc chống rò rỉ
│   ├── demo.md                     # Đặc tả thiết kế giao diện ứng dụng Demo
│   ├── evalution.md                # Đặc tả kịch bản đánh giá (MAE, RMSE, R2, Latency, RAM/VRAM)
│   ├── model.md                    # Đặc tả kiến trúc 3 mô hình
│   └── training.md                 # Đặc tả quy trình huấn luyện & Teacher Forcing
│
├── data/                           # Thư mục chứa bộ dữ liệu gốc Jena Climate
│   └── jena_climate_2009_2016.csv  # (Tải về và đặt tại đây)
│
├── src/                            # Mã nguồn chính theo chuẩn OOP và Module hóa
│   ├── __init__.py                 # Đóng gói package src
│   ├── data_module.py              # JenaDataModule (Resample 1h, chia Train/Val/Test, Scaler, Sliding Window)
│   ├── model_builder.py            # Khởi tạo 3 mô hình (Seq2SeqLSTM, Seq2SeqAttention, Transformer)
│   ├── trainer.py                  # ModelTrainer (Vòng lặp train, val, Early Stopping, Gradient Clipping)
│   ├── logger.py                   # TensorBoardLogger (Ghi nhận loss, lr, hardware, forecast figures)
│   ├── eda.py                      # Phân tích khám phá dữ liệu chỉ trên tập Train (Zero Leakage)
│   └── evaluation.py               # Đánh giá Test, giải chuẩn hóa °C, đo độ trễ ms, vẽ Attention Heatmap
│
├── tests/                          # Bộ kiểm thử tự động (Unit & Smoke Tests) cho gói src/
│   ├── test_data_module.py         # Kiểm thử DataModule & quy tắc chống rò rỉ dữ liệu
│   ├── test_model_builder.py       # Kiểm thử 3 kiến trúc mô hình học sâu & Factory
│   ├── test_trainer.py             # Kiểm thử quy trình huấn luyện & Checkpoints
│   ├── test_evaluation.py          # Kiểm thử kịch bản đánh giá & Benchmark độ trễ
│   ├── test_logger.py              # Kiểm thử bộ ghi nhận trực quan TensorBoard
│   ├── test_eda.py                 # Kiểm thử EDA pipeline
│   └── run_all_tests.py            # Master Test Runner điều phối chạy toàn bộ kiểm thử
│
├── testing/                        # Thư mục kiểm thử mở rộng ánh xạ tới bộ test suite
│
├── evalution_results/              # Thư mục lưu bảng chỉ số và biểu đồ đầu ra thực nghiệm
│   └── evaluation_YYYYMMDD/        # Thư mục con theo từng ngày chạy thực nghiệm
│
├── runs/                           # Nhật ký sự kiện trực quan của TensorBoard
│
├── checkpoints/                    # File trọng số mô hình tốt nhất (.pth)
│
├── notebooks/                      # Jupyter Notebooks phục vụ thử nghiệm và trực quan hóa
│   ├── experiment_demo.ipynb       # Thử nghiệm so sánh mô hình, đo độ trễ và trực quan hóa Attention
│   └── jena_climate_eda.ipynb      # Khám phá dữ liệu EDA chỉ trên tập Train
│
├── app.py                          # Demo CLI: nạp model/checkpoint, dự báo 72h nhiệt độ, lưu biểu đồ PNG và Attention Heatmap vào evalution_results/
├── main.py                         # Điểm khởi chạy chính điều phối toàn bộ Pipeline
├── requirements.txt                # Danh sách thư viện cần thiết
└── README.md                       # Hướng dẫn cài đặt và sử dụng
```

---

## 2. Quy Trình Chống Rò Rỉ Dữ Liệu 4 Pha (Strict Anti-Leakage Protocol)

Hệ thống tuân thủ nghiêm ngặt chuẩn khoa học chống "lộ đề":
- **Pha 1 (Split First):** Đổi các giá trị đo gió lỗi (`-9999.0`) thành `NaN`, gom trung bình theo từng giờ (`1h`), và chia ngay thành 3 tập tuần tự theo thời gian: **70% Train**, **15% Validation**, và **15% Test**.
- **Pha 2 (EDA on Train Only):** Chỉ phân tích và vẽ biểu đồ trên 70% tập Train để tránh rò rỉ nhận thức (Cognitive Leakage).
- **Pha 3 (Fit on Train, Apply to All):** Xử lý `ffill()` nội bộ từng phân tập và chỉ gọi `.fit()` của `StandardScaler` trên tập Train, sau đó dùng đúng thông số đó để `.transform()` cho Val và Test.
- **Pha 4 (Isolated Sliding Windows):** Cắt cửa sổ trượt (168 giờ quá khứ $\rightarrow$ 72 giờ tương lai) độc lập bên trong từng tập, không cắt vắt ngang qua ranh giới.

---

## 3. Cài Đặt Môi Trường (Installation)

### 3.1 Khởi tạo môi trường ảo Python
```bash
# Trên Windows (cmd)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.bat

# Trên Linux / macOS
python -3.11 -m venv venv
source venv/bin/activate
```

### 3.2 Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

---

## 4. Tải Dữ Liệu Jena Climate

Tải file `jena_climate_2009_2016.csv.zip` từ trang web chính thức của Max Planck Institute for Biogeochemistry hoặc TensorFlow Datasets:
- [Link tải trực tiếp Jena Climate CSV](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)
- Giải nén file `jena_climate_2009_2016.csv` và đặt vào thư mục `data/`:
```text
data/jena_climate_2009_2016.csv
```

---

## 5. Hướng Dẫn Sử Dụng (Usage)

### 5.1 Chạy Phân Tích Khám Phá Dữ Liệu (EDA trên tập Train)
```bash
python main.py --mode eda
```
Biểu đồ sẽ được lưu tự động tại `evalution_results/eda/`.

### 5.2 Huấn Luyện Mô Hình (Training)
Lựa chọn một trong ba mô hình:
```bash
# Model 1: Baseline Seq2Seq LSTM
python main.py --mode train --model seq2seq_lstm --epochs 30

# Model 2: Seq2Seq Attention LSTM (Khuyên dùng)
python main.py --mode train --model seq2seq_attention --epochs 30

# Model 3: TimeSeries Transformer
python main.py --mode train --model transformer --epochs 30
```

### 5.3 Đánh Giá Mô Hình Trên Tập Test (Evaluation)
```bash
python main.py --mode evaluate --model seq2seq_attention
```
Kết quả báo cáo chỉ số (MAE, MSE, RMSE, R2, Tỷ lệ sai số $\le \pm 1.5^\circ\text{C}$), biểu đồ so sánh chuỗi 72h và Attention Heatmap được lưu tại thư mục `evalution_results/evaluation_YYYYMMDD/`.

### 5.4 Chạy Toàn Bộ Đường Ống (End-to-End Pipeline)
```bash
python main.py --mode all --model seq2seq_attention
```

### 5.5 Theo Dõi Qua TensorBoard
```bash
tensorboard --logdir runs
```
Truy cập trình duyệt tại `http://localhost:6006` để theo dõi đồ thị loss, learning rate và biểu đồ dự báo.

### 5.6 Chạy Demo CLI (Demo dự báo nhiệt độ 72 giờ)
```bash
# Tự động tìm checkpoint tốt nhất cho mô hình Seq2Seq Attention
python app.py --model seq2seq_attention

# Chỉ định rõ checkpoint và thiết bị
python app.py --model transformer --checkpoint checkpoints/transformer_20261116-204500_best.pth --device cpu

# Chỉ định thư mục đầu ra tùy chỉnh
python app.py --model seq2seq_lstm --output_dir my_results
```
Biểu đồ dự báo và Attention Heatmap sẽ được lưu tự động tại `evalution_results/`.

### 5.7 Chạy Kiểm Thử Tự Động (Smoke Tests & Unit Tests)
Hệ thống cung cấp bộ kiểm thử khói hoàn chỉnh tại thư mục `tests/` để xác minh mọi thành phần trong gói `src/`:
```bash
# Cách 1: Chạy toàn bộ Smoke Tests qua main.py
python main.py --mode test

# Cách 2: Chạy trực tiếp qua Master Test Runner (Khuyên dùng)
python tests/run_all_tests.py

# Cách 3: Chạy Unit Tests tự động bằng unittest discovery
python -m unittest discover -s tests -p "test_*.py"

# Cách 4: Chạy qua thư mục testing/
python testing/run_smoke_tests.py
```

---

## 6. Tiêu Chí Nghiệm Thu Đạt Được

1. **Zero Leakage:** Thời gian $train < val < test$, `StandardScaler` chỉ đếm đúng số dòng của $train\_df$, độc lập cửa sổ trượt.
2. **Độ chính xác (Accuracy):** Đo lường MAE, MSE, RMSE, $R^2$ trên thang nhiệt độ Celsius thực tế, kèm tỷ lệ % số giờ dự đoán nằm trong ngưỡng $\pm 1.5^\circ\text{C}$.
3. **Hiệu năng & Tài nguyên:** Đo lường chính xác độ trễ suy luận (`ms/batch`, `ms/sample`) và mức tiêu thụ RAM / VRAM.
4. **Tính giải thích được (Explainability):** Trực quan hóa Attention Heatmap ma trận $72 \times 168$.
