# THƯ MỤC KIỂM THỬ TỰ ĐỘNG (TESTING SUITE) - TOPIC 5

Thư mục này chứa toàn bộ các kịch bản kiểm thử khói (**Smoke Tests**) và kiểm thử đơn vị (**Unit Tests**) cho tất cả các mô-đun trong gói `src/` của đồ án:
**Sequence-to-Sequence Weather Forecasting (Topic 5)**.

---

## 1. Cấu Trúc Thư Mục Kiểm Thử

```text
tests/
├── __init__.py             # Đóng gói tests thành Python package
├── test_data_module.py     # Kiểm thử DataModule & Quy tắc 4 pha chống rò rỉ dữ liệu (Zero Leakage)
├── test_model_builder.py   # Kiểm thử 3 kiến trúc mô hình (LSTM, Attention, Transformer) & Factory
├── test_trainer.py         # Kiểm thử ModelTrainer, Scheduled Teacher Forcing, Early Stopping & Checkpoints
├── test_evaluation.py      # Kiểm thử ModelEvaluator, MAE/RMSE/R2, Benchmark độ trễ ms & xuất JSON/CSV
├── test_logger.py          # Kiểm thử TensorBoardLogger (Scalar metrics, Heatmaps, Figures)
├── test_eda.py             # Kiểm thử EDA pipeline (Chỉ vẽ trên tập Train - Zero Cognitive Leakage)
├── run_all_tests.py        # Master Test Runner điều phối chạy toàn bộ test suite
└── README.md               # Tài liệu hướng dẫn sử dụng thư mục tests
```

---

## 2. Hướng Dẫn Chạy Kiểm Thử

### 2.1 Chạy toàn bộ Smoke Tests (Khuyên dùng - Nhanh dưới 10 giây)
Lệnh này chạy tuần tự tất cả 6 bài kiểm thử với dữ liệu giả lập (Synthetic Mock Data) để kiểm chứng tính toàn vẹn hệ thống:
```bash
python tests/run_all_tests.py
# hoặc chỉ định rõ chế độ:
python tests/run_all_tests.py --mode smoke
```

### 2.2 Chạy toàn bộ Unit Tests (Python Unittest Discovery)
```bash
python tests/run_all_tests.py --mode unit
# hoặc dùng lệnh unittest chuẩn của Python:
python -m unittest discover -s tests -p "test_*.py" -v
```

### 2.3 Chạy bằng Pytest (Nếu đã cài đặt `pytest`)
```bash
pytest tests/ -v
```

### 2.4 Chạy riêng lẻ từng Mô-đun Kiểm Thử
```bash
# 1. Kiểm thử DataModule & Chống rò rỉ dữ liệu
python tests/test_data_module.py

# 2. Kiểm thử 3 Kiến trúc Mô hình Học sâu
python tests/test_model_builder.py

# 3. Kiểm thử Quy trình Huấn luyện & Checkpoint
python tests/test_trainer.py

# 4. Kiểm thử Kịch bản Đánh giá & Benchmark độ trễ
python tests/test_evaluation.py

# 5. Kiểm thử Bộ ghi nhận trực quan TensorBoard
python tests/test_logger.py

# 6. Kiểm thử EDA Pipeline
python tests/test_eda.py
```

---

## 3. Đầu Ra Mẫu Khi Chạy Master Test Runner

```text
================================================================================
   BỘ KIỂM THỬ KHÓI TỔNG THỂ (MASTER SMOKE TESTS) - TOPIC 5
   Dự án: Sequence-to-Sequence Weather Forecasting (Jena Climate)
================================================================================

>>> [1/6] THỰC THI KIỂM THỬ: DataModule (Anti-Leakage & Sliding Windows)
... [SUCCESS] TOÀN BỘ BÀI KIỂM THỬ DATA MODULE ĐÃ VƯỢT QUA XUẤT SẮC!

>>> [2/6] THỰC THI KIỂM THỬ: ModelBuilder (3 Architectures & Factory)
... [SUCCESS] TOÀN BỘ BÀI KIỂM THỬ MÔ HÌNH ĐÃ VƯỢT QUA XUẤT SẮC!

>>> [3/6] THỰC THI KIỂM THỬ: ModelTrainer (Early Stopping & Checkpoints)
... [SUCCESS] TOÀN BỘ BÀI KIỂM THỬ MODEL TRAINER ĐÃ VƯỢT QUA XUẤT SẮC!

>>> [4/6] THỰC THI KIỂM THỬ: ModelEvaluator (Metrics & Latency Benchmark)
... [SUCCESS] TOÀN BỘ BÀI KIỂM THỬ ĐÁNH GIÁ ĐÃ VƯỢT QUA XUẤT SẮC!

>>> [5/6] THỰC THI KIỂM THỬ: TensorBoardLogger (Scalars & Heatmaps)
... [SUCCESS] TOÀN BỘ BÀI KIỂM THỬ LOGGER ĐÃ VƯỢT QUA XUẤT SẮC!

>>> [6/6] THỰC THI KIỂM THỬ: EDA Pipeline (Train-only Analysis)
... [SUCCESS] TOÀN BỘ BÀI KIỂM THỬ EDA ĐÃ VƯỢT QUA XUẤT SẮC!

================================================================================
                     BẢNG TỔNG HỢP KẾT QUẢ KIỂM THỬ
================================================================================
Mô-đun kiểm thử                                    | Trạng thái   | Thời gian (ms) 
--------------------------------------------------------------------------------
DataModule (Anti-Leakage & Sliding Windows)        | [OK] PASSED  |       125.40 ms
ModelBuilder (3 Architectures & Factory)           | [OK] PASSED  |       340.12 ms
ModelTrainer (Early Stopping & Checkpoints)        | [OK] PASSED  |       450.85 ms
ModelEvaluator (Metrics & Latency Benchmark)       | [OK] PASSED  |       180.20 ms
TensorBoardLogger (Scalars & Heatmaps)             | [OK] PASSED  |        95.10 ms
EDA Pipeline (Train-only Analysis)                 | [OK] PASSED  |       210.50 ms
================================================================================
Tổng thời gian thực thi: 1.40 giây
[FINAL RESULT] TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ VƯỢT QUA XUẤT SẮC! (100% PASSED)
```
