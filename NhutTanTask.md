# BÁO CÁO HOÀN THÀNH NHIỆM VỤ
## Chương 3: Phân Tích Dữ Liệu Khám Phá (EDA) & Tiền Xử Lý Chống Rò Rỉ

---

## 1. Thông Tin Chung

| Hạng mục | Nội dung |
| :--- | :--- |
| **Đề tài** | Dự báo nhiệt độ chuỗi thời gian đa bước bằng mô hình Sequence-to-Sequence (Jena Climate Dataset — Topic 5) |
| **Phần việc phụ trách** | Chương 3: Phân tích dữ liệu khám phá (EDA), trực quan hóa dữ liệu và tiền xử lý chống rò rỉ |
| **Người thực hiện** | Lưu Nhựt Tân |
| **Tệp bàn giao chính** | `src/eda.py`, `notebooks/jena_climate_eda.ipynb` |
| **Tệp cập nhật đồng bộ** | `main.py`, `tests/test_eda.py` |
| **Trạng thái** | Hoàn thành (Đã chạy thực nghiệm trên dữ liệu thật và vượt qua toàn bộ kiểm thử tự động) |

---

## 2. Mục Tiêu Nhiệm Vụ

1. Xây dựng và hoàn thiện module trực quan hóa dữ liệu (`src/eda.py`) và sổ tay phân tích (`notebooks/jena_climate_eda.ipynb`) cho bộ dữ liệu thời tiết **Jena Climate (2009–2016)**.
2. Tuân thủ nghiêm ngặt **Quy trình 4 pha chống rò rỉ dữ liệu (Strict Anti-Leakage Protocol)**: chia dữ liệu tuần tự theo thời gian trước khi phân tích, chỉ thực hiện EDA và `fit` `StandardScaler` trên 70% tập Train, cắt cửa sổ trượt độc lập `(168h → 72h)`.
3. Đảm bảo mỗi biểu đồ trực quan hóa đều có phần **giải thích rõ ý nghĩa vật lý/thống kê** và **kết nối trực tiếp tới thiết kế mô hình Học sâu** (Seq2Seq LSTM, Bahdanau Attention LSTM, TimeSeries Transformer).
4. Đồng bộ kết quả đầu ra giữa Notebook và pipeline dòng lệnh (`python main.py --mode eda`), đảm bảo mã nguồn gọn gàng, dễ đọc và vượt qua toàn bộ kiểm thử.

---

## 3. Các Công Việc Đã Thực Hiện

### 3.1. Cải tiến, sửa lỗi và tối ưu hóa `src/eda.py`
- **Khắc phục lỗi kỹ thuật ở phiên bản ban đầu:**
  - Sửa hàm `plot_diurnal_cycle()` từ cố định `np.arange(24)` sang sử dụng chỉ số giờ thực tế `stats.index.to_numpy()`, tránh lỗi khi tập dữ liệu kiểm thử có ít hơn 24 khung giờ; bổ sung đánh dấu trực quan khung giờ nóng nhất và lạnh nhất.
  - Xử lý an toàn đường dẫn lưu ảnh (`os.makedirs`) khi tên tệp không chứa thư mục cha.
  - Nâng cấp `plot_temperature_trend()` bằng đường trung bình trượt 30 ngày (`rolling(720)`) và đường giá trị trung bình toàn tập Train để làm rõ tính chu kỳ mùa.
- **Tái cấu trúc mã nguồn theo hướng ngắn gọn, sạch và dễ bảo trì:**
  - Xây dựng hàm dùng chung `_finalize_plot()` gom toàn bộ định dạng đồ thị (tiêu đề, nhãn trục, lưới, chú giải, lưu ảnh 300 DPI và giải phóng bộ nhớ `plt.close()`).
  - Bổ sung 7 hàm trực quan hóa chuyên sâu mới và hàm tổng hợp thống kê định lượng `summarize_eda_insights()`.
  - Chuẩn hóa đầy đủ Docstring tiếng Việt (mô tả chức năng, Input, Output) cho toàn bộ các hàm theo đặc tả kỹ thuật của dự án.

### 3.2. Xây dựng và thực thi `notebooks/jena_climate_eda.ipynb`
- Thiết kế Notebook gồm **26 cells** mạch lạc theo 5 phần:
  1. **Phần 1:** Nạp cấu hình từ `configs/config.yaml`, kiểm tra dữ liệu gốc và phát hiện/lọc lỗi cảm biến gió `-9999.0`.
  2. **Phần 2 (Pha 1 — Split First):** Gom mẫu `1h` và chia tuần tự 70% Train / 15% Validation / 15% Test theo dòng thời gian.
  3. **Phần 3 (Pha 2 — EDA on Train Only):** Thống kê mô tả và trực quan hóa 6 biểu đồ phân tích cốt lõi chỉ trên 70% tập Train.
  4. **Phần 4 (Pha 3 & Pha 4):** Trực quan hóa hiệu quả `StandardScaler`, minh họa cửa sổ trượt độc lập `(168h → 72h)` và kiểm định `verify_zero_leakage()`.
  5. **Phần 5:** Tổng kết quy trình tiền xử lý và đề xuất hướng Feature Engineering mở rộng.
- Thực thi toàn bộ Notebook (**Run All**) trên dữ liệu thật `data/jena_climate_2009_2016.csv` và lưu sẵn đầy đủ 10 biểu đồ cùng phần giải thích chi tiết ngay dưới mỗi hình.

### 3.3. Đồng bộ Pipeline và Kiểm thử tự động
- Cập nhật `run_eda_pipeline()` trong `src/eda.py` và điểm gọi trong `main.py` (`--mode eda`) để đồng bộ 1-1 với Notebook: xuất đầy đủ **10 biểu đồ PNG** độ phân giải 300 DPI vào thư mục `evalution_results/eda/`.
- Cập nhật bộ kiểm thử `tests/test_eda.py` tương ứng với tập 6 biểu đồ EDA cốt lõi trên tập Train và xác minh toàn bộ unit test, smoke test đều đạt.

---

## 4. Hệ Thống 10 Biểu Đồ Trực Quan Hóa Đã Triển Khai

Toàn bộ biểu đồ được chọn lọc theo nguyên tắc **không trùng lặp nội dung**, mỗi biểu đồ phục vụ một mục tiêu phân tích cụ thể: gồm **3 biểu đồ gốc (đã sửa lỗi & nâng cấp)** và **7 biểu đồ mới bổ sung**.

| STT | Trạng thái | Tên biểu đồ & Hàm (`src/eda.py`) | Loại biểu đồ | Mục đích & Ý nghĩa phân tích |
| :-: | :---: | :--- | :--- | :--- |
| 1 | **Bổ sung mới** | Lọc lỗi cảm biến gió (`plot_wind_anomalies`) | Line (2 khung so sánh) | Phát hiện các giá trị nhiễu cực đoan `-9999.0` (18 điểm ở `wv`, 20 điểm ở `max. wv`) và chứng minh hiệu quả làm sạch trước khi chuẩn hóa. |
| 2 | **Bổ sung mới** | Pha 1: Chia tập theo thời gian (`plot_data_split_timeline`) | Line Timeline | Chứng minh phân chia tuần tự 70% Train – 15% Val – 15% Test ($\max(\text{Train}) < \min(\text{Val}) < \min(\text{Test})$), ngăn chặn rò rỉ tương lai. |
| 3 | *Gốc (nâng cấp)* | Xu hướng nhiệt độ dài hạn (`plot_temperature_trend`) | Line + Rolling Mean 30 ngày | Làm rõ tính chu kỳ mùa lặp lại hàng năm (biên độ từ `-22.65°C` đến `+35.52°C`), cho thấy chuỗi mang tính mùa vụ mạnh. |
| 4 | **Bổ sung mới** | Phân phối nhiệt độ `T (degC)` (`plot_temperature_distribution`) | Boxplot + Histogram/KDE | Kiểm tra hình dạng phân phối (`Mean ≈ Median ≈ 9.1–9.2°C`, `Skew = -0.095`) và nhận diện các điểm cực trị rét đậm mùa đông (`< -15°C`). |
| 5 | *Gốc (sửa & nâng cấp)* | Chu kỳ ngày đêm 24 giờ (`plot_diurnal_cycle`) | Line Mean ± 1 Std | Xác định quy luật ngày-đêm ổn định: thấp nhất lúc `5h` (`5.99°C`), cao nhất lúc `15h` (`12.44°C`); giải thích bài toán dự báo `72h` tương ứng đúng 3 chu kỳ ngày-đêm. |
| 6 | *Gốc (nâng cấp)* | Ma trận tương quan 14 đặc trưng (`plot_correlation_heatmap`) | Heatmap (14×14) | Tổng quan toàn bộ tương quan tuyến tính Pearson, nhận diện đa cộng tuyến cao giữa nhóm nhiệt–ẩm (`T`, `Tpot`, `VPmax`, `Tdew`, `rho`). |
| 7 | **Bổ sung mới** | Quan hệ tuyến tính & phi tuyến (`plot_scatter_relationships`) | Scatter Plot (3 khung) | Phơi bày quan hệ **phi tuyến hàm mũ** giữa `T (degC)` và `VPmax (mbar)` (định luật Clausius–Clapeyron), lý giải sự cần thiết của mô hình Học sâu phi tuyến. |
| 8 | **Bổ sung mới** | Hàm tự tương quan ACF 1h–168h (`plot_autocorrelation`) | Line ACF + Đánh dấu 24h | Chứng minh các đỉnh tự tương quan lặp lại đúng mỗi 24 giờ đến tận mốc `168h` (`ACF = 0.77`), tạo cơ sở khoa học cho cửa sổ `168h → 72h` và cơ chế Attention/Transformer. |
| 9 | **Bổ sung mới** | Pha 3: So sánh trước & sau chuẩn hóa (`plot_scaling_comparison`) | Boxplot (2 khung) | Chứng minh `StandardScaler` (fit duy nhất trên Train) đưa các biến chênh lệch biên độ lớn (`rho ~1200`, `p ~990` so với `wv ~2`) về cùng thang `Mean ≈ 0, Std ≈ 1`. |
| 10 | **Bổ sung mới** | Pha 4: Cửa sổ trượt độc lập (`plot_sliding_window_example`) | Line + Vùng tô màu | Minh họa trực quan 1 cặp mẫu `(168h quá khứ X → 72h tương lai Y)` cắt độc lập nội bộ tập Train, không vắt ngang ranh giới giữa các phân tập. |

---

## 5. Kết Quả Phân Tích Định Lượng & Kiểm Định Chống Rò Rỉ

Thực nghiệm trên dữ liệu thực tế `data/jena_climate_2009_2016.csv` ghi nhận các kết quả sau:

- **Quy mô dữ liệu:** `420.551` dòng gốc (tần suất 10 phút) $\rightarrow$ sau gom mẫu `1h` thu được **`70.129` giờ**.
- **Phân chia tuần tự (Pha 1):**
  - **Train (70%):** `49.090` giờ (`2009-01-01 00:00` → `2014-08-08 09:00`) $\rightarrow$ `48.851` cửa sổ trượt.
  - **Validation (15%):** `10.519` giờ (`2014-08-08 10:00` → `2015-10-20 16:00`) $\rightarrow$ `10.280` cửa sổ trượt.
  - **Test (15%):** `10.520` giờ (`2015-10-20 17:00` → `2017-01-01 00:00`) $\rightarrow$ `10.281` cửa sổ trượt.
- **Thống kê biến mục tiêu `T (degC)` trên tập Train (Pha 2):**
  - `Mean = 9.11°C` | `Median = 9.20°C` | `Std = 8.65°C` | `Min = -22.65°C` | `Max = 35.52°C` | `Skewness = -0.095`.
  - Tương quan thuận cao nhất: `Tpot (K)` (`+0.997`), `VPmax (mbar)` (`+0.950`), `Tdew (degC)` (`+0.902`).
  - Tương quan nghịch cao nhất: `rho (g/m**3)` (`-0.965`), `rh (%)` (`-0.563`).
  - Tự tương quan ACF theo độ trễ: `24h = 0.9222` | `48h = 0.8672` | `72h = 0.8309` | `168h = 0.7721`.
- **Kiểm định chống rò rỉ (Pha 3 & Pha 4):**
  - `dm.verify_zero_leakage()` đạt **PASSED** (`scaler.n_samples_seen_ = 49.090 == len(train_df)`).
  - Hợp đồng kích thước Tensor từ DataLoader: `X = (64, 168, 14)`, `Y = (64, 72, 1)`.

---

## 6. Tổng Kết Tiền Xử Lý & Đề Xuất Feature Engineering Mở Rộng

- **5 bước tiền xử lý và biến đổi đặc trưng đã triển khai trong `src/data_module.py`:**
  1. Lọc giá trị cảm biến gió lỗi (`-9999.0 → NaN`).
  2. Gom mẫu thời gian `resample('1h').mean()` nhằm khử nhiễu tần số cao và giảm độ dài chuỗi 6 lần.
  3. Điền khuyết nội bộ độc lập trong từng tập (`ffill().bfill()`).
  4. Chuẩn hóa Z-score (`StandardScaler`) chỉ `fit` trên tập Train.
  5. Tạo cặp mẫu chuỗi thời gian độc lập bằng `SlidingWindowDataset` (`168h → 72h`).
- **Đề xuất hướng Feature Engineering mở rộng (Future Work):**
  Để đảm bảo tuân thủ hợp đồng `input_dim = 14` dùng chung cho 3 kiến trúc mô hình trong toàn dự án, các đặc trưng gốc được giữ nguyên. Trong giai đoạn cải tiến tiếp theo, có thể thử nghiệm bổ sung:
  - Chuyển đổi biến góc hướng gió `wd (deg)` thành vector gió 2D ($W_x = wv\cos(\text{wd}_{\text{rad}}),\; W_y = wv\sin(\text{wd}_{\text{rad}})$) để loại bỏ điểm gián đoạn `0°/360°`.
  - Mã hóa chu kỳ thời gian bằng hàm lượng giác (`Day sin/cos` theo chu kỳ 24h và `Year sin/cos` theo chu kỳ 365.2425 ngày).

---

## 7. Kết Quả Kiểm Thử & Hướng Dẫn Chạy Lại

Toàn bộ các kiểm thử đều thực thi trên môi trường ảo `.venv` (Python 3.11.18) và đạt kết quả **PASSED 100%**:

| Lệnh thực thi | Kết quả | Ghi chú |
| :--- | :---: | :--- |
| `python main.py --mode eda` | **PASSED** | Xuất đầy đủ 10 ảnh PNG vào `evalution_results/eda/` |
| `python src/eda.py` | **PASSED** | Vượt qua smoke test nội bộ và hàm `summarize_eda_insights()` |
| `python tests/test_eda.py` | **PASSED** | Vượt qua smoke test EDA độc lập |
| `python -m unittest tests/test_eda.py -v` | **OK** | 1/1 unit test thành công |
| `jupyter nbconvert --execute notebooks/jena_climate_eda.ipynb` | **PASSED** | Chạy trọn vẹn 26 cells, nhúng đủ 10 biểu đồ trên dữ liệu thật |

```bash
# Kích hoạt môi trường ảo và tái lập kết quả
source .venv/bin/activate
python main.py --mode eda
python -m unittest tests/test_eda.py -v
```
