# Tóm Tắt Nhiệm Vụ — Chương 3: Phân Tích Dữ Liệu Khám Phá (EDA) & Tiền Xử Lý Chống Rò Rỉ

- **Người thực hiện:** Lưu Nhựt Tân
- **Đề tài:** Sequence-to-Sequence Weather Forecasting (Jena Climate Dataset — Topic 5)
- **Sản phẩm bàn giao:** `src/eda.py` và `notebooks/jena_climate_eda.ipynb`
- **Trạng thái:** Hoàn thành & đã kiểm thử PASS 100% trên dữ liệu thật (`data/jena_climate_2009_2016.csv`)

---

## 1. Các Hạng Mục Đã Thực Hiện

### 1.1 Sửa lỗi, tối ưu & nâng cấp `src/eda.py`
- **Sửa lỗi tiềm ẩn ở bản cũ:**
  - Sửa `plot_diurnal_cycle` (trước đây cố định `np.arange(24)` gây lỗi nếu dữ liệu ít hơn 24 giờ) sang dùng `stats.index.to_numpy()` và tự động đánh dấu giờ nóng nhất / lạnh nhất.
  - Sửa `os.makedirs(os.path.dirname(save_path))` để không lỗi khi `save_path` không có thư mục cha.
  - Nâng cấp `plot_temperature_trend` thêm đường trung bình trượt 30 ngày (`720h`) và đường `Mean`.
- **Tối ưu cấu trúc code dễ học & dễ thuyết trình:**
  - Gom toàn bộ phần trang trí lặp lại (`set_title`, `set_xlabel`, `set_ylabel`, `grid`, `legend`, `tight_layout`, `savefig`/`show`) vào hàm dùng chung `_finalize_plot()`.
  - Mỗi hàm vẽ chỉ còn ~8–15 dòng logic rõ ràng, chia thành 4 khối mạch lạc và đầy đủ Docstring tiếng Việt theo chuẩn dự án.
- **Giữ tương thích 100% với toàn nhóm:**
  - `run_eda_pipeline(train_df, output_dir, include_extended=False)` mặc định vẫn xuất đúng 3 ảnh cốt lõi để `main.py --mode eda` và `tests/test_eda.py` vượt qua kiểm thử; khi bật `include_extended=True` sẽ xuất đủ 10 ảnh EDA.

### 1.2 Hoàn thiện `notebooks/jena_climate_eda.ipynb`
- Xây dựng **33 cells** gọn gàng (mỗi ô code chỉ 1–10 dòng), chạy **Run All** thành công trên dữ liệu thật và lưu sẵn **14 biểu đồ độ phân giải cao** bên trong notebook.
- Ngay dưới mỗi biểu đồ đều có mục **Giải thích biểu đồ** (Quan sát → Ý nghĩa vật lý/khí tượng → Hàm ý thiết kế mô hình Học sâu).
- Bổ sung **Phần 5: Tổng kết Tiền xử lý & Ghi chú Feature Engineering** phục vụ trực tiếp phần vấn đáp.

### 1.3 Thiết lập môi trường, dữ liệu & kiểm thử
- Cài **Python 3.11.18** và tạo môi trường ảo `.venv` đầy đủ thư viện (`torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`, `ipykernel`, `nbconvert`, `pytest`, ...).
- Tải và giải nén bộ dữ liệu thật `data/jena_climate_2009_2016.csv` (420.551 dòng × 15 cột).
- Chạy kiểm thử toàn bộ:
  - `.venv/bin/python src/eda.py` → **PASSED**
  - `.venv/bin/python tests/test_eda.py` → **PASSED**
  - `.venv/bin/python -m unittest tests/test_eda.py -v` → **OK**
  - `.venv/bin/python main.py --mode eda` → Xuất 3 ảnh tại `evalution_results/eda/`

---

## 2. Chi Tiết Biểu Đồ: Cũ (3 Biểu Đồ) vs Mới Bổ Sung (11 Biểu Đồ)

### 2.1 3 biểu đồ gốc có sẵn (đã sửa lỗi & nâng cấp)
1. **Xu hướng nhiệt độ dài hạn (`plot_temperature_trend` — Line):** Thêm đường trung bình trượt 30 ngày (`720h`) và đường `Mean`.
2. **Chu kỳ ngày đêm 24 giờ (`plot_diurnal_cycle` — Line + ±1 Std):** Sửa lỗi crash khi dữ liệu `< 24h`, tự động đánh dấu đỉnh `15h (12.44°C)` và đáy `5h (5.99°C)`.
3. **Ma trận tương quan 14 biến (`plot_correlation_heatmap` — Heatmap):** Chuẩn hóa thang màu `[-1, 1]` đối xứng qua `0`.

### 2.2 11 biểu đồ MỚI đã bổ sung thêm
1. **`plot_wind_anomalies` (Line 2 khung):** So sánh tốc độ gió `wv (m/s)` trước và sau khi lọc lỗi cảm biến `-9999.0`.
2. **`plot_data_split_timeline` (Line timeline):** Trực quan hóa Pha 1 — chia tuần tự 70% Train / 15% Val / 15% Test chứng minh Zero Time Leakage.
3. **`plot_temperature_distribution` (Histogram + KDE + Boxplot):** Phân phối biến mục tiêu `T (degC)` kèm Mean, Median, Std, Skew.
4. **`plot_temperature_regime_pie` (Biểu đồ tròn / Donut):** Tỷ lệ % số giờ theo 4 ngưỡng nhiệt độ (`<0°C`, `0–10°C`, `10–20°C`, `>20°C`).
5. **`plot_seasonal_monthly_cycle` (Boxplot + Line):** Chu kỳ nhiệt độ theo 12 tháng trong năm.
6. **`plot_target_correlations` (Bar ngang):** Xếp hạng hệ số tương quan Pearson của 13 biến khí tượng với `T (degC)`.
7. **`plot_scatter_relationships` (Biểu đồ phân tán / Scatter 3 khung):** Quan hệ giữa `T (degC)` với `VPmax` (phi tuyến hàm mũ), `rho` (tuyến tính nghịch), `rh` (phân tán rộng).
8. **`plot_wind_rose` (Biểu đồ hoa gió / Polar):** Tần suất (%) và tốc độ gió trung bình theo 16 hướng gió `wd (deg)`.
9. **`plot_autocorrelation` (Line ACF 1h–168h):** Tự tương quan nhiệt độ với các đỉnh lặp lại mỗi 24h (`24h..168h`), chứng minh cơ sở chọn `168h -> 72h` và vai trò Attention/Transformer.
10. **`plot_scaling_comparison` (Boxplot 2 khung):** Pha 3 — so sánh thang đo 6 biến tiêu biểu trước và sau `StandardScaler` (chỉ `fit` trên Train).
11. **`plot_sliding_window_example` (Line + Shaded Regions):** Pha 4 — minh họa trực quan 1 mẫu cửa sổ trượt `168h quá khứ (X) -> 72h tương lai (Y)` nội bộ tập Train.

### 2.3 Bảng tra cứu nhanh toàn bộ 14 biểu đồ trong `notebooks/jena_climate_eda.ipynb`

| STT | Trạng thái | Biểu đồ | Loại biểu đồ | Hàm trong `src/eda.py` | Ý nghĩa chính để thuyết trình |
| :-: | :---: | :--- | :--- | :--- | :--- |
| 1 | **Mới** | Lọc cảm biến gió lỗi `-9999.0` | Line (2 khung) | `plot_wind_anomalies` | Phát hiện 18 lỗi ở `wv`, 20 lỗi ở `max. wv`; lọc trước để không làm méo Scaler & gradient. |
| 2 | **Mới** | Pha 1: Chia Train / Val / Test | Line (timeline) | `plot_data_split_timeline` | Chứng minh chia tuần tự 70/15/15 theo thời gian ($\max(\text{Train}) < \min(\text{Val}) < \min(\text{Test})$), chống Time Leakage. |
| 3 | *Gốc (nâng cấp)* | Xu hướng nhiệt độ dài hạn | Line + Rolling 30d | `plot_temperature_trend` | Sóng hình sin lặp lại chu kỳ 1 năm (`-22.65°C` đến `+35.52°C`, Mean `9.11°C`) → cần đầu vào 168h nắm nền nhiệt mùa. |
| 4 | **Mới** | Phân phối `T (degC)` | Histogram + KDE + Boxplot | `plot_temperature_distribution` | Phân phối rộng (`Std = 8.65°C`), đối xứng (`Skew = -0.095`), có đuôi rét đậm mùa đông. |
| 5 | **Mới** | Tỷ lệ 4 ngưỡng nhiệt độ | Donut / Pie Chart | `plot_temperature_regime_pie` | Trực quan hóa tỷ lệ % số giờ Rét đậm (`<0°C`), Lạnh/Mát (`0–10°C`), Ôn hòa (`10–20°C`), Nóng (`>20°C`). |
| 6 | **Mới** | Chu kỳ theo 12 tháng | Boxplot + Line Mean | `plot_seasonal_monthly_cycle` | Lạnh nhất tháng 1 (`-0.78°C`), nóng nhất tháng 7 (`18.97°C`); mùa đông/xuân biến động mạnh. |
| 7 | *Gốc (sửa & nâng cấp)* | Chu kỳ ngày đêm 24 giờ | Line + Dải `±1 Std` | `plot_diurnal_cycle` | Đáy lúc 5h sáng (`5.99°C`), đỉnh lúc 15h chiều (`12.44°C`) → dự báo 72h là đúng 3 chu kỳ ngày-đêm. |
| 8 | *Gốc (nâng cấp)* | Ma trận tương quan 14 biến | Heatmap (14×14) | `plot_correlation_heatmap` | Thấy rõ đa cộng tuyến cao giữa nhóm nhiệt–ẩm (`T`, `Tpot`, `VPmax`, `Tdew`, `sh`, `H2OC`). |
| 9 | **Mới** | Xếp hạng tương quan với `T` | Horizontal Bar Chart | `plot_target_correlations` | Thuận mạnh nhất: `Tpot (+0.997)`, `VPmax (+0.950)`, `Tdew (+0.902)`; Nghịch mạnh nhất: `rho (-0.965)`, `rh (-0.563)`. |
| 10 | **Mới** | Quan hệ tuyến tính & phi tuyến | Scatter Plot (3 khung) | `plot_scatter_relationships` | `T` vs `VPmax` cong hàm mũ **phi tuyến** (Clausius–Clapeyron) → lý giải vì sao cần Deep Learning thay vì Linear Regression. |
| 11 | **Mới** | Hoa gió hướng & tốc độ gió | Polar Wind Rose | `plot_wind_rose` | Hướng gió chủ đạo Tây Nam & Đông Bắc; thể hiện đúng bản chất biến góc tuần hoàn `wd (0° ≡ 360°)`. |
| 12 | **Mới** | Hàm tự tương quan ACF (1h–168h) | Line + Marker 24h | `plot_autocorrelation` | Đỉnh lặp lại đúng mỗi 24h (`24h: 0.92`, `72h: 0.83`, `168h: 0.77`) → chứng minh chọn `168h -> 72h` và vai trò Attention/Transformer. |
| 13 | **Mới** | Pha 3: Trước & Sau StandardScaler | Boxplot (2 khung) | `plot_scaling_comparison` | Đưa 14 biến từ thang đo chênh lệch hàng trăm lần về cùng chuẩn `Mean ≈ 0, Std ≈ 1` (chỉ `fit` trên Train). |
| 14 | **Mới** | Pha 4: Cửa sổ trượt độc lập | Line + Shaded Regions | `plot_sliding_window_example` | Minh họa trực quan 1 mẫu `168h quá khứ (X) -> 72h tương lai (Y)` cắt nội bộ trong tập Train. |

---

## 3. Số Liệu Thực Tế Cần Nhớ Khi Vấn Đáp

- **Dữ liệu gốc:** `420.551` dòng (10 phút/lần) → sau Resample `1h`: **`70.129` giờ**.
- **Phân chia 70 / 15 / 15 (Zero Time Leakage):**
  - **Train (70%):** `49.090` giờ (`2009-01-01 00:00` → `2014-08-08 09:00`) → `48.851` cửa sổ trượt.
  - **Val (15%):** `10.519` giờ (`2014-08-08 10:00` → `2015-10-20 16:00`) → `10.280` cửa sổ trượt.
  - **Test (15%):** `10.520` giờ (`2015-10-20 17:00` → `2017-01-01 00:00`) → `10.281` cửa sổ trượt.
- **Kiểm định Zero Leakage:** `scaler.n_samples_seen_ == 49.090 == len(train_df)` (**PASSED**).
- **Hợp đồng Tensor DataLoader:** `X = (64, 168, 14)`, `Y = (64, 72, 1)`.

---

## 4. Tóm Tắt Tiền Xử Lý & Feature Engineering (Trả Lời Vấn Đáp)

1. **5 bước Tiền xử lý / Biến đổi đặc trưng đã làm trong `src/data_module.py`:**
   - Lọc nhiễu cảm biến gió (`-9999.0 -> NaN`).
   - Gom mẫu thời gian `resample('1h').mean()`.
   - Điền khuyết nội bộ từng tập (`ffill().bfill()`).
   - Chuẩn hóa `StandardScaler` chỉ `fit` trên Train.
   - Đóng gói cửa sổ trượt độc lập `(168h × 14) -> (72h × 1)`.
2. **Hướng Feature Engineering mở rộng (chưa thêm vào model để giữ đúng hợp đồng `input_dim = 14` của cả nhóm):**
   - **Vector gió 2D:** $W_x = wv \cdot \cos(\text{wd}_{\text{rad}})$, $W_y = wv \cdot \sin(\text{wd}_{\text{rad}})$ xử lý tính gián đoạn `0°/360°` của `wd (deg)`.
   - **Mã hóa chu kỳ thời gian:** `Day sin/cos` (24h) và `Year sin/cos` (365.2425 ngày).

---

## 5. Câu Lệnh Chạy Lại Nhanh

```bash
# Kích hoạt môi trường ảo Python 3.11
source .venv/bin/activate

# Chạy smoke test của module EDA
python src/eda.py

# Chạy unit test EDA
python -m unittest tests/test_eda.py -v

# Xuất 3 biểu đồ EDA cốt lõi vào evalution_results/eda/
python main.py --mode eda
```
