# HƯỚNG DẪN HỌC & THUYẾT TRÌNH CODE `src/eda.py` THEO TỪNG HÀM (TỪ TRÊN XUỐNG DƯỚI)
## Chương 3: Phân Tích Dữ Liệu Khám Phá (EDA) & Tiền Xử Lý Chống Rò Rỉ — Jena Climate (Topic 5)

> **Cách học nhanh nhất:** Đọc **Phần Mở Đầu** để lấy lời dẫn khái quát trước khi chiếu code, sau đó mở song song `src/eda.py` và học lần lượt **từ Hàm 1 đến Khối 14 đúng theo thứ tự dòng 1 → dòng 516**.

---

## PHẦN MỞ ĐẦU: KHÁI QUÁT VỀ EDA TRƯỚC KHI VÀO CODE

### 1. EDA là gì?
- **EDA (Exploratory Data Analysis — Phân tích dữ liệu khám phá)** là bước kiểm tra và khám phá dữ liệu bằng thống kê mô tả và biểu đồ trực quan **trước khi đưa vào huấn luyện mô hình**.
- Mục đích cốt lõi của EDA là trả lời 3 câu hỏi:
  1. *Dữ liệu có sạch không?* (Có giá trị lỗi, ngoại lai, hay dữ liệu khuyết không?)
  2. *Dữ liệu có quy luật gì?* (Phân phối ra sao, có xu hướng mùa hay chu kỳ lặp lại không?)
  3. *Các biến liên hệ với nhau thế nào và ảnh hưởng gì đến việc chọn mô hình Học sâu?*

### 2. Tại sao EDA đặc biệt quan trọng trong bài toán Jena Climate?
Với bộ dữ liệu chuỗi thời gian khí tượng **Jena Climate** (`420.551` dòng đo 10 phút/lần, `14` đặc trưng), EDA giúp nhóm giải quyết **4 vấn đề quyết định**:
1. **Phát hiện lỗi phần cứng cảm biến:** Nhận diện các giá trị lỗi cực đoan `-9999.0` ở cảm biến gió để lọc bỏ trước khi chuẩn hóa.
2. **Chứng minh cơ sở chọn kiến trúc & cửa sổ thời gian (`168h → 72h`):** Nhờ phân tích chu kỳ 24h, tính mùa vụ và tự tương quan **ACF**, nhóm chứng minh sau 7 ngày (`168h`) nhiệt độ vẫn có tương quan cao (`0.77`) — vừa giải thích độ dài đầu vào, vừa lý giải vì sao **Attention** và **Transformer** vượt trội LSTM cơ sở.
3. **Chứng minh tính phi tuyến của dữ liệu:** Biểu đồ tương quan & Scatter cho thấy quan hệ **phi tuyến mạnh** (ví dụ `T` vs `VPmax` theo đường cong hàm mũ) → khẳng định cần dùng **Deep Learning** thay vì hồi quy tuyến tính.
4. **Đảm bảo nguyên tắc chống rò rỉ dữ liệu 4 pha (Strict Anti-Leakage):** Trong bài toán chuỗi thời gian, chia dữ liệu sai hoặc EDA trên toàn bộ tập sẽ gây *Time Leakage* và *Cognitive Leakage*. Vì vậy nhóm chia tuần tự **70% Train – 15% Val – 15% Test** trước (Pha 1), sau đó **chỉ làm EDA và `fit` `StandardScaler` trên 70% tập Train**.

### 3. Luồng tổng thể của Chương 3 (nhớ trong 1 dòng)
`Dữ liệu gốc (420.551 dòng, 10 phút)` → `Lọc lỗi -9999.0 & Resample 1h (70.129h)` → **`Pha 1:`** `Chia tuần tự 70% Train / 15% Val / 15% Test` → **`Pha 2:`** `EDA chỉ trên 70% Train` → **`Pha 3:`** `StandardScaler chỉ fit trên Train` → **`Pha 4:`** `Cắt Sliding Window độc lập (168h → 72h)` → `Sẵn sàng huấn luyện 3 mô hình`.

### 4. Lời dẫn mở đầu mẫu (nói trong ~30–40 giây trước khi mở `src/eda.py`)
> *"Chào cô và các bạn, em là Nhựt Tân, mở đầu phần trình bày của nhóm với Chương 3: Phân tích dữ liệu khám phá (EDA) và tiền xử lý chống rò rỉ dữ liệu cho bộ Jena Climate.*
>
> *Trước khi xây dựng mô hình dự báo nhiệt độ 72 giờ tới từ 168 giờ quá khứ, EDA đóng vai trò cực kỳ quan trọng giúp nhóm kiểm tra lỗi cảm biến, nắm rõ quy luật chu kỳ ngày–đêm, chu kỳ mùa, cũng như kiểm chứng tính phi tuyến và tự tương quan của 14 đặc trưng khí tượng. Đặc biệt, để tránh hiện tượng rò rỉ dữ liệu trong chuỗi thời gian, nhóm áp dụng quy trình 4 pha nghiêm ngặt: chia tuần tự 70-15-15 trước rồi chỉ thực hiện toàn bộ EDA và chuẩn hóa trên 70% tập Train.*
>
> *Sau đây em xin trình bày chi tiết cấu trúc mã nguồn trong file `src/eda.py` và ý nghĩa của 10 biểu đồ mà file này sinh ra."*

---

## MỤC LỤC THEO ĐÚNG THỨ TỰ TRONG `src/eda.py`

> **Tóm gọn cấu trúc:** File có **13 hàm `def`** (gồm đúng **10 hàm vẽ 10 biểu đồ** + **3 hàm hỗ trợ/điều phối** không sinh thêm hình) và **1 khối Smoke Test** (`if __name__ == "__main__":`).

| TT | Tên hàm / Khối | Dòng | Vai trò nhanh | Hình ảnh sinh ra (`evalution_results/eda/`) |
| :-: | :--- | :---: | :--- | :--- |
| 0 | Import thư viện | `1–6` | Nạp các thư viện xử lý & vẽ đồ thị | — |
| 1 | `_finalize_plot()` | `9–47` | Định dạng chung & lưu/hiển thị biểu đồ | *(Hàm phụ trợ dùng chung)* |
| 2 | `plot_wind_anomalies()` | `54–88` | So sánh gió trước & sau lọc `-9999.0` | `eda_wind_anomalies.png` |
| 3 | `plot_data_split_timeline()` | `91–123` | Minh họa Pha 1 chia 70/15/15 theo thời gian | `eda_data_split_timeline.png` |
| 4 | `plot_temperature_trend()` | `130–157` | Vẽ xu hướng nhiệt độ 5 năm & chu kỳ mùa | `eda_temperature_trend.png` |
| 5 | `plot_temperature_distribution()` | `160–184` | Vẽ phân phối, độ lệch & ngoại lai của `T` | `eda_temperature_distribution.png` |
| 6 | `plot_diurnal_cycle()` | `187–215` | Vẽ chu kỳ ngày đêm 24 giờ | `eda_diurnal_cycle.png` |
| 7 | `plot_correlation_heatmap()` | `218–233` | Vẽ ma trận tương quan 14 biến | `eda_correlation_heatmap.png` |
| 8 | `plot_scatter_relationships()` | `236–268` | Vẽ tương quan phi tuyến/tuyến tính bằng Scatter | `eda_scatter_relationships.png` |
| 9 | `plot_autocorrelation()` | `271–305` | Vẽ tự tương quan ACF 1h–168h | `eda_autocorrelation.png` |
| 10 | `plot_scaling_comparison()` | `312–346` | Minh họa Pha 3 trước & sau `StandardScaler` | `eda_scaling_comparison.png` |
| 11 | `plot_sliding_window_example()` | `349–387` | Minh họa Pha 4 cửa sổ trượt `168h → 72h` | `eda_sliding_window_example.png` |
| 12 | `summarize_eda_insights()` | `394–424` | Tính nhanh dict số liệu định lượng tập Train | *(Không vẽ hình)* |
| 13 | `run_eda_pipeline()` | `427–493` | Điều phối gọi các hàm vẽ xuất file PNG tự động | *(Xuất 6 hoặc 10 ảnh PNG)* |
| 14 | `if __name__ == "__main__":` | `496–516` | Chạy Smoke Test kiểm thử nhanh file `eda.py` | *(Tự xóa ảnh tạm)* |

---

## 0. IMPORT THƯ VIỆN (Dòng 1–6)

```python
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
```

### Vai trò
Khai báo toàn bộ thư viện nền tảng để thao tác tệp/thư mục, xử lý bảng chuỗi thời gian và vẽ biểu đồ thống kê.

### Giải thích từng dòng
- `os`: Tạo thư mục (`makedirs`), kiểm tra file tồn tại (`exists`), ghép đường dẫn (`path.join`).
- `typing` (`Dict`, `List`, `Optional`): Khai báo kiểu dữ liệu đầu vào/đầu ra (Type Hints) giúp code chuẩn, dễ đọc.
- `matplotlib.pyplot as plt`: Thư viện vẽ biểu đồ nền tảng (Line, Scatter, Subplots, lưu ảnh PNG).
- `numpy as np`: Xử lý mảng số học (`np.nan`, `np.arange`, `np.array`).
- `pandas as pd`: Xử lý bảng chuỗi thời gian (`DataFrame`, `rolling`, `groupby`, `corr`, `autocorr`).
- `seaborn as sns`: Thư viện vẽ đồ thị thống kê đẹp hơn trên nền Matplotlib (`boxplot`, `histplot`, `heatmap`).

---

## HÀM 1: `_finalize_plot()` (Dòng 9–47)

### Code chính
```python
def _finalize_plot(fig, ax=None, title="", xlabel="", ylabel="", save_path=None, show_legend=False):
    if ax is not None:
        if title:  ax.set_title(title, fontsize=12, fontweight="bold")
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)
        if show_legend: ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
```

### Vai trò của hàm
> Là **hàm phụ trợ dùng chung (helper)** để hoàn thiện định dạng (tiêu đề, nhãn trục, lưới, chú giải) và quyết định **lưu biểu đồ ra file PNG 300 DPI** (khi chạy CLI) hay **hiển thị trực tiếp** (khi chạy Notebook), giúp 10 hàm vẽ bên dưới không bị lặp code.

### Giải thích từng dòng
- **Tên có dấu `_` ở đầu (`_finalize_plot`):** Quy ước Python cho hàm nội bộ (*private helper*).
- `if ax is not None:` Nếu biểu đồ có 1 trục chính `ax` thì tự gắn tiêu đề (`set_title`), nhãn trục X/Y, bật lưới chấm mờ (`grid`) và khung chú giải (`legend(loc="best")` tự tìm góc trống không che dữ liệu).
- `fig.tight_layout()`: Tự căn chỉnh khoảng cách giữa tiêu đề, nhãn trục và các khung con để chữ không bị đè lên nhau.
- `if save_path:` Chế độ **lưu ảnh ra ổ cứng** (khi chạy `main.py --mode eda`):
  - `parent = os.path.dirname(save_path)` + `os.makedirs(parent, exist_ok=True)`: Lấy tên thư mục cha và tạo tự động nếu chưa tồn tại (`if parent:` tránh lỗi khi `save_path` chỉ có tên file).
  - `fig.savefig(save_path, dpi=300, bbox_inches="tight")`: Xuất ảnh sắc nét **300 DPI** và cắt gọn viền thừa.
  - `plt.close(fig)`: Đóng đối tượng hình để **giải phóng RAM** ngay sau khi lưu (rất hay bị cô hỏi!).
- `else: plt.show()`: Chế độ **hiển thị trực tiếp** trong Jupyter Notebook (`save_path=None`).

---

## HÀM 2: `plot_wind_anomalies()` (Dòng 54–88)
**Ảnh sinh ra:** `evalution_results/eda/eda_wind_anomalies.png`

### Code chính
```python
col = "wv (m/s)"
raw_s = raw_df[col]
clean_s = raw_s.replace(invalid_val, np.nan) if cleaned_df is None else cleaned_df[col]
n_err = int((raw_s == invalid_val).sum())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(raw_s.values, color="#d62728", lw=0.7)   # Trái: Trước lọc (màu đỏ)
...
ax2.plot(clean_s.values, color="#2ca02c", lw=0.7) # Phải: Sau lọc (màu xanh)
...
fig.suptitle("Kiểm tra cảm biến: Loại bỏ giá trị gió lỗi (-9999.0)", ...)
_finalize_plot(fig, save_path=save_path)
```

### Vai trò của hàm
> **Kiểm tra chất lượng dữ liệu gốc:** Phát hiện và chứng minh trực quan tác hại của lỗi cảm biến tốc độ gió `-9999.0` trước và sau khi lọc sang `NaN`.

### Giải thích từng dòng
- `raw_s = raw_df["wv (m/s)"]`: Lấy cột tốc độ gió gốc (420.551 dòng).
- `raw_s.replace(invalid_val, np.nan)`: Thay toàn bộ giá trị lỗi cảm biến `-9999.0` thành `NaN` (ô trống).
- `n_err = int((raw_s == invalid_val).sum())`: Đếm chính xác số điểm lỗi (dữ liệu thật có **18 điểm** ở `wv (m/s)` và 20 điểm ở `max. wv (m/s)`).
- `plt.subplots(1, 2, ...)`: Chia khung thành 1 hàng × 2 cột (`ax1` bên trái vẽ dữ liệu lỗi, `ax2` bên phải vẽ dữ liệu sạch).
- `clean_s.min()` / `clean_s.max()`: Tự động tính miền tốc độ gió hợp lệ sau khi lọc để in lên tiêu đề (`[0.0, 28.5] m/s`).

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** 2 biểu đồ đường (Line) đặt cạnh nhau so sánh tốc độ gió trước và sau khi lọc lỗi `-9999.0`.
- **Tại sao chọn Line:** Nhìn vào đường bên trái thấy ngay các gai lao xuống `-10.000 m/s` kéo dẹt toàn bộ dữ liệu gió thật sát về 0.
- **Ý nghĩa:** Phải lọc `-9999.0 -> NaN` trước khi gom mẫu `1h` và chuẩn hóa, nếu không sẽ làm sai lệch hoàn toàn Mean/Std của `StandardScaler` và gây nổ gradient khi huấn luyện.
- **Câu nói mẫu:** *"Hàm này kiểm tra chất lượng cảm biến gió gốc: phát hiện 18 điểm lỗi -9999.0 kéo dẹt biểu đồ bên trái; sau khi đổi sang NaN ở bên phải, tốc độ gió trở về đúng dải vật lý 0 đến ~28 m/s."*

---

## HÀM 3: `plot_data_split_timeline()` (Dòng 91–123)
**Ảnh sinh ra:** `evalution_results/eda/eda_data_split_timeline.png`

### Code chính
```python
fig, ax = plt.subplots(figsize=(14, 4.5))
splits = [
    (train_df, "#1f77b4", f"Train 70% ({len(train_df):,}h)"),
    (val_df, "#ff7f0e", f"Val 15% ({len(val_df):,}h)"),
    (test_df, "#2ca02c", f"Test 15% ({len(test_df):,}h)"),
]
for df_part, color, label in splits:
    ax.plot(df_part.index, df_part[target_col], color=color, lw=0.7, alpha=0.85, label=label)

ax.axvline(val_df.index.min(), color="#d62728", ls="--", lw=1.5, label="Ranh giới Train | Val")
ax.axvline(test_df.index.min(), color="#9467bd", ls="--", lw=1.5, label="Ranh giới Val | Test")
_finalize_plot(fig, ax, ..., save_path=save_path, show_legend=True)
```

### Vai trò của hàm
> **Trực quan hóa Pha 1 (Split First):** Vẽ 3 tập Train (70%), Val (15%), Test (15%) nối tiếp nhau trên cùng trục thời gian để chứng minh tuyệt đối không rò rỉ tương lai (*Zero Time Leakage*).

### Giải thích từng dòng & các chỉ số trong hàm
- `figsize=(14, 4.5)`: Kích thước khung ảnh tính bằng **inch** (rộng 14 inch, cao 4.5 inch) — chọn dáng ngang dài để chuỗi thời gian gần 8 năm trải rộng, không bị nén dính.
- `splits = [...]`: Gom 3 bộ `(DataFrame, mã màu, nhãn chú thích)` gồm Train (`49,090h`), Val (`10,519h`), Test (`10,520h`) để vẽ gọn bằng 1 vòng `for`:
  - Mã màu Hex chuẩn `tab10` của Matplotlib: `#1f77b4` (xanh dương — Train), `#ff7f0e` (cam — Val), `#2ca02c` (xanh lá — Test) để 3 tập tương phản rõ ràng.
  - `:,` trong `f"{len(train_df):,}h"`: Định dạng tự động thêm dấu phẩy ngăn cách hàng nghìn (`49090` → `49,090h`) cho dễ đọc.
- `ax.plot(df_part.index, df_part[target_col], lw=0.7, alpha=0.85)`:
  - `lw=0.7` (*linewidth*): Độ dày nét mảnh `0.7` vì có tới 70.129 điểm giờ, nếu nét dày sẽ bị bết màu.
  - `alpha=0.85`: Độ đậm/trong suốt (thang `0` trong suốt → `1` đục hoàn toàn), đặt `0.85` cho dịu mắt.
- `ax.axvline(val_df.index.min(), color="#d62728", ls="--", lw=1.5)` & `test_df.index.min()` (`#9467bd`):
  - Kẻ 2 đường dọc đánh dấu mốc cắt `Train | Val` (đỏ `#d62728`) và `Val | Test` (tím `#9467bd`).
  - `ls="--"` (*linestyle*): Nét gạch đứt để phân biệt với đường dữ liệu nét liền.
  - `lw=1.5`: Dày hơn nét dữ liệu (`0.7`) để ranh giới nổi bật ngay khi nhìn vào.

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** Biểu đồ dòng thời gian minh họa **Pha 1 (Split First)** chia dữ liệu 1h thành 70% Train – 15% Val – 15% Test.
- **Tại sao chọn Line Timeline:** Chứng minh trực quan rằng dữ liệu được cắt tuần tự từ quá khứ đến tương lai, **không trộn ngẫu nhiên**.
- **Ý nghĩa chống rò rỉ:** Đảm bảo $\max(\text{Train}) < \min(\text{Val}) < \min(\text{Test})$ (**Zero Time Leakage**). Sau bước này khóa kín Val và Test, toàn bộ 6 hàm EDA tiếp theo chỉ dùng `train_df`.
- **Câu nói mẫu:** *"Ở Pha 1, dữ liệu được cắt nối tiếp theo trục thời gian 70-15-15 với 2 đường ranh giới tách bạch hoàn toàn để chống rò rỉ tương lai."*

---

## HÀM 4: `plot_temperature_trend()` (Dòng 130–157)
**Ảnh sinh ra:** `evalution_results/eda/eda_temperature_trend.png`

### Code chính
```python
s = train_df[target_col]
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(train_df.index, s, color="tab:blue", lw=0.6, alpha=0.55, label="Nhiệt độ 1h")

if len(train_df) >= 720:
    ax.plot(train_df.index, s.rolling(720, min_periods=1).mean(), color="#d62728", lw=2.0, label="Trung bình trượt 30 ngày (720h)")

ax.axhline(s.mean(), color="#2ca02c", ls="--", lw=1.3, label=f"Mean: {s.mean():.2f}°C (Min {s.min():.1f}°C | Max {s.max():.1f}°C)")
_finalize_plot(fig, ax, ..., save_path=save_path, show_legend=True)
```

### Vai trò của hàm
> **Phân tích xu hướng dài hạn & tính mùa vụ (Pha 2):** Vẽ chuỗi nhiệt độ 1h kèm đường trung bình trượt 30 ngày (`720h`) trên 70% tập Train để làm rõ chu kỳ mùa lặp lại hàng năm.

### Giải thích từng dòng
- `s = train_df[target_col]`: Chỉ lấy chuỗi nhiệt độ trên 70% tập Train.
- `ax.plot(..., alpha=0.55)`: Vẽ đường nhiệt độ từng giờ làm lớp nền mờ.
- `if len(train_df) >= 720:` Kiểm tra đủ dữ liệu tối thiểu 30 ngày mới vẽ đường xu hướng.
- `s.rolling(720, min_periods=1).mean()`: Tính **trung bình trượt 30 ngày** (`30 ngày × 24 giờ = 720` bước) để làm mượt dao động ngày-đêm và làm nổi bật sóng mùa vụ màu đỏ. `min_periods=1` để đầu chuỗi không bị trống `NaN`.
- `ax.axhline(s.mean(), ls="--")`: Vẽ đường ngang nét đứt tại nhiệt độ trung bình `9.11°C`, kèm Min `-22.7°C` và Max `+35.5°C`.

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** Xu hướng nhiệt độ dài hạn (2009–2014) trên tập Train kèm đường trung bình 30 ngày.
- **Tại sao chọn Line + Rolling Mean 30 ngày:** Đường 1h dao động rất dày; đường trung bình trượt 720h giúp thấy rõ dạng sóng hình sin lặp lại đúng chu kỳ **1 năm**.
- **Ý nghĩa cho mô hình:** Nhiệt độ có tính **mùa vụ mạnh** (chênh gần `60°C` giữa đông và hè), vì vậy dùng cửa sổ đầu vào **168 giờ (7 ngày)** giúp mô hình nhận diện chính xác nền nhiệt của mùa hiện tại.
- **Câu nói mẫu:** *"Đường trung bình trượt 30 ngày (720h) màu đỏ cho thấy tính mùa vụ hình sin rất rõ lặp lại theo từng năm, với biên độ nhiệt rộng từ -22.7°C đến +35.5°C."*

---

## HÀM 5: `plot_temperature_distribution()` (Dòng 160–184)
**Ảnh sinh ra:** `evalution_results/eda/eda_temperature_distribution.png`

### Code chính
```python
s = train_df[target_col].dropna()
fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [0.22, 0.78]})

sns.boxplot(x=s, ax=ax_box, color="#6baed6", fliersize=3)
ax_box.set_title(f"Phân phối {target_col} (Mean={s.mean():.2f}°C, Std={s.std():.2f}°C, Skew={s.skew():.2f})", fontweight="bold")
ax_box.set_xlabel("")

sns.histplot(s, bins=50, kde=True, ax=ax_hist, color="#3182bd", alpha=0.55)
ax_hist.axvline(s.mean(), color="#d62728", ls="--", lw=1.8, label=f"Mean: {s.mean():.2f}°C")
ax_hist.axvline(s.median(), color="#2ca02c", ls="-.", lw=1.8, label=f"Median: {s.median():.2f}°C")
_finalize_plot(fig, ax_hist, xlabel="Nhiệt độ (°C)", ylabel="Số giờ", save_path=save_path, show_legend=True)
```

### Vai trò của hàm
> **Phân tích phân phối biến mục tiêu `T (degC)` (Pha 2):** Kết hợp Boxplot và Histogram/KDE để kiểm tra độ lệch (*Skewness*), độ trải rộng (*Std*) và các giá trị cực trị/ngoại lai mùa đông.

### Giải thích từng dòng
- `s = train_df[target_col].dropna()`: Lọc bỏ giá trị rỗng (nếu có) trước khi tính thống kê phân phối.
- `plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [0.22, 0.78]})`: Tạo 2 khung xếp dọc dùng chung trục X nhiệt độ (`sharex=True`); khung trên 22% vẽ Boxplot, khung dưới 78% vẽ Histogram.
- `sns.boxplot(x=s, fliersize=3)`: Vẽ hộp tứ phân vị (Q1–Q3) và chấm nhỏ các điểm ngoại lai (outliers).
- `s.skew()`: Tính hệ số bất đối xứng (*Skewness = -0.095*, gần 0 tức là phân phối khá cân đối).
- `sns.histplot(s, bins=50, kde=True)`: Chia dải nhiệt độ thành 50 cột tần suất kèm đường cong mật độ mượt **KDE**.
- 2 đường `axvline` đánh dấu `Mean (9.11°C)` và `Median (9.20°C)` gần trùng nhau.

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** Biểu đồ phân phối của biến mục tiêu `T (degC)` trên tập Train.
- **Tại sao chọn kết hợp Boxplot + Histogram/KDE:** Histogram/KDE cho thấy hình dạng phân phối tổng thể, còn Boxplot phía trên chỉ rõ các điểm cực trị rét đậm (`< -15°C`).
- **Ý nghĩa cho mô hình:** Độ lệch chuẩn nhiệt độ lớn (`Std = 8.65°C`), nên tiêu chí đánh giá tỷ lệ % số giờ dự báo lệch $\le \pm 1.5^\circ\text{C}$ cùng `MAE`/`RMSE` phản ánh rất nghiêm khắc chất lượng dự báo.
- **Câu nói mẫu:** *"Biểu đồ phân phối cho thấy nhiệt độ khá đối xứng quanh 9.1–9.2°C nhưng trải rộng (Std 8.65°C) và có các đợt rét cực đoan dưới -15°C ở đuôi trái."*

---

## HÀM 6: `plot_diurnal_cycle()` (Dòng 187–215)
**Ảnh sinh ra:** `evalution_results/eda/eda_diurnal_cycle.png`

### Code chính
```python
stats = train_df.groupby(train_df.index.hour)[target_col].agg(["mean", "std"]).fillna(0.0)
hours = stats.index.to_numpy()
peak_h, trough_h = int(stats["mean"].idxmax()), int(stats["mean"].idxmin())

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(hours, stats["mean"], marker="o", color="crimson", lw=2.0, label="Nhiệt độ trung bình")
ax.fill_between(hours, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color="crimson", alpha=0.18, label="±1 Std")
ax.scatter([peak_h, trough_h], [stats.loc[peak_h, "mean"], stats.loc[trough_h, "mean"]], color=["#b2182b", "#2166ac"], s=70, zorder=5)
ax.set_xticks(hours)
_finalize_plot(fig, ax, ..., save_path=save_path, show_legend=True)
```

### Vai trò của hàm
> **Phân tích chu kỳ ngày đêm 24 giờ (Pha 2):** Gom nhóm nhiệt độ theo từng giờ trong ngày (`0–23h`) để xác định giờ nóng nhất, lạnh nhất và quy luật lặp lại 24h của bài toán dự báo 72h.

### Giải thích từng dòng
- `train_df.groupby(train_df.index.hour)[target_col].agg(["mean", "std"])`: Lấy giờ trong ngày (`0..23`) từ `DateTimeIndex`, gom nhóm và tính đồng thời `mean` (trung bình) và `std` (độ lệch chuẩn) cho từng khung giờ. `.fillna(0.0)` tránh lỗi nếu 1 giờ chỉ có 1 mẫu (`std` bị `NaN`).
- `hours = stats.index.to_numpy()`: Lấy đúng các giờ hiện có (fix lỗi crash nếu tập test nhỏ hơn 24 giờ).
- `idxmax()` / `idxmin()`: Tự động tìm ra giờ nóng nhất (**15h = `12.44°C`**) và giờ lạnh nhất (**5h = `5.99°C`**).
- `ax.fill_between(..., mean - std, mean + std, alpha=0.18)`: Tô dải bóng mờ khoảng dao động `±1 Std`.
- `ax.scatter(..., zorder=5)`: Vẽ nổi 2 chấm đỉnh/đáy đè lên trên đường vẽ (`zorder=5`).

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** Đường cong biến thiên nhiệt độ trung bình theo 24 giờ trong ngày (kèm vùng `±1 Std`).
- **Tại sao chọn Line + `fill_between`:** Thể hiện trực quan pha lên–xuống của chu kỳ ngày đêm và độ biến thiên theo mùa.
- **Ý nghĩa cho mô hình:** Chu kỳ 24h cực kỳ ổn định (đáy 5h sáng trước bình minh, đỉnh 15h chiều). Vì bài toán dự báo **72 giờ = đúng 3 chu kỳ 24h**, mô hình phải tái tạo tốt nhịp dao động ngày–đêm này.
- **Câu nói mẫu:** *"Chu kỳ 24 giờ rất đều đặn: thấp nhất lúc 5h sáng (6.0°C) và cao nhất lúc 15h chiều (12.4°C). Dự báo 72 giờ tương lai chính là dự báo liên tiếp 3 chu kỳ ngày đêm."*

---

## HÀM 7: `plot_correlation_heatmap()` (Dòng 218–233)
**Ảnh sinh ra:** `evalution_results/eda/eda_correlation_heatmap.png`

### Code chính
```python
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
    vmin=-1, vmax=1, center=0, square=True, lw=0.5, ax=ax
)
ax.set_title("Ma trận tương quan Pearson giữa 14 đặc trưng khí tượng (Tập Train)", fontsize=12, fontweight="bold")
_finalize_plot(fig, save_path=save_path)
```

### Vai trò của hàm
> **Phân tích tương quan đa biến (Pha 2):** Vẽ bản đồ nhiệt ma trận tương quan Pearson `14 × 14` trên tập Train để phát hiện đa cộng tuyến và các đặc trưng khí tượng ảnh hưởng mạnh nhất đến nhiệt độ `T`.

### Giải thích từng dòng
- `train_df.corr()`: Tính ma trận tương quan tuyến tính Pearson kích thước `14 × 14` trên tập Train.
- `annot=True, fmt=".2f"`: Hiển thị số tương quan làm tròn 2 chữ số ngay trong từng ô.
- `cmap="coolwarm", vmin=-1, vmax=1, center=0`: Khóa dải màu cố định từ `-1` (xanh đậm: nghịch biến) qua `0` (trắng: không tương quan tuyến tính) tới `+1` (đỏ đậm: đồng biến).
- `square=True, lw=0.5`: Mỗi ô là hình vuông đều có đường viền mảnh `0.5` giúp bảng rất dễ nhìn.

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** Ma trận tương quan Pearson giữa toàn bộ 14 biến khí tượng.
- **Tại sao chọn Heatmap:** Nhìn nhanh 196 cặp tương quan trong 1 hình duy nhất để nhận diện biến mạnh và đa cộng tuyến.
- **Ý nghĩa cho mô hình:**
  - Tương quan thuận cực mạnh với `T`: `Tpot (+0.997)`, `VPmax (+0.950)`, `Tdew (+0.902)`.
  - Tương quan nghịch cực mạnh với `T`: `rho (-0.965)` (trời nóng không khí loãng), `rh (-0.563)`.
  - Biến áp suất `p`, tốc độ gió `wv`, hướng gió `wd` có tương quan tuyến tính yếu nhưng là tín hiệu phi tuyến báo trước không khí lạnh/mưa bão → **giữ đủ 14 đặc trưng đầu vào**.
- **Câu nói mẫu:** *"Heatmap cho thấy nhóm nhiệt-ẩm tương quan rất mạnh với T (Tpot +0.997, VPmax +0.95, rho -0.965). Các biến áp suất và gió tuy tương quan tuyến tính yếu nhưng mang thông tin đổi thời tiết nên nhóm giữ cả 14 đặc trưng."*

---

## HÀM 8: `plot_scatter_relationships()` (Dòng 236–268)
**Ảnh sinh ra:** `evalution_results/eda/eda_scatter_relationships.png`

### Code chính
```python
clean = train_df.dropna()
sample = clean.sample(n=min(3000, len(clean)), random_state=42)
pairs = [
    ("VPmax (mbar)", "#d62728", "Phi tuyến hàm mũ"),
    ("rho (g/m**3)", "#1f77b4", "Tuyến tính nghịch chặt"),
    ("rh (%)", "#2ca02c", "Nghịch biến, phân tán rộng"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (col, color, desc) in zip(axes, pairs):
    r = sample[target_col].corr(sample[col])
    ax.scatter(sample[target_col], sample[col], s=10, alpha=0.35, color=color)
    ax.set_title(f"{target_col} vs {col}\n(r = {r:+.2f} | {desc})", fontsize=10, fontweight="bold")
    ...
_finalize_plot(fig, save_path=save_path)
```

### Vai trò của hàm
> **Kiểm chứng tính phi tuyến của dữ liệu (Pha 2):** Vẽ 3 biểu đồ phân tán giữa `T (degC)` với `VPmax`, `rho`, `rh` để phơi bày quan hệ phi tuyến mà Heatmap không nhìn thấy được, qua đó chứng minh sự cần thiết của Deep Learning.

### Giải thích từng dòng
- `clean.sample(n=min(3000, len(clean)), random_state=42)`: Tập Train có gần 50.000 điểm, vẽ hết sẽ bị bết đặc (overplotting) nên lấy mẫu ngẫu nhiên 3.000 điểm cố định `random_state=42` để hình rõ đẹp và kết quả tái lập giống nhau mỗi lần chạy.
- `pairs = [...]` + `zip(axes, pairs)`: Duyệt song song 3 khung con (`axes[0..2]`) với 3 biến tiêu biểu (`VPmax`, `rho`, `rh`).
- `ax.scatter(..., s=10, alpha=0.35)`: Vẽ điểm nhỏ kích thước `10` và độ trong suốt `0.35` để thấy vùng nào mật độ điểm dày đặc.

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** 3 biểu đồ phân tán (Scatter) giữa nhiệt độ `T` với `VPmax`, `rho` và `rh`.
- **Tại sao phải vẽ thêm Scatter sau Heatmap?** Heatmap chỉ đo tương quan **tuyến tính**; Scatter phơi bày hình dạng thật của dữ liệu — điển hình là **`T` vs `VPmax` tạo thành đường cong hàm mũ phi tuyến sắc nét** (phương trình vật lý *Clausius–Clapeyron*).
- **Ý nghĩa cho mô hình:** Bằng chứng trực quan giải thích **tại sao phải chọn Deep Learning (LSTM / Attention / Transformer)** có khả năng học quan hệ phi tuyến phức tạp thay vì hồi quy tuyến tính.
- **Câu nói mẫu:** *"Heatmap chỉ đo tuyến tính, nên em vẽ thêm Scatter Plot để xem hình dạng thật. Khung đầu tiên cho thấy T và VPmax tạo đường cong hàm mũ phi tuyến rõ rệt — chứng minh cần dùng Deep Learning thay vì mô hình tuyến tính."*

---

## HÀM 9: `plot_autocorrelation()` (Dòng 271–305)
**Ảnh sinh ra:** `evalution_results/eda/eda_autocorrelation.png`

### Code chính
```python
s = train_df[target_col].ffill().bfill()
n_lags = min(max_lag, max(1, len(s) - 2))
lags = np.arange(1, n_lags + 1)
acf = np.array([s.autocorr(lag=int(k)) for k in lags])

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(lags, acf, color="#1f77b4", lw=2.0, label="Autocorrelation (ACF)")
for lag_h in range(24, n_lags + 1, 24):
    ax.axvline(lag_h, color="#d62728", ls="--", lw=1.0, alpha=0.6)
    ax.scatter([lag_h], [acf[lag_h - 1]], color="#d62728", s=35, zorder=5)
    ax.text(lag_h, acf[lag_h - 1] + 0.03, f"{lag_h}h\n({acf[lag_h - 1]:.2f})", ha="center", ...)
```

### Vai trò của hàm
> **Phân tích tính tự tương quan theo độ trễ (Pha 2):** Đo mức độ phụ thuộc giữa nhiệt độ hiện tại với quá khứ từ `1h → 168h` để tạo căn cứ khoa học chọn cửa sổ đầu vào `168h` và lý giải vai trò của Attention/Transformer.

### Giải thích từng dòng
- `s = train_df[target_col].ffill().bfill()`: Điền khuyết nội bộ chuỗi Train trước khi tính tự tương quan.
- `lags = np.arange(1, n_lags + 1)`: Tạo mảng độ trễ từ `1` đến `168` giờ (7 ngày).
- `[s.autocorr(lag=int(k)) for k in lags]`: Tính hệ số tự tương quan giữa $T(t)$ và chính nó cách $k$ giờ trong quá khứ $T(t - k)$.
- `for lag_h in range(24, n_lags + 1, 24)`: Lặp qua đúng các mốc bội số 24h (`24, 48, 72, 96, 120, 144, 168`) để kẻ vạch đỏ, chấm đỉnh và in số (`acf[lag_h - 1]` vì chỉ số mảng bắt đầu từ `0`).

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** Biểu đồ hàm tự tương quan **ACF** của nhiệt độ từ độ trễ 1h đến 168h.
- **Tại sao chọn biểu đồ ACF:** Là căn cứ khoa học quyết định **độ dài chuỗi đầu vào** và lý giải **sự vượt trội của Attention/Transformer**.
- **Ý nghĩa cho mô hình (ĐIỂM CHỐT ĂN ĐIỂM NHẤT):**
  1. Các đỉnh tự tương quan rơi chính xác vào các mốc **24h (`0.92`), 48h (`0.87`), 72h (`0.83`), ..., và tới tận 168h (7 ngày) vẫn còn `0.77`** → **lý giải tại sao chọn đầu vào `168h` dự báo `72h`**.
  2. Baseline Seq2Seq LSTM nén toàn bộ 168 bước vào 1 vector ẩn dễ quên thông tin xa, còn **Bahdanau Attention** và **Transformer** có thể đặt trọng số chú ý thẳng vào các đỉnh bội số 24h trong quá khứ.
- **Câu nói mẫu:** *"Biểu đồ ACF cho thấy các đỉnh tự tương quan lặp lại đúng mỗi 24h và sau 7 ngày (168h) vẫn đạt 0.77. Điều này vừa chứng minh cơ sở chọn cửa sổ 168h→72h, vừa giải thích vì sao Attention và Transformer vượt trội LSTM thường."*

---

## HÀM 10: `plot_scaling_comparison()` (Dòng 312–346)
**Ảnh sinh ra:** `evalution_results/eda/eda_scaling_comparison.png`

### Code chính
```python
cols = selected_features or [c for c in ["p (mbar)", "rho (g/m**3)", "rh (%)", "T (degC)", "VPmax (mbar)", "wv (m/s)"] if c in feature_names]
scaled_df = pd.DataFrame(train_scaled, columns=feature_names)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(data=train_raw[cols], ax=ax1, palette="Set2", fliersize=2)
...
sns.boxplot(data=scaled_df[cols], ax=ax2, palette="Set2", fliersize=2)
ax2.axhline(0, color="#d62728", ls="--", lw=1.2)
...
_finalize_plot(fig, save_path=save_path)
```

### Vai trò của hàm
> **Trực quan hóa Pha 3 (Chuẩn hóa chống rò rỉ):** So sánh biên độ 6 đặc trưng khí tượng tiêu biểu trước và sau khi áp dụng `StandardScaler` (chỉ `fit` trên tập Train) để chứng minh hiệu quả cân bằng thang đo.

### Giải thích từng dòng
- `cols = [...]`: Chọn 6 cột đại diện có thang đo chênh lệch rõ nhất (áp suất ~990, mật độ không khí ~1200, độ ẩm ~75, nhiệt độ ~9, gió ~2).
- `scaled_df = pd.DataFrame(train_scaled, columns=feature_names)`: Bọc mảng NumPy sau chuẩn hóa thành DataFrame để Seaborn vẽ theo tên cột.
- `ax1` vẽ Boxplot dữ liệu gốc, `ax2` vẽ Boxplot dữ liệu sau `StandardScaler` kèm đường chuẩn `0` (`ax2.axhline(0)`). `rotation=25` xoay nghiêng tên cột 25° cho khỏi dính chữ.

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** So sánh thang đo của 6 đặc trưng **Trước** và **Sau `StandardScaler`** (**Pha 3**).
- **Tại sao chọn Boxplot 2 khung:** Nhìn thấy ngay sự chênh lệch biên độ hàng trăm lần ở bên trái đã được chuẩn hóa đồng đều về **`Mean ≈ 0, Std ≈ 1`** ở bên phải.
- **Ý nghĩa cho mô hình & chống rò rỉ:** Đưa 14 biến về cùng thang đo giúp optimizer Adam cập nhật gradient cân bằng, hội tụ nhanh. Đặc biệt `StandardScaler` chỉ gọi `.fit()` duy nhất trên tập Train (`n_samples_seen_ == 49.090`) rồi `.transform()` cho Val & Test.
- **Câu nói mẫu:** *"Ở Pha 3, Boxplot chứng minh trước chuẩn hóa các biến lệch biên độ hàng trăm lần (rho ~1200 vs gió ~2); sau StandardScaler fit duy nhất trên Train, mọi biến đều về chuẩn Mean=0, Std=1."*

---

## HÀM 11: `plot_sliding_window_example()` (Dòng 349–387)
**Ảnh sinh ra:** `evalution_results/eda/eda_sliding_window_example.png`

### Code chính
```python
total = seq_len + pred_len
if len(train_df) < start_idx + total:
    start_idx = 0
win = train_df.iloc[start_idx : start_idx + total]
past, future = win.iloc[:seq_len], win.iloc[seq_len:]

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(past.index, past[target_col], color="#1f77b4", lw=2.0, label=f"Đầu vào X ({seq_len}h quá khứ, 14 biến)")
ax.plot(future.index, future[target_col], color="#d62728", lw=2.0, ls="--", label=f"Nhãn Y ({pred_len}h tương lai, T degC)")
ax.axvline(future.index[0], color="#111111", ls=":", lw=1.6, label="Mốc hiện tại (t = 0)")
ax.axvspan(past.index[0], future.index[0], color="#1f77b4", alpha=0.08)
ax.axvspan(future.index[0], future.index[-1], color="#d62728", alpha=0.08)
_finalize_plot(fig, ax, ..., save_path=save_path, show_legend=True)
```

### Vai trò của hàm
> **Trực quan hóa Pha 4 (Cửa sổ trượt độc lập):** Minh họa trực quan 1 cặp mẫu huấn luyện `(168 giờ quá khứ X → 72 giờ tương lai Y)` được cắt hoàn toàn nội bộ trong tập Train.

### Giải thích từng dòng
- `total = seq_len + pred_len`: Tổng chiều dài 1 mẫu là `168 + 72 = 240` giờ liên tiếp.
- `win = train_df.iloc[start_idx : start_idx + total]`: Cắt đúng 1 đoạn 240 giờ nội bộ trong `train_df`.
- `past = win.iloc[:168]` (đầu vào $X$) và `future = win.iloc[168:]` (nhãn mục tiêu $Y$).
- `ax.axvline(future.index[0], ...)`: Kẻ vạch đứng tại mốc hiện tại $t = 0$ ngăn cách quá khứ và tương lai.
- `ax.axvspan(..., alpha=0.08)`: Tô nền xanh nhạt vùng đầu vào $X$ và nền đỏ nhạt vùng nhãn dự báo $Y$.

### Hình ảnh là gì & Ý nghĩa khi thuyết trình
- **Là gì:** Minh họa 1 cặp mẫu **cửa sổ trượt (Sliding Window)** trong **Pha 4**.
- **Tại sao chọn biểu đồ này:** Biến khái niệm trừu tượng `(Batch, 168, 14) → (Batch, 72, 1)` thành hình ảnh dễ hiểu ngay lập tức.
- **Ý nghĩa chống rò rỉ:** Các cửa sổ được cắt **độc lập trong từng phân tập** (`Train: 48.851`, `Val: 10.280`, `Test: 10.281` mẫu), tuyệt đối không cắt vắt qua ranh giới giữa 2 tập.
- **Câu nói mẫu:** *"Hình cuối cùng minh họa Pha 4: 1 mẫu cửa sổ trượt lấy 168 giờ quá khứ vùng xanh làm đầu vào X để dự báo 72 giờ tương lai vùng đỏ làm nhãn Y, cắt hoàn toàn nội bộ từng tập."*

---

## HÀM 12: `summarize_eda_insights()` (Dòng 394–424)

### Code chính
```python
s = train_df[target_col].dropna()
h_mean = train_df.groupby(train_df.index.hour)[target_col].mean()
corr = train_df.corr()[target_col].drop(target_col).sort_values(ascending=False)
filled = train_df[target_col].ffill().bfill()

return {
    "n_rows_train": len(train_df),
    "missing_total": int(train_df.isna().sum().sum()),
    "temp_mean_degC": round(float(s.mean()), 2),
    ...
    "top_positive_corr": corr.head(3).round(3).to_dict(),
    "top_negative_corr": corr.tail(3).round(3).to_dict(),
    "autocorrelation": {f"lag_{k}h": round(float(filled.autocorr(lag=k)), 4) for k in [24, 48, 72, 168] if len(filled) > k + 1},
}
```

### Vai trò của hàm
> **Tổng hợp số liệu định lượng (không vẽ hình):** Tính tự động toàn bộ các chỉ số thống kê cốt lõi trên tập Train (Mean, Std, Skew, giờ nóng/lạnh nhất, top biến tương quan, ACF) trả về dạng `dict` để kiểm chứng báo cáo và in ra Notebook.

### Giải thích từng dòng
- `s = train_df[target_col].dropna()`: Lấy chuỗi nhiệt độ sạch để tính `mean`, `median`, `std`, `min`, `max`, `skew`.
- `h_mean = train_df.groupby(train_df.index.hour)[target_col].mean()`: Tìm giờ nóng nhất (`idxmax()`) và lạnh nhất (`idxmin()`).
- `corr = train_df.corr()[target_col].drop(target_col).sort_values(ascending=False)`: Lấy cột tương quan với `T (degC)`, bỏ chính `T` (bằng 1.0), sắp xếp giảm dần → `head(3)` là 3 biến tương quan thuận mạnh nhất (`Tpot`, `VPmax`, `Tdew`), `tail(3)` là 3 biến tương quan nghịch mạnh nhất (`p`, `rh`, `rho`).
- Dict comprehension ở dòng cuối tính nhanh ACF tại 4 mốc vàng `24h, 48h, 72h, 168h` (có kiểm tra `len(filled) > k + 1` để không lỗi trên dữ liệu test nhỏ).

---

## HÀM 13: `run_eda_pipeline()` (Dòng 427–493)

### Code chính
```python
def run_eda_pipeline(train_df, output_dir="evalution_results/eda", include_extended=True, dm=None):
    os.makedirs(output_dir, exist_ok=True)
    tasks = [
        ("eda_temperature_trend.png", plot_temperature_trend),
        ("eda_correlation_heatmap.png", plot_correlation_heatmap),
        ("eda_diurnal_cycle.png", plot_diurnal_cycle),
    ]
    if include_extended:
        tasks += [
            ("eda_temperature_distribution.png", plot_temperature_distribution),
            ("eda_scatter_relationships.png", plot_scatter_relationships),
            ("eda_autocorrelation.png", plot_autocorrelation),
        ]

    generated_files = []
    for filename, plot_fn in tasks:
        path = os.path.join(output_dir, filename)
        plot_fn(train_df, save_path=path)
        generated_files.append(path)

    if include_extended and dm is not None:
        # Xuất thêm 4 biểu đồ quy trình chống rò rỉ -> tổng đúng 10 ảnh PNG
        ...
    return generated_files
```

### Vai trò của hàm
> **Hàm điều phối ("nhạc trưởng") của toàn bộ module `src/eda.py`:** Được `main.py --mode eda` và `tests/test_eda.py` gọi để tự động xuất toàn bộ biểu đồ EDA ra file PNG vào `evalution_results/eda/`.

### Giải thích từng dòng
- `tasks = [(tên_file, hàm_vẽ), ...]`: Gom danh sách các hàm vẽ vào 1 list rồi duyệt vòng `for` gọi lần lượt, giúp code rất gọn và dễ thêm bớt biểu đồ.
- **2 chế độ hoạt động linh hoạt:**
  - Nếu chỉ truyền `train_df` (như trong unit test `tests/test_eda.py`): xuất **6 ảnh EDA thuần tập Train** (nếu `include_extended=False` chỉ xuất 3 ảnh gốc).
  - Nếu truyền thêm `dm` (`JenaDataModule` từ `main.py --mode eda`): xuất thêm **4 ảnh quy trình chống rò rỉ** (`eda_wind_anomalies`, `eda_data_split_timeline`, `eda_scaling_comparison`, `eda_sliding_window_example`) → **đủ 10 ảnh PNG** giống hệt Notebook.
- `pd.read_csv(dm.raw_csv_path, usecols=["wv (m/s)"])`: Chỉ đọc đúng 1 cột `wv (m/s)` từ file gốc 41MB để chạy nhanh và tiết kiệm RAM.

---

## KHỐI 14: `if __name__ == "__main__":` — SMOKE TEST (Dòng 496–516)

### Code chính
```python
if __name__ == "__main__":
    cols = [...]  # 14 cột khí tượng
    mock_train = pd.DataFrame(np.random.randn(500, 14), index=pd.date_range("2012-01-01", periods=500, freq="1h"), columns=cols)

    test_out = "evalution_results/test_eda"
    files = run_eda_pipeline(mock_train, output_dir=test_out)
    insights = summarize_eda_insights(mock_train)

    assert len(files) == 6 and "temp_mean_degC" in insights
    for f in files:
        assert os.path.exists(f)
        os.remove(f)
    if os.path.exists(test_out):
        os.rmdir(test_out)
```

### Vai trò của khối
> **Smoke Test tự kiểm tra độc lập:** Cho phép chạy trực tiếp `python src/eda.py` để xác minh nhanh toàn bộ pipeline EDA hoạt động đúng trong vài giây mà không cần tải file CSV 41MB.

### Giải thích từng dòng
- `mock_train = pd.DataFrame(...)`: Tạo dữ liệu giả lập 500 giờ × 14 cột khí tượng có `DateTimeIndex`.
- Gọi `run_eda_pipeline()` xuất 6 ảnh vào thư mục tạm `evalution_results/test_eda` và gọi `summarize_eda_insights()`.
- `assert len(files) == 6 and "temp_mean_degC" in insights`: Tự kiểm định đủ 6 file ảnh và dict thống kê hợp lệ.
- Vòng `for` cuối kiểm tra từng file đã tồn tại rồi tự động xóa sạch file và thư mục tạm để giữ dự án sạch.

---

## TỪ ĐIỂN GIẢI THÍCH TOÀN BỘ CHỈ SỐ & CON SỐ TRONG `src/eda.py` ("TẠI SAO LẠI LÀ SỐ NÀY?")

> **Mẹo nhớ khi cô hỏi bất ngờ vào một con số bất kỳ trong code:** Toàn bộ các con số trong `src/eda.py` chỉ thuộc **2 nhóm**: (1) **Con số mang ý nghĩa bài toán/thống kê** (có cơ sở khoa học rõ ràng) và (2) **Con số thẩm mỹ Matplotlib/Seaborn** (chọn để biểu đồ rõ nét, không bị chồng chữ — hoàn toàn có thể trả lời thẳng *"em chọn thông số thẩm mỹ này để ảnh rõ nét/cân đối trên slide"*!).

### Nhóm 1: Các con số bài toán & thống kê (CÔ RẤT HAY HỎI — PHẢI NHỚ CHẮC)

| Chỉ số / Con số | Xuất hiện ở hàm | Tại sao lại là con số này? |
| :--- | :--- | :--- |
| **`-9999.0`** (`invalid_val`) | `plot_wind_anomalies` | Mã lỗi phần cứng của cảm biến gió trong bộ dữ liệu gốc Jena Climate (tốc độ gió vật lý phải $\ge 0$). |
| **`70% / 15% / 15%`** | `plot_data_split_timeline` | Tỷ lệ chia tập chuẩn cho dữ liệu ~70.000 giờ: 70% Train (~5.6 năm) đủ học nhiều chu kỳ năm; Val và Test mỗi tập ~1.2 năm phủ trọn 4 mùa. |
| **`720`** (`rolling(720)`) | `plot_temperature_trend` | Vì dữ liệu đã resample `1h`, nên **30 ngày × 24 giờ = 720 giờ** (trung bình trượt 1 tháng) để lọc nhiễu ngày–đêm và làm nổi sóng mùa vụ 1 năm. |
| **`min_periods=1`** | `plot_temperature_trend` | Cho phép tính trung bình ngay từ giờ đầu tiên thay vì đợi đủ 720 giờ (tránh 719 ô `NaN` bị trống ở đầu biểu đồ). |
| **`bins=50`** | `plot_temperature_distribution` | Nhiệt độ trải dài gần `58°C` (`-22.7°C → +35.5°C`), chia `50` cột thì mỗi cột rộng ~`1.16°C` — đủ mịn để thấy hình dạng phân phối mà không bị răng cưa. |
| **`vmin=-1, vmax=1, center=0`** | `plot_correlation_heatmap` | Hệ số tương quan Pearson luôn nằm trong đoạn `[-1, +1]`; khóa cố định thang màu và đặt tâm trắng tại `0` giúp màu sắc phản ánh đúng tương quan thuận/nghịch. |
| **`n=3000`, `random_state=42`** | `plot_scatter_relationships` | Tập Train có `49.090` điểm, vẽ hết sẽ bị chồng kín (overplotting). Lấy mẫu `3.000` điểm đủ thấy rõ đường cong phi tuyến `VPmax`; `random_state=42` cố định seed ngẫu nhiên để tái lập kết quả. |
| **`max_lag=168`**, bước **`24`** | `plot_autocorrelation` | `168 = 7 ngày × 24 giờ` (khớp đúng `seq_len`), bước nhảy `24h` đánh dấu đúng các mốc chu kỳ ngày (`24h, 48h, ..., 168h`). |
| **`acf[lag_h - 1]`** | `plot_autocorrelation` | Mảng `lags` bắt đầu từ `1` (`[1, 2, ..., 168]`), còn chỉ số mảng Python bắt đầu từ `0`, nên độ trễ `24h` nằm tại vị trí `24 - 1 = 23`. |
| **`seq_len=168`, `pred_len=72`** | `plot_sliding_window_example` | Hợp đồng đầu vào/đầu ra cố định của đề tài: **7 ngày quá khứ (`168h`) → 3 ngày dự báo (`72h`)**. |
| **`start_idx=240`** | `run_eda_pipeline` | Bắt đầu lấy mẫu minh họa từ giờ thứ 240 (ngày thứ 10 của tập Train) để đường nhiệt độ tự nhiên, tránh ngay ngày đầu năm mới. |
| **`periods=500`, `14`** | Khối `__main__` | `500` giờ đủ dài hơn `168 + 72 = 240` để chạy test nhanh trong ~1 giây; `14` là đúng số cột đặc trưng khí tượng của Jena Climate. |

---

### Nhóm 2: Các tham số định dạng đồ thị Matplotlib / Seaborn (CÔ HỎI ĐỂ CHECK XEM CÓ HIỂU CODE KHÔNG)

| Tham số | Ý nghĩa & Tại sao chọn giá trị đó? |
| :--- | :--- |
| **`figsize=(W, H)`** *(ví dụ `(14, 4.5)`, `(11, 9)`, `(15, 4.5)`)* | Kích thước hình tính theo **inch** (`rộng × cao`). Biểu đồ chuỗi thời gian để dáng ngang dài `(14, 4.5)` cho dễ nhìn dòng thời gian; Heatmap `14×14` để gần vuông `(11, 9)`; 3 biểu đồ Scatter ghép ngang để rất rộng `(15, 4.5)`. |
| **Mã màu Hex** (`#1f77b4`, `#ff7f0e`, `#2ca02c`, `#d62728`, `#9467bd`) | Bộ mã màu chuẩn `tab10` của Matplotlib lần lượt là: **Xanh dương**, **Cam**, **Xanh lá**, **Đỏ**, **Tím** — có độ tương phản cao, nhìn rõ trên máy chiếu. |
| **`lw`** (*linewidth*: `0.6`, `0.7`, `1.5`, `2.0`) | Độ dày nét vẽ (đơn vị point). Dữ liệu dày 50k–70k giờ để nét mảnh `0.6–0.7` khỏi bết màu; đường xu hướng 30 ngày / ranh giới / dự báo để dày `1.5–2.0` cho nổi bật. |
| **`alpha`** (`0.08`, `0.18`, `0.35`, `0.55`, `0.6`, `0.85`) | Độ đục / trong suốt (từ `0` tàng hình → `1` đục đặc): vùng tô nền Sliding Window để rất nhạt `0.08`; dải `±1 Std` để `0.18`; chấm Scatter 3.000 điểm để `0.35` để thấy chỗ chồng điểm; lưới để `0.6`. |
| **`ls`** (*linestyle*: `"--"`, `":"`, `"-."`) | Kiểu nét đường: `"--"` là **nét gạch đứt** (ranh giới, đường Mean, đường dự báo Y), `":"` là **nét chấm** (lưới grid, mốc `t=0`), `"-."` là **gạch-chấm** (đường Median phân biệt với Mean). |
| **`height_ratios=[0.22, 0.78]`** | Chia tỷ lệ chiều cao trong `plot_temperature_distribution`: Boxplot chỉ là 1 thanh ngang nên chiếm `22%`, Histogram chiếm `78%` (tổng bằng `1.0`). |
| **`fliersize=2` / `3`** | Kích thước chấm tròn nhỏ biểu diễn các điểm ngoại lai (*outliers*) trong `sns.boxplot` để đỡ rối mắt. |
| **`s=10`, `s=35`, `s=70`** (`scatter`) | Kích thước điểm chấm (tính bằng diện tích `pt²`): 3.000 điểm Scatter để nhỏ `s=10`; điểm đỉnh ACF để vừa `s=35`; 2 điểm đỉnh/đáy chu kỳ 24h để lớn `s=70` cho nổi. |
| **`zorder=5`** | Thứ tự lớp hiển thị (*z-index*): đường line mặc định ở lớp `2`, đặt chấm tròn ở lớp `5` để chấm nổi lên trên đè lên đường vẽ. |
| **`fmt=".2f"`**, **`f"{n:,}"`** | Định dạng hiển thị số: `.2f` làm tròn 2 chữ số thập phân (tránh số dài như `0.9501234`); `:,` thêm dấu phẩy hàng nghìn (`49090` → `49,090`). |
| **`acf[lag_h - 1] + 0.03`**, **`ylim(..., 1.1)`** | Đẩy nhãn chữ cao hơn chấm đỉnh `0.03` đơn vị để chữ không đè lên chấm, và mở trần trục Y lên `1.1` (trên mức tối đa `1.0`) để nhãn `"24h (0.92)"` không bị cắt mép trên. |
| **`rotation=25`** | Xoay nhãn tên 6 đặc trưng dưới trục X nghiêng `25°` trong `plot_scaling_comparison` để tên dài (`rho (g/m**3)`, `VPmax (mbar)`) không đè lên nhau. |
| **`dpi=300`**, **`bbox_inches="tight"`** | Độ phân giải ảnh `300` chấm/inch (chuẩn báo cáo/in ấn không bị vỡ khi phóng to) và cắt gọn viền trắng thừa. |
| **`fontsize=8..13`** | Cỡ chữ phân cấp rõ ràng: tiêu đề chính `12–13` (in đậm), tiêu đề khung con `10`, chú giải `9`, nhãn số nhỏ trên đỉnh ACF `8`. |

---

## PHỤ LỤC: 8 CÂU HỎI VẤN ĐÁP CÔ HAY HỎI & BẢNG SỐ LIỆU NHỚ NHANH

1. **Tại sao không chia `train_test_split` ngẫu nhiên?** → Dữ liệu chuỗi thời gian có tự tương quan cao; chia ngẫu nhiên làm mô hình nhìn thấy giờ kế cận ở tương lai (**Time Leakage**).
2. **Cognitive Leakage là gì?** → Nhìn vào Val/Test trong lúc EDA rồi sửa thiết kế mô hình theo nó. Tránh bằng cách chia tập ở Pha 1 trước, chỉ EDA trên 70% Train.
3. **Tại sao chỉ `fit` `StandardScaler` trên Train?** → Không để Mean/Std của tập tương lai lọt vào lúc huấn luyện; kiểm chứng bằng `scaler.n_samples_seen_ == len(train_df) == 49.090`.
4. **Tại sao có `plt.close(fig)` trong `_finalize_plot()`?** → Giải phóng RAM sau khi lưu file PNG, tránh tràn bộ nhớ khi vẽ nhiều ảnh.
5. **Con số `720` trong `rolling(720)` nghĩa là gì?** → `30 ngày × 24 giờ = 720 giờ` (trung bình trượt 1 tháng) để lọc nhiễu ngày-đêm và thấy rõ chu kỳ năm.
6. **Tại sao `plot_scatter_relationships` lấy `sample(3000)`?** → Tránh 49.090 điểm chồng kín nhau (overplotting), cố định `random_state=42` để kết quả tái lập giống nhau.
7. **Biến `wd (deg)` có tương quan `r ≈ 0` trong Heatmap, có nên bỏ không?** → Không vội bỏ vì `wd` là biến góc tuần hoàn (`0° ≡ 360°`) nên tương quan tuyến tính không đo hết bản chất; hướng gió + áp suất chứa tín hiệu phi tuyến báo đổi thời tiết.
8. **Nhóm có làm Feature Engineering không?** → Đã có 5 bước tiền xử lý (lọc `-9999`, resample `1h`, `ffill/bfill`, `StandardScaler` trên Train, Sliding Window `168h→72h`); chưa tạo thêm cột mới để giữ cố định `input_dim = 14` chung cho 3 mô hình (hướng mở rộng: vector gió `Wx, Wy` và `Day/Year sin-cos`).

### Bảng số liệu vàng (liếc lại 1 phút trước khi lên bảng)
- **Dữ liệu gốc:** `420.551` dòng (10 phút) → **Resample `1h`:** `70.129` giờ (`Train: 49.090h | Val: 10.519h | Test: 10.520h`).
- **Lỗi gió `-9999.0`:** `18` điểm ở `wv (m/s)`, `20` điểm ở `max. wv (m/s)`.
- **Nhiệt độ `T (degC)` (Train):** `Mean = 9.11°C` | `Median = 9.20°C` | `Std = 8.65°C` | `Min = -22.65°C` | `Max = 35.52°C` | `Skew = -0.095`.
- **Chu kỳ 24h:** Đáy **`5h (5.99°C)`** | Đỉnh **`15h (12.44°C)`**.
- **Tương quan với `T`:** Thuận mạnh `Tpot (+0.997)`, `VPmax (+0.950)`, `Tdew (+0.902)` | Nghịch mạnh `rho (-0.965)`, `rh (-0.563)`.
- **Tự tương quan ACF:** `24h = 0.922` | `48h = 0.867` | `72h = 0.831` | **`168h = 0.772`**.
- **Số mẫu Sliding Window:** `Train = 48.851` | `Val = 10.280` | `Test = 10.281` | **Tensor:** `X = (64, 168, 14)` → `Y = (64, 72, 1)`.
