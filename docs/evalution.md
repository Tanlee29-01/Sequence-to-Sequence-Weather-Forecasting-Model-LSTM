# ĐẶC TẢ KỊCH BẢN ĐÁNH GIÁ MÔ HÌNH (EVALUATION SPEC)

---

## 1. Purpose (Mục đích)
- **Mục tiêu:** Thực hiện đánh giá khoa học, toàn diện và khách quan năng lực tổng quát hóa của 3 mô hình học sâu trên tập **Test độc lập (15% cuối cùng)** — tương đương với kỳ thi tốt nghiệp cuối cùng trong quy trình chống rò rỉ dữ liệu.
- **Trọng tâm đánh giá:**
  1. *Độ chính xác trên thang đo vật lý thực tế:* Chuyển đổi toàn bộ kết quả dự báo đã chuẩn hóa về lại độ C ($^\circ\text{C}$) thực tế trước khi tính toán các chỉ số sai số (MAE, MSE, RMSE, $R^2$) và tỷ lệ dự báo chuẩn xác trong ngưỡng dung sai $\pm 1.5^\circ\text{C}$.
  2. *Độ trễ suy luận (Inference Latency):* Đo chính xác thời gian xử lý từ khi nhận đầu vào $168$ giờ đến khi trả ra kết quả $72$ giờ tính bằng mili-giây (`ms/batch` và `ms/sample`).
  3. *Tài nguyên phần cứng (Resource Footprint):* Đo lường lượng RAM hệ thống và GPU VRAM tối đa mà mô hình chiếm dụng khi chạy suy luận.
  4. *Khả năng giải thích (Explainable AI):* Trực quan hóa ma trận Attention Heatmap $72 \times 168$ để đối chiếu với tính chu kỳ ngày/đêm và các đợt biến đổi thời tiết cực đoan.

---

## 2. Input / Output Contract (Hợp đồng Đầu vào & Đầu ra)

### 2.1 Đầu vào của quy trình đánh giá
- **File Checkpoint mô hình:** Hỗ trợ chỉ định phiên bản trực tiếp qua `--model <model_name>_v<N>` (ví dụ `seq2seq_attention_v1`, `seq2seq_attention_v2`, `seq2seq_attention_v3`) hoặc truyền file cụ thể qua `--checkpoint <path>`. Nếu không truyền hậu tố `_vN`, hệ thống tự động chọn phiên bản lớn nhất hiện có và nạp tệp `<base_model>_best.pth` hoặc `*_best.pth` mới nhất trong [`checkpoints/<model_name>_v<N>/`](file:///d:/UTH/Deeplearning/course_work_topic_5/checkpoints/). Đồng thời tự động đọc `config.json` của phiên bản đó để tái tạo đúng kiến trúc mạng.
  - Ví dụ: `checkpoints/seq2seq_attention_v1/seq2seq_attention_20261121-080327_best.pth` hoặc `checkpoints/seq2seq_attention_v2/seq2seq_attention_best.pth`
- **Tập dữ liệu kiểm thử:** `test_loader` từ `JenaDataModule` với `batch_size = 64` (không shuffle, giữ nguyên thứ tự dòng thời gian).
- **Hàm giải chuẩn hóa:** Phương thức `inverse_transform_target` của `JenaDataModule` sử dụng thông số $\mu_{train}$ và $\sigma_{train}$ từ `StandardScaler`.
- **Ngưỡng dung sai nhiệt độ:** `tolerance_degC = 1.5` ($\pm 1.5^\circ\text{C}$).

### 2.2 Đầu ra kết quả đánh giá (Evaluation Outputs)
- **Thư mục lưu trữ theo ngày thực nghiệm:** `evalution_results/evaluation_YYYYMMDD/` (ví dụ: `evalution_results/evaluation_20261126/`).
- **Các tệp kết quả được xuất theo từng version (`<base_model>_v<N>`, tránh ghi đè giữa v1/v2/vN):**
  1. `<model_name>_v<N>_metrics.json`: Báo cáo chi tiết toàn bộ các chỉ số của từng phiên bản dạng JSON.
  2. `summary_evaluation.csv`: Bảng tổng hợp đối đầu so sánh các mô hình và các phiên bản trên cùng các tiêu chí.
  3. `<model_name>_v<N>_sample_<idx>_forecast.png`: Biểu đồ đường so sánh giữa 72 giờ thực tế vs dự báo kèm dải dung sai $\pm 1.5^\circ\text{C}$.
  4. `<model_name>_v<N>_attention_heatmap.png`: Bản đồ nhiệt ma trận Attention Heatmap $72 \times 168$.

### 2.3 Ghi nhận kết quả vào TensorBoard (`runs/`)
- Nếu chạy `--mode all` sẽ ghi chung vào phiên vừa train; nếu chạy `--mode evaluate` riêng (không truyền `--run_name`) thì `main.py` tự gắn vào thư mục `runs/<model_name>_v<N>*` tương ứng của phiên bản đó:
  - `Loss/Test`: Sai số MSE trên toàn bộ tập Test sau khi hoàn tất đánh giá (ghi qua `logger.log_test_metrics()`).
  - `Accuracy/Test`: Tỷ lệ % số giờ dự đoán nằm trong khoảng lệch $\le \pm 1.5^\circ\text{C}$ trên tập Test.
  - `HParams`: Ghi siêu tham số huấn luyện kèm `hparam/MAE`, `hparam/RMSE`, `hparam/R2`, `hparam/Acc_1_5C` qua `logger.log_hparams(..., run_name=".")` để đối chiếu giữa các lần chạy trên giao diện TensorBoard.

---

## 3. Target Benchmarks (Các Chỉ Tiêu Tham Chiếu Kỳ Vọng)

Dưới đây là bộ chỉ tiêu tham chiếu kỳ vọng để đối chiếu và đánh giá chất lượng mô hình trên tập Test (trong đó các ngưỡng độ trễ áp dụng cho cấu hình tăng tốc GPU):

1. **Chỉ số sai số hồi quy:**
   - **MAE (Mean Absolute Error):** Tham chiếu $\le 2.0^\circ\text{C}$ (khung ngắn hạn 24h đầu kỳ vọng $\le 1.8^\circ\text{C}$).
   - **RMSE (Root Mean Squared Error):** Tham chiếu $\le 2.5^\circ\text{C}$.
   - **Hệ số xác định ($R^2$ Score):** $\ge 0.70$ (Giải thích được tối thiểu 70% độ biến thiên của nhiệt độ 3 ngày tới).
   - **Tỷ lệ trong ngưỡng dung sai (`accuracy_test` $\le \pm 1.5^\circ\text{C}$):** Tham chiếu $\ge 65\%$ (cao nhất ở 24h đầu và giảm dần theo độ dài dự báo).
2. **Tốc độ suy luận (Speed — mục tiêu trên GPU CUDA):**
   - Độ trễ trên mỗi batch 64 mẫu: $\le 100$ ms (trên GPU).
   - Độ trễ trên mỗi mẫu đơn lẻ: $\le 5$ ms (trên GPU, hoặc tính theo `amortized_per_sample_ms` khi chạy batch).
3. **Mức tiêu hao tài nguyên phần cứng (Hardware Footprint):**
   - RAM hệ thống: $\le 2000$ MB.
   - GPU VRAM: $\le 4000$ MB.
4. **Tính giải thích của Attention:**
   - Ma trận Attention thể hiện trọng số chú ý phân hóa theo chu kỳ ngày/đêm (bội số của 24 giờ: $24h, 48h, 72h, \dots$) trong 168 giờ quá khứ.

---

## 4. How to do (Quy trình Hiện thực Đánh giá Chi Tiết)

### 4.1 Công thức toán học các chỉ số đánh giá
- **MAE:**
  $$\text{MAE} = \frac{1}{N \times 72} \sum_{i=1}^{N} \sum_{t=1}^{72} |y_{i, t} - \hat{y}_{i, t}|$$
- **RMSE:**
  $$\text{RMSE} = \sqrt{\frac{1}{N \times 72} \sum_{i=1}^{N} \sum_{t=1}^{72} (y_{i, t} - \hat{y}_{i, t})^2}$$
- **Hệ số xác định $R^2$:**
  $$R^2 = 1 - \frac{\sum_{i, t} (y_{i, t} - \hat{y}_{i, t})^2}{\sum_{i, t} (y_{i, t} - \bar{y})^2}$$
- **Tỷ lệ dung sai Accuracy ($\le \pm 1.5^\circ\text{C}$):**
  $$\text{Accuracy} = \frac{1}{N \times 72} \sum_{i, t} \mathbb{I}(|y_{i, t} - \hat{y}_{i, t}| \le 1.5) \times 100\%$$

### 4.2 Kịch bản đo độ trễ suy luận chuẩn hóa (Latency Benchmark Protocol)
Để kết quả đo đạc chính xác tuyệt đối và không bị ảnh hưởng bởi quá trình khởi tạo đồ họa hay phân mảnh bộ nhớ:
1. **Warmup:** Chạy thử nghiệm $10$ lần forward đầu tiên và bỏ qua thời gian đo.
2. **Đồng bộ hóa GPU:** Gọi `torch.cuda.synchronize()` trước khi bấm giờ và ngay sau khi kết thúc vòng lặp đo.
3. **Đo thời gian:** Sử dụng hàm `time.perf_counter()` đo trung bình qua $50$ lần lặp forward độc lập.

```python
with torch.no_grad():
    # 1. Warmup
    for _ in range(10):
        _ = model(sample_x, trg=None, teacher_forcing_ratio=0.0)
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 2. Benchmark
    start_t = time.perf_counter()
    for _ in range(50):
        _ = model(sample_x, trg=None, teacher_forcing_ratio=0.0)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_t = time.perf_counter()

latency_batch_ms = ((end_t - start_t) / 50) * 1000.0
```

### 4.3 Trực quan hóa và Lưu trữ Báo cáo
1. Lưu kết quả chỉ số thành file JSON:
   ```python
   with open(os.path.join(run_dir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
       json.dump(full_results, f, indent=4, ensure_ascii=False)
   ```
2. Cập nhật bảng CSV so sánh tổng hợp `summary_evaluation.csv`:
   ```python
   df = pd.DataFrame([full_results])
   df.to_csv(os.path.join(run_dir, "summary_evaluation.csv"), index=False)
   ```
3. Vẽ biểu đồ đường 72 giờ và biểu đồ nhiệt Attention Heatmap với Matplotlib, xuất file với độ phân giải cao (`dpi=300`).

---

## 5. Smoke Test & Verification (Hướng Dẫn Kiểm Thử & Đầu Ra Mẫu)

### 5.1 Hướng dẫn kiểm thử nhanh (Smoke Test Instructions)

Mục đích của bài kiểm thử khói quy trình đánh giá là xác minh tính đúng đắn về mặt toán học của bộ chỉ số hồi quy, quy trình đo độ trễ chuẩn hóa (Latency Benchmark), và khả năng xuất báo cáo JSON/CSV mà không cần phải chạy qua toàn bộ tập Test 10.514 giờ. Quá trình kiểm thử diễn ra trong vòng chưa đầy **2 giây**.

#### Các tiêu chí kiểm thử bắt buộc:
1. **Độ chính xác của các công thức chỉ số sai số:**
   - MAE, MSE, RMSE không âm; hệ số $R^2 \le 1.0$.
   - Tỷ lệ dự báo chuẩn xác $\text{Within\_Tolerance\_Rate\_Pct} \in [0.0, 100.0]\%$ đối với dung sai $\pm 1.5^\circ\text{C}$.
2. **Kịch bản đo độ trễ suy luận chuẩn hóa (Latency Benchmark Protocol):**
   - Khởi động bộ đệm GPU (Warmup) độc lập tối thiểu $2$ lần trước khi bấm giờ.
   - Sử dụng `time.perf_counter()` đo trung bình qua nhiều lần lặp, trả về độ trễ hợp lệ $> 0$ mili-giây (`ms/batch` và `ms/sample`).
3. **Giám sát dấu chân phần cứng (Resource Footprint):** Đo lường và báo cáo mức RAM (MB) tiến trình đang tiêu thụ và GPU VRAM (MB).
4. **Đóng gói báo cáo khoa học tự động:** Tạo mới hoặc cập nhật tệp JSON `<model_name>_metrics.json` và bảng so sánh `summary_evaluation.csv` mà không làm hỏng cấu trúc dữ liệu cũ.

#### Các câu lệnh thực thi kiểm thử:

```bash
# Cách 1: Chạy trực tiếp khối kiểm thử trong src/evaluation.py
python src/evaluation.py

# Cách 2: Chạy qua bộ kiểm thử tự động trong thư mục tests/
python tests/test_evaluation.py

# Cách 3: Chạy bằng pytest với cờ hiển thị chi tiết
pytest tests/test_evaluation.py -v
```

#### Mã nguồn kịch bản Smoke Test độc lập:
```python
import numpy as np
import torch
from src.model_builder import ModelBuilder_Seq2SeqAttention
from src.evaluation import ModelEvaluator

device = torch.device("cpu")
model = ModelBuilder_Seq2SeqAttention(input_dim=14, hidden_dim=16, num_layers=1)
model.eval()

evaluator = ModelEvaluator(
    model=model,
    device=device,
    tolerance_degC=1.5,
    results_dir="evalution_results/smoke_eval",
)

# 1. Kiểm tra tính toán metrics
mock_true = np.random.randn(20, 72, 1) * 5 + 15
mock_pred = mock_true + np.random.randn(20, 72, 1) * 0.8
res = evaluator.compute_metrics(mock_true, mock_pred)
assert "MAE" in res and "RMSE" in res and "R2" in res

# 2. Kiểm tra đo độ trễ
x_sample = torch.randn(2, 168, 14)
perf = evaluator.measure_latency_and_resources(x_sample, num_runs=3, warmup=1)
assert perf["Latency_ms_per_batch"] > 0

print("Smoke Test for ModelEvaluator PASSED!")
```

---

### 5.2 Đầu ra thực tế của kiểm thử (Actual Smoke Test Output)

Khi thực thi câu lệnh `python tests/test_evaluation.py`, terminal xuất ra kết quả thực tế như sau:

```text
======================================================================
[SMOKE TEST] BẮT ĐẦU KIỂM THỬ QUY TRÌNH ĐÁNH GIÁ VÀ BENCHMARK (TOPIC 5)
======================================================================
[1/3] Kiểm tra tính toán các chỉ số sai số tổng thể và theo horizon...
      -> MAE=0.6391 | RMSE=0.8128 | R2=0.9722 | 1-24h=0.6085 | 25-48h=0.6172 | 49-72h=0.6915 [OK]
[2/3] Kiểm tra kịch bản đánh giá & đo độ trễ suy luận (Latency Benchmark Protocol)...
      -> Batch: 16.1177 ms | Amortized: 8.0589 ms | Single: 13.0526 ms | RAM: 352.46 MB [OK]
[3/3] Xác minh cô lập thư mục tạm và tự động dọn dẹp...
      -> Dọn dẹp hoàn tất (không ghi rác vào evalution_results/) [OK]
----------------------------------------------------------------------
[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ ĐÁNH GIÁ ĐÃ VƯỢT QUA XUẤT SẮC!
======================================================================
```

#### Bảng đối chiếu chỉ tiêu tham chiếu kiểm thử đánh giá & benchmark:
| Chỉ số kiểm thử | Chỉ tiêu tham chiếu | Kết quả Smoke Test | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Sai số MAE** | $\le 2.0^\circ\text{C}$ | $0.6391^\circ\text{C}$ | **ĐẠT (PASSED)** |
| **Sai số RMSE** | $\le 2.5^\circ\text{C}$ | $0.8128^\circ\text{C}$ | **ĐẠT (PASSED)** |
| **Hệ số xác định $R^2$** | $\ge 0.70$ | $0.9722$ | **ĐẠT (PASSED)** |
| **Độ trễ suy luận / Batch** | $\le 100\text{ ms}$ | $16.1177\text{ ms}$ | **ĐẠT (PASSED)** |
| **Bộ nhớ RAM tiến trình** | $\le 2000\text{ MB}$ | $352.46\text{ MB}$ | **ĐẠT (PASSED)** |
| **Cô lập & dọn dẹp tệp tạm** | Không ghi rác `evalution_results/` | Tự động dọn dẹp hoàn tất | **ĐẠT (PASSED)** |

---

## 6. Báo Cáo Kết Quả Thực Nghiệm Thực Tế (`evaluation_20261127`)

Phần này tổng hợp, đối chiếu và nhận xét kết quả đánh giá thực tế trên **tập Test độc lập (`10.281` mẫu)** được lưu tại [`evalution_results/evaluation_20261127/`](../evalution_results/evaluation_20261127/). Trong đợt chạy mới nhất này, hệ thống đánh giá hai phiên bản của mô hình **Seq2Seq Attention LSTM** (`seq2seq_attention_v1` và `seq2seq_attention_v2`) trên thiết bị `cpu` (`batch_size = 64`, `warmup_runs = 10`, `benchmark_runs = 50`).

### 6.1 Khác biệt cấu hình giữa hai phiên bản
Theo tệp cấu hình lưu kèm tại `checkpoints/seq2seq_attention_v1/config.json` và `checkpoints/seq2seq_attention_v2/config.json`, cả hai phiên bản dùng chung kiến trúc (`input_dim=14`, `hidden_dim=128`, `num_layers=2`, `dropout=0.2`, `seq_len=168`, `pred_len=72`, `epochs=50`, `early_stopping_patience=7`, `learning_rate=0.001`), chỉ khác nhau ở chiến lược điều chỉnh tốc độ học:
- **`seq2seq_attention_v1`:** `scheduler_type = "none"` (giữ nguyên learning rate cố định).
- **`seq2seq_attention_v2`:** `scheduler_type = "ReduceLROnPlateau"` (`lr_patience = 3`, `lr_factor = 0.5`, `lr_min = 1e-05`).

### 6.2 Tổng hợp chỉ số định lượng trên toàn tập Test (10.281 mẫu)

| Chỉ số đánh giá | Chỉ tiêu tham chiếu (Mục 3) | `seq2seq_attention_v1` | `seq2seq_attention_v2` | Chênh lệch (`v2` vs `v1`) | Nhận xét (`v2`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MAE (°C)** | $\le 2.0^\circ\text{C}$ | `2.6807` | **`2.5503`** | `-0.1304°C` (tốt hơn) | **Đạt chuẩn ở 24h đầu (`1.76°C`)**; tăng dần ở ngày 2–3 |
| **MSE (°C²)** | Càng nhỏ càng tốt | `11.9590` | **`10.8977`** | `-1.0613` (tốt hơn) | Cải thiện rõ rệt so với `v1` |
| **RMSE (°C)** | $\le 2.5^\circ\text{C}$ | `3.4582` | **`3.3012`** | `-0.1570°C` (tốt hơn) | **Đạt chuẩn ở 24h đầu (`2.32°C`)**; tích lũy ở ngày 2–3 |
| **Hệ số $R^2$** | $\ge 0.70$ | `0.8040` | **`0.8214`** | `+0.0174` (tốt hơn) | **ĐẠT kỳ vọng** (giải thích `82.14%` biến thiên; 24h đầu đạt `0.9117`) |
| **Tỷ lệ dung sai $\le \pm 1.5^\circ\text{C}$** | $\ge 65.0\%$ | `37.26%` | **`39.13%`** | `+1.87%` (tốt hơn) | Cải thiện so với `v1`; khung 24h đầu đạt `53.54%` |
| **Latency / Batch (64 mẫu, CPU)** | $\le 100\text{ ms}$ (GPU) | **`228.13 ms`** | `237.58 ms` | `+9.45 ms` | Đo trên **CPU** (mục tiêu `100 ms` đặt cho GPU) |
| **Latency / Single Sample (CPU)** | $\le 5\text{ ms}$ (GPU) | `14.68 ms` | **`14.60 ms`** | `-0.08 ms` | Đo đơn lẻ trên **CPU** (chưa chạy trên GPU) |
| **Latency Amortized / Sample (CPU)** | $\le 5\text{ ms}$ | **`3.56 ms`** | `3.71 ms` | `+0.15 ms` | **Đạt tốc độ cao** khi xử lý theo batch (`3.71 ms/mẫu`) |
| **RAM tiến trình (RSS)** | $\le 2000\text{ MB}$ | **`585.36 MB`** | `585.84 MB` | `+0.48 MB` | **ĐẠT kỳ vọng** (tiết kiệm ~70% so với ngưỡng) |
| **Peak GPU VRAM** | $\le 4000\text{ MB}$ | `0.0 MB` (CPU) | `0.0 MB` (CPU) | `—` | Chạy chế độ CPU |

### 6.3 Phân rã sai số theo từng khung thời gian dự báo (1–24h, 25–48h, 49–72h)

| Khung dự báo | Chỉ số | `seq2seq_attention_v1` | `seq2seq_attention_v2` | Nhận xét |
| :--- | :--- | :---: | :---: | :--- |
| **Ngày 1 (`1–24h`)** | `MAE` / `RMSE` / `R2` / `Acc ±1.5°C` | `1.8597°C` / `2.4457°C` / `0.9016` / `51.86%` | **`1.7629°C` / `2.3171°C` / `0.9117` / `53.54%`** | Đạt chuẩn `MAE <= 2.0°C`, `RMSE <= 2.5°C`, `R2 >= 0.70` |
| **Ngày 2 (`25–48h`)** | `MAE` / `RMSE` / `R2` / `Acc ±1.5°C` | `2.8524°C` / `3.5784°C` / `0.7900` / `32.61%` | **`2.7160°C` / `3.4168°C` / `0.8086` / `34.63%`** | Sai số tăng rõ rệt so với ngày 1 |
| **Ngày 3 (`49–72h`)** | `MAE` / `RMSE` / `R2` / `Acc ±1.5°C` | `3.3299°C` / `4.1342°C` / `0.7212` / `27.29%` | **`3.1721°C` / `3.9559°C` / `0.7448` / `29.22%`** | Dự báo xa nhất nên độ bất định lớn nhất |

> **Nhận xét xu hướng sai số:** Cả hai phiên bản đều thể hiện rõ quy luật tự nhiên của dự báo chuỗi thời gian đa bước (Autoregressive Multi-step Forecasting): sai số tích lũy dần theo độ dài dự báo. Ở khung ngắn hạn **24 giờ đầu tiên**, mô hình hoạt động rất tốt ($R^2 > 0.91$, `MAE = 1.76°C` đạt ngưỡng $\le 2.0^\circ\text{C}$), nhưng khi kéo dài sang **ngày 2 và ngày 3**, sai số `MAE` tăng lên `2.72°C` và `3.17°C`, kéo chỉ số trung bình chung cả 72 giờ vượt ngưỡng `2.0°C`.

### 6.4 Phân tích trực quan từ biểu đồ thực nghiệm

1. **Biểu đồ dự báo chuỗi 72 giờ (`*_sample_0_forecast.png`):**
   - Trong [biểu đồ dự báo của `v1`](../evalution_results/evaluation_20261127/seq2seq_attention_v1_sample_0_forecast.png) và [biểu đồ dự báo của `v2`](../evalution_results/evaluation_20261127/seq2seq_attention_v2_sample_0_forecast.png), đường dự báo (nét đứt đỏ) bắt kịp tốt **chu kỳ ngày – đêm 24 giờ** (lên đỉnh vào khoảng giờ +22, +47, +69 và xuống đáy vào khoảng giờ +14, +37).
   - **Điểm cải thiện của `v2` so với `v1`:** Ở đỉnh ngày 1 (~+22h), đỉnh ngày 2 (~+47h) và đỉnh ngày 3 (~+69h), `v2` bám sát biên độ thực tế hơn (ví dụ tại đỉnh +47h, `v1` vọt lên gần `15°C` so với thực tế `~10.3°C`, trong khi `v2` giảm độ vọt lố xuống còn `~13.1°C`).
   - **Hạn chế còn tồn tại:** Tại các điểm đáy nhiệt độ thấp ban đêm (quanh +14h và +37h), cả `v1` và `v2` đều có xu hướng dự báo cao hơn thực tế (nhiệt độ thực xuống `~1–2.7°C` nhưng dự báo chỉ xuống `~5.7–6.4°C`), làm phần lớn các giờ ban đêm vượt ra ngoài dải dung sai tô mờ $\pm 1.5^\circ\text{C}$.

2. **Bản đồ nhiệt Attention (`*_attention_heatmap.png`):**
   - Ở [`seq2seq_attention_v1_attention_heatmap.png`](../evalution_results/evaluation_20261127/seq2seq_attention_v1_attention_heatmap.png), dải trọng số chú ý nằm trong khoảng rất hẹp (`0.0054 – 0.0066`, tiệm cận mức chia đều $1/168 \approx 0.00595$) và các dải màu gần như **thẳng đứng đồng nhất** từ bước tương lai `1` đến `72`. Điều này cho thấy cơ chế Attention ở `v1` gần như giữ một phân phối chú ý tĩnh lên một vài mốc quá khứ cố định (quanh giờ `~68`, `~40`, `~86`), chưa linh hoạt thay đổi theo từng bước giải mã.
   - Ở [`seq2seq_attention_v2_attention_heatmap.png`](../evalution_results/evaluation_20261127/seq2seq_attention_v2_attention_heatmap.png), biên độ phân hóa trọng số rộng gấp đôi (`0.0040 – 0.0080`) và bắt đầu hình thành các **nhịp sáng – tối biến thiên theo trục dọc** (khi bước dự báo tương lai dịch chuyển theo chu kỳ, cường độ chú ý tại các cột quá khứ cũng tăng giảm theo nhịp). Tuy nhiên, bản đồ nhiệt vẫn chưa tạo thành các đường chéo sắc nét cách đều đúng 24 giờ như kỳ vọng lý tưởng ở Mục 3.

### 6.5 Kết luận và đề xuất cải tiến cho đợt chạy tiếp theo
- **Kết luận chính:** Việc bật **`ReduceLROnPlateau` ở `seq2seq_attention_v2`** đã mang lại cải thiện nhất quán trên **toàn bộ các chỉ số chất lượng** (`MAE` giảm `4.86%`, `RMSE` giảm `4.54%`, $R^2$ tăng từ `0.8040` lên `0.8214`, và tỷ lệ dung sai tăng ở cả 3 ngày dự báo). Mô hình đạt tiêu chuẩn ở khả năng giải thích xu hướng ($R^2 = 0.8214 \ge 0.70$) và bộ nhớ RAM gọn nhẹ (`~585.8 MB <= 2000 MB`).
- **Khoảng trống cần hoàn thiện:**
  1. Thư mục `evaluation_20261127` hiện mới chạy đánh giá cho 2 phiên bản của `seq2seq_attention`, chưa bao gồm `seq2seq_lstm` và `transformer` để lập bảng đối đầu đầy đủ giữa 3 kiến trúc.
  2. Độ trễ suy luận (`~237 ms/batch`, `~14.6 ms/sample`) hiện đo trên **CPU**; cần đo lại trên **GPU (CUDA)** để đánh giá đúng năng lực đáp ứng ngưỡng $\le 100\text{ ms/batch}$ và $\le 5\text{ ms/sample}$.
  3. Để kéo `MAE` tổng thể xuống $\le 2.0^\circ\text{C}$ và cải thiện tỷ lệ $\pm 1.5^\circ\text{C}$ ở ngày 2–3 (nhất là các đáy nhiệt độ ban đêm), có thể thử nghiệm: (i) dùng `CosineAnnealingLR` hoặc điều chỉnh `teacher_forcing_decay`, (ii) tăng `hidden_dim` / bổ sung đặc trưng thời gian tuần hoàn (`sin/cos` của giờ trong ngày và ngày trong năm), hoặc (iii) đối chiếu với `transformer` có khả năng chú ý trực tiếp không bị suy giảm theo bước đệ quy.


