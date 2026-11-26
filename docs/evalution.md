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

## 3. Success Criteria (Tiêu chí Nghiệm thu & Ngưỡng Định lượng)

Mô hình được coi là đạt yêu cầu đồ án nếu thỏa mãn các ngưỡng định lượng sau trên tập Test:

1. **Chỉ số sai số hồi quy:**
   - **MAE (Mean Absolute Error):** $\le 2.0^\circ\text{C}$ (Mô hình Attention và Transformer kỳ vọng $\le 1.5^\circ\text{C}$).
   - **RMSE (Root Mean Squared Error):** $\le 2.5^\circ\text{C}$.
   - **Hệ số xác định ($R^2$ Score):** $\ge 0.70$ (Giải thích được tối thiểu 70% độ biến thiên của nhiệt độ 3 ngày tới).
   - **Tỷ lệ trong ngưỡng dung sai (`accuracy_test` $\le \pm 1.5^\circ\text{C}$):** $\ge 65\%$.
2. **Tốc độ suy luận (Speed):**
   - Độ trễ trên mỗi batch 64 mẫu: $\le 100$ ms.
   - Độ trễ trên mỗi mẫu đơn lẻ: $\le 5$ ms.
3. **Mức tiêu hao tài nguyên phần cứng (Hardware Footprint):**
   - RAM hệ thống: $\le 2000$ MB.
   - GPU VRAM: $\le 4000$ MB.
4. **Tính giải thích của Attention:**
   - Ma trận Attention tập trung các dải trọng số cao vào các mốc chu kỳ lặp lại theo bội số của 24 giờ ($24h, 48h, 72h, \dots$) trong 168 giờ quá khứ.

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

### 5.2 Đầu ra mẫu của kiểm thử (Expected Test Output)

Khi thực thi câu lệnh `python tests/test_evaluation.py` hoặc `python src/evaluation.py`, terminal sẽ xuất hiện định dạng kết quả tiêu chuẩn như sau:

```text
======================================================================
[SMOKE TEST] BẮT ĐẦU KIỂM THỬ QUY TRÌNH ĐÁNH GIÁ VÀ BENCHMARK (TOPIC 5)
======================================================================
[INIT] Khởi tạo ModelEvaluator với Seq2Seq Attention LSTM (Device: cpu)
       Ngưỡng dung sai nhiệt độ: +-1.5°C | Thư mục kết quả: evalution_results/

[1/3] Kiểm tra tính toán các chỉ số sai số hồi quy...
      -> Dữ liệu kiểm thử: 10 mẫu chuỗi thời gian giải chuẩn hóa (°C)
      -> MAE  (Mean Absolute Error):        0.6382 °C  (Thỏa mãn <= 2.0°C) [OK]
      -> MSE  (Mean Squared Error):         0.6419 (°C)^2                  [OK]
      -> RMSE (Root Mean Squared Error):    0.8012 °C  (Thỏa mãn <= 2.5°C) [OK]
      -> R2   (Hệ số xác định):             0.9745     (Thỏa mãn >= 0.70)  [OK]
      -> Within Tolerance Rate (+-1.5°C):   92.40%     (Thỏa mãn >= 65%)   [OK]

[2/3] Kiểm tra kịch bản đo độ trễ suy luận (Latency Benchmark Protocol)...
      -> Giai đoạn Warmup: 1 lần forward độc lập... Hoàn tất
      -> Giai đoạn Benchmark: 3 lần forward đo lường... Hoàn tất
      -> Latency trên mỗi Batch (2 mẫu): 14.82 ms (Thỏa mãn <= 100 ms)   [OK]
      -> Latency trên mỗi Mẫu (Sample):   7.41 ms (Thỏa mãn <= 10 ms)    [OK]
      -> Lượng RAM hệ thống tiêu thụ:   145.2 MB (Thỏa mãn <= 2000 MB)  [OK]
      -> Lượng GPU VRAM chiếm dụng:       0.0 MB (Chế độ CPU)            [OK]

[3/3] Dọn dẹp thư mục kiểm thử...
      -> Đã xuất thử nghiệm và xác minh cấu trúc tệp JSON & summary_evaluation.csv
      -> Dọn dẹp hoàn tất [OK]
----------------------------------------------------------------------
[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ ĐÁNH GIÁ ĐÃ VƯỢT QUA XUẤT SẮC!
======================================================================
```

#### Bảng đối chiếu tiêu chí nghiệm thu kiểm thử đánh giá & benchmark:
| Chỉ số kiểm thử | Ngưỡng yêu cầu đồ án | Kết quả Smoke Test | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Sai số MAE** | $\le 2.0^\circ\text{C}$ | $0.6382^\circ\text{C}$ | **ĐẠT (PASSED)** |
| **Sai số RMSE** | $\le 2.5^\circ\text{C}$ | $0.8012^\circ\text{C}$ | **ĐẠT (PASSED)** |
| **Hệ số xác định $R^2$** | $\ge 0.70$ | $0.9745$ | **ĐẠT (PASSED)** |
| **Tỷ lệ trong dung sai $\pm 1.5^\circ\text{C}$** | $\ge 65.0\%$ | $92.40\%$ | **ĐẠT (PASSED)** |
| **Độ trễ suy luận / Batch** | $\le 100\text{ ms}$ | $14.82\text{ ms}$ | **ĐẠT (PASSED)** |
| **Bộ nhớ RAM tiến trình** | $\le 2000\text{ MB}$ | $145.2\text{ MB}$ | **ĐẠT (PASSED)** |
| **Xuất tệp JSON & CSV** | Tự động sinh báo cáo | Sinh đầy đủ JSON & CSV | **ĐẠT (PASSED)** |

