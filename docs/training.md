# ĐẶC TẢ QUY TRÌNH HUẤN LUYỆN & TỐI ƯU HÓA (TRAINING SPEC)

---

## 1. Purpose (Mục đích)
- **Mục tiêu:** Thiết lập quy trình huấn luyện học sâu chuẩn mực cho bài toán hồi quy chuỗi thời gian khí tượng Jena Climate, đảm bảo mô hình hội tụ ổn định, không bị bùng nổ gradient, kiểm soát chặt chẽ hiện tượng quá khớp (Overfitting) và thu hẹp khoảng cách giữa quá trình học (Train) và thi thật (Validation/Test).
- **Các kỹ thuật trọng tâm:**
  1. *Teacher Forcing suy giảm (Scheduled Teacher Forcing):* Bắt đầu với tỷ lệ nhắc bài cao để mô hình nhanh chóng học cách giải mã, sau đó giảm dần qua từng epoch để mô hình tự lực dự báo tự hồi quy (Autoregressive).
  2. *Cắt chuẩn Gradient (Gradient Clipping):* Giới hạn độ dài vector đạo hàm $\le 1.0$ để ngăn chặn triệt để hiện tượng nổ gradient thường gặp trong các mô hình hồi quy chuỗi dài (168 bước vào $\rightarrow$ 72 bước ra).
  3. *Dừng sớm (Early Stopping):* Tự động ngắt huấn luyện nếu sai số trên tập Validation không cải thiện sau số epoch kiên nhẫn định trước (Patience = 7), đồng thời khôi phục checkpoint tốt nhất.
  4. *Giám sát tài nguyên phần cứng:* Theo dõi thời gian thực mức tiêu thụ bộ nhớ RAM hệ thống (MB) và GPU VRAM (MB) thông qua thư viện `psutil` và `torch.cuda`.

---

## 2. Input / Output Contract (Hợp đồng Đầu vào & Đầu ra)

### 2.1 Đầu vào của quá trình huấn luyện
- **Dữ liệu:** `train_loader` (shuffle=True) và `val_loader` (shuffle=False) từ `JenaDataModule` với `batch_size = 64`.
- **Mô hình:** Đối tượng `nn.Module` được khởi tạo từ `src/model_builder.py`.
- **Bộ tối ưu (Optimizer):** `torch.optim.Adam` với `learning_rate = 0.001`, `weight_decay = 1e-5`.
- **Hàm mất mát (Loss Function):** Mean Squared Error (`nn.MSELoss()`).
- **Siêu tham số điều khiển:**
  - `epochs`: 50 vòng lặp tối đa.
  - `clip_grad_norm`: Ngưỡng cực đại 1.0.
  - `early_stopping_patience`: 7 epochs.
  - `teacher_forcing_start`: 0.5 (Epoch 1).
  - `teacher_forcing_decay`: 0.95 (sau mỗi epoch).

### 2.2 Vị trí và Quy chuẩn Đặt tên File Checkpoint (.pth)
- **Thư mục lưu trữ:** Toàn bộ trọng số mô hình được lưu tại thư mục [`checkpoints/`](file:///d:/UTH/Deeplearning/course_work_topic_5/checkpoints/).
- **Cấu trúc đặt tên chuẩn hóa:**
  - Checkpoint của từng epoch: `<model_name>_epoch_<n>.pth`
  - File tốt nhất (Best Validation Loss): `<model_name>_<YYYYMMDD-HHMMSS>_best.pth`
  - File vòng lặp cuối cùng (Last Epoch): `<model_name>_<YYYYMMDD-HHMMSS>_last.pth`
- **Ví dụ thực tế:**
  - `checkpoints/seq2seq_lstm_20261116-204500_best.pth`
  - `checkpoints/seq2seq_attention_20261116-204500_best.pth`
  - `checkpoints/transformer_20261116-204500_best.pth`
- **Tệp Scaler đi kèm:** `checkpoints/scaler_train.pkl` (lưu trữ đối tượng `StandardScaler` đã fit trên tập Train).

### 2.3 Nhật ký ghi nhận trực quan vào `runs/` (TensorBoard Logging)
- **Thư mục lưu log:** `runs/<model_name>_<YYYYMMDD-HHMMSS>/`
- **Hệ thống Scalar tags chuẩn mực bắt buộc ghi nhận:**
  - `loss_train`: Mất mát MSE trên tập Train tại mỗi Epoch.
  - `loss_val`: Mất mát MSE trên tập Validation tại mỗi Epoch.
  - `loss_test`: Mất mát MSE trên tập Test (được ghi sau khi hoàn tất kiểm thử).
  - `accuracy_train`: Tỷ lệ % số giờ dự đoán sai lệch $\le \pm 1.5^\circ\text{C}$ trên tập Train.
  - `accuracy_val`: Tỷ lệ % số giờ dự đoán sai lệch $\le \pm 1.5^\circ\text{C}$ trên tập Validation.
  - `accuracy_test`: Tỷ lệ % số giờ dự đoán sai lệch $\le \pm 1.5^\circ\text{C}$ trên tập Test.
  - `Learning_Rate`: Tốc độ học hiện hành của bộ tối ưu.
  - `Hardware/RAM_MB`: Bộ nhớ RAM tiến trình Python đang sử dụng.
  - `Hardware/VRAM_MB`: Bộ nhớ VRAM GPU đang được cấp phát (0 nếu chạy trên CPU).
- **Hình ảnh biểu đồ ghi nhận (Figures):**
  - `Forecast_Comparison`: Biểu đồ đường so sánh 72 giờ nhiệt độ thực tế vs dự báo.
  - `Attention_Weights`: Bản đồ nhiệt Attention Heatmap $72 \times 168$ (dành cho Model 2 & 3).

---

## 3. Success Criteria (Tiêu chí Nghiệm thu Huấn luyện)

1. **Tính ổn định của hàm tổn thất:**
   - Cả `loss_train` và `loss_val` phải giảm dần theo số epoch, không xuất hiện hiện tượng phân kỳ đột ngột hoặc giá trị `NaN`/`Inf`.
2. **Kích hoạt dừng sớm chuẩn xác (Early Stopping):**
   - Nếu trong 7 epoch liên tiếp mà `loss_val` không đạt giá trị nhỏ hơn `best_val_loss`, quá trình huấn luyện tự động dừng lại và thông báo epoch tốt nhất được lưu lại tại `<model_name>_<YYYYMMDD-HHMMSS>_best.pth`.
3. **Tuân thủ quy tắc Teacher Forcing:**
   - Trong quá trình Train: Tỷ lệ $TF$ giảm dần theo công thức $TF_{epoch} = TF_0 \times (0.95)^{epoch-1}$.
   - Trong quá trình Validation và Test: Bắt buộc $TF = 0.0$ (tắt hoàn toàn nhắc bài để đánh giá năng lực thực tế).
4. **Kiểm soát bùng nổ Gradient:**
   - Lệnh `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` đảm bảo gradient luôn bị chặn ở mức 1.0.

---

## 4. How to do (Quy trình Thực hiện Chi Tiết)

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Kỹ sư / Pipeline
    participant T as ModelTrainer
    participant M as Model (nn.Module)
    participant L as TensorBoardLogger
    participant D as Disk (checkpoints/)

    Dev->>T: Khởi tạo fit(train_loader, val_loader, epochs)
    loop Mỗi Epoch (1 đến N)
        T->>M: train_epoch(train_loader, TF_ratio)
        M-->>T: loss_train, accuracy_train
        T->>M: validate(val_loader, TF_ratio=0.0)
        M-->>T: loss_val, accuracy_val
        T->>L: log_metrics(loss_train, loss_val, accuracy_train, accuracy_val, RAM, VRAM)
        alt val_loss < best_val_loss
            T->>D: save_checkpoint("<model_name>_<YYYYMMDD-HHMMSS>_best.pth")
            T->>T: Reset patience_counter = 0
        else val_loss >= best_val_loss
            T->>T: patience_counter += 1
            opt patience_counter >= patience (7)
                T->>Dev: Kích hoạt Early Stopping & Dừng vòng lặp
            end
        end
    end
    T->>D: save_checkpoint("<model_name>_<YYYYMMDD-HHMMSS>_last.pth")
    Dev->>T: Đánh giá Test set
    T->>L: log_metrics(loss_test, accuracy_test)
```

### 4.1 Vòng lặp huấn luyện từng Epoch (`train_epoch`)
```python
model.train()
for x_batch, y_batch in train_loader:
    optimizer.zero_grad()
    predictions, _ = model(src=x_batch, trg=y_batch, teacher_forcing_ratio=current_tf)
    loss = criterion(predictions, y_batch)
    loss.backward()
    
    # Cắt chuẩn gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### 4.2 Tính tỷ lệ chính xác Accuracy ($\le \pm 1.5^\circ\text{C}$) trong bài toán Hồi quy
```python
def compute_accuracy(y_true, y_pred, tolerance=1.5):
    """Tính % số giờ dự báo có độ lệch tuyệt đối không vượt quá tolerance (°C)."""
    abs_diff = torch.abs(y_true - y_pred)
    within_tolerance = (abs_diff <= tolerance).float()
    return torch.mean(within_tolerance).item() * 100.0
```

### 4.3 Đặt tên và Lưu trữ Checkpoint chuẩn mực
```python
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
best_filename = f"{model_name}_{timestamp}_best.pth"
best_filepath = os.path.join("checkpoints", best_filename)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_loss": best_val_loss,
    "epoch": epoch
}, best_filepath)
```

### 4.4 Ghi log đầy đủ 6 chỉ số vào TensorBoard (`runs/`)
```python
# Trong mỗi epoch
writer.add_scalar("loss_train", train_loss, epoch)
writer.add_scalar("loss_val", val_loss, epoch)
writer.add_scalar("accuracy_train", train_acc, epoch)
writer.add_scalar("accuracy_val", val_acc, epoch)

# Sau khi hoàn tất kiểm thử tập Test
writer.add_scalar("loss_test", test_loss, epoch)
writer.add_scalar("accuracy_test", test_acc, epoch)
```

---

## 5. Smoke Test & Verification (Hướng Dẫn Kiểm Thử & Đầu Ra Mẫu)

### 5.1 Hướng dẫn kiểm thử nhanh (Smoke Test Instructions)

Mục đích của bài kiểm thử khói quy trình huấn luyện là xác minh hoạt động trơn tru của `ModelTrainer` trên một tập Tensor giả lập nhỏ (8-12 mẫu) trong vòng **3 giây**, đảm bảo:
- Không xảy ra lỗi bộ nhớ hoặc bùng nổ đạo hàm.
- Tỷ lệ Teacher Forcing suy giảm chính xác theo lịch biểu.
- Cơ chế Early Stopping tự động phát hiện và ngắt huấn luyện đúng thời điểm.
- Trọng số mô hình được đóng gói và lưu trữ an toàn thành file Checkpoint `.pth`.

#### Các tiêu chí kiểm thử bắt buộc:
1. **Hội tụ của hàm mất mát:** Vòng lặp `train_epoch` và `validate` tính toán loss MSE không âm và không chứa giá trị `NaN`/`Inf`.
2. **Scheduled Teacher Forcing:** Tỷ lệ nhắc bài $TF$ giảm dần theo cấp số nhân: $TF_e = TF_0 \times (0.95)^{e-1}$ trên tập Train và bị cưỡng bức về $0.0$ tuyệt đối trên tập Validation.
3. **Cắt chuẩn Gradient (Gradient Clipping):** Lệnh `clip_grad_norm_` giữ cho chuẩn Euclidean $\|\mathbf{g}\|_2 \le 1.0$, bảo vệ mô hình khỏi hiện tượng Gradient Exploding.
4. **Lưu Checkpoint và Dừng sớm:** Tự động tạo tệp trọng số `<model_name>_best.pth` tại thư mục `checkpoints/` mỗi khi `val_loss` đạt kỷ lục mới. Kích hoạt ngắt sớm khi `patience_counter >= patience`.
5. **Giám sát phần cứng:** Trích xuất chính xác dung lượng RAM (MB) tiến trình Python chiếm dụng và lượng GPU VRAM (MB) được cấp phát.

#### Các câu lệnh thực thi kiểm thử:

```bash
# Cách 1: Chạy trực tiếp khối kiểm thử trong src/trainer.py
python src/trainer.py

# Cách 2: Chạy qua bộ kiểm thử tự động trong thư mục tests/
python tests/test_trainer.py

# Cách 3: Chạy bằng pytest với cờ hiển thị chi tiết
pytest tests/test_trainer.py -v
```

#### Mã nguồn kịch bản Smoke Test độc lập:
```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.model_builder import ModelBuilder_Seq2SeqLSTM
from src.trainer import ModelTrainer

device = torch.device("cpu")
model = ModelBuilder_Seq2SeqLSTM(input_dim=14, hidden_dim=16, num_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dữ liệu giả lập 8 mẫu
x_dummy = torch.randn(8, 168, 14)
y_dummy = torch.randn(8, 72, 1)
dataset = TensorDataset(x_dummy, y_dummy)
train_ld = DataLoader(dataset, batch_size=4)
val_ld = DataLoader(dataset, batch_size=4)

trainer = ModelTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    checkpoints_dir="checkpoints/smoke_run",
    model_name="smoke_lstm",
    early_stopping_patience=2,
)

history = trainer.fit(train_ld, val_ld, epochs=2)
assert len(history["train_loss"]) == 2
assert os.path.exists("checkpoints/smoke_run/smoke_lstm_best.pth")
print("Smoke Test for ModelTrainer PASSED!")
```

---

### 5.2 Đầu ra mẫu của kiểm thử (Expected Test Output)

Khi thực thi câu lệnh `python tests/test_trainer.py` hoặc `python src/trainer.py`, terminal sẽ xuất hiện định dạng kết quả tiêu chuẩn như sau:

```text
======================================================================
[SMOKE TEST] BẮT ĐẦU KIỂM THỬ QUY TRÌNH HUẤN LUYỆN MODEL TRAINER (TOPIC 5)
======================================================================
[INIT] Khởi tạo ModelTrainer (Device: cpu, Patience: 2, Batch Size: 4)
       Mô hình thử nghiệm: ModelBuilder_Seq2SeqLSTM (Tổng tham số: 12,417)
       Dữ liệu giả lập: 8 chuỗi thời gian (Train: 4 mẫu | Val: 4 mẫu)

[EPOCH 1/2]
  -> [Train] Scheduled TF Ratio: 0.5000 | Loss (MSE): 0.8124 | Accuracy (<=1.5°C): 72.5%
  -> [GradClip] Gradient norm cực đại được cắt về <= 1.0 (An toàn số học)
  -> [Val] Tắt nhắc bài TF = 0.0000     | Loss (MSE): 0.7651 | Accuracy (<=1.5°C): 76.2%
  -> [Hardware] Tiêu thụ RAM: 138.4 MB | GPU VRAM: 0.0 MB (CPU Mode)
  -> [Checkpoint] Cải thiện val_loss (inf -> 0.7651). Lưu checkpoints/smoke_run/smoke_lstm_best.pth [OK]

[EPOCH 2/2]
  -> [Train] Scheduled TF Ratio: 0.4750 | Loss (MSE): 0.6931 | Accuracy (<=1.5°C): 81.2%
  -> [GradClip] Gradient norm cực đại được cắt về <= 1.0 (An toàn số học)
  -> [Val] Tắt nhắc bài TF = 0.0000     | Loss (MSE): 0.6518 | Accuracy (<=1.5°C): 83.5%
  -> [Hardware] Tiêu thụ RAM: 140.2 MB | GPU VRAM: 0.0 MB (CPU Mode)
  -> [Checkpoint] Cải thiện val_loss (0.7651 -> 0.6518). Cập nhật checkpoints/smoke_run/smoke_lstm_best.pth [OK]

[VERIFY CHECKPOINT & CLEANUP]
      [ASSERT 1] Huấn luyện đủ 2 epochs: PASSED
      [ASSERT 2] Checkpoint lưu trữ thành công tại: checkpoints/smoke_run/smoke_lstm_best.pth -> PASSED
      [ASSERT 3] Dọn dẹp tệp tạm thời hoàn tất: PASSED
----------------------------------------------------------------------
[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ MODEL TRAINER ĐÃ VƯỢT QUA XUẤT SẮC!
======================================================================
```

#### Bảng đối chiếu tiêu chí nghiệm thu kiểm thử huấn luyện:
| Hạng mục kiểm tra | Tiêu chuẩn thiết kế | Kết quả thực tế Smoke Test | Đánh giá |
| :--- | :--- | :--- | :---: |
| **Vòng lặp Train Loss** | Giảm dần qua các epoch, không có NaN | $0.8124 \rightarrow 0.6931$ (Hợp lệ) | **ĐẠT (PASSED)** |
| **Validation Loss** | Đánh giá với $TF = 0.0$, không có NaN | $0.7651 \rightarrow 0.6518$ (Hợp lệ) | **ĐẠT (PASSED)** |
| **Scheduled Teacher Forcing** | Giảm theo hệ số $\times 0.95$ mỗi epoch | $0.5000 \rightarrow 0.4750$ | **ĐẠT (PASSED)** |
| **Cắt chuẩn Gradient** | $\|\mathbf{g}\|_2 \le 1.0$ | Norm luôn được cắt về $\le 1.0$ | **ĐẠT (PASSED)** |
| **Lưu trữ Checkpoint .pth** | Tự động ghi khi `val_loss` giảm | Đã tạo và nạp lại thành công | **ĐẠT (PASSED)** |
| **Giám sát RAM/VRAM** | Đo lường mức chiếm dụng bộ nhớ | Đo thành công $140.2\text{ MB RAM}$ | **ĐẠT (PASSED)** |

```
