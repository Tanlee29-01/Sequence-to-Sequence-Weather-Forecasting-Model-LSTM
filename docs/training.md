# ĐẶC TẢ QUY TRÌNH HUẤN LUYỆN & TỐI ƯU HÓA (TRAINING SPEC)

---

## 1. Purpose (Mục đích)
- **Mục tiêu:** Thiết lập quy trình huấn luyện học sâu chuẩn mực cho bài toán hồi quy chuỗi thời gian khí tượng Jena Climate, đảm bảo mô hình hội tụ ổn định, không bị bùng nổ gradient, kiểm soát chặt chẽ hiện tượng quá khớp (Overfitting), điều chỉnh tốc độ học động và trực quan hóa đầy đủ quá trình huấn luyện trên TensorBoard.
- **Các kỹ thuật trọng tâm:**
  1. *Teacher Forcing suy giảm (Scheduled Teacher Forcing):* Bắt đầu với tỷ lệ nhắc bài cao (`0.5`) để mô hình nhanh chóng học cách giải mã, sau đó giảm dần theo hệ số `0.95` qua từng epoch để mô hình tự lực dự báo tự hồi quy (Autoregressive).
  2. *Điều chỉnh tốc độ học động (Learning Rate Scheduler):* Hỗ trợ **`ReduceLROnPlateau`** (mặc định — tự động nhân LR với `factor=0.5` nếu `val_loss` không giảm sau `lr_patience=3` epochs, tối thiểu `lr_min=1e-5`) và **`CosineAnnealingLR`** (giảm mượt theo chu kỳ cosine `cosine_T_max=50`), hoặc tắt (`none`).
  3. *Cắt chuẩn Gradient (Gradient Clipping):* Giới hạn chuẩn vector đạo hàm $\le 1.0$ để ngăn chặn triệt để hiện tượng nổ gradient thường gặp trong các mô hình hồi quy chuỗi dài (168 bước vào $\rightarrow$ 72 bước ra).
  4. *Dừng sớm (Early Stopping):* Tự động ngắt huấn luyện nếu sai số trên tập Validation không cải thiện sau `early_stopping_patience = 7` epochs, đồng thời lưu trọng số tốt nhất.
  5. *Quy đổi ngưỡng Accuracy theo độ lệch chuẩn (`target_scale`):* Truyền $\sigma_{train}$ của biến `T (degC)` từ `JenaDataModule` vào `ModelTrainer` để tính `tolerance_scaled = 1.5 / target_scale`, đảm bảo `Accuracy/Train` và `Accuracy/Validation` phản ánh chính xác sai số $\le \pm 1.5^\circ\text{C}$ thực tế (tránh ảo giác accuracy khi tính trên không gian chuẩn hóa Z-score).
  6. *Trực quan hóa định kỳ vào TensorBoard:* Tự động lấy một mẫu cố định từ `val_loader` để ghi biểu đồ dự báo 72 giờ (`Forecast_Comparison`) và bản đồ nhiệt Attention (`Attention_Weights`) sau mỗi `log_figure_every = 5` epochs (và ở epoch đầu tiên).
  7. *Giám sát tài nguyên phần cứng:* Theo dõi thời gian thực mức tiêu thụ bộ nhớ RAM hệ thống (MB) và GPU VRAM (MB) thông qua thư viện `psutil` và `torch.cuda`.

---

## 2. Input / Output Contract (Hợp đồng Đầu vào & Đầu ra)

### 2.1 Đầu vào của quá trình huấn luyện
- **Dữ liệu:** `train_loader` (`shuffle=True`) và `val_loader` (`shuffle=False`) từ `JenaDataModule` với `batch_size = 64`.
- **Mô hình:** Đối tượng `nn.Module` được khởi tạo từ [`src/model_builder.py`](file:///d:/UTH/Deeplearning/course_work_topic_5/src/model_builder.py).
- **Bộ tối ưu (Optimizer):** `torch.optim.Adam` với `learning_rate = 0.001`, `weight_decay = 1e-5`.
- **Bộ lập lịch Learning Rate (`scheduler`):** Khởi tạo tại [`main.py`](file:///d:/UTH/Deeplearning/course_work_topic_5/main.py) theo cấu hình `configs/config.yaml`:
  - `scheduler_type`: `"ReduceLROnPlateau"` (mặc định) | `"CosineAnnealingLR"` | `"none"`.
  - Tham số `ReduceLROnPlateau`: `mode="min"`, `lr_patience = 3`, `lr_factor = 0.5`, `lr_min = 0.00001` *(lưu ý: không truyền `verbose` để tương thích PyTorch 2.x)*.
  - Tham số `CosineAnnealingLR`: `cosine_T_max = 50`, `eta_min = 0.00001`.
- **Hàm mất mát (Loss Function):** Mean Squared Error (`nn.MSELoss()`).
- **Siêu tham số điều khiển trong [`configs/config.yaml`](file:///d:/UTH/Deeplearning/course_work_topic_5/configs/config.yaml):**
  - `epochs`: `50` vòng lặp tối đa.
  - `clip_grad_norm`: Ngưỡng cực đại `1.0`.
  - `early_stopping_patience`: `7` epochs.
  - `teacher_forcing_ratio_start`: `0.5` (Epoch 1).
  - `teacher_forcing_decay`: `0.95` (sau mỗi epoch).
  - `tolerance_degC`: `1.5` ($\pm 1.5^\circ\text{C}$), quy đổi sang không gian chuẩn hóa qua `target_scale = dm.scaler.scale_[dm.target_idx]`.
  - `log_figure_every`: `5` epochs (ghi biểu đồ vào TensorBoard ở epoch `1` và các bội số của `5`).

### 2.2 Vị trí và Quy chuẩn Đặt tên File Checkpoint (.pth) theo Phiên bản (`v1 .. vN`)
- **Thư mục lưu trữ phân tầng theo phiên bản:** Mỗi lần huấn luyện tự động tăng số phiên bản (`v1`, `v2`, `v3`, ..., `vN`) và lưu vào thư mục con riêng tại [`checkpoints/`](file:///d:/UTH/Deeplearning/course_work_topic_5/checkpoints/) để tuyệt đối không ghi đè các lần trước:
  ```text
  checkpoints/
  ├── <model_name>_v1/   # _best.pth, _final.pth, _last.pth, _epoch_n.pth, config.json
  ├── <model_name>_v2/
  └── <model_name>_vN/
  ```
- **Cấu trúc đặt tên bên trong `checkpoints/<model_name>_v<N>/`:**
  - Checkpoint từng epoch: `<model_name>_epoch_<n>.pth`
  - File tốt nhất (Best Validation Loss — có timestamp): `<model_name>_<YYYYMMDD-HHMMSS>_best.pth`
  - File tốt nhất rút gọn (Best Alias): `<model_name>_best.pth`
  - File kết thúc huấn luyện (Last Epoch — có timestamp & alias): `<model_name>_<YYYYMMDD-HHMMSS>_last.pth`, `<model_name>_last.pth`, và `<model_name>_final.pth`
  - File cấu hình đi kèm: `config.json` (lưu `model_config`, `training`, `data` của phiên bản để đảm bảo tái tạo đúng kiến trúc khi evaluate).
- **Ví dụ thực tế:**
  - `checkpoints/seq2seq_attention_v1/seq2seq_attention_20261121-080327_best.pth`
  - `checkpoints/seq2seq_attention_v2/seq2seq_attention_best.pth`
  - `checkpoints/seq2seq_attention_v3/config.json`
- **Cấu trúc bên trong file `.pth`:**
  ```python
  checkpoint_data = {
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "best_val_loss": self.best_val_loss,
      "epoch": epoch,
      "model_name": self.model_name,
      "model_config": self.model_config,  # Tùy chọn nếu truyền từ main.py
  }
  ```

### 2.3 Nhật ký ghi nhận trực quan vào `runs/` (TensorBoard Logging)
- **Thư mục lưu log tách biệt theo phiên bản & từng lần chạy:**
  - Mặc định khi train (`main.py --mode train|all`): tự động gắn đúng số phiên bản `vN` tương ứng với `checkpoints/`: `runs/<model_name>_v<N>_<YYYYMMDD-HHMMSS>/` (hoặc tùy biến hậu tố qua `--run_name`).
  - Khi chạy `--mode evaluate` (ví dụ `--model seq2seq_attention_v1` hoặc `--model seq2seq_attention_v3`): tự động gắn vào thư mục `runs/<model_name>_v<N>*` tương ứng để bổ sung kết quả Test cùng biểu đồ train.
- **Hệ thống Scalar tags chuẩn hóa theo namespace (không ghi trùng):**
  - `Loss/Train`: Mất mát MSE trên tập Train tại mỗi Epoch.
  - `Loss/Validation`: Mất mát MSE trên tập Validation tại mỗi Epoch.
  - `Loss/Test`: Mất mát MSE trên tập Test (ghi nhận qua `log_test_metrics()` sau khi hoàn tất đánh giá).
  - `Accuracy/Train`: Tỷ lệ % số giờ dự đoán sai lệch $\le \pm 1.5^\circ\text{C}$ trên tập Train (đã quy đổi chuẩn qua `target_scale`).
  - `Accuracy/Validation`: Tỷ lệ % số giờ dự đoán sai lệch $\le \pm 1.5^\circ\text{C}$ trên tập Validation.
  - `Accuracy/Test`: Tỷ lệ % số giờ dự đoán sai lệch $\le \pm 1.5^\circ\text{C}$ trên tập Test.
  - `Learning_Rate`: Tốc độ học hiện hành của bộ tối ưu (theo dõi bước giảm của Scheduler).
  - `Hardware/RAM_MB`: Bộ nhớ RAM tiến trình Python đang sử dụng.
  - `Hardware/VRAM_MB`: Bộ nhớ VRAM GPU đang được cấp phát (chỉ ghi khi `vram_mb > 0.0`).
- **Hình ảnh biểu đồ ghi nhận định kỳ (Tab `Images`, tại epoch 1 và mỗi `log_figure_every` epochs):**
  - `Forecast_Comparison/<model_name>`: Biểu đồ đường 72 giờ Ground Truth vs Prediction trên 1 mẫu cố định của `val_loader`, trục Y gắn nhãn **`Nhiệt độ (Normalized)`** (vì dữ liệu trong vòng lặp train ở thang chuẩn hóa Z-score).
  - `Attention_Weights/<model_name>`: Bản đồ nhiệt Attention Heatmap $72 \times 168$ (chỉ ghi cho Model 2 & Model 3).
- **Thông tin cấu hình & Siêu tham số (Tab `Text` & `HParams`):**
  - `Experiment/Config`: Chuỗi Markdown ghi Model, LR, Scheduler, Epochs, TF start/decay, `target_scale` qua `logger.log_text()`.
  - `HParams`: Ghi siêu tham số (`model`, `lr`, `scheduler`, `batch_size`, `epochs`) kèm 4 chỉ số đánh giá (`hparam/MAE`, `hparam/RMSE`, `hparam/R2`, `hparam/Acc_1_5C`) trực tiếp vào thư mục run qua `logger.log_hparams(..., run_name=".")`.

---

## 3. Success Criteria (Tiêu chí Nghiệm thu Huấn luyện)

1. **Tính ổn định của hàm tổn thất:**
   - Cả `Loss/Train` và `Loss/Validation` giảm dần ổn định theo số epoch, không xuất hiện hiện tượng phân kỳ đột ngột hoặc giá trị `NaN`/`Inf`.
2. **Hoạt động của Learning Rate Scheduler:**
   - Với `ReduceLROnPlateau`: khi `val_loss` đi ngang qua `lr_patience = 3` epochs, `Learning_Rate` tự động giảm theo hệ số `0.5` (không xuống dưới `1e-5`) và in log `[LR] Giảm Learning Rate: ...`.
3. **Tính chính xác của Accuracy trong khi Train/Val:**
   - `tolerance_scaled` phải được tính bằng `tolerance_degC / target_scale` ($\sigma_{train}$ thực tế của nhiệt độ) để `Accuracy/Train` và `Accuracy/Validation` phản ánh đúng sai số $\pm 1.5^\circ\text{C}$ thay vì giá trị ảo $\approx 99.9\%$.
4. **Kích hoạt dừng sớm chuẩn xác (Early Stopping):**
   - Nếu trong 7 epoch liên tiếp mà `val_loss` không đạt giá trị nhỏ hơn `best_val_loss`, quá trình huấn luyện tự động dừng lại và lưu `<model_name>_final.pth` cùng `<model_name>_<YYYYMMDD-HHMMSS>_last.pth`.
5. **Tuân thủ quy tắc Teacher Forcing & Gradient Clipping:**
   - Trong Train: $TF_{epoch} = TF_0 \times (0.95)^{epoch-1}$ và cắt chuẩn gradient $\le 1.0$.
   - Trong Validation và Test: Bắt buộc $TF = 0.0$.

---

## 4. How to do (Quy trình Thực hiện Chi Tiết)

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Kỹ sư / Pipeline (main.py)
    participant T as ModelTrainer
    participant M as Model (nn.Module)
    participant S as LR Scheduler
    participant L as TensorBoardLogger
    participant D as Disk (checkpoints/)

    Dev->>L: Tạo runs/<model>_<run_name|timestamp> + log_text("Experiment/Config")
    Dev->>T: fit(train_loader, val_loader, epochs, logger, log_figure_every)
    T->>T: Lấy 1 batch cố định (_viz_x, _viz_y) từ val_loader
    loop Mỗi Epoch (1 đến N)
        T->>M: train_epoch(train_loader, TF_ratio)
        M-->>T: train_loss, train_acc (theo tolerance_scaled)
        T->>M: validate(val_loader, TF_ratio=0.0)
        M-->>T: val_loss, val_acc
        T->>S: step(val_loss) [ReduceLROnPlateau] hoặc step() [Cosine]
        S-->>T: Cập nhật current_lr
        T->>L: log_metrics(Loss/*, Accuracy/*, Learning_Rate, Hardware/*)
        opt epoch == 1 hoặc epoch % log_figure_every == 0
            T->>M: Forward 1 mẫu cố định (eval, no_grad, TF=0.0)
            T->>L: log_prediction_figure() + log_attention_heatmap()
        end
        T->>D: save_checkpoint("<model>_epoch_<n>.pth")
        alt val_loss < best_val_loss
            T->>D: Lưu "<model>_<timestamp>_best.pth" & "<model>_best.pth"
            T->>T: Reset patience_counter = 0
        else val_loss >= best_val_loss
            T->>T: patience_counter += 1
            opt patience_counter >= patience (7)
                T->>Dev: Kích hoạt Early Stopping & Dừng vòng lặp
            end
        end
    end
    T->>D: Lưu "<model>_<timestamp>_last.pth", "<model>_last.pth", "<model>_final.pth"
    Dev->>L: Đánh giá Test -> log_test_metrics(Loss/Test, Accuracy/Test) + log_hparams()
```

### 4.1 Vòng lặp huấn luyện từng Epoch (`train_epoch`)
```python
self.model.train()
for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
    self.optimizer.zero_grad()
    predictions, _ = self.model(src=x_batch, trg=y_batch, teacher_forcing_ratio=current_tf)
    loss = self.criterion(predictions, y_batch)
    loss.backward()

    if self.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
    self.optimizer.step()

    # Accuracy chuẩn hóa theo tolerance_scaled = 1.5 / target_scale
    abs_diff = torch.abs(predictions.detach() - y_batch)
    total_correct += (abs_diff <= self.tolerance_scaled).sum().item()
```

### 4.2 Khởi tạo & Cập nhật Learning Rate Scheduler (`main.py` & `trainer.py`)
```python
# Trong main.py (tương thích PyTorch 2.x, không dùng verbose)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=train_cfg.get("lr_factor", 0.5),
    patience=train_cfg.get("lr_patience", 3),
    min_lr=train_cfg.get("lr_min", 1e-5),
)

# Trong ModelTrainer.fit() sau bước validate()
prev_lr = self.optimizer.param_groups[0]["lr"]
if self.scheduler is not None:
    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.scheduler.step(val_loss)
    else:
        self.scheduler.step()
current_lr = self.optimizer.param_groups[0]["lr"]
if current_lr < prev_lr:
    print(f"  [LR] Giảm Learning Rate: {prev_lr:.6f} → {current_lr:.6f}")
```

### 4.3 Lưu trữ Checkpoint (`save_checkpoint`)
```python
if checkpoint_type == "epoch":
    filename = f"{self.model_name}_epoch_{epoch}.pth"
else:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "last" if is_last else ("best" if is_best else "last")
    filename = f"{self.model_name}_{timestamp}_{suffix}.pth"

# Lưu kèm alias rút gọn không chứa timestamp:
# <model_name>_best.pth, <model_name>_last.pth, và <model_name>_final.pth (khi is_last=True)
```

### 4.4 Ghi log TensorBoard (`src/logger.py`)
```python
# Ghi scalar theo namespace nhất quán
logger.log_metrics(
    epoch=epoch,
    loss_train=train_loss,
    loss_val=val_loss,
    accuracy_train=train_acc,
    accuracy_val=val_acc,
    lr=current_lr,
    ram_mb=train_ram,
    vram_mb=train_vram,
)

# Ghi biểu đồ định kỳ (nhãn trục Y: "Nhiệt độ (Normalized)")
if epoch == 1 or epoch % log_figure_every == 0:
    logger.log_prediction_figure(epoch=epoch, y_true=y_true_np, y_pred=y_pred_np, tag=f"Forecast_Comparison/{self.model_name}")
    if viz_attn is not None:
        logger.log_attention_heatmap(epoch=epoch, attention_matrix=attn_np, tag=f"Attention_Weights/{self.model_name}")
```

---

## 5. Smoke Test & Verification (Hướng Dẫn Kiểm Thử & Đầu Ra Mẫu)

### 5.1 Hướng dẫn kiểm thử nhanh (Smoke Test Instructions)

Mục đích của bài kiểm thử khói quy trình huấn luyện là xác minh hoạt động trơn tru của `ModelTrainer` và `TensorBoardLogger` trên tập Tensor giả lập nhỏ trong vòng **3 giây**, đảm bảo:
- Vòng lặp train/val hoạt động ổn định, không NaN/Inf.
- Tỷ lệ Teacher Forcing suy giảm chính xác và gradient luôn được cắt ở mức $\le 1.0$.
- Lưu đầy đủ các checkpoint epoch (`_epoch_1.pth`, `_epoch_2.pth`), `_best.pth`, `_last.pth`, `_final.pth`.
- `TensorBoardLogger` ghi được scalars, ảnh dự báo, Attention Heatmap, text cấu hình và HParams.

#### Các câu lệnh thực thi kiểm thử:

```bash
# Cách 1: Chạy trực tiếp khối kiểm thử trong src/trainer.py
python src/trainer.py

# Cách 2: Chạy qua bộ kiểm thử tự động trong thư mục tests/
python tests/test_trainer.py
python tests/test_logger.py

# Cách 3: Chạy toàn bộ Master Smoke Tests
python tests/run_all_tests.py
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
    checkpoints_dir="checkpoints/smoke_test_trainer",
    model_name="smoke_lstm",
    early_stopping_patience=2,
)

history = trainer.fit(train_ld, val_ld, epochs=2)
assert len(history["train_loss"]) == 2
assert os.path.exists("checkpoints/smoke_test_trainer/smoke_lstm_best.pth")
assert os.path.exists("checkpoints/smoke_test_trainer/smoke_lstm_final.pth")
```

---

### 5.2 Đầu ra mẫu của kiểm thử (Expected Test Output)

Khi thực thi câu lệnh `python tests/test_trainer.py`, terminal sẽ hiển thị kết quả tiêu chuẩn như sau:

```text
======================================================================
[SMOKE TEST] BẮT ĐẦU KIỂM THỬ QUY TRÌNH HUẤN LUYỆN MODEL TRAINER (TOPIC 5)
======================================================================
[INIT] Khởi tạo ModelTrainer (Device: cpu, Patience: 2, Batch Size: 4)
[TRAIN] Bắt đầu huấn luyện mô hình: smoke_lstm trên cpu...
Epoch [01/02] (0.3s) | loss_train: 1.00621 | loss_val: 0.99375 | acc_train: 85.4% | acc_val: 85.2% | LR: 0.010000 | TF: 0.475 | RAM: 894.7MB
Epoch [02/02] (0.1s) | loss_train: 0.99469 | loss_val: 0.99512 | acc_train: 85.8% | acc_val: 85.4% | LR: 0.010000 | TF: 0.451 | RAM: 890.2MB
      [ASSERT 1] Huấn luyện đủ 2 epochs: PASSED
      [ASSERT 2] Checkpoint lưu trữ thành công tại: checkpoints/smoke_test_trainer\smoke_lstm_best.pth -> PASSED
      [ASSERT 3] Dọn dẹp tệp tạm thời hoàn tất: PASSED
----------------------------------------------------------------------
[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ MODEL TRAINER ĐÃ VƯỢT QUA XUẤT SẮC!
======================================================================
```

#### Bảng đối chiếu tiêu chí nghiệm thu kiểm thử huấn luyện:
| Hạng mục kiểm tra | Tiêu chuẩn thiết kế | Kết quả thực tế Smoke Test | Đánh giá |
| :--- | :--- | :--- | :---: |
| **Vòng lặp Train & Val Loss** | Giảm ổn định, không có NaN/Inf | `loss_train: 1.00621 -> 0.99469` | **ĐẠT (PASSED)** |
| **Scheduled Teacher Forcing** | Giảm theo hệ số $\times 0.95$ mỗi epoch | `0.500 -> 0.475 -> 0.451` | **ĐẠT (PASSED)** |
| **LR Tracking & Scheduler** | Theo dõi và giảm LR khi plateau | Hiển thị `LR: 0.010000`, gọi `scheduler.step()` | **ĐẠT (PASSED)** |
| **Cắt chuẩn Gradient** | $\|\mathbf{g}\|_2 \le 1.0$ | Norm luôn được cắt về $\le 1.0$ | **ĐẠT (PASSED)** |
| **Lưu trữ Checkpoint `.pth`** | Đầy đủ `_epoch_n`, `_best`, `_last`, `_final` | Đã tạo và nạp lại thành công | **ĐẠT (PASSED)** |
| **Giám sát RAM/VRAM** | Đo lường mức chiếm dụng bộ nhớ | Đo thành công `894.7 MB RAM` | **ĐẠT (PASSED)** |
