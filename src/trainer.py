from datetime import datetime
import os
import time
from typing import Dict, Optional, Tuple
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ModelTrainer:
    """Class điều phối toàn bộ chu trình huấn luyện, đánh giá và lưu trữ mô hình cho bài toán Jena Climate.
    - Mục đích: Quản lý vòng lặp huấn luyện (Training loop), kiểm định (Validation), cơ chế nhắc bài (Teacher Forcing), dừng sớm (Early Stopping), cắt gradient (Gradient Clipping) và giám sát tài nguyên phần cứng (RAM/VRAM).
    - Dữ liệu & Kiến trúc bên trong bao gồm: Mô hình mạng (kế thừa nn.Module), bộ tối ưu Optimizer (Adam), hàm mất mát Loss (MSE), và bộ lập lịch tỷ lệ nhắc bài.
    - Bản chất bài toán: Tối ưu hóa trọng số cho bài toán Hồi quy chuỗi thời gian đa bước 72 giờ, ghi nhận các chỉ số loss_train, loss_val, accuracy_train, accuracy_val vào TensorBoard và lưu checkpoint định dạng timestamp.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoints_dir: str = "checkpoints",
        model_name: str = "model",
        clip_grad_norm: float = 1.0,
        early_stopping_patience: int = 7,
        teacher_forcing_start: float = 0.5,
        teacher_forcing_decay: float = 0.95,
        tolerance_degC: float = 1.5,
        target_scale: float = 1.0,
    ):
        """Khởi tạo Trainer với mô hình, thuật toán tối ưu và các tham số điều khiển huấn luyện.
        - Hàm này thực hiện: Lưu trữ các tham chiếu đối tượng mạng, cấu hình thư mục lưu checkpoint và khởi tạo các biến theo dõi Early Stopping.
        - Input:
            + model: Đối tượng mô hình PyTorch cần huấn luyện.
            + optimizer: Bộ tối ưu hóa tham số (ví dụ: Adam, AdamW).
            + criterion: Hàm tổn thất mục tiêu (ví dụ: MSELoss, L1Loss).
            + device: Thiết bị tính toán (CPU hoặc CUDA GPU).
            + checkpoints_dir: Đường dẫn thư mục lưu file trọng số mô hình tốt nhất (.pth).
            + model_name: Tên tiền tố để đặt tên file lưu checkpoint.
            + clip_grad_norm: Ngưỡng cực đại để cắt chuẩn gradient chống vọt số (mặc định 1.0).
            + early_stopping_patience: Số epoch kiên nhẫn chờ đợi trước khi kích hoạt dừng sớm (mặc định 7).
            + teacher_forcing_start: Tỷ lệ nhắc bài ban đầu ở Epoch đầu tiên (mặc định 0.5).
            + teacher_forcing_decay: Hệ số suy giảm tỷ lệ nhắc bài sau mỗi Epoch (mặc định 0.95).
            + tolerance_degC: Ngưỡng sai số nhiệt độ để tính độ chính xác (mặc định 1.5°C).
            + target_scale: Độ lệch chuẩn của biến nhiệt độ trên tập Train để chuẩn hóa ngưỡng sai số.
        - Output:
            + Không trả về giá trị (khởi tạo thực thể ModelTrainer).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.model_name = model_name
        self.clip_grad_norm = clip_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.teacher_forcing_ratio = teacher_forcing_start
        self.teacher_forcing_decay = teacher_forcing_decay
        self.tolerance_scaled = tolerance_degC / max(1e-6, target_scale)

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.latest_best_checkpoint = ""

    def _measure_resources(self) -> Tuple[float, float]:
        """Đo lường lượng tài nguyên RAM hệ thống và VRAM GPU đang được sử dụng.
        - Hàm này thực hiện: Sử dụng thư viện psutil để lấy dung lượng RAM tiến trình (MB) và torch.cuda để lấy dung lượng VRAM cấp phát (MB).
        - Input:
            + Không nhận tham số bên ngoài.
        - Output:
            + ram_mb: Bộ nhớ RAM tiến trình Python đang chiếm dụng (Megabytes).
            + vram_mb: Bộ nhớ VRAM GPU đang được cấp phát (Megabytes), trả về 0.0 nếu chạy trên CPU.
        """
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)

        vram_mb = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

        return ram_mb, vram_mb

    def get_ram_usage_mb(self) -> float:
        """Đo lường lượng RAM tiến trình Python đang sử dụng (MB)."""
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))

    def get_vram_usage_mb(self) -> float:
        """Đo lường lượng GPU VRAM đang cấp phát (MB)."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            return float(torch.cuda.memory_allocated(self.device) / (1024 * 1024))
        return 0.0

    def compute_accuracy(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, tolerance: float = 1.5
    ) -> float:
        """Tính % số giá trị dự đoán có sai số tuyệt đối <= tolerance.
        - Hàm này thực hiện: So sánh độ lệch giữa nhãn thật và dự đoán với ngưỡng dung sai.
        - Input:
            + y_true: Tensor nhãn thực tế.
            + y_pred: Tensor nhãn dự đoán.
            + tolerance: Ngưỡng dung sai (mặc định 1.5).
        - Output:
            + Tỷ lệ % số điểm đạt chuẩn trong khoảng [0.0, 100.0].
        """
        abs_diff = torch.abs(y_pred - y_true)
        within_tolerance = (abs_diff <= tolerance).float()
        return float(torch.mean(within_tolerance).item() * 100.0)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int = 1,
        teacher_forcing_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Thực hiện một Epoch huấn luyện trên toàn bộ tập dữ liệu huấn luyện.
        - Hàm này thực hiện: Đưa mô hình vào chế độ train, lặp qua các batch dữ liệu, áp dụng Teacher Forcing, tính loss, accuracy, cắt chuẩn gradient và cập nhật trọng số.
        - Input:
            + train_loader: Đối tượng DataLoader chứa các batch dữ liệu huấn luyện.
            + epoch: Chỉ số vòng lặp hiện tại (bắt đầu từ 1).
            + teacher_forcing_ratio: Tùy chọn ghi đè tỷ lệ nhắc bài.
        - Output:
            + avg_loss: Giá trị hàm mất mát MSE trung bình của Epoch trên tập Train.
            + accuracy: Tỷ lệ % số giờ dự đoán lệch không quá 1.5°C trên tập Train.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_points = 0
        num_batches = 0
        current_tf = (
            teacher_forcing_ratio
            if teacher_forcing_ratio is not None
            else self.teacher_forcing_ratio
        )

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            # Gọi mô hình qua forward interface thống nhất
            predictions, _ = self.model(
                src=x_batch,
                trg=y_batch,
                teacher_forcing_ratio=current_tf,
            )

            loss = self.criterion(predictions, y_batch)
            loss.backward()

            # Cắt gradient chuẩn chống bùng nổ số
            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad_norm
                )

            self.optimizer.step()

            # Đo tỷ lệ % số bước dự đoán trong ngưỡng sai số tolerance
            abs_diff = torch.abs(predictions.detach() - y_batch)
            total_correct += (abs_diff <= self.tolerance_scaled).sum().item()
            total_points += y_batch.numel()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = (total_correct / max(1, total_points)) * 100.0

        # Giảm dần tỷ lệ nhắc bài sau mỗi epoch nếu dùng tỷ lệ mặc định
        if teacher_forcing_ratio is None:
            self.teacher_forcing_ratio *= self.teacher_forcing_decay

        return avg_loss, accuracy

    def validate(
        self, val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Đánh giá mô hình trên tập kiểm định (Validation) mà không nhắc bài.
        - Hàm này thực hiện: Đưa mô hình vào chế độ eval, tắt tính toán gradient (torch.no_grad), ép teacher_forcing_ratio = 0.0, tính hàm tổn thất và tỷ lệ chính xác.
        - Input:
            + val_loader: Đối tượng DataLoader chứa các batch kiểm định.
        - Output:
            + avg_loss: Giá trị mất mát MSE trung bình trên tập Validation.
            + accuracy: Tỷ lệ % số giờ dự đoán lệch không quá 1.5°C trên tập Validation.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_points = 0
        num_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Bắt buộc tắt nhắc bài khi validation
                predictions, _ = self.model(
                    src=x_batch,
                    trg=None,
                    teacher_forcing_ratio=0.0,
                )

                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()

                abs_diff = torch.abs(predictions - y_batch)
                total_correct += (abs_diff <= self.tolerance_scaled).sum().item()
                total_points += y_batch.numel()

                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = (total_correct / max(1, total_points)) * 100.0
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        logger: Optional[object] = None,
    ) -> Dict[str, list]:
        """Điều phối toàn bộ chu trình huấn luyện xuyên suốt số Epoch chỉ định.
        - Hàm này thực hiện: Chạy tuần tự từng epoch, ghi log lịch sử (loss_train, loss_val, accuracy_train, accuracy_val), kiểm tra Early Stopping và lưu Checkpoint tốt nhất.
        - Input:
            + train_loader: DataLoader tập huấn luyện.
            + val_loader: DataLoader tập kiểm định.
            + epochs: Tổng số Epoch huấn luyện tối đa (mặc định 50).
            + logger: Đối tượng TensorBoardLogger để ghi nhận trực quan (tùy chọn).
        - Output:
            + history: Từ điển lưu lịch sử train_loss, val_loss, train_acc, val_acc và tài nguyên theo từng Epoch.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "ram_mb": [],
            "vram_mb": [],
        }

        print(f"[TRAIN] Bắt đầu huấn luyện mô hình: {self.model_name} trên {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            train_ram, train_vram = self._measure_resources()
            elapsed = time.time() - start_time

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["ram_mb"].append(train_ram)
            history["vram_mb"].append(train_vram)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if logger is not None and hasattr(logger, "log_metrics"):
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

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"loss_train: {train_loss:.5f} | loss_val: {val_loss:.5f} | "
                f"acc_train: {train_acc:.1f}% | acc_val: {val_acc:.1f}% | "
                f"TF: {self.teacher_forcing_ratio:.3f} | RAM: {train_ram:.1f}MB"
            )

            # Lưu checkpoint của epoch và checkpoint tốt nhất
            self.save_checkpoint(epoch=epoch, checkpoint_type="epoch")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                saved_ckpt = self.save_checkpoint(epoch=epoch, is_best=True)
                self.latest_best_checkpoint = saved_ckpt
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(
                        f"[EARLY STOPPING] Kích hoạt dừng sớm tại Epoch {epoch} "
                        f"(Patience {self.early_stopping_patience} không cải thiện)."
                    )
                    break

        completed_epochs = len(history["train_loss"])
        self.save_checkpoint(epoch=completed_epochs, is_last=True)
        return history

    def save_checkpoint(
        self,
        epoch: int = 1,
        is_best: bool = False,
        is_last: bool = False,
        checkpoint_type: Optional[str] = None,
    ) -> str:
        """Lưu trữ trọng số mạng và trạng thái tối ưu xuống đĩa cứng với định dạng timestamp.
        - Hàm này thực hiện: Đóng gói state_dict của mô hình, optimizer và best_val_loss thành
          file .pth theo loại checkpoint được yêu cầu.
        - Input:
            + epoch: Chỉ số epoch hiện tại.
            + is_best: Cờ boolean đánh dấu nếu đây là mô hình đạt loss tốt nhất.
            + is_last: Cờ boolean đánh dấu checkpoint ở epoch cuối cùng.
            + checkpoint_type: Đặt là "epoch" để lưu theo mẫu <model_name>_epoch_<n>.pth.
        - Output:
            + save_path: Chuỗi đường dẫn file checkpoint đã lưu thành công.
        """
        if checkpoint_type == "epoch":
            filename = f"{self.model_name}_epoch_{epoch}.pth"
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            suffix = "best" if is_best else "last"
            if is_last:
                suffix = "last"
            filename = f"{self.model_name}_{timestamp}_{suffix}.pth"
        save_path = os.path.join(self.checkpoints_dir, filename)

        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epoch": epoch,
            "model_name": self.model_name,
        }
        torch.save(checkpoint_data, save_path)

        # Lưu đồng thời file alias không kèm timestamp cho best/last và final.
        if checkpoint_type != "epoch":
            suffix = "best" if is_best else "last"
            if is_last:
                suffix = "last"
            alias_path = os.path.join(self.checkpoints_dir, f"{self.model_name}_{suffix}.pth")
            torch.save(checkpoint_data, alias_path)
            if is_last:
                final_path = os.path.join(self.checkpoints_dir, f"{self.model_name}_final.pth")
                torch.save(checkpoint_data, final_path)

        return save_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Nạp trọng số mô hình đã lưu từ file checkpoint .pth.
        - Hàm này thực hiện: Đọc file .pth và cập nhật state_dict vào mô hình mạng.
        - Input:
            + checkpoint_path: Đường dẫn tới file trọng số .pth cần nạp.
        - Output:
            + Không trả về giá trị (cập nhật trực tiếp vào thuộc tính self.model).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Không tìm thấy file checkpoint tại {checkpoint_path}."
            )
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        if "best_val_loss" in checkpoint_data:
            self.best_val_loss = checkpoint_data["best_val_loss"]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    print("[TEST] Bắt đầu kiểm thử ModelTrainer...")
    from src.model_builder import ModelBuilder_Seq2SeqLSTM
    from torch.utils.data import TensorDataset

    device = torch.device("cpu")
    dummy_model = ModelBuilder_Seq2SeqLSTM(input_dim=14, hidden_dim=32, num_layers=1)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dữ liệu giả lập
    x_dummy = torch.randn(10, 168, 14)
    y_dummy = torch.randn(10, 72, 1)
    dataset = TensorDataset(x_dummy, y_dummy)
    train_ld = DataLoader(dataset, batch_size=4)
    val_ld = DataLoader(dataset, batch_size=4)

    trainer = ModelTrainer(
        model=dummy_model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoints_dir="checkpoints",
        model_name="test_lstm",
        early_stopping_patience=2,
    )

    history = trainer.fit(train_ld, val_ld, epochs=2)
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["train_acc"]) == 2
    assert len(history["val_acc"]) == 2
    assert os.path.exists("checkpoints/test_lstm_best.pth")

    # Dọn dẹp file test
    for f in os.listdir("checkpoints"):
        if f.startswith("test_lstm"):
            os.remove(os.path.join("checkpoints", f))

    print("[SUCCESS] Toàn bộ bài kiểm tra cho ModelTrainer đã vượt qua xuất sắc!")
