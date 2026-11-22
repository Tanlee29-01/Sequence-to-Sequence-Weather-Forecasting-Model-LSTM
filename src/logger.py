import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Class quản lý việc ghi nhận nhật ký huấn luyện và biểu đồ trực quan vào TensorBoard.
    - Mục đích: Giám sát trực quan hóa quá trình hội tụ mô hình (Train/Val Loss), biến thiên Learning Rate, mức độ sử dụng tài nguyên (RAM/VRAM) và chất lượng dự báo 72 giờ.
    - Dữ liệu & Kiến trúc bên trong bao gồm: SummaryWriter từ PyTorch, các hàm vẽ đồ thị Matplotlib cho chuỗi nhiệt độ và ma trận nhiệt trọng số Attention Heatmap.
    - Bản chất bài toán: Theo dõi, phát hiện hiện tượng quá khớp (Overfitting) và giải thích trực quan cơ chế hoạt động của mô hình dự báo thời tiết.
    """

    def __init__(self, log_dir: str = "runs", experiment_name: str = "exp_default"):
        """Khởi tạo SummaryWriter với đường dẫn lưu trữ nhật ký thực nghiệm.
        - Hàm này thực hiện: Tạo thư mục log con tương ứng với từng thử nghiệm và thiết lập bộ ghi nhận TensorBoard.
        - Input:
            + log_dir: Thư mục gốc lưu trữ log TensorBoard (mặc định 'runs').
            + experiment_name: Tên của phiên thực nghiệm (mặc định 'exp_default').
        - Output:
            + Không trả về giá trị (khởi tạo thực thể TensorBoardLogger).
        """
        self.exp_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.exp_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.exp_path)

    def log_metrics(
        self,
        epoch: int,
        loss_train: Optional[float] = None,
        loss_val: Optional[float] = None,
        accuracy_train: float = 0.0,
        accuracy_val: float = 0.0,
        lr: float = 0.001,
        ram_mb: float = 0.0,
        vram_mb: float = 0.0,
        loss_test: Optional[float] = None,
        accuracy_test: Optional[float] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> None:
        """Ghi nhận các chỉ số số học vào TensorBoard tại mỗi Epoch.
        - Hàm này thực hiện: Ghi lại loss_train, loss_val, accuracy_train, accuracy_val, loss_test, accuracy_test, Learning Rate và tài nguyên.
        - Input:
            + epoch: Chỉ số vòng lặp hiện tại.
            + loss_train / train_loss: Mất mát MSE trên tập Train.
            + loss_val / val_loss: Mất mát MSE trên tập Validation.
            + accuracy_train: Tỷ lệ % số giờ dự đoán sai lệch <= 1.5°C trên tập Train.
            + accuracy_val: Tỷ lệ % số giờ dự đoán sai lệch <= 1.5°C trên tập Validation.
            + lr: Tốc độ học hiện thời của optimizer.
            + ram_mb: Bộ nhớ RAM tiến trình (MB).
            + vram_mb: Bộ nhớ VRAM GPU (MB).
            + loss_test: Mất mát MSE trên tập Test (tùy chọn).
            + accuracy_test: Tỷ lệ % số giờ sai lệch <= 1.5°C trên tập Test (tùy chọn).
        - Output:
            + Không trả về giá trị (ghi trực tiếp vào file log event).
        """
        if loss_train is None:
            loss_train = train_loss if train_loss is not None else 0.0
        if loss_val is None:
            loss_val = val_loss if val_loss is not None else 0.0
        self.writer.add_scalar("loss_train", loss_train, epoch)
        self.writer.add_scalar("loss_val", loss_val, epoch)
        self.writer.add_scalar("accuracy_train", accuracy_train, epoch)
        self.writer.add_scalar("accuracy_val", accuracy_val, epoch)

        if loss_test is not None:
            self.writer.add_scalar("loss_test", loss_test, epoch)
        if accuracy_test is not None:
            self.writer.add_scalar("accuracy_test", accuracy_test, epoch)

        self.writer.add_scalar("Loss/Train", loss_train, epoch)
        self.writer.add_scalar("Loss/Validation", loss_val, epoch)
        self.writer.add_scalar("Learning_Rate", lr, epoch)
        self.writer.add_scalar("Hardware/RAM_MB", ram_mb, epoch)
        if vram_mb > 0.0:
            self.writer.add_scalar("Hardware/VRAM_MB", vram_mb, epoch)

    def log_test_metrics(
        self, loss_test: float, accuracy_test: float, epoch: int = 1
    ) -> None:
        """Ghi nhận riêng chỉ số kiểm thử trên tập Test vào TensorBoard.
        - Hàm này thực hiện: Đăng ký scalar loss_test và accuracy_test phục vụ đối chiếu trên giao diện.
        - Input:
            + loss_test: Sai số MSE trên toàn bộ tập Test.
            + accuracy_test: Tỷ lệ % số giờ lệch <= 1.5°C trên tập Test.
            + epoch: Mốc thời gian ghi nhận (mặc định 1).
        - Output:
            + Không trả về giá trị.
        """
        self.writer.add_scalar("loss_test", loss_test, epoch)
        self.writer.add_scalar("accuracy_test", accuracy_test, epoch)

    def log_prediction_figure(
        self,
        epoch: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tag: str = "Forecast_Comparison",
    ) -> None:
        """Vẽ biểu đồ so sánh chuỗi nhiệt độ thực tế và nhiệt độ dự báo 72 giờ và đưa vào TensorBoard.
        - Hàm này thực hiện: Khởi tạo đồ thị đường thẳng biểu diễn 72 mốc giờ, so sánh đường Ground Truth và đường Prediction rồi gửi sang SummaryWriter.
        - Input:
            + epoch: Chỉ số Epoch hiện tại.
            + y_true: Chuỗi nhiệt độ thực tế (°C), shape (72,) hoặc (72, 1).
            + y_pred: Chuỗi nhiệt độ do mô hình dự báo (°C), shape (72,) hoặc (72, 1).
            + tag: Thẻ định danh biểu đồ trong giao diện TensorBoard.
        - Output:
            + Không trả về giá trị (lưu hình vẽ vào TensorBoard và tự động giải phóng bộ nhớ đồ họa).
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        hours = np.arange(1, len(y_true) + 1)

        ax.plot(hours, y_true.flatten(), label="Thực tế (Ground Truth)", color="blue", linewidth=2)
        ax.plot(hours, y_pred.flatten(), label="Dự báo (Prediction)", color="red", linestyle="--", linewidth=2)
        ax.set_title(f"Dự báo nhiệt độ 72 giờ tiếp theo - Epoch {epoch}")
        ax.set_xlabel("Giờ trong tương lai (+1h đến +72h)")
        ax.set_ylabel("Nhiệt độ (°C)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        self.writer.add_figure(tag, fig, global_step=epoch)
        plt.close(fig)

    def log_attention_heatmap(
        self,
        epoch: int,
        attention_matrix: np.ndarray,
        tag: str = "Attention_Weights",
    ) -> None:
        """Vẽ và ghi nhận bản đồ nhiệt Attention Heatmap vào TensorBoard để giải thích mô hình.
        - Hàm này thực hiện: Vẽ ma trận tương quan giữa 72 giờ dự báo (trục Y) và 168 giờ lịch sử (trục X) dưới dạng Heatmap.
        - Input:
            + epoch: Chỉ số Epoch hiện tại.
            + attention_matrix: Ma trận trọng số Attention 2D, shape (72, 168).
            + tag: Tên thẻ trong TensorBoard.
        - Output:
            + Không trả về giá trị (gửi hình ảnh Heatmap vào SummaryWriter).
        """
        if attention_matrix is None:
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        cax = ax.imshow(attention_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        fig.colorbar(cax, ax=ax, label="Attention Weight")

        ax.set_title(f"Attention Heatmap (72h Tương lai x 168h Quá khứ) - Epoch {epoch}")
        ax.set_xlabel("Giờ quá khứ (1 đến 168)")
        ax.set_ylabel("Giờ dự báo tương lai (1 đến 72)")

        self.writer.add_figure(tag, fig, global_step=epoch)
        plt.close(fig)

    def close(self) -> None:
        """Đóng bộ ghi nhận SummaryWriter và xả sạch dữ liệu đệm ra đĩa.
        - Hàm này thực hiện: Đảm bảo toàn bộ sự kiện đã được lưu trọn vẹn xuống file và giải phóng tài nguyên.
        - Input:
            + Không nhận tham số.
        - Output:
            + Không trả về giá trị.
        """
        self.writer.flush()
        self.writer.close()


if __name__ == "__main__":
    print("[TEST] Bắt đầu kiểm thử TensorBoardLogger...")
    logger = TensorBoardLogger(log_dir="runs", experiment_name="test_logger")

    # Ghi metric thử nghiệm
    logger.log_metrics(epoch=1, train_loss=0.45, val_loss=0.42, lr=0.001, ram_mb=120.0, vram_mb=0.0)

    # Ghi hình so sánh dự báo
    dummy_true = np.sin(np.linspace(0, 3 * np.pi, 72)) * 10 + 20
    dummy_pred = dummy_true + np.random.randn(72) * 0.5
    logger.log_prediction_figure(epoch=1, y_true=dummy_true, y_pred=dummy_pred)

    # Ghi Attention Heatmap
    dummy_attn = np.random.rand(72, 168)
    dummy_attn = dummy_attn / dummy_attn.sum(axis=1, keepdims=True)
    logger.log_attention_heatmap(epoch=1, attention_matrix=dummy_attn)

    logger.close()
    assert os.path.exists("runs/test_logger")
    print("[SUCCESS] Toàn bộ bài kiểm tra cho TensorBoardLogger đã vượt qua xuất sắc!")
