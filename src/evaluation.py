from datetime import datetime
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ModelEvaluator:
    """Class chuyên trách chấm điểm, đo lường tốc độ suy luận và trực quan hóa giải thích mô hình trên tập Test.
    - Mục đích: Giải chuẩn hóa kết quả về độ C thực tế, tính toán hệ thống chỉ số đánh giá (MAE, MSE, RMSE, R2, tỷ lệ sai số <= 1.5°C), đo độ trễ mili-giây (ms), mức chiếm dụng RAM/VRAM và vẽ Attention Heatmap.
    - Dữ liệu & Kiến trúc bên trong bao gồm: Dữ liệu tập Test, bộ giải chuẩn hóa từ DataModule, các hàm đo đạc hiệu năng từ sklearn và các bộ trích xuất ma trận Attention.
    - Bản chất bài toán: Đánh giá khách quan năng lực tổng quát hóa của các mô hình Hồi quy đa bước 72 giờ và xuất báo cáo khoa học theo ngày thực nghiệm.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        tolerance_degC: float = 1.5,
        results_dir: str = "evalution_results",
    ):
        """Khởi tạo đối tượng đánh giá mô hình.
        - Hàm này thực hiện: Lưu trữ thực thể mô hình, thiết lập ngưỡng dung sai nhiệt độ và tạo thư mục con lưu trữ theo định dạng ngày evaluation_YYYYMMDD.
        - Input:
            + model: Đối tượng mô hình PyTorch đã nạp trọng số tốt nhất.
            + device: Thiết bị chạy suy luận (CPU hoặc GPU).
            + tolerance_degC: Dung sai nhiệt độ cho phép để tính tỷ lệ chính xác (mặc định ±1.5°C).
            + results_dir: Thư mục gốc lưu trữ toàn bộ kết quả đánh giá.
        - Output:
            + Không trả về giá trị (khởi tạo thực thể ModelEvaluator).
        """
        self.model = model.to(device)
        self.device = device
        self.tolerance_degC = tolerance_degC

        today_str = datetime.now().strftime("%Y%m%d")
        self.run_dir = os.path.join(results_dir, f"evaluation_{today_str}")
        os.makedirs(self.run_dir, exist_ok=True)

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Tính toán toàn bộ các chỉ số thống kê sai số hồi quy trên thang đo độ C thực tế.
        - Hàm này thực hiện: Tính toán MAE, MSE, RMSE, R2 và tỷ lệ phần trăm số giờ dự đoán nằm trong khoảng lệch không quá ±1.5°C.
        - Input:
            + y_true: Mảng nhiệt độ thực tế theo độ C, shape (N_samples, 72) hoặc (N_samples, 72, 1).
            + y_pred: Mảng nhiệt độ dự báo theo độ C, shape (N_samples, 72) hoặc (N_samples, 72, 1).
        - Output:
            + metrics: Từ điển chứa các giá trị sai số số thực {mae, mse, rmse, r2, within_tolerance_rate_pct}.
        """
        y_t_flat = y_true.flatten()
        y_p_flat = y_pred.flatten()

        mae = float(mean_absolute_error(y_t_flat, y_p_flat))
        mse = float(mean_squared_error(y_t_flat, y_p_flat))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_t_flat, y_p_flat))

        # Tỷ lệ % số giờ dự đoán sai lệch <= ± tolerance_degC
        abs_diff = np.abs(y_t_flat - y_p_flat)
        within_tol = float(np.mean(abs_diff <= self.tolerance_degC) * 100.0)

        return {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Within_Tolerance_Rate_Pct": round(within_tol, 2),
            "loss_test": round(mse, 4),
            "accuracy_test": round(within_tol, 2),
        }

    def measure_latency_and_resources(
        self,
        sample_input: torch.Tensor,
        num_runs: int = 50,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Đo lường thời gian suy luận (Latency) và tài nguyên bộ nhớ phần cứng (RAM/VRAM).
        - Hàm này thực hiện: Thực hiện các lần chạy khởi động (Warmup), đo tổng thời gian chạy num_runs lần tiến trình forward để tính thời gian mili-giây trên mỗi batch và mỗi mẫu.
        - Input:
            + sample_input: Tensor dữ liệu thử nghiệm đầu vào, shape (Batch_size, 168, 14).
            + num_runs: Số lần lặp để tính trung bình thời gian thực thi (mặc định 50).
            + warmup: Số lần chạy khởi động bộ nhớ GPU/CPU (mặc định 10).
        - Output:
            + perf_stats: Từ điển chứa thông tin thời gian độ trễ và tài nguyên {latency_ms_per_batch, latency_ms_per_sample, ram_mb, vram_mb}.
        """
        self.model.eval()
        sample_input = sample_input.to(self.device)
        batch_size = sample_input.size(0)

        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = self.model(sample_input, trg=None, teacher_forcing_ratio=0.0)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_t = time.perf_counter()
            for _ in range(num_runs):
                _ = self.model(sample_input, trg=None, teacher_forcing_ratio=0.0)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_t = time.perf_counter()

        total_time_ms = (end_t - start_t) * 1000.0
        avg_batch_ms = total_time_ms / num_runs
        avg_sample_ms = avg_batch_ms / batch_size

        # Đo RAM & VRAM
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)

        vram_mb = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

        return {
            "Latency_ms_per_batch": round(avg_batch_ms, 2),
            "Latency_ms_per_sample": round(avg_sample_ms, 4),
            "RAM_MB": round(ram_mb, 2),
            "VRAM_MB": round(vram_mb, 2),
        }

    def evaluate(
        self,
        test_loader: DataLoader,
        inverse_transform_fn,
        model_name: str = "model",
        logger: Optional[object] = None,
    ) -> Dict[str, Union[float, str]]:
        """Thực hiện quy trình đánh giá toàn diện trên toàn bộ tập dữ liệu kiểm thử Test.
        - Hàm này thực hiện: Thu thập toàn bộ dự đoán, giải chuẩn hóa về độ C, tính toán các chỉ số độ chính xác (loss_test, accuracy_test), đo tốc độ suy luận và lưu báo cáo.
        - Input:
            + test_loader: DataLoader của tập kiểm thử độc lập (không shuffle).
            + inverse_transform_fn: Hàm chuyển đổi ngược từ dữ liệu chuẩn hóa về °C thực tế.
            + model_name: Tên mô hình để phân biệt trong báo cáo.
            + logger: Đối tượng TensorBoardLogger để ghi nhận loss_test và accuracy_test.
        - Output:
            + full_results: Từ điển tổng hợp toàn diện chỉ số sai số, tốc độ và tài nguyên.
        """
        self.model.eval()
        all_preds = []
        all_trues = []
        last_attention = None
        first_batch = None

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                if first_batch is None:
                    first_batch = x_batch

                x_batch = x_batch.to(self.device)
                preds, attn = self.model(
                    src=x_batch, trg=None, teacher_forcing_ratio=0.0
                )

                all_preds.append(preds.cpu().numpy())
                all_trues.append(y_batch.numpy())

                if attn is not None:
                    last_attention = attn.cpu().numpy()

        y_pred_scaled = np.concatenate(all_preds, axis=0)
        y_true_scaled = np.concatenate(all_trues, axis=0)

        # Giải chuẩn hóa về độ C
        y_pred_degC = inverse_transform_fn(y_pred_scaled)
        y_true_degC = inverse_transform_fn(y_true_scaled)

        # Tính toán sai số
        metrics = self.compute_metrics(y_true_degC, y_pred_degC)

        # Đo độ trễ và tài nguyên
        perf = self.measure_latency_and_resources(first_batch)

        full_results = {
            "Model": model_name,
            **metrics,
            **perf,
        }

        # Lưu kết quả dạng JSON & CSV
        self.save_results(full_results, model_name)

        # Ghi nhận loss_test và accuracy_test vào TensorBoard
        if logger is not None and hasattr(logger, "log_test_metrics"):
            logger.log_test_metrics(
                loss_test=metrics["loss_test"],
                accuracy_test=metrics["accuracy_test"],
            )

        # Vẽ biểu đồ mẫu dự báo 72h
        self.plot_forecast_sample(
            y_true=y_true_degC[0],
            y_pred=y_pred_degC[0],
            model_name=model_name,
        )

        # Vẽ Attention Heatmap nếu mô hình có Attention
        if last_attention is not None:
            self.plot_attention_heatmap(
                attention_matrix=last_attention[0],
                model_name=model_name,
            )

        return full_results

    def plot_forecast_sample(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        sample_idx: int = 0,
    ) -> str:
        """Vẽ biểu đồ đường so sánh giữa 72 giờ nhiệt độ thực tế và nhiệt độ dự báo.
        - Hàm này thực hiện: Tạo hình vẽ so sánh, ghi rõ các mốc từ giờ +1 đến +72 và lưu thành file PNG trong thư mục kết quả.
        - Input:
            + y_true: Chuỗi nhiệt độ thực tế của một mẫu (°C), shape (72,) hoặc (72, 1).
            + y_pred: Chuỗi nhiệt độ dự báo của một mẫu (°C), shape (72,) hoặc (72, 1).
            + model_name: Tên mô hình đang thực nghiệm.
            + sample_idx: Chỉ số mẫu cần vẽ biểu đồ (mặc định 0).
        - Output:
            + save_path: Đường dẫn file PNG đã lưu trữ thành công.
        """
        fig, ax = plt.subplots(figsize=(11, 4))
        hours = np.arange(1, len(y_true) + 1)

        ax.plot(hours, y_true.flatten(), label="Thực tế (Ground Truth)", color="#1f77b4", linewidth=2.5)
        ax.plot(hours, y_pred.flatten(), label=f"Dự báo ({model_name})", color="#d62728", linestyle="--", linewidth=2.0)

        # Vùng dung sai ±1.5°C
        ax.fill_between(
            hours,
            y_true.flatten() - self.tolerance_degC,
            y_true.flatten() + self.tolerance_degC,
            color="#1f77b4",
            alpha=0.15,
            label=f"Dung sai ±{self.tolerance_degC}°C",
        )

        ax.set_title(f"Dự báo nhiệt độ 72 giờ (3 ngày) trên tập Test - {model_name}")
        ax.set_xlabel("Thời gian trong tương lai (+1h đến +72h)")
        ax.set_ylabel("Nhiệt độ (°C)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        save_path = os.path.join(self.run_dir, f"{model_name}_sample_{sample_idx}_forecast.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_attention_heatmap(
        self, attention_matrix: np.ndarray, model_name: str
    ) -> str:
        """Vẽ bản đồ nhiệt Attention Heatmap giải thích cơ chế chú ý theo thời gian.
        - Hàm này thực hiện: Vẽ lưới trọng số 72x168 thể hiện mức độ liên kết giữa từng giờ dự báo tương lai với 168 giờ quá khứ.
        - Input:
            + attention_matrix: Ma trận 2D trọng số chú ý, shape (72, 168).
            + model_name: Tên mô hình (Model 2 hoặc Model 3).
        - Output:
            + save_path: Đường dẫn tới file ảnh PNG Heatmap đã lưu.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(
            attention_matrix,
            aspect="auto",
            cmap="inferno",
            interpolation="nearest",
        )
        fig.colorbar(cax, ax=ax, label="Attention Weight (Xác suất chú ý)")

        ax.set_title(f"Bản đồ nhiệt Attention (72h Tương lai x 168h Quá khứ) - {model_name}")
        ax.set_xlabel("Giờ trong quá khứ (1 đến 168)")
        ax.set_ylabel("Giờ dự báo tương lai (1 đến 72)")
        plt.tight_layout()

        save_path = os.path.join(self.run_dir, f"{model_name}_attention_heatmap.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def save_results(self, results: Dict, model_name: str) -> None:
        """Lưu trữ bảng số liệu kết quả đánh giá xuống định dạng JSON và CSV.
        - Hàm này thực hiện: Ghi dữ liệu từ điển thành file JSON chi tiết và cập nhật/bổ sung vào bảng tổng hợp CSV.
        - Input:
            + results: Từ điển chứa các chỉ số đánh giá.
            + model_name: Tên của mô hình.
        - Output:
            + Không trả về giá trị (ghi các file kết quả vào thư mục run_dir).
        """
        # Lưu file JSON
        json_path = os.path.join(self.run_dir, f"{model_name}_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Lưu hoặc nối vào file CSV tổng hợp
        csv_path = os.path.join(self.run_dir, "summary_evaluation.csv")
        df_new = pd.DataFrame([results])
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["Model"], keep="last")
            df_combined.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, index=False)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    print("[TEST] Bắt đầu kiểm thử ModelEvaluator...")
    from src.model_builder import ModelBuilder_Seq2SeqAttention
    from torch.utils.data import TensorDataset

    device = torch.device("cpu")
    dummy_model = ModelBuilder_Seq2SeqAttention(input_dim=14, hidden_dim=32, num_layers=1)
    dummy_model.eval()

    evaluator = ModelEvaluator(
        model=dummy_model,
        device=device,
        tolerance_degC=1.5,
        results_dir="evalution_results",
    )

    # 1. Kiểm tra tính toán metrics
    mock_true = np.random.randn(20, 72, 1) * 5 + 20
    mock_pred = mock_true + np.random.randn(20, 72, 1) * 0.8
    res = evaluator.compute_metrics(mock_true, mock_pred)
    assert "MAE" in res and "R2" in res and "Within_Tolerance_Rate_Pct" in res
    print(f"  -> Metrics PASSED: MAE={res['MAE']} | R2={res['R2']} | Within Tol={res['Within_Tolerance_Rate_Pct']}%")

    # 2. Kiểm tra đo latency
    x_sample = torch.randn(4, 168, 14)
    perf = evaluator.measure_latency_and_resources(x_sample, num_runs=5, warmup=2)
    assert "Latency_ms_per_batch" in perf and perf["Latency_ms_per_batch"] > 0
    print(f"  -> Latency PASSED: {perf['Latency_ms_per_batch']} ms/batch")

    # 3. Kiểm tra hàm evaluate đầy đủ với DataLoader
    y_sample = torch.randn(4, 72, 1)
    test_ds = TensorDataset(x_sample, y_sample)
    test_ld = DataLoader(test_ds, batch_size=2)

    def dummy_inverse(arr):
        return arr * 8.0 + 15.0

    eval_out = evaluator.evaluate(test_ld, dummy_inverse, model_name="test_attention_eval")
    assert eval_out["Model"] == "test_attention_eval"
    print("[SUCCESS] Toàn bộ bài kiểm tra cho ModelEvaluator đã vượt qua xuất sắc!")
