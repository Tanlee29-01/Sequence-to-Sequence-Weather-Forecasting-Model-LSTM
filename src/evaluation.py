from datetime import datetime
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


DEFAULT_HORIZONS: Tuple[Tuple[str, int, int], ...] = (
    ("1_24h", 0, 24),
    ("25_48h", 24, 48),
    ("49_72h", 48, 72),
)


class ModelEvaluator:
    """Class chuyên trách chấm điểm, đo lường tốc độ suy luận và trực quan hóa giải thích mô hình trên tập Test.
    - Mục đích: Giải chuẩn hóa kết quả về độ C thực tế, tính toán hệ thống chỉ số đánh giá tổng thể và theo từng horizon (1-24h, 25-48h, 49-72h), đo độ trễ batch/amortized/single-sample (ms), mức chiếm dụng RAM RSS / Peak VRAM và vẽ Attention Heatmap.
    - Dữ liệu & Kiến trúc bên trong bao gồm: Dữ liệu tập Test, bộ giải chuẩn hóa từ DataModule, các hàm đo đạc hiệu năng từ sklearn và các bộ trích xuất ma trận Attention.
    - Bản chất bài toán: Đánh giá khách quan, tái lập được năng lực tổng quát hóa của các mô hình Hồi quy đa bước 72 giờ và xuất báo cáo khoa học theo khóa phức hợp (Model, Version).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        tolerance_degC: float = 1.5,
        results_dir: str = "evalution_results",
        run_dir: Optional[str] = None,
    ):
        """Khởi tạo đối tượng đánh giá mô hình.
        - Hàm này thực hiện: Lưu trữ thực thể mô hình, thiết lập ngưỡng dung sai nhiệt độ và tạo thư mục con lưu trữ theo định dạng ngày evaluation_YYYYMMDD (hoặc run_dir chỉ định).
        - Input:
            + model: Đối tượng mô hình PyTorch đã nạp trọng số tốt nhất.
            + device: Thiết bị chạy suy luận (CPU hoặc GPU).
            + tolerance_degC: Dung sai nhiệt độ cho phép để tính tỷ lệ chính xác (mặc định ±1.5°C).
            + results_dir: Thư mục gốc lưu trữ toàn bộ kết quả đánh giá.
            + run_dir: Đường dẫn thư mục lưu kết quả cụ thể (tùy chọn; mặc định results_dir/evaluation_YYYYMMDD).
        - Output:
            + Không trả về giá trị (khởi tạo thực thể ModelEvaluator).
        """
        self.model = model.to(device)
        self.device = device
        self.tolerance_degC = tolerance_degC

        if run_dir is not None:
            self.run_dir = run_dir
        else:
            today_str = datetime.now().strftime("%Y%m%d")
            self.run_dir = os.path.join(results_dir, f"evaluation_{today_str}")
        os.makedirs(self.run_dir, exist_ok=True)

    @staticmethod
    def _to_2d_time_series(arr: np.ndarray, name: str) -> np.ndarray:
        """Chuẩn hóa mảng dự báo/nhãn về dạng 2D (N_samples, pred_len) để cắt lát theo chiều thời gian.
        - Hàm này thực hiện: Kiểm tra số chiều (2D hoặc 3D với trục cuối bằng 1) và loại bỏ chiều singleton cuối cùng nếu có.
        - Input:
            + arr: Mảng NumPy nhiệt độ, shape (N_samples, T) hoặc (N_samples, T, 1).
            + name: Tên biến dùng cho thông báo lỗi.
        - Output:
            + Mảng NumPy 2D có shape (N_samples, T).
        """
        arr_np = np.asarray(arr, dtype=np.float64)
        if arr_np.ndim == 3 and arr_np.shape[-1] == 1:
            return arr_np[:, :, 0]
        if arr_np.ndim == 2:
            return arr_np
        raise ValueError(
            f"Mảng '{name}' phải có shape (N, T) hoặc (N, T, 1), nhận được {arr_np.shape}."
        )

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Tính toán toàn bộ các chỉ số thống kê sai số hồi quy tổng thể trên thang đo độ C thực tế.
        - Hàm này thực hiện: Tính toán MAE, MSE, RMSE, R2 và tỷ lệ phần trăm số giờ dự đoán nằm trong khoảng lệch không quá ±1.5°C.
        - Input:
            + y_true: Mảng nhiệt độ thực tế theo độ C, shape (N_samples, 72) hoặc (N_samples, 72, 1).
            + y_pred: Mảng nhiệt độ dự báo theo độ C, shape (N_samples, 72) hoặc (N_samples, 72, 1).
        - Output:
            + metrics: Từ điển chứa các giá trị sai số số thực {MAE, MSE, RMSE, R2, Within_Tolerance_Rate_Pct, loss_test, accuracy_test}.
        """
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        if y_t.shape != y_p.shape:
            raise ValueError(
                f"Kích thước y_true {y_t.shape} không khớp y_pred {y_p.shape}."
            )

        y_t_flat = y_t.reshape(-1)
        y_p_flat = y_p.reshape(-1)

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

    def compute_horizon_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizons: Tuple[Tuple[str, int, int], ...] = DEFAULT_HORIZONS,
    ) -> Dict[str, float]:
        """Tính toán các chỉ số sai số trên từng phân đoạn chân trời dự báo (1-24h, 25-48h, 49-72h).
        - Hàm này thực hiện: Đưa mảng về shape (N_samples, pred_len), cắt lát dọc theo trục thời gian (axis=1) và tính MAE, RMSE, R2, tỷ lệ dung sai cho từng khoảng giờ.
        - Input:
            + y_true: Mảng nhiệt độ thực tế theo độ C, shape (N_samples, 72) hoặc (N_samples, 72, 1).
            + y_pred: Mảng nhiệt độ dự báo theo độ C, shape (N_samples, 72) hoặc (N_samples, 72, 1).
            + horizons: Bộ các tuple (nhãn, chỉ số bắt đầu, chỉ số kết thúc) dọc theo chiều thời gian.
        - Output:
            + horizon_metrics: Từ điển chứa MAE_*, RMSE_*, R2_*, Within_Tol_*_Pct cho từng phân đoạn.
        """
        y_t_2d = self._to_2d_time_series(y_true, "y_true")
        y_p_2d = self._to_2d_time_series(y_pred, "y_pred")
        if y_t_2d.shape != y_p_2d.shape:
            raise ValueError(
                f"Kích thước 2D của y_true {y_t_2d.shape} không khớp y_pred {y_p_2d.shape}."
            )

        pred_len = y_t_2d.shape[1]
        horizon_metrics: Dict[str, float] = {}
        for label, start, end in horizons:
            if start < 0 or end <= start or end > pred_len:
                raise ValueError(
                    f"Khoảng horizon '{label}' [{start}:{end}] không hợp lệ với pred_len={pred_len}."
                )
            yt_slice = y_t_2d[:, start:end].reshape(-1)
            yp_slice = y_p_2d[:, start:end].reshape(-1)

            mae = float(mean_absolute_error(yt_slice, yp_slice))
            mse = float(mean_squared_error(yt_slice, yp_slice))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(yt_slice, yp_slice))
            within_tol = float(np.mean(np.abs(yt_slice - yp_slice) <= self.tolerance_degC) * 100.0)

            horizon_metrics[f"MAE_{label}"] = round(mae, 4)
            horizon_metrics[f"RMSE_{label}"] = round(rmse, 4)
            horizon_metrics[f"R2_{label}"] = round(r2, 4)
            horizon_metrics[f"Within_Tol_{label}_Pct"] = round(within_tol, 2)

        return horizon_metrics

    def _run_timed_forward_benchmark(
        self,
        input_tensor: torch.Tensor,
        num_runs: int,
        warmup: int,
        track_peak_vram: bool = False,
    ) -> Tuple[List[float], float]:
        """Thực thi vòng lặp warmup và đo thời gian từng lượt forward pass chính xác trên CPU/GPU.
        - Hàm này thực hiện: Chạy warmup, đồng bộ hóa CUDA trước/sau mỗi lần đo và tùy chọn reset/đọc Peak VRAM.
        - Input:
            + input_tensor: Tensor đầu vào đã đặt trên self.device, shape (B, seq_len, num_features).
            + num_runs: Số lượt đo lường chính thức.
            + warmup: Số lượt khởi động bỏ qua trước khi bấm giờ.
            + track_peak_vram: Có reset và thu nhận torch.cuda.max_memory_allocated() hay không.
        - Output:
            + run_times_ms: Danh sách thời gian thực thi (ms) của từng lượt chạy.
            + peak_vram_mb: Đỉnh VRAM cấp phát (MiB) trong quá trình chạy (0.0 nếu chạy CPU hoặc không theo dõi).
        """
        is_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        run_times_ms: List[float] = []
        peak_vram_mb = 0.0

        with torch.no_grad():
            if is_cuda and track_peak_vram:
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

            # Giai đoạn Warmup
            for _ in range(warmup):
                _ = self.model(input_tensor, trg=None, teacher_forcing_ratio=0.0)

            if is_cuda:
                torch.cuda.synchronize(self.device)
                if track_peak_vram:
                    torch.cuda.reset_peak_memory_stats(self.device)

            # Giai đoạn Benchmark từng lượt độc lập
            for _ in range(num_runs):
                if is_cuda:
                    torch.cuda.synchronize(self.device)
                start_t = time.perf_counter()
                _ = self.model(input_tensor, trg=None, teacher_forcing_ratio=0.0)
                if is_cuda:
                    torch.cuda.synchronize(self.device)
                end_t = time.perf_counter()
                run_times_ms.append((end_t - start_t) * 1000.0)

            if is_cuda and track_peak_vram:
                peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        return run_times_ms, float(peak_vram_mb)

    def measure_latency_and_resources(
        self,
        sample_input: torch.Tensor,
        num_runs: int = 50,
        warmup: int = 10,
    ) -> Dict[str, Union[float, str]]:
        """Đo lường độ trễ suy luận (batch, amortized/sample, single-sample) và tài nguyên bộ nhớ (RAM RSS, Peak VRAM).
        - Hàm này thực hiện: Khởi động (warmup) và benchmark riêng biệt cho batch đầy đủ và cho đơn mẫu (batch_size=1), đồng bộ hóa CUDA trước/sau forward pass, reset và đọc Peak VRAM bằng max_memory_allocated().
        - Input:
            + sample_input: Tensor dữ liệu thử nghiệm đầu vào, shape (Batch_size, 168, 14).
            + num_runs: Số lần lặp để tính thống kê thời gian thực thi (mặc định 50).
            + warmup: Số lần chạy khởi động bộ nhớ GPU/CPU (mặc định 10).
        - Output:
            + perf_stats: Từ điển chứa Latency_batch_ms (mean/median/std), Latency_amortized_per_sample_ms, Latency_single_sample_ms (mean/median/std), RAM_MB, Peak_VRAM_MB và ghi chú RAM_Measurement.
        """
        if sample_input is None or sample_input.ndim != 3 or sample_input.size(0) < 1:
            raise ValueError(
                f"sample_input phải là Tensor 3D (B >= 1, seq_len, features), nhận được {None if sample_input is None else tuple(sample_input.shape)}."
            )
        if num_runs < 1:
            raise ValueError(f"num_runs phải >= 1, nhận được {num_runs}.")
        if warmup < 0:
            raise ValueError(f"warmup phải >= 0, nhận được {warmup}.")

        self.model.eval()
        batch_input = sample_input.to(self.device)
        batch_size = int(batch_input.size(0))

        # 1. Benchmark theo Batch (B = batch_size) + theo dõi Peak VRAM
        batch_times_ms, peak_vram_mb = self._run_timed_forward_benchmark(
            input_tensor=batch_input,
            num_runs=num_runs,
            warmup=warmup,
            track_peak_vram=True,
        )
        avg_batch_ms = float(np.mean(batch_times_ms))
        med_batch_ms = float(np.median(batch_times_ms))
        std_batch_ms = float(np.std(batch_times_ms))
        amortized_sample_ms = avg_batch_ms / batch_size

        # 2. Benchmark đơn mẫu thực tế (batch_size = 1) với warmup riêng
        single_input = batch_input[:1].contiguous()
        single_times_ms, _ = self._run_timed_forward_benchmark(
            input_tensor=single_input,
            num_runs=num_runs,
            warmup=warmup,
            track_peak_vram=False,
        )
        avg_single_ms = float(np.mean(single_times_ms))
        med_single_ms = float(np.median(single_times_ms))
        std_single_ms = float(np.std(single_times_ms))

        # 3. Đo RAM RSS tức thời của tiến trình Python (MiB = 1024^2 bytes)
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)

        return {
            "Latency_batch_ms": round(avg_batch_ms, 4),
            "Latency_batch_median_ms": round(med_batch_ms, 4),
            "Latency_batch_std_ms": round(std_batch_ms, 4),
            "Latency_amortized_per_sample_ms": round(amortized_sample_ms, 4),
            "Latency_single_sample_ms": round(avg_single_ms, 4),
            "Latency_single_sample_median_ms": round(med_single_ms, 4),
            "Latency_single_sample_std_ms": round(std_single_ms, 4),
            "RAM_MB": round(ram_mb, 2),
            "Peak_VRAM_MB": round(peak_vram_mb, 2),
            "RAM_Measurement": "Process RSS (MiB, 1024^2 bytes) at measurement time; not peak memory.",
        }

    @staticmethod
    def _build_artifact_prefix(model_name: str, version: str = "") -> str:
        """Tạo tiền tố tên tệp đầu ra tách biệt theo cặp (model_name, version).
        - Hàm này thực hiện: Ghép '<model_name>_<version>' nếu version không rỗng và chưa nằm ở cuối model_name.
        - Input:
            + model_name: Tên mô hình (ví dụ 'seq2seq_attention').
            + version: Nhãn phiên bản (ví dụ 'v1', 'v2').
        - Output:
            + Chuỗi tiền tố định danh duy nhất cho tệp JSON và PNG.
        """
        ver = (version or "").strip()
        if ver and not model_name.endswith(f"_{ver}"):
            return f"{model_name}_{ver}"
        return model_name

    def evaluate(
        self,
        test_loader: DataLoader,
        inverse_transform_fn,
        model_name: str = "model",
        logger: Optional[object] = None,
        version: str = "",
        checkpoint_path: str = "",
        checkpoint_sha256: str = "",
        config_path: str = "",
        version_config_path: str = "",
        warmup_runs: int = 10,
        benchmark_runs: int = 50,
    ) -> Dict[str, Union[float, int, str]]:
        """Thực hiện quy trình đánh giá toàn diện trên toàn bộ tập dữ liệu kiểm thử Test.
        - Hàm này thực hiện: Thu thập toàn bộ dự đoán, giải chuẩn hóa về độ C, tính toán chỉ số tổng thể và theo horizon (1-24h, 25-48h, 49-72h), đo tốc độ suy luận chuẩn hóa và lưu báo cáo JSON/CSV cùng biểu đồ.
        - Input:
            + test_loader: DataLoader của tập kiểm thử độc lập (không shuffle).
            + inverse_transform_fn: Hàm chuyển đổi ngược từ dữ liệu chuẩn hóa về °C thực tế.
            + model_name: Tên kiến trúc mô hình (ví dụ 'seq2seq_lstm', 'seq2seq_attention', 'transformer').
            + logger: Đối tượng TensorBoardLogger để ghi nhận loss_test và accuracy_test (tùy chọn).
            + version: Nhãn phiên bản của mô hình (ví dụ 'v1', 'v2').
            + checkpoint_path: Đường dẫn file checkpoint .pth đã nạp.
            + checkpoint_sha256: Mã băm SHA-256 của file checkpoint để truy xuất nguồn gốc.
            + config_path: Đường dẫn file cấu hình YAML chính.
            + version_config_path: Đường dẫn file config.json của phiên bản (nếu có).
            + warmup_runs: Số lượt warmup trước khi benchmark latency.
            + benchmark_runs: Số lượt lặp đo latency.
        - Output:
            + full_results: Từ điển tổng hợp toàn diện chỉ số sai số tổng thể, horizon, hiệu năng và metadata tái lập.
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

        if first_batch is None or not all_preds:
            raise ValueError("test_loader rỗng, không có mẫu nào để đánh giá.")

        y_pred_scaled = np.concatenate(all_preds, axis=0)
        y_true_scaled = np.concatenate(all_trues, axis=0)

        # Giải chuẩn hóa về độ C thực tế
        y_pred_degC = inverse_transform_fn(y_pred_scaled)
        y_true_degC = inverse_transform_fn(y_true_scaled)

        # Tính toán sai số tổng thể và theo từng phân đoạn horizon (1-24h, 25-48h, 49-72h)
        metrics = self.compute_metrics(y_true_degC, y_pred_degC)
        horizon_metrics = self.compute_horizon_metrics(y_true_degC, y_pred_degC)

        # Đo độ trễ (batch, amortized, single-sample) và tài nguyên (RAM RSS, Peak VRAM)
        perf = self.measure_latency_and_resources(
            first_batch, num_runs=benchmark_runs, warmup=warmup_runs
        )

        ver = (version or "").strip()
        artifact_prefix = self._build_artifact_prefix(model_name, ver)

        full_results: Dict[str, Union[float, int, str]] = {
            "Model": model_name,
            "Version": ver,
            **metrics,
            **horizon_metrics,
            **perf,
            "Checkpoint_Path": checkpoint_path,
            "Checkpoint_SHA256": checkpoint_sha256,
            "Config_Path": config_path,
            "Version_Config_Path": version_config_path,
            "Device": str(self.device),
            "Batch_Size": int(first_batch.size(0)),
            "Warmup_Runs": int(warmup_runs),
            "Benchmark_Runs": int(benchmark_runs),
            "Test_Samples": int(y_true_degC.shape[0]),
        }

        # Lưu kết quả dạng JSON & CSV (upsert theo khóa phức hợp Model + Version)
        self.save_results(full_results, model_name=model_name, version=ver)

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
            model_name=artifact_prefix,
        )

        # Vẽ Attention Heatmap nếu mô hình có Attention
        if last_attention is not None:
            self.plot_attention_heatmap(
                attention_matrix=last_attention[0],
                model_name=artifact_prefix,
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
            + model_name: Nhãn định danh mô hình/phiên bản đang thực nghiệm.
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
            + model_name: Nhãn định danh mô hình/phiên bản (Model 2 hoặc Model 3).
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

    def save_results(
        self,
        results: Dict,
        model_name: str,
        version: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Lưu trữ bảng số liệu kết quả đánh giá xuống định dạng JSON và CSV theo khóa phức hợp (Model, Version).
        - Hàm này thực hiện: Ghi dữ liệu từ điển thành file JSON chi tiết và upsert vào summary_evaluation.csv dựa trên khóa phức hợp (Model, Version) để không ghi đè nhầm giữa các phiên bản.
        - Input:
            + results: Từ điển chứa các chỉ số đánh giá.
            + model_name: Tên của mô hình.
            + version: Nhãn phiên bản (tùy chọn; tự lấy từ results['Version'] nếu không truyền).
        - Output:
            + Tuple (json_path, csv_path) tới hai tệp báo cáo đã ghi.
        """
        ver = str(version if version is not None else results.get("Version", "") or "").strip()
        record = dict(results)
        record.setdefault("Model", model_name)
        record["Version"] = ver

        artifact_prefix = self._build_artifact_prefix(model_name, ver)

        # Lưu file JSON
        json_path = os.path.join(self.run_dir, f"{artifact_prefix}_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)

        # Lưu hoặc upsert vào file CSV tổng hợp theo khóa phức hợp (Model, Version)
        csv_path = os.path.join(self.run_dir, "summary_evaluation.csv")
        df_new = pd.DataFrame([record])
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path, keep_default_na=False)
            if "Version" not in df_old.columns:
                df_old["Version"] = ""
            df_old["Model"] = df_old["Model"].astype(str)
            df_old["Version"] = df_old["Version"].astype(str)
            df_new["Model"] = df_new["Model"].astype(str)
            df_new["Version"] = df_new["Version"].astype(str)

            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(
                subset=["Model", "Version"], keep="last"
            )
            df_combined = df_combined.sort_values(
                by=["Model", "Version"], kind="stable"
            ).reset_index(drop=True)
            df_combined.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, index=False)

        return json_path, csv_path


if __name__ == "__main__":
    import sys
    import tempfile
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    print("[TEST] Bắt đầu kiểm thử ModelEvaluator trong thư mục tạm cô lập...")
    from src.model_builder import ModelBuilder_Seq2SeqAttention
    from torch.utils.data import TensorDataset

    device = torch.device("cpu")
    dummy_model = ModelBuilder_Seq2SeqAttention(input_dim=14, hidden_dim=32, num_layers=1)
    dummy_model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        evaluator = ModelEvaluator(
            model=dummy_model,
            device=device,
            tolerance_degC=1.5,
            results_dir=tmp_dir,
        )

        # 1. Kiểm tra tính toán metrics tổng thể và theo horizon
        mock_true = np.random.randn(20, 72, 1) * 5 + 20
        mock_pred = mock_true + np.random.randn(20, 72, 1) * 0.8
        res = evaluator.compute_metrics(mock_true, mock_pred)
        h_res = evaluator.compute_horizon_metrics(mock_true, mock_pred)
        assert "MAE" in res and "R2" in res and "Within_Tolerance_Rate_Pct" in res
        assert "MAE_1_24h" in h_res and "MAE_25_48h" in h_res and "MAE_49_72h" in h_res
        print(
            f"  -> Metrics PASSED: MAE={res['MAE']} | 1-24h={h_res['MAE_1_24h']} "
            f"| 25-48h={h_res['MAE_25_48h']} | 49-72h={h_res['MAE_49_72h']}"
        )

        # 2. Kiểm tra đo latency (batch, amortized, single-sample) & RAM/Peak VRAM
        x_sample = torch.randn(4, 168, 14)
        perf = evaluator.measure_latency_and_resources(x_sample, num_runs=3, warmup=1)
        assert perf["Latency_batch_ms"] > 0 and perf["Latency_single_sample_ms"] > 0
        assert "Peak_VRAM_MB" in perf and "RAM_Measurement" in perf
        print(
            f"  -> Latency PASSED: batch={perf['Latency_batch_ms']}ms | "
            f"amortized={perf['Latency_amortized_per_sample_ms']}ms | "
            f"single={perf['Latency_single_sample_ms']}ms"
        )

        # 3. Kiểm tra hàm evaluate đầy đủ với DataLoader trong thư mục tạm
        y_sample = torch.randn(4, 72, 1)
        test_ds = TensorDataset(x_sample, y_sample)
        test_ld = DataLoader(test_ds, batch_size=2)

        def dummy_inverse(arr):
            return arr * 8.0 + 15.0

        eval_out = evaluator.evaluate(
            test_ld,
            dummy_inverse,
            model_name="seq2seq_attention",
            version="v1",
            warmup_runs=1,
            benchmark_runs=2,
        )
        assert eval_out["Model"] == "seq2seq_attention" and eval_out["Version"] == "v1"
        assert eval_out["Test_Samples"] == 4

    print("[SUCCESS] Toàn bộ bài kiểm tra cho ModelEvaluator đã vượt qua xuất sắc (không ghi rác)!")
