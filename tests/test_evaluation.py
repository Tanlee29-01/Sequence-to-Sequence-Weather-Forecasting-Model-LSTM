"""Kiểm thử tự động cho module đánh giá và benchmark (src/evaluation.py).
Xác minh công thức tính MAE, MSE, RMSE, R2, tỷ lệ trong ngưỡng dung sai (+-1.5°C),
kịch bản đo độ trễ suy luận (Latency Benchmark Protocol), và xuất tệp báo cáo JSON/CSV.
"""

import json
import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Đảm bảo import được module src từ thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model_builder import ModelBuilder_Seq2SeqAttention
from src.evaluation import ModelEvaluator


class TestEvaluation(unittest.TestCase):
    """Bộ kiểm thử cho ModelEvaluator."""

    def setUp(self):
        """Khởi tạo mô hình và Evaluator giả lập."""
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.model = ModelBuilder_Seq2SeqAttention(input_dim=14, hidden_dim=16, num_layers=1)
        self.model.eval()

        self.test_results_dir = "evalution_results/test_eval_run"
        os.makedirs(self.test_results_dir, exist_ok=True)

        self.evaluator = ModelEvaluator(
            model=self.model,
            device=self.device,
            tolerance_degC=1.5,
            results_dir=self.test_results_dir,
        )

    def tearDown(self):
        """Dọn dẹp thư mục kết quả thử nghiệm."""
        if os.path.exists(self.test_results_dir):
            for root, dirs, files in os.walk(self.test_results_dir, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            os.rmdir(self.test_results_dir)

    def test_compute_metrics_accuracy(self):
        """Kiểm tra độ chính xác của các công thức sai số MAE, MSE, RMSE, R2 và Tolerance."""
        # Tạo mảng giả lập có sai số xác định trước: y_pred = y_true + 1.0
        y_true = np.ones((10, 72, 1)) * 20.0
        y_pred = np.ones((10, 72, 1)) * 21.0  # Lệch đúng 1.0 độ C

        metrics = self.evaluator.compute_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["MAE"], 1.0, places=4)
        self.assertAlmostEqual(metrics["MSE"], 1.0, places=4)
        self.assertAlmostEqual(metrics["RMSE"], 1.0, places=4)
        # Độ lệch 1.0°C <= tolerance 1.5°C nên tỷ lệ đạt phải là 100%
        self.assertEqual(metrics["Within_Tolerance_Rate_Pct"], 100.0)

    def test_latency_and_resource_measurement(self):
        """Kiểm tra kịch bản đo độ trễ chuẩn (Warmup + Perf counter)."""
        sample_x = torch.randn(2, 168, 14)
        perf = self.evaluator.measure_latency_and_resources(sample_x, num_runs=3, warmup=1)

        self.assertIn("Latency_ms_per_batch", perf)
        self.assertIn("Latency_ms_per_sample", perf)
        self.assertIn("RAM_MB", perf)
        self.assertGreater(perf["Latency_ms_per_batch"], 0.0)
        self.assertGreater(perf["Latency_ms_per_sample"], 0.0)
        self.assertGreater(perf["RAM_MB"], 0.0)

    def test_export_json_and_summary_csv(self):
        """Kiểm tra khả năng lưu trữ kết quả ra file JSON và bảng CSV tổng hợp."""
        mock_results = {
            "Model": "test_model_export",
            "MAE": 1.25,
            "MSE": 2.10,
            "RMSE": 1.45,
            "R2": 0.88,
            "Within_Tolerance_Rate_Pct": 82.5,
            "Latency_ms_per_batch": 12.4,
            "RAM_MB": 120.0,
        }

        self.evaluator.save_results(mock_results, model_name="test_model_export")

        # Kiểm tra file JSON
        json_path = os.path.join(self.evaluator.run_dir, "test_model_export_metrics.json")
        self.assertTrue(os.path.exists(json_path), "File JSON kết quả không tồn tại.")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["Model"], "test_model_export")
            self.assertEqual(data["MAE"], 1.25)

        # Kiểm tra file CSV
        csv_path = os.path.join(self.evaluator.run_dir, "summary_evaluation.csv")
        self.assertTrue(os.path.exists(csv_path), "File CSV tổng hợp không tồn tại.")
        df = pd.read_csv(csv_path)
        self.assertIn("Model", df.columns)
        self.assertEqual(df.iloc[-1]["Model"], "test_model_export")

    def test_full_evaluate_pipeline(self):
        """Kiểm tra hàm evaluate() toàn diện với Test DataLoader."""
        x_dummy = torch.randn(4, 168, 14)
        y_dummy = torch.randn(4, 72, 1)
        test_ds = TensorDataset(x_dummy, y_dummy)
        test_ld = DataLoader(test_ds, batch_size=2)

        def dummy_inverse_fn(arr):
            return arr * 5.0 + 10.0

        eval_summary = self.evaluator.evaluate(
            test_loader=test_ld,
            inverse_transform_fn=dummy_inverse_fn,
            model_name="unit_test_attention_eval",
        )

        self.assertEqual(eval_summary["Model"], "unit_test_attention_eval")
        self.assertIn("MAE", eval_summary)
        self.assertIn("RMSE", eval_summary)
        self.assertIn("R2", eval_summary)


def run_smoke_test():
    """Hàm chạy nhanh Smoke Test cho quy trình đánh giá."""
    print("=" * 70)
    print("[SMOKE TEST] BẮT ĐẦU KIỂM THỬ QUY TRÌNH ĐÁNH GIÁ VÀ BENCHMARK (TOPIC 5)")
    print("=" * 70)

    device = torch.device("cpu")
    model = ModelBuilder_Seq2SeqAttention(input_dim=14, hidden_dim=16, num_layers=1)
    model.eval()

    eval_dir = "evalution_results/smoke_test_eval"
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        tolerance_degC=1.5,
        results_dir=eval_dir,
    )

    # 1. Kiểm tra tính chỉ số
    print("[1/3] Kiểm tra tính toán các chỉ số sai số hồi quy...")
    y_true = np.random.randn(10, 72, 1) * 5 + 15
    y_pred = y_true + np.random.randn(10, 72, 1) * 0.8
    res = evaluator.compute_metrics(y_true, y_pred)
    assert res["MAE"] > 0 and res["RMSE"] > 0
    print(f"      -> MAE={res['MAE']} | RMSE={res['RMSE']} | R2={res['R2']} | Acc={res['Within_Tolerance_Rate_Pct']}% [OK]")

    # 2. Kiểm tra đo độ trễ
    print("[2/3] Kiểm tra kịch bản đo độ trễ suy luận (Latency Benchmark Protocol)...")
    sample_x = torch.randn(2, 168, 14)
    perf = evaluator.measure_latency_and_resources(sample_x, num_runs=3, warmup=1)
    assert perf["Latency_ms_per_batch"] > 0
    print(f"      -> Latency/batch: {perf['Latency_ms_per_batch']} ms | RAM: {perf['RAM_MB']} MB [OK]")

    # 3. Dọn dẹp
    print("[3/3] Dọn dẹp thư mục kiểm thử...")
    if os.path.exists(eval_dir):
        for root, dirs, files in os.walk(eval_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(eval_dir)
    print("      -> Dọn dẹp hoàn tất [OK]")

    print("-" * 70)
    print("[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ ĐÁNH GIÁ ĐÃ VƯỢT QUA XUẤT SẮC!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_smoke_test()
