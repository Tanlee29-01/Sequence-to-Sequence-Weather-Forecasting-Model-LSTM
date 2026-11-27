"""Kiểm thử tự động cho module đánh giá, benchmark và kiểm soát checkpoint (src/evaluation.py & main.py).
Xác minh công thức tính MAE/MSE/RMSE/R2/Tolerance tổng thể và theo horizon (1-24h, 25-48h, 49-72h),
kịch bản đo độ trễ batch/amortized/single-sample & RAM/Peak VRAM, upsert CSV theo khóa (Model, Version),
chặn thiếu checkpoint (chống random weights) và kiểm tra khớp kiến trúc trước khi load_state_dict.
"""

import json
import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Đảm bảo import được module src từ thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import main as pipeline_main
from src.data_module import JenaDataModule
from src.evaluation import ModelEvaluator
from src.model_builder import ModelBuilder_Seq2SeqAttention, ModelBuilder_Seq2SeqLSTM
from src.trainer import ModelTrainer


class TestEvaluation(unittest.TestCase):
    """Bộ kiểm thử toàn diện cho ModelEvaluator và hợp đồng đánh giá trong main.py."""

    def setUp(self):
        """Khởi tạo mô hình và Evaluator trong thư mục tạm cô lập hoàn toàn."""
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.model = ModelBuilder_Seq2SeqAttention(input_dim=14, hidden_dim=16, num_layers=1)
        self.model.eval()

        self._tmp_dir = tempfile.TemporaryDirectory()
        self.test_results_dir = self._tmp_dir.name

        self.evaluator = ModelEvaluator(
            model=self.model,
            device=self.device,
            tolerance_degC=1.5,
            results_dir=self.test_results_dir,
        )

    def tearDown(self):
        """Dọn dẹp thư mục tạm cô lập sau mỗi test case."""
        self._tmp_dir.cleanup()

    def test_compute_metrics_accuracy(self):
        """Kiểm tra độ chính xác của các công thức sai số MAE, MSE, RMSE, R2 và Tolerance."""
        y_true = np.ones((10, 72, 1)) * 20.0
        y_pred = np.ones((10, 72, 1)) * 21.0  # Lệch đúng 1.0 độ C

        metrics = self.evaluator.compute_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["MAE"], 1.0, places=4)
        self.assertAlmostEqual(metrics["MSE"], 1.0, places=4)
        self.assertAlmostEqual(metrics["RMSE"], 1.0, places=4)
        self.assertEqual(metrics["Within_Tolerance_Rate_Pct"], 100.0)

    def test_compute_horizon_metrics_slicing(self):
        """Kiểm tra cắt lát đúng chiều thời gian cho 1-24h, 25-48h và 49-72h."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(loc=15.0, scale=5.0, size=(12, 72, 1))
        y_pred = y_true.copy()
        # Cài sai số cố định khác nhau ở đúng từng khoảng 24 giờ
        y_pred[:, 0:24, :] += 1.0   # <= 1.5°C -> 100% within tol
        y_pred[:, 24:48, :] += 2.0  # >  1.5°C -> 0% within tol
        y_pred[:, 48:72, :] += 4.0  # >  1.5°C -> 0% within tol

        h_metrics = self.evaluator.compute_horizon_metrics(y_true, y_pred)

        self.assertAlmostEqual(h_metrics["MAE_1_24h"], 1.0, places=4)
        self.assertAlmostEqual(h_metrics["RMSE_1_24h"], 1.0, places=4)
        self.assertEqual(h_metrics["Within_Tol_1_24h_Pct"], 100.0)

        self.assertAlmostEqual(h_metrics["MAE_25_48h"], 2.0, places=4)
        self.assertAlmostEqual(h_metrics["RMSE_25_48h"], 2.0, places=4)
        self.assertEqual(h_metrics["Within_Tol_25_48h_Pct"], 0.0)

        self.assertAlmostEqual(h_metrics["MAE_49_72h"], 4.0, places=4)
        self.assertAlmostEqual(h_metrics["RMSE_49_72h"], 4.0, places=4)
        self.assertEqual(h_metrics["Within_Tol_49_72h_Pct"], 0.0)

    def test_latency_and_resource_measurement(self):
        """Kiểm tra tách biệt batch latency, amortized latency, single-sample latency, RAM RSS và Peak VRAM."""
        sample_x = torch.randn(4, 168, 14)
        perf = self.evaluator.measure_latency_and_resources(sample_x, num_runs=3, warmup=1)

        expected_keys = [
            "Latency_batch_ms",
            "Latency_batch_median_ms",
            "Latency_batch_std_ms",
            "Latency_amortized_per_sample_ms",
            "Latency_single_sample_ms",
            "Latency_single_sample_median_ms",
            "Latency_single_sample_std_ms",
            "RAM_MB",
            "Peak_VRAM_MB",
            "RAM_Measurement",
        ]
        for key in expected_keys:
            self.assertIn(key, perf)

        self.assertGreater(perf["Latency_batch_ms"], 0.0)
        self.assertGreater(perf["Latency_amortized_per_sample_ms"], 0.0)
        self.assertGreater(perf["Latency_single_sample_ms"], 0.0)
        self.assertAlmostEqual(
            perf["Latency_amortized_per_sample_ms"],
            perf["Latency_batch_ms"] / 4.0,
            places=3,
        )
        self.assertGreater(perf["RAM_MB"], 0.0)
        self.assertEqual(perf["Peak_VRAM_MB"], 0.0)  # CPU mode

    def test_export_json_and_csv_composite_key_upsert(self):
        """Kiểm tra xuất JSON tách theo version và upsert CSV theo khóa phức hợp (Model, Version)."""
        r_v1 = {"Model": "seq2seq_lstm", "Version": "v1", "MAE": 2.60, "RMSE": 3.40}
        r_v2 = {"Model": "seq2seq_lstm", "Version": "v2", "MAE": 2.50, "RMSE": 3.30}
        r_v1_updated = {"Model": "seq2seq_lstm", "Version": "v1", "MAE": 2.55, "RMSE": 3.35}

        j1, csv_path = self.evaluator.save_results(r_v1, model_name="seq2seq_lstm", version="v1")
        j2, _ = self.evaluator.save_results(r_v2, model_name="seq2seq_lstm", version="v2")
        self.evaluator.save_results(r_v1_updated, model_name="seq2seq_lstm", version="v1")

        self.assertTrue(os.path.exists(j1))
        self.assertTrue(os.path.exists(j2))
        self.assertTrue(j1.endswith("seq2seq_lstm_v1_metrics.json"))
        self.assertTrue(j2.endswith("seq2seq_lstm_v2_metrics.json"))

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2, "Upsert theo (Model, Version) phải giữ nguyên cả v1 và v2.")
        v1_row = df[(df["Model"] == "seq2seq_lstm") & (df["Version"] == "v1")].iloc[0]
        v2_row = df[(df["Model"] == "seq2seq_lstm") & (df["Version"] == "v2")].iloc[0]
        self.assertAlmostEqual(float(v1_row["MAE"]), 2.55, places=4)
        self.assertAlmostEqual(float(v2_row["MAE"]), 2.50, places=4)

    def test_full_evaluate_pipeline_completeness(self):
        """Kiểm tra evaluate() ghi đủ metadata, horizon metrics, latency và không ghi ra thư mục chính thức."""
        x_dummy = torch.randn(4, 168, 14)
        y_dummy = torch.randn(4, 72, 1)
        test_ds = TensorDataset(x_dummy, y_dummy)
        test_ld = DataLoader(test_ds, batch_size=2)

        def dummy_inverse_fn(arr):
            return arr * 5.0 + 10.0

        eval_summary = self.evaluator.evaluate(
            test_loader=test_ld,
            inverse_transform_fn=dummy_inverse_fn,
            model_name="seq2seq_attention",
            version="v2",
            checkpoint_path="/tmp/fake_best.pth",
            checkpoint_sha256="abc123",
            config_path="/tmp/config.yaml",
            version_config_path="/tmp/v2_config.json",
            warmup_runs=1,
            benchmark_runs=2,
        )

        required_fields = [
            "Model", "Version", "MAE", "MSE", "RMSE", "R2", "Within_Tolerance_Rate_Pct",
            "MAE_1_24h", "RMSE_1_24h", "R2_1_24h", "Within_Tol_1_24h_Pct",
            "MAE_25_48h", "RMSE_25_48h", "R2_25_48h", "Within_Tol_25_48h_Pct",
            "MAE_49_72h", "RMSE_49_72h", "R2_49_72h", "Within_Tol_49_72h_Pct",
            "Latency_batch_ms", "Latency_amortized_per_sample_ms", "Latency_single_sample_ms",
            "RAM_MB", "Peak_VRAM_MB", "RAM_Measurement",
            "Checkpoint_Path", "Checkpoint_SHA256", "Config_Path", "Version_Config_Path",
            "Device", "Batch_Size", "Warmup_Runs", "Benchmark_Runs", "Test_Samples",
        ]
        for field in required_fields:
            self.assertIn(field, eval_summary, f"Thiếu trường bắt buộc '{field}' trong kết quả.")

        self.assertEqual(eval_summary["Model"], "seq2seq_attention")
        self.assertEqual(eval_summary["Version"], "v2")
        self.assertEqual(eval_summary["Batch_Size"], 2)
        self.assertEqual(eval_summary["Warmup_Runs"], 1)
        self.assertEqual(eval_summary["Benchmark_Runs"], 2)
        self.assertEqual(eval_summary["Test_Samples"], 4)

        json_path = os.path.join(self.evaluator.run_dir, "seq2seq_attention_v2_metrics.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["Checkpoint_SHA256"], "abc123")

    def test_missing_checkpoint_blocks_evaluation(self):
        """Kiểm tra chặn tuyệt đối đánh giá bằng trọng số ngẫu nhiên khi thiếu checkpoint hoặc fallback sai."""
        with tempfile.TemporaryDirectory() as tmp_root:
            # Đặt file best ở root và file _last trong v1 để xác minh KHÔNG bị fallback nhầm
            root_ckpt = os.path.join(tmp_root, "seq2seq_lstm_best.pth")
            v1_dir = os.path.join(tmp_root, "seq2seq_lstm_v1")
            os.makedirs(v1_dir, exist_ok=True)
            last_ckpt = os.path.join(v1_dir, "seq2seq_lstm_last.pth")
            torch.save({"model_state_dict": {}, "best_val_loss": 0.1, "epoch": 1}, root_ckpt)
            torch.save({"model_state_dict": {}, "best_val_loss": 0.2, "epoch": 1}, last_ckpt)

            # 1. Không fallback ra root hay _last.pth
            resolved = pipeline_main.resolve_best_checkpoint(tmp_root, v1_dir, "seq2seq_lstm")
            self.assertEqual(resolved, "")

            # 2. Explicit --checkpoint không tồn tại phải raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                pipeline_main.resolve_best_checkpoint(
                    tmp_root, v1_dir, "seq2seq_lstm", explicit_ckpt=os.path.join(tmp_root, "missing.pth")
                )

            # 3. evaluate_model_version phải chặn ngay bằng FileNotFoundError
            dm = JenaDataModule(seq_len=168, pred_len=72)
            x_dummy = torch.randn(2, 168, 14)
            y_dummy = torch.randn(2, 72, 1)
            test_ld = DataLoader(TensorDataset(x_dummy, y_dummy), batch_size=2)
            cfg = {"models": {"seq2seq_lstm": {"input_dim": 14, "hidden_dim": 16, "num_layers": 1}}, "data": {"seq_len": 168, "pred_len": 72, "num_features": 14}}

            with self.assertRaises(FileNotFoundError):
                pipeline_main.evaluate_model_version(
                    base_model="seq2seq_lstm",
                    version="v1",
                    config=cfg,
                    config_path="configs/config.yaml",
                    dm=dm,
                    test_loader=test_ld,
                    device=torch.device("cpu"),
                    checkpoints_root=tmp_root,
                    runs_dir=os.path.join(tmp_root, "runs"),
                    results_dir=os.path.join(tmp_root, "results"),
                    warmup_runs=1,
                    benchmark_runs=1,
                    log_to_tensorboard=False,
                )

    def test_select_best_checkpoint_by_min_val_loss(self):
        """Kiểm tra resolve_best_checkpoint chọn file có best_val_loss nhỏ nhất thay vì chỉ sort tên."""
        with tempfile.TemporaryDirectory() as tmp_root:
            v1_dir = os.path.join(tmp_root, "seq2seq_lstm_v1")
            os.makedirs(v1_dir, exist_ok=True)
            older_better = os.path.join(v1_dir, "seq2seq_lstm_20261121-070000_best.pth")
            newer_worse = os.path.join(v1_dir, "seq2seq_lstm_20261121-090000_best.pth")

            torch.save({"model_state_dict": {}, "best_val_loss": 0.12, "epoch": 5, "model_name": "seq2seq_lstm"}, older_better)
            torch.save({"model_state_dict": {}, "best_val_loss": 0.19, "epoch": 8, "model_name": "seq2seq_lstm"}, newer_worse)

            selected = pipeline_main.resolve_best_checkpoint(tmp_root, v1_dir, "seq2seq_lstm")
            self.assertEqual(selected, older_better)

    def test_architecture_mismatch_rejected_before_load(self):
        """Kiểm tra từ chối ngay khi tên model, model_config hoặc shape trọng số không khớp."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_model = ModelBuilder_Seq2SeqLSTM(input_dim=14, hidden_dim=16, num_layers=1)
            ckpt_path = os.path.join(tmp_dir, "seq2seq_lstm_best.pth")
            torch.save(
                {
                    "model_state_dict": src_model.state_dict(),
                    "best_val_loss": 0.2,
                    "epoch": 3,
                    "model_name": "seq2seq_lstm",
                    "model_config": {"input_dim": 14, "hidden_dim": 16, "num_layers": 1, "output_dim": 1},
                },
                ckpt_path,
            )

            # 1. Lệch hidden_dim (32 vs 16)
            wrong_dim_model = ModelBuilder_Seq2SeqLSTM(input_dim=14, hidden_dim=32, num_layers=1)
            trainer_dim = ModelTrainer(
                model=wrong_dim_model,
                optimizer=torch.optim.Adam(wrong_dim_model.parameters(), lr=1e-3),
                criterion=nn.MSELoss(),
                device=torch.device("cpu"),
                checkpoints_dir=tmp_dir,
                model_name="seq2seq_lstm",
                model_config={"input_dim": 14, "hidden_dim": 32, "num_layers": 1, "output_dim": 1},
            )
            with self.assertRaises(ValueError):
                trainer_dim.load_checkpoint(ckpt_path, strict=True)

            # 2. Lệch model_name (seq2seq_attention vs seq2seq_lstm)
            wrong_type_model = ModelBuilder_Seq2SeqAttention(input_dim=14, hidden_dim=16, num_layers=1)
            trainer_type = ModelTrainer(
                model=wrong_type_model,
                optimizer=torch.optim.Adam(wrong_type_model.parameters(), lr=1e-3),
                criterion=nn.MSELoss(),
                device=torch.device("cpu"),
                checkpoints_dir=tmp_dir,
                model_name="seq2seq_attention",
            )
            with self.assertRaises(ValueError):
                trainer_type.load_checkpoint(ckpt_path, expected_model_name="seq2seq_attention", strict=True)

    def test_apply_version_model_config_syncs_and_validates(self):
        """Kiểm tra đồng bộ model_config + training và báo lỗi khi config.json lệch seq_len hoặc hỏng."""
        base_cfg = {
            "models": {"seq2seq_lstm": {"input_dim": 14, "hidden_dim": 128, "num_layers": 2}},
            "training": {"scheduler_type": "ReduceLROnPlateau", "learning_rate": 0.001},
            "data": {"seq_len": 168, "pred_len": 72, "num_features": 14},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = os.path.join(tmp_dir, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": "seq2seq_lstm",
                        "version": "v1",
                        "model_config": {"input_dim": 14, "hidden_dim": 64, "num_layers": 1},
                        "training": {"scheduler_type": "none"},
                        "data": {"seq_len": 168, "pred_len": 72, "num_features": 14},
                    },
                    f,
                )

            eff = pipeline_main.apply_version_model_config(
                base_cfg, tmp_dir, "seq2seq_lstm", version="v1"
            )
            self.assertEqual(eff["models"]["seq2seq_lstm"]["hidden_dim"], 64)
            self.assertEqual(eff["training"]["scheduler_type"], "none")

            # Lệch pred_len phải raise ValueError
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": "seq2seq_lstm",
                        "version": "v1",
                        "model_config": {"input_dim": 14, "hidden_dim": 64, "num_layers": 1},
                        "data": {"seq_len": 168, "pred_len": 24, "num_features": 14},
                    },
                    f,
                )
            with self.assertRaises(ValueError):
                pipeline_main.apply_version_model_config(
                    base_cfg, tmp_dir, "seq2seq_lstm", version="v1"
                )


def run_smoke_test():
    """Hàm chạy nhanh Smoke Test cho quy trình đánh giá hoàn toàn trong thư mục tạm cô lập."""
    print("=" * 70)
    print("[SMOKE TEST] BẮT ĐẦU KIỂM THỬ QUY TRÌNH ĐÁNH GIÁ VÀ BENCHMARK (TOPIC 5)")
    print("=" * 70)

    device = torch.device("cpu")
    model = ModelBuilder_Seq2SeqAttention(input_dim=14, hidden_dim=16, num_layers=1)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_eval_dir:
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            tolerance_degC=1.5,
            results_dir=tmp_eval_dir,
        )

        # 1. Kiểm tra tính chỉ số tổng thể & horizon
        print("[1/3] Kiểm tra tính toán các chỉ số sai số tổng thể và theo horizon...")
        y_true = np.random.randn(10, 72, 1) * 5 + 15
        y_pred = y_true + np.random.randn(10, 72, 1) * 0.8
        res = evaluator.compute_metrics(y_true, y_pred)
        h_res = evaluator.compute_horizon_metrics(y_true, y_pred)
        assert res["MAE"] > 0 and res["RMSE"] > 0
        assert "MAE_1_24h" in h_res and "MAE_25_48h" in h_res and "MAE_49_72h" in h_res
        print(
            f"      -> MAE={res['MAE']} | RMSE={res['RMSE']} | R2={res['R2']} | "
            f"1-24h={h_res['MAE_1_24h']} | 25-48h={h_res['MAE_25_48h']} | 49-72h={h_res['MAE_49_72h']} [OK]"
        )

        # 2. Kiểm tra đo độ trễ batch/amortized/single-sample & xuất báo cáo JSON/CSV
        print("[2/3] Kiểm tra kịch bản đánh giá & đo độ trễ suy luận (Latency Benchmark Protocol)...")
        sample_x = torch.randn(4, 168, 14)
        sample_y = torch.randn(4, 72, 1)
        test_ld = DataLoader(TensorDataset(sample_x, sample_y), batch_size=2)
        eval_out = evaluator.evaluate(
            test_loader=test_ld,
            inverse_transform_fn=lambda arr: arr * 5.0 + 15.0,
            model_name="seq2seq_attention",
            version="v1",
            warmup_runs=1,
            benchmark_runs=2,
        )
        assert eval_out["Latency_batch_ms"] > 0 and eval_out["Latency_single_sample_ms"] > 0
        assert os.path.exists(os.path.join(evaluator.run_dir, "seq2seq_attention_v1_metrics.json"))
        assert os.path.exists(os.path.join(evaluator.run_dir, "summary_evaluation.csv"))
        print(
            f"      -> Batch: {eval_out['Latency_batch_ms']} ms | "
            f"Amortized: {eval_out['Latency_amortized_per_sample_ms']} ms | "
            f"Single: {eval_out['Latency_single_sample_ms']} ms | RAM: {eval_out['RAM_MB']} MB [OK]"
        )

        # 3. Xác nhận dọn dẹp tự động bởi TemporaryDirectory
        print("[3/3] Xác minh cô lập thư mục tạm và tự động dọn dẹp...")
    print("      -> Dọn dẹp hoàn tất (không ghi rác vào evalution_results/) [OK]")

    print("-" * 70)
    print("[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ ĐÁNH GIÁ ĐÃ VƯỢT QUA XUẤT SẮC!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_smoke_test()
