"""Kiểm thử tự động cho module ghi nhận trực quan TensorBoard (src/logger.py).
Xác minh ghi nhận các chỉ số Scalar (loss, accuracy, lr, RAM, VRAM)
và hình ảnh trực quan (Forecast Figure, Attention Heatmap).
"""

import os
import sys
import unittest
import numpy as np

# Đảm bảo import được module src từ thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.logger import TensorBoardLogger


class TestLogger(unittest.TestCase):
    """Bộ kiểm thử cho TensorBoardLogger."""

    def setUp(self):
        """Khởi tạo logger trong thư mục tạm thời."""
        self.test_log_dir = "runs/test_logger_run"
        self.logger = TensorBoardLogger(log_dir=self.test_log_dir, experiment_name="unit_test")

    def tearDown(self):
        """Đóng logger và dọn dẹp các tệp log."""
        self.logger.close()
        if os.path.exists(self.test_log_dir):
            for root, dirs, files in os.walk(self.test_log_dir, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            os.rmdir(self.test_log_dir)

    def test_log_scalars(self):
        """Kiểm tra ghi nhận scalar metrics vào TensorBoard."""
        self.logger.log_metrics(
            epoch=1,
            train_loss=0.35,
            val_loss=0.32,
            lr=0.001,
            ram_mb=128.0,
            vram_mb=0.0,
        )
        self.assertTrue(os.path.exists(self.logger.exp_dir))

    def test_log_prediction_and_attention(self):
        """Kiểm tra ghi nhận biểu đồ dự báo và Attention Heatmap."""
        y_true = np.sin(np.linspace(0, 2 * np.pi, 72)) * 10 + 15
        y_pred = y_true + np.random.randn(72) * 0.5
        self.logger.log_prediction_figure(epoch=1, y_true=y_true, y_pred=y_pred)

        attn = np.random.rand(72, 168)
        attn = attn / attn.sum(axis=1, keepdims=True)
        self.logger.log_attention_heatmap(epoch=1, attention_matrix=attn)


def run_smoke_test():
    """Hàm chạy nhanh Smoke Test cho Logger."""
    print("=" * 70)
    print("[SMOKE TEST] BẮT ĐẦU KIỂM THỬ TENSORBOARD LOGGER (TOPIC 5)")
    print("=" * 70)

    test_dir = "runs/smoke_test_logger"
    logger = TensorBoardLogger(log_dir=test_dir, experiment_name="smoke_test")

    print("[1/3] Ghi nhận Scalar metrics (Loss, Accuracy, Hardware RAM/VRAM)...")
    logger.log_metrics(epoch=1, train_loss=0.45, val_loss=0.42, lr=0.001, ram_mb=110.0, vram_mb=0.0)
    print("      -> Scalars logged [OK]")

    print("[2/3] Ghi nhận hình ảnh dự báo và Attention Heatmap 72x168...")
    dummy_true = np.ones(72) * 20.0
    dummy_pred = np.ones(72) * 20.5
    logger.log_prediction_figure(epoch=1, y_true=dummy_true, y_pred=dummy_pred)
    dummy_attn = np.random.rand(72, 168)
    dummy_attn = dummy_attn / dummy_attn.sum(axis=1, keepdims=True)
    logger.log_attention_heatmap(epoch=1, attention_matrix=dummy_attn)
    print("      -> Figures logged [OK]")

    print("[3/3] Đóng logger và dọn dẹp...")
    logger.close()
    if os.path.exists(test_dir):
        for root, dirs, files in os.walk(test_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(test_dir)
    print("      -> Dọn dẹp hoàn tất [OK]")

    print("-" * 70)
    print("[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ LOGGER ĐÃ VƯỢT QUA XUẤT SẮC!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_smoke_test()
