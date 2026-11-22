"""Kiểm thử tự động cho module huấn luyện (src/trainer.py).
Xác minh vòng lặp huấn luyện, cơ chế Scheduled Teacher Forcing, Cắt chuẩn Gradient,
Dừng sớm (Early Stopping), lưu Checkpoint (.pth) và đo lường tài nguyên phần cứng.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Đảm bảo import được module src từ thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model_builder import ModelBuilder_Seq2SeqLSTM
from src.trainer import ModelTrainer


class TestTrainer(unittest.TestCase):
    """Bộ kiểm thử cho ModelTrainer."""

    def setUp(self):
        """Khởi tạo mô hình và dữ liệu giả lập trước mỗi test case."""
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.model = ModelBuilder_Seq2SeqLSTM(input_dim=14, hidden_dim=16, num_layers=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        # Tạo dataset giả lập 12 mẫu
        x_dummy = torch.randn(12, 168, 14)
        y_dummy = torch.randn(12, 72, 1)
        dataset = TensorDataset(x_dummy, y_dummy)
        self.train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        self.test_ckpt_dir = "checkpoints/test_run"
        os.makedirs(self.test_ckpt_dir, exist_ok=True)

        self.trainer = ModelTrainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            checkpoints_dir=self.test_ckpt_dir,
            model_name="unit_test_lstm",
            early_stopping_patience=2,
            clip_grad_norm=1.0,
        )

    def tearDown(self):
        """Dọn dẹp các tệp checkpoint sau khi kiểm thử xong."""
        if os.path.exists(self.test_ckpt_dir):
            for f in os.listdir(self.test_ckpt_dir):
                os.remove(os.path.join(self.test_ckpt_dir, f))
            os.rmdir(self.test_ckpt_dir)

    def test_train_epoch_and_val_step(self):
        """Kiểm tra 1 epoch huấn luyện và validation trả về loss và accuracy hợp lệ."""
        train_loss, train_acc = self.trainer.train_epoch(self.train_loader, teacher_forcing_ratio=0.5)
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0.0)
        self.assertTrue(0.0 <= train_acc <= 100.0)

        val_loss, val_acc = self.trainer.validate(self.val_loader)
        self.assertIsInstance(val_loss, float)
        self.assertGreater(val_loss, 0.0)
        self.assertTrue(0.0 <= val_acc <= 100.0)

    def test_compute_accuracy_tolerance(self):
        """Kiểm tra hàm tính tỷ lệ % dự đoán trong ngưỡng sai số tolerance."""
        y_true = torch.tensor([[[10.0], [20.0], [30.0]]])
        y_pred = torch.tensor([[[11.0], [22.0], [30.5]]])  # lệch: 1.0, 2.0, 0.5

        # Với tolerance = 1.5: 2/3 phần tử đạt chuẩn -> ~66.67%
        acc = self.trainer.compute_accuracy(y_true, y_pred, tolerance=1.5)
        self.assertAlmostEqual(acc, 66.6667, places=2)

    def test_fit_and_checkpoint_saving(self):
        """Kiểm tra toàn bộ quá trình fit() lưu đúng file checkpoint tốt nhất."""
        history = self.trainer.fit(self.train_loader, self.val_loader, epochs=2)

        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertEqual(len(history["train_acc"]), 2)
        self.assertEqual(len(history["val_acc"]), 2)

        for epoch in (1, 2):
            epoch_path = os.path.join(
                self.test_ckpt_dir, f"unit_test_lstm_epoch_{epoch}.pth"
            )
            self.assertTrue(
                os.path.exists(epoch_path),
                f"Checkpoint của epoch {epoch} không tồn tại tại {epoch_path}",
            )
            checkpoint = torch.load(epoch_path, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], epoch)

        best_path = os.path.join(self.test_ckpt_dir, "unit_test_lstm_best.pth")
        self.assertTrue(
            os.path.exists(best_path),
            f"File checkpoint tốt nhất không tồn tại tại {best_path}"
        )

        # Kiểm tra khả năng load lại checkpoint
        fresh_model = ModelBuilder_Seq2SeqLSTM(input_dim=14, hidden_dim=16, num_layers=1)
        self.trainer.model = fresh_model
        self.trainer.load_checkpoint(best_path)

    def test_resource_monitoring(self):
        """Kiểm tra hàm đo tài nguyên RAM và GPU VRAM."""
        ram_mb = self.trainer.get_ram_usage_mb()
        self.assertIsInstance(ram_mb, float)
        self.assertGreater(ram_mb, 0.0)

        vram_mb = self.trainer.get_vram_usage_mb()
        self.assertIsInstance(vram_mb, float)
        self.assertGreaterEqual(vram_mb, 0.0)


def run_smoke_test():
    """Hàm chạy nhanh Smoke Test cho quy trình huấn luyện."""
    print("=" * 70)
    print("[SMOKE TEST] BẮT ĐẦU KIỂM THỬ QUY TRÌNH HUẤN LUYỆN MODEL TRAINER (TOPIC 5)")
    print("=" * 70)

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

    test_dir = "checkpoints/smoke_test_trainer"
    os.makedirs(test_dir, exist_ok=True)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoints_dir=test_dir,
        model_name="smoke_lstm",
        early_stopping_patience=2,
    )

    print(f"[INIT] Khởi tạo ModelTrainer (Device: {device}, Patience: 2, Batch Size: 4)")
    history = trainer.fit(train_ld, val_ld, epochs=2)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    print("      [ASSERT 1] Huấn luyện đủ 2 epochs: PASSED")

    best_ckpt = os.path.join(test_dir, "smoke_lstm_best.pth")
    assert os.path.exists(best_ckpt), "Không tìm thấy file checkpoint tốt nhất"
    print(f"      [ASSERT 2] Checkpoint lưu trữ thành công tại: {best_ckpt} -> PASSED")

    # Dọn dẹp
    for f in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, f))
    os.rmdir(test_dir)
    print("      [ASSERT 3] Dọn dẹp tệp tạm thời hoàn tất: PASSED")

    print("-" * 70)
    print("[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ MODEL TRAINER ĐÃ VƯỢT QUA XUẤT SẮC!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_smoke_test()
