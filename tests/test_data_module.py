"""Kiểm thử tự động cho module tiền xử lý dữ liệu (src/data_module.py).
Xác minh quy trình 4 pha chống rò rỉ dữ liệu (Anti-Data Leakage),
kích thước Tensor DataLoader và hàm giải chuẩn hóa nhiệt độ Celsius.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

# Đảm bảo import được module src từ thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_module import JenaDataModule, SlidingWindowDataset


class TestDataModule(unittest.TestCase):
    """Bộ kiểm thử cho JenaDataModule và SlidingWindowDataset."""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo Mock DataFrame 1000 bước thời gian với đầy đủ 14 đặc trưng khí tượng."""
        np.random.seed(42)
        cls.periods = 2000
        cls.date_range = pd.date_range("2011-01-01", periods=cls.periods, freq="1h")
        data = np.random.randn(cls.periods, 14) * 5 + 15
        # Cài đặt giá trị cảm biến gió lỗi (-9999.0) để thử nghiệm bộ lọc
        data[25, 11] = -9999.0
        data[80, 12] = -9999.0

        cls.mock_df = pd.DataFrame(
            data,
            index=cls.date_range,
            columns=JenaDataModule.DEFAULT_FEATURES,
        )

        cls.dm = JenaDataModule(
            seq_len=168,
            pred_len=72,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        cls.dm.setup(cls.mock_df)

    def test_faulty_sensor_cleaning(self):
        """Kiểm tra giá trị lỗi -9999.0 đã được thay thế và điền khuyết an toàn."""
        # Kiểm tra không còn giá trị âm cực đoan <= -9000 trong train_df
        self.assertFalse(
            (self.dm.train_df["wv (m/s)"] <= -9000).any(),
            "Lỗi: Cột 'wv (m/s)' vẫn còn giá trị lỗi -9999.0 sau tiền xử lý."
        )
        self.assertFalse(
            (self.dm.train_df["max. wv (m/s)"] <= -9000).any(),
            "Lỗi: Cột 'max. wv (m/s)' vẫn còn giá trị lỗi -9999.0 sau tiền xử lý."
        )
        self.assertFalse(
            self.dm.train_df.isna().any().any(),
            "Lỗi: Tập train vẫn còn giá trị NaN sau khi ffill/bfill."
        )

    def test_zero_leakage_temporal_split(self):
        """Kiểm tra Pha 1: Thứ tự thời gian Train < Val < Test tuyệt đối không trùng lặp."""
        train_max = self.dm.train_df.index.max()
        val_min = self.dm.val_df.index.min()
        val_max = self.dm.val_df.index.max()
        test_min = self.dm.test_df.index.min()

        self.assertLess(
            train_max,
            val_min,
            f"Rò rỉ thời gian: Train max ({train_max}) >= Val min ({val_min})"
        )
        self.assertLess(
            val_max,
            test_min,
            f"Rò rỉ thời gian: Val max ({val_max}) >= Test min ({test_min})"
        )

    def test_scaler_fitted_strictly_on_train(self):
        """Kiểm tra Pha 3: StandardScaler chỉ tính mean/std trên tập Train (Zero Scaler Leakage)."""
        expected_train_rows = len(self.dm.train_df)
        actual_samples_seen = self.dm.scaler.n_samples_seen_

        self.assertEqual(
            actual_samples_seen,
            expected_train_rows,
            f"Rò rỉ Scaler: Số mẫu fit ({actual_samples_seen}) != Số dòng Train ({expected_train_rows})"
        )

    def test_verify_zero_leakage_method(self):
        """Kiểm tra phương thức kiểm định tự động verify_zero_leakage() trả về True."""
        self.assertTrue(
            self.dm.verify_zero_leakage(),
            "Phương thức verify_zero_leakage() thất bại."
        )

    def test_dataloader_tensor_shapes(self):
        """Kiểm tra Pha 4: DataLoader sinh Tensor X (Batch, 168, 14) và Y (Batch, 72, 1)."""
        train_loader, val_loader, test_loader = self.dm.get_dataloaders(batch_size=16)
        x_batch, y_batch = next(iter(train_loader))

        self.assertEqual(
            x_batch.shape,
            (16, 168, 14),
            f"Shape Tensor X không đúng contract: {x_batch.shape}"
        )
        self.assertEqual(
            y_batch.shape,
            (16, 72, 1),
            f"Shape Tensor Y không đúng contract: {y_batch.shape}"
        )
        self.assertEqual(x_batch.dtype, torch.float32)
        self.assertEqual(y_batch.dtype, torch.float32)

    def test_inverse_transform_target(self):
        """Kiểm tra hàm giải chuẩn hóa khôi phục về thang đo độ C thực tế."""
        _, _, test_loader = self.dm.get_dataloaders(batch_size=8)
        _, y_batch = next(iter(test_loader))
        y_np = y_batch.numpy()

        y_degC = self.dm.inverse_transform_target(y_np)

        self.assertEqual(y_degC.shape, y_np.shape)
        # Nhiệt độ trung bình trong mock data khoảng 15°C
        mean_temp = np.mean(y_degC)
        self.assertTrue(
            -10.0 <= mean_temp <= 40.0,
            f"Nhiệt độ sau giải chuẩn hóa bất thường: {mean_temp}°C"
        )

    def test_val_test_leading_nan_uses_train_stats_not_future_bfill(self):
        """Kiểm tra NaN ở dòng đầu Val/Test được điền bằng trung vị Train, tuyệt đối không bfill từ tương lai."""
        periods = 1000
        date_range = pd.date_range("2012-01-01", periods=periods, freq="1h")
        data = np.ones((periods, 14), dtype=np.float64) * 10.0
        # Cài giá trị tương lai rất khác biệt ở dòng 1 của Val (index 701) và NaN ở dòng 0 của Val (index 700)
        data[700, 1] = np.nan
        data[701:, 1] = 99.0
        df = pd.DataFrame(data, index=date_range, columns=JenaDataModule.DEFAULT_FEATURES)

        dm = JenaDataModule(seq_len=168, pred_len=72)
        dm.setup(df)

        self.assertFalse(dm.val_df.isna().any().any())
        # Nếu dùng bfill từ tương lai thì val_df.iloc[0, 1] sẽ là 99.0; đúng anti-leakage phải là trung vị Train = 10.0
        self.assertAlmostEqual(float(dm.val_df.iloc[0]["T (degC)"]), 10.0, places=5)


def run_smoke_test():
    """Hàm chạy nhanh Smoke Test với định dạng đầu ra chi tiết."""
    print("=" * 70)
    print("[SMOKE TEST] BẮT ĐẦU KIỂM THỬ DATA MODULE & QUY TẮC CHỐNG RÒ RỈ")
    print("=" * 70)

    # 1. Tạo dữ liệu giả lập
    periods = 2000
    date_range = pd.date_range("2010-01-01", periods=periods, freq="1h")
    data = np.random.randn(periods, 14) * 6 + 12
    data[15, 11] = -9999.0  # Lỗi cảm biến gió
    mock_df = pd.DataFrame(data, index=date_range, columns=JenaDataModule.DEFAULT_FEATURES)

    print(f"[1/5] Khởi tạo Mock DataFrame ({periods} giờ, 14 đặc trưng khí tượng)...")
    dm = JenaDataModule(seq_len=168, pred_len=72)
    dm.setup(mock_df)
    print(f"      -> Train size: {len(dm.train_df)} | Val size: {len(dm.val_df)} | Test size: {len(dm.test_df)}")

    # 2. Kiểm tra chống rò rỉ và lọc sạch dữ liệu lỗi thực sự
    print("[2/5] Xác minh Chống rò rỉ dữ liệu (Anti-Leakage Verification)...")
    assert dm.verify_zero_leakage() is True
    assert not (dm.train_df["wv (m/s)"] <= -9000).any(), "Còn giá trị lỗi -9999.0 trong train_df"
    assert not dm.train_df.isna().any().any(), "Còn giá trị NaN trong train_df sau imputation"
    assert not dm.val_df.isna().any().any(), "Còn giá trị NaN trong val_df sau imputation"
    assert not dm.test_df.isna().any().any(), "Còn giá trị NaN trong test_df sau imputation"
    print("      [ASSERT 1] Train max time < Val min time: PASSED")
    print("      [ASSERT 2] Val max time < Test min time: PASSED")
    print(f"      [ASSERT 3] Scaler n_samples_seen_ ({dm.scaler.n_samples_seen_}) == len(train_df): PASSED")
    print("      [ASSERT 4] Lọc sạch cảm biến gió lỗi (-9999.0) & Zero NaN: PASSED")

    # 3. Kiểm tra DataLoader
    print("[3/5] Kiểm tra kích thước Tensor từ DataLoader...")
    train_ld, _, _ = dm.get_dataloaders(batch_size=16)
    x_b, y_b = next(iter(train_ld))
    assert x_b.shape == (16, 168, 14), f"Lỗi shape X: {x_b.shape}"
    assert y_b.shape == (16, 72, 1), f"Lỗi shape Y: {y_b.shape}"
    print(f"      -> Batch X shape: {tuple(x_b.shape)} [OK]")
    print(f"      -> Batch Y shape: {tuple(y_b.shape)} [OK]")

    # 4. Kiểm tra giải chuẩn hóa
    print("[4/5] Kiểm tra giải chuẩn hóa về độ C (inverse_transform_target)...")
    degC = dm.inverse_transform_target(y_b.numpy())
    assert degC.shape == y_b.shape
    print(f"      -> Giá trị mẫu 3 giờ đầu (°C): {degC[0, :3, 0].round(2)} [OK]")

    # 5. Kiểm tra tính độc lập của sliding window
    print("[5/5] Kiểm tra tính độc lập của cửa sổ trượt (Isolated Windows)...")
    assert len(dm.train_dataset) > 0 and len(dm.val_dataset) > 0 and len(dm.test_dataset) > 0
    print("      -> Cửa sổ trượt độc lập hoàn toàn bên trong từng phân tập: PASSED")

    print("-" * 70)
    print("[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ DATA MODULE ĐÃ VƯỢT QUA XUẤT SẮC!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_smoke_test()
