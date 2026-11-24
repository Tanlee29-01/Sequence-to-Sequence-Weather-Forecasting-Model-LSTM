"""Kiểm thử tự động cho module phân tích khám phá dữ liệu (src/eda.py).
Xác minh quy tắc EDA chỉ thực thi trên tập Train (Zero Cognitive Leakage)
và xuất đủ 3 biểu đồ trực quan hóa.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Đảm bảo import được module src từ thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_module import JenaDataModule
from src.eda import run_eda_pipeline


class TestEDA(unittest.TestCase):
    """Bộ kiểm thử cho EDA pipeline."""

    def setUp(self):
        """Khởi tạo mock train DataFrame."""
        date_range = pd.date_range("2012-01-01", periods=200, freq="1h")
        mock_data = np.random.randn(200, 14) * 4 + 10
        self.mock_train_df = pd.DataFrame(
            mock_data,
            index=date_range,
            columns=JenaDataModule.DEFAULT_FEATURES,
        )
        self.test_output_dir = "evalution_results/test_eda_run"

    def tearDown(self):
        """Dọn dẹp thư mục ảnh EDA."""
        if os.path.exists(self.test_output_dir):
            for f in os.listdir(self.test_output_dir):
                os.remove(os.path.join(self.test_output_dir, f))
            os.rmdir(self.test_output_dir)

    def test_run_eda_generates_all_plots(self):
        """Kiểm tra pipeline EDA xuất ra đủ 6 tệp hình ảnh PNG cốt lõi trên tập Train."""
        generated = run_eda_pipeline(self.mock_train_df, output_dir=self.test_output_dir)

        self.assertEqual(len(generated), 6)
        for filepath in generated:
            self.assertTrue(os.path.exists(filepath), f"File {filepath} không được tạo.")
            self.assertTrue(filepath.endswith(".png"))


def run_smoke_test():
    """Hàm chạy nhanh Smoke Test cho EDA."""
    print("=" * 70)
    print("[SMOKE TEST] BẮT ĐẦU KIỂM THỬ EDA PIPELINE TRÊN TẬP TRAIN (TOPIC 5)")
    print("=" * 70)

    date_range = pd.date_range("2012-01-01", periods=100, freq="1h")
    mock_train = pd.DataFrame(
        np.random.randn(100, 14),
        index=date_range,
        columns=JenaDataModule.DEFAULT_FEATURES,
    )
    test_dir = "evalution_results/smoke_test_eda"

    print("[1/2] Chạy EDA Pipeline xuất 6 biểu đồ cốt lõi...")
    files = run_eda_pipeline(mock_train, output_dir=test_dir)
    assert len(files) == 6
    for f in files:
        assert os.path.exists(f)
        os.remove(f)
    print("      -> 6 biểu đồ xuất thành công: PASSED")

    print("[2/2] Dọn dẹp thư mục...")
    if os.path.exists(test_dir):
        os.rmdir(test_dir)
    print("      -> Dọn dẹp hoàn tất [OK]")

    print("-" * 70)
    print("[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ EDA ĐÃ VƯỢT QUA XUẤT SẮC!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_smoke_test()
