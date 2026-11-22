"""Script điều phối toàn bộ các bài kiểm thử (Master Test Runner).
Cho phép chạy nhanh toàn bộ Smoke Tests hoặc Unit Tests của dự án Topic 5.
"""

import argparse
import os
import sys
import time
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def run_all_smoke_tests() -> int:
    """Chạy tuần tự toàn bộ các bài kiểm thử khói (Smoke Tests) cho các module trong src/."""
    from tests.test_data_module import run_smoke_test as test_dm
    from tests.test_model_builder import run_smoke_test as test_mb
    from tests.test_trainer import run_smoke_test as test_tr
    from tests.test_evaluation import run_smoke_test as test_ev
    from tests.test_logger import run_smoke_test as test_lg
    from tests.test_eda import run_smoke_test as test_ed

    tests = [
        ("DataModule (Anti-Leakage & Sliding Windows)", test_dm),
        ("ModelBuilder (3 Architectures & Factory)", test_mb),
        ("ModelTrainer (Early Stopping & Checkpoints)", test_tr),
        ("ModelEvaluator (Metrics & Latency Benchmark)", test_ev),
        ("TensorBoardLogger (Scalars & Heatmaps)", test_lg),
        ("EDA Pipeline (Train-only Analysis)", test_ed),
    ]

    print("=" * 80)
    print("   BỘ KIỂM THỬ KHÓI TỔNG THỂ (MASTER SMOKE TESTS) - TOPIC 5")
    print("   Dự án: Sequence-to-Sequence Weather Forecasting (Jena Climate)")
    print("=" * 80)

    start_total = time.perf_counter()
    results = []

    for idx, (name, test_func) in enumerate(tests, 1):
        print(f"\n>>> [{idx}/{len(tests)}] THỰC THI KIỂM THỬ: {name}")
        t0 = time.perf_counter()
        try:
            test_func()
            elapsed = (time.perf_counter() - t0) * 1000.0
            results.append((name, "PASSED", elapsed))
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            results.append((name, f"FAILED: {e}", elapsed))
            print(f"[ERROR] Kiểm thử '{name}' thất bại: {e}")

    total_time_s = time.perf_counter() - start_total

    print("\n" + "=" * 80)
    print("                     BẢNG TỔNG HỢP KẾT QUẢ KIỂM THỬ")
    print("=" * 80)
    print(f"{'Mô-đun kiểm thử':<50} | {'Trạng thái':<12} | {'Thời gian (ms)':<15}")
    print("-" * 80)

    all_passed = True
    for name, status, elapsed in results:
        status_str = "[OK] PASSED" if status == "PASSED" else "[X] FAILED"
        if status != "PASSED":
            all_passed = False
        print(f"{name:<50} | {status_str:<12} | {elapsed:>12.2f} ms")

    print("=" * 80)
    print(f"Tổng thời gian thực thi: {total_time_s:.2f} giây")

    if all_passed:
        print("[FINAL RESULT] TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ VƯỢT QUA XUẤT SẮC! (100% PASSED)")
        return 0
    else:
        print("[FINAL RESULT] CÓ BÀI KIỂM THỬ THẤT BẠI. VUI LÒNG KIỂM TRA LỖI TRÊN!")
        return 1


def run_unit_tests() -> int:
    """Chạy toàn bộ unittest test suite bằng TestLoader chuẩn của Python."""
    print("=" * 80)
    print("   CHẠY BỘ UNIT TESTS TỰ ĐỘNG (PYTHON UNITTEST DISCOVERY)")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description="Chạy kiểm thử cho dự án Topic 5")
    parser.add_argument(
        "--mode",
        choices=["smoke", "unit", "all"],
        default="smoke",
        help="Chế độ kiểm thử: 'smoke' (kiểm thử nhanh), 'unit' (unittest), hoặc 'all' (cả hai)",
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        code = run_all_smoke_tests()
    elif args.mode == "unit":
        code = run_unit_tests()
    else:
        c1 = run_all_smoke_tests()
        c2 = run_unit_tests()
        code = 0 if (c1 == 0 and c2 == 0) else 1

    sys.exit(code)


if __name__ == "__main__":
    main()
