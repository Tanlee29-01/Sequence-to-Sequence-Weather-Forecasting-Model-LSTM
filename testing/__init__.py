"""Thư mục testing cho mã nguồn src/.
Điều phối và ánh xạ tới toàn bộ bộ kiểm thử tự động của dự án.
"""
from tests.run_all_tests import run_all_smoke_tests, run_unit_tests

__all__ = ["run_all_smoke_tests", "run_unit_tests"]
