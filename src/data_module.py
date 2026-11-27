import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset


class SlidingWindowDataset(Dataset):
    """Lớp Dataset dạng cửa sổ trượt (Sliding Window) cho dữ liệu chuỗi thời gian khí tượng.
    - Mục đích: Tạo các cặp mẫu (X: 168 giờ quá khứ, Y: 72 giờ tương lai) một cách độc lập trong từng tập dữ liệu.
    - Dữ liệu bên trong bao gồm: Mảng NumPy chứa 14 đặc trưng thời tiết đã được chuẩn hóa và chỉ số cột mục tiêu T (degC).
    - Bản chất bài toán: Trích xuất các mẫu chuỗi thời gian cho bài toán Hồi quy đa bước mà không bị rò rỉ dữ liệu giữa các phân tập.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 168,
        pred_len: int = 72,
        target_idx: int = 1,
    ):
        """Khởi tạo tập dữ liệu cửa sổ trượt độc lập.
        - Hàm này thực hiện: Lưu trữ mảng dữ liệu, thiết lập độ dài cửa sổ quá khứ, độ dài dự báo tương lai và vị trí cột mục tiêu.
        - Input:
            + data: Mảng dữ liệu thời tiết 2D, shape (N_samples, 14).
            + seq_len: Số bước thời gian đầu vào trong quá khứ (mặc định 168 giờ).
            + pred_len: Số bước thời gian đầu ra cần dự báo (mặc định 72 giờ).
            + target_idx: Chỉ số cột của biến nhiệt độ T (degC) trong 14 đặc trưng (mặc định là 1).
        - Output:
            + Không trả về giá trị (khởi tạo đối tượng Dataset).
        """
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.total_len = seq_len + pred_len

    def __len__(self) -> int:
        """Đo lường tổng số mẫu cửa sổ trượt có thể tạo ra từ tập dữ liệu.
        - Hàm này thực hiện: Tính toán số lượng mẫu hợp lệ thỏa mãn cả cửa sổ quá khứ và nhãn tương lai.
        - Input:
            + Không có (nhận thông tin nội tại từ đối tượng self).
        - Output:
            + Số lượng mẫu nguyên hợp lệ trong tập dữ liệu.
        """
        if len(self.data) < self.total_len:
            return 0
        return len(self.data) - self.total_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trích xuất một cặp mẫu (Đầu vào quá khứ, Nhãn tương lai) tại chỉ số tương ứng.
        - Hàm này thực hiện: Cắt lát mảng dữ liệu để lấy 168 bước quá khứ làm X và 72 bước tương lai của nhiệt độ làm Y.
        - Input:
            + idx: Chỉ mục của mẫu cần lấy, số nguyên trong khoảng [0, len(self) - 1].
        - Output:
            + x: Chuỗi 14 đặc trưng trong 168 giờ quá khứ, shape (168, 14).
            + y: Chuỗi nhiệt độ T (degC) trong 72 giờ tương lai, shape (72, 1).
        """
        x_start = idx
        x_end = x_start + self.seq_len
        y_end = x_end + self.pred_len

        x = self.data[x_start:x_end, :]
        y = self.data[x_end:y_end, self.target_idx : self.target_idx + 1]
        return x, y


class JenaDataModule:
    """Module quản lý toàn bộ vòng đời dữ liệu thời tiết Jena Climate tuân thủ 4 pha chống rò rỉ (Strict Anti-Leakage).
    - Mục đích: Nạp dữ liệu gốc, gộp mẫu 1 giờ (1h), chia Train/Val/Test theo thứ tự thời gian, chuẩn hóa chỉ fit trên Train, và cung cấp DataLoader độc lập.
    - Dữ liệu bên trong bao gồm: 14 thuộc tính khí tượng (p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh, H2OC, rho, wv, max. wv, wd) cùng bộ StandardScaler học từ tập Train.
    - Bản chất bài toán: Đảm bảo tính trung thực khoa học tuyệt đối, ngăn chặn triệt để hiện tượng Data Leakage trước khi huấn luyện mô hình dự báo nhiệt độ 72 giờ.
    """

    DEFAULT_FEATURES: List[str] = [
        "p (mbar)",
        "T (degC)",
        "Tpot (K)",
        "Tdew (degC)",
        "rh (%)",
        "VPmax (mbar)",
        "VPact (mbar)",
        "VPdef (mbar)",
        "sh (g/kg)",
        "H2OC (mmol/mol)",
        "rho (g/m**3)",
        "wv (m/s)",
        "max. wv (m/s)",
        "wd (deg)",
    ]

    def __init__(
        self,
        raw_csv_path: str = "data/jena_climate_2009_2016.csv",
        seq_len: int = 168,
        pred_len: int = 72,
        target_col: str = "T (degC)",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        resample_rule: str = "1h",
        invalid_wind_val: float = -9999.0,
        features: Optional[List[str]] = None,
    ):
        """Khởi tạo Data Module với các tham số phân chia và tiền xử lý dữ liệu thời tiết.
        - Hàm này thực hiện: Thiết lập đường dẫn dữ liệu, tỷ lệ chia tập, độ dài chuỗi vào/ra và khởi tạo bộ scaler.
        - Input:
            + raw_csv_path: Đường dẫn tới file CSV Jena Climate gốc.
            + seq_len: Độ dài chuỗi quá khứ đầu vào (168 giờ).
            + pred_len: Độ dài chuỗi nhiệt độ cần dự báo (72 giờ).
            + target_col: Tên cột mục tiêu cần dự báo ("T (degC)").
            + train_ratio: Tỷ lệ phân bổ tập huấn luyện (mặc định 0.70).
            + val_ratio: Tỷ lệ phân bổ tập kiểm định (mặc định 0.15).
            + test_ratio: Tỷ lệ phân bổ tập kiểm thử (mặc định 0.15).
            + resample_rule: Quy tắc gom mẫu thời gian (mặc định '1h').
            + invalid_wind_val: Giá trị lỗi của cảm biến đo gió cần lọc (mặc định -9999.0).
            + features: Danh sách cột đặc trưng khí tượng cần giữ (mặc định 14 cột DEFAULT_FEATURES).
        - Output:
            + Không trả về giá trị (khởi tạo đối tượng DataModule).
        """
        self.raw_csv_path = raw_csv_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.resample_rule = resample_rule
        self.invalid_wind_val = invalid_wind_val

        self.features: List[str] = list(features) if features is not None else list(self.DEFAULT_FEATURES)
        if self.target_col not in self.features:
            raise ValueError(
                f"Cột mục tiêu '{self.target_col}' không nằm trong danh sách đặc trưng: {self.features}"
            )
        self.target_idx: int = self.features.index(self.target_col)

        self.scaler: StandardScaler = StandardScaler()
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.train_dataset: Optional[SlidingWindowDataset] = None
        self.val_dataset: Optional[SlidingWindowDataset] = None
        self.test_dataset: Optional[SlidingWindowDataset] = None

    def load_and_resample(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Đọc file CSV gốc, xử lý giá trị cảm biến lỗi và gom mẫu theo chu kỳ 1 giờ.
        - Hàm này thực hiện: Chuyển đổi chỉ số thời gian Date Time, thay thế giá trị đo gió âm -9999.0 bằng NaN, và tính trung bình theo từng giờ độc lập.
        - Input:
            + df: DataFrame tùy chọn (nếu truyền vào để kiểm thử), shape (N_raw_rows, 15). Nếu None, đọc từ raw_csv_path.
        - Output:
            + hourly_df: DataFrame sau khi resample 1 giờ với 14 thuộc tính khí tượng, shape (~70091, 14).
        """
        if df is None:
            if not os.path.exists(self.raw_csv_path):
                raise FileNotFoundError(
                    f"Không tìm thấy file dữ liệu tại {self.raw_csv_path}."
                )
            df = pd.read_csv(self.raw_csv_path)

        date_time_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_time_col = col
                break

        if date_time_col is not None:
            # File Jena Climate lưu theo định dạng Châu Âu: DD.MM.YYYY HH:MM:SS (cần format rõ để tránh lỗi tháng > 12)
            try:
                df[date_time_col] = pd.to_datetime(
                    df[date_time_col], format="%d.%m.%Y %H:%M:%S"
                )
            except Exception:
                df[date_time_col] = pd.to_datetime(
                    df[date_time_col], dayfirst=True, format="mixed"
                )
            df = df.set_index(date_time_col)

        # Lọc giá trị cảm biến gió hỏng
        for col in ["wv (m/s)", "max. wv (m/s)"]:
            if col in df.columns:
                df[col] = df[col].replace(self.invalid_wind_val, np.nan)

        # Giữ đúng 14 đặc trưng
        available_cols = [c for c in self.features if c in df.columns]
        df = df[available_cols]

        # Resample trung bình từng giờ
        hourly_df = df.resample(self.resample_rule).mean()
        return hourly_df

    def split_data(
        self, hourly_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Pha 1: Chia tách dữ liệu theo đúng thứ tự dòng thời gian trước khi thực hiện bất kỳ phép biến đổi nào.
        - Hàm này thực hiện: Cắt dữ liệu thành 3 phần tuần tự 70% Train, 15% Val và 15% Test để ngăn chặn rò rỉ tương lai.
        - Input:
            + hourly_df: DataFrame thời tiết 1 giờ hoàn chỉnh, shape (N_hours, 14).
        - Output:
            + train_df: Dữ liệu tập huấn luyện (70% đầu), shape (N_train, 14).
            + val_df: Dữ liệu tập kiểm định (15% kế tiếp), shape (N_val, 14).
            + test_df: Dữ liệu tập kiểm thử (15% cuối cùng), shape (N_test, 14).
        """
        n = len(hourly_df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        self.train_df = hourly_df.iloc[:train_end].copy()
        self.val_df = hourly_df.iloc[train_end:val_end].copy()
        self.test_df = hourly_df.iloc[val_end:].copy()

        return self.train_df, self.val_df, self.test_df

    def fit_scaler_and_impute(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pha 3: Điền khuyết nội bộ từng tập và chuẩn hóa chỉ học tham số từ tập Train.
        - Hàm này thực hiện: Dùng ffill trong nội bộ từng phân tập để xử lý NaN, gọi .fit() của StandardScaler trên duy nhất tập Train, sau đó .transform() cho cả 3 tập.
        - Input:
            + Không nhận trực tiếp tham số (sử dụng self.train_df, self.val_df, self.test_df đã chia ở Pha 1).
        - Output:
            + train_scaled: Mảng dữ liệu Train đã chuẩn hóa, shape (N_train, 14).
            + val_scaled: Mảng dữ liệu Val đã chuẩn hóa, shape (N_val, 14).
            + test_scaled: Mảng dữ liệu Test đã chuẩn hóa, shape (N_test, 14).
        """
        if (
            self.train_df is None
            or self.val_df is None
            or self.test_df is None
        ):
            raise ValueError(
                "Dữ liệu chưa được phân chia! Vui lòng gọi split_data() trước."
            )

        # Xử lý ô trống nội bộ từng tập:
        # - Train: ffill theo chiều thời gian, chỉ bfill cho các hàng đầu của riêng Train.
        # - Val & Test: CHỈ ffill theo chiều quá khứ; nếu hàng đầu vẫn là NaN thì điền bằng trung vị Train
        #   (tuyệt đối không dùng bfill trên Val/Test để tránh nhìn trước tương lai trong nội bộ tập).
        train_imputed = self.train_df.ffill().bfill()
        train_fill_values = train_imputed.median()
        val_imputed = self.val_df.ffill().fillna(train_fill_values)
        test_imputed = self.test_df.ffill().fillna(train_fill_values)

        # Cập nhật lại các DataFrame nội tại để đồng nhất giữa EDA, kiểm thử và scaler
        self.train_df = train_imputed
        self.val_df = val_imputed
        self.test_df = test_imputed

        train_clean = self.train_df.values
        val_clean = self.val_df.values
        test_clean = self.test_df.values

        # CHỈ FIT TRÊN TẬP HUẤN LUYỆN
        self.scaler.fit(train_clean)

        # Áp dụng tham số từ tập Train sang Val và Test
        train_scaled = self.scaler.transform(train_clean)
        val_scaled = self.scaler.transform(val_clean)
        test_scaled = self.scaler.transform(test_clean)

        return train_scaled, val_scaled, test_scaled

    def setup(
        self, df: Optional[pd.DataFrame] = None
    ) -> None:
        """Thiết lập toàn bộ quy trình tiền xử lý 4 pha chống rò rỉ dữ liệu.
        - Hàm này thực hiện: Điều phối tuần tự Resample -> Split -> Fit Scaler -> Tạo Dataset cửa sổ trượt.
        - Input:
            + df: DataFrame dữ liệu đầu vào tùy chọn (dùng khi kiểm thử hoặc nạp trước).
        - Output:
            + Không trả về giá trị (lưu trữ dataset vào các thuộc tính nội tại của class).
        """
        hourly_df = self.load_and_resample(df)
        self.split_data(hourly_df)
        train_s, val_s, test_s = self.fit_scaler_and_impute()

        # Pha 4: Cắt cửa sổ trượt riêng biệt
        self.train_dataset = SlidingWindowDataset(
            train_s, self.seq_len, self.pred_len, self.target_idx
        )
        self.val_dataset = SlidingWindowDataset(
            val_s, self.seq_len, self.pred_len, self.target_idx
        )
        self.test_dataset = SlidingWindowDataset(
            test_s, self.seq_len, self.pred_len, self.target_idx
        )

    def get_dataloaders(
        self, batch_size: int = 64, num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Tạo các DataLoader cho Train, Validation và Test.
        - Hàm này thực hiện: Đóng gói các Dataset thành DataLoader với batch_size được chỉ định, chỉ shuffle trên tập Train.
        - Input:
            + batch_size: Kích thước mỗi batch dữ liệu (mặc định 64).
            + num_workers: Số luồng phụ xử lý dữ liệu (mặc định 0 trên Windows).
        - Output:
            + train_loader: DataLoader cho tập Train (shuffle=True).
            + val_loader: DataLoader cho tập Validation (shuffle=False).
            + test_loader: DataLoader cho tập Test (shuffle=False).
        """
        if self.train_dataset is None:
            self.setup()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader, test_loader

    def inverse_transform_target(
        self, scaled_target: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Chuyển đổi nhiệt độ đã chuẩn hóa về lại độ C (°C) thực tế.
        - Hàm này thực hiện: Áp dụng công thức giải chuẩn hóa (y * std + mean) dựa trên tham số scaler của biến nhiệt độ T (degC).
        - Input:
            + scaled_target: Tensor hoặc mảng NumPy chứa nhiệt độ chuẩn hóa, shape (Batch_size, 72, 1) hoặc (Batch_size, 72).
        - Output:
            + original_temp: Mảng NumPy nhiệt độ thực tế theo độ C, cùng kích thước với đầu vào.
        """
        if isinstance(scaled_target, torch.Tensor):
            scaled_target = scaled_target.detach().cpu().numpy()

        target_mean = self.scaler.mean_[self.target_idx]
        target_scale = self.scaler.scale_[self.target_idx]

        return scaled_target * target_scale + target_mean

    def verify_zero_leakage(self) -> bool:
        """Kiểm tra và chứng minh không rò rỉ dữ liệu (Zero Leakage Check).
        - Hàm này thực hiện: Tự động kiểm định 2 tiêu chí khắt khe: thời gian train < val < test và scaler chỉ đếm đúng số dòng của train_df.
        - Input:
            + Không có tham số đầu vào.
        - Output:
            + True nếu vượt qua toàn bộ các phép kiểm tra chống rò rỉ, ngược lại gây ngoại lệ AssertionError.
        """
        if (
            self.train_df is None
            or self.val_df is None
            or self.test_df is None
        ):
            raise AssertionError("Các tập dữ liệu chưa được khởi tạo đầy đủ.")

        # 1. Kiểm tra tính tuần tự thời gian
        train_max_time = self.train_df.index.max()
        val_min_time = self.val_df.index.min()
        val_max_time = self.val_df.index.max()
        test_min_time = self.test_df.index.min()

        assert (
            train_max_time < val_min_time
        ), f"Lộ đề: Thời gian Train ({train_max_time}) >= Val ({val_min_time})"
        assert (
            val_max_time < test_min_time
        ), f"Lộ đề: Thời gian Val ({val_max_time}) >= Test ({test_min_time})"

        # 2. Kiểm tra bộ Scaler chỉ học trên đúng số mẫu của tập Train
        expected_samples = len(self.train_df)
        actual_samples = self.scaler.n_samples_seen_
        assert (
            actual_samples == expected_samples
        ), f"Scaler Leakage: Số mẫu fit ({actual_samples}) != Số dòng Train ({expected_samples})"

        return True


if __name__ == "__main__":
    print("[TEST] Bắt đầu kiểm thử DataModule và quy tắc chống rò rỉ dữ liệu...")

    # Tạo DataFrame giả lập cấu trúc Jena Climate (14 features, đo theo giờ)
    date_range = pd.date_range("2010-01-01", periods=1000, freq="1h")
    mock_data = np.random.randn(1000, 14)
    # Cài giá trị lỗi để test bộ lọc
    mock_data[10, 11] = -9999.0
    mock_df = pd.DataFrame(
        mock_data, index=date_range, columns=JenaDataModule.DEFAULT_FEATURES
    )

    dm = JenaDataModule(seq_len=168, pred_len=72)
    dm.setup(mock_df)

    # 1. Kiểm tra không rò rỉ dữ liệu
    assert dm.verify_zero_leakage() is True
    print("  -> Vượt qua kiểm tra Zero Data Leakage thành công.")

    # 2. Kiểm tra kích thước mẫu trong DataLoader
    train_loader, val_loader, test_loader = dm.get_dataloaders(batch_size=16)
    x_batch, y_batch = next(iter(train_loader))

    assert x_batch.shape == (
        16,
        168,
        14,
    ), f"Shape X không đúng: {x_batch.shape}"
    assert y_batch.shape == (
        16,
        72,
        1,
    ), f"Shape Y không đúng: {y_batch.shape}"
    print(f"  -> Batch X shape: {x_batch.shape} | Batch Y shape: {y_batch.shape}")

    # 3. Kiểm tra giải chuẩn hóa nhiệt độ
    sample_y = y_batch[:2]
    original_degC = dm.inverse_transform_target(sample_y)
    assert original_degC.shape == sample_y.shape
    print(
        f"  -> Giải chuẩn hóa thành công, giá trị mẫu (°C): {original_degC[0, :3, 0]}"
    )

    print("[SUCCESS] Toàn bộ bài kiểm tra cho DataModule đã vượt qua xuất sắc!")
