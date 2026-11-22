import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_temperature_trend(
    train_df: pd.DataFrame,
    target_col: str = "T (degC)",
    save_path: Optional[str] = None,
) -> None:
    """Vẽ biểu đồ xu hướng biến thiên nhiệt độ theo thời gian trên tập Train.
    - Hàm này thực hiện: Trích xuất chuỗi thời gian của cột nhiệt độ trên 70% tập dữ liệu Train và trực quan hóa tính chu kỳ mùa.
    - Input:
        + train_df: DataFrame chỉ chứa tập dữ liệu huấn luyện, shape (N_train, 14).
        + target_col: Tên cột mục tiêu nhiệt độ (mặc định 'T (degC)').
        + save_path: Đường dẫn lưu ảnh biểu đồ (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh đồ thị).
    """
    plt.figure(figsize=(14, 5))
    plt.plot(train_df.index, train_df[target_col], color="tab:blue", linewidth=0.8, alpha=0.85)
    plt.title("Xu hướng nhiệt độ T (degC) theo dòng thời gian (Chỉ trên tập Train - Zero Leakage)")
    plt.xlabel("Thời gian")
    plt.ylabel("Nhiệt độ (°C)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(
    train_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Vẽ ma trận hệ số tương quan nhiệt (Correlation Heatmap) giữa 14 biến khí tượng trên tập Train.
    - Hàm này thực hiện: Tính ma trận tương quan Pearson giữa 14 thuộc tính thời tiết và vẽ bản đồ nhiệt trực quan.
    - Input:
        + train_df: DataFrame của tập Train, shape (N_train, 14).
        + save_path: Đường dẫn lưu ảnh ma trận tương quan (tùy chọn).
    - Output:
        + Không trả về giá trị (lưu hoặc hiển thị biểu đồ Heatmap).
    """
    corr = train_df.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Ma trận tương quan Pearson giữa 14 đặc trưng khí tượng (Tập Train)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_diurnal_cycle(
    train_df: pd.DataFrame,
    target_col: str = "T (degC)",
    save_path: Optional[str] = None,
) -> None:
    """Phân tích chu kỳ dao động nhiệt độ theo từng giờ trong ngày (24 giờ).
    - Hàm này thực hiện: Gom nhóm nhiệt độ theo giờ trong ngày (0-23h) trên tập Train và vẽ khoảng tin cậy dao động nhiệt.
    - Input:
        + train_df: DataFrame tập Train có chỉ số DateTimeIndex, shape (N_train, 14).
        + target_col: Cột nhiệt độ phân tích (mặc định 'T (degC)').
        + save_path: Đường dẫn lưu file đồ thị (tùy chọn).
    - Output:
        + Không trả về giá trị (lưu hoặc hiển thị biểu đồ).
    """
    df_temp = train_df[[target_col]].copy()
    df_temp["hour"] = df_temp.index.hour

    hourly_stats = df_temp.groupby("hour")[target_col].agg(["mean", "std"])

    plt.figure(figsize=(10, 5))
    hours = np.arange(24)
    plt.plot(hours, hourly_stats["mean"], marker="o", color="crimson", label="Nhiệt độ trung bình")
    plt.fill_between(
        hours,
        hourly_stats["mean"] - hourly_stats["std"],
        hourly_stats["mean"] + hourly_stats["std"],
        color="crimson",
        alpha=0.2,
        label="± 1 Độ lệch chuẩn (Std)",
    )
    plt.title("Chu kỳ ngày đêm: Biến động nhiệt độ theo 24 giờ trong ngày (Tập Train)")
    plt.xlabel("Giờ trong ngày (0 - 23h)")
    plt.ylabel("Nhiệt độ (°C)")
    plt.xticks(hours)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def run_eda_pipeline(
    train_df: pd.DataFrame, output_dir: str = "evalution_results/eda"
) -> List[str]:
    """Chạy toàn bộ đường ống phân tích khám phá dữ liệu tuân thủ quy tắc chống rò rỉ.
    - Hàm này thực hiện: Điều phối tuần tự các tác vụ vẽ xu hướng thời gian, ma trận tương quan và chu kỳ ngày đêm, lưu file vào thư mục đầu ra.
    - Input:
        + train_df: DataFrame 70% tập Train thuần túy, shape (N_train, 14).
        + output_dir: Thư mục lưu các biểu đồ hình ảnh kết quả EDA.
    - Output:
        + generated_files: Danh sách các đường dẫn file ảnh biểu đồ đã sinh ra.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    p1 = os.path.join(output_dir, "eda_temperature_trend.png")
    plot_temperature_trend(train_df, save_path=p1)
    generated_files.append(p1)

    p2 = os.path.join(output_dir, "eda_correlation_heatmap.png")
    plot_correlation_heatmap(train_df, save_path=p2)
    generated_files.append(p2)

    p3 = os.path.join(output_dir, "eda_diurnal_cycle.png")
    plot_diurnal_cycle(train_df, save_path=p3)
    generated_files.append(p3)

    return generated_files


if __name__ == "__main__":
    print("[TEST] Bắt đầu kiểm thử script EDA trên tập Train giả lập...")
    date_range = pd.date_range("2012-01-01", periods=500, freq="1h")
    mock_cols = [
        "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
        "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
        "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"
    ]
    mock_train = pd.DataFrame(np.random.randn(500, 14), index=date_range, columns=mock_cols)

    test_out = "evalution_results/test_eda"
    files = run_eda_pipeline(mock_train, output_dir=test_out)

    assert len(files) == 3
    for f in files:
        assert os.path.exists(f), f"File {f} không được tạo."
        os.remove(f)

    if os.path.exists(test_out):
        os.rmdir(test_out)

    print("[SUCCESS] Toàn bộ bài kiểm tra cho EDA pipeline đã vượt qua xuất sắc!")
