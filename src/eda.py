import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _finalize_plot(
    fig: plt.Figure,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
    show_legend: bool = False,
) -> None:
    """Định dạng chung (tiêu đề, nhãn trục, lưới, chú giải) và lưu/hiển thị biểu đồ.
    - Hàm này thực hiện: Gom phần trang trí lặp lại vào 1 hàm dùng chung để code các hàm vẽ bên dưới ngắn gọn, dễ đọc.
    - Input:
        + fig, ax: Đối tượng Figure và Axes của Matplotlib.
        + title, xlabel, ylabel: Tiêu đề và nhãn 2 trục.
        + save_path: Đường dẫn lưu ảnh PNG (None nếu hiển thị trực tiếp trong Notebook).
        + show_legend: Bật/tắt khung chú giải (legend).
    - Output:
        + Không trả về giá trị (lưu PNG hoặc plt.show()).
    """
    if ax is not None:
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)
        if show_legend:
            ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ==============================================================================
# 1. CHẤT LƯỢNG DỮ LIỆU & PHA 1 (SPLIT FIRST)
# ==============================================================================

def plot_wind_anomalies(
    raw_df: pd.DataFrame,
    cleaned_df: Optional[pd.DataFrame] = None,
    invalid_val: float = -9999.0,
    save_path: Optional[str] = None,
) -> None:
    """So sánh tốc độ gió wv (m/s) trước và sau khi lọc giá trị lỗi cảm biến (-9999.0).
    - Hàm này thực hiện: Vẽ 2 biểu đồ Line cạnh nhau để chứng minh sự tồn tại và hiệu quả loại bỏ nhiễu -9999.0.
    - Input:
        + raw_df: DataFrame gốc chứa cột 'wv (m/s)'.
        + cleaned_df: DataFrame đã lọc (tùy chọn; nếu None tự thay invalid_val -> NaN).
        + invalid_val: Giá trị lỗi cảm biến cần lọc (mặc định -9999.0).
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    col = "wv (m/s)"
    raw_s = raw_df[col]
    clean_s = raw_s.replace(invalid_val, np.nan) if cleaned_df is None else cleaned_df[col]
    n_err = int((raw_s == invalid_val).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(raw_s.values, color="#d62728", lw=0.7)
    ax1.set_title(f"Trước lọc: {col} ({n_err} điểm lỗi = {invalid_val})")
    ax1.set_xlabel("Mẫu thời gian")
    ax1.set_ylabel("Tốc độ gió (m/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(clean_s.values, color="#2ca02c", lw=0.7)
    ax2.set_title(f"Sau lọc: [{clean_s.min():.1f}, {clean_s.max():.1f}] m/s")
    ax2.set_xlabel("Mẫu thời gian")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Kiểm tra cảm biến: Loại bỏ giá trị gió lỗi (-9999.0)", fontsize=13, fontweight="bold")
    _finalize_plot(fig, save_path=save_path)


def plot_data_split_timeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "T (degC)",
    save_path: Optional[str] = None,
) -> None:
    """Trực quan hóa Pha 1 (Split First): Chia tuần tự 70% Train - 15% Val - 15% Test.
    - Hàm này thực hiện: Vẽ 3 tập dữ liệu nối tiếp theo trục thời gian kèm 2 đường ranh giới để chứng minh Zero Time Leakage.
    - Input:
        + train_df, val_df, test_df: 3 DataFrame đã chia tuần tự theo thời gian.
        + target_col: Cột nhiệt độ mục tiêu (mặc định 'T (degC)').
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    fig, ax = plt.subplots(figsize=(14, 4.5))
    splits = [
        (train_df, "#1f77b4", f"Train 70% ({len(train_df):,}h)"),
        (val_df, "#ff7f0e", f"Val 15% ({len(val_df):,}h)"),
        (test_df, "#2ca02c", f"Test 15% ({len(test_df):,}h)"),
    ]
    for df_part, color, label in splits:
        ax.plot(df_part.index, df_part[target_col], color=color, lw=0.7, alpha=0.85, label=label)

    ax.axvline(val_df.index.min(), color="#d62728", ls="--", lw=1.5, label="Ranh giới Train | Val")
    ax.axvline(test_df.index.min(), color="#9467bd", ls="--", lw=1.5, label="Ranh giới Val | Test")
    _finalize_plot(
        fig, ax,
        title="Pha 1 (Split First): Phân chia tuần tự theo thời gian (Zero Time Leakage)",
        xlabel="Thời gian", ylabel="Nhiệt độ (°C)",
        save_path=save_path, show_legend=True,
    )


# ==============================================================================
# 2. PHA 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU CHỈ TRÊN TẬP TRAIN (6 BIỂU ĐỒ CỐT LÕI)
# ==============================================================================

def plot_temperature_trend(
    train_df: pd.DataFrame,
    target_col: str = "T (degC)",
    save_path: Optional[str] = None,
) -> None:
    """Vẽ xu hướng biến thiên nhiệt độ dài hạn theo thời gian trên tập Train.
    - Hàm này thực hiện: Vẽ chuỗi nhiệt độ 1h, đường trung bình trượt 30 ngày (720h) và đường Mean để thấy rõ chu kỳ mùa.
    - Input:
        + train_df: DataFrame tập Train, shape (N_train, 14).
        + target_col: Cột nhiệt độ (mặc định 'T (degC)').
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    s = train_df[target_col]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(train_df.index, s, color="tab:blue", lw=0.6, alpha=0.55, label="Nhiệt độ 1h")

    if len(train_df) >= 720:
        ax.plot(train_df.index, s.rolling(720, min_periods=1).mean(), color="#d62728", lw=2.0, label="Trung bình trượt 30 ngày (720h)")

    ax.axhline(s.mean(), color="#2ca02c", ls="--", lw=1.3, label=f"Mean: {s.mean():.2f}°C (Min {s.min():.1f}°C | Max {s.max():.1f}°C)")
    _finalize_plot(
        fig, ax,
        title="Xu hướng nhiệt độ T (degC) theo thời gian (Chỉ trên tập Train)",
        xlabel="Thời gian", ylabel="Nhiệt độ (°C)",
        save_path=save_path, show_legend=True,
    )


def plot_temperature_distribution(
    train_df: pd.DataFrame,
    target_col: str = "T (degC)",
    save_path: Optional[str] = None,
) -> None:
    """Vẽ phân phối (Boxplot + Histogram/KDE) của nhiệt độ T (degC) trên tập Train.
    - Hàm này thực hiện: Trực quan hóa hình dạng phân phối, độ lệch (Skew), ngoại lai, Mean và Median của biến mục tiêu.
    - Input:
        + train_df: DataFrame tập Train, shape (N_train, 14).
        + target_col: Cột nhiệt độ (mặc định 'T (degC)').
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    s = train_df[target_col].dropna()
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [0.22, 0.78]})

    sns.boxplot(x=s, ax=ax_box, color="#6baed6", fliersize=3)
    ax_box.set_title(f"Phân phối {target_col} (Mean={s.mean():.2f}°C, Std={s.std():.2f}°C, Skew={s.skew():.2f})", fontweight="bold")
    ax_box.set_xlabel("")

    sns.histplot(s, bins=50, kde=True, ax=ax_hist, color="#3182bd", alpha=0.55)
    ax_hist.axvline(s.mean(), color="#d62728", ls="--", lw=1.8, label=f"Mean: {s.mean():.2f}°C")
    ax_hist.axvline(s.median(), color="#2ca02c", ls="-.", lw=1.8, label=f"Median: {s.median():.2f}°C")
    _finalize_plot(fig, ax_hist, xlabel="Nhiệt độ (°C)", ylabel="Số giờ", save_path=save_path, show_legend=True)


def plot_diurnal_cycle(
    train_df: pd.DataFrame,
    target_col: str = "T (degC)",
    save_path: Optional[str] = None,
) -> None:
    """Phân tích chu kỳ dao động nhiệt độ ngày đêm theo 24 giờ trên tập Train.
    - Hàm này thực hiện: Gom nhóm theo giờ (0–23h), vẽ Mean ± 1 Std và đánh dấu giờ nóng nhất / lạnh nhất trong ngày.
    - Input:
        + train_df: DataFrame tập Train có DateTimeIndex.
        + target_col: Cột nhiệt độ (mặc định 'T (degC)').
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    stats = train_df.groupby(train_df.index.hour)[target_col].agg(["mean", "std"]).fillna(0.0)
    hours = stats.index.to_numpy()
    peak_h, trough_h = int(stats["mean"].idxmax()), int(stats["mean"].idxmin())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hours, stats["mean"], marker="o", color="crimson", lw=2.0, label="Nhiệt độ trung bình")
    ax.fill_between(hours, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color="crimson", alpha=0.18, label="±1 Std")
    ax.scatter([peak_h, trough_h], [stats.loc[peak_h, "mean"], stats.loc[trough_h, "mean"]], color=["#b2182b", "#2166ac"], s=70, zorder=5)
    ax.set_xticks(hours)
    _finalize_plot(
        fig, ax,
        title=f"Chu kỳ ngày đêm 24h (Đáy: {trough_h}h = {stats.loc[trough_h, 'mean']:.1f}°C | Đỉnh: {peak_h}h = {stats.loc[peak_h, 'mean']:.1f}°C)",
        xlabel="Giờ trong ngày (0 - 23h)", ylabel="Nhiệt độ (°C)",
        save_path=save_path, show_legend=True,
    )


def plot_correlation_heatmap(
    train_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Vẽ ma trận tương quan Pearson giữa 14 đặc trưng khí tượng trên tập Train.
    - Hàm này thực hiện: Tính train_df.corr() và vẽ bản đồ nhiệt Heatmap để phát hiện đa cộng tuyến và các biến ảnh hưởng mạnh đến T.
    - Input:
        + train_df: DataFrame tập Train, shape (N_train, 14).
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True, lw=0.5, ax=ax)
    ax.set_title("Ma trận tương quan Pearson giữa 14 đặc trưng khí tượng (Tập Train)", fontsize=12, fontweight="bold")
    _finalize_plot(fig, save_path=save_path)


def plot_scatter_relationships(
    train_df: pd.DataFrame,
    target_col: str = "T (degC)",
    save_path: Optional[str] = None,
) -> None:
    """Vẽ 3 biểu đồ phân tán (Scatter) giữa T (degC) với VPmax, rho và rh trên tập Train.
    - Hàm này thực hiện: Lấy mẫu ngẫu nhiên 3.000 điểm trên tập Train để chỉ rõ quan hệ phi tuyến (VPmax) và tuyến tính nghịch (rho, rh).
    - Input:
        + train_df: DataFrame tập Train, shape (N_train, 14).
        + target_col: Cột nhiệt độ mục tiêu (mặc định 'T (degC)').
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    clean = train_df.dropna()
    sample = clean.sample(n=min(3000, len(clean)), random_state=42)
    pairs = [
        ("VPmax (mbar)", "#d62728", "Phi tuyến hàm mũ"),
        ("rho (g/m**3)", "#1f77b4", "Tuyến tính nghịch chặt"),
        ("rh (%)", "#2ca02c", "Nghịch biến, phân tán rộng"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, color, desc) in zip(axes, pairs):
        r = sample[target_col].corr(sample[col])
        ax.scatter(sample[target_col], sample[col], s=10, alpha=0.35, color=color)
        ax.set_title(f"{target_col} vs {col}\n(r = {r:+.2f} | {desc})", fontsize=10, fontweight="bold")
        ax.set_xlabel(target_col)
        ax.set_ylabel(col)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Biểu đồ phân tán (Scatter): Quan hệ tuyến tính & phi tuyến với nhiệt độ (Tập Train)", fontsize=12, fontweight="bold")
    _finalize_plot(fig, save_path=save_path)


def plot_autocorrelation(
    train_df: pd.DataFrame,
    target_col: str = "T (degC)",
    max_lag: int = 168,
    save_path: Optional[str] = None,
) -> None:
    """Vẽ hàm tự tương quan (ACF) của nhiệt độ từ độ trễ 1h đến 168h (7 ngày) trên tập Train.
    - Hàm này thực hiện: Tính hệ số tự tương quan theo từng giờ và đánh dấu các đỉnh bội số 24h (24h..168h) để lý giải chọn cửa sổ 168h -> 72h.
    - Input:
        + train_df: DataFrame tập Train, shape (N_train, 14).
        + target_col: Cột nhiệt độ (mặc định 'T (degC)').
        + max_lag: Độ trễ tối đa theo giờ (mặc định 168h = 7 ngày).
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    s = train_df[target_col].ffill().bfill()
    n_lags = min(max_lag, max(1, len(s) - 2))
    lags = np.arange(1, n_lags + 1)
    acf = np.array([s.autocorr(lag=int(k)) for k in lags])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(lags, acf, color="#1f77b4", lw=2.0, label="Autocorrelation (ACF)")
    for lag_h in range(24, n_lags + 1, 24):
        ax.axvline(lag_h, color="#d62728", ls="--", lw=1.0, alpha=0.6)
        ax.scatter([lag_h], [acf[lag_h - 1]], color="#d62728", s=35, zorder=5)
        ax.text(lag_h, acf[lag_h - 1] + 0.03, f"{lag_h}h\n({acf[lag_h - 1]:.2f})", ha="center", fontsize=8, color="#b2182b", fontweight="bold")

    ax.set_ylim(min(0.0, float(np.nanmin(acf)) - 0.1), 1.1)
    _finalize_plot(
        fig, ax,
        title=f"Hàm tự tương quan (ACF) của {target_col} từ 1h đến {n_lags}h (Tập Train)",
        xlabel="Độ trễ (giờ)", ylabel="ACF",
        save_path=save_path, show_legend=True,
    )


# ==============================================================================
# 3. PHA 3 & PHA 4: CHUẨN HÓA & CỬA SỔ TRƯỢT ĐỘC LẬP
# ==============================================================================

def plot_scaling_comparison(
    train_raw: pd.DataFrame,
    train_scaled: np.ndarray,
    feature_names: List[str],
    selected_features: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """So sánh thang đo đặc trưng trước và sau StandardScaler (chỉ fit trên tập Train).
    - Hàm này thực hiện: Vẽ 2 Boxplot cạnh nhau để chứng minh chuẩn hóa đưa các biến về cùng thang đo Mean ≈ 0, Std ≈ 1.
    - Input:
        + train_raw: DataFrame tập Train gốc.
        + train_scaled: Mảng NumPy tập Train đã chuẩn hóa, shape (N_train, 14).
        + feature_names: Danh sách tên 14 cột đặc trưng.
        + selected_features: Danh sách cột tiêu biểu cần vẽ (tùy chọn).
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    cols = selected_features or [c for c in ["p (mbar)", "rho (g/m**3)", "rh (%)", "T (degC)", "VPmax (mbar)", "wv (m/s)"] if c in feature_names]
    scaled_df = pd.DataFrame(train_scaled, columns=feature_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.boxplot(data=train_raw[cols], ax=ax1, palette="Set2", fliersize=2)
    ax1.set_title("Trước StandardScaler (Đơn vị gốc chênh lệch lớn)")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    sns.boxplot(data=scaled_df[cols], ax=ax2, palette="Set2", fliersize=2)
    ax2.axhline(0, color="#d62728", ls="--", lw=1.2)
    ax2.set_title("Sau StandardScaler (Fit CHỈ trên Train: Mean ≈ 0, Std ≈ 1)")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.suptitle("Pha 3: Hiệu quả chuẩn hóa dữ liệu chống rò rỉ", fontsize=12, fontweight="bold")
    _finalize_plot(fig, save_path=save_path)


def plot_sliding_window_example(
    train_df: pd.DataFrame,
    seq_len: int = 168,
    pred_len: int = 72,
    start_idx: int = 0,
    target_col: str = "T (degC)",
    save_path: Optional[str] = None,
) -> None:
    """Minh họa 1 cặp mẫu cửa sổ trượt (168 giờ quá khứ -> 72 giờ tương lai) nội bộ tập Train.
    - Hàm này thực hiện: Tô màu vùng đầu vào X (168h) và vùng nhãn dự báo Y (72h) ngăn cách bởi mốc t=0.
    - Input:
        + train_df: DataFrame tập Train.
        + seq_len: Độ dài đầu vào (168h).
        + pred_len: Độ dài dự báo (72h).
        + start_idx: Vị trí bắt đầu cắt mẫu minh họa.
        + target_col: Cột nhiệt độ (mặc định 'T (degC)').
        + save_path: Đường dẫn lưu ảnh PNG (tùy chọn).
    - Output:
        + Không trả về giá trị (hiển thị hoặc lưu file ảnh).
    """
    total = seq_len + pred_len
    if len(train_df) < start_idx + total:
        start_idx = 0
    win = train_df.iloc[start_idx : start_idx + total]
    past, future = win.iloc[:seq_len], win.iloc[seq_len:]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(past.index, past[target_col], color="#1f77b4", lw=2.0, label=f"Đầu vào X ({seq_len}h quá khứ, 14 biến)")
    ax.plot(future.index, future[target_col], color="#d62728", lw=2.0, ls="--", label=f"Nhãn Y ({pred_len}h tương lai, T degC)")
    ax.axvline(future.index[0], color="#111111", ls=":", lw=1.6, label="Mốc hiện tại (t = 0)")
    ax.axvspan(past.index[0], future.index[0], color="#1f77b4", alpha=0.08)
    ax.axvspan(future.index[0], future.index[-1], color="#d62728", alpha=0.08)

    _finalize_plot(
        fig, ax,
        title=f"Pha 4 (Isolated Sliding Window): Minh họa 1 mẫu ({seq_len}h → {pred_len}h) trong tập Train",
        xlabel="Thời gian", ylabel="Nhiệt độ (°C)",
        save_path=save_path, show_legend=True,
    )


# ==============================================================================
# 4. TỔNG HỢP SỐ LIỆU & PIPELINE XUẤT BIỂU ĐỒ TỰ ĐỘNG
# ==============================================================================

def summarize_eda_insights(
    train_df: pd.DataFrame, target_col: str = "T (degC)"
) -> Dict[str, object]:
    """Tính nhanh các chỉ số định lượng cốt lõi trên tập Train phục vụ thuyết trình Chương 3.
    - Hàm này thực hiện: Tổng hợp Mean/Median/Std/Min/Max, giờ nóng-lạnh nhất, top biến tương quan và ACF.
    - Input:
        + train_df: DataFrame tập Train, shape (N_train, 14).
        + target_col: Cột nhiệt độ mục tiêu (mặc định 'T (degC)').
    - Output:
        + insights: Từ điển các chỉ số thống kê trọng yếu.
    """
    s = train_df[target_col].dropna()
    h_mean = train_df.groupby(train_df.index.hour)[target_col].mean()
    corr = train_df.corr()[target_col].drop(target_col).sort_values(ascending=False)
    filled = train_df[target_col].ffill().bfill()

    return {
        "n_rows_train": len(train_df),
        "missing_total": int(train_df.isna().sum().sum()),
        "temp_mean_degC": round(float(s.mean()), 2),
        "temp_median_degC": round(float(s.median()), 2),
        "temp_std_degC": round(float(s.std()), 2),
        "temp_min_degC": round(float(s.min()), 2),
        "temp_max_degC": round(float(s.max()), 2),
        "temp_skew": round(float(s.skew()), 3),
        "warmest_hour": f"{int(h_mean.idxmax())}h ({h_mean.max():.2f}°C)",
        "coldest_hour": f"{int(h_mean.idxmin())}h ({h_mean.min():.2f}°C)",
        "top_positive_corr": corr.head(3).round(3).to_dict(),
        "top_negative_corr": corr.tail(3).round(3).to_dict(),
        "autocorrelation": {f"lag_{k}h": round(float(filled.autocorr(lag=k)), 4) for k in [24, 48, 72, 168] if len(filled) > k + 1},
    }


def run_eda_pipeline(
    train_df: pd.DataFrame,
    output_dir: str = "evalution_results/eda",
    include_extended: bool = True,
    dm: Optional[object] = None,
) -> List[str]:
    """Chạy tự động đường ống xuất toàn bộ 10 biểu đồ EDA & chống rò rỉ (Zero Leakage).
    - Hàm này thực hiện: Mặc định xuất đủ 6 biểu đồ EDA trên tập Train; nếu truyền thêm dm (JenaDataModule) sẽ xuất đủ cả 10 biểu đồ giống hệt Notebook.
    - Input:
        + train_df: DataFrame 70% tập Train, shape (N_train, 14).
        + output_dir: Thư mục lưu các ảnh PNG đầu ra.
        + include_extended: True (mặc định) xuất toàn bộ biểu đồ cốt lõi; False chỉ xuất 3 ảnh cơ bản.
        + dm: Đối tượng JenaDataModule đã setup (tùy chọn) để xuất thêm 4 biểu đồ quy trình chống rò rỉ.
    - Output:
        + generated_files: Danh sách đường dẫn các file PNG đã tạo.
    """
    os.makedirs(output_dir, exist_ok=True)
    tasks = [
        ("eda_temperature_trend.png", plot_temperature_trend),
        ("eda_correlation_heatmap.png", plot_correlation_heatmap),
        ("eda_diurnal_cycle.png", plot_diurnal_cycle),
    ]
    if include_extended:
        tasks += [
            ("eda_temperature_distribution.png", plot_temperature_distribution),
            ("eda_scatter_relationships.png", plot_scatter_relationships),
            ("eda_autocorrelation.png", plot_autocorrelation),
        ]

    generated_files: List[str] = []
    for filename, plot_fn in tasks:
        path = os.path.join(output_dir, filename)
        plot_fn(train_df, save_path=path)
        generated_files.append(path)

    # Nếu truyền DataModule (từ main.py --mode eda), xuất thêm 4 biểu đồ quy trình chống rò rỉ -> đủ 10 ảnh
    if include_extended and dm is not None:
        if getattr(dm, "raw_csv_path", None) and os.path.exists(dm.raw_csv_path):
            raw_df = pd.read_csv(dm.raw_csv_path, usecols=["wv (m/s)"])
            p_wind = os.path.join(output_dir, "eda_wind_anomalies.png")
            plot_wind_anomalies(raw_df, invalid_val=getattr(dm, "invalid_wind_val", -9999.0), save_path=p_wind)
            generated_files.append(p_wind)

        if dm.train_df is not None and dm.val_df is not None and dm.test_df is not None:
            p_split = os.path.join(output_dir, "eda_data_split_timeline.png")
            plot_data_split_timeline(dm.train_df, dm.val_df, dm.test_df, save_path=p_split)
            generated_files.append(p_split)

        if hasattr(dm, "scaler") and hasattr(dm.scaler, "mean_"):
            train_clean = dm.train_df.ffill().bfill()
            train_scaled = dm.scaler.transform(train_clean.values)
            p_scale = os.path.join(output_dir, "eda_scaling_comparison.png")
            plot_scaling_comparison(train_clean, train_scaled, dm.features, save_path=p_scale)
            generated_files.append(p_scale)

        if len(train_df) >= getattr(dm, "seq_len", 168) + getattr(dm, "pred_len", 72):
            p_win = os.path.join(output_dir, "eda_sliding_window_example.png")
            plot_sliding_window_example(
                train_df.ffill().bfill(),
                seq_len=getattr(dm, "seq_len", 168),
                pred_len=getattr(dm, "pred_len", 72),
                start_idx=240,
                save_path=p_win,
            )
            generated_files.append(p_win)

    return generated_files


if __name__ == "__main__":
    print("[TEST] Bắt đầu kiểm thử script EDA trên tập Train giả lập...")
    cols = [
        "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
        "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
        "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)",
    ]
    mock_train = pd.DataFrame(np.random.randn(500, 14), index=pd.date_range("2012-01-01", periods=500, freq="1h"), columns=cols)

    test_out = "evalution_results/test_eda"
    files = run_eda_pipeline(mock_train, output_dir=test_out)
    insights = summarize_eda_insights(mock_train)

    assert len(files) == 6 and "temp_mean_degC" in insights
    for f in files:
        assert os.path.exists(f)
        os.remove(f)
    if os.path.exists(test_out):
        os.rmdir(test_out)

    print("[SUCCESS] Toàn bộ bài kiểm tra cho EDA pipeline đã vượt qua xuất sắc!")
