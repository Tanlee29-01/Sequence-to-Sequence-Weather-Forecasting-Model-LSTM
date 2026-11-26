import argparse
from datetime import datetime
import os
import sys
from typing import Dict, Optional
import torch
import torch.nn as nn
import yaml

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from src.data_module import JenaDataModule
from src.eda import run_eda_pipeline
from src.evaluation import ModelEvaluator
from src.logger import TensorBoardLogger
from src.model_builder import build_model
from src.trainer import ModelTrainer


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Nạp file cấu hình hệ thống YAML phục vụ toàn bộ quy trình thực nghiệm.
    - Hàm này thực hiện: Đọc tệp cấu hình .yaml và phân tích cú pháp thành một từ điển Python chuẩn.
    - Input:
        + config_path: Đường dẫn chuỗi tới file config.yaml (mặc định 'configs/config.yaml').
    - Output:
        + config: Từ điển chứa toàn bộ siêu tham số của dữ liệu, mô hình và quy trình huấn luyện.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại {config_path}.")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_str: str = "auto") -> torch.device:
    """Xác định và trả về thiết bị tính toán tối ưu nhất (CUDA GPU hoặc CPU).
    - Hàm này thực hiện: Kiểm tra khả năng tương thích phần cứng CUDA và lựa chọn thiết bị theo yêu cầu hoặc tự động.
    - Input:
        + device_str: Chuỗi cấu hình thiết bị ('auto', 'cuda', hoặc 'cpu').
    - Output:
        + device: Đối tượng torch.device tương ứng.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def resolve_experiment_name(
    runs_dir: str,
    model_name: str,
    mode: str,
    run_name: Optional[str] = None,
) -> str:
    """Tạo tên thư mục run tách biệt cho mỗi lần chạy trên TensorBoard.
    - Hàm này thực hiện: Gắn nhãn run_name hoặc timestamp YYYYMMDD-HHMMSS để các lần huấn luyện không ghi đè lên nhau; riêng chế độ evaluate sẽ gắn vào run mới nhất của mô hình nếu không chỉ định run_name.
    - Input:
        + runs_dir: Thư mục gốc chứa log TensorBoard (mặc định 'runs').
        + model_name: Tên kiến trúc mô hình.
        + mode: Chế độ chạy ('train', 'evaluate', 'all', ...).
        + run_name: Tên phiên thực nghiệm do người dùng truyền qua CLI (tùy chọn).
    - Output:
        + experiment_name: Tên thư mục con duy nhất trong runs/.
    """
    if run_name:
        return f"{model_name}_{run_name}"

    if mode == "evaluate" and os.path.exists(runs_dir):
        candidates = [
            d for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith(f"{model_name}_")
        ]
        if candidates:
            candidates.sort(key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
            return candidates[-1]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_{timestamp}"


def resolve_checkpoint_path(
    model_name: str,
    checkpoints_dir: str,
    explicit_path: Optional[str] = None,
    trained_checkpoint: Optional[str] = None,
) -> str:
    """Resolve a checkpoint without silently selecting ambiguous or random weights."""
    if trained_checkpoint is not None:
        checkpoint_path = trained_checkpoint
    elif explicit_path:
        checkpoint_path = explicit_path
    else:
        candidates = [
            os.path.join(checkpoints_dir, filename)
            for filename in os.listdir(checkpoints_dir)
            if filename.startswith(f"{model_name}_")
            and filename.endswith("_best.pth")
        ] if os.path.isdir(checkpoints_dir) else []

        if not candidates:
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint tốt nhất cho '{model_name}' trong "
                f"'{checkpoints_dir}'. Hãy truyền --checkpoint <đường_dẫn_checkpoint>."
            )
        if len(candidates) > 1:
            candidate_list = "\n  - ".join(sorted(candidates))
            raise ValueError(
                f"Có nhiều checkpoint cho '{model_name}', không thể xác định run "
                f"đúng một cách an toàn. Hãy truyền --checkpoint:\n  - {candidate_list}"
            )
        checkpoint_path = candidates[0]

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại '{checkpoint_path}'.")
    return checkpoint_path


def run_pipeline(args) -> None:
    """Điều phối toàn bộ Pipeline thực thi bài toán Sequence-to-Sequence Weather Forecasting.
    - Hàm này thực hiện: Khởi tạo DataModule theo 4 pha chống rò rỉ, chạy EDA trên tập Train, xây dựng mô hình, huấn luyện và đánh giá trên tập Test.
    - Input:
        + args: Đối tượng Namespace chứa các tham số dòng lệnh người dùng truyền vào.
    - Output:
        + Không trả về giá trị (thực thi và in kết quả ra terminal cùng các thư mục lưu trữ).
    """
    config = load_config(args.config)
    device = get_device(args.device if args.device != "auto" else config.get("project", {}).get("device", "auto"))
    print(f"[SYSTEM] Sử dụng thiết bị tính toán: {device}")

    # 1. Khởi tạo Data Module và thực hiện quy trình 4 pha
    data_cfg = config.get("data", {})
    dm = JenaDataModule(
        raw_csv_path=data_cfg.get("raw_csv_path", "data/jena_climate_2009_2016.csv"),
        seq_len=data_cfg.get("seq_len", 168),
        pred_len=data_cfg.get("pred_len", 72),
        target_col=data_cfg.get("target_col", "T (degC)"),
        train_ratio=data_cfg.get("train_ratio", 0.70),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        resample_rule=data_cfg.get("resample_rule", "1h"),
    )

    # 2. Xử lý theo từng chế độ chạy
    mode = args.mode.lower()
    if getattr(args, "checkpoint", None) and mode != "evaluate":
        raise ValueError("--checkpoint chỉ dùng khi chạy --mode evaluate.")

    if mode == "test":
        from tests.run_all_tests import run_all_smoke_tests
        exit_code = run_all_smoke_tests()
        sys.exit(exit_code)

    if mode in ["eda", "all"]:
        print("\n==================== [GIAI ĐOẠN: EDA TRÊN TẬP TRAIN] ====================")
        if not os.path.exists(dm.raw_csv_path):
            print(f"[CẢNH BÁO] Không tìm thấy file dữ liệu tại '{dm.raw_csv_path}'. Vui lòng tải dữ liệu Jena Climate đặt vào thư mục data/.")
            if mode == "eda":
                return
        else:
            dm.setup()
            # Kiểm chứng không rò rỉ
            dm.verify_zero_leakage()
            print("[ANTI-LEAKAGE] Đã kiểm chứng Zero Leakage: Chỉ thực hiện EDA trên tập Train!")
            eda_files = run_eda_pipeline(
                dm.train_df,
                output_dir=os.path.join(config.get("paths", {}).get("evaluation_dir", "evalution_results"), "eda"),
                dm=dm,
            )
            for f in eda_files:
                print(f"  + Đã lưu biểu đồ EDA: {f}")

    if mode in ["train", "evaluate", "all"]:
        if not os.path.exists(dm.raw_csv_path):
            print(f"[CẢNH BÁO] Không tìm thấy dữ liệu gốc tại '{dm.raw_csv_path}'. Không thể tiếp tục huấn luyện/đánh giá.")
            return

        dm.setup()
        dm.verify_zero_leakage()

        train_loader, val_loader, test_loader = dm.get_dataloaders(
            batch_size=data_cfg.get("batch_size", 64),
            num_workers=data_cfg.get("num_workers", 0),
        )

        model_name = args.model.lower()
        print(f"\n==================== [KHỞI TẠO MÔ HÌNH: {model_name}] ====================")
        model = build_model(model_name, config).to(device)

        train_cfg = config.get("training", {})
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        criterion = nn.MSELoss()

        # --- Khởi tạo Learning Rate Scheduler ---
        scheduler_type = train_cfg.get("scheduler_type", "ReduceLROnPlateau").lower()
        scheduler = None
        if scheduler_type == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=train_cfg.get("lr_factor", 0.5),
                patience=train_cfg.get("lr_patience", 3),
                min_lr=train_cfg.get("lr_min", 1e-5),
            )
            print(f"[SCHEDULER] ReduceLROnPlateau | factor={train_cfg.get('lr_factor', 0.5)} | patience={train_cfg.get('lr_patience', 3)} | min_lr={train_cfg.get('lr_min', 1e-5)}")
        elif scheduler_type == "cosineannealinglr":
            t_max = train_cfg.get("cosine_T_max", train_cfg.get("epochs", 50))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=train_cfg.get("lr_min", 1e-5),
            )
            print(f"[SCHEDULER] CosineAnnealingLR | T_max={t_max} | eta_min={train_cfg.get('lr_min', 1e-5)}")
        else:
            print("[SCHEDULER] Không dùng LR Scheduler (scheduler_type='none').")

        # --- Lấy target_scale từ DataModule để tính ngưỡng accuracy đúng ---
        # target_scale = độ lệch chuẩn σ của T(degC) trên tập Train (vd: ~8°C)
        # Nếu không truyền vào Trainer, tolerance_scaled = 1.5 / 1.0 = 1.5 (sai lớn)
        # Với σ≈8: tolerance_scaled = 1.5 / 8 = 0.1875 (đúng trong không gian chuẩn hoá)
        target_scale = float(dm.scaler.scale_[dm.target_idx]) if hasattr(dm, "scaler") and dm.scaler is not None else 1.0
        eval_cfg = config.get("evaluation", {})
        tolerance_degC = eval_cfg.get("tolerance_degC", 1.5)
        print(f"[TRAINER] target_scale (σ của T°C trên Train) = {target_scale:.4f} | tolerance_scaled = {tolerance_degC / target_scale:.4f}")

        paths_cfg = config.get("paths", {})
        checkpoints_dir = paths_cfg.get("checkpoints_dir", "checkpoints")

        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoints_dir=checkpoints_dir,
            model_name=model_name,
            clip_grad_norm=train_cfg.get("clip_grad_norm", 1.0),
            early_stopping_patience=train_cfg.get("early_stopping_patience", 7),
            teacher_forcing_start=train_cfg.get("teacher_forcing_ratio_start", 0.5),
            teacher_forcing_decay=train_cfg.get("teacher_forcing_decay", 0.95),
            scheduler=scheduler,
            target_scale=target_scale,
            tolerance_degC=tolerance_degC,
        )

        runs_dir = paths_cfg.get("runs_dir", "runs")
        experiment_name = resolve_experiment_name(
            runs_dir=runs_dir,
            model_name=model_name,
            mode=mode,
            run_name=getattr(args, "run_name", None),
        )
        trainer.experiment_name = experiment_name
        print(f"[TENSORBOARD] Ghi log thực nghiệm tại: {os.path.join(runs_dir, experiment_name)}")

        logger = TensorBoardLogger(
            log_dir=runs_dir,
            experiment_name=experiment_name,
        )

        epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 50)
        if mode in ["train", "all"]:
            config_summary = (
                f"**Model:** `{model_name}` | **LR:** `{train_cfg.get('learning_rate', 0.001)}` | "
                f"**Scheduler:** `{train_cfg.get('scheduler_type', 'ReduceLROnPlateau')}` | "
                f"**Epochs:** `{epochs}` | **TF start/decay:** `{train_cfg.get('teacher_forcing_ratio_start', 0.5)}/{train_cfg.get('teacher_forcing_decay', 0.95)}` | "
                f"**target_scale:** `{target_scale:.4f}`"
            )
            logger.log_text("Experiment/Config", config_summary, step=0)

        # Chạy Huấn luyện
        if mode in ["train", "all"]:
            log_figure_every = train_cfg.get("log_figure_every", 5)
            trainer.fit(train_loader, val_loader, epochs=epochs, logger=logger, log_figure_every=log_figure_every)

        # Chạy Đánh giá trên tập Test
        if mode in ["evaluate", "all"]:
            print(f"\n==================== [ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST] ====================")
            best_ckpt = resolve_checkpoint_path(
                model_name=model_name,
                checkpoints_dir=checkpoints_dir,
                explicit_path=getattr(args, "checkpoint", None),
                trained_checkpoint=(
                    trainer.latest_best_checkpoint if mode == "all" else None
                ),
            )
            print(f"[CHECKPOINT] Nạp trọng số từ: {best_ckpt}")
            checkpoint_data = trainer.load_checkpoint(best_ckpt)

            evaluator = ModelEvaluator(
                model=model,
                device=device,
                tolerance_degC=config.get("evaluation", {}).get("tolerance_degC", 1.5),
                results_dir=paths_cfg.get("evaluation_dir", "evalution_results"),
            )

            results = evaluator.evaluate(
                test_loader=test_loader,
                inverse_transform_fn=dm.inverse_transform_target,
                model_name=model_name,
                logger=logger,
                checkpoint_path=best_ckpt,
                checkpoint_run=checkpoint_data.get("experiment_name"),
            )

            logger.log_hparams(
                hparam_dict={
                    "model": model_name,
                    "lr": float(train_cfg.get("learning_rate", 0.001)),
                    "scheduler": str(train_cfg.get("scheduler_type", "ReduceLROnPlateau")),
                    "batch_size": int(data_cfg.get("batch_size", 64)),
                    "epochs": int(epochs),
                },
                metric_dict={
                    "hparam/MAE": float(results.get("MAE", 0.0)),
                    "hparam/RMSE": float(results.get("RMSE", 0.0)),
                    "hparam/R2": float(results.get("R2", 0.0)),
                    "hparam/Acc_1_5C": float(results.get("Within_Tolerance_Rate_Pct", 0.0)),
                },
            )

            print("\n------------------ KẾT QUẢ ĐÁNH GIÁ (TEST SET) ------------------")
            for k, v in results.items():
                print(f"  * {k}: {v}")
            print("-----------------------------------------------------------------")

        logger.close()


def main() -> None:
    """Điểm khởi chạy chương trình từ giao diện dòng lệnh (Command Line Interface).
    - Hàm này thực hiện: Tiếp nhận các tham số truyền vào từ terminal qua ArgumentParser và chuyển tiếp tới hàm run_pipeline.
    - Input:
        + Không nhận trực tiếp tham số (đọc từ sys.argv).
    - Output:
        + Không trả về giá trị (chạy tiến trình chính).
    """
    parser = argparse.ArgumentParser(
        description="Master Pipeline: Sequence-to-Sequence Weather Forecasting (Topic 5)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Đường dẫn file cấu hình YAML (mặc định: configs/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "eda", "train", "evaluate", "test"],
        help="Chế độ thực thi: 'all', 'eda', 'train', 'evaluate', 'test' (mặc định: 'all')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="seq2seq_attention",
        choices=["seq2seq_lstm", "seq2seq_attention", "transformer"],
        help="Tên mô hình: 'seq2seq_lstm', 'seq2seq_attention', 'transformer'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Số lượng Epoch muốn chạy (ghi đè giá trị trong config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị tính toán: 'auto', 'cuda', hoặc 'cpu'",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nhãn định danh cho lần chạy TensorBoard (ví dụ: 'v3_cosine_lr'). Mặc định tự gắn timestamp.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Đường dẫn checkpoint chính xác cần đánh giá (bắt buộc nếu có nhiều checkpoint cùng model).",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
