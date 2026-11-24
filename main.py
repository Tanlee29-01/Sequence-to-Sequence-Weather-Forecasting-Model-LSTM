import argparse
import os
import sys
from typing import Dict
import torch
import torch.nn as nn
import yaml

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
        )

        logger = TensorBoardLogger(
            log_dir=paths_cfg.get("runs_dir", "runs"),
            experiment_name=f"{model_name}_run",
        )

        # Chạy Huấn luyện
        if mode in ["train", "all"]:
            epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 50)
            trainer.fit(train_loader, val_loader, epochs=epochs, logger=logger)

        # Chạy Đánh giá trên tập Test
        if mode in ["evaluate", "all"]:
            print(f"\n==================== [ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST] ====================")
            # Tìm file checkpoint tốt nhất có timestamp hoặc alias mặc định
            best_ckpt = ""
            if os.path.exists(checkpoints_dir):
                candidates = sorted([
                    f for f in os.listdir(checkpoints_dir)
                    if f.startswith(model_name) and f.endswith("_best.pth")
                ])
                if candidates:
                    best_ckpt = os.path.join(checkpoints_dir, candidates[-1])

            if best_ckpt and os.path.exists(best_ckpt):
                print(f"[CHECKPOINT] Nạp trọng số tốt nhất từ: {best_ckpt}")
                trainer.load_checkpoint(best_ckpt)

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

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
