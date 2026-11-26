import argparse
import copy
from datetime import datetime
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
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


VALID_MODELS: Tuple[str, ...] = ("seq2seq_lstm", "seq2seq_attention", "transformer")


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


def parse_model_arg(model_arg: str) -> Tuple[str, Optional[str]]:
    """Tách tên mô hình gốc và hậu tố phiên bản (v1, v2, ..., v_n) từ tham số CLI --model.
    - Hàm này thực hiện: Kiểm tra định dạng '<base_model>' hoặc '<base_model>_v<n>' bằng biểu thức chính quy.
    - Input:
        + model_arg: Chuỗi tên mô hình từ CLI (ví dụ: 'seq2seq_attention', 'seq2seq_attention_v1', 'transformer_v3').
    - Output:
        + base_model: Tên kiến trúc gốc trong VALID_MODELS.
        + version: Nhãn phiên bản dạng 'v1', 'v2', ... hoặc None nếu không chỉ định.
    """
    cleaned = model_arg.strip().lower()
    pattern = r"^(seq2seq_lstm|seq2seq_attention|transformer)(?:_(v\d+))?$"
    match = re.match(pattern, cleaned)
    if not match:
        raise ValueError(
            f"Tên mô hình '{model_arg}' không hợp lệ. Hỗ trợ: {', '.join(VALID_MODELS)} "
            f"(có thể kèm hậu tố _v1, _v2, ..., _vN)."
        )
    return match.group(1), match.group(2)


def list_model_versions(base_dir: str, base_model: str) -> List[int]:
    """Liệt kê tất cả số hiệu phiên bản v1, v2, ..., vN đã tồn tại của một mô hình trong thư mục.
    - Hàm này thực hiện: Quét các thư mục con có tiền tố '<base_model>_v<số>' và trích xuất danh sách số nguyên tăng dần.
    - Input:
        + base_dir: Thư mục gốc cần quét ('checkpoints' hoặc 'runs').
        + base_model: Tên kiến trúc mô hình gốc.
    - Output:
        + Danh sách các số phiên bản nguyên đã sắp xếp tăng dần.
    """
    if not os.path.exists(base_dir):
        return []
    pattern = re.compile(rf"^{re.escape(base_model)}_v(\d+)(?:_.*)?$")
    versions: List[int] = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            m = pattern.match(entry)
            if m:
                versions.append(int(m.group(1)))
    return sorted(set(versions))


def get_next_version(checkpoints_root: str, runs_dir: str, base_model: str) -> str:
    """Tự động xác định số hiệu phiên bản kế tiếp (v1 -> v2 -> v3 -> ... -> vN) khi huấn luyện mới.
    - Hàm này thực hiện: Tổng hợp các version hiện có trong checkpoints/ và runs/ để trả về v(max + 1).
    - Input:
        + checkpoints_root: Thư mục gốc checkpoints/.
        + runs_dir: Thư mục gốc runs/.
        + base_model: Tên kiến trúc mô hình gốc.
    - Output:
        + Chuỗi phiên bản tiếp theo dạng 'v3', 'v4', ...
    """
    ckpt_versions = list_model_versions(checkpoints_root, base_model)
    run_versions = list_model_versions(runs_dir, base_model)
    all_versions = ckpt_versions + run_versions
    next_num = max(all_versions, default=0) + 1
    return f"v{next_num}"


def resolve_version_for_mode(
    checkpoints_root: str,
    runs_dir: str,
    base_model: str,
    explicit_version: Optional[str],
    mode: str,
    run_name: Optional[str] = None,
) -> str:
    """Xác định phiên bản mục tiêu cho chế độ chạy hiện tại (train, evaluate hoặc all).
    - Hàm này thực hiện: Ưu tiên version chỉ định tường minh; nếu train sẽ tự tăng vN tiếp theo; nếu evaluate sẽ chọn vN mới nhất hiện có.
    - Input:
        + checkpoints_root: Thư mục gốc checkpoints/.
        + runs_dir: Thư mục gốc runs/.
        + base_model: Tên mô hình gốc.
        + explicit_version: Version trích xuất từ '--model <model>_vN' (nếu có).
        + mode: Chế độ thực thi ('train', 'evaluate', 'all', ...).
        + run_name: Tham số '--run_name' nếu người dùng truyền kèm (tùy chọn).
    - Output:
        + version: Chuỗi phiên bản chuẩn dạng 'v1', 'v2', 'v3', ...
    """
    if explicit_version:
        return explicit_version

    if run_name:
        m = re.match(r"^(v\d+)(?:_.*)?$", run_name.strip().lower())
        if m:
            return m.group(1)

    if mode in ["train", "all"]:
        return get_next_version(checkpoints_root, runs_dir, base_model)

    existing_ckpts = list_model_versions(checkpoints_root, base_model)
    if existing_ckpts:
        return f"v{max(existing_ckpts)}"

    existing_runs = list_model_versions(runs_dir, base_model)
    if existing_runs:
        return f"v{max(existing_runs)}"

    return "v1"


def resolve_best_checkpoint(
    checkpoints_root: str,
    version_ckpt_dir: str,
    base_model: str,
    explicit_ckpt: Optional[str] = None,
) -> str:
    """Resolve the best checkpoint for a version without falling back to last/random weights."""
    if explicit_ckpt:
        return resolve_checkpoint_path(
            base_model, checkpoints_root, explicit_path=explicit_ckpt
        )

    search_dir = (
        version_ckpt_dir
        if os.path.isdir(version_ckpt_dir)
        else checkpoints_root
    )
    try:
        return resolve_checkpoint_path(base_model, search_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tốt nhất cho '{base_model}' trong "
            f"'{search_dir}'. Hãy truyền --checkpoint với đường dẫn chính xác."
        ) from exc


def save_version_config(
    version_ckpt_dir: str,
    base_model: str,
    version: str,
    config: Dict,
    epochs: int,
) -> str:
    """Lưu file config.json ghi lại siêu tham số mô hình và cấu hình huấn luyện của từng version.
    - Hàm này thực hiện: Đóng gói thông số kiến trúc, training và data vào checkpoints/<model>_vN/config.json để phục vụ đánh giá lại chính xác.
    - Input:
        + version_ckpt_dir: Thư mục checkpoint của phiên bản.
        + base_model: Tên kiến trúc mô hình gốc.
        + version: Nhãn phiên bản ('v1', 'v2', ...).
        + config: Từ điển cấu hình gốc.
        + epochs: Số epochs huấn luyện thực tế.
    - Output:
        + config_json_path: Đường dẫn file config.json vừa ghi.
    """
    os.makedirs(version_ckpt_dir, exist_ok=True)
    train_cfg = dict(config.get("training", {}))
    train_cfg["epochs"] = epochs
    data_cfg = config.get("data", {})

    payload = {
        "model": base_model,
        "version": version,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": config.get("models", {}).get(base_model, {}),
        "training": train_cfg,
        "data": {
            "seq_len": data_cfg.get("seq_len", 168),
            "pred_len": data_cfg.get("pred_len", 72),
            "num_features": data_cfg.get("num_features", 14),
        },
    }
    config_json_path = os.path.join(version_ckpt_dir, "config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return config_json_path


def apply_version_model_config(
    config: Dict,
    version_ckpt_dir: str,
    base_model: str,
    ckpt_path: str = "",
) -> Dict:
    """Nạp lại siêu tham số kiến trúc từ config.json hoặc file .pth của version cũ để tránh lệch shape mạng.
    - Hàm này thực hiện: Đọc model_config đã lưu của version cần đánh giá và ghi đè vào bản sao từ điển cấu hình trước khi build_model().
    - Input:
        + config: Từ điển cấu hình hiện hành từ config.yaml.
        + version_ckpt_dir: Thư mục checkpoint của phiên bản.
        + base_model: Tên kiến trúc mô hình gốc.
        + ckpt_path: Đường dẫn file .pth dự phòng nếu chưa có config.json.
    - Output:
        + effective_config: Bản sao từ điển cấu hình đã đồng bộ đúng kiến trúc của version.
    """
    effective_config = copy.deepcopy(config)
    effective_config.setdefault("models", {})

    config_json_path = os.path.join(version_ckpt_dir, "config.json")
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved.get("model_config"), dict) and saved["model_config"]:
                effective_config["models"][base_model] = saved["model_config"]
                return effective_config
        except Exception:
            pass

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt_meta = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt_meta, dict) and isinstance(ckpt_meta.get("model_config"), dict):
                effective_config["models"][base_model] = ckpt_meta["model_config"]
        except Exception:
            pass

    return effective_config


def resolve_experiment_name(
    runs_dir: str,
    base_model: str,
    version: str,
    mode: str,
    run_name: Optional[str] = None,
) -> str:
    """Tạo tên thư mục run tách biệt theo phiên bản cho mỗi lần chạy trên TensorBoard.
    - Hàm này thực hiện: Gắn '<base_model>_<version>_<timestamp>' khi train mới; khi evaluate sẽ gắn vào thư mục '<base_model>_<version>*' tương ứng.
    - Input:
        + runs_dir: Thư mục gốc chứa log TensorBoard (mặc định 'runs').
        + base_model: Tên kiến trúc mô hình gốc.
        + version: Nhãn phiên bản ('v1', 'v2', 'v3', ...).
        + mode: Chế độ chạy ('train', 'evaluate', 'all', ...).
        + run_name: Tên phiên thực nghiệm do người dùng truyền qua CLI (tùy chọn).
    - Output:
        + experiment_name: Tên thư mục con duy nhất trong runs/.
    """
    prefix = f"{base_model}_{version}"
    if run_name:
        cleaned_run = run_name.strip()
        if cleaned_run.lower().startswith(f"{version}_") or cleaned_run.lower() == version:
            return f"{base_model}_{cleaned_run}"
        return f"{prefix}_{cleaned_run}"

    if mode == "evaluate" and os.path.exists(runs_dir):
        candidates = [
            d for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith(prefix)
        ]
        if candidates:
            candidates.sort(key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
            return candidates[-1]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}"


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
        alias = os.path.join(checkpoints_dir, f"{model_name}_best.pth")
        if os.path.isfile(alias):
            return alias

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
    - Hàm này thực hiện: Khởi tạo DataModule theo 4 pha chống rò rỉ, chạy EDA trên tập Train, quản lý version tự động (v1..vN), huấn luyện và đánh giá trên tập Test.
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

        base_model, explicit_version = parse_model_arg(args.model)
        paths_cfg = config.get("paths", {})
        checkpoints_root = paths_cfg.get("checkpoints_dir", "checkpoints")
        runs_dir = paths_cfg.get("runs_dir", "runs")

        version = resolve_version_for_mode(
            checkpoints_root=checkpoints_root,
            runs_dir=runs_dir,
            base_model=base_model,
            explicit_version=explicit_version,
            mode=mode,
            run_name=getattr(args, "run_name", None),
        )
        versioned_name = f"{base_model}_{version}"
        version_ckpt_dir = os.path.join(checkpoints_root, versioned_name)

        explicit_ckpt = getattr(args, "checkpoint", None)
        best_ckpt = ""
        effective_config = config
        if mode == "evaluate":
            best_ckpt = resolve_best_checkpoint(
                checkpoints_root=checkpoints_root,
                version_ckpt_dir=version_ckpt_dir,
                base_model=base_model,
                explicit_ckpt=explicit_ckpt,
            )
            effective_config = apply_version_model_config(
                config=config,
                version_ckpt_dir=version_ckpt_dir,
                base_model=base_model,
                ckpt_path=best_ckpt,
            )

        print(f"\n==================== [KHỞI TẠO MÔ HÌNH: {versioned_name} (Kiến trúc: {base_model})] ====================")
        print(f"[VERSION] Thư mục Checkpoint: {version_ckpt_dir}")
        model = build_model(base_model, effective_config).to(device)

        train_cfg = effective_config.get("training", {})
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

        target_scale = float(dm.scaler.scale_[dm.target_idx]) if hasattr(dm, "scaler") and dm.scaler is not None else 1.0
        eval_cfg = effective_config.get("evaluation", {})
        tolerance_degC = eval_cfg.get("tolerance_degC", 1.5)
        print(f"[TRAINER] target_scale (σ của T°C trên Train) = {target_scale:.4f} | tolerance_scaled = {tolerance_degC / target_scale:.4f}")

        model_arch_cfg = effective_config.get("models", {}).get(base_model, {})
        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoints_dir=version_ckpt_dir,
            model_name=base_model,
            clip_grad_norm=train_cfg.get("clip_grad_norm", 1.0),
            early_stopping_patience=train_cfg.get("early_stopping_patience", 7),
            teacher_forcing_start=train_cfg.get("teacher_forcing_ratio_start", 0.5),
            teacher_forcing_decay=train_cfg.get("teacher_forcing_decay", 0.95),
            scheduler=scheduler,
            target_scale=target_scale,
            tolerance_degC=tolerance_degC,
            model_config=model_arch_cfg,
        )

        experiment_name = resolve_experiment_name(
            runs_dir=runs_dir,
            base_model=base_model,
            version=version,
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
            saved_cfg_path = save_version_config(
                version_ckpt_dir=version_ckpt_dir,
                base_model=base_model,
                version=version,
                config=effective_config,
                epochs=epochs,
            )
            print(f"[CONFIG] Đã lưu cấu hình phiên bản tại: {saved_cfg_path}")

            config_summary = (
                f"**Model:** `{versioned_name}` | **LR:** `{train_cfg.get('learning_rate', 0.001)}` | "
                f"**Scheduler:** `{train_cfg.get('scheduler_type', 'ReduceLROnPlateau')}` | "
                f"**Epochs:** `{epochs}` | **TF start/decay:** `{train_cfg.get('teacher_forcing_ratio_start', 0.5)}/{train_cfg.get('teacher_forcing_decay', 0.95)}` | "
                f"**target_scale:** `{target_scale:.4f}`"
            )
            logger.log_text("Experiment/Config", config_summary, step=0)

            log_figure_every = train_cfg.get("log_figure_every", 5)
            trainer.fit(train_loader, val_loader, epochs=epochs, logger=logger, log_figure_every=log_figure_every)

        # Chạy Đánh giá trên tập Test
        if mode in ["evaluate", "all"]:
            print(f"\n==================== [ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST: {versioned_name}] ====================")
            if mode == "all":
                best_ckpt = trainer.latest_best_checkpoint
                if not best_ckpt:
                    raise RuntimeError(
                        f"Huấn luyện '{versioned_name}' kết thúc nhưng không tạo checkpoint tốt nhất."
                    )
            elif not best_ckpt:
                best_ckpt = resolve_best_checkpoint(
                    checkpoints_root=checkpoints_root,
                    version_ckpt_dir=version_ckpt_dir,
                    base_model=base_model,
                    explicit_ckpt=explicit_ckpt,
                )

            print(f"[CHECKPOINT] Nạp trọng số tốt nhất từ: {best_ckpt}")
            checkpoint_data = trainer.load_checkpoint(best_ckpt)

            evaluator = ModelEvaluator(
                model=model,
                device=device,
                tolerance_degC=effective_config.get("evaluation", {}).get("tolerance_degC", 1.5),
                results_dir=paths_cfg.get("evaluation_dir", "evalution_results"),
            )

            results = evaluator.evaluate(
                test_loader=test_loader,
                inverse_transform_fn=dm.inverse_transform_target,
                model_name=versioned_name,
                logger=logger,
                checkpoint_path=best_ckpt,
                checkpoint_run=checkpoint_data.get("experiment_name"),
            )

            logger.log_hparams(
                hparam_dict={
                    "model": versioned_name,
                    "base_model": base_model,
                    "version": version,
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
        help="Tên mô hình ('seq2seq_lstm', 'seq2seq_attention', 'transformer'), có thể kèm hậu tố _v1, _v2, ..., _vN",
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
        help="Nhãn định danh bổ sung cho lần chạy TensorBoard (mặc định gắn theo version vN và timestamp).",
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
