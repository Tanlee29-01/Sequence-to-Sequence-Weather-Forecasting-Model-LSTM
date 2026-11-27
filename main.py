import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data_module import JenaDataModule
from src.eda import run_eda_pipeline
from src.evaluation import ModelEvaluator
from src.logger import TensorBoardLogger
from src.model_builder import build_model
from src.trainer import ModelTrainer


VALID_MODELS: Tuple[str, ...] = ("seq2seq_lstm", "seq2seq_attention", "transformer")


def seed_everything(seed: int = 42) -> None:
    """Cố định toàn bộ bộ sinh số ngẫu nhiên (Python, NumPy, PyTorch CPU/CUDA) nhằm tăng khả năng tái lập.
    - Hàm này thực hiện: Thiết lập cùng một seed cho random, numpy.random, torch.manual_seed và torch.cuda.manual_seed_all.
    - Input:
        + seed: Số nguyên hạt giống ngẫu nhiên (mặc định 42).
    - Output:
        + Không trả về giá trị (cập nhật trạng thái RNG toàn cục).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_file_sha256(file_path: str) -> str:
    """Tính mã băm SHA-256 của một tệp trên đĩa để truy xuất nguồn gốc checkpoint/dữ liệu.
    - Hàm này thực hiện: Đọc tệp theo từng khối 1MB và trả về chuỗi hex SHA-256 đầy đủ.
    - Input:
        + file_path: Đường dẫn tới tệp cần băm.
    - Output:
        + Chuỗi hex SHA-256 (64 ký tự) hoặc chuỗi rỗng nếu tệp không tồn tại.
    """
    if not file_path or not os.path.exists(file_path):
        return ""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Đã yêu cầu --device cuda nhưng hệ thống không khả dụng CUDA.")
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

def discover_all_model_versions(checkpoints_root: str) -> List[Tuple[str, str]]:
    """Quét và trả về toàn bộ các cặp (base_model, version) hợp lệ có thư mục trong checkpoints/.
    - Hàm này thực hiện: Duyệt theo thứ tự chuẩn của VALID_MODELS và số hiệu phiên bản tăng dần (v1, v2, ...).
    - Input:
        + checkpoints_root: Đường dẫn thư mục gốc checkpoints/.
    - Output:
        + Danh sách các tuple (base_model, 'vN') theo thứ tự chuẩn hóa.
    """
    discovered: List[Tuple[str, str]] = []
    for base_model in VALID_MODELS:
        for ver_num in list_model_versions(checkpoints_root, base_model):
            discovered.append((base_model, f"v{ver_num}"))
    return discovered


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
        + base_model: Tên kiến trúc mô hình gốc.
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
    """Tìm đường dẫn file checkpoint tốt nhất (theo min best_val_loss) trong đúng thư mục phiên bản đang đánh giá.
    - Hàm này thực hiện: Ưu tiên --checkpoint chỉ định (bắt buộc tồn tại); nếu không chỉ định thì quét duy nhất các tệp *_best.pth trong version_ckpt_dir và chọn tệp có best_val_loss thấp nhất (tuyệt đối không fallback xuyên thư mục gốc hay rơi về _last/_final).
    - Input:
        + checkpoints_root: Thư mục gốc checkpoints/ (giữ chữ ký tương thích, không dùng fallback xuyên version).
        + version_ckpt_dir: Thư mục riêng của phiên bản (ví dụ: checkpoints/seq2seq_attention_v1).
        + base_model: Tên kiến trúc mô hình gốc.
        + explicit_ckpt: Đường dẫn file .pth người dùng truyền trực tiếp (tùy chọn).
    - Output:
        + Đường dẫn tới file checkpoint tốt nhất (hoặc chuỗi rỗng nếu chưa có tệp *_best.pth trong version_ckpt_dir).
    """
    _ = checkpoints_root  # Giữ chữ ký tương thích ngược; cấm fallback xuyên version

    if explicit_ckpt:
        if not os.path.exists(explicit_ckpt):
            raise FileNotFoundError(
                f"Không tìm thấy tệp checkpoint được chỉ định tại '{explicit_ckpt}'."
            )
        return explicit_ckpt

    if not os.path.exists(version_ckpt_dir):
        return ""

    alias_name = f"{base_model}_best.pth"
    best_candidates = sorted(
        f for f in os.listdir(version_ckpt_dir)
        if os.path.isfile(os.path.join(version_ckpt_dir, f))
        and f.startswith(base_model)
        and f.endswith("_best.pth")
    )
    if not best_candidates:
        return ""

    scored_candidates: List[Tuple[float, int, int, str, str]] = []
    for fname in best_candidates:
        full_path = os.path.join(version_ckpt_dir, fname)
        val_loss = float("inf")
        epoch = -1
        try:
            meta = torch.load(full_path, map_location="cpu")
            if isinstance(meta, dict):
                raw_loss = meta.get("best_val_loss")
                if raw_loss is not None:
                    val_loss = float(raw_loss)
                raw_epoch = meta.get("epoch")
                if raw_epoch is not None:
                    epoch = int(raw_epoch)
        except Exception as exc:
            raise ValueError(
                f"Không thể đọc metadata từ checkpoint '{full_path}': {exc}"
            ) from exc

        # Ưu tiên: (1) best_val_loss nhỏ nhất, (2) file alias <model>_best.pth nếu bằng loss, (3) epoch lớn nhất
        is_not_alias = 0 if fname == alias_name else 1
        scored_candidates.append((val_loss, is_not_alias, -epoch, fname, full_path))

    scored_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return scored_candidates[0][4]


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
        "seed": int(config.get("project", {}).get("seed", 42)),
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
    version: Optional[str] = None,
) -> Dict:
    """Nạp và xác thực siêu tham số kiến trúc, training và data từ config.json hoặc .pth của version trước khi build_model().
    - Hàm này thực hiện: Đọc nghiêm ngặt checkpoints/<model>_vN/config.json (không nuốt lỗi), kiểm chứng khớp tên model/version và các tham số dữ liệu (seq_len, pred_len, num_features), đồng bộ model_config và training vào bản sao cấu hình.
    - Input:
        + config: Từ điển cấu hình hiện hành từ config.yaml.
        + version_ckpt_dir: Thư mục checkpoint của phiên bản.
        + base_model: Tên kiến trúc mô hình gốc.
        + ckpt_path: Đường dẫn file .pth dự phòng nếu chưa có config.json.
        + version: Nhãn phiên bản kỳ vọng ('v1', 'v2', ...) để kiểm chứng chéo (tùy chọn).
    - Output:
        + effective_config: Bản sao từ điển cấu hình đã đồng bộ và xác thực khớp kiến trúc của version.
    """
    effective_config = copy.deepcopy(config)
    effective_config.setdefault("models", {})
    effective_config.setdefault("training", {})
    effective_config.setdefault("data", {})

    config_json_path = os.path.join(version_ckpt_dir, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if not isinstance(saved, dict):
            raise ValueError(f"Tệp cấu hình phiên bản '{config_json_path}' phải là JSON object.")

        saved_model = saved.get("model")
        if saved_model is not None and str(saved_model).strip().lower() != base_model.lower():
            raise ValueError(
                f"Lệch tên mô hình trong '{config_json_path}': "
                f"config.json='{saved_model}' != requested='{base_model}'."
            )

        saved_version = saved.get("version")
        if version and saved_version is not None and str(saved_version).strip().lower() != version.lower():
            raise ValueError(
                f"Lệch phiên bản trong '{config_json_path}': "
                f"config.json='{saved_version}' != requested='{version}'."
            )

        saved_data = saved.get("data")
        if isinstance(saved_data, dict):
            active_data = effective_config["data"]
            for key in ("seq_len", "pred_len", "num_features"):
                if key in saved_data and key in active_data and int(saved_data[key]) != int(active_data[key]):
                    raise ValueError(
                        f"Lệch cấu hình dữ liệu '{key}' giữa '{config_json_path}' ({saved_data[key]}) "
                        f"và config.yaml ({active_data[key]})."
                    )
                if key in saved_data:
                    active_data[key] = saved_data[key]

        if isinstance(saved.get("model_config"), dict) and saved["model_config"]:
            effective_config["models"][base_model] = copy.deepcopy(saved["model_config"])

        if isinstance(saved.get("training"), dict) and saved["training"]:
            effective_config["training"].update(saved["training"])
    elif ckpt_path and os.path.exists(ckpt_path):
        ckpt_meta = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt_meta, dict) and isinstance(ckpt_meta.get("model_config"), dict):
            effective_config["models"][base_model] = copy.deepcopy(ckpt_meta["model_config"])

    # Kiểm tra nhất quán input_dim của kiến trúc với num_features của dữ liệu
    arch_cfg = effective_config["models"].get(base_model, {})
    num_features = effective_config["data"].get("num_features")
    if (
        isinstance(arch_cfg, dict)
        and "input_dim" in arch_cfg
        and num_features is not None
        and int(arch_cfg["input_dim"]) != int(num_features)
    ):
        raise ValueError(
            f"Lệch số đặc trưng đầu vào cho '{base_model}': "
            f"model.input_dim={arch_cfg['input_dim']} != data.num_features={num_features}."
        )

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


def evaluate_model_version(
    base_model: str,
    version: str,
    config: Dict,
    config_path: str,
    dm: JenaDataModule,
    test_loader,
    device: torch.device,
    checkpoints_root: str,
    runs_dir: str,
    results_dir: str,
    warmup_runs: int,
    benchmark_runs: int,
    explicit_ckpt: Optional[str] = None,
    run_name: Optional[str] = None,
    log_to_tensorboard: bool = True,
) -> Dict:
    """Thực thi quy trình đánh giá nghiêm ngặt cho đúng một cặp (base_model, version).
    - Hàm này thực hiện: Tìm checkpoint best của version (chặn ngay nếu thiếu để tránh random weights), nạp & kiểm tra khớp config.json, xác thực kiến trúc trước khi load_state_dict(..., strict=True), chấm điểm tổng thể/horizon và benchmark hiệu năng.
    - Input:
        + base_model: Tên kiến trúc gốc ('seq2seq_lstm', 'seq2seq_attention', 'transformer').
        + version: Nhãn phiên bản ('v1', 'v2', ...).
        + config: Từ điển cấu hình gốc từ config.yaml.
        + config_path: Đường dẫn file config.yaml.
        + dm: Thực thể JenaDataModule đã gọi setup() và verify_zero_leakage().
        + test_loader: DataLoader của tập Test (không shuffle).
        + device: Thiết bị tính toán cố định (CPU hoặc CUDA GPU).
        + checkpoints_root: Thư mục gốc checkpoints/.
        + runs_dir: Thư mục gốc runs/ cho TensorBoard.
        + results_dir: Thư mục gốc lưu báo cáo đánh giá.
        + warmup_runs: Số lượt warmup trước khi đo latency.
        + benchmark_runs: Số lượt lặp benchmark latency.
        + explicit_ckpt: Đường dẫn file checkpoint chỉ định tay (tùy chọn).
        + run_name: Nhãn run TensorBoard tùy chọn.
        + log_to_tensorboard: Bật/tắt ghi kết quả vào TensorBoard.
    - Output:
        + results: Từ điển kết quả đánh giá đầy đủ của cặp (base_model, version).
    """
    versioned_name = f"{base_model}_{version}"
    version_ckpt_dir = os.path.join(checkpoints_root, versioned_name)

    best_ckpt = resolve_best_checkpoint(
        checkpoints_root=checkpoints_root,
        version_ckpt_dir=version_ckpt_dir,
        base_model=base_model,
        explicit_ckpt=explicit_ckpt,
    )
    if not best_ckpt or not os.path.exists(best_ckpt):
        raise FileNotFoundError(
            f"[BLOCKED] Không tìm thấy checkpoint hợp lệ (*_best.pth) cho '{versioned_name}' "
            f"trong '{version_ckpt_dir}'. Từ chối đánh giá bằng trọng số ngẫu nhiên!"
        )

    effective_config = apply_version_model_config(
        config=config,
        version_ckpt_dir=version_ckpt_dir,
        base_model=base_model,
        ckpt_path=best_ckpt,
        version=version,
    )
    version_cfg_file = os.path.join(version_ckpt_dir, "config.json")
    version_cfg_path = os.path.abspath(version_cfg_file) if os.path.exists(version_cfg_file) else ""

    print(f"\n==================== [ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST: {versioned_name}] ====================")
    print(f"[VERSION] Thư mục Checkpoint: {version_ckpt_dir}")
    print(f"[CHECKPOINT] Nạp trọng số tốt nhất từ: {best_ckpt}")

    model = build_model(base_model, effective_config).to(device)
    model_arch_cfg = effective_config.get("models", {}).get(base_model, {})
    train_cfg = effective_config.get("training", {})
    eval_cfg = effective_config.get("evaluation", {})
    data_cfg = effective_config.get("data", {})

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    target_scale = float(dm.scaler.scale_[dm.target_idx]) if hasattr(dm, "scaler") and dm.scaler is not None else 1.0
    tolerance_degC = float(eval_cfg.get("tolerance_degC", 1.5))

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device=device,
        checkpoints_dir=version_ckpt_dir,
        model_name=base_model,
        target_scale=target_scale,
        tolerance_degC=tolerance_degC,
        model_config=model_arch_cfg,
    )

    ckpt_meta = trainer.load_checkpoint(
        best_ckpt,
        expected_model_name=base_model,
        expected_model_config=model_arch_cfg,
        strict=True,
    )

    logger = None
    if log_to_tensorboard:
        experiment_name = resolve_experiment_name(
            runs_dir=runs_dir,
            base_model=base_model,
            version=version,
            mode="evaluate",
            run_name=run_name,
        )
        print(f"[TENSORBOARD] Ghi log thực nghiệm tại: {os.path.join(runs_dir, experiment_name)}")
        logger = TensorBoardLogger(log_dir=runs_dir, experiment_name=experiment_name)

    try:
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            tolerance_degC=tolerance_degC,
            results_dir=results_dir,
        )

        results = evaluator.evaluate(
            test_loader=test_loader,
            inverse_transform_fn=dm.inverse_transform_target,
            model_name=base_model,
            version=version,
            logger=logger,
            checkpoint_path=os.path.abspath(best_ckpt),
            checkpoint_sha256=compute_file_sha256(best_ckpt),
            config_path=os.path.abspath(config_path),
            version_config_path=version_cfg_path,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )

        if logger is not None:
            ckpt_epoch = int(ckpt_meta.get("epoch", train_cfg.get("epochs", 50)))
            logger.log_hparams(
                hparam_dict={
                    "model": versioned_name,
                    "base_model": base_model,
                    "version": version,
                    "lr": float(train_cfg.get("learning_rate", 0.001)),
                    "scheduler": str(train_cfg.get("scheduler_type", "ReduceLROnPlateau")),
                    "batch_size": int(results.get("Batch_Size", data_cfg.get("batch_size", 64))),
                    "epochs": ckpt_epoch,
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
        return results
    finally:
        if logger is not None:
            logger.close()


def run_pipeline(args) -> None:
    """Điều phối toàn bộ Pipeline thực thi bài toán Sequence-to-Sequence Weather Forecasting.
    - Hàm này thực hiện: Khóa seed tái lập, khởi tạo DataModule theo 4 pha chống rò rỉ, chạy EDA trên tập Train, quản lý version tự động (v1..vN), huấn luyện và đánh giá công bằng trên tập Test.
    - Input:
        + args: Đối tượng Namespace chứa các tham số dòng lệnh người dùng truyền vào.
    - Output:
        + Không trả về giá trị (thực thi và in kết quả ra terminal cùng các thư mục lưu trữ).
    """
    config = load_config(args.config)
    seed = int(config.get("project", {}).get("seed", 42))
    seed_everything(seed)
    device = get_device(args.device if args.device != "auto" else config.get("project", {}).get("device", "auto"))
    print(f"[SYSTEM] Sử dụng thiết bị tính toán: {device} | Seed: {seed}")

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
        invalid_wind_val=data_cfg.get("invalid_wind_val", -9999.0),
        features=data_cfg.get("all_features", None),
    )

    # 2. Xử lý theo từng chế độ chạy
    mode = args.mode.lower()
    paths_cfg = config.get("paths", {})
    eval_cfg = config.get("evaluation", {})
    checkpoints_root = paths_cfg.get("checkpoints_dir", "checkpoints")
    runs_dir = paths_cfg.get("runs_dir", "runs")
    results_dir = getattr(args, "results_dir", None) or paths_cfg.get("evaluation_dir", "evalution_results")
    batch_size = getattr(args, "batch_size", None) or int(data_cfg.get("batch_size", 64))
    warmup_runs = (
        args.warmup_runs
        if getattr(args, "warmup_runs", None) is not None
        else int(eval_cfg.get("warmup_runs", 10))
    )
    benchmark_runs = (
        args.benchmark_runs
        if getattr(args, "benchmark_runs", None) is not None
        else int(eval_cfg.get("benchmark_runs", 50))
    )

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
                output_dir=os.path.join(results_dir, "eda"),
                dm=dm,
            )
            for f in eda_files:
                print(f"  + Đã lưu biểu đồ EDA: {f}")

    if mode in ["train", "evaluate", "all"]:
        if not os.path.exists(dm.raw_csv_path):
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu gốc tại '{dm.raw_csv_path}'. Không thể tiếp tục huấn luyện/đánh giá."
            )

        dm.setup()
        dm.verify_zero_leakage()

        train_loader, val_loader, test_loader = dm.get_dataloaders(
            batch_size=batch_size,
            num_workers=data_cfg.get("num_workers", 0),
        )

        # Chế độ đánh giá riêng lẻ hoặc đồng loạt toàn bộ các phiên bản (Fair Comparison Protocol)
        if mode == "evaluate":
            if getattr(args, "all_versions", False):
                targets = discover_all_model_versions(checkpoints_root)
                if not targets:
                    raise FileNotFoundError(
                        f"Không tìm thấy thư mục phiên bản nào (<model>_vN) trong '{checkpoints_root}'."
                    )
                print(
                    f"\n==================== [FAIR COMPARISON PROTOCOL: {len(targets)} PHIÊN BẢN] ===================="
                )
                print(
                    f"[PROTOCOL] Device={device} | Batch_Size={batch_size} | Warmup={warmup_runs} "
                    f"| Benchmark_Runs={benchmark_runs} | Test_Samples={len(dm.test_dataset)}"
                )
                for base_model, version in targets:
                    evaluate_model_version(
                        base_model=base_model,
                        version=version,
                        config=config,
                        config_path=args.config,
                        dm=dm,
                        test_loader=test_loader,
                        device=device,
                        checkpoints_root=checkpoints_root,
                        runs_dir=runs_dir,
                        results_dir=results_dir,
                        warmup_runs=warmup_runs,
                        benchmark_runs=benchmark_runs,
                        explicit_ckpt=None,
                        run_name=getattr(args, "run_name", None),
                        log_to_tensorboard=True,
                    )
                return

            base_model, explicit_version = parse_model_arg(args.model)
            version = resolve_version_for_mode(
                checkpoints_root=checkpoints_root,
                runs_dir=runs_dir,
                base_model=base_model,
                explicit_version=explicit_version,
                mode=mode,
                run_name=getattr(args, "run_name", None),
            )
            evaluate_model_version(
                base_model=base_model,
                version=version,
                config=config,
                config_path=args.config,
                dm=dm,
                test_loader=test_loader,
                device=device,
                checkpoints_root=checkpoints_root,
                runs_dir=runs_dir,
                results_dir=results_dir,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
                explicit_ckpt=getattr(args, "checkpoint", None),
                run_name=getattr(args, "run_name", None),
                log_to_tensorboard=True,
            )
            return

        # Chế độ train hoặc all
        base_model, explicit_version = parse_model_arg(args.model)
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

        print(f"\n==================== [KHỞI TẠO MÔ HÌNH: {versioned_name} (Kiến trúc: {base_model})] ====================")
        print(f"[VERSION] Thư mục Checkpoint: {version_ckpt_dir}")
        model = build_model(base_model, config).to(device)

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

        target_scale = float(dm.scaler.scale_[dm.target_idx]) if hasattr(dm, "scaler") and dm.scaler is not None else 1.0
        tolerance_degC = float(eval_cfg.get("tolerance_degC", 1.5))
        print(f"[TRAINER] target_scale (σ của T°C trên Train) = {target_scale:.4f} | tolerance_scaled = {tolerance_degC / target_scale:.4f}")

        model_arch_cfg = config.get("models", {}).get(base_model, {})
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
        print(f"[TENSORBOARD] Ghi log thực nghiệm tại: {os.path.join(runs_dir, experiment_name)}")

        logger = TensorBoardLogger(
            log_dir=runs_dir,
            experiment_name=experiment_name,
        )

        epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 50)
        try:
            saved_cfg_path = save_version_config(
                version_ckpt_dir=version_ckpt_dir,
                base_model=base_model,
                version=version,
                config=config,
                epochs=epochs,
            )
            print(f"[CONFIG] Đã lưu cấu hình phiên bản tại: {saved_cfg_path}")

            config_summary = (
                f"**Model:** `{versioned_name}` | **Seed:** `{seed}` | **LR:** `{train_cfg.get('learning_rate', 0.001)}` | "
                f"**Scheduler:** `{train_cfg.get('scheduler_type', 'ReduceLROnPlateau')}` | "
                f"**Epochs:** `{epochs}` | **TF start/decay:** `{train_cfg.get('teacher_forcing_ratio_start', 0.5)}/{train_cfg.get('teacher_forcing_decay', 0.95)}` | "
                f"**target_scale:** `{target_scale:.4f}`"
            )
            logger.log_text("Experiment/Config", config_summary, step=0)

            log_figure_every = train_cfg.get("log_figure_every", 5)
            trainer.fit(train_loader, val_loader, epochs=epochs, logger=logger, log_figure_every=log_figure_every)

            if mode == "all":
                best_ckpt = resolve_best_checkpoint(
                    checkpoints_root=checkpoints_root,
                    version_ckpt_dir=version_ckpt_dir,
                    base_model=base_model,
                    explicit_ckpt=getattr(args, "checkpoint", None),
                )
                if not best_ckpt or not os.path.exists(best_ckpt):
                    raise FileNotFoundError(
                        f"[BLOCKED] Không tìm thấy checkpoint best sau khi huấn luyện trong '{version_ckpt_dir}'."
                    )
                print(f"[CHECKPOINT] Nạp trọng số tốt nhất từ: {best_ckpt}")
                trainer.load_checkpoint(
                    best_ckpt,
                    expected_model_name=base_model,
                    expected_model_config=model_arch_cfg,
                    strict=True,
                )
                evaluator = ModelEvaluator(
                    model=model,
                    device=device,
                    tolerance_degC=tolerance_degC,
                    results_dir=results_dir,
                )
                results = evaluator.evaluate(
                    test_loader=test_loader,
                    inverse_transform_fn=dm.inverse_transform_target,
                    model_name=base_model,
                    version=version,
                    logger=logger,
                    checkpoint_path=os.path.abspath(best_ckpt),
                    checkpoint_sha256=compute_file_sha256(best_ckpt),
                    config_path=os.path.abspath(args.config),
                    version_config_path=os.path.abspath(saved_cfg_path),
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                )
                logger.log_hparams(
                    hparam_dict={
                        "model": versioned_name,
                        "base_model": base_model,
                        "version": version,
                        "lr": float(train_cfg.get("learning_rate", 0.001)),
                        "scheduler": str(train_cfg.get("scheduler_type", "ReduceLROnPlateau")),
                        "batch_size": int(batch_size),
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
        finally:
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
        "--all_versions",
        action="store_true",
        help="Khi dùng cùng --mode evaluate: đánh giá công bằng đồng loạt toàn bộ các phiên bản (<model>_vN) trong checkpoints/ trên cùng tập test và thiết bị.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Số lượng Epoch muốn chạy (ghi đè giá trị trong config.yaml)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Kích thước batch đánh giá/huấn luyện (ghi đè data.batch_size trong config.yaml)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=None,
        help="Số lần chạy warmup trước khi benchmark latency (ghi đè evaluation.warmup_runs)",
    )
    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=None,
        help="Số lần lặp benchmark latency (ghi đè evaluation.benchmark_runs)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Thư mục gốc lưu kết quả đánh giá (ghi đè paths.evaluation_dir trong config.yaml)",
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
        help="Đường dẫn file checkpoint .pth cụ thể (tùy chọn, dùng khi muốn nạp file lẻ thay vì best.pth).",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
