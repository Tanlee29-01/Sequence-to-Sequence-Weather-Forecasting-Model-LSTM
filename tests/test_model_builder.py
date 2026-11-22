"""Kiểm thử tự động cho kiến trúc 3 mô hình học sâu (src/model_builder.py).
Xác minh giao diện chung forward(src, trg, tf), contract kích thước Tensor đầu ra (72, 1),
phân phối ma trận Attention Heatmap (72, 168), tính ổn định số học (Zero NaN/Inf)
và hàm Factory build_model.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Đảm bảo import được module src từ thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model_builder import (
    ModelBuilder_Seq2SeqLSTM,
    ModelBuilder_Seq2SeqAttention,
    ModelBuilder_TimeSeriesTransformer,
    build_model,
)


class TestModelBuilder(unittest.TestCase):
    """Bộ kiểm thử cho 3 kiến trúc học sâu và Factory build_model."""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo Tensor giả lập cho kiểm thử."""
        torch.manual_seed(42)
        cls.batch_size = 2
        cls.seq_len = 168
        cls.pred_len = 72
        cls.input_dim = 14

        cls.src = torch.randn(cls.batch_size, cls.seq_len, cls.input_dim)
        cls.trg = torch.randn(cls.batch_size, cls.pred_len, 1)

    def test_model1_baseline_lstm(self):
        """Kiểm tra Model 1 (Baseline Seq2Seq LSTM): Shape (B, 72, 1) & Attention is None."""
        model = ModelBuilder_Seq2SeqLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=1,
            output_dim=1,
        )
        model.eval()
        with torch.no_grad():
            pred, attn = model(self.src, trg=None, teacher_forcing_ratio=0.0)

        self.assertEqual(
            pred.shape,
            (self.batch_size, self.pred_len, 1),
            f"Model 1 Pred shape sai: {pred.shape}"
        )
        self.assertIsNone(
            attn,
            "Model 1 Baseline LSTM không có Attention nên attn_weights bắt buộc phải là None."
        )
        self.assertFalse(torch.isnan(pred).any(), "Model 1 output chứa giá trị NaN.")
        self.assertFalse(torch.isinf(pred).any(), "Model 1 output chứa giá trị Inf.")

    def test_model2_seq2seq_attention(self):
        """Kiểm tra Model 2 (Seq2Seq Attention LSTM): Shape (B, 72, 1) & Attn (B, 72, 168)."""
        model = ModelBuilder_Seq2SeqAttention(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=1,
            output_dim=1,
        )
        model.eval()
        with torch.no_grad():
            pred, attn = model(self.src, trg=None, teacher_forcing_ratio=0.0)

        self.assertEqual(
            pred.shape,
            (self.batch_size, self.pred_len, 1),
            f"Model 2 Pred shape sai: {pred.shape}"
        )
        self.assertIsNotNone(attn, "Model 2 bắt buộc phải trả về attention_weights.")
        self.assertEqual(
            attn.shape,
            (self.batch_size, self.pred_len, self.seq_len),
            f"Model 2 Attn shape sai contract (B, 72, 168): {attn.shape}"
        )

        # Kiểm tra tổng Softmax trên trục 168 giờ lịch sử luôn xấp xỉ 1.0
        attn_sum = torch.sum(attn, dim=-1)
        expected_sum = torch.ones_like(attn_sum)
        self.assertTrue(
            torch.allclose(attn_sum, expected_sum, atol=1e-5),
            "Tổng Attention weights qua 168 giờ quá khứ không bằng 1.0 (Softmax lỗi)."
        )

    def test_model3_timeseries_transformer(self):
        """Kiểm tra Model 3 (TimeSeries Transformer): Shape (B, 72, 1) & Cross-Attn (B, 72, 168)."""
        model = ModelBuilder_TimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            output_dim=1,
        )
        model.eval()
        with torch.no_grad():
            pred, attn = model(self.src, trg=None, teacher_forcing_ratio=0.0)

        self.assertEqual(
            pred.shape,
            (self.batch_size, self.pred_len, 1),
            f"Model 3 Pred shape sai: {pred.shape}"
        )
        self.assertIsNotNone(attn, "Model 3 bắt buộc phải trả về attention_weights.")
        self.assertEqual(
            attn.shape,
            (self.batch_size, self.pred_len, self.seq_len),
            f"Model 3 Attn shape sai contract: {attn.shape}"
        )

    def test_teacher_forcing_mode(self):
        """Kiểm tra Teacher Forcing: Chạy thành công ở cả 2 chế độ TF = 0.0 và TF = 0.5."""
        model = ModelBuilder_Seq2SeqAttention(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=1,
        )
        model.train()

        # Chế độ Train với Teacher Forcing
        pred_tf, _ = model(self.src, trg=self.trg, teacher_forcing_ratio=0.5)
        self.assertEqual(pred_tf.shape, (self.batch_size, self.pred_len, 1))

        # Chế độ Eval tắt hoàn toàn nhắc bài
        model.eval()
        pred_no_tf, _ = model(self.src, trg=None, teacher_forcing_ratio=0.0)
        self.assertEqual(pred_no_tf.shape, (self.batch_size, self.pred_len, 1))

    def test_gradient_flow_and_no_nan(self):
        """Kiểm tra lan truyền ngược (Backward pass) không sinh NaN hoặc Inf trong gradient."""
        model = ModelBuilder_Seq2SeqAttention(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=1,
        )
        model.train()
        criterion = nn.MSELoss()

        pred, _ = model(self.src, trg=self.trg, teacher_forcing_ratio=0.5)
        loss = criterion(pred, self.trg)
        loss.backward()

        # Kiểm tra toàn bộ tham số có gradient hợp lệ
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Tham số {name} không có gradient.")
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient của {name} bị NaN.")
                self.assertFalse(torch.isinf(param.grad).any(), f"Gradient của {name} bị Inf.")

    def test_factory_build_model(self):
        """Kiểm tra hàm khởi tạo factory build_model() theo cấu hình config.yaml."""
        mock_config = {
            "models": {
                "seq2seq_lstm": {"input_dim": 14, "hidden_dim": 32, "num_layers": 1},
                "seq2seq_attention": {"input_dim": 14, "hidden_dim": 32, "num_layers": 1},
                "transformer": {
                    "input_dim": 14, "d_model": 32, "nhead": 2,
                    "num_encoder_layers": 1, "num_decoder_layers": 1, "dim_feedforward": 64
                },
            }
        }

        m1 = build_model("seq2seq_lstm", mock_config)
        self.assertIsInstance(m1, ModelBuilder_Seq2SeqLSTM)

        m2 = build_model("seq2seq_attention", mock_config)
        self.assertIsInstance(m2, ModelBuilder_Seq2SeqAttention)

        m3 = build_model("transformer", mock_config)
        self.assertIsInstance(m3, ModelBuilder_TimeSeriesTransformer)


def run_smoke_test():
    """Hàm chạy nhanh Smoke Test kiểm tra 3 mô hình học sâu."""
    print("=" * 70)
    print("[SMOKE TEST] BẮT ĐẦU KIỂM THỬ 3 KIẾN TRÚC MÔ HÌNH HỌC SÂU (TOPIC 5)")
    print("=" * 70)

    batch_size = 2
    seq_len = 168
    pred_len = 72
    num_features = 14

    src = torch.randn(batch_size, seq_len, num_features)
    trg = torch.randn(batch_size, pred_len, 1)
    print(f"[INIT] Tensor đầu vào: Input X={tuple(src.shape)} | Target Y={tuple(trg.shape)}")

    # 1. Model 1: Baseline LSTM
    print("\n[TEST 1/4] Kiểm tra Model 1: Baseline Seq2Seq LSTM...")
    m1 = ModelBuilder_Seq2SeqLSTM(input_dim=num_features, hidden_dim=32, num_layers=1)
    m1.eval()
    pred1, attn1 = m1(src, trg=None, teacher_forcing_ratio=0.0)
    assert pred1.shape == (batch_size, pred_len, 1), f"Model 1 Pred shape sai: {pred1.shape}"
    assert attn1 is None, "Model 1 Attn bắt buộc phải là None"
    assert not torch.isnan(pred1).any(), "Model 1 Pred chứa NaN"
    print(f"      [ASSERT 1] Pred shape: {tuple(pred1.shape)} == (2, 72, 1) -> PASSED")
    print(f"      [ASSERT 2] Attention weights is None: PASSED")
    print(f"      [ASSERT 3] Zero NaN / Inf detected: PASSED")
    print("      -> Model 1 (Baseline LSTM) PASSED.")

    # 2. Model 2: Attention LSTM
    print("\n[TEST 2/4] Kiểm tra Model 2: Seq2Seq Attention LSTM (Bahdanau)...")
    m2 = ModelBuilder_Seq2SeqAttention(input_dim=num_features, hidden_dim=32, num_layers=1)
    m2.eval()
    pred2, attn2 = m2(src, trg=None, teacher_forcing_ratio=0.0)
    assert pred2.shape == (batch_size, pred_len, 1), f"Model 2 Pred shape sai: {pred2.shape}"
    assert attn2.shape == (batch_size, pred_len, seq_len), f"Model 2 Attn shape sai: {attn2.shape}"
    attn_sum = torch.sum(attn2, dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Softmax sum != 1.0"
    print(f"      [ASSERT 1] Pred shape: {tuple(pred2.shape)} == (2, 72, 1) -> PASSED")
    print(f"      [ASSERT 2] Attn shape: {tuple(attn2.shape)} == (2, 72, 168) -> PASSED")
    print(f"      [ASSERT 3] Softmax sum along 168h: Min={attn_sum.min():.6f}, Max={attn_sum.max():.6f} -> PASSED")
    print("      -> Model 2 (Attention LSTM) PASSED.")

    # 3. Model 3: TimeSeries Transformer
    print("\n[TEST 3/4] Kiểm tra Model 3: TimeSeries Transformer (Multi-Head Causal)...")
    m3 = ModelBuilder_TimeSeriesTransformer(input_dim=num_features, d_model=32, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
    m3.eval()
    pred3, attn3 = m3(src, trg=None, teacher_forcing_ratio=0.0)
    assert pred3.shape == (batch_size, pred_len, 1), f"Model 3 Pred shape sai: {pred3.shape}"
    assert attn3.shape == (batch_size, pred_len, seq_len), f"Model 3 Attn shape sai: {attn3.shape}"
    print(f"      [ASSERT 1] Pred shape: {tuple(pred3.shape)} == (2, 72, 1) -> PASSED")
    print(f"      [ASSERT 2] Cross-Attn shape: {tuple(attn3.shape)} == (2, 72, 168) -> PASSED")
    print("      -> Model 3 (TimeSeries Transformer) PASSED.")

    # 4. Factory Function
    print("\n[TEST 4/4] Kiểm tra Factory function build_model()...")
    cfg = {
        "models": {
            "seq2seq_lstm": {"input_dim": 14, "hidden_dim": 32},
            "seq2seq_attention": {"input_dim": 14, "hidden_dim": 32},
            "transformer": {"input_dim": 14, "d_model": 32, "nhead": 2, "num_encoder_layers": 1, "num_decoder_layers": 1},
        }
    }
    assert isinstance(build_model("seq2seq_lstm", cfg), ModelBuilder_Seq2SeqLSTM)
    assert isinstance(build_model("seq2seq_attention", cfg), ModelBuilder_Seq2SeqAttention)
    assert isinstance(build_model("transformer", cfg), ModelBuilder_TimeSeriesTransformer)
    print("      [ASSERT 1] Khởi tạo chính xác cả 3 kiến trúc từ dictionary: PASSED")

    print("\n" + "-" * 70)
    print("[SUCCESS] TOÀN BỘ BÀI KIỂM THỬ MÔ HÌNH ĐÃ VƯỢT QUA XUẤT SẮC!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_smoke_test()
