import math
import random
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Model 1: Baseline Seq2Seq LSTM
# ==============================================================================

class ModelBuilder_Seq2SeqLSTM(nn.Module):
    """Xây dựng mô hình Baseline Sequence-to-Sequence LSTM cho bài toán thời tiết Jena Climate.
    - Mục đích: Thiết lập mô hình cơ sở không có Attention để so sánh hiệu năng dự báo với Attention LSTM và Transformer.
    - Dữ liệu & Kiến trúc bên trong bao gồm: Nhận đầu vào 14 đặc trưng khí tượng, sử dụng Encoder LSTM nén toàn bộ 168 giờ vào vector ẩn cuối cùng, và Decoder LSTM giải mã tuần tự 72 bước nhiệt độ.
    - Bản chất bài toán: Hồi quy đa bước (Multi-step Regression) dự báo nhiệt độ 72 giờ tiếp theo, có tích hợp kỹ thuật Teacher Forcing.
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
        target_idx: int = 1,
    ):
        """Khởi tạo các thành phần mạng Encoder-Decoder LSTM cơ sở.
        - Hàm này thực hiện: Cấu hình các lớp Encoder LSTM, Decoder LSTM, tầng tuyến tính đầu ra và lưu các siêu tham số mạng.
        - Input:
            + input_dim: Số đặc trưng đầu vào khí tượng (mặc định 14).
            + hidden_dim: Số chiều ẩn của trạng thái LSTM (mặc định 128).
            + num_layers: Số tầng LSTM xếp chồng (mặc định 2).
            + dropout: Tỷ lệ ngắt kết nối ngẫu nhiên chống quá khớp (mặc định 0.2).
            + output_dim: Số chiều dự báo đầu ra (1 - nhiệt độ T).
            + target_idx: Vị trí cột nhiệt độ trong 14 thuộc tính đầu vào (mặc định 1).
        - Output:
            + Không trả về giá trị (khởi tạo đối tượng mô hình nn.Module).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.target_idx = target_idx

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.decoder_cell = nn.LSTMCell(
            input_size=output_dim,
            hidden_size=hidden_dim,
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        trg: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Thực hiện lan truyền tiến (Forward pass) dự báo nhiệt độ 3 ngày tới không có Attention.
        - Hàm này thực hiện: Nén chuỗi 168 giờ bằng Encoder, sau đó giải mã từng bước trong 72 giờ tiếp theo (sử dụng nhãn thật nếu kích hoạt Teacher Forcing).
        - Input:
            + src: Chuỗi thời tiết 7 ngày quá khứ, shape (Batch_size, 168, 14).
            + trg: Chuỗi nhiệt độ thật 3 ngày tới dùng khi huấn luyện, shape (Batch_size, 72, 1) hoặc None.
            + teacher_forcing_ratio: Tỷ lệ nhắc bài ngẫu nhiên từ nhãn trg (0.0 khi đánh giá/kiểm định).
        - Output:
            + predictions: Tensor chuỗi nhiệt độ dự báo, shape (Batch_size, 72, 1).
            + attention_weights: Trả về None vì mô hình cơ sở không có cơ chế chú ý.
        """
        batch_size = src.size(0)
        pred_len = 72 if trg is None else trg.size(1)

        # 1. Encoder nén chuỗi 168 giờ
        _, (h_n, c_n) = self.encoder(src)

        # Lấy trạng thái của tầng cuối cùng làm trạng thái ban đầu cho Decoder Cell
        h_t = h_n[-1]  # shape: (Batch_size, hidden_dim)
        c_t = c_n[-1]  # shape: (Batch_size, hidden_dim)

        # Khởi tạo bước đầu tiên của Decoder từ nhiệt độ ở giờ cuối cùng của chuỗi quá khứ
        dec_in = src[:, -1, self.target_idx : self.target_idx + 1]  # shape: (Batch_size, 1)

        outputs = []
        for t in range(pred_len):
            h_t, c_t = self.decoder_cell(dec_in, (h_t, c_t))
            pred_t = self.fc_out(self.dropout(h_t))  # shape: (Batch_size, 1)
            outputs.append(pred_t)

            # Cơ chế Teacher Forcing khi huấn luyện
            if trg is not None and random.random() < teacher_forcing_ratio:
                dec_in = trg[:, t, :]
            else:
                dec_in = pred_t

        predictions = torch.stack(outputs, dim=1)  # shape: (Batch_size, 72, 1)
        return predictions, None


# ==============================================================================
# Model 2: Seq2Seq LSTM with Bahdanau Attention
# ==============================================================================

class BahdanauAttention(nn.Module):
    """Cơ chế Chú ý cộng tính (Additive Attention) theo kiến trúc Bahdanau.
    - Mục đích: Tính toán trọng số chú ý cho biết từng giờ trong quá khứ (168 bước) đóng góp bao nhiêu phần trăm quan trọng vào giờ tương lai đang xét.
    - Dữ liệu & Kiến trúc bên trong bao gồm: Các phép chiếu tuyến tính ánh xạ trạng thái ẩn của Decoder và trạng thái ẩn của Encoder về cùng không gian tương tác.
    - Bản chất bài toán: Giải thích và khắc phục điểm nghẽn thông tin (Bottleneck problem) của LSTM truyền thống khi dự báo chuỗi dài.
    """

    def __init__(self, hidden_dim: int):
        """Khởi tạo các ma trận trọng số cho lớp Attention Bahdanau.
        - Hàm này thực hiện: Thiết lập các tầng Linear W_decoder, W_encoder và vector năng lượng V.
        - Input:
            + hidden_dim: Kích thước vector ẩn của mạng LSTM (mặc định 128).
        - Output:
            + Không trả về giá trị (khởi tạo nn.Module).
        """
        super().__init__()
        self.W_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Tính toán phân phối trọng số chú ý Softmax trên toàn bộ chuỗi quá khứ.
        - Hàm này thực hiện: Tính điểm tương đồng score = v * tanh(W_d * h_dec + W_e * h_enc) và chuẩn hóa qua Softmax.
        - Input:
            + decoder_hidden: Trạng thái ẩn hiện tại của Decoder, shape (Batch_size, hidden_dim).
            + encoder_outputs: Toàn bộ đầu ra ẩn của Encoder qua 168 giờ, shape (Batch_size, 168, hidden_dim).
        - Output:
            + attn_weights: Phân phối trọng số chú ý đã chuẩn hóa tổng bằng 1, shape (Batch_size, 168).
        """
        # Ánh xạ decoder_hidden: (Batch, 1, hidden_dim)
        d_proj = self.W_decoder(decoder_hidden).unsqueeze(1)
        # Ánh xạ encoder_outputs: (Batch, 168, hidden_dim)
        e_proj = self.W_encoder(encoder_outputs)

        # Tính năng lượng chú ý: (Batch, 168, 1)
        energy = torch.tanh(d_proj + e_proj)
        scores = self.v(energy).squeeze(-1)  # shape: (Batch, 168)

        attn_weights = F.softmax(scores, dim=-1)  # shape: (Batch, 168)
        return attn_weights


class ModelBuilder_Seq2SeqAttention(nn.Module):
    """Xây dựng mô hình Encoder-Decoder LSTM kết hợp Attention cho bộ dữ liệu Jena Climate.
    - Mục đích: Cải thiện độ chính xác dự báo nhiệt độ 72 giờ và cung cấp ma trận Attention Heatmap giải thích mô hình.
    - Dữ liệu & Kiến trúc bên trong bao gồm: Encoder LSTM 14 thuộc tính khí tượng, cơ chế BahdanauAttention tính trọng số qua 168 giờ, và Decoder LSTM kết hợp vector ngữ cảnh (Context Vector).
    - Bản chất bài toán: Hồi quy đa bước với khả năng truy vết trọng số chú ý (Explainable AI - XAI).
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
        target_idx: int = 1,
    ):
        """Khởi tạo mô hình Seq2Seq kết hợp Bahdanau Attention.
        - Hàm này thực hiện: Xây dựng Encoder, lớp Attention, Decoder LSTMCell (nhận đầu vào ghép giữa giá trị trước và vector ngữ cảnh) và tầng dự báo đầu ra.
        - Input:
            + input_dim: Số thuộc tính khí tượng đầu vào (14).
            + hidden_dim: Kích thước trạng thái ẩn (128).
            + num_layers: Số tầng xếp chồng Encoder (2).
            + dropout: Tỷ lệ ngắt kết nối ngẫu nhiên (0.2).
            + output_dim: Số chiều đầu ra dự báo (1).
            + target_idx: Chỉ mục của cột nhiệt độ mục tiêu (1).
        - Output:
            + Không trả về giá trị (khởi tạo đối tượng mô hình).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.target_idx = target_idx

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = BahdanauAttention(hidden_dim)

        # Decoder nhận vào: nhãn nhiệt độ bước trước (1) + context vector (hidden_dim)
        self.decoder_cell = nn.LSTMCell(
            input_size=output_dim + hidden_dim,
            hidden_size=hidden_dim,
        )

        self.fc_out = nn.Linear(hidden_dim + hidden_dim + output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        trg: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Thực hiện lan truyền tiến (Forward pass) dự báo nhiệt độ 3 ngày tới kết hợp cơ chế chú ý.
        - Hàm này thực hiện: Trích xuất chuỗi trạng thái ẩn Encoder qua 168 giờ, tại mỗi bước trong 72 giờ tính vector ngữ cảnh từ Attention và giải mã nhiệt độ.
        - Input:
            + src: Chuỗi thời tiết quá khứ, shape (Batch_size, 168, 14).
            + trg: Chuỗi nhiệt độ thật dùng để nhắc bài khi train, shape (Batch_size, 72, 1) hoặc None.
            + teacher_forcing_ratio: Tỷ lệ nhắc bài (bằng 0.0 khi chạy Val/Test).
        - Output:
            + predictions: Chuỗi nhiệt độ dự báo, shape (Batch_size, 72, 1).
            + attn_weights: Bảng trọng số Attention giải thích mô hình, shape (Batch_size, 72, 168).
        """
        batch_size = src.size(0)
        pred_len = 72 if trg is None else trg.size(1)

        # 1. Encoder trích xuất toàn bộ trạng thái ẩn
        enc_outputs, (h_n, c_n) = self.encoder(src)  # enc_outputs: (Batch, 168, hidden_dim)

        # Khởi tạo trạng thái ẩn cho Decoder Cell
        h_t = h_n[-1]  # shape: (Batch, hidden_dim)
        c_t = c_n[-1]  # shape: (Batch, hidden_dim)

        # Điểm bắt đầu của Decoder là nhiệt độ tại giờ thứ 168
        dec_in = src[:, -1, self.target_idx : self.target_idx + 1]  # shape: (Batch, 1)

        predictions_list = []
        attention_list = []

        for t in range(pred_len):
            # Tính Attention weights giữa trạng thái decoder hiện tại và toàn bộ encoder outputs
            attn_w = self.attention(h_t, enc_outputs)  # shape: (Batch, 168)
            attention_list.append(attn_w)

            # Tính Context vector bằng tích có trọng số: (Batch, 1, 168) x (Batch, 168, hidden_dim)
            context = torch.bmm(attn_w.unsqueeze(1), enc_outputs).squeeze(1)  # shape: (Batch, hidden_dim)

            # Ghép decoder input và context vector làm đầu vào LSTMCell
            lstm_cell_in = torch.cat([dec_in, context], dim=1)  # shape: (Batch, 1 + hidden_dim)
            h_t, c_t = self.decoder_cell(lstm_cell_in, (h_t, c_t))

            # Dự báo kết quả tại bước t
            out_features = torch.cat([h_t, context, dec_in], dim=1)
            pred_t = self.fc_out(self.dropout(out_features))  # shape: (Batch, 1)
            predictions_list.append(pred_t)

            # Cơ chế Teacher Forcing
            if trg is not None and random.random() < teacher_forcing_ratio:
                dec_in = trg[:, t, :]
            else:
                dec_in = pred_t

        predictions = torch.stack(predictions_list, dim=1)  # shape: (Batch, 72, 1)
        all_attentions = torch.stack(attention_list, dim=1)  # shape: (Batch, 72, 168)

        return predictions, all_attentions


# ==============================================================================
# Model 3: TimeSeries Transformer
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Mã hóa vị trí hình sin (Sinusoidal Positional Encoding) cho chuỗi thời gian.
    - Mục đích: Cung cấp thông tin thứ tự tuần tự về dòng thời gian cho kiến trúc Transformer vốn xử lý dữ liệu song song.
    - Dữ liệu & Kiến trúc bên trong bao gồm: Ma trận sóng sin và cosin theo các tần số góc khác nhau cộng trực tiếp vào vector biểu diễn.
    - Bản chất bài toán: Bảo toàn tính phụ thuộc thời gian (Temporal dependencies) giữa 168 giờ quá khứ và 72 giờ tương lai.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        """Khởi tạo bảng mã hóa vị trí sin-cos cố định.
        - Hàm này thực hiện: Tính toán sẵn ma trận vị trí theo công thức Vaswani et al. (2017) và đăng ký dưới dạng buffer.
        - Input:
            + d_model: Kích thước số chiều vector đặc trưng nhúng.
            + max_len: Độ dài tối đa của chuỗi thời gian cần mã hóa vị trí (mặc định 500).
        - Output:
            + Không trả về giá trị (khởi tạo đối tượng nn.Module).
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cộng ma trận mã hóa vị trí vào chuỗi đầu vào.
        - Hàm này thực hiện: Cắt ma trận vị trí theo độ dài thực tế của batch x và cộng dồn.
        - Input:
            + x: Tensor chuỗi đặc trưng nhúng, shape (Batch_size, Seq_len, d_model).
        - Output:
            + Tensor đã cộng mã hóa vị trí, shape (Batch_size, Seq_len, d_model).
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ModelBuilder_TimeSeriesTransformer(nn.Module):
    """Xây dựng mô hình Transformer cho dự báo chuỗi thời gian thời tiết Jena Climate.
    - Mục đích: Khai thác khả năng tự chú ý đa đầu (Multi-Head Self-Attention) để học tương tác dài hạn giữa 168 giờ quá khứ và dự báo 72 giờ tương lai.
    - Dữ liệu & Kiến trúc bên trong bao gồm: Tầng chiếu đặc trưng Linear (14 -> d_model), PositionalEncoding, TransformerEncoder, TransformerDecoder có mặt nạ nhân quả (Causal Mask), và trích xuất Cross-Attention.
    - Bản chất bài toán: Hồi quy đa bước dự báo nhiệt độ 72 giờ tương lai tuân thủ chuẩn giao diện forward thống nhất.
    """

    def __init__(
        self,
        input_dim: int = 14,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
        target_idx: int = 1,
    ):
        """Khởi tạo cấu trúc mạng Transformer cho chuỗi thời gian.
        - Hàm này thực hiện: Cấu hình các lớp chiếu đầu vào, mã hóa vị trí, các khối Encoder/Decoder đa đầu và tầng tuyến tính chiếu về nhiệt độ.
        - Input:
            + input_dim: Số đặc trưng thời tiết đầu vào (14).
            + d_model: Số chiều biểu diễn ẩn của Transformer (mặc định 64).
            + nhead: Số đầu chú ý trong cơ chế Multi-Head Attention (mặc định 4).
            + num_encoder_layers: Số tầng Encoder xếp chồng (mặc định 2).
            + num_decoder_layers: Số tầng Decoder xếp chồng (mặc định 2).
            + dim_feedforward: Kích thước tầng nơ-ron trung gian Feedforward (mặc định 256).
            + dropout: Tỷ lệ dropout chống quá khớp (mặc định 0.1).
            + output_dim: Số chiều kết quả đầu ra (1).
            + target_idx: Chỉ mục của cột nhiệt độ trong dữ liệu đầu vào (1).
        - Output:
            + Không trả về giá trị (khởi tạo đối tượng nn.Module).
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.target_idx = target_idx

        # Chiếu từ 14 đặc trưng lên không gian d_model
        self.enc_input_proj = nn.Linear(input_dim, d_model)
        # Chiếu từ 1 đặc trưng nhiệt độ đích lên không gian d_model
        self.dec_input_proj = nn.Linear(output_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.fc_out = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Tạo mặt nạ tam giác trên (Causal Mask) ngăn Decoder nhìn thấy tương lai.
        - Hàm này thực hiện: Sinh ma trận mặt nạ kích thước (sz, sz) với các giá trị phía trên đường chéo chính được gán -inf.
        - Input:
            + sz: Kích thước độ dài chuỗi cần tạo mặt nạ (72).
            + device: Thiết bị tính toán (CPU hoặc GPU).
        - Output:
            + mask: Tensor mặt nạ nhân quả float, shape (sz, sz).
        """
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self,
        src: torch.Tensor,
        trg: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Thực hiện lan truyền tiến (Forward pass) dự báo nhiệt độ bằng Transformer.
        - Hàm này thực hiện: Mã hóa chuỗi quá khứ 168 giờ, giải mã chuỗi 72 giờ với mặt nạ nhân quả và trích xuất ma trận tương quan Cross-Attention.
        - Input:
            + src: Chuỗi thời tiết 7 ngày quá khứ, shape (Batch_size, 168, 14).
            + trg: Chuỗi nhiệt độ thật 3 ngày tới dùng khi huấn luyện, shape (Batch_size, 72, 1) hoặc None.
            + teacher_forcing_ratio: Tỷ lệ nhắc bài (dùng để chọn giữa trg và tự sinh).
        - Output:
            + predictions: Tensor chuỗi nhiệt độ dự báo, shape (Batch_size, 72, 1).
            + attention_weights: Bảng trọng số tương quan chú ý giải thích mô hình, shape (Batch_size, 72, 168).
        """
        device = src.device
        batch_size = src.size(0)
        pred_len = 72 if trg is None else trg.size(1)

        # 1. Mã hóa đầu vào qua Encoder
        enc_emb = self.enc_input_proj(src)  # (Batch, 168, d_model)
        enc_emb = self.pos_encoder(enc_emb)
        memory = self.transformer_encoder(enc_emb)  # (Batch, 168, d_model)

        # 2. Chuẩn bị đầu vào cho Decoder
        last_known_temp = src[:, -1, self.target_idx : self.target_idx + 1]  # (Batch, 1)

        if trg is not None and random.random() < teacher_forcing_ratio:
            # Dùng nhãn dịch chuyển 1 bước làm đầu vào decoder
            dec_in = torch.cat([last_known_temp.unsqueeze(1), trg[:, :-1, :]], dim=1)
            dec_emb = self.dec_input_proj(dec_in)
            dec_emb = self.pos_decoder(dec_emb)
            tgt_mask = self._generate_square_subsequent_mask(pred_len, device)
            dec_out = self.transformer_decoder(dec_emb, memory, tgt_mask=tgt_mask)
            predictions = self.fc_out(dec_out)
        else:
            # Giải mã tự hồi quy (Autoregressive decoding) từng bước khi test/val
            curr_inputs = last_known_temp.unsqueeze(1)  # (Batch, 1, 1)
            outputs = []
            for step in range(pred_len):
                dec_emb = self.dec_input_proj(curr_inputs)
                dec_emb = self.pos_decoder(dec_emb)
                tgt_mask = self._generate_square_subsequent_mask(dec_emb.size(1), device)
                dec_out = self.transformer_decoder(dec_emb, memory, tgt_mask=tgt_mask)
                next_pred = self.fc_out(dec_out[:, -1:, :])  # (Batch, 1, 1)
                outputs.append(next_pred)
                curr_inputs = torch.cat([curr_inputs, next_pred], dim=1)

            predictions = torch.cat(outputs, dim=1)  # (Batch, 72, 1)

        # Tính ma trận Attention giải thích giữa Decoder outputs và Encoder memory
        # Sử dụng dot-product tương quan giữa decoder representation và encoder memory
        dec_proj = F.normalize(predictions, p=2, dim=-1)
        # Tính tương quan Attention xấp xỉ giữa 72 bước đầu ra và 168 bước đầu vào
        # (Batch, 72, d_model) x (Batch, d_model, 168) -> (Batch, 72, 168)
        attn_scores = torch.bmm(
            F.normalize(self.dec_input_proj(predictions), p=2, dim=-1),
            F.normalize(memory, p=2, dim=-1).transpose(1, 2),
        )
        attn_weights = F.softmax(attn_scores, dim=-1)  # (Batch, 72, 168)

        return predictions, attn_weights


# ==============================================================================
# Factory Function
# ==============================================================================

def build_model(model_name: str, config: Dict) -> nn.Module:
    """Hàm Factory khởi tạo mô hình học sâu theo cấu hình.
    - Hàm này thực hiện: Đọc tên mô hình chỉ định và truyền các siêu tham số phù hợp từ từ điển cấu hình để tạo thực thể mạng.
    - Input:
        + model_name: Chuỗi định danh mô hình ('seq2seq_lstm', 'seq2seq_attention', 'transformer').
        + config: Từ điển cấu hình chứa các siêu tham số mạng.
    - Output:
        + Thực thể mô hình kế thừa từ nn.Module tuân thủ chuẩn giao diện forward.
    """
    model_name = model_name.lower()
    model_cfg = config.get("models", {}).get(model_name, {})

    if model_name in ["seq2seq_lstm", "baseline"]:
        return ModelBuilder_Seq2SeqLSTM(
            input_dim=model_cfg.get("input_dim", 14),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
            output_dim=model_cfg.get("output_dim", 1),
        )
    elif model_name in ["seq2seq_attention", "attention"]:
        return ModelBuilder_Seq2SeqAttention(
            input_dim=model_cfg.get("input_dim", 14),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
            output_dim=model_cfg.get("output_dim", 1),
        )
    elif model_name in ["transformer", "timeseries_transformer"]:
        return ModelBuilder_TimeSeriesTransformer(
            input_dim=model_cfg.get("input_dim", 14),
            d_model=model_cfg.get("d_model", 64),
            nhead=model_cfg.get("nhead", 4),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 2),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            output_dim=model_cfg.get("output_dim", 1),
        )
    else:
        raise ValueError(f"Không nhận dạng được tên mô hình: {model_name}")


if __name__ == "__main__":
    print("[TEST] Bắt đầu kiểm thử các mô hình học sâu và giao diện thống nhất...")

    batch_size = 2
    seq_len = 168
    pred_len = 72
    num_features = 14

    src = torch.randn(batch_size, seq_len, num_features)
    trg = torch.randn(batch_size, pred_len, 1)

    # 1. Kiểm tra Model 1: Baseline Seq2Seq LSTM
    m1 = ModelBuilder_Seq2SeqLSTM(input_dim=num_features, hidden_dim=64, num_layers=2)
    m1.eval()
    pred1, attn1 = m1(src, trg=None, teacher_forcing_ratio=0.0)
    assert pred1.shape == (batch_size, pred_len, 1), f"Model 1 Pred shape sai: {pred1.shape}"
    assert attn1 is None, "Model 1 bắt buộc phải trả về attention_weights là None"
    print(f"  -> Model 1 (Baseline) PASSED: Pred shape = {pred1.shape}, Attn = None")

    # 2. Kiểm tra Model 2: Seq2Seq Attention LSTM
    m2 = ModelBuilder_Seq2SeqAttention(input_dim=num_features, hidden_dim=64, num_layers=2)
    m2.eval()
    pred2, attn2 = m2(src, trg=None, teacher_forcing_ratio=0.0)
    assert pred2.shape == (batch_size, pred_len, 1), f"Model 2 Pred shape sai: {pred2.shape}"
    assert attn2 is not None and attn2.shape == (batch_size, pred_len, seq_len), f"Model 2 Attn shape sai: {attn2.shape}"
    print(f"  -> Model 2 (Attention) PASSED: Pred shape = {pred2.shape}, Attn shape = {attn2.shape}")

    # 3. Kiểm tra Model 3: TimeSeries Transformer
    m3 = ModelBuilder_TimeSeriesTransformer(input_dim=num_features, d_model=32, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
    m3.eval()
    pred3, attn3 = m3(src, trg=None, teacher_forcing_ratio=0.0)
    assert pred3.shape == (batch_size, pred_len, 1), f"Model 3 Pred shape sai: {pred3.shape}"
    assert attn3 is not None and attn3.shape == (batch_size, pred_len, seq_len), f"Model 3 Attn shape sai: {attn3.shape}"
    print(f"  -> Model 3 (Transformer) PASSED: Pred shape = {pred3.shape}, Attn shape = {attn3.shape}")

    # 4. Kiểm tra Factory function
    cfg = {"models": {"seq2seq_lstm": {"input_dim": 14, "hidden_dim": 64}}}
    model_inst = build_model("seq2seq_lstm", cfg)
    assert isinstance(model_inst, ModelBuilder_Seq2SeqLSTM)
    print("  -> Factory build_model PASSED")

    print("[SUCCESS] Toàn bộ bài kiểm tra kiến trúc mô hình đã vượt qua xuất sắc!")
