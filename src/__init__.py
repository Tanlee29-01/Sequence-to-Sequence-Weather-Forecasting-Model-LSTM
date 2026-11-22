"""Package mã nguồn cho bài toán Sequence-to-Sequence Weather Forecasting (Topic 5).
Bao gồm các module:
- data_module: Xử lý dữ liệu thời tiết Jena Climate theo chuẩn chống rò rỉ 4 pha.
- model_builder: Định nghĩa 3 kiến trúc mô hình (Seq2Seq LSTM, Seq2Seq Attention, TimeSeries Transformer).
- trainer: Quản lý quá trình huấn luyện, Teacher Forcing, Early Stopping, Gradient Clipping và đo tài nguyên.
- logger: Ghi nhận TensorBoard log và biểu đồ trực quan hóa.
- eda: Phân tích khám phá dữ liệu chỉ trên tập Train.
- evaluation: Đánh giá mô hình trên tập Test, đo độ trễ suy luận, tài nguyên và Attention weights.
"""

__version__ = "1.0.0"
