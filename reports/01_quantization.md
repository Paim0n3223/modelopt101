# Bài 1: Quantization

## 1. Bài này làm gì

- Notebook: `examples/01_quantization.ipynb`, chạy trên Google Colab (GPU T4).
- Dữ liệu MNIST, model là CNN kiểu VGG (6 lớp conv + 1 Linear, khoảng 1.1M tham số), train FP32 12 epoch.
- Mục tiêu: lưu weight của Conv và Linear bằng INT8 thay cho FP32 (post-training quantization, không train lại).
- Đo size, accuracy, latency trước và sau.

## 2. Ý tưởng em hiểu được

- Đề bài yêu cầu viết các hàm để ánh xạ (maps) các giá trị FP32 sang INT8, theo các hiểu của em thì có thể phát biểu nôm na là "Ướm thước float lên thước int": tìm xem mỗi số thực thuộc về vạch nào trên thước INT8.
- Độ dài thước float không lấy theo khả năng của FP32 vì khoảng số thực của FP32 là quá lớn so với bài toán nên em lấy bằng việc đo từ chính dữ liệu: min và max của tensor (hoặc của từng channel).
- Không làm tròn thẳng được vì weight cỡ 0.1 làm tròn thẳng sẽ thành 0 hết. Phải ướm thước (tính scale) trước, rồi chia cho scale để biết mỗi số nằm ở vạch nào, sau đó mới làm tròn.
- Ví dụ tính tay: x = [0.31, -0.12, 0.05, -0.40], scale = 0.40 / 127 ≈ 0.00315, q = [98, -38, 16, -127], dequantize ra [0.3087, -0.1197, 0.0504, -0.4000].
- Notebool có bước truy ngược, tuy nhiên không thể truy ngược được chính xác. Phần mất đi là quantization error, tối đa bằng nửa bước thước (scale / 2).
- Có 2 cách scale là symmetric và affine: symmetric là thước đối xứng quanh 0 (zero_point = 0), affine là thước ôm sát [min, max] (cần zero_point để biết số 0 thật nằm ở vạch nào).
- Per-tensor và per-channel: một cây thước cho cả tensor, hoặc mỗi filter (output channel) một cây thước riêng. Lớp conv đầu có shape [64, 1, 3, 3], tức 64 filter, mỗi filter 9 weight.

## 3. Phần em viết (3 TODO)

- TODO 1, symmetric: `scale = (torch.maximum(xmax.abs(), xmin.abs()) / qmax).clamp(min=1e-8)`, zero_point = 0.
- TODO 2, affine: `scale = ((xmax - xmin) / (qmax - qmin)).clamp(min=1e-8)`, `zero_point = qmin - torch.round(xmin / scale)`.
- TODO 3: `q = (torch.round(x / s) + z).clamp(min=qmin, max=qmax)`.
- `.clamp(min=1e-8)` để tránh chia cho 0 khi một channel có mọi giá trị bằng nhau với affine, còn với symmetric, scale = 0 khi channel toàn số 0.

## 4. Những chỗ em vấp

- Chạy với code placeholder không báo lỗi, nhưng kết quả "đẹp giả": MSE = 0, `int codes in [-0, 0]`, accuracy INT8 bằng đúng FP32, size vẫn báo 3.9x nhỏ hơn vì placeholder trả lại nguyên weight gốc, chưa quantize gì.
- Lần đầu viết TODO 3 dùng `scale` thay vì `s`. Chạy per-tensor thì không sao, nhưng per-channel sẽ sai vì broadcast căn theo chiều cuối.
- Dùng `round()` của Python thay vì `torch.round()`.
- Quên `.clamp(min=1e-8)`. Với affine, thiếu clamp có thể ra NaN hoặc inf.

## 5. Kết quả (8 bit, symmetric, per-channel)

| | Size | Accuracy | CPU latency |
|---|---|---|---|
| FP32 | 4.61 MB | 99.67% | 5.42 ms |
| INT8 | 1.18 MB | 99.67% | 5.54 ms |

- MSE trên lớp conv đầu: per-tensor 3.000e-07, per-channel 1.332e-07 (nhỏ hơn khoảng 2.3 lần).
- Sau khi sửa: `int codes in [-127, 127]`.
- Size nhỏ đi 3.9x chứ không tròn 4x vì bias và BatchNorm vẫn FP32, mỗi channel lưu thêm một scale.
- Latency không đổi (chênh lệch là nhiễu, chạy lại có lúc đảo ngược) vì notebook dequantize weight về FP32 rồi tính như thường (fake-quant). Muốn nhanh thật cần kernel INT8, như trong llama.cpp hoặc vla.cpp.

## 6. Thí nghiệm thêm

### Symmetric và affine (8 bit, per-channel)

| | MSE | Khoảng q |
|---|---|---|
| Symmetric | 1.332e-07 | [-127, 127] |
| Affine | 1.183e-07 | [-128, 127] |

- Affine tốt hơn khoảng 11%. Ít vì weight gần đối xứng quanh 0 (khoảng -0.24 đến 0.21).
- Affine sẽ đáng giá hơn với dữ liệu lệch một phía, ví dụ activation sau ReLU.

### Giảm số bit

| Bit | Số mức dùng | MSE per-tensor | MSE per-channel | Acc per-tensor | Acc per-channel |
|---|---|---|---|---|---|
| 8 | 255 | 3.000e-07 | 1.332e-07 | 99.67% | 99.67% |
| 4 | 15 | 9.400e-05 | 4.353e-05 | 99.67% | 99.67% |
| 3 | 7 | (quên không ghi) | (quên không ghi) | (quên không ghi) | 99.53% |
| 2 | 3 | 6.313e-03 | 2.145e-03 | 9.80% | 20.23% |

- Điểm gãy nằm ở đâu đó giữa 3 bit và 2 bit. Accuracy không giảm dần mà giữ nguyên rồi sập khi đi qua điểm gãy
- 2 bit: per-tensor 9.80% tương đương đoán bừa (10 lớp - xác suất đoán bừa trúng là 10%).
- Histogram 4 bit: một cột lớn ở 0 (khoảng 38 weight bị làm tròn về 0).
- Histogram 2 bit: chỉ còn 3 cụm, khoảng 265 / 576 weight thành 0. Weight chỉ còn giữ dấu, mất thông tin độ lớn.
- 3 bit còn 7 mức, vẫn phân biệt được yếu, vừa, mạnh. 2 bit chỉ còn 3 mức.
- Ước lượng sai số trung bình so với weight cỡ 0.1: khoảng 0.4% (8 bit), 7% (4 bit), 46% (2 bit).
- Size vẫn 1.18 MB ở mọi mức bit, vì q vẫn lưu bằng kiểu int8 (1 byte). Muốn nhỏ hơn thật phải gói nhiều số vào một byte (packing).
- Mức bit chịu được tùy model và bài toán. MNIST dễ nên chịu được tới 3 bit.
- Cách xuống thấp hơn (chưa thử): affine ở 2 bit (4 mức thay vì 3), giữ lớp đầu và cuối ở 8 bit (mixed precision), hoặc train có quantization (QAT) như BitNet.

## 7. Điều em rút ra

- Quá trình quantization chắc chắn giảm bộ nhớ, còn tốc độ phụ thuộc kernel và phần cứng.
- Việc khó khăn nhất theo em là đặt thước cho khéo vì đôi khi có những lúc outlier cực lớn, nếu cứ lấy max(abs()) thì sẽ tạo ra một thước quá dài, phần lớn giá trị có ích sẽ bị map sai.
- Bài toán MNIST vẫn còn khá dễ, đến mức khả năng biểu thị hạn chế của 3 bits vẫn có thể xử lý tốt, các bài toán phức tạp thường sẽ cần số bits lớn hơn.
