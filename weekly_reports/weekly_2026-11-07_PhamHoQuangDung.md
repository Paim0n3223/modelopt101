# Báo cáo tuần

- Người viết: Phạm Hồ Quang Dũng
- Thời gian: 07/11/2026
- Repo: https://github.com/Paim0n3223/modelopt101

## 1. Summary

### Highlight công việc trong tuần

- Đọc slide về các phương pháp tối ưu model cho robot
- Thực hành 5 examples trên Google Colab/máy cá nhân, sau đây tóm tắt ngắn gọn kết quả:
  - Quantization giảm size đi xấp xỉ 4 lần khi lưu dưới dạng INT8, accuracy và latency giữ nguyên.
  - Knowledge distillation giúp student nhỏ hơn 79 lần, nhanh hơn 8.6 lần, và KD giúp student giảm lỗi khoảng 20%.
  - Với Pruning: Unstructured cut khiến accuracy giảm nhẹ, size và latency không đổi, Structured cut giảm size 4 lần và latency 2 lần, accuracy giữ nguyên.
  - NAS chọn ra kiến trúc d4-w12-k3, có size giảm 1.65 lần, chạy nhanh hơn 1.7 lần tuy nhiên con số này không ổn định, accuracy giữ nguyên so với baseline là kiến trúc d3-w32-k3.
  - Kernel Optimization: Kernel Triton em tự viết chạy đúng nhưng chậm hơn cuBLAS 15–25 lần, lí do là vì trên GPU T4 của Google Colab, Triton biên dịch lệnh tl.dot thành nhân cộng thường trên CUDA core, còn cuBLAS dùng tensor core - chuyên dụng cho nhân ma trận nên nhanh hơn nhiều lần.
- Thực hành SmolVLA trên máy cá nhân, chạy bằng vla.cpp và đo trong LIBERO:
  - Đã cài đặt và chạy thử test, CLI, server, Python binding và vla_predict_check.
  - Chạy baseline cho ra kết quả latency p50 và p90 tương ứng là 163.9 ms và 164.3 ms (đo bằng vla-bench), VRAM đỉnh đạt 2039 MiB.
  - Cấu hình tốt nhất em tìm được sau khâu sàng lọc là q8v_fa_mmdef_s5 (Q8_0 (có vision) + flash-attn + mm-prec default + 5 bước denoise), nhanh hơn so với baseline đo khi chạy LIBERO (168.5 → 99.5 ms) mà vẫn giữ nguyên SR (37/40 so với 35/40).
  - Vẫn đang tiến hành đo chính thức cho bản gốc và 3 cấu hình tốt nhất.

### Insight rút ra

- Quantization làm giảm bộ nhớ, tuy nhiên tốc độ thì vẫn phụ thuộc vào kernel và phần cứng, nếu kernel vẫn dequantize lại về FP32 khi chạy thì latency sẽ không đổi như kết quả đo của em. Em thử tìm hiểu thì hiện tượng này gọi là fake-quant, cần chú ý cho những lần sau khi áp dụng quantization.
- Việc chọn quantize đến mức nào còn phụ thuộc vào bài toán, như ví dụ với MNIST là một bài toán dễ, giảm từ 32 bit xuống 3 bit vẫn giữ được accuracy cao, điểm gãy chỉ xuất hiện khi đã giảm xuống 2 bit. Với các bài toán phức tạp hơn, không nên quantize quá mức vì làm tròn ở các bài toán phức tạp có thể tạo ra sai số lớn hơn đáng kể khi được cộng dồn lại so với bài toán đơn giản như MNIST, bằng chứng là với SmolVLA trên LIBERO, giảm từ BF16 xuống Q8 thì SR giữ nguyên, nhưng xuống Q4 thì gãy (SR từ 35/40 giảm xuống 16/40).
- Trong Knowledge Distillation, ở thí nghiệm thay đổi α, với α = 1 - nghĩa là student chỉ học từ teacher mà bỏ qua label gốc, student mắc ít lỗi hơn chỉ học từ nhãn - α = 0, nên em kết luận student chỉ học từ teacher học ngang hoặc thậm chí tốt hơn student học từ nhãn, lí do là vì câu trả lời của teacher đã chứa thông tin của nhãn, ngoài ra còn đi kèm thêm mức độ giống nhau giữa các lớp (dark knowledge). Điều này giống như student được teacher cho biết đáp án đúng và lí do tại sao, thay vì chỉ biết được đáp án đúng.
- Model student càng nhỏ có xu hướng hưởng lợi càng nhiều từ teacher, bằng chứng là với student 14k tham số, KD giúp giảm 15 lỗi, student 3.8k tham số thì KD giúp giảm 31 lỗi. Em nghĩ điều này có thể hiểu theo một cách trực giác là student càng yếu thì càng cần teacher dạy, trong khi student đã giỏi thì ít cần hơn.
- Khi Pruning, unstructured cut không loại bỏ hẳn w ra khỏi mạng mà thay w bị loại bằng 0, đây chính là lí do khiến size và latency không đổi vì tensor vẫn lưu cùng số lượng w và phép nhân thì vẫn nhân với 0. Ngoài ra, sau khi structured cut thì việc fine-tune lại model mới là rất đáng làm vì nó làm tăng đáng kể accuracy.
- Ở bài kernel optimization, việc tối ưu không đúng chỗ không mang lại hiệu quả tổng thể: kỹ thuật Grouped ordering - xếp tile theo nhóm giúp đọc bộ nhớ hiệu quả hơn, nhưng vấn đề của kernel là không dùng tensor core trên GPU T4 nên việc tính toán khá chậm, dữ liệu được đọc nhanh hơn từ bộ nhớ vẫn phải "xếp hàng chờ" trước khi được mang vào tính toán. Do vậy, bài học em rút ra là phải tìm đúng chỗ nghẽn trước khi tối ưu.
- Ở phần SmolVLA, nhìn output trên một input thì Q4_0 + vision không có vấn đề gì khi độ lệch chỉ khoảng gấp đôi bản 5 bước denoise, nhưng khi cho robot làm thật thì SR của bản 5 bước là 35/40 trong khi bản Q4_0 + vision sập xuống 16/40. Bài học em rút ra là độ lệch output chỉ dùng để lọc sơ bộ và kiểm tra nhanh, còn để đánh giá phương pháp tối ưu có làm hỏng robot hay không thì vẫn phải dựa vào SR.
- Khi so sánh cấu hình Q8_0 có và không quantize vision, khi chạy với flag mặc định thì vision chỉ nhanh hơn khoảng 4 ms, còn chạy với `--flash-attn --mm-prec default` thì vision chạy nhanh hơn khoảng 11 ms nên em đã vội kết luận là sự cải thiện có được là nhờ vào flag flash-attn hoặc mm-prec. Đây là một sai lầm khi đo do đã thay đổi 2 yếu tố cùng lúc và em sau đó đã thử đo lại đủ các kiểu tổ hợp, thêm biến fused-ops và có được kết luận là `--mm-prec` không ảnh hưởng tới vision, và khoảng chênh lệch 11 ms bị thu hẹp còn 4 ms chỉ khi flash-attn tắt và fused-ops bật. Flash-attn cũng không phải nguyên nhân vì tắt fused ops thì dù flash-attn vẫn tắt, Q8 vẫn nhanh hơn khoảng 11 ms nên giả thuyết của em về nguyên nhân là fused-ops tạo ra một lợi thế riêng cho w dạng BF16. Bài học rút ra là khi đo, mỗi lần chỉ đổi một biến và phải để ý các biến ẩn đang bật mặc định.

## 2. Completed Work

| Đầu việc | Trạng thái | Chi tiết |
|---|---|---|
| Bài 1: Quantization | Xong (notebook + report) | 2.1.1 |
| Bài 2: Knowledge distillation | Xong (notebook + report) | 2.1.2 |
| Bài 3: Pruning | Xong (notebook + report) | 2.1.3 |
| Bài 4: NAS | Xong (notebook + report) | 2.1.4 |
| Bài 5: Kernel optimization | Xong (notebook + report) | 2.1.5 |
| vla.cpp: cài đặt và chạy thử | Xong | 2.2 |
| vla.cpp: baseline | Xong | 2.3 |
| vla.cpp: runtime flag | Xong | 2.4 |
| vla.cpp: quantize | Xong | 2.5 |
| vla.cpp: đo SR trên LIBERO | Smoke test và sàng lọc xong; vòng chính thức đang chạy | 2.6 |
| vla.cpp: profile (timing theo phase, nsys) | Chưa làm | Plan |
| vla.cpp: pruning | Chưa làm | Plan |

### 2.1. Các bài examples

Chi tiết từng bài: [01](https://github.com/Paim0n3223/modelopt101/blob/main/reports/01_quantization.md), [02](https://github.com/Paim0n3223/modelopt101/blob/main/reports/02_knowledge_distillation.md), [03](https://github.com/Paim0n3223/modelopt101/blob/main/reports/03_pruning.md), [04](https://github.com/Paim0n3223/modelopt101/blob/main/reports/04_nas.md), [05](https://github.com/Paim0n3223/modelopt101/blob/main/reports/05_kernel_optimization.md). Bài 1–3 và 5 chạy trên Colab T4, bài 4 chạy trên laptop (latency đo trên CPU), nên latency giữa các bài không so trực tiếp được.

#### 2.1.1. Quantization (MNIST, CNN kiểu VGG, 8 bit symmetric per-channel)

**Bài này làm gì**

- Notebook: `examples/01_quantization.ipynb`, chạy trên Google Colab (GPU T4).
- Dữ liệu MNIST, model là CNN kiểu VGG (6 lớp conv + 1 Linear, khoảng 1.1M tham số), train FP32 12 epoch.
- Mục tiêu: lưu weight của Conv và Linear bằng INT8 thay cho FP32 (post-training quantization, không train lại).
- Đo size, accuracy, latency trước và sau.

**Phần em viết (3 TODO)**

- TODO 1: tính scale cho symmetric (zero_point = 0).
- TODO 2: tính scale và zero_point cho affine.
- TODO 3: quantize: chia cho scale, làm tròn, cộng zero_point, kẹp vào [qmin, qmax].

**Thí nghiệm thêm**

- Symmetric và affine (8 bit, per-channel)
- Giảm số bit

**Kết quả**

*Bảng 1. Kết quả chính bài quantization: FP32 so với INT8 (8 bit, symmetric, per-channel). Colab T4, latency đo trên CPU.*

| | Size | Accuracy | CPU latency |
|---|---|---|---|
| FP32 | 4.61 MB | 99.67% | 5.42 ms |
| INT8 | 1.18 MB | 99.67% | 5.54 ms |

*Bảng 2. Thí nghiệm giảm số bit: accuracy theo số bit, per-channel và per-tensor.*

| Bit | 8 | 4 | 3 | 2 |
|---|--:|--:|--:|--:|
| Acc per-channel | 99.67% | 99.67% | 99.53% | 20.23% |
| Acc per-tensor | 99.67% | 99.67% | – | 9.80% |

- Quantize per-channel cho kết quả tốt hơn per-tensor ở 2 bit dù cả 2 đều sập, nguyên nhân là vì sử dụng scale riêng cho từng filter/channel giúp chúng giữ lại thông tin tốt hơn là dùng 1 scale cho cả tensor.

#### 2.1.2. Knowledge distillation (MNIST, teacher 1.15M tham số, student 14.5k)

**Bài này làm gì**

- Notebook: `examples/02_knowledge_distillation.ipynb`, chạy trên Google Colab (GPU T4), dữ liệu MNIST.
- Train 3 model, mỗi model 15 epoch:
  - Teacher: CNN 64 → 128 → 256 kênh, 1,147,722 tham số.
  - Student không KD: CNN 16 → 32 → 32 kênh, 14,458 tham số, train bằng nhãn như bình thường.
  - Student có KD: cùng kiến trúc, chỉ khác loss (học thêm từ teacher).
- Mục tiêu: student có KD chính xác hơn student không KD, mà vẫn cùng kích thước và tốc độ.

**Phần em viết (3 TODO)**

- TODO 1: kiến trúc teacher (64 → 128 → 256 kênh).
- TODO 2: KD term: KL divergence giữa output của student và teacher đã làm mềm bằng T, nhân T².
- TODO 3: ghép loss: α · KD term + (1 − α) · CE term.

**Thí nghiệm thêm**

- A. Temperature (α = 0.7)
- B. Alpha (T = 4)
- C. Student nhỏ hơn (T = 4)

**Kết quả**

*Bảng 3. Kết quả chính bài KD (T = 4, α = 0.7): teacher, student không KD, student có KD. Colab T4.*

| Model | Tham số | Size | Accuracy | Số lỗi / 10,000 |
|---|---|---|---|---|
| Teacher | 1,147,722 | 4.61 MB | 99.68% | 32 |
| Student không KD | 14,458 | 0.07 MB | 99.24% | 76 |
| Student có KD | 14,458 | 0.07 MB | 99.38% | 62 |

- Student nhỏ hơn teacher 79.4 lần, nhanh hơn 8.6 lần (CPU latency 0.64 ms).

Bảng 4–6 chạy lại với cùng seed (`torch.manual_seed(0)`). Nhiễu giữa các lần chạy khoảng ±0.1% (±10 lỗi): cùng T = 4, α = 0.7, lần chạy đầu được 62 lỗi (Bảng 3), chạy lại với seed 0 được 58 lỗi (Bảng 5).

*Bảng 4. Thí nghiệm A: thay đổi temperature T (α = 0.7, cùng seed).*

| T (α = 0.7) | 1 | 4 | 8 | 20 |
|---|--:|--:|--:|--:|
| Số lỗi | 62 | 58 | 62 | 67 |

*Bảng 5. Thí nghiệm B: thay đổi α (T = 4, cùng seed).*

| α (T = 4, cùng seed) | 0.0 (chỉ nhãn) | 0.3 | 0.7 | 1.0 (chỉ teacher) |
|---|--:|--:|--:|--:|
| Số lỗi | 73 | 58 | 58 | 64 |

*Bảng 6. Thí nghiệm C: KD với student nhỏ hơn (T = 4).*

| Student | Tham số | Không KD | Có KD (α = 0.7) | KD giảm được |
|---|---|---|---|---|
| Student | 14,458 | 73 lỗi | 58 lỗi | 15 lỗi (~21%) |
| TinyStudent | 3,778 | 131 lỗi (98.69%) | 100 lỗi (99.00%) | 31 lỗi (~24%) |

- Xu hướng chung của T và α là đạt mức ít lỗi nhất tại T = 4 và α = 0.3–0.7, tuy nhiên các chênh lệch này (tối đa 9 lỗi với T, 6 lỗi so với α = 1.0) vẫn nằm trong vùng nhiễu ±10 lỗi nên mới chỉ là xu hướng. Riêng α = 0 (73 lỗi) kém rõ rệt.

#### 2.1.3. Pruning (MNIST, CNN kiểu VGG, cắt 50%)

**Bài này làm gì**

- Notebook: `examples/03_pruning.ipynb`, chạy trên Google Colab (GPU T4), dữ liệu MNIST.
- Model: CNN kiểu VGG (6 lớp conv 3×3 chia 3 stage, kênh 64 → 128 → 256, khoảng 1.1M tham số), train 12 epoch.
- Cắt 50% (`PRUNE_RATIO = 0.5`) theo hai cách rồi so sánh size, accuracy, latency:
  - Unstructured: đặt 50% weight có |w| nhỏ nhất về 0, không fine-tune.
  - Structured: bỏ 50% filter có L1 nhỏ nhất ở mỗi lớp, dựng lại model hẹp hơn, fine-tune 4 epoch.

**Phần em viết (3 TODO)**

- TODO 1: tính ngưỡng cắt theo quantile của |w| (unstructured).
- TODO 2: đặt các weight nhỏ hơn ngưỡng về 0.
- TODO 3: chọn filter giữ lại theo L1 (structured).

**Thí nghiệm thêm**

- A. Tiêu chí chọn filter, ngay sau khi cắt (không fine-tune)
- B. Tiêu chí chọn filter, sau fine-tune 1 epoch (ratio 0.3)
- C. Prune rồi quantize

**Kết quả**

*Bảng 7. Kết quả chính bài pruning (cắt 50%): dense, unstructured, structured. Colab T4, latency đo trên CPU.*

| Model | Size | Accuracy | CPU latency |
|---|---|---|---|
| Dense | 4.61 MB | 99.68% | 7.94 ms |
| Unstructured (masked) | 4.61 MB | 99.59% | 7.97 ms |
| Structured (rebuilt + fine-tune) | 1.17 MB | 99.68% | 3.48 ms |

- Unstructured cắt 571,680 / 1,143,360 weight (50.0%), accuracy sau khi cắt vẫn cao mà không cần fine-tune.
- Structured: tham số 1,147,722 → 288,170 (4.0x); accuracy ngay sau khi cắt, chưa fine-tune chỉ có 9.74%.

*Bảng 8. Accuracy của unstructured theo mức sparsity (không fine-tune).*

| Sparsity (unstructured) | 0% | 50% | 70% | 80% | 90% | 95% |
|---|---|---|---|---|---|---|
| Accuracy | 99.68% | 99.59% | 98.90% | 87.65% | 26.95% | 10.10% |

- Accuracy phải tới khi cắt từ 80% trở lên mới giảm mạnh, em cho rằng nguyên nhân đến từ độ khó thấp của bài toán MNIST, với bài toán phức tạp hơn thì em đoán là mức giảm mạnh sẽ đến sớm hơn.

*Bảng 9. Thí nghiệm A: tiêu chí chọn filter của structured, accuracy ngay sau khi cắt.*

| Tiêu chí chọn filter, chưa fine-tune | Ratio 0.1 | 0.2 | 0.3 | 0.5 |
|---|--:|--:|--:|--:|
| L1 | 84.04% | 26.87% | 9.74% | 9.74% |
| BN gamma | 89.81% | 51.21% | 24.49% | 9.80% |
| Ngẫu nhiên (TB 3 seed) | 97.01% | 60.83% | 25.30% | 10.35% |

- Bước này đã đi ngược dự đoán của em khi em cho rằng cắt L1 sẽ cho kết quả tốt nhất, nhưng thực tế số liệu cho thấy cắt ngẫu nhiên lại cho kết quả tốt hơn L1 ở mọi mức tỉ lệ thí nghiệm.

*Bảng 10. Thí nghiệm B: tiêu chí chọn filter, trước và sau fine-tune 1 epoch (ratio 0.3).*

| Tiêu chí chọn filter (ratio 0.3) | Ngay sau khi cắt | Sau fine-tune 1 epoch | Số lỗi |
|---|---|---|---|
| L1 | 9.74% | 99.58% | 42 |
| BN gamma | 24.49% | 99.46% | 54 |
| Ngẫu nhiên | 25.30% (TB 3 seed) | 99.40% | 60 |

- Sau khi fine-tune thì câu chuyện đã trở nên đúng như dự đoán khi L1 có accuracy cao nhất. Em cho rằng không nên đánh giá accuracy ngay sau khi cắt mà phải fine-tune rồi mới đánh giá.

*Bảng 11. Thí nghiệm C: quantize model dense và model đã cắt (structured L1, ratio 0.5, fine-tune 4 epoch).*

| Prune rồi quantize (fake-quant per-channel) | FP32 | 8 bit | 4 bit | 3 bit | 2 bit |
|---|--:|--:|--:|--:|--:|
| Dense (1,147,722 tham số) | 99.68% | 99.68% | 99.68% | 99.61% | 25.40% |
| Pruned structured (288,170 tham số) | 99.68% | 99.67% | 99.68% | 99.51% | 24.06% |

- Quantize model sau cắt vẫn cho độ chính xác gần như tương đương ở tất cả các mức bit nên em cho rằng có thể kết hợp 2 phương pháp optimization này với nhau.

#### 2.1.4. NAS (MNIST, không gian 30 kiến trúc, random + evolutionary search, chạy trên laptop)

**Bài này làm gì**

- Notebook: `examples/04_nas.ipynb`, dữ liệu MNIST.
- Chạy trên laptop (GPU RTX 3050 Ti 4 GB, PyTorch 2.11.0+cu128, VS Code), vì Colab đã hết hạn mức GPU miễn phí. Latency vẫn đo trên CPU (của laptop, không phải Colab), nên không so trực tiếp với latency của 3 bài trước.
- Thay vì thu nhỏ một model có sẵn (bài 1–3), NAS định nghĩa một không gian kiến trúc rồi tìm kiến trúc cân bằng tốt giữa accuracy và chi phí.
- Mục tiêu: kiến trúc chính xác nhất (theo proxy) mà không to hơn baseline thiết kế tay `d3-w32-k3`, rồi train đầy đủ cả hai để so sánh.

**Phần em viết (4 TODO)**

- TODO 1: dựng model từ gen (depth, width, kernel).
- TODO 2: proxy: train ứng viên trước khi chấm điểm.
- TODO 3: đột biến: đổi 1 gen sang giá trị khác.
- TODO 4: chọn 2 cha mẹ có proxy accuracy cao nhất.

**Thí nghiệm thêm**

- Proxy có xếp hạng đúng không? (train đầy đủ 5 kiến trúc, cùng seed 0)

**Kết quả**

*Bảng 12. Kết quả chính bài NAS: baseline và kiến trúc NAS chọn, train đầy đủ (10 epoch, 60,000 ảnh). Laptop, latency đo trên CPU.*

| Model | Size | Accuracy | Số lỗi | CPU latency |
|---|---|---|---|---|
| Baseline d3-w32-k3 | 0.38 MB | 99.37% | 63 | 0.76 ms |
| NAS chọn d4-w12-k3 | 0.23 MB | 99.46% | 54 | 0.45 ms |

- Ngân sách tìm kiếm tối đa mà bài cho là 16 lượt, thực tế chỉ thử 12 kiến trúc, số còn lại bị loại do trùng lặp (8 random + 4 evolved) trong 30. Pareto front gồm 5 kiến trúc sau: d2-w8-k5, d3-w8-k3, d4-w12-k3, d4-w16-k3, d4-w12-k5.
- Train thử proxy (2 epoch, 10k ảnh) cho kết quả về khoảng accuracy theo depth như sau: depth d2 có accuracy trong khoảng 74–76%, d3 là 93–97.5%, d4 là 95.5–98.3%. Dựa trên kết quả này, em cho rằng depth sẽ là yếu tố quan trọng nhất của kiến trúc trong bài toán này.

*Bảng 13. Thí nghiệm thêm: train đầy đủ 5 kiến trúc (cùng seed 0) để kiểm tra xếp hạng của proxy.*

| Kiến trúc | Vì sao chọn | Proxy | Train đầy đủ | Số lỗi | Size | CPU latency |
|---|---|---|---|---|---|---|
| d3-w32-k3 | Baseline | 95.08% | 99.40% | 60 | 0.38 MB | 0.47 ms |
| d4-w12-k3 | NAS chọn | 95.56% | 99.49% | 51 | 0.23 MB | 0.49 ms |
| d3-w16-k5 | Suýt thắng | 95.53% | 99.49% | 51 | 0.27 MB | 0.64 ms |
| d4-w12-k5 | Proxy cao nhất | 98.26% | 99.54% | 46 | 0.62 MB | 0.90 ms |
| d4-w32-k5 | To nhất | 96.58% | 99.57% | 43 | 2.68 MB | 1.05 ms |

- Trả lời câu hỏi của thí nghiệm: proxy xếp hạng gần đúng, chỉ 2 vị trí đầu đổi chỗ cho nhau.
- Khi đã train đủ thì accuracy của các kiến trúc gần như ngang nhau.
- Model to nhất sau khi train thì ít lỗi nhất, nhưng so với kiến trúc NAS chọn thì to hơn 11.7 lần và chậm hơn 2.1 lần, chỉ để giảm 8 lỗi, mà 8 lỗi vẫn nằm trong vùng nhiễu. Dù vậy, em có một phỏng đoán rằng model lớn sẽ cần nhiều epoch hơn để phát huy sức mạnh thật sự.

#### 2.1.5. Kernel optimization (Triton matmul FP16, Colab T4, Triton 3.6.0)

**Bài này làm gì**

- Notebook: `examples/05_kernel_optimization.ipynb`, chạy trên Google Colab (Tesla T4, Triton 3.6.0, compute capability 7.5, đời Turing).
- Bài yêu cầu viết một kernel nhân ma trận C = A × B bằng Triton (FP16), rồi so với `torch.matmul` (gọi thư viện cuBLAS của NVIDIA) ở các cỡ 512, 1024, 2048, 4096.
- Bài này khác bốn bài trước: không đổi model, không đổi phép toán, chỉ đổi cách GPU thực thi. Kết quả phải giống hệt, chỉ khác tốc độ.
- Các chỉ số được đo: đúng sai so với `torch.matmul`, latency (ms) và throughput (TFLOP/s = 2MNK / thời gian).

**Phần em viết (3 TODO)**

- TODO 1: xếp tile theo nhóm (grouped ordering).
- TODO 2: tiling: cộng dồn từng mảnh A × B vào tile của C.
- TODO 3: mask khi ghi kết quả cho ma trận không chia hết cho tile.
- Thứ tự làm: TODO 2 → TODO 3 → đo → TODO 1 → đo lại, để tách riêng tác dụng của TODO 1.

**Thí nghiệm thêm**

- A. Vì sao chậm: Triton có dùng tensor core không?
- B. Ma trận không vuông
- C. Kích thước tile (cỡ 2048)

**Kết quả**

- Placeholder (chưa làm TODO): kết quả toàn số 0 (`match: False`) nhưng benchmark báo 880 TFLOP/s ở cỡ 4096, gấp hơn 13 lần đỉnh lý thuyết của T4.
- Sau khi làm: max abs difference 0.0625, khớp `torch.matmul`; đúng cả với ma trận không vuông (500×300×700, 512×4096×1024, 1000³, 77×33×129).

*Bảng 14. Latency (ms) kernel Triton sau TODO 2–3 (xếp tile theo hàng) và sau TODO 1 (xếp tile theo nhóm). Colab T4, FP16.*

| Size | 512 | 1024 | 2048 | 4096 |
|---|--:|--:|--:|--:|
| Theo hàng | 0.301 | 1.828 | 16.027 | 142.010 |
| Theo nhóm | 0.301 | 1.861 | 15.796 | 148.901 |

*Bảng 15. Kernel Triton bản cuối so với `torch.matmul` (cuBLAS). Colab T4, FP16.*

| Size | torch ms | triton ms | torch TF/s | triton TF/s | speedup |
|---|---|---|---|---|---|
| 512 | 0.034 | 0.301 | 7.9 | 0.9 | 0.11x |
| 1024 | 0.080 | 1.861 | 26.8 | 1.2 | 0.04x |
| 2048 | 0.922 | 15.796 | 18.6 | 1.1 | 0.06x |
| 4096 | 9.044 | 148.901 | 15.2 | 0.9 | 0.06x |

*Bảng 16. Thí nghiệm C: kích thước tile, ma trận 2048 × 2048.*

| Tile (cỡ 2048, BLOCK_K 32 nếu không ghi) | ms | TFLOP/s | Thanh ghi | Spill |
|---|---|---|---|---|
| 32×32 | 11.140 | 1.54 | 165 | 0 |
| 64×64 (mặc định) | 15.787 | 1.09 | 255 | 92 |
| 64×64, BLOCK_K 64 | 86.850 | 0.20 | 64 | 586 |
| 128×64 | 187.182 | 0.09 | 64 | 958 |
| 128×128 | 316.520 | 0.05 | 64 | 2,772 |

- Kết quả cho thấy tile càng to càng chậm, ngược với trực giác ban đầu nên em đã tiến hành kiểm tra và phát hiện tile càng to thì tình trạng tràn thanh ghi diễn ra càng nghiêm trọng, dữ liệu tràn bị đẩy ra bộ nhớ bên ngoài và làm chậm quá trình chạy.

### 2.2. vla.cpp: cài đặt và chạy thử

**Yêu cầu:** Cài đặt và chạy thử. Build vla.cpp có CUDA qua WSL2 theo docs/backend/wsl.md, tự chọn CUDA architecture cho RTX 3060 Laptop và bật build test. Sau đó chạy thử test, CLI, server, Python binding và vla_predict_check để hiểu mỗi công cụ dùng làm gì.

**Đã làm**

- Cài công cụ build qua apt và CUDA toolkit 13.1 từ repo `wsl-ubuntu` của NVIDIA (không dùng 12.6 như wsl.md vì Ubuntu 26.04 có gcc 15).
- Vá header CUDA để nvcc chạy được với glibc 2.43.
- Clone vla.cpp, configure với CUDA architecture 86 và `VLA_BUILD_TESTS=ON`, build `-j4`.
- Chạy `ctest`, tìm nguyên nhân 2 test fail.
- Tải model bằng `vla-cli -hf` (cài thêm `hf` CLI); chạy CLI với `--tokens`, rồi với `--text` (tạo venv riêng có `transformers` cho tokenizer).
- Kiểm tra tính tất định: chạy `vla-cli` và `vla_predict_check` 2 lần với cùng input; so `vla_predict_check` khi bật và tắt fused ops.
- Chạy `vla-server`, gửi request qua một client ZMQ nhỏ với noise cố định.
- Chạy Python binding qua `VLA_LIBRARY=build/libvla.so`, so output với server.

**Kết quả**

*Bảng 17. Môi trường cài đặt vla.cpp.*

| Thành phần | Giá trị |
|---|---|
| Máy | Ryzen 7 5800H, RTX 3050 Ti Laptop 4 GB (sm_86), driver 591.44 |
| WSL2 | Ubuntu 26.04.1, gcc 15.2.0, glibc 2.43, cmake 4.2.3 |
| CUDA | 13.1 (wsl.md dùng 12.6, không hợp với gcc 15) |
| vla.cpp | commit `61f20f1`, build `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DVLA_BUILD_TESTS=ON` |
| Model | `vrfai/smolvla-libero-gguf`, `smolvla-libero.gguf` (BF16, 1.126 GB) |

*Bảng 18. Kết quả cài đặt và chạy thử từng công cụ.*

| Việc | Kết quả |
|---|---|
| Build | Lỗi nvcc `rsqrt` (glibc 2.43 khai báo `noexcept`, CUDA 13.1 thì không) → vá 2 dòng header `crt/math_functions.h`. Build `-j4` khoảng 16 phút, 0 lỗi |
| ctest | 13/15 pass. `foldquant_cuda_op`, `fused_cuda_ops` fail: test so bit-exact, lệch khoảng 1 ULP |
| vla-cli | Cần cài `hf` (để `-hf` tải model) và `transformers` (để `--text`). Noise lấy ngẫu nhiên nên 2 lần chạy cho output khác nhau |
| vla-server | ZMQ REQ/REP, request nhận noise từ ngoài → cố định noise thì tất định. Cùng noise, khác câu lệnh: action lệch tới 2.01. 1 view: request đầu 1001 ms, ổn định khoảng 112 ms |
| Python binding | Dùng `VLA_LIBRARY=build/libvla.so`, output khớp server |
| vla_predict_check | Tất định. Fused ops bật/tắt lệch max 1.06e-3, mean 2.09e-4 (mean \|action\| = 0.187) |

### 2.3. vla.cpp: baseline

**Yêu cầu:** Baseline. Đo latency (p50/p90) và VRAM đỉnh bằng vla-bench theo protocol và cấu hình input cho SmolVLA trong docs/benchmark/ (tham khảo rtx-3060.md). Ghi rõ điều kiện đo vì laptop dễ bị throttle.

**Đã làm**

- Viết script `apps/vla.cpp/scripts/run_baseline.sh`: chạy `vla-bench` theo protocol của `docs/benchmark/TEMPLATE.md` (3 warmup + 20 lần × 3 process), lấy mẫu `nvidia-smi` mỗi 0.5 s, ghi lại môi trường đo.
- Chạy baseline với flag mặc định, 2 view 512 px.
- So với số liệu RTX 3060 trong `docs/benchmark/rtx-3060.md`.

**Kết quả**

Điều kiện: `vla-bench --images 2 --size 512 --warmup 3 --reps 20`, flag mặc định, 3 process (báo process có mean thấp nhất), cắm sạc, Best performance, GPU 49 °C lúc bắt đầu. VRAM lấy `memory.used` của toàn GPU (trong WSL `nvidia-smi` không báo được VRAM theo process).

*Bảng 19. Baseline SmolVLA đo bằng `vla-bench`, flag mặc định, so với RTX 3060 desktop.*

| | p50 ms | p90 ms | vision ms | VRAM đỉnh MiB | Host RSS MiB |
|---|--:|--:|--:|--:|--:|
| RTX 3050 Ti Laptop (máy này) | 163.9 | 164.3 | 91.1 | 2039 | 772 |
| RTX 3060 desktop (`docs/benchmark/rtx-3060.md`) | 125.3 | 128.0 | 71.6 | 2038 | 721 |

### 2.4. vla.cpp: tối ưu không cần train, runtime flag

**Yêu cầu:** Tối ưu không cần train. Thử các runtime flag trong docs/USAGE.md và docs/backend/cuda.md (attention, dtype, số bước denoise, …). Mỗi cấu hình ghi kích thước file, VRAM, latency và độ lệch output so với bản gốc.

**Đã làm**

- Viết script `run_flags.sh` (mỗi cấu hình: `vla-bench` 1 process × 5 lần, `vla_predict_check` để đo độ lệch, nghỉ 60 s) và `analyze_flags.py` (lập bảng).
- Sàng lọc 18 cấu hình: `--flash-attn`, `--mm-prec`, `--weight-dtype`, `--act-dtype`, `--num-steps` và các biến môi trường `VLA_FUSED_OPS`, `VLA_GEMM_EXACT`, `VLA_ATTN_EXACT`, `GGML_CUDA_DISABLE_GRAPHS`.
- Đo đầy đủ (3 process × 20 lần) các bộ flag nhanh nhất.

**Kết quả**

Cùng điều kiện mục 2.3. Độ lệch = mean \|lệch\| của action so với mặc định, đo bằng `vla_predict_check` (1 input tổng hợp).

*Bảng 20. Runtime flag, đo đầy đủ bằng `vla-bench` (3 process × 20 lần) cho các bộ flag nhanh nhất.*

| Cấu hình | p50 ms | vision ms | So với mặc định | VRAM MiB | Độ lệch |
|---|--:|--:|--:|--:|--:|
| Mặc định | 163.5 | 90.6 | – | 2039 | 0 |
| `--flash-attn` | 143.4 | 70.5 | −12.1% | 1971 | 3.7e-4 |
| `--mm-prec default` | 153.3 | 91.0 | −6.2% | 1269 | 1.5e-4 |
| `--flash-attn --mm-prec default` | 132.2 | 70.1 | −19.1% | 1201 | 3.2e-4 |
| + `--num-steps 5` | 111.0 | 70.0 | −32.1% | 1173 | 1.1e-2 |

*Bảng 21. Runtime flag, sàng lọc bằng `vla-bench` (1 process × 5 lần) cho tất cả flag.*

| Cấu hình | p50 ms | vision ms | So với mặc định | VRAM MiB | Độ lệch | Kẹp đổi ở bước |
|---|--:|--:|--:|--:|--:|---|
| Mặc định | 163.3 | 91.1 | – | 2039 | 0 | 7 |
| `--flash-attn` | 143.3 | 71.0 | −12.1% | 1971 | 3.7e-4 | 7 |
| `--mm-prec default` | 152.9 | 91.2 | −6.4% | 1269 | 1.5e-4 | 7 |
| `--flash-attn --mm-prec default` | 132.8 | 70.9 | −18.8% | 1201 | 3.2e-4 | 7 |
| `--weight-dtype bf16` | 197.3 | 91.5 | +21.0% | 1281 | 0 | 7 |
| `--weight-dtype bf16 --flash-attn` | 177.0 | 70.9 | +8.5% | 1211 | 3.7e-4 | 7 |
| `--weight-dtype f16` | 200.2 | 94.0 | +22.5% | 1285 | 1.8e-4 | 7 |
| `--weight-dtype f16 --flash-attn` | 172.2 | 66.2 | +5.4% | 1209 | 2.0e-4 | 7 |
| `--act-dtype bf16` | – | – | – | – | – | không hỗ trợ cho SmolVLA |
| `--num-steps 8` | 154.4 | 92.0 | −5.4% | 2029 | 4.2e-3 | 7 |
| `--num-steps 5` | 139.0 | 91.2 | −14.8% | 2015 | 1.1e-2 | 7 |
| `--num-steps 3` | 129.1 | 91.2 | −20.8% | 2007 | 3.0e-2 | 9 |
| `--num-steps 2` | 124.2 | 91.1 | −23.9% | 2005 | 5.9e-2 | 10 |
| `--num-steps 1` | 119.2 | 91.0 | −27.0% | 1989 | 1.4e-1 | đóng/mở liên tục |
| `VLA_FUSED_OPS=0` | 183.1 | 110.7 | +12.1% | 2013 | 2.1e-4 | 7 |
| `VLA_GEMM_EXACT=0` | 163.7 | 91.5 | +0.4% | 2033 | 0 | 7 |
| `VLA_ATTN_EXACT=0` | 162.6 | 90.3 | −0.5% | 2045 | 1.8e-4 | 7 |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 185.6 | 92.1 | +13.4% | 1987 | 0 | 7 |

### 2.5. vla.cpp: tối ưu không cần train, quantize

**Yêu cầu:** Quantize theo docs/MODELS.md ở nhiều mức, có và không quantize vision tower. Mỗi cấu hình ghi kích thước file, VRAM, latency và độ lệch output so với bản gốc.

**Đã làm**

- Tạo 10 file GGUF bằng script có sẵn của vla.cpp (`scripts/quantize_gguf.py`): Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, mỗi mức có và không `--vision`.
- Sàng lọc từng file với flag mặc định và với fa + mmdef (dùng lại `run_flags.sh`); đo đầy đủ 3 file tốt nhất.
- Thí nghiệm tách biến: 2×2×2 (weight × flash-attn × mm-prec), rồi tắt fused ops.

**Kết quả**

Quantize bằng `scripts/quantize_gguf.py` (ggml block-32). Không nén embedding, norm, action expert; vision chỉ nén khi thêm `--vision`. Cùng điều kiện đo mục 2.3; fa + mmdef = `--flash-attn --mm-prec default`. Độ lệch tính so với BF16 flag mặc định, trên input tổng hợp, với fa + mmdef.

*Bảng 22. Các file quantize, sàng lọc bằng `vla-bench` (1 process × 5 lần) với flag mặc định và với fa + mmdef.*

| File | Kích thước | p50 ms mặc định | p50 ms fa + mmdef | vision ms fa + mmdef | VRAM MiB mặc định | VRAM MiB fa + mmdef | Độ lệch | Kẹp đổi ở bước |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| BF16 (gốc) | 1.126 GB | 163.6 | 131.9 | 70.5 | 2039 | 1201 | 3.2e-4 | 7 |
| Q8_0 | 0.820 GB | 156.5 | 129.2 | 70.3 | 1149 | 911 | 1.4e-3 | 7 |
| Q8_0 + vision | 0.741 GB | 152.6 | 117.9 | 59.2 | 1079 | 835 | 3.6e-3 | 7 |
| Q5_1 | 0.718 GB | 159.1 | 131.8 | 70.7 | 1053 | 813 | 7.4e-3 | 7 |
| Q5_1 + vision | 0.612 GB | 157.2 | 121.9 | 60.8 | 957 | 713 | 2.5e-2 | 8 |
| Q5_0 | 0.698 GB | 157.6 | 130.4 | 70.5 | 1033 | 795 | 2.7e-2 | 10 |
| Q5_0 + vision | 0.587 GB | 155.2 | 120.1 | 60.2 | 933 | 689 | 2.1e-2 | 9 |
| Q4_1 | 0.678 GB | 158.0 | 130.5 | 70.2 | 1013 | 775 | 1.7e-2 | 7 |
| Q4_1 + vision | 0.561 GB | 155.8 | 120.6 | 60.6 | 907 | 663 | 2.9e-2 | 8 |
| Q4_0 | 0.657 GB | 157.4 | 129.5 | 70.4 | 993 | 755 | 1.6e-2 | 8 |
| Q4_0 + vision | 0.535 GB | 152.9 | 117.4 | 58.8 | 883 | 639 | 2.4e-2 | 7 |

Đo đầy đủ (3 process × 20 lần), fa + mmdef: BF16 p50 132.2 ms; Q8_0 129.5 ms; Q8_0 + vision 118.4 ms (−27.5% so với mặc định); Q4_0 + vision 118.4 ms.

### 2.6. vla.cpp: đo SR trên LIBERO

**Yêu cầu:** Đo SR trên LIBERO. Cài theo mục "End-to-end evaluation (LIBERO client on WSL)" trong docs/backend/wsl.md. Ba mức: smoke test 1 task × 2 episode (mỗi lần đổi GGUF); sàng lọc 10 task × 4 = 40 episode (cấu hình flag, quantize, pruning); chính thức 10 task × 20 = 200 episode (bản gốc và tối đa 3 cấu hình tốt nhất).

**Đã làm**

- Cài LIBERO bằng `eval/sim/libero/setup_libero.sh` (tăng timeout, số lần thử lại và giảm số file tải song song của uv vì mạng chập chờn).
- Kiểm tra render EGL; tìm cách render bằng GPU NVIDIA qua d3d12.
- Smoke test bản gốc (task 0 × 2 episode), so render bằng CPU và bằng GPU.
- Viết script `run_libero_sr.sh` (chạy lần lượt nhiều cấu hình, tự bật/tắt server, chạy tiếp được khi bị ngắt) và `analyze_sr.py` (SR, khoảng tin cậy 95%, latency phía server).
- Sàng lọc 6 cấu hình × 40 episode; chọn 3 cấu hình cùng bản gốc cho vòng chính thức và bắt đầu chạy (200 episode mỗi cấu hình).

**Kết quả**

*Bảng 23. Cài LIBERO, render và smoke test.*

| Việc | Kết quả |
|---|---|
| Cài đặt | `eval/sim/libero/setup_libero.sh`, venv khoảng 10 GB; mạng chậm, phải chạy 3 lần |
| Render EGL | Không có `/dev/dri` → mặc định render bằng CPU (llvmpipe). `GALLIUM_DRIVER=d3d12` chọn GPU AMD → segfault; thêm `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` → chạy được |
| Smoke test (bản gốc, libero_object task 0, 2 episode) | SR 2/2. Mỗi bước: render CPU 0.41 s, render GPU 0.29 s; suy luận 187 ms ở cả hai |
| Script | `apps/vla.cpp/scripts/run_libero_sr.sh`, `analyze_sr.py` |

Sàng lọc: libero_object, 10 task × 4 episode, seed 42, giới hạn 500 bước, render GPU. Latency = median phía server khi chạy LIBERO (không phải `vla-bench`).

*Bảng 24. Sàng lọc SR trên LIBERO (libero_object, 10 task × 4 episode); latency lấy từ log server trong lúc chạy LIBERO.*

| Cấu hình | SR | 95% CI | Server total / vision / inf ms |
|---|--:|---|---|
| BF16 (gốc) | 35/40 (87.5%) | 74–95% | 168.5 / 92.8 / 74.7 |
| BF16, fa + mmdef | 32/40 (80.0%) | 65–90% | 137.2 / 72.0 / 64.3 |
| BF16, fa + mmdef + 5 bước | 35/40 (87.5%) | 74–95% | 113.3 / 71.5 / 40.9 |
| Q8_0 + vision, fa + mmdef | 35/40 (87.5%) | 74–95% | 121.8 / 60.1 / 60.8 |
| Q8_0 + vision, fa + mmdef + 5 bước | 37/40 (92.5%) | 80–97% | 99.5 / 60.1 / 38.6 |
| Q4_0 + vision, fa + mmdef | 16/40 (40.0%) | 26–55% | 123.2 / 60.1 / 62.2 |

- Task 5 khó nhất với mọi cấu hình (10/24 lần thành công).
- Vòng chính thức (10 task × 20 episode) vẫn đang chạy: bản gốc, Q8_0 + vision fa + mmdef + 5 bước, Q8_0 + vision fa + mmdef, BF16 fa + mmdef + 5 bước.

## 3. Plan

### Action items tuần tới

Hoàn thành nốt các phần còn lại của task vla.cpp:

- [ ] Đo SR chính thức (10 task × 20 episode): chạy nốt Q8_0 + vision, fa + mmdef (task 5–9) và BF16, fa + mmdef + 5 bước; tổng hợp cùng bản gốc và Q8_0 + vision, fa + mmdef + 5 bước (đã xong).
- [ ] Profile: mô tả kiến trúc SmolVLA (vision, connector, VLM, action expert, vòng denoise) từ paper và `src/models/smolvla.cpp`; đo latency từng thành phần bằng timing theo phase; dùng nsys cho bản gốc và bản tốt nhất, ghi lại hạn chế của nsys trên WSL2.
- [ ] Pruning: tìm hiểu các phương pháp pruning phù hợp với SmolVLA (tham khảo CLP), chọn và implement một vài phương pháp; đo SR (sàng lọc) và kiểm tra latency từng module bằng vla-bench và nsys.
- [ ] Viết report cho phần vla.cpp.

### Đề xuất

- Em dự tính cần 2–3 ngày để hoàn thành task của modelopt101 vì trên máy em chạy khá lâu (lượt chạy chính thức dự tính tốn 10 tiếng). Sau đó em xin nhận task mới từ các anh.
