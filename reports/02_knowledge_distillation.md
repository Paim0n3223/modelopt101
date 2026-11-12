# Bài 2: Knowledge Distillation

## 1. Bài này làm gì

- Notebook: `examples/02_knowledge_distillation.ipynb`, chạy trên Google Colab (GPU T4), dữ liệu MNIST.
- Train 3 model, mỗi model 15 epoch:
  - Teacher: CNN 64 → 128 → 256 kênh, 1,147,722 tham số.
  - Student không KD: CNN 16 → 32 → 32 kênh, 14,458 tham số, train bằng nhãn như bình thường.
  - Student có KD: cùng kiến trúc, chỉ khác loss (học thêm từ teacher).
- Mục tiêu: student có KD chính xác hơn student không KD, mà vẫn cùng kích thước và tốc độ.

## 2. Ý tưởng em hiểu được

- Theo em, model bình thường chỉ học "đáp án là 7". Teacher cho thêm một số thông tin hữu ích, ví dụ như "giống 7 nhất, hơi giống 1, chẳng giống 0" (dark knowledge).
- Vì teacher và student khác kiến trúc nên không copy weight được. Kiến thức truyền qua hành vi: student tập cho ra output giống teacher với cùng input.
- Teacher đã train xong và bị đóng băng (eval mode, không tính gradient).
- Loss = α · KD_term + (1 − α) · CE_term. CE so student với nhãn, KD so student với teacher (dùng KL divergence).
- Gradient descent kéo student về hai hướng cùng lúc, theo tỉ lệ α. Hai hướng phần lớn đồng thuận về lớp đúng, chỉ khác ở các lớp sai và mức độ tự tin.
- Temperature T: chia logits cho T trước softmax để làm mềm phân phối. Softmax chỉ phụ thuộc chênh lệch giữa các logits, chia cho T là thu hẹp chênh lệch.
- Ví dụ logits [8, 4, 1]: T = 1 cho [0.981, 0.018, 0.001], T = 4 cho [0.649, 0.239, 0.113], T = 20 cho [0.396, 0.324, 0.279].
- T quá nhỏ thì dark knowledge bị che, T quá lớn thì bị pha loãng (tiến tới phân phối đều).
- Nhân T² để gradient của KD_term cân bằng với CE_term (chia cho T làm gradient nhỏ đi khoảng T² lần).
- T chỉ dùng lúc train, khi chạy thật student dùng softmax bình thường.

## 3. Phần em viết (3 TODO)

- TODO 1, kiến trúc teacher: `c1, c2, c3 = 64, 128, 256`.
- TODO 2, KD term:
  ```python
  s = F.log_softmax(student_logits / T, dim=1)
  t = F.softmax(teacher_logits / T, dim=1)
  kd = F.kl_div(s, t, reduction="batchmean")
  kd_term = kd * T**2
  ```
- `F.kl_div` đòi tham số đầu là log-xác suất (student), tham số thứ hai là xác suất thường (teacher).
- TODO 3, ghép hai phần: `alpha * kd_term + (1 - alpha) * ce_term`.

## 4. Những chỗ em vấp

- Khác bài 1: TODO 1 quyết định kiến trúc teacher và nằm trước bước train. Nếu "Chạy tất cả" ngay với placeholder (16, 16, 16) thì sẽ train một teacher còn nhỏ hơn student. Phải sửa TODO 1 trước rồi mới chạy.
- Placeholder của TODO 3 trả về `ce_term`, nghĩa là student "có KD" thực ra giống hệt student không KD.
- TODO 2 viết một dòng thì khó, tách thành 4 dòng nhỏ thì dễ hơn.
- Viết TODO 3 bằng cách gán lại vào biến `ce_term`. Chạy đúng nhưng tên biến dễ gây hiểu nhầm (biến tên CE nhưng chứa tổng hai phần).

## 5. Kết quả (T = 4, α = 0.7, như notebook)

| Model | Tham số | Size | Accuracy | Số lỗi / 10,000 |
|---|---|---|---|---|
| Teacher | 1,147,722 | 4.61 MB | 99.68% | 32 |
| Student không KD | 14,458 | 0.07 MB | 99.24% | 76 |
| Student có KD | 14,458 | 0.07 MB | 99.38% | 62 |

- Student nhỏ hơn teacher 79.4 lần, nhanh hơn 8.6 lần (CPU latency 0.64 ms).
- Accuracy chỉ tăng 0.14% nhưng số lỗi giảm 76 → 62 (khoảng 18%). Ở mức gần 100%, nhìn số lỗi dễ thấy hơn nhìn accuracy.
- MNIST quá dễ nên student tự học đã gần chạm trần, KD còn ít chỗ để giúp.

## 6. Thí nghiệm thêm

Cách làm: mỗi lần chạy đặt lại cùng seed (`torch.manual_seed(0)`) để so sánh công bằng. Nhiễu giữa các lần chạy khoảng ±0.1% (±10 lỗi). Ví dụ: cùng T = 4, α = 0.7, lần đầu được 62 lỗi, lần chạy lại với seed 0 được 58 lỗi. Các mục A, B, C chỉ chạy một seed; mục D chạy lại các so sánh chính với 5 seed.

### A. Temperature (α = 0.7)

| T | Accuracy | Số lỗi |
|---|---|---|
| 1 | 99.38% | 62 |
| 4 | 99.42% | 58 |
| 8 | 99.38% | 62 |
| 20 | 99.33% | 67 |

- T = 4 tốt nhất, T = 20 kém nhất, đúng dự đoán "ở giữa là tốt nhất".
- Nhưng chênh lệch tốt nhất và kém nhất chỉ 9 lỗi, nằm trong vùng nhiễu. Chỉ là xu hướng, cần chạy nhiều seed để khẳng định.

### B. Alpha (T = 4)

| α | Học từ | Accuracy | Số lỗi |
|---|---|---|---|
| 0.0 | Chỉ nhãn | 99.27% | 73 |
| 0.3 | | 99.42% | 58 |
| 0.7 | | 99.42% | 58 |
| 1.0 | Chỉ teacher | 99.36% | 64 |

- α = 0.0 chính là student không KD chạy cùng seed. So với nó, KD giảm 73 xuống 58 lỗi (khoảng 20%). Chênh lệch 15 lỗi lớn hơn mức nhiễu nhưng không nhiều, và mới chạy một seed.
- Kết hợp với thí nghiệm A: T = 1 (62 lỗi) ít lỗi hơn không KD (73 lỗi). Chênh 11 lỗi chỉ vừa sát mức nhiễu, nên em chưa kết luận được KD có giúp khi không làm mềm hay không.
- α = 1.0: student không nhìn thấy nhãn mà vẫn đạt 64 lỗi, so với 73 lỗi của student học từ nhãn. Chênh 9 lỗi nằm trong vùng nhiễu, nên em chỉ nói được là học hoàn toàn từ teacher không kém học từ nhãn, chưa nói được là tốt hơn.
- Kết hợp cả hai (α = 0.3 hoặc 0.7) cho số lỗi thấp nhất, nhưng chênh lệch với α = 1.0 (6 lỗi) cũng nằm trong vùng nhiễu.
- Khi chạy lại với 5 seed (mục D), α = 1 ít lỗi hơn cả α = 0 lẫn α = 0.7 ở mọi seed, nên thứ tự ở bảng này không lặp lại.

### C. Student nhỏ hơn (T = 4)

TinyStudent: 8 → 16 → 16 kênh, 3,778 tham số (nhỏ hơn Student khoảng 3.8 lần, nhỏ hơn teacher khoảng 300 lần).

| Student | Tham số | Không KD | Có KD (α = 0.7) | KD giảm được |
|---|---|---|---|---|
| Student | 14,458 | 73 lỗi | 58 lỗi | 15 lỗi (~21%) |
| TinyStudent | 3,778 | 131 lỗi (98.69%) | 100 lỗi (99.00%) | 31 lỗi (~24%) |

- Tính theo số lỗi tuyệt đối thì KD giúp TinyStudent nhiều hơn (31 so với 15 lỗi). Nhưng TinyStudent có nhiều lỗi hơn ngay từ đầu, nên tính theo tỉ lệ thì hai student giảm gần như bằng nhau (khoảng 21% và 24%), chênh lệch này nằm trong vùng nhiễu.
- Vì vậy em chưa kết luận được "student càng yếu thì KD giúp càng nhiều". Chạy lại với 5 seed (mục D) cũng không ủng hộ nhận định này.
- KD không bù được hết cho model quá nhỏ: TinyStudent có KD (100 lỗi) vẫn kém Student không KD (73 lỗi).

### D. Chạy lại nhiều seed

Theo feedback của mentor, em chạy lại các so sánh chính của B và C với 5 seed (0 đến 4), bằng script `examples/02_kd_seeds.py`. Kết quả từng lượt nằm ở `examples/02_kd_seeds.csv`.
- Giữ nguyên kiến trúc, loss, optimizer, cách xử lý dữ liệu và 15 epoch như notebook, T = 4.
- Lần này chạy trên laptop (RTX 3050 Ti, PyTorch 2.11) chứ không phải Colab, nên số lỗi không ghép trực tiếp với các bảng A, B, C. Teacher được train lại một lần với seed 0, đạt 33 lỗi.
- Cùng một seed thì mọi cấu hình có cùng khởi tạo và cùng thứ tự dữ liệu, nên em so theo từng cặp seed bằng kiểm định t theo cặp.

| Cấu hình | Số lỗi trung bình | Độ lệch chuẩn | Thấp nhất – cao nhất |
|---|--:|--:|---|
| Student, không KD (α = 0) | 71.2 | 4.9 | 64–76 |
| Student, α = 0.7 | 62.2 | 2.6 | 59–65 |
| Student, α = 1 | 59.2 | 2.5 | 57–63 |
| TinyStudent, không KD | 117.2 | 11.9 | 104–136 |
| TinyStudent, α = 0.7 | 104.4 | 9.4 | 93–113 |

| So sánh theo cặp seed | Ít lỗi hơn trung bình | Số seed ít lỗi hơn | p |
|---|--:|--:|--:|
| Student: α = 0.7 so với không KD | 9.0 lỗi (khoảng 12%) | 4/5 | 0.03 |
| Student: α = 1 so với α = 0 | 12.0 lỗi | 5/5 | 0.007 |
| Student: α = 1 so với α = 0.7 | 3.0 lỗi | 5/5 | 0.01 |
| TinyStudent: α = 0.7 so với không KD | 12.8 lỗi (khoảng 10%) | 3/5 | 0.24 |

- KD vẫn giúp Student, nhưng ít hơn lần chạy đơn: trung bình khoảng 12%, dao động từ −2% đến 21% giữa các seed. Con số khoảng 20% ở mục 5 và bảng B rơi vào nhóm cao nhất.
- α = 1 ít lỗi hơn α = 0 ở cả 5 seed, trung bình 12 lỗi. Với 5 seed, em có thể nói trong thiết lập này, học hoàn toàn từ teacher tốt hơn học từ nhãn.
- α = 1 cũng ít lỗi hơn α = 0.7 ở cả 5 seed, dù chỉ khoảng 3 lỗi. Thứ tự này ngược với bảng B (α = 0.7 được 58 lỗi, α = 1 được 64 lỗi), nên kết luận "kết hợp nhãn và teacher là tốt nhất" không đứng vững.
- TinyStudent dao động mạnh hơn hẳn Student (độ lệch chuẩn khoảng 12 lỗi so với khoảng 5 lỗi), và có 2/5 seed KD còn làm TinyStudent kém đi. Em chưa khẳng định được KD có giúp TinyStudent hay không.
- Tính theo tỉ lệ, KD giảm lỗi của hai student gần như bằng nhau (12.2% và 9.7%, p = 0.80). Vì vậy không có cơ sở cho nhận định "student càng nhỏ càng hưởng lợi từ KD".
- Mức nhiễu giữa các seed phụ thuộc vào model, không phải một con số ±10 lỗi chung như em ước lượng ban đầu.
- Giới hạn: mới có 5 seed, teacher chỉ train với một seed, và có nhiều phép so sánh cùng lúc nên p chỉ để tham khảo.

### Code để chạy lại các thí nghiệm

Mục D: `python examples/02_kd_seeds.py` (từ thư mục gốc của repo), xem lại bảng tổng hợp bằng `python examples/02_kd_seeds.py --summary`.

Mục A, B, C:

Các ô này chỉ chạy trong Colab, không commit vào notebook. Chạy sau khi đã chạy hết notebook (cần `teacher`, `train`, `evaluate`, `distillation_loss`).

```python
def run_kd(T, alpha, model_class=Student, seed=0):
    torch.manual_seed(seed)
    student = model_class()

    def loss_fn(out, x, y):
        with torch.no_grad():
            teacher_out = teacher(x)
        return distillation_loss(out, teacher_out, y, T=T, alpha=alpha)

    student = train(student, loss_fn=loss_fn, tag=f"T={T}, alpha={alpha}")
    return evaluate(student)

# A: run_kd(T, 0.7) với T = 1, 4, 8, 20
# B: run_kd(4, alpha) với alpha = 0.0, 0.3, 1.0
# C: run_kd(4, alpha, model_class=TinyStudent) với alpha = 0.0, 0.7
```

TinyStudent giống Student, chỉ đổi số kênh thành 8, 16, 16 và lớp Linear thành `nn.Linear(16, num_classes)`.

## 7. Điều em rút ra

- Từ kết quả, em thấy có KD hay không thì khác biệt rõ ràng hơn so với việc chọn T. Với T, các chênh lệch ở mục A nhỏ và chỉ chạy một seed nên có thể là nhiễu. Với α, ban đầu em nghĩ giá trị ở giữa là tốt nhất, nhưng chạy 5 seed thì giá trị cực đoan α = 1 lại tốt nhất.
- KD không tốn thêm gì lúc chạy, cái giá chỉ nằm ở lúc train (cần một teacher).
- Với 5 seed, KD giúp Student giảm khoảng 12% lỗi, và học hoàn toàn từ teacher (α = 1) cho kết quả tốt nhất ở mọi seed. Với TinyStudent, KD chưa cho thấy tác dụng rõ ràng vì kết quả dao động quá mạnh giữa các seed. KD cũng không thay được sức chứa của model: TinyStudent có KD vẫn kém Student không KD.
- Một lần chạy có thể sai cả về mức độ (khoảng 20% so với khoảng 12%) lẫn về thứ tự (α = 0.7 so với α = 1). Khi chênh lệch chỉ vài chục lỗi, phải chạy nhiều seed và so theo cặp rồi mới kết luận.
- Em đang nghĩ về ý tưởng kết hợp với bài 1: distill ra student nhỏ rồi quantize student đó, việc này sẽ giúp tối ưu bộ nhớ trên disk đến mức cực đoan.
- Cũng khá tương tự với bài quantization, bài toán MNIST vẫn là quá dễ để có thể nhìn thấy được sự khác biệt khi chỉnh T và α.
