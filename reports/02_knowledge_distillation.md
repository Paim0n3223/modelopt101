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

Cách làm: mỗi lần chạy đặt lại cùng seed (`torch.manual_seed(0)`) để so sánh công bằng. Nhiễu giữa các lần chạy khoảng ±0.1% (±10 lỗi). Ví dụ: cùng T = 4, α = 0.7, lần đầu được 62 lỗi, lần chạy lại với seed 0 được 58 lỗi.

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

- α = 0.0 chính là student không KD chạy cùng seed. So với nó, KD giảm 73 → 58 lỗi (khoảng 20%), vượt mức nhiễu.
- Kết hợp với thí nghiệm A: T = 1 (62 lỗi) vẫn tốt hơn không KD (73 lỗi), nên KD giúp được ngay cả khi không làm mềm.
- α = 1.0: student không hề nhìn thấy nhãn mà vẫn tốt hơn student học từ nhãn (64 so với 73), vì output của teacher đã chứa gần đủ thông tin của nhãn cộng thêm dark knowledge.
- Kết hợp cả hai (α = 0.3 hoặc 0.7) có xu hướng tốt nhất, nhãn giúp sửa những chỗ teacher sai. Chênh lệch với α = 1.0 (6 lỗi) nằm trong vùng nhiễu.

### C. Student nhỏ hơn (T = 4)

TinyStudent: 8 → 16 → 16 kênh, 3,778 tham số (nhỏ hơn Student khoảng 3.8 lần, nhỏ hơn teacher khoảng 300 lần).

| Student | Tham số | Không KD | Có KD (α = 0.7) | KD giảm được |
|---|---|---|---|---|
| Student | 14,458 | 73 lỗi | 58 lỗi | 15 lỗi (~21%) |
| TinyStudent | 3,778 | 131 lỗi (98.69%) | 100 lỗi (99.00%) | 31 lỗi (~24%) |

- Student càng yếu, KD giúp càng nhiều (tính theo số lỗi). Kết quả này vượt xa mức nhiễu.
- Nhưng KD không bù được hết cho model quá nhỏ: TinyStudent có KD (100 lỗi) vẫn kém Student không KD (73 lỗi).

### Code để chạy lại các thí nghiệm

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

- Từ kết quả, em thấy có KD hay không thì khác biệt rõ ràng, trong khi đó chọn T và α trong khoảng hợp lý dù có xu hướng cải thiện kết quả nhưng khác biệt nhỏ, có thể là nhiễu và khoảng hợp lí thường là nằm giữa trong các thí nghiệm đã thực hiện, ít khi nằm ở các giá trị cực đoan.
- KD không tốn thêm gì lúc chạy, cái giá chỉ nằm ở lúc train (cần một teacher).
- KD hữu ích nhất khi buộc phải dùng model rất nhỏ, nhưng không thay được sức chứa của model.
- Em đang nghĩ về ý tưởng kết hợp với bài 1: distill ra student nhỏ rồi quantize student đó, việc này sẽ giúp tối ưu bộ nhớ trên disk đến mức cực đoan.
- Cũng khá tương tự với bài quantization, bài toán MNIST vẫn là quá dễ để có thể nhìn thấy được sự khác biệt khi chỉnh T và α.
