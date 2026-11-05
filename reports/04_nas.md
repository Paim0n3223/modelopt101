# Bài 4: Neural Architecture Search (NAS)

## 1. Bài này làm gì

- Notebook: `examples/04_nas.ipynb`, dữ liệu MNIST.
- Chạy trên laptop (GPU RTX 3050 Ti 4 GB, PyTorch 2.11.0+cu128, VS Code), vì Colab đã hết hạn mức GPU miễn phí. Latency vẫn đo trên CPU (của laptop, không phải Colab), nên không so trực tiếp với latency của 3 bài trước.
- Thay vì thu nhỏ một model có sẵn (bài 1–3), NAS định nghĩa một không gian kiến trúc rồi tìm kiến trúc cân bằng tốt giữa accuracy và chi phí.
- Mục tiêu: kiến trúc chính xác nhất (theo proxy) mà không to hơn baseline thiết kế tay `d3-w32-k3`, rồi train đầy đủ cả hai để so sánh.

## 2. Ý tưởng em hiểu được

- Ba thành phần cốt lõi của bài toán NAS: search space - tập hợp architecture khả thi, search strategy - chiến lược tìm kiếm, trong bài đề xuất 2 cách là random search và evolutionary search, cách chấm điểm - đánh giá architecture dựa trên ít epochs và ít dữ liệu.
- Search space: 3 gen, tổng 3 × 5 × 2 = 30 kiến trúc.
  - depth (2, 3, 4): số stage, mỗi stage là conv → BatchNorm → ReLU → MaxPool. Ảnh 28 → 14 → 7 → 3 → 1.
  - width (8, 12, 16, 24, 32): số kênh stage đầu, gấp đôi mỗi stage, tối đa 128.
  - kernel (3, 5): kích thước ô conv. Kernel 5 nhìn vùng rộng hơn nhưng tốn gần gấp 3 lần weight và phép tính.
- Proxy: train mỗi ứng viên 2 epoch trên 10,000 ảnh (thay vì 10 epoch trên 60,000 ảnh), đủ để xếp hạng thô. Chi phí trong lúc tìm kiếm đo bằng size thay cho latency.
- Random search: bốc ngẫu nhiên 8 kiến trúc.
- Evolutionary search: lấy 2 kiến trúc tốt nhất làm cha mẹ, đột biến (đổi 1 gen) để tạo con, 2 thế hệ × 4 con. Dành công tìm kiếm quanh vùng đã thấy tốt.
- Pareto front: các kiến trúc không bị kiến trúc nào khác vừa chính xác hơn vừa nhỏ hơn.

## 3. Phần em viết (4 TODO)

- TODO 1, `build_model`: lấy số stage, số kênh, kernel từ gen.
  ```python
  for i in range(arch["depth"]):
      cout = min(arch["width"] * (2 ** i), 128)
      k = arch["kernel"]
  ```
- TODO 2, proxy: train trước khi chấm điểm: `train(model, search_loader, epochs=SEARCH_EPOCHS)`. `train` sửa trực tiếp model nên không bắt buộc gán lại.
- TODO 3, đột biến:
  ```python
  gene = random.choice(list(CHOICES))
  child[gene] = random.choice([v for v in CHOICES[gene] if v != arch[gene]])
  ```
  `dict(arch)` tạo bản sao nên sửa con không làm hỏng cha.
- TODO 4, chọn cha mẹ: `parents = sorted(results, key=lambda r: r["acc"], reverse=True)[:2]`.

## 4. Những chỗ em vấp

- Placeholder lần này làm mọi thứ vô nghĩa: `build_model` bỏ qua gen (mọi ứng viên giống nhau), proxy không train (mọi ứng viên ~10%), đột biến trả về con giống cha. Nên phải làm hết TODO rồi mới chạy.
- Colab báo hết hạn mức GPU, phải chuyển sang chạy local
- Cài môi trường trên laptop:
  - `nvidia-smi` không có trong PATH, phải gọi `C:\Windows\System32\nvidia-smi.exe`. Driver 591.44 hỗ trợ CUDA tối đa 13.1.
  - Python mặc định là conda base 3.13, nên tạo môi trường conda riêng với Python 3.12.
  - Lần đầu chạy nguyên văn `cuXXX` trong lệnh cài PyTorch (đó là chỗ để điền), phải đổi thành `cu128`.

## 5. Kết quả

### Tìm kiếm (proxy)

| Kiến trúc | Proxy acc | Size | Nguồn |
|---|---|---|---|
| d3-w32-k5 | 97.54% | 1.04 MB | random |
| d3-w16-k5 | 95.53% | 0.27 MB | random |
| d3-w32-k3 | 95.08% | 0.38 MB | random |
| d4-w12-k5 | 98.26% | 0.62 MB | random |
| d2-w8-k5 | 74.12% | 0.02 MB | random |
| d4-w32-k3 | 98.09% | 0.98 MB | random |
| d3-w8-k3 | 93.50% | 0.03 MB | random |
| d4-w16-k5 | 97.62% | 1.09 MB | random |
| d4-w32-k5 | 96.58% | 2.68 MB | evolved (thế hệ 1) |
| d4-w16-k3 | 97.83% | 0.40 MB | evolved (thế hệ 1) |
| d4-w12-k3 | 95.56% | 0.23 MB | evolved (thế hệ 1) |
| d2-w12-k5 | 76.00% | 0.04 MB | evolved (thế hệ 2) |

- Pareto front: d2-w8-k5, d3-w8-k3, d4-w12-k3, d4-w16-k3, d4-w12-k5.
- Depth quan trọng nhất trong không gian này: d2 chỉ 74–76%, d3 từ 93–97.5%, d4 từ 95.5–98.3%.
- To hơn không có nghĩa là tốt hơn ở proxy: d4-w32-k5 (2.68 MB) chỉ 96.58%, thua d4-w12-k5 (0.62 MB, 98.26%).
- Tiến hóa tạo ra kiến trúc thắng cuộc (d4-w12-k3) và d4-w16-k3, cả hai thuộc Pareto front. Nhưng thế hệ 1 chỉ ra 3 con mới, thế hệ 2 chỉ 1 con (các con còn lại trùng kiến trúc đã thử, bị bỏ qua). Thế hệ 2 dùng lại cùng cha mẹ vì chưa con nào vượt được. Không gian chỉ 30 kiến trúc, đã thử 12.
- Trong ngân sách size (≤ 0.38 MB), d4-w12-k3 (95.56%) thắng d3-w16-k5 (95.53%) chỉ 0.03%.

### Baseline và kiến trúc NAS chọn (train đầy đủ)

| Model | Size | Accuracy | Số lỗi | CPU latency |
|---|---|---|---|---|
| Baseline d3-w32-k3 | 0.38 MB | 99.37% | 63 | 0.76 ms |
| NAS chọn d4-w12-k3 | 0.23 MB | 99.46% | 54 | 0.45 ms |

- Notebook in ra: nhỏ hơn 1.7x, nhanh hơn 1.7x, +0.09% accuracy.
- Chênh 9 lỗi nằm trong vùng nhiễu (khoảng ±10 lỗi), nên chỉ nói được là chính xác ngang baseline.
- Con số "nhanh hơn 1.7x" không lặp lại được (xem thí nghiệm bên dưới).

## 6. Thí nghiệm thêm: proxy có xếp hạng đúng không?

Train đầy đủ (10 epoch, 60,000 ảnh, cùng seed 0) 5 kiến trúc đáng chú ý.

| Kiến trúc | Vì sao chọn | Proxy | Train đầy đủ | Số lỗi | Size | CPU latency |
|---|---|---|---|---|---|---|
| d3-w32-k3 | Baseline | 95.08% | 99.40% | 60 | 0.38 MB | 0.47 ms |
| d4-w12-k3 | NAS chọn | 95.56% | 99.49% | 51 | 0.23 MB | 0.49 ms |
| d3-w16-k5 | Suýt thắng | 95.53% | 99.49% | 51 | 0.27 MB | 0.64 ms |
| d4-w12-k5 | Proxy cao nhất | 98.26% | 99.54% | 46 | 0.62 MB | 0.90 ms |
| d4-w32-k5 | To nhất | 96.58% | 99.57% | 43 | 2.68 MB | 1.05 ms |

- Proxy xếp hạng gần đúng: chỉ hai vị trí đầu đổi chỗ, phần còn lại giữ thứ tự. Proxy thiên vị nhẹ chống lại model lớn (cần nhiều epoch hơn mới phát huy).
- Train đầy đủ thì mọi kiến trúc gần nhau: proxy trải hơn 3%, train đầy đủ chỉ chênh 17 lỗi. Các vị trí liền nhau chỉ hơn kém 3–5 lỗi, trong vùng nhiễu.
- Cặp suýt thắng hòa nhau hoàn toàn (51 lỗi). Khi accuracy ngang nhau thì xét chi phí: d4-w12-k3 nhỏ hơn và nhanh hơn, nên lựa chọn của NAS vẫn đúng (phần nào nhờ may).
- Model to nhất có accuracy cao nhất, nhưng so với kiến trúc NAS chọn thì to hơn 11.7 lần, chậm hơn 2.1 lần để đổi lấy 8 lỗi ít hơn (trong vùng nhiễu). "Tốt nhất" phụ thuộc vào ngân sách.
- Latency của baseline và kiến trúc NAS chọn lần này gần bằng nhau (0.47 so với 0.49 ms), không lặp lại được "nhanh hơn 1.7x" của lần chạy chính. Dù d4-w12-k3 ít phép nhân-cộng hơn khoảng 5 lần (khoảng 1.5 triệu so với 7.4 triệu mỗi ảnh), model quá nhỏ chạy từng ảnh trên CPU bị chi phí cố định mỗi lớp chiếm phần lớn, mà d4 lại có nhiều lớp hơn. Điều lặp lại được: kernel 5 luôn chậm hơn rõ rệt.
- Baseline train lại cùng seed ra 99.40% (lần chạy chính 99.37%), khớp nhau.

### Code để chạy lại

Chạy sau khi đã chạy hết notebook (cần `results`, `build_model`, `arch_str`, `train`, `train_loader`, `evaluate`, `model_size_mb`, `latency_ms`).

```python
proxy = {arch_str(r["arch"]): r["acc"] for r in results}

def parse(s):
    d, w, k = s.split("-")
    return {"depth": int(d[1:]), "width": int(w[1:]), "kernel": int(k[1:])}

cands = ["d3-w32-k3", "d4-w12-k3", "d3-w16-k5", "d4-w12-k5", "d4-w32-k5"]
rows = []
for s in cands:
    torch.manual_seed(0)
    m = train(build_model(parse(s)), train_loader, epochs=EPOCHS)
    acc = evaluate(m)
    rows.append((s, proxy.get(s), acc, model_size_mb(m), latency_ms(m)))

for s, p, a, sz, lat in rows:
    print(s, p, a, round((100 - a) * 100), sz, lat)
```

Thứ tự kiến trúc trong random search phụ thuộc seed, chạy lại notebook có thể tìm ra kiến trúc khác.

## 7. Điều em rút ra

- NAS giúp tìm ra kiến trúc nhỏ hơn 1.65x với accuracy ngang baseline, từ đó mang đến lợi thế về size, tuy nhiên lợi thế về latency thì không ổn định.
- Sử dụng proxy là đủ để loại kiến trúc tệ và xếp hạng thô, nhưng không phân định được các kiến trúc sát nhau, và thiên vị nhẹ chống model lớn.
- Notebook gợi ý NAS kết hợp được với các bài trước theo thứ tự tìm kiến trúc, rồi distill, prune, quantize, em sẽ thử sau.
