# Bài 3: Pruning

## 1. Bài này làm gì

- Notebook: `examples/03_pruning.ipynb`, chạy trên Google Colab (GPU T4), dữ liệu MNIST.
- Model: CNN kiểu VGG (6 lớp conv 3×3 chia 3 stage, kênh 64 → 128 → 256, khoảng 1.1M tham số), train 12 epoch.
- Cắt 50% (`PRUNE_RATIO = 0.5`) theo hai cách rồi so sánh size, accuracy, latency:
  - Unstructured: đặt 50% weight có |w| nhỏ nhất về 0, không fine-tune.
  - Structured: bỏ 50% filter có L1 nhỏ nhất ở mỗi lớp, dựng lại model hẹp hơn, fine-tune 4 epoch.

## 2. Ý tưởng em hiểu được

- Pruning gồm 3 quyết định trước khi cắt: criterion (chấm điểm cái gì quan trọng), granularity (cắt theo đơn vị nào) và ratio (cắt bao nhiêu). Sau khi prune, accuracy có thể rất tệ (structured còn khoảng 10%) nên phải thêm bước fine-tune để hồi phục.
- Mạng node và ma trận weight là một: mỗi đường nối là một ô. Hàng = mọi đường đi vào một node lớp sau, cột = mọi đường đi ra từ một node lớp trước.
- Với conv: mỗi ô là một kernel 3×3, weight có shape [out, in, 3, 3]. Một hàng (mọi kernel đi vào một node) là một filter, tức một bộ dò hoa văn hoàn chỉnh, tạo ra một feature map.
- Cắt một filter = xóa một hàng ở lớp này và một cột ở lớp sau (cùng các kênh tương ứng trong BatchNorm). Vì vậy không xóa trực tiếp trong model cũ được (shape không khớp), mà phải dựng `Net(widths=...)` mới hẹp hơn rồi chép phần giữ lại.
- Cắt 50% filter mỗi lớp thì weight chỉ còn khoảng 25% (mất cả hàng lẫn cột).
- Unstructured: chấm điểm từng weight bằng |w|, ngưỡng là quantile ở mức `ratio` của |w| toàn model (global). Ví dụ 10 số, ratio 0.5: vị trí 0.5 × 9 = 4.5, nội suy giữa số thứ 4 và 5.
- Structured: chấm điểm mỗi filter bằng L1 (tổng |w| của cả hàng), giữ top-k ở mỗi lớp (per-layer).
- Unstructured giữ shape, số 0 vẫn lưu 4 byte và vẫn được nhân, nên không nhỏ hơn, không nhanh hơn (trừ khi có lưu trữ và kernel sparse). Structured tạo model đặc nhỏ hơn thật.

## 3. Phần em viết (3 TODO)

- TODO 1, ngưỡng: `thresh = torch.quantile(all_w, ratio)`.
- TODO 2, đặt weight nhỏ về 0: `w.mul_((w.abs() > thresh).float())`.
- TODO 3, chọn filter:
  ```python
  l1 = blk[0].weight.data.abs().sum(dim=(1, 2, 3))
  idx = torch.topk(l1, k).indices.sort().values
  ```
- `.sum(dim=(1, 2, 3))` cộng theo chiều in và 3×3, ra một điểm cho mỗi filter.
- `.sort().values` giữ thứ tự gốc của filter. Không bắt buộc về đúng sai, vì cùng một `idx` dùng cho hàng lớp này và cột lớp sau.

## 4. Những chỗ em vấp

- Placeholder lại "đẹp giả": mục 7 in sparsity 50%... 95% mà accuracy không đổi, thực ra `threshold = 0` và điều kiện `|w| >= 0` luôn đúng, nên 0 weight bị cắt. Các con số 50%, 70%... chỉ là tỉ lệ được yêu cầu, không phải sparsity thật.
- Lúc đầu thấy mask khó hiểu: `w.abs() > thresh` là danh sách giữ lại (True = nhân 1 = giữ), weight dưới ngưỡng nhân 0.
- `.mul_` sửa trực tiếp tensor. Nếu viết `w = w * mask` thì chỉ tạo tensor mới, weight trong model không đổi, code vẫn chạy không báo lỗi.
- Thêm `torch.abs` thừa ở TODO 1 (`all_w` đã là |w|). Dùng `>=` thay vì `>` ở TODO 2.
- Bị ngắt kết nối Colab giữa chừng, mất hết biến, phải chạy lại từ đầu (model dense train lại ra 99.68% thay vì 99.71%).
- Dễ so sai hai con số "50%": unstructured 50% = bỏ 50% weight, structured 50% = bỏ 50% filter, tương đương bỏ khoảng 75% weight.

## 5. Kết quả (ratio 0.5)

| Model | Size | Accuracy | CPU latency |
|---|---|---|---|
| Dense | 4.61 MB | 99.68% | 7.94 ms |
| Unstructured (masked) | 4.61 MB | 99.59% | 7.97 ms |
| Structured (rebuilt + fine-tune) | 1.17 MB | 99.68% | 3.48 ms |

- Unstructured: threshold = 0.0064, cắt đúng 571,680 / 1,143,360 weight (50.0%). Một nửa số weight có |w| < 0.0064. Accuracy giảm nhẹ, size và latency không đổi.
- Structured: widths (64, 64, 128, 128, 256, 256) → (32, 32, 64, 64, 128, 128), tham số 1,147,722 → 288,170 (4.0x). Nhỏ hơn 3.9x, nhanh hơn 2.0x, accuracy giữ nguyên sau fine-tune.
- Accuracy structured ngay sau khi cắt (chưa fine-tune): 9.74%. Đây không phải đoán bừa mà là model luôn trả lời "8" (MNIST test có đúng 974 ảnh số 8). Tương tự 11.35% = luôn trả lời "1", 10.10% = luôn trả lời "3".
- Lần chạy trước với placeholder TODO 3 (giữ k filter đầu, gần như ngẫu nhiên): dense 99.71%, ngay sau khi cắt 11.35%, sau fine-tune 99.68%, latency 5.42 → 2.42 ms. Latency tuyệt đối thay đổi giữa các phiên Colab, tỉ lệ nhanh hơn 2.0–2.4x thì nhất quán.

### Unstructured: tăng tỉ lệ cắt (không fine-tune)

| Sparsity | 0% | 50% | 70% | 80% | 90% | 95% |
|---|---|---|---|---|---|---|
| Accuracy | 99.68% | 99.59% | 98.90% | 87.65% | 26.95% | 10.10% |

- Giữ nguyên rồi sập, điểm gãy giữa 70% và 90%. Cùng hình dạng với thí nghiệm giảm bit ở bài 1.
- Cùng mức bỏ khoảng 75% weight, chưa fine-tune: unstructured còn khoảng 88–99%, structured chỉ 9.74%. Unstructured chọn chính xác từng weight nhỏ nên chịu cắt tốt, structured bỏ cả hàng (kể cả weight tốt) nhưng cho model nhỏ và nhanh thật.

## 6. Thí nghiệm thêm

Chạy trong Colab, không commit vào notebook.

### A. Tiêu chí chọn filter, ngay sau khi cắt (không fine-tune)

So 3 cách chọn filter: L1 của weight conv, |γ| của BatchNorm (ý tưởng của Network Slimming), và ngẫu nhiên thật (`randperm`, trung bình 3 seed).

| Ratio | L1 | BN gamma | Ngẫu nhiên (TB 3 seed) |
|---|---|---|---|
| 0.1 | 84.04% | 89.81% | 97.01% |
| 0.2 | 26.87% | 51.21% | 60.83% |
| 0.3 | 9.74% | 24.49% | 25.30% |
| 0.5 | 9.74% | 9.80% | 10.35% |

- Ngược kỳ vọng của notebook ("magnitude should win"): ngẫu nhiên tốt nhất, L1 tệ nhất.
- Kiểm tra code: ratio 0.5 nhánh L1 ra 9.74%, khớp mục 8 của notebook.
- Giả thuyết 1 (BatchNorm): conv → BatchNorm → ReLU, BatchNorm chuẩn hóa lại output mỗi kênh, nên L1 của weight conv không phản ánh đóng góp thật. BN gamma tốt hơn L1 rõ rệt, ủng hộ một phần giả thuyết này.
- Nhưng ngẫu nhiên vẫn ngang hoặc hơn BN gamma, nên BatchNorm chưa giải thích hết. Giả thuyết thêm (chưa kiểm chứng): kênh bị cắt vẫn đang cho ra hằng số khác 0 (ReLU(β)); độ lớn không đo được sự trùng lặp giữa các filter.
- Hiện tượng cắt ngẫu nhiên không thua cắt theo độ lớn đã được ghi nhận trong nghiên cứu (ví dụ "Rethinking the Value of Network Pruning", Liu và cộng sự, 2019).

### B. Tiêu chí chọn filter, sau fine-tune 1 epoch (ratio 0.3)

| Tiêu chí | Ngay sau khi cắt | Sau 1 epoch | Số lỗi |
|---|---|---|---|
| L1 | 9.74% | 99.58% | 42 |
| BN gamma | 24.49% | 99.46% | 54 |
| Ngẫu nhiên (seed 0) | 25.30% (TB 3 seed) | 99.40% | 60 |

- Thứ tự đảo ngược: L1 từ tệ nhất thành tốt nhất.
- Cách hiểu: ngay sau khi cắt đo việc phần còn lại có còn ăn khớp không (BatchNorm dùng thống kê cũ, lớp sau thiếu đầu vào), fine-tune sửa rất nhanh. Sau fine-tune mới đo được phần còn lại giữ bao nhiêu kiến thức.
- Dè dặt: chênh L1 và ngẫu nhiên 18 lỗi, chỉ hơn mức nhiễu (±10) một chút; ngẫu nhiên chỉ 1 seed, chỉ 1 lần chạy, 1 epoch. Fine-tune đủ 4 epoch có thể về gần bằng nhau (L1 và "giữ k đầu" đều ra 99.68%).
- Bài học: không nên đánh giá tiêu chí cắt bằng accuracy ngay sau khi cắt.

### C. Prune rồi quantize

Quantize (symmetric, per-channel, fake-quant như bài 1) model dense và model đã cắt (structured L1, ratio 0.5, fine-tune 4 epoch).

| Bit | Dense (1,147,722 tham số) | Pruned (288,170 tham số) |
|---|---|---|
| FP32 | 99.68% | 99.68% |
| 8 | 99.68% | 99.67% |
| 4 | 99.68% | 99.68% |
| 3 | 99.61% | 99.51% |
| 2 | 25.40% | 24.06% |

- Dự đoán model đã cắt gãy sớm hơn, nhưng cả hai gãy cùng chỗ (giữa 3 và 2 bit). Ở 3 bit pruned thấp hơn 10 lỗi, trong vùng nhiễu.
- Cột dense khớp bài 1 (cũng gãy giữa 3 và 2 bit).
- Cách hiểu: ở 2 bit mỗi weight chỉ còn giữ dấu, chuyện này xảy ra với mọi model. Model đã cắt vẫn gấp khoảng 20 lần student bài 2 (14,458 tham số, 99.24%), nên vẫn còn dư với MNIST. Model đã được fine-tune ổn định trước khi quantize.
- Ước lượng: 288K tham số ở 4 bit (nếu có packing) khoảng 0.14 MB, nhỏ hơn dense FP32 khoảng 32 lần, accuracy giữ 99.68%.
- Muốn thấy gãy sớm hơn phải thử model thật sự ít dư (student bài 2, hoặc cắt 80–90% filter).

### Code để chạy lại

Chạy sau khi đã chạy hết notebook (cần `model_dense`, `pruned`, `Net`, `conv_blocks`, `train`, `evaluate`, `count_params`).

```python
def structured_prune_c(model, ratio, criterion="l1", seed=0):
    """Giống structured_prune, thêm lựa chọn criterion: 'l1', 'bn' hoặc 'random'."""
    g = torch.Generator().manual_seed(seed)
    blocks = conv_blocks(model)
    keep = []
    for blk in blocks:
        n = blk[0].weight.shape[0]
        k = max(1, round(n * (1 - ratio)))
        if criterion == "l1":
            l1 = blk[0].weight.data.abs().sum(dim=(1, 2, 3))
            idx = torch.topk(l1, k).indices.sort().values
        elif criterion == "bn":
            gamma = blk[1].weight.data.abs()
            idx = torch.topk(gamma, k).indices.sort().values
        else:
            idx = torch.randperm(n, generator=g)[:k].sort().values
        keep.append(idx)

    new = Net(widths=tuple(len(k) for k in keep)).eval()
    prev = torch.tensor([0])
    for ob, nb, idx in zip(blocks, conv_blocks(new), keep):
        nb[0].weight.data  = ob[0].weight.data[idx][:, prev, :, :].clone()
        nb[1].weight.data  = ob[1].weight.data[idx].clone()
        nb[1].bias.data    = ob[1].bias.data[idx].clone()
        nb[1].running_mean = ob[1].running_mean[idx].clone()
        nb[1].running_var  = ob[1].running_var[idx].clone()
        prev = idx
    new.head[2].weight.data = model.head[2].weight.data[:, prev].clone()
    new.head[2].bias.data   = model.head[2].bias.data.clone()
    return new

# A: không fine-tune
for r in [0.1, 0.2, 0.3, 0.5]:
    acc_l1 = evaluate(structured_prune_c(model_dense, r, "l1").to(device))
    acc_bn = evaluate(structured_prune_c(model_dense, r, "bn").to(device))
    acc_rd = sum(evaluate(structured_prune_c(model_dense, r, "random", s).to(device))
                 for s in range(3)) / 3
    print(r, acc_l1, acc_bn, acc_rd)

# B: fine-tune 1 epoch ở ratio 0.3
for crit in ["l1", "bn", "random"]:
    torch.manual_seed(0)
    m = structured_prune_c(model_dense, 0.3, crit, seed=0)
    m = train(m, epochs=1, lr=LR * 0.2, tag=f"finetune {crit}")
    print(crit, evaluate(m))
```

Thí nghiệm C: chép hàm `quantize` và `dequantize` từ bài 1, cùng bản rút gọn của `quantize_model`:

```python
def quantize_model(model, n_bits=8, per_channel=True):
    deq = copy.deepcopy(model).cpu().eval()
    for mod in deq.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            q, s, z = quantize(mod.weight.data, n_bits=n_bits, symmetric=True, per_channel=per_channel)
            mod.weight.data = dequantize(q, s, z, per_channel=per_channel)
    return deq

for b in [8, 4, 3, 2]:
    print(b, evaluate(quantize_model(model_dense, n_bits=b).to(device)),
             evaluate(quantize_model(pruned, n_bits=b).to(device)))
```

## 7. Điều em rút ra

- Granularity quyết định pruning có "thật" hay không. Unstructured chịu cắt tốt nhưng không nhỏ hơn, không nhanh hơn; structured nhỏ và nhanh thật (lần đầu tiên latency giảm trong 3 bài).
- Việc fine-tune sau khi cắt là rất nên làm vì cải thiện accuracy rất mạnh (từ ~10% lên 99.68%), và nó làm sai lệch do cắt không dồn sang bước sau.
- Ở thí nghiệm đầu tiên, kết quả của em mâu thuẫn với kỳ vọng của notebook ở thời điểm ngay sau khi cắt (ngẫn nhiên mạnh hơn L1), nhưng có xu hướng khớp sau fine-tune nên em cho rằng không nên đánh giá tiêu chí cắt bằng accuracy ngay sau khi cắt.
- Sau fine-tune, khác biệt giữa các tiêu chí chỉ vài chục lỗi, gần mức nhiễu.
- Thí nghiệm prune rồi quantize kết hợp tốt trên MNIST, nhỏ hơn khoảng 32 lần mà không mất accuracy.
