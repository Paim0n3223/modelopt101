# Bài 5: Kernel Optimization

## 1. Bài này làm gì

- Notebook: `examples/05_kernel_optimization.ipynb`, chạy trên Google Colab (Tesla T4, Triton 3.6.0, compute capability 7.5, đời Turing).
- Bài yêu cầu viết một kernel nhân ma trận C = A × B bằng Triton (FP16), rồi so với `torch.matmul` (gọi thư viện cuBLAS của NVIDIA) ở các cỡ 512, 1024, 2048, 4096.
- Bài này khác bốn bài trước: không đổi model, không đổi phép toán, chỉ đổi cách GPU thực thi. Kết quả phải giống hệt, chỉ khác tốc độ.
- Các chỉ số được đo: đúng sai so với `torch.matmul`, latency (ms) và throughput (TFLOP/s = 2MNK / thời gian).

## 2. Ý tưởng em hiểu được

- Mỗi ô của C là tích vô hướng một hàng A với một cột B, nên tổng số phép tính luôn là 2MNK, kernel nào cũng phải làm đúng chừng đó. Khác nhau chỉ ở việc làm nhanh đến đâu, nên throughput (TFLOP/s) là thước đo công bằng để so với nhau và so với đỉnh của phần cứng.
- Vấn đề cốt lõi là "bức tường bộ nhớ": GPU tính nhanh hơn rất nhiều so với tốc độ đọc dữ liệu. T4 tính được khoảng 65 nghìn tỉ phép/giây (FP16, tensor core) nhưng chỉ đọc được khoảng 320 GB/giây, tức phải dùng mỗi byte đọc vào cho khoảng 200 phép tính thì mạch tính mới không phải ngồi chờ.
- Em hình dung bằng ví dụ nhà bếp: đầu bếp (GPU) nấu cực nhanh nhưng nguyên liệu nằm trong kho dưới tầng hầm (bộ nhớ chính), bên cạnh chỉ có một mặt bàn nhỏ (bộ nhớ nhanh trên chip). Nếu làm món nào chạy xuống kho lấy đồ cho món đó thì gần hết thời gian là chạy cầu thang.
- Cách ngây thơ là tính xong từng ô của C rồi mới sang ô khác. Mỗi ô đọc nguyên một hàng A và một cột B, nên cùng một hàng, một cột bị đọc lại rất nhiều lần. Với ví dụ 4×4 chia khối 2×2, cách ngây thơ cần 32 lần đọc, còn chia khối chỉ cần 16.
- Tiling là gom việc theo khối: mỗi program phụ trách một tile của C (64×64). Mỗi bước, nó nạp một mảnh A và một mảnh B lên bộ nhớ nhanh, nhân hai mảnh rồi cộng vào toàn bộ tile (`acc`). Mỗi số nạp vào được dùng cho cả hàng hoặc cả cột của tile, nên lượng đọc từ bộ nhớ chính giảm khoảng bằng kích thước tile.
- Cách dễ nhất để phân biệt hai cách là nhìn ma trận C được cập nhật thế nào: cách ngây thơ thì từng ô xong hẳn rồi mới sang ô khác, còn chia khối thì cả khối cùng nhích lên một phần mỗi bước và cùng xong ở bước cuối. Chính vì cả khối cùng tiến lên nên một mảnh dữ liệu vừa nạp mới phục vụ được nhiều ô cùng lúc.
- Input là FP16 nhưng `acc` là FP32: mỗi ô là tổng của hàng nghìn tích, nếu cộng dồn bằng FP16 (chỉ 10 bit mantissa) thì sai số làm tròn tích lũy rất nhanh. Nên kernel nhân bằng FP16, cộng bằng FP32, cuối cùng mới đổi về FP16 khi ghi ra.
- Grid layout quyết định program nào tính tile nào. Các program có số gần nhau thường chạy cùng lúc, nên nếu xếp chúng thành một ô vuông nhỏ (theo nhóm) thay vì một hàng dài thì chúng cần ít dải dữ liệu A, B hơn, và dữ liệu dễ còn nằm trong cache L2.
- Mask chặn việc đọc, ghi ra ngoài ma trận khi kích thước không chia hết cho tile: chỉ ghi những ô có hàng < M và cột < N, còn phần thiếu khi nạp theo chiều K thì điền 0.
- Em từng thắc mắc sao không để nhà sản xuất phần cứng lo luôn chuyện này. Thực ra họ đã làm (tensor core, TPU, thư viện cuBLAS), nhưng phần cứng không tự quyết được hết: bộ nhớ buộc phải phân tầng theo vật lý, cách chia khối tốt nhất phụ thuộc hình dạng ma trận, kiểu số và từng con chip, và các phép tính mới (FlashAttention, định dạng ternary của BitVLA, gộp nhiều phép tính) xuất hiện nhanh hơn chu kỳ thiết kế chip vài năm.
- Trước khi tối ưu phải biết kernel đang nghẽn ở bộ nhớ hay ở sức tính (phân tích roofline). Bài này cho em thấy điều đó rất rõ ở mục 5 và 6.

## 3. Phần em viết (3 TODO)

- TODO 2, tiling: `acc += tl.dot(a, b)`, nhân mảnh A 64×32 với mảnh B 32×64 rồi cộng vào tile 64×64.
- TODO 3, mask khi ghi: `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)`.
- TODO 1, xếp tile theo nhóm:
  ```python
  num_pid_in_group = GROUP_M * num_pid_n
  group_id = pid // num_pid_in_group; first_pid_m = group_id * GROUP_M
  group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  ```
- Em làm theo thứ tự TODO 2 → TODO 3 → đo → TODO 1 → đo lại, để tách riêng được tác dụng của việc xếp tile theo nhóm.

## 4. Những chỗ em vấp

- Colab hết hạn mức GPU từ bài 4, em phải đổi sang tài khoản Google khác để có GPU.
- Placeholder lần này "đẹp giả" rất mạnh. Kernel chỉ cộng 0 nên compiler bỏ luôn phần tính, kết quả toàn số 0 (sai lệch 173.75, `match: False`), vậy mà benchmark lại báo nhanh hơn cuBLAS tới 40 lần, 880 TFLOP/s ở cỡ 4096. Con số này gấp hơn 13 lần đỉnh lý thuyết của T4, tức là vô lý về mặt vật lý. Em rút ra hai thói quen: luôn kiểm tra đúng sai trước khi đo tốc độ, và luôn so con số đo được với đỉnh của phần cứng.
- Lúc đầu em hiểu sai tiling thành "tính xong ô (1,1) rồi sang ô (1,2)". Thực ra đó chính là cách ngây thơ; phải xem minh họa từng bước em mới thấy điểm khác là cả khối cùng tiến lên.
- Sau khi sửa xong, kernel đúng nhưng chậm hơn cuBLAS 15–25 lần, trái với kỳ vọng "cùng tầm" của notebook. Nguyên nhân hóa ra không nằm ở code mà ở compiler và đời GPU (mục 6A).

## 5. Kết quả

### Đúng sai

- Sau khi sửa: `max abs difference = 0.0625`, `match torch.matmul: True`.

### Benchmark

Sau TODO 2 và 3 (xếp tile theo hàng):

| Size | torch ms | triton ms | torch TF/s | triton TF/s | speedup |
|---|---|---|---|---|---|
| 512 | 0.032 | 0.301 | 8.3 | 0.9 | 0.11x |
| 1024 | 0.077 | 1.828 | 27.9 | 1.2 | 0.04x |
| 2048 | 0.916 | 16.027 | 18.7 | 1.1 | 0.06x |
| 4096 | 8.506 | 142.010 | 16.2 | 1.0 | 0.06x |

Sau TODO 1 (xếp tile theo nhóm, bản trong notebook đã commit):

| Size | torch ms | triton ms | torch TF/s | triton TF/s | speedup |
|---|---|---|---|---|---|
| 512 | 0.034 | 0.301 | 7.9 | 0.9 | 0.11x |
| 1024 | 0.080 | 1.861 | 26.8 | 1.2 | 0.04x |
| 2048 | 0.922 | 15.796 | 18.6 | 1.1 | 0.06x |
| 4096 | 9.044 | 148.901 | 15.2 | 0.9 | 0.06x |

- Kernel chỉ đạt khoảng 1 TFLOP/s ở mọi cỡ, chậm hơn cuBLAS 15–25 lần.
- TODO 1 gần như không thay đổi gì, chênh lệch nằm trong vùng nhiễu đo. Xếp theo nhóm giúp đọc bộ nhớ hiệu quả hơn, nhưng kernel trên T4 đang nghẽn ở sức tính chứ không phải ở bộ nhớ, nên tối ưu đúng kỹ thuật mà sai chỗ nghẽn thì không có tác dụng.
- `torch.matmul` đạt cao nhất ở cỡ 1024 rồi giảm dần ở 2048 và 4096. Em đoán là T4 (giới hạn 70 W) bị hạ xung nhịp khi chạy nặng lâu, nhưng chưa kiểm chứng.

## 6. Thí nghiệm thêm

Các thí nghiệm chạy trong Colab, không commit vào notebook.

### A. Vì sao chậm: Triton có dùng tensor core không?

Em đếm các lệnh trong PTX (mã trung gian mà Triton sinh ra):

| | Kết quả |
|---|---|
| Lệnh tensor core (`mma.sync`) | 0 |
| Lệnh nhân-cộng thường (`fma.rn`) | 1024 |
| Compute capability | (7, 5) |

- Trên T4, Triton 3.6 biên dịch `tl.dot` thành nhân-cộng thường trên CUDA core, hoàn toàn không dùng tensor core. Trong khi đó cuBLAS vẫn dùng tensor core trên đời Turing, nên nhanh hơn rất nhiều.
- Code của em đúng, giới hạn nằm ở compiler và đời GPU. Đây là ví dụ rõ nhất cho câu "kernel optimization is hardware-constrained" trong slide: cùng một dòng `tl.dot`, chạy nhanh hay chậm phụ thuộc compiler có hỗ trợ con chip đó không.
- GPU trên laptop của em (RTX 3050 Ti, đời Ampere, compute capability 8.6) có thể cho kết quả khác, nhưng Triton không chạy chính thức trên Windows (cần WSL2), nên em chưa thử.

### B. Ma trận không vuông

| M | N | K | Đúng | Sai lệch lớn nhất |
|---|---|---|---|---|
| 500 | 300 | 700 | True | 0.0625 |
| 512 | 4096 | 1024 | True | 0.0625 |
| 1000 | 1000 | 1000 | True | 0.0625 |
| 77 | 33 | 129 | True | 0.0000 |

- Mask khi ghi (TODO 3) và mask khi nạp (`other=0.0`) đều hoạt động đúng với kích thước không chia hết cho tile, kể cả trường hợp rất lẻ 77 × 33 × 129.
- Sai lệch 0.0625 đúng bằng khoảng cách giữa hai số FP16 liền nhau ở độ lớn 64–128. Tức là kernel của em và cuBLAS chỉ lệch nhau một bước làm tròn do thứ tự cộng khác nhau, giống chuyện "thước" ở bài quantization.

### C. Kích thước tile (cỡ 2048)

| Tile | BLOCK_K | ms | TFLOP/s | Thanh ghi | Spill |
|---|---|---|---|---|---|
| 32×32 | 32 | 11.140 | 1.54 | 165 | 0 |
| 64×64 (mặc định) | 32 | 15.787 | 1.09 | 255 | 92 |
| 64×64 | 64 | 86.850 | 0.20 | 64 | 586 |
| 128×64 | 32 | 187.182 | 0.09 | 64 | 958 |
| 128×128 | 32 | 316.520 | 0.05 | 64 | 2,772 |

- Kết quả ngược với trực giác "tile to dùng lại dữ liệu nhiều hơn nên nhanh hơn": tile càng to càng chậm, tile nhỏ nhất lại nhanh nhất, nhanh hơn cấu hình mặc định 1.4 lần. Mọi cấu hình vẫn cho kết quả đúng.
- Số spill tăng đúng theo độ chậm. Tile to cần nhiều thanh ghi hơn; khi vượt quá, compiler phải đẩy dữ liệu ra bộ nhớ chậm (register spill), giống như rổ nguyên liệu quá to so với mặt bàn. 255 là số thanh ghi tối đa mỗi luồng trên GPU NVIDIA, và cấu hình mặc định của notebook đã chạm trần, bắt đầu spill trên T4. Chỉ tile 32×32 là vừa thanh ghi, không spill.
- Ba cấu hình lớn chỉ ghi nhận 64 thanh ghi nhưng spill rất nhiều. Em chưa rõ vì sao compiler chọn con số 64.
- Tile to còn tạo ra ít program hơn (2048 chia tile 32×32 được 4,096 program, chia 128×128 chỉ còn 256), nên khó giữ cả 40 khối tính của T4 cùng bận.
- Kích thước tile tốt nhất phụ thuộc vào phần cứng. Đó là lý do Triton có `triton.autotune` để tự thử nhiều cấu hình rồi chọn, thay vì đoán.

### Code để chạy lại

Chạy sau khi đã chạy hết notebook.

```python
# A: đếm lệnh trong PTX
c_chk = torch.empty((M, N), device=device, dtype=DTYPE)
grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
ck = matmul_kernel[grid](
    a, b, c_chk, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_chk.stride(0), c_chk.stride(1),
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8,
)
ptx = ck.asm["ptx"]
print(ptx.count("mma.sync"), ptx.count("fma.rn"), torch.cuda.get_device_capability())

# B: ma trận không vuông
for Ms, Ns, Ks in [(500, 300, 700), (512, 4096, 1024), (1000, 1000, 1000), (77, 33, 129)]:
    a_ = torch.randn(Ms, Ks, device=device, dtype=DTYPE)
    b_ = torch.randn(Ks, Ns, device=device, dtype=DTYPE)
    out, ref_ = triton_matmul(a_, b_), torch.matmul(a_, b_)
    print(Ms, Ns, Ks, torch.allclose(out, ref_, atol=1e-1, rtol=1e-1), (out - ref_).abs().max().item())

# C: kích thước tile, thanh ghi và spill
size = 2048
a2 = torch.randn(size, size, device=device, dtype=DTYPE)
b2 = torch.randn(size, size, device=device, dtype=DTYPE)
for bm, bn, bk in [(32, 32, 32), (64, 64, 32), (64, 64, 64), (128, 64, 32), (128, 128, 32)]:
    ms = triton.testing.do_bench(lambda: triton_matmul(a2, b2, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk))
    c_ = torch.empty((size, size), device=device, dtype=DTYPE)
    grid = (triton.cdiv(size, bm) * triton.cdiv(size, bn),)
    ck = matmul_kernel[grid](
        a2, b2, c_, size, size, size,
        a2.stride(0), a2.stride(1), b2.stride(0), b2.stride(1), c_.stride(0), c_.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
    )
    print(bm, bn, bk, ms, 2 * size**3 / ms / 1e9, ck.n_regs, ck.n_spills)
```

## 7. Điều em rút ra

- Tối ưu kernel không thay đổi kết quả, chỉ thay đổi cách dữ liệu di chuyển giữa các tầng bộ nhớ. Nhưng hiệu quả của nó phụ thuộc rất nhiều vào phần cứng và compiler: cùng một đoạn code đúng, trên T4 với Triton 3.6 lại chậm hơn cuBLAS 15–25 lần chỉ vì không được dùng tensor core.
- Phải biết nút thắt nằm ở đâu trước khi tối ưu. TODO 1 là một kỹ thuật đúng, nhưng không có tác dụng ở đây vì kernel không nghẽn ở bộ nhớ.
- Lý thuyết "tile to thì dùng lại dữ liệu nhiều hơn" chỉ đúng khi tài nguyên trên chip (thanh ghi, bộ nhớ nhanh) cho phép. Trên T4, tile nhỏ nhất lại nhanh nhất vì không bị spill. Phải đo chứ không nên đoán, và đó cũng là lý do có autotune.
- Luôn kiểm tra đúng sai trước khi đo tốc độ, và so con số đo được với đỉnh lý thuyết của phần cứng. Placeholder cho thấy một kernel sai hoàn toàn vẫn có thể báo "nhanh gấp 40 lần".
- Bài này giúp em hiểu vì sao nhóm vla.cpp phải tự viết kernel cho BitVLA: định dạng ternary chưa có thư viện nào hỗ trợ sẵn, và kernel riêng của họ làm mỗi bước nhanh hơn 4.5 lần. Khi phép tính hoặc phần cứng nằm ngoài những gì thư viện có sẵn, tối ưu kernel trở thành việc bắt buộc.
