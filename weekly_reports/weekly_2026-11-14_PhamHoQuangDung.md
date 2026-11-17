# Báo cáo tuần

- Người viết: Phạm Hồ Quang Dũng
- Thời gian: 14/11/2026
- Repo: https://github.com/Paim0n3223/modelopt101

## 1. Summary

### Highlight công việc trong tuần

- Em đã hoàn thành cả 6 bước của task SmolVLA trên vla.cpp và viết report chính thức ([reports/06_vla_cpp.md](../reports/06_vla_cpp.md)). Script và kết quả đo em đã đưa lên repo.
- Cấu hình em đề xuất cuối cùng là `q8v_fa_mmdef_s4` (Q8_0 cả vision, `--flash-attn --mm-prec default --num-steps 4`):
  - So với bản gốc, latency phía server giảm từ 170.7 ms xuống 96.1 ms (nhanh hơn khoảng 44%), và VRAM giảm khoảng một nửa (từ 2457 xuống 1221 MiB).
  - Trên libero_object, SR đạt 87.0% (174/200) so với 87.5% của bản gốc. Kiểm định McNemar theo từng cặp episode không cho thấy khác biệt (p = 1.00).
  - Trên bộ task dài libero_10, SR đạt 41.0% so với 39.0% của bản gốc (McNemar p = 0.72), nên em cũng chưa thấy SR giảm.
- Em đã chạy xong vòng đo SR chính thức (10 task × 20 episode) cho bản gốc và 3 cấu hình tối ưu. Em chưa thấy cấu hình nào làm giảm SR so với bản gốc (McNemar p từ 0.54 trở lên).
- Về profile, em đo bằng nsys và thấy ở bản gốc, vision chiếm khoảng 55% thời gian, denoise khoảng 36% và LM prefill khoảng 9%. Ở cấu hình đã tối ưu `q8v_fa_mmdef_s5`, vision chiếm khoảng 60%.
- Về pruning không cần train, em thấy nó chỉ có ích khi có thêm bước bù trọng số, và cũng chỉ nhanh thêm khoảng 3.5–5.5% ở BF16. Ghép lên cấu hình đề xuất thì em thấy không đáng, vì mỗi phần trăm tốc độ thêm đều phải trả bằng độ chính xác.
- Em đã làm lại các điểm anh góp ý ở báo cáo tuần trước (chi tiết ở mục 3): chạy KD với 5 seed, tính lại BatchNorm sau khi cắt, bổ sung bằng chứng mã máy cho bài kernel, đo Q4 không quantize vision và đo libero_10. Sau khi đo thêm, em phải sửa lại một số kết luận cũ.

### Insight rút ra

- Với những gì em đo được, pruning không fine-tune đem lại ít lợi ích trên SmolVLA. Em nghĩ có hai lý do. Thứ nhất, vision chiếm nhiều thời gian nhất nhưng em không cắt được, vì kích thước ViT bị ghi cứng trong vla.cpp. Thứ hai, action expert bị giới hạn bởi số lượng kernel chứ không phải băng thông, nên phải bỏ cả layer thì mới giảm được thời gian thật.
- Điều em thấy đáng giá nhất ở phần pruning là bù trọng số bằng bình phương tối thiểu. Cách này không cần gradient, chạy trong vài phút và vừa với 4 GB VRAM, mà hồi phục được nhiều: cùng bỏ 8 layer, SR tăng từ 2/40 lên 28/40.
- Em rút ra là nên đo độ lệch action trên các request thật ghi lại từ LIBERO trước khi chạy SR. Độ lệch này xếp gần đúng thứ tự SR của 8 cấu hình em đã đo (quantize, pruning, giảm bước denoise), nên dùng được để sàng lọc. Có hai cặp bị đảo thứ tự, nhưng cả hai đều có độ lệch sát nhau và SR chênh trong mức nhiễu của 40 episode. Còn độ lệch đo trên một input tổng hợp thì không dự đoán được SR.
- Em cũng học được rằng công cụ đo có thể làm sai kết quả. Cờ đo thời gian theo phần của repo làm tổng thời gian tăng 54–60% và làm sai tỉ lệ giữa các phần, nên em phải đối chiếu với nsys.

## 2. Completed Work

| Đầu việc | Trạng thái | Chi tiết trong [report 06](../reports/06_vla_cpp.md) |
|---|---|---|
| Đo SR chính thức trên LIBERO | Xong | Mục 4.3 |
| Profile (cờ timing theo phase, nsys) | Xong | Mục 4.4 |
| Pruning (CLP trước bước train, width, bù trọng số, khảo sát phương pháp không cần train) | Xong | Mục 4.5 |
| Giảm số bước denoise xuống 4 | Xong | Mục 4.6 |
| Viết report chính thức và đưa script, kết quả lên repo | Xong | Toàn bộ report, mục 6 |

### 2.1. Đo SR chính thức

- Em chạy 10 task × 20 episode (seed 42) cho bản gốc và 3 cấu hình em chọn sau vòng sàng lọc tuần trước.
- Mọi cấu hình có cùng trạng thái ban đầu ở episode thứ k, nên em so theo từng cặp episode bằng kiểm định McNemar.
- Kết quả: bản gốc đạt 87.5%, `bf16_fa_mmdef_s5` 88.0%, `q8v_fa_mmdef` 85.0% và `q8v_fa_mmdef_s5` 86.0%. Khoảng tin cậy 95% rộng khoảng ±5 điểm, nên em chỉ kết luận là chưa thấy SR giảm.

### 2.2. Profile

- Em đo bản gốc, `q8v_fa_mmdef_s5` và `bf16_mmdef`. Cấu hình thứ ba dùng để tách riêng tác dụng của `--mm-prec default`.
- Em chỉ dùng cờ timing theo phase để biết model gồm những phần nào, vì cờ này làm chương trình chậm đi nhiều. Thời gian thật của từng phần em lấy từ nsys.
- Q8 cùng flash-attn giúp giảm khoảng 50 ms ở attention và GEMM của vision, nhưng mất lại khoảng 19 ms cho các kernel phụ, vì kernel gộp của vla.cpp chỉ hỗ trợ trọng số BF16.
- Trên WSL2 em không đọc được bộ đếm hiệu năng GPU (lỗi `ERR_NVGPUCTRPERM`), nên các kết luận của em dựa trên thời gian của từng kernel.

### 2.3. Pruning

- Em làm CLP tới bước trước khi train như anh hướng dẫn, và so 4 cách chọn layer: CKA, cosine, bỏ layer cuối và ngẫu nhiên.
- Trước khi kết luận, em kiểm tra lại pipeline: cùng một bản cắt chạy trên vla.cpp và trên PyTorch chỉ lệch nhau 2.2e-4.
- Khi không bù trọng số, bỏ 8 layer thì SR chỉ còn 0–5%. Khi chưa fine-tune, CKA cũng không chọn layer tốt hơn ngẫu nhiên (tiêu chí ngẫu nhiên em mới chạy một seed).
- Width pruning em chỉ làm được ở action expert. Bỏ 25% neuron FFN thì output lệch ít, nhưng chỉ nhanh hơn khoảng 3 ms.
- Khi có bù trọng số, bản bỏ 4 layer (`cos_k2_comp`) đạt 35/40 ở vòng sàng lọc, bằng bản gốc.
- Em khảo sát 17 phương pháp pruning không cần train. Chỉ nhóm bỏ cả layer là chạy được trên vla.cpp gốc mà vẫn giảm latency thật. Các phương pháp còn lại hoặc cần sửa vla.cpp, hoặc không làm nhanh hơn vì ggml không có kernel nhân ma trận thưa.

### 2.4. Giảm số bước denoise

- Trên nền `q8v_fa_mmdef_s5`, em so các cách nhanh thêm theo mức độ lệch phải trả cho mỗi ms. Giảm còn 4 bước là cách rẻ nhất.
- Ở vòng chính thức, bản 4 bước đạt 87.0% (174/200). Em không thấy khác biệt so với bản 5 bước (McNemar p = 0.87) và bản gốc (p = 1.00).

## 3. Sửa theo feedback tuần trước

Em cảm ơn anh đã góp ý cho báo cáo tuần trước. Dưới đây là những gì em đã làm với từng ý.

- **Kết luận NAS** ([report 04](../reports/04_nas.md)): Em đã bỏ con số "nhanh hơn 1.7x", vì khi đo lại thì bản NAS còn chậm hơn baseline một chút. Em cũng ghi rõ rằng phép đo latency trên CPU dao động lớn hơn chênh lệch giữa hai kiến trúc, nên em chưa kết luận được bản nào nhanh hơn.
- **Kết luận KD** ([report 02](../reports/02_knowledge_distillation.md), mục 6D): Em đã chạy lại 5 cấu hình với 5 seed và so theo từng cặp seed. Kết quả cho thấy lần trước em kết luận hơi vội:
  - KD giúp Student giảm khoảng 12% lỗi, chứ không phải khoảng 20% như lần chạy đơn.
  - Alpha bằng 1 ít lỗi hơn alpha bằng 0 ở cả 5 seed. Vì vậy trong thiết lập này (MNIST, 5 seed), em có cơ sở để nói học hoàn toàn từ teacher tốt hơn học từ nhãn. Alpha bằng 1 cũng nhỉnh hơn alpha bằng 0.7, ngược với lần chạy đơn.
  - Nhận định "student nhỏ hưởng lợi nhiều hơn" không còn đứng vững. Tỉ lệ giảm lỗi của hai student gần như bằng nhau (12% và 10%), và kết quả của TinyStudent dao động quá mạnh để em khẳng định KD có giúp.
- **Bằng chứng tensor core** ([report 05](../reports/05_kernel_optimization.md), mục 6A): Em đã kiểm tra mã máy (SASS) của kernel Triton. Kernel chỉ có lệnh nhân-cộng FP32 trên CUDA core, không có lệnh tensor core nào, còn cuBLAS thì dùng kernel tensor core (em suy ra từ tên kernel). Đúng như anh nhận xét, tensor core chỉ giải thích được một phần độ chậm: kernel của em chỉ đạt khoảng 1 TFLOP/s, trong khi đỉnh FP32 của CUDA core là khoảng 8 TFLOP/s. Phần còn lại là do chính kernel, trong đó một phần do tràn thanh ghi.
- **Tính lại BatchNorm sau khi cắt** ([report 03](../reports/03_pruning.md), mục 6D): Đúng như anh nói, con số 9.74% là do model đoán gần như mọi ảnh vào một lớp. Em thử chỉ tính lại thống kê BN, không train, và accuracy hồi phục được phần lớn: cắt 10–20% thì gần về mức dense, cắt 50% thì L1 từ khoảng 10% lên 72% và ngẫu nhiên lên khoảng 89%. Ở tỉ lệ cắt cao thì vẫn cần fine-tune (4 epoch đưa lên 99.6%). Tuy nhiên sau khi tính lại BN, L1 vẫn không hơn cắt ngẫu nhiên, nên em nghĩ thống kê BN cũ không phải lý do khiến L1 thua.
- **Q4 không quantize vision trên LIBERO** ([report 06](../reports/06_vla_cpp.md), mục 4.3): Em đã đo thêm bản Q4 giữ vision ở BF16. SR sàng lọc vẫn chỉ 11/40, không khác biệt so với Q4 cả vision (16/40, McNemar p = 0.33) và kém xa bản gốc (35/40). Vì vậy em kết luận SR sập chủ yếu là do Q4 ở backbone, không phải do quantize vision. Độ lệch `rel` em đo trước đó (0.244 so với 0.258) cũng dự đoán đúng điều này.
- **So sánh SR theo từng cặp episode** ([report 06](../reports/06_vla_cpp.md), mục 4.3): Em đã dùng kiểm định McNemar cho vòng chính thức.
- **libero_10 cho cấu hình tốt nhất** ([report 06](../reports/06_vla_cpp.md), mục 4.7): Em đã đo `q8v_fa_mmdef_s4` cùng bản gốc, mỗi cấu hình 200 episode. SR là 41.0% so với 39.0% (McNemar p = 0.72), tức cấu hình đề xuất cũng không làm giảm SR trên bộ task dài, và server nhanh hơn khoảng 46%. libero_10 khó hơn libero_object nhiều (bản gốc chỉ đạt 39% so với 87.5%). Em chưa có SR tham chiếu nên chưa biết mức này có bình thường không.
- **Profile trước khi pruning** ([report 06](../reports/06_vla_cpp.md), mục 4.4): Em đã profile trước khi làm pruning. Vision chiếm 55–60% thời gian, đúng như anh nhận định. Nhưng kích thước ViT bị ghi cứng trong vla.cpp, nên em không cắt được phía vision nếu không sửa code vla.cpp.

## 4. Plan

### Action items tuần tới

- [ ] Đọc paper về RoboAuto.
- [ ] Viết report về paper này.
