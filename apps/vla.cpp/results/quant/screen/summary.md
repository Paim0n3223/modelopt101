| Config | Flags / env | min ms | mean ms | p50 ms | p90 ms | vision ms | mean vs default | peak VRAM MiB | T start °C | max abs dev | mean abs dev | gripper sign flips | arm max dev | arm mean dev | gripper switch rows |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| default | `(defaults)` | 163.4 | 164.2 | 163.6 | 165.4 | 91.2 | +0.0% | 2039 | 49 | 0.00e+00 | 0.00e+00 | 0/50 | 0.00e+00 | 0.00e+00 | [7]->[7] |
| q8_0 | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q8_0.gguf` | 156.3 | 156.8 | 156.5 | 157.7 | 91.1 | -4.5% | 1149 | 48 | 1.05e-02 | 1.41e-03 | 0/50 | 1.05e-02 | 1.58e-03 | [7]->[7] |
| q8_0_vis | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q8_0-vis.gguf` | 152.3 | 153.0 | 152.6 | 153.9 | 87.2 | -6.8% | 1079 | 48 | 3.07e-02 | 3.95e-03 | 0/50 | 3.07e-02 | 4.49e-03 | [7]->[7] |
| q5_1 | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q5_1.gguf` | 158.6 | 159.3 | 159.1 | 160.3 | 91.0 | -3.0% | 1053 | 48 | 1.95e+00 | 1.37e-02 | 1/50 | 5.34e-02 | 8.70e-03 | [7]->[8] |
| q5_1_vis | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q5_1-vis.gguf` | 156.9 | 157.6 | 157.2 | 158.6 | 89.2 | -4.0% | 957 | 49 | 1.95e+00 | 3.07e-02 | 2/50 | 1.44e-01 | 2.21e-02 | [7]->[9] |
| q5_0 | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q5_0.gguf` | 157.3 | 157.9 | 157.6 | 158.8 | 91.1 | -3.8% | 1033 | 49 | 1.93e+00 | 2.75e-02 | 3/50 | 7.18e-02 | 1.15e-02 | [7]->[10] |
| q5_0_vis | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q5_0-vis.gguf` | 154.9 | 155.6 | 155.2 | 156.6 | 88.5 | -5.2% | 933 | 48 | 1.94e+00 | 2.01e-02 | 2/50 | 6.77e-02 | 1.01e-02 | [7]->[9] |
| q4_1 | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q4_1.gguf` | 157.9 | 158.5 | 158.0 | 159.5 | 91.2 | -3.5% | 1013 | 49 | 1.37e-01 | 1.79e-02 | 0/50 | 1.37e-01 | 2.04e-02 | [7]->[7] |
| q4_1_vis | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q4_1-vis.gguf` | 155.1 | 156.0 | 155.8 | 156.8 | 88.7 | -5.0% | 907 | 50 | 1.93e+00 | 2.72e-02 | 1/50 | 9.09e-02 | 2.34e-02 | [7]->[8] |
| q4_0 | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q4_0.gguf` | 157.2 | 157.9 | 157.4 | 159.1 | 91.6 | -3.8% | 993 | 50 | 5.92e-02 | 1.02e-02 | 0/50 | 5.92e-02 | 1.12e-02 | [7]->[7] |
| q4_0_vis | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q4_0-vis.gguf` | 152.4 | 153.2 | 152.9 | 154.2 | 86.9 | -6.7% | 883 | 51 | 1.12e-01 | 2.42e-02 | 0/50 | 1.12e-01 | 2.65e-02 | [7]->[7] |
| fa_mmdef | `--flash-attn --mm-prec default` | 131.8 | 132.5 | 131.9 | 133.6 | 70.5 | -19.3% | 1201 | 49 | 1.72e-03 | 3.23e-04 | 0/50 | 1.72e-03 | 3.42e-04 | [7]->[7] |
| q8_0_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q8_0.gguf --flash-attn --mm-prec default` | 128.7 | 129.5 | 129.2 | 130.4 | 70.3 | -21.1% | 911 | 48 | 7.44e-03 | 1.37e-03 | 0/50 | 7.44e-03 | 1.53e-03 | [7]->[7] |
| q8_0_vis_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q8_0-vis.gguf --flash-attn --mm-prec default` | 117.6 | 118.3 | 117.9 | 119.2 | 59.2 | -28.0% | 835 | 48 | 2.79e-02 | 3.59e-03 | 0/50 | 2.79e-02 | 4.04e-03 | [7]->[7] |
| q5_1_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q5_1.gguf --flash-attn --mm-prec default` | 131.3 | 132.9 | 131.8 | 135.4 | 70.7 | -19.1% | 813 | 48 | 4.76e-01 | 7.35e-03 | 0/50 | 4.07e-02 | 6.18e-03 | [7]->[7] |
| q5_1_vis_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q5_1-vis.gguf --flash-attn --mm-prec default` | 121.8 | 122.4 | 121.9 | 123.3 | 60.8 | -25.5% | 713 | 47 | 1.95e+00 | 2.49e-02 | 1/50 | 1.37e-01 | 2.10e-02 | [7]->[8] |
| q5_0_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q5_0.gguf --flash-attn --mm-prec default` | 129.7 | 130.6 | 130.4 | 131.7 | 70.5 | -20.5% | 795 | 47 | 1.93e+00 | 2.68e-02 | 3/50 | 6.87e-02 | 1.06e-02 | [7]->[10] |
| q5_0_vis_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q5_0-vis.gguf --flash-attn --mm-prec default` | 120.0 | 120.5 | 120.1 | 121.4 | 60.2 | -26.6% | 689 | 49 | 1.94e+00 | 2.05e-02 | 2/50 | 7.01e-02 | 1.06e-02 | [7]->[9] |
| q4_1_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q4_1.gguf --flash-attn --mm-prec default` | 130.2 | 130.9 | 130.5 | 131.7 | 70.2 | -20.3% | 775 | 47 | 1.23e-01 | 1.70e-02 | 0/50 | 1.23e-01 | 1.92e-02 | [7]->[7] |
| q4_1_vis_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q4_1-vis.gguf --flash-attn --mm-prec default` | 120.4 | 121.1 | 120.6 | 122.2 | 60.6 | -26.2% | 663 | 47 | 1.96e+00 | 2.86e-02 | 1/50 | 1.13e-01 | 2.47e-02 | [7]->[8] |
| q4_0_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q4_0.gguf --flash-attn --mm-prec default` | 129.1 | 129.8 | 129.5 | 130.6 | 70.4 | -21.0% | 755 | 46 | 1.92e+00 | 1.57e-02 | 1/50 | 5.48e-02 | 1.13e-02 | [7]->[8] |
| q4_0_vis_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q4_0-vis.gguf --flash-attn --mm-prec default` | 117.2 | 117.9 | 117.4 | 118.9 | 58.8 | -28.2% | 639 | 46 | 1.14e-01 | 2.41e-02 | 0/50 | 1.14e-01 | 2.63e-02 | [7]->[7] |

Reference: `default`, mean |action| over the 7 real dims = 1.868e-01
