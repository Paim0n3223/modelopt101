| Config | Flags / env | min ms | mean ms | p50 ms | p90 ms | vision ms | mean vs default | peak VRAM MiB | T start °C | max abs dev | mean abs dev | gripper sign flips | arm max dev | arm mean dev | gripper switch rows |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bf16 | `(defaults)` | 162.8 | 163.6 | 163.4 | 163.8 | 91.0 | +0.0% | 2039 | 47 | 0.00e+00 | 0.00e+00 | 0/50 | 0.00e+00 | 0.00e+00 | [7]->[7] |
| bf16_fa | `--flash-attn` | 142.9 | 143.5 | 143.4 | 143.8 | 70.5 | -12.3% | 1971 | 48 | 2.16e-03 | 3.68e-04 | 0/50 | 2.16e-03 | 3.97e-04 | [7]->[7] |
| bf16_mmdef | `--mm-prec default` | 152.8 | 153.5 | 153.1 | 153.8 | 91.3 | -6.2% | 1269 | 49 | 6.75e-04 | 1.51e-04 | 0/50 | 6.75e-04 | 1.38e-04 | [7]->[7] |
| bf16_fa_mmdef | `--flash-attn --mm-prec default` | 132.0 | 132.5 | 132.3 | 132.8 | 70.5 | -19.0% | 1201 | 50 | 1.72e-03 | 3.23e-04 | 0/50 | 1.72e-03 | 3.42e-04 | [7]->[7] |
| q8v | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q8_0-vis.gguf` | 152.7 | 153.3 | 153.1 | 153.5 | 87.2 | -6.3% | 1079 | 50 | 3.07e-02 | 3.95e-03 | 0/50 | 3.07e-02 | 4.49e-03 | [7]->[7] |
| q8v_fa | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q8_0-vis.gguf --flash-attn` | 124.9 | 125.3 | 125.1 | 125.5 | 59.2 | -23.4% | 1003 | 50 | 2.79e-02 | 3.59e-03 | 0/50 | 2.79e-02 | 4.05e-03 | [7]->[7] |
| q8v_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q8_0-vis.gguf --mm-prec default` | 146.2 | 146.7 | 146.5 | 147.1 | 87.3 | -10.3% | 911 | 50 | 3.12e-02 | 3.98e-03 | 0/50 | 3.12e-02 | 4.52e-03 | [7]->[7] |
| q8v_fa_mmdef | `VLA_GGUF=/home/qdung/models/smolvla-quant/smolvla-q8_0-vis.gguf --flash-attn --mm-prec default` | 118.1 | 118.7 | 118.6 | 119.1 | 59.2 | -27.4% | 835 | 50 | 2.79e-02 | 3.59e-03 | 0/50 | 2.79e-02 | 4.04e-03 | [7]->[7] |

Reference: `bf16`, mean |action| over the 7 real dims = 1.868e-01
