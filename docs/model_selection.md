# Model Selection

## Provisional VLM
`Qwen/Qwen2.5-VL-3B-Instruct`.

Rationale: compact multimodal instruction model, open local inference path, sufficiently small for constrained free-GPU experiments, and supported by Hugging Face Transformers. The code uses lazy loading, optional 4-bit quantization, deterministic decoding, configurable image-pixel budget, and records the exact revision/environment during execution.

The repository does **not** claim that 3B is globally optimal. Final selection is conditional on measured VRAM, stability, latency, and development accuracy in the actual free-GPU environment.

## Confidence signal
Generation confidence is the geometric mean probability of generated tokens. It is treated only as an **uncalibrated likelihood signal**. It is never called a correctness probability until a calibrator is fitted using held-out validation data.

## GPU gate
Current ChatGPT execution container is CPU-only. Exact GPU/VRAM claims remain NOT EXECUTED until the notebook runs on Kaggle/Colab GPU.
