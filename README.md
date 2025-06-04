# URP_Quant-MARLIN

> Comprehensive performance evaluation of LLaMA-3.1 8B using state-of-the-art quantization methods and GPU kernel acceleration techniques.

---

## 📌 Overview

This repository provides an end-to-end benchmark suite for evaluating **quantization** and **inference acceleration** methods on the [`Meta-LLaMA-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) model.

We evaluate and compare:

| Method          | Quantization | Acceleration (Kernel) |
|-----------------|--------------|------------------------|
| `Base`          | FP16         | None                  |
| `GPTQ`          | INT4         | Torch                 |
| `AWQ`           | INT4         | Torch                 |
| `GPTQ+Marlin`   | INT4         | Marlin (Triton)       |
| `AWQ+Marlin`    | INT4         | Marlin (Triton)       |
| `QQQ`           | INT4         | Torch (custom)        |

---

## 🔧 Setup

### Requirements

- GPU: RTX 3090 or A100 (CUDA 12.4 tested)
- Python: 3.10+
- PyTorch: 2.7.0 + cu126
- Install dependencies:

```bash
conda create -n gptq-marlin python=3.10 -y
conda activate gptq-marlin
pip install -r requirements.txt

