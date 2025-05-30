import torch

# 1) torch shims
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not hasattr(torch, "xpu"):
    class _XPU:
        @staticmethod
        def is_available(): return False
    torch.xpu = _XPU()

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path  = "/home/gunwoong/URP2025-1/llama3-8b"
tokenizer   = AutoTokenizer.from_pretrained(model_path, use_fast=False)
common_cfg  = {"zero_point": True, "q_group_size": 128, "w_bit": 4}

versions = [
    ("GEMM","/home/gunwoong/URP2025-1/Meta-Llama-3.1-8B-AWQ-GEMM-INT4"),
    ("Marlin","/home/gunwoong/URP2025-1/Meta-Llama-3.1-8B-AWQ-Marlin-INT4")
]

for version, out_path in versions:
    print(f"\n→ Quantizing version={version}")
    cfg = {**common_cfg, "version": version}
    if version == "Marlin":
        # Marlin은 zero-point 없이
        cfg["zero_point"] = False

    # 1) load onto GPU
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    # 2) quantize (+ verbose & fewer samples if too 느림)
    model.quantize(
        tokenizer,
        quant_config=cfg,

    )

    # 3) pack into GPU memory
    model.pack()

    # 4) save
    model.save_quantized(out_path, safetensors=True)
    tokenizer.save_pretrained(out_path)
    print(f"  Saved to {out_path}")
