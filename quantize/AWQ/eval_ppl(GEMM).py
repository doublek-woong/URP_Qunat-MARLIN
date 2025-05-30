import os
import time
import psutil
import subprocess

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AwqConfig,
)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME     = "Meta-Llama-3.1-8B"
TOKENIZER_PATH = "/home/gunwoong/URP2025-1/llama3-8b"
QUANT_PATH     = "/home/gunwoong/URP2025-1/Meta-Llama-3.1-8B-AWQ-GEMM-INT4"
SEQLEN         = 2048
DEVICE         = torch.device("cuda:1")
# ────────────────────────────────────────────────────────────────────────────────

def fmt_mib(x: int) -> float:
    return x / (1024**2)

def get_driver_gpu_used(gpu_index: int = 0) -> float:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i", str(gpu_index)
    ])
    return max(float(x) for x in out.decode().splitlines() if x.strip())

@torch.no_grad()
def eval_ppl_and_speed(model, tokens, seqlen, device):
    total_nll, total_toks = 0.0, 0
    # make sure all kernels are done before timing
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    for i in range(tokens.size(1) // seqlen):
        chunk = tokens[:, i*seqlen:(i+1)*seqlen].to(DEVICE)
        out   = model(chunk, labels=chunk)
        total_nll += out.loss.item() * chunk.numel()
        total_toks += chunk.numel()

    # wait for last kernels
    torch.cuda.synchronize(DEVICE)
    t1 = time.time()
    peak_inference = torch.cuda.max_memory_reserved(device)
    ppl       = torch.exp(torch.tensor(total_nll/total_toks)).item()
    tok_per_s = total_toks / (t1 - t0)
    return ppl, tok_per_s, peak_inference

def main():
    # 1) Prepare tokens
    ds       = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer= AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    big_txt  = "\n\n".join(ds["text"])
    ids      = tokenizer.encode(big_txt, add_special_tokens=False)
    tokens   = torch.tensor([ids], dtype=torch.long)

    # 2) Load & measure alloc stats
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    awq_cfg = AwqConfig(bits=4, do_fuse=False, fuse_max_seq_len=SEQLEN)
    model   = AutoModelForCausalLM.from_pretrained(
        QUANT_PATH,
        quantization_config=awq_cfg,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={ "": DEVICE.index },
    ).to(DEVICE)
    
    torch.cuda.synchronize(DEVICE)
    peak = torch.cuda.max_memory_reserved(DEVICE)
    print(f"Peak PT alloc : {peak/1024**2:.1f} MiB")
    load_time    = time.time() - t_start
    
    # PyTorch allocator usage (optional)
    gpu_used_tensor = fmt_mib(torch.cuda.memory_allocated(DEVICE))

    # Real driver-reported peak (≈nvidia-smi)
    gpu_used_driver = get_driver_gpu_used(1)

    # CPU RSS
    cpu_rss = fmt_mib(psutil.Process(os.getpid()).memory_info().rss)

    # 3) Eval PPL + speed
    ppl, speed, inf_peak = eval_ppl_and_speed(model, tokens, SEQLEN, DEVICE)

    # 4) Print bench
    print(f"=== {MODEL_NAME} 8B INT4 Bench ===")
    print(f"Load time           : {load_time:5.1f} s")
    print(f"Load GPU memory     : {gpu_used_tensor:7.1f} MiB  (PyTorch)")
    print(f"Load GPU memory     : {gpu_used_driver:7.1f} MiB  (nvidia-smi)")  # ← your 9793 MiB
    print(f"Load CPU RSS        : {cpu_rss:7.1f} MiB")
    print(f"Inference peak GPU  : {inf_peak/1024**2:7.1f} MiB")
    print(f"Inference speed     : {speed:7.1f} tokens/s (+{SEQLEN})")
    print(f"Wikitext-2 PPL      : {ppl:7.2f}")

if __name__ == "__main__":
    main()
