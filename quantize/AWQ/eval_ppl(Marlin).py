import os
import time
import psutil
import subprocess

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME     = "Meta-Llama-3.1-8B"
TOKENIZER_PATH = "/home/gunwoong/URP2025-1/llama3-8b"
QUANT_PATH     = "/home/gunwoong/URP2025-1/Meta-Llama-3.1-8B-AWQ-Marlin-INT4"
SEQLEN         = 2048
DEVICE         = torch.device("cuda:0")
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
    torch.cuda.synchronize(device)
    t0 = time.time()

    for i in range(tokens.size(1) // seqlen):
        chunk = tokens[:, i*seqlen:(i+1)*seqlen].to(device)
        out   = model(chunk, labels=chunk)
        total_nll += out.loss.item() * chunk.numel()
        total_toks += chunk.numel()

    torch.cuda.synchronize(device)
    t1 = time.time()
    inf_peak = torch.cuda.max_memory_reserved(device)
    ppl       = torch.exp(torch.tensor(total_nll/total_toks)).item()
    speed     = total_toks / (t1 - t0)
    return ppl, speed, inf_peak

def main():
    # 1) Prepare tokens
    ds        = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    big_txt   = "\n\n".join(ds["text"])
    ids       = tokenizer.encode(big_txt, add_special_tokens=False)
    tokens    = torch.tensor([ids], dtype=torch.long)

    # 2) Load & measure alloc stats
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    # → Marlin 전용 로드
    model = AutoAWQForCausalLM.from_quantized(
        QUANT_PATH,
        device_map={"": "cuda:0"},  # 단일 GPU로 몰아넣기
        safetensors=True,
        fuse_layers=False,
        trust_remote_code=True,
    )

    torch.cuda.synchronize(DEVICE)
    peak_load = torch.cuda.max_memory_reserved(DEVICE)
    load_time = time.time() - t_start

    gpu_allocator = fmt_mib(torch.cuda.memory_allocated(DEVICE))
    gpu_driver    = get_driver_gpu_used(1)
    cpu_rss       = fmt_mib(psutil.Process(os.getpid()).memory_info().rss)

    # 3) Eval PPL + speed
    ppl, speed, inf_peak = eval_ppl_and_speed(model, tokens, SEQLEN, DEVICE)

    # 4) Print bench
    print(f"\n=== {MODEL_NAME} 8B AWQ-Marlin INT4 Bench ===")
    print(f"Load time           : {load_time:5.1f} s")
    print(f"Peak PT alloc       : {peak_load/1024**2:7.1f} MiB")
    print(f"Load GPU memory     : {gpu_allocator:7.1f} MiB  (torch)")
    print(f"Load GPU memory     : {gpu_driver:7.1f} MiB  (nvidia-smi)")
    print(f"Load CPU RSS        : {cpu_rss:7.1f} MiB")
    print(f"Inference peak GPU  : {inf_peak/1024**2:7.1f} MiB")
    print(f"Inference speed     : {speed:7.1f} tokens/s (+{SEQLEN})")
    print(f"Wikitext-2 PPL      : {ppl:7.2f}")
    print()
    
if __name__ == "__main__":
    main()
