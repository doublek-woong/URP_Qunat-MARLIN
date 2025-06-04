from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils import BACKEND
# 1. Define model and output paths
model_id = "./llama-3.1-8B"
quant_model_dir = "Llama-3.1-8B-GPTQ"  # output directory for quantized model

# 2. Prepare calibration dataset from WikiText2 (use training split)
calib_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# Filter out very short/empty lines for better calibration
calib_texts = [text for text in calib_data["text"] if text and len(text) > 10]
# (Optionally, use only a subset of texts to limit calibration size, e.g., first 1000 lines)
calib_texts = calib_texts[:1000]

# 3. Set up quantization config for 4-bit with group size 128
quant_config = QuantizeConfig(bits=4, group_size=128)
# (Optional: quant_config.desc_act = True can enable act-order (descending activation) for even better accuracy at the cost of more compute.)

# 4. Load the model with quantization config (in preload mode)
print("Loading the model... (this may take a while)")
model = GPTQModel.load(model_id, quant_config)

# 5. Run GPTQ quantization using the calibration data
print("Quantizing the model with GPTQ 4-bit...")
model.quantize(calib_texts, 
               batch_size=1,
               calibration_dataset_concat_size=2048,
               backend=BACKEND.AUTO,          # 그대로 두면 CuBLAS
               #desc_act=True,                 # 정확도 강화
               buffered_fwd=True,             # 3090 메모리 세이프
               ) # Use batch_size=1 for safe VRAM usage on 24GB

# 6. Save the quantized model in Transformers-compatible format
model.save(quant_model_dir)
print(f"Quantized model saved to: {quant_model_dir}")
