from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

model_name = "ibm-granite/granite-3.2-8b-instruct"
adapter_name = "ibm-granite/granite-3.2-8b-alora-jailbreak"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_name)

# 取出第 0 層的 lora_A 和 lora_B
A = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight
B = model.base_model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

# 計算 ΔW = B × A
A_fp = A.to(torch.float32)
B_fp = B.to(torch.float32)
delta_W = B_fp @ A_fp  # @ 是矩陣乘法

print(f"\nΔW shape: {delta_W.shape}")
print(f"ΔW 的數值範圍：min={delta_W.min():.4f}, max={delta_W.max():.4f}")
print(f"ΔW 的平均絕對值：{delta_W.abs().mean():.6f}")

# 看 ΔW 的分布
total = delta_W.numel()
near_zero = (delta_W.abs() < 0.001).sum().item()
print(f"\nΔW 中接近 0 的元素比例：{near_zero/total*100:.1f}%")
