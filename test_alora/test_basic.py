from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 指定模型名稱（HuggingFace Hub 上的路徑）
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("正在載入 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("正在載入模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   # 用半精度節省記憶體
    device_map="auto"            # 自動分配到 GPU/CPU
)

# 2. 準備輸入
messages = [
    {"role": "user", "content": "什麼是 LoRA？請用一句話解釋。"}
]

# 3. 套用 chat template（把對話格式轉成模型看得懂的 token 序列）
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 看看 chat template 長什麼樣子
print("=== Chat Template 輸出 ===")
print(repr(text))  # repr() 會顯示隱藏的特殊字元

# 看看 tokenizer 把文字切成哪些 token
print("\n=== Token IDs ===")
print(inputs["input_ids"])

print("\n=== 每個 Token 對應的文字 ===")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(tokens)

# 4. 生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False   # 關掉隨機性，讓輸出可重現
    )

# 5. 解碼輸出
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],  # 只取新生成的部分
    skip_special_tokens=True
)
print("\n模型回答：", response)
