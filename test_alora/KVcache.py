from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

text = "台灣的首都是"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 第一次：不用 KV cache，計時
import time
t0 = time.time()
with torch.no_grad():
    out1 = model(**inputs)
t1 = time.time()

# 看 KV cache 長什麼樣子
past = out1.past_key_values
print(f"沒有 cache，耗時：{(t1-t0)*1000:.1f} ms")
print(f"\nKV cache 結構：")
print(f"  層數：{len(past)}")
print(f"  每層 key shape：{past[0][0].shape}")
print(f"  每層 val shape：{past[0][1].shape}")
# shape 是 [batch, heads, tokens, head_dim]

# 第二次：用 KV cache 繼續生成下一個 token
next_token_input = tokenizer("台", return_tensors="pt").to(model.device)
t2 = time.time()
with torch.no_grad():
    out2 = model(
        **next_token_input,
        past_key_values=past  # 重用剛才的 cache
    )
t3 = time.time()
print(f"\n有 cache，只算新 token，耗時：{(t3-t2)*1000:.1f} ms")
