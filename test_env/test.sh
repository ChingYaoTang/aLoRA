python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

invocation_ids = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj'],
    alora_invocation_tokens=invocation_ids,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

prompt = '<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n'
inputs = tokenizer(prompt, return_tensors='pt')

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)

print('生成結果:', tokenizer.decode(out[0], skip_special_tokens=True))
print('✅ aLoRA 端對端測試通過')
"