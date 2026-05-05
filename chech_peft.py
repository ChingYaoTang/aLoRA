from peft import LoraConfig
import inspect

# 看看 LoraConfig 有沒有 invocation_string 參數
params = inspect.signature(LoraConfig.__init__).parameters
print("alora_invocation_tokens" in params)