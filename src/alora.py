"""
手動實作 Activated LoRA（論文公式 6）
不依賴 PEFT 的 use_alora 參數
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def patch_model_to_alora(model, invoke_token_id: int):
    """
    把已套用 LoRA 的 model 改為 aLoRA 行為：
    只對 invoke_token_id 之後的 token 套用 adapter weights
    """
    original_forwards = {}

    for name, module in model.named_modules():
        # 找到所有 LoRA linear layers
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            original_forwards[name] = module.forward

            def make_alora_forward(mod):
                orig_fwd = mod.forward

                def alora_forward(x, *args, **kwargs):
                    # x shape: (batch, seq_len, hidden)
                    # 這裡先用完整 forward，之後可根據 invoke 位置遮罩
                    return orig_fwd(x, *args, **kwargs)

                return alora_forward

            module.forward = make_alora_forward(module)

    return model

# 測試是否可以 import
if __name__ == "__main__":
    print("aLoRA wrapper 載入成功")
    print("下一步：接上 HuggingFace model 做完整測試")
