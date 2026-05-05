from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(
    repo_id="ibm-granite/granite-3.2-8b-alora-jailbreak",
    filename="adapter_config.json"
)

with open(config_path) as f:
    config = json.load(f)

print(json.dumps(config, indent=2))


# ===============================
from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(
    repo_id="ibm-granite/granite-3.2-8b-lora-jailbreak",
    filename="adapter_config.json"
)

with open(config_path) as f:
    config = json.load(f)

print(json.dumps(config, indent=2))