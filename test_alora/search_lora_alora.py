from huggingface_hub import list_models
results = list_models(author='ibm-granite', search='alora')
for m in results:
    print(m.id)

print("==================")
results = list_models(author='ibm-granite', search='lora')
for m in results:
    print(m.id)
