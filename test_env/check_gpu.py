import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 數量: {torch.cuda.device_count()}")
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 簡單運算測試
    x = torch.randn(1000, 1000).cuda()
    y = x @ x.T
    print(f"GPU 運算測試通過，結果 shape: {y.shape}")
else:
    print("⚠️  CUDA 不可用，請確認模組是否正確載入")
