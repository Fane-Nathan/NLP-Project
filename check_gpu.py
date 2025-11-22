import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"VRAM Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
