import sys
import os
# Add parent dir to path to find src if needed, but we rely on installed package
from kokoro import KPipeline
import inspect
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print("KPipeline Init Signature:")
print(inspect.signature(KPipeline.__init__))
