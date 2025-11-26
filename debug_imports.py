import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

print("\nAttempting imports...")
try:
    import llama_index
    print(f"✓ llama_index found: {llama_index.__file__}")
except ImportError as e:
    print(f"✗ llama_index NOT found: {e}")

try:
    import llama_index.llms.gemini
    print(f"✓ llama_index.llms.gemini found")
except ImportError as e:
    print(f"✗ llama_index.llms.gemini NOT found: {e}")
