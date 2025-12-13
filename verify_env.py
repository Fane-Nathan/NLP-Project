import sys
import importlib.util

def check_package(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"[-] {package_name} NOT installed")
        return False
    else:
        try:
            lib = importlib.import_module(package_name)
            version = getattr(lib, '__version__', 'unknown')
            print(f"[+] {package_name} installed (v{version})")
            return True
        except ImportError as e:
            print(f"[-] {package_name} installed but failed to import: {e}")
            return False

print(f"Python Directory: {sys.executable}")
print(f"Python Version: {sys.version}")

print("\n--- Checking Critical Dependencies ---")
packages = ['transformers', 'torch', 'sastrawi', 'networkx', 'numpy', 'pandas', 'scikit-learn', 'datasets', 'kokoro']
all_good = True
for pkg in packages:
    if not check_package(pkg):
        all_good = False

print("\n--- Checking GPU ---")
try:
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("Torch not installed, cannot check CUDA.")

if all_good:
    print("\n[SUCCESS] All critical packages found.")
else:
    print("\n[WARNING] Some packages are missing.")
