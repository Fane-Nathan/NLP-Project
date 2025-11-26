import sys
import os
import traceback

# Add project root to path
sys.path.append(os.getcwd())

print("Testing HoaxClassifier loading...")

try:
    from src.hoax_detection import HoaxClassifier
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

try:
    print("Initializing classifier...")
    classifier = HoaxClassifier("models/hoax_indobert_lora")
    print("Initialization successful.")
    
    text = "Ini adalah berita palsu yang sangat berbahaya."
    result = classifier.predict(text)
    print(f"Prediction result: {result}")
    
except Exception as e:
    print(f"Error during loading/prediction: {e}")
    traceback.print_exc()
