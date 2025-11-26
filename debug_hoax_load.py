from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_name = "indobenchmark/indobert-base-p1"

print(f"Attempting to load {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
