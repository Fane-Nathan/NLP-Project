import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

class MT5Model:
    """
    Wrapper for fine-tuned mT5 model for Hoax Detection & Summarization.
    """
    
    def __init__(self, model_path: str = "models/mt5_lora", base_model_name: str = "google/mt5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading mT5 from {model_path} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            print(f"Warning: Could not load tokenizer from {model_path}, using {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model then adapters
        try:
            # Try loading as PeftModel
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                use_safetensors=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        except Exception as e:
            print(f"Warning: Could not load adapter from {model_path}, using base model only. Error: {e}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            
        self.model.to(self.device)
        self.model.eval()
        
    def generate(self, text: str, task: str = "classify", max_length: int = 128) -> str:
        input_text = f"{task}: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def classify(self, text: str) -> dict:
        """
        Classify text as HOAX or VALID.
        Returns: {'label': 'HOAX'|'VALID', 'confidence': float (dummy for now)}
        """
        result = self.generate(text, task="classify", max_length=10)
        label = result.strip().upper()
        
        # Basic parsing
        if "HOAX" in label:
            final_label = "HOAX"
        elif "VALID" in label:
            final_label = "VALID"
        else:
            final_label = "UNKNOWN"
            
        return {"label": final_label, "raw_output": result}

    def summarize(self, text: str) -> str:
        return self.generate(text, task="summarize", max_length=150)

    def explain(self, text: str) -> str:
        return self.generate(text, task="explain", max_length=200)
