"""
Hoax Classifier using IndoBERT + LoRA

This module provides inference capabilities for the fine-tuned
hoax detection model. Optimized for low-memory GPU inference.

Usage:
    classifier = HoaxClassifier("models/hoax_indobert_lora")
    result = classifier.predict("Berita ini sangat mencurigakan...")
    print(result)  # {"label": "HOAX", "confidence": 0.92, "probabilities": {...}}
"""

import os
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


@dataclass
class ClassificationResult:
    """Result of hoax classification."""
    text: str
    label: str  # "VALID" or "HOAX"
    confidence: float  # 0.0 - 1.0
    hoax_probability: float
    valid_probability: float
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "hoax_probability": round(self.hoax_probability, 4),
            "valid_probability": round(self.valid_probability, 4)
        }
    
    @property
    def is_hoax(self) -> bool:
        return self.label == "HOAX"


class HoaxClassifier:
    """
    IndoBERT + LoRA Hoax Detection Classifier.
    
    Attributes:
        model_path: Path to saved LoRA model.
        device: Torch device (cuda/cpu).
        threshold: Classification threshold (default 0.5).
    """
    
    # Default base model
    BASE_MODEL = "indobenchmark/indobert-base-p1"
    
    def __init__(
        self,
        model_path: str = "models/hoax_indobert_lora",
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize the hoax classifier.
        
        Args:
            model_path: Path to saved LoRA adapters.
            device: Device to use ("cuda", "cpu", or None for auto).
            threshold: Probability threshold for hoax classification.
        """
        self.model_path = model_path
        self.threshold = threshold
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[HoaxClassifier] Initializing on {self.device}...")
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        print(f"[HoaxClassifier] Ready. Threshold: {self.threshold}")
    
    def _load_config(self) -> Dict:
        """Load model configuration."""
        config_path = os.path.join(self.model_path, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        
        # Default config
        return {
            "base_model": self.BASE_MODEL,
            "max_length": 256,
            "labels": {0: "VALID", 1: "HOAX"}
        }
    
    def _load_model(self):
        """Load tokenizer and model with LoRA adapters."""
        base_model_name = self.config.get("base_model", self.BASE_MODEL)
        max_length = self.config.get("max_length", 256)
        
        # Load tokenizer
        tokenizer_path = self.model_path if os.path.exists(
            os.path.join(self.model_path, "tokenizer_config.json")
        ) else base_model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        
        # Check if LoRA adapters exist
        adapter_config = os.path.join(self.model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config):
            print(f"[HoaxClassifier] Loading LoRA adapters from {self.model_path}")
            
            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=2,
                use_safetensors=True
            )
            
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path
            )
            
            # Merge for faster inference (optional)
            # self.model = self.model.merge_and_unload()
        else:
            print(f"[HoaxClassifier] No LoRA adapters found. Loading base model only.")
            print(f"[HoaxClassifier] Train a model first: python -m src.hoax_detection.train_lora")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=2,
                id2label={0: "VALID", 1: "HOAX"},
                label2id={"VALID": 0, "HOAX": 1}
            )
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, text: str) -> ClassificationResult:
        """
        Classify a single text as hoax or valid.
        
        Args:
            text: Input text to classify.
            
        Returns:
            ClassificationResult with label and confidence.
        """
        if not text or len(text.strip()) < 10:
            return ClassificationResult(
                text=text,
                label="VALID",
                confidence=0.0,
                hoax_probability=0.0,
                valid_probability=1.0
            )
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        valid_prob = float(probs[0])
        hoax_prob = float(probs[1])
        
        # Classification
        if hoax_prob >= self.threshold:
            label = "HOAX"
            confidence = hoax_prob
        else:
            label = "VALID"
            confidence = valid_prob
        
        return ClassificationResult(
            text=text,
            label=label,
            confidence=confidence,
            hoax_probability=hoax_prob,
            valid_probability=valid_prob
        )
    
    @torch.no_grad()
    def predict_batch(
        self, 
        texts: List[str], 
        batch_size: int = 8
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.
        
        Args:
            texts: List of texts to classify.
            batch_size: Batch size for inference.
            
        Returns:
            List of ClassificationResult objects.
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Filter empty texts
            valid_indices = []
            valid_texts = []
            for j, t in enumerate(batch_texts):
                if t and len(t.strip()) >= 10:
                    valid_indices.append(j)
                    valid_texts.append(t)
            
            if not valid_texts:
                # All empty - return default results
                for t in batch_texts:
                    results.append(ClassificationResult(
                        text=t,
                        label="VALID",
                        confidence=0.0,
                        hoax_probability=0.0,
                        valid_probability=1.0
                    ))
                continue
            
            # Tokenize batch
            encoding = self.tokenizer(
                valid_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Build results
            batch_results = []
            valid_idx = 0
            
            for j, t in enumerate(batch_texts):
                if j in valid_indices:
                    p = probs[valid_idx]
                    valid_idx += 1
                    
                    hoax_prob = float(p[1])
                    valid_prob = float(p[0])
                    
                    if hoax_prob >= self.threshold:
                        label = "HOAX"
                        confidence = hoax_prob
                    else:
                        label = "VALID"
                        confidence = valid_prob
                    
                    batch_results.append(ClassificationResult(
                        text=t,
                        label=label,
                        confidence=confidence,
                        hoax_probability=hoax_prob,
                        valid_probability=valid_prob
                    ))
                else:
                    batch_results.append(ClassificationResult(
                        text=t,
                        label="VALID",
                        confidence=0.0,
                        hoax_probability=0.0,
                        valid_probability=1.0
                    ))
            
            results.extend(batch_results)
        
        return results
    
    def get_hoax_score(self, text: str) -> float:
        """
        Get hoax probability score for a text.
        
        Args:
            text: Input text.
            
        Returns:
            Float between 0.0 (definitely valid) and 1.0 (definitely hoax).
        """
        result = self.predict(text)
        return result.hoax_probability
    
    def set_threshold(self, threshold: float):
        """Update classification threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        print(f"[HoaxClassifier] Threshold updated to {threshold}")


# Convenience function
def load_classifier(model_path: str = "models/hoax_indobert_lora") -> HoaxClassifier:
    """Load a pre-trained hoax classifier."""
    return HoaxClassifier(model_path)


if __name__ == "__main__":
    # Demo
    print("=" * 50)
    print("Hoax Classifier Demo")
    print("=" * 50)
    
    classifier = HoaxClassifier()
    
    test_texts = [
        "VIRAL! Vaksin COVID-19 mengandung microchip 5G! Bagikan sebelum dihapus!",
        "Menteri Kesehatan mengumumkan program vaksinasi gratis untuk lansia.",
        "AWAS! Air minum kemasan merek X mengandung racun berbahaya!",
        "Berdasarkan data BPS, inflasi Indonesia pada kuartal III mencapai 3.2%."
    ]
    
    print("\nClassification Results:")
    print("-" * 50)
    
    for text in test_texts:
        result = classifier.predict(text)
        emoji = "⚠️" if result.is_hoax else "✓"
        print(f"\n{emoji} [{result.label}] (confidence: {result.confidence:.2%})")
        print(f"   Hoax prob: {result.hoax_probability:.2%}")
        print(f"   Text: {text[:60]}...")
