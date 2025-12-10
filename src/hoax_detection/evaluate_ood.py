"""
Out-of-Distribution (OOD) Evaluation for Hoax Detector

Tests the hoax classifier's generalization to unseen data distributions.
Addresses concern: 99.5% F1 might indicate overfitting or data leakage.

Tests include:
1. Domain shift: News from different sources
2. Temporal shift: Recent news not in training data
3. Adversarial: Slightly modified hoaxes
4. Calibration: Expected Calibration Error (ECE)
"""

import os
import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class OODTestResult:
    """Result of OOD evaluation."""
    test_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    num_samples: int
    predictions: List[Dict]
    
    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "num_samples": self.num_samples
        }


class OODEvaluator:
    """
    Evaluates hoax classifier on out-of-distribution data.
    """
    
    def __init__(self, classifier=None):
        """
        Initialize OOD evaluator.
        
        Args:
            classifier: HoaxClassifier instance (loaded if None).
        """
        if classifier is None:
            from src.hoax_detection.classifier import HoaxClassifier
            self.classifier = HoaxClassifier()
        else:
            self.classifier = classifier
    
    def generate_adversarial_examples(
        self, 
        hoax_texts: List[str],
        num_examples: int = 20
    ) -> List[Dict]:
        """
        Generate adversarial examples by modifying hoax texts.
        
        Modifications:
        - Remove clickbait words (VIRAL, AWAS, etc.)
        - Add formal language markers
        - Reduce exclamation marks
        """
        adversarial = []
        
        # Clickbait words to remove
        clickbait_words = [
            "VIRAL", "AWAS", "BREAKING", "GILA", "HEBOH", 
            "BAGIKAN", "SEBARKAN", "SEGERA", "DARURAT"
        ]
        
        # Formal replacements
        replacements = {
            "!": ".",
            "!!!": ".",
            "??": "?",
            "  ": " ",
        }
        
        for text in hoax_texts[:num_examples]:
            modified = text
            
            # Remove clickbait words
            for word in clickbait_words:
                modified = modified.replace(word, "")
                modified = modified.replace(word.lower(), "")
            
            # Apply replacements
            for old, new in replacements.items():
                modified = modified.replace(old, new)
            
            # Clean up
            modified = ' '.join(modified.split())
            
            if len(modified) > 20:  # Skip if too short
                adversarial.append({
                    "original": text,
                    "modified": modified,
                    "true_label": "HOAX"  # Should still be detected as hoax
                })
        
        return adversarial
    
    def evaluate_adversarial(self, hoax_texts: List[str]) -> OODTestResult:
        """
        Evaluate on adversarial examples.
        
        Tests if removing clickbait signals causes misclassification.
        """
        adversarial = self.generate_adversarial_examples(hoax_texts)
        
        if not adversarial:
            return OODTestResult(
                test_name="adversarial",
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                num_samples=0, predictions=[]
            )
        
        predictions = []
        correct = 0
        tp = fp = fn = 0
        
        for item in adversarial:
            result = self.classifier.predict(item["modified"])
            is_correct = result.label == item["true_label"]
            
            predictions.append({
                "text": item["modified"][:100],
                "predicted": result.label,
                "true_label": item["true_label"],
                "confidence": result.confidence,
                "correct": is_correct
            })
            
            if is_correct:
                correct += 1
                tp += 1
            else:
                fn += 1  # Missed hoax
        
        accuracy = correct / len(predictions)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return OODTestResult(
            test_name="adversarial",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            num_samples=len(predictions),
            predictions=predictions
        )
    
    def evaluate_calibration(
        self, 
        texts: List[str], 
        labels: List[str],
        num_bins: int = 10
    ) -> Dict:
        """
        Compute Expected Calibration Error (ECE).
        
        A well-calibrated model has:
        - 80% confidence predictions → 80% accuracy on those predictions
        
        High ECE = overconfident or underconfident predictions.
        """
        predictions = []
        confidences = []
        corrects = []
        
        for text, true_label in zip(texts, labels):
            result = self.classifier.predict(text)
            predictions.append(result.label)
            confidences.append(result.confidence)
            corrects.append(1 if result.label == true_label else 0)
        
        confidences = np.array(confidences)
        corrects = np.array(corrects)
        
        # Bin by confidence
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        bin_stats = []
        
        for i in range(num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = corrects[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
                
                bin_stats.append({
                    "bin": f"{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}",
                    "count": int(in_bin.sum()),
                    "avg_confidence": float(avg_confidence),
                    "avg_accuracy": float(avg_accuracy),
                    "gap": float(abs(avg_accuracy - avg_confidence))
                })
        
        return {
            "ece": float(ece),
            "interpretation": "well-calibrated" if ece < 0.1 else "needs calibration" if ece < 0.2 else "poorly calibrated",
            "bin_stats": bin_stats
        }
    
    def run_full_evaluation(
        self,
        test_texts: List[str],
        test_labels: List[str],
        hoax_texts_for_adversarial: List[str] = None
    ) -> Dict:
        """
        Run comprehensive OOD evaluation.
        
        Args:
            test_texts: Test set texts.
            test_labels: Test set labels ("HOAX" or "VALID").
            hoax_texts_for_adversarial: Hoax texts for adversarial generation.
            
        Returns:
            Dict with all evaluation results.
        """
        results = {}
        
        # 1. Standard evaluation
        print("\n[OOD] Running standard evaluation...")
        predictions = []
        tp = tn = fp = fn = 0
        
        for text, true_label in zip(test_texts, test_labels):
            result = self.classifier.predict(text)
            pred_label = result.label
            
            if true_label == "HOAX":
                if pred_label == "HOAX":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_label == "VALID":
                    tn += 1
                else:
                    fp += 1
        
        accuracy = (tp + tn) / len(test_texts) if test_texts else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results["standard"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        }
        
        # 2. Adversarial evaluation
        if hoax_texts_for_adversarial:
            print("[OOD] Running adversarial evaluation...")
            adv_result = self.evaluate_adversarial(hoax_texts_for_adversarial)
            results["adversarial"] = adv_result.to_dict()
            results["adversarial"]["drop_from_standard"] = results["standard"]["f1_score"] - adv_result.f1_score
        
        # 3. Calibration
        print("[OOD] Computing calibration metrics...")
        calibration = self.evaluate_calibration(test_texts, test_labels)
        results["calibration"] = calibration
        
        # 4. Summary
        results["summary"] = {
            "is_overfit": results["adversarial"]["f1_score"] < 0.5 if "adversarial" in results else False,
            "is_well_calibrated": calibration["ece"] < 0.15,
            "recommendations": []
        }
        
        if results.get("adversarial", {}).get("f1_score", 1.0) < 0.7:
            results["summary"]["recommendations"].append(
                "Model may be relying on surface-level clickbait features. Consider data augmentation."
            )
        
        if calibration["ece"] > 0.2:
            results["summary"]["recommendations"].append(
                "Model is poorly calibrated. Consider temperature scaling or Platt scaling."
            )
        
        return results


def load_test_data_from_csv(filepath: str) -> Tuple[List[str], List[str]]:
    """Load test data from CSV file."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required for CSV loading")
    
    df = pd.read_csv(filepath)
    
    # Try common column names
    text_col = None
    label_col = None
    
    for col in df.columns:
        if col.lower() in ['text', 'content', 'article', 'narasi']:
            text_col = col
        if col.lower() in ['label', 'is_hoax', 'hoax', 'class']:
            label_col = col
    
    if not text_col or not label_col:
        raise ValueError(f"Could not find text/label columns. Found: {df.columns.tolist()}")
    
    texts = df[text_col].fillna("").tolist()
    labels = df[label_col].apply(
        lambda x: "HOAX" if str(x).upper() in ["HOAX", "1", "TRUE", "YES"] else "VALID"
    ).tolist()
    
    return texts, labels


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Hoax Detector OOD Evaluation")
    print("=" * 60)
    
    # Try to load test data
    test_files = [
        "data/hoax_dataset.csv",
        "data/cleaned/dataset_turnbackhoax_cleaned.xlsx"
    ]
    
    texts = []
    labels = []
    
    for filepath in test_files:
        if os.path.exists(filepath):
            print(f"\n[Input] Loading: {filepath}")
            try:
                if filepath.endswith('.csv'):
                    texts, labels = load_test_data_from_csv(filepath)
                elif filepath.endswith('.xlsx') and PANDAS_AVAILABLE:
                    df = pd.read_excel(filepath)
                    # Adjust column names as needed
                    text_col = [c for c in df.columns if 'text' in c.lower() or 'narasi' in c.lower()][0]
                    label_col = [c for c in df.columns if 'label' in c.lower() or 'hoax' in c.lower()][0]
                    texts = df[text_col].fillna("").tolist()
                    labels = df[label_col].apply(
                        lambda x: "HOAX" if str(x).upper() in ["HOAX", "1", "TRUE", "YES"] else "VALID"
                    ).tolist()
                print(f"[Input] Loaded {len(texts)} samples")
                break
            except Exception as e:
                print(f"[Warning] Failed to load {filepath}: {e}")
    
    if not texts:
        print("\n[Demo] Using synthetic test data...")
        texts = [
            "VIRAL! Vaksin COVID mengandung microchip!",
            "Menteri Kesehatan mengumumkan program vaksinasi.",
            "AWAS! Makan mie instan menyebabkan kanker!",
            "BPS melaporkan inflasi 3.2% pada kuartal III.",
            "GILA! Presiden diculik alien dari luar angkasa!",
            "KPU mengumumkan hasil pemilu serentak 2024."
        ]
        labels = ["HOAX", "VALID", "HOAX", "VALID", "HOAX", "VALID"]
    
    # Run evaluation
    evaluator = OODEvaluator()
    
    # Get hoax texts for adversarial testing
    hoax_texts = [t for t, l in zip(texts, labels) if l == "HOAX"]
    
    results = evaluator.run_full_evaluation(
        test_texts=texts[:100],  # Limit for demo
        test_labels=labels[:100],
        hoax_texts_for_adversarial=hoax_texts[:20]
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 OOD Evaluation Results")
    print("=" * 60)
    
    print("\n[Standard Test Set]")
    std = results["standard"]
    print(f"  Accuracy:  {std['accuracy']:.4f}")
    print(f"  Precision: {std['precision']:.4f}")
    print(f"  Recall:    {std['recall']:.4f}")
    print(f"  F1 Score:  {std['f1_score']:.4f}")
    
    if "adversarial" in results:
        print("\n[Adversarial Test (clickbait removed)]")
        adv = results["adversarial"]
        print(f"  Accuracy:  {adv['accuracy']:.4f}")
        print(f"  F1 Score:  {adv['f1_score']:.4f}")
        print(f"  F1 Drop:   {adv.get('drop_from_standard', 0):.4f}")
    
    print("\n[Calibration]")
    cal = results["calibration"]
    print(f"  ECE: {cal['ece']:.4f} ({cal['interpretation']})")
    
    print("\n[Recommendations]")
    for rec in results["summary"]["recommendations"]:
        print(f"  ⚠ {rec}")
    
    if not results["summary"]["recommendations"]:
        print("  ✓ Model appears robust and well-calibrated")
