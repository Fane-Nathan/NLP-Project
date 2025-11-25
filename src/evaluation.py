"""
Evaluation Module for Summarization

Uses ROUGE metrics to evaluate summary quality.
Supports Indonesian text with optional stemming.
"""

from typing import List, Dict, Optional
import numpy as np

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    rouge_scorer = None


class Evaluator:
    """
    Evaluates summarization results using ROUGE metrics.
    
    Supports:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap  
    - ROUGE-L: Longest common subsequence
    """
    
    def __init__(
        self, 
        metrics: Optional[List[str]] = None,
        use_stemmer: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            metrics: ROUGE metrics to compute.
            use_stemmer: Use Porter stemmer (English-based, approximate for Indonesian).
        """
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score not installed. Run: pip install rouge-score")
        
        self.metrics = metrics or ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=use_stemmer)
    
    def calculate_rouge(
        self, 
        references: List[str], 
        predictions: List[str]
    ) -> Dict[str, float]:
        """
        Calculate average ROUGE scores for a batch.
        
        Args:
            references: List of reference summaries (ground truth).
            predictions: List of generated summaries.
            
        Returns:
            Dictionary of average F1 scores for each metric.
        """
        if len(references) != len(predictions):
            raise ValueError("Number of references and predictions must match.")
        
        scores = {metric: [] for metric in self.metrics}
        
        for ref, pred in zip(references, predictions):
            if not ref or not pred:
                continue
                
            score = self.scorer.score(ref, pred)
            for metric in self.metrics:
                scores[metric].append(score[metric].fmeasure)
        
        # Average scores
        avg_scores = {
            metric: np.mean(val) if val else 0.0 
            for metric, val in scores.items()
        }
        
        return avg_scores
    
    def evaluate_single(
        self, 
        reference: str, 
        prediction: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a single reference-prediction pair.
        
        Returns detailed scores (precision, recall, f1).
        """
        if not reference or not prediction:
            return {
                metric: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
                for metric in self.metrics
            }
        
        scores = self.scorer.score(reference, prediction)
        
        return {
            metric: {
                "precision": scores[metric].precision,
                "recall": scores[metric].recall,
                "fmeasure": scores[metric].fmeasure
            }
            for metric in self.metrics
        }
    
    def print_scores(self, scores: Dict[str, float]):
        """Print scores in formatted table."""
        print("\n" + "=" * 40)
        print("ROUGE Evaluation Results")
        print("=" * 40)
        
        for metric, value in scores.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("=" * 40)


if __name__ == "__main__":
    # Demo
    evaluator = Evaluator()
    
    references = [
        "Pemerintah mengumumkan program vaksinasi untuk masyarakat.",
        "Ekonomi Indonesia tumbuh 5% pada kuartal ketiga."
    ]
    
    predictions = [
        "Program vaksinasi diumumkan oleh pemerintah.",
        "Pertumbuhan ekonomi Indonesia mencapai 5%."
    ]
    
    scores = evaluator.calculate_rouge(references, predictions)
    evaluator.print_scores(scores)