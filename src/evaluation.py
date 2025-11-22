`from typing import List, Dict
from rouge_score import rouge_scorer
import numpy as np

class Evaluator:
    """
    Evaluates summarization results using ROUGE metrics.
    """
    def __init__(self, metrics: List[str] = ['rouge1', 'rouge2', 'rougeL']):
        self.scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    def calculate_rouge(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """
        Calculate average ROUGE scores for a batch of references and predictions.
        
        Args:
            references: List of reference summaries (ground truth).
            predictions: List of generated summaries.
            
        Returns:
            Dictionary of average F1 scores for each metric.
        """
        if len(references) != len(predictions):
            raise ValueError("Number of references and predictions must match.")
            
        scores = {metric: [] for metric in self.scorer.metrics}
        
        for ref, pred in zip(references, predictions):
            score = self.scorer.score(ref, pred)
            for metric in self.scorer.metrics:
                scores[metric].append(score[metric].fmeasure)
                
        # Average scores
        avg_scores = {metric: np.mean(val) for metric, val in scores.items()}
        return avg_scores
