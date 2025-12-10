"""
Evaluation Module for Summarization

Uses ROUGE metrics to evaluate summary quality.
Supports Indonesian text with optional stemming.
"""

from typing import List, Dict, Optional
import numpy as np
import re

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    rouge_scorer = None

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    StemmerFactory = None


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


class IndonesianEvaluator(Evaluator):
    """
    ROUGE Evaluator with Indonesian-specific preprocessing.
    
    Applies Sastrawi stemming to normalize Indonesian morphological variants
    before computing ROUGE scores. This provides fairer evaluation for 
    Indonesian text where different affixes (me-, ber-, di-, -kan, -i) 
    create many surface forms of the same root word.
    
    Example:
        "meluncurkan" -> "luncur"
        "diluncurkan" -> "luncur"
        Now these will match in ROUGE calculation!
    """
    
    # Indonesian stopwords to optionally remove
    STOPWORDS = {
        'yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'untuk', 'pada',
        'adalah', 'sebagai', 'dengan', 'juga', 'akan', 'sudah', 'atau',
        'ia', 'dia', 'mereka', 'kita', 'kami', 'anda', 'saya', 'tidak',
        'bisa', 'ada', 'oleh', 'sebuah', 'dalam', 'tersebut', 'dapat',
        'lebih', 'telah', 'hanya', 'karena', 'agar', 'seperti', 'saat',
        'bahwa', 'jika', 'maka', 'hal', 'sehingga', 'namun', 'tetapi'
    }
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        use_stemmer: bool = True,
        remove_stopwords: bool = False
    ):
        """
        Initialize Indonesian evaluator.
        
        Args:
            metrics: ROUGE metrics to compute.
            use_stemmer: Apply Sastrawi stemming (recommended for Indonesian).
            remove_stopwords: Remove Indonesian stopwords before evaluation.
        """
        # Initialize parent with use_stemmer=False since we'll use Sastrawi
        super().__init__(metrics=metrics, use_stemmer=False)
        
        self.use_indonesian_stemmer = use_stemmer
        self.remove_stopwords = remove_stopwords
        
        if self.use_indonesian_stemmer:
            if SASTRAWI_AVAILABLE:
                factory = StemmerFactory()
                self.stemmer = factory.create_stemmer()
                print("[IndonesianEvaluator] Sastrawi stemmer loaded ✓")
            else:
                print("[Warning] Sastrawi not available. Install with: pip install Sastrawi")
                self.use_indonesian_stemmer = False
                self.stemmer = None
        else:
            self.stemmer = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess Indonesian text for ROUGE evaluation.
        
        Steps:
        1. Lowercase
        2. Remove punctuation
        3. Apply Sastrawi stemming
        4. Optionally remove stopwords
        
        Args:
            text: Raw Indonesian text.
            
        Returns:
            Preprocessed text.
        """
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS]
        
        # Apply Sastrawi stemming
        if self.use_indonesian_stemmer and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Rejoin
        return ' '.join(tokens)
    
    def calculate_rouge(
        self, 
        references: List[str], 
        predictions: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores with Indonesian preprocessing.
        
        Preprocesses both references and predictions before ROUGE calculation.
        """
        # Preprocess all texts
        preprocessed_refs = [self.preprocess_text(r) for r in references]
        preprocessed_preds = [self.preprocess_text(p) for p in predictions]
        
        # Use parent's calculate_rouge on preprocessed texts
        return super().calculate_rouge(preprocessed_refs, preprocessed_preds)
    
    def evaluate_single(
        self, 
        reference: str, 
        prediction: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate single pair with Indonesian preprocessing.
        """
        preprocessed_ref = self.preprocess_text(reference)
        preprocessed_pred = self.preprocess_text(prediction)
        
        return super().evaluate_single(preprocessed_ref, preprocessed_pred)
    
    def compare_with_standard(
        self,
        references: List[str],
        predictions: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare Indonesian-preprocessed scores with standard ROUGE.
        
        Returns both scores for comparison, useful for research papers.
        """
        # Standard ROUGE (no preprocessing)
        standard_evaluator = Evaluator(metrics=self.metrics, use_stemmer=True)
        standard_scores = standard_evaluator.calculate_rouge(references, predictions)
        
        # Indonesian-preprocessed ROUGE
        indo_scores = self.calculate_rouge(references, predictions)
        
        return {
            "standard": standard_scores,
            "indonesian": indo_scores
        }


if __name__ == "__main__":
    # Demo comparing standard vs Indonesian ROUGE
    print("=== Standard Evaluator ===")
    evaluator = Evaluator()
    
    references = [
        "Pemerintah meluncurkan program vaksinasi untuk masyarakat.",
        "Ekonomi Indonesia tumbuh 5% pada kuartal ketiga."
    ]
    
    predictions = [
        "Program vaksinasi diluncurkan oleh pemerintah.",
        "Pertumbuhan ekonomi Indonesia mencapai 5%."
    ]
    
    scores = evaluator.calculate_rouge(references, predictions)
    evaluator.print_scores(scores)
    
    print("\n=== Indonesian Evaluator (with Sastrawi) ===")
    if SASTRAWI_AVAILABLE:
        indo_evaluator = IndonesianEvaluator(use_stemmer=True)
        indo_scores = indo_evaluator.calculate_rouge(references, predictions)
        indo_evaluator.print_scores(indo_scores)
        
        print("\nNote: 'meluncurkan' and 'diluncurkan' now match after stemming!")
    else:
        print("Install Sastrawi to test: pip install Sastrawi")
