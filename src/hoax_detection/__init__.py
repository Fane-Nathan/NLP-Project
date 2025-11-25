"""
Hoax Detection Module for Indonesian News Summarization

This module provides:
1. IndoBERT + LoRA fine-tuned classifier for hoax detection
2. Outlier detection using cosine similarity
3. Credibility report generation

Architecture: Parallel processing with combined scoring
- Hoax classification runs independently
- Outlier detection runs independently  
- Scores are combined for final filtering decision
"""

from .classifier import HoaxClassifier
from .outlier_detector import OutlierDetector
from .credibility_report import CredibilityAnalyzer, CredibilityReport

__all__ = [
    'HoaxClassifier',
    'OutlierDetector', 
    'CredibilityAnalyzer',
    'CredibilityReport'
]
