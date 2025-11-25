"""
LexRank Summarization for Indonesian Text

Eigenvector centrality-based extractive summarization.
Similar to TextRank but uses power method on similarity graph.
"""

import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import TextPreprocessor


class LexRankSummarizer:
    """
    LexRank-based extractive summarizer for Indonesian text.
    
    Uses eigenvector centrality on sentence similarity graph.
    """
    
    # Indonesian stopwords
    STOPWORDS = [
        'yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'untuk', 'pada',
        'adalah', 'sebagai', 'dengan', 'juga', 'akan', 'sudah', 'atau',
        'ia', 'dia', 'mereka', 'kita', 'kami', 'anda', 'saya', 'tidak',
        'bisa', 'ada', 'oleh', 'sebuah', 'dalam', 'tersebut', 'dapat'
    ]
    
    def __init__(
        self, 
        num_sentences: int = 3,
        threshold: float = 0.1,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ):
        """
        Initialize LexRank summarizer.
        
        Args:
            num_sentences: Number of sentences to extract.
            threshold: Minimum similarity to create edge.
            tolerance: Convergence tolerance for power iteration.
            max_iterations: Maximum power iterations.
        """
        self.num_sentences = num_sentences
        self.threshold = threshold
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        self.preprocessor = TextPreprocessor(use_stemmer=True)
        self.vectorizer = TfidfVectorizer(
            stop_words=self.STOPWORDS,
            max_features=5000,
            ngram_range=(1, 2)
        )
    
    def _compute_centrality(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute eigenvector centrality using power iteration.
        
        Args:
            similarity_matrix: NxN sentence similarity matrix.
            
        Returns:
            Centrality scores for each sentence.
        """
        n = similarity_matrix.shape[0]
        
        # Create transition matrix (row-normalize)
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = similarity_matrix / row_sums
        
        # Initialize scores uniformly
        scores = np.ones(n) / n
        
        # Power iteration
        for _ in range(self.max_iterations):
            new_scores = transition_matrix.T @ scores
            
            # Normalize
            new_scores /= np.linalg.norm(new_scores, ord=1)
            
            # Check convergence
            if np.abs(new_scores - scores).sum() < self.tolerance:
                break
            
            scores = new_scores
        
        return scores
    
    def summarize(
        self, 
        text: str, 
        num_sentences: Optional[int] = None
    ) -> str:
        """
        Generate extractive summary using LexRank.
        
        Args:
            text: Input text to summarize.
            num_sentences: Override default number of sentences.
            
        Returns:
            Summary string.
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        n_sent = num_sentences or self.num_sentences
        
        # Tokenize into sentences
        original_sentences = self.preprocessor.tokenize_sentences(text)
        
        if len(original_sentences) <= n_sent:
            return text
        
        # Preprocess for similarity calculation
        processed_sentences = [
            self.preprocessor.preprocess_sentence(s) 
            for s in original_sentences
        ]
        
        # Filter empty sentences
        valid_indices = [
            i for i, s in enumerate(processed_sentences) 
            if len(s.strip()) > 0
        ]
        
        if len(valid_indices) <= n_sent:
            return ' '.join(original_sentences[:n_sent])
        
        valid_processed = [processed_sentences[i] for i in valid_indices]
        
        # Get TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(valid_processed)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Apply threshold
        similarity_matrix[similarity_matrix < self.threshold] = 0
        
        # Remove self-loops
        np.fill_diagonal(similarity_matrix, 0)
        
        # Compute centrality scores
        scores = self._compute_centrality(similarity_matrix)
        
        # Get top sentences
        ranked_indices = np.argsort(scores)[::-1][:n_sent]
        
        # Map back to original indices and sort by position
        original_indices = sorted([valid_indices[i] for i in ranked_indices])
        
        # Build summary
        summary_sentences = [original_sentences[i] for i in original_indices]
        
        return ' '.join(summary_sentences)
    
    def summarize_multi(
        self, 
        documents: List[str], 
        num_sentences: Optional[int] = None
    ) -> str:
        """
        Summarize multiple documents.
        
        Args:
            documents: List of document texts.
            num_sentences: Number of sentences to extract.
            
        Returns:
            Combined summary.
        """
        combined_text = ' '.join(documents)
        return self.summarize(combined_text, num_sentences)


if __name__ == "__main__":
    # Demo
    text = """
    Pemerintah Indonesia mengumumkan kebijakan baru tentang vaksinasi COVID-19.
    Program ini ditargetkan menjangkau 70% populasi dalam waktu enam bulan.
    Kementerian Kesehatan telah menyiapkan lebih dari 100 juta dosis vaksin.
    Vaksinasi akan dilakukan secara bertahap mulai dari tenaga kesehatan.
    Masyarakat diimbau untuk mendaftar melalui aplikasi resmi pemerintah.
    WHO memberikan dukungan penuh terhadap program vaksinasi Indonesia.
    """
    
    summarizer = LexRankSummarizer(num_sentences=2)
    summary = summarizer.summarize(text)
    
    print("Original:")
    print(text)
    print("\nSummary:")
    print(summary)
