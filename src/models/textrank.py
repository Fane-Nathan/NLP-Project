"""
TextRank Summarization for Indonesian Text

Graph-based extractive summarization using sentence similarity.
Uses TF-IDF vectors and cosine similarity to build the graph.
"""

import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from src.preprocessing import TextPreprocessor


class TextRankSummarizer:
    """
    TextRank-based extractive summarizer for Indonesian text.
    
    Uses TF-IDF sentence embeddings and PageRank algorithm
    to select the most important sentences.
    """
    
    # Indonesian stopwords for TF-IDF
    STOPWORDS = [
        'yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'untuk', 'pada',
        'adalah', 'sebagai', 'dengan', 'juga', 'akan', 'sudah', 'atau',
        'ia', 'dia', 'mereka', 'kita', 'kami', 'anda', 'saya', 'tidak',
        'bisa', 'ada', 'oleh', 'sebuah', 'dalam', 'tersebut', 'dapat'
    ]
    
    def __init__(
        self, 
        num_sentences: int = 3,
        similarity_threshold: float = 0.1,
        damping: float = 0.85
    ):
        """
        Initialize TextRank summarizer.
        
        Args:
            num_sentences: Number of sentences to extract.
            similarity_threshold: Minimum similarity to create edge.
            damping: PageRank damping factor.
        """
        self.num_sentences = num_sentences
        self.similarity_threshold = similarity_threshold
        self.damping = damping
        
        self.preprocessor = TextPreprocessor(use_stemmer=True)
        self.vectorizer = TfidfVectorizer(
            stop_words=self.STOPWORDS,
            max_features=5000,
            ngram_range=(1, 2)
        )
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Build sentence similarity matrix using TF-IDF."""
        # Get TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Apply threshold
        similarity_matrix[similarity_matrix < self.similarity_threshold] = 0
        
        # Remove self-loops
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def _rank_sentences(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Apply PageRank to get sentence scores."""
        # Create graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank
        try:
            scores = nx.pagerank(graph, alpha=self.damping, max_iter=100)
            return np.array([scores[i] for i in range(len(scores))])
        except nx.PowerIterationFailedConvergence:
            # Fallback to uniform scores
            return np.ones(similarity_matrix.shape[0]) / similarity_matrix.shape[0]
    
    def summarize(
        self, 
        text: str, 
        num_sentences: Optional[int] = None
    ) -> str:
        """
        Generate extractive summary using TextRank.
        
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
        
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(valid_processed)
        
        # Rank sentences
        scores = self._rank_sentences(similarity_matrix)
        
        # Get top sentences (maintain original order)
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
        # Concatenate all documents
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
    
    summarizer = TextRankSummarizer(num_sentences=2)
    summary = summarizer.summarize(text)
    
    print("Original:")
    print(text)
    print("\nSummary:")
    print(summary)