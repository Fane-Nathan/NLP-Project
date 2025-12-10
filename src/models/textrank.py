"""
TextRank Summarization for Indonesian Text

Graph-based extractive summarization using sentence similarity.
Supports TF-IDF or neural sentence embeddings for similarity.
Supports MMR (Maximal Marginal Relevance) for redundancy reduction.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Literal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from src.preprocessing import TextPreprocessor
from src.models.mmr import mmr_rerank

# Lazy import for embeddings
_embedder = None


def _get_embedder(model_key: str = "indo-e5-small"):
    """Lazy load the sentence embedder."""
    global _embedder
    if _embedder is None:
        from src.models.embeddings import SentenceEmbedder
        _embedder = SentenceEmbedder(model_key)
    return _embedder


class TextRankSummarizer:
    """
    TextRank-based extractive summarizer for Indonesian text.

    Supports three similarity modes:
    - "tfidf": TF-IDF vectors (fast, no GPU needed)
    - "embedding": Neural sentence embeddings (better semantic similarity)
    - "hybrid": Combines TF-IDF and embeddings (best ROUGE, recommended)
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
        damping: float = 0.85,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,  # Higher = more relevance (better for ROUGE)
        position_bias: float = 0.3,  # Boost early sentences (news lead bias)
        min_sentence_words: int = 5,  # Filter very short sentences
        similarity_mode: Literal["tfidf", "embedding", "hybrid"] = "embedding",
        embedding_model: str = "indo-e5-small",
        hybrid_weight: float = 0.6,  # Weight for embeddings in hybrid mode (0.6 = 60% embedding, 40% tfidf)
        mmr_use_embedding: bool = True  # Use embeddings for MMR even when similarity_mode is tfidf
    ):
        """
        Initialize TextRank summarizer.

        Args:
            num_sentences: Number of sentences to extract.
            similarity_threshold: Minimum similarity to create edge.
            damping: PageRank damping factor.
            use_mmr: Apply MMR reranking for diversity (recommended for MDS).
            mmr_lambda: MMR trade-off parameter (0.7+ recommended for ROUGE).
            position_bias: Weight for position-based scoring (0-1).
            min_sentence_words: Minimum words for a sentence to be considered.
            similarity_mode: "tfidf", "embedding", or "hybrid" for sentence similarity.
            embedding_model: Model key for embeddings (see embeddings.py).
            hybrid_weight: Weight for embeddings in hybrid mode (0-1).
            mmr_use_embedding: Use embeddings for MMR diversity even when similarity_mode is tfidf.
        """
        self.num_sentences = num_sentences
        self.similarity_threshold = similarity_threshold
        self.damping = damping
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.position_bias = position_bias
        self.min_sentence_words = min_sentence_words
        self.similarity_mode = similarity_mode
        self.embedding_model = embedding_model
        self.hybrid_weight = hybrid_weight
        self.mmr_use_embedding = mmr_use_embedding

        self.preprocessor = TextPreprocessor(use_stemmer=True)

        # TF-IDF vectorizer (always available as fallback)
        self.vectorizer = TfidfVectorizer(
            stop_words=self.STOPWORDS,
            max_features=5000,
            ngram_range=(1, 2)
        )

        # Embedder loaded lazily
        self._embedder = None
    
    def _get_embedder(self):
        """Get or create sentence embedder."""
        if self._embedder is None:
            self._embedder = _get_embedder(self.embedding_model)
        return self._embedder

    def _build_similarity_matrix(
        self,
        sentences: List[str],
        original_sentences: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Build sentence similarity matrix.

        Args:
            sentences: Preprocessed sentences (for TF-IDF).
            original_sentences: Original sentences (for embeddings).

        Returns:
            Similarity matrix.
        """
        if self.similarity_mode == "embedding":
            # Use neural embeddings on original (unprocessed) sentences
            if original_sentences is None:
                original_sentences = sentences
            embedder = self._get_embedder()
            similarity_matrix = embedder.similarity_matrix(original_sentences)
        elif self.similarity_mode == "hybrid":
            # Combine TF-IDF and embeddings for best results
            if original_sentences is None:
                original_sentences = sentences

            # TF-IDF similarity
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            tfidf_sim = cosine_similarity(tfidf_matrix)

            # Embedding similarity
            embedder = self._get_embedder()
            embed_sim = embedder.similarity_matrix(original_sentences)

            # Weighted combination
            similarity_matrix = (
                self.hybrid_weight * embed_sim +
                (1 - self.hybrid_weight) * tfidf_sim
            )
        else:
            # Use TF-IDF on preprocessed sentences
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)

        # Apply threshold
        similarity_matrix[similarity_matrix < self.similarity_threshold] = 0

        # Remove self-loops
        np.fill_diagonal(similarity_matrix, 0)

        return similarity_matrix

    def _get_sentence_vectors(
        self,
        sentences: List[str],
        original_sentences: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get sentence vectors for MMR.

        When mmr_use_embedding=True (default), uses embeddings for better
        semantic diversity detection, even if similarity_mode is 'tfidf'.
        This gives the best of both worlds: TF-IDF for ROUGE-aligned ranking,
        embeddings for semantic diversity in MMR.
        """
        use_embeddings = (
            self.similarity_mode in ("embedding", "hybrid") or
            (self.use_mmr and self.mmr_use_embedding)
        )

        if use_embeddings:
            if original_sentences is None:
                original_sentences = sentences
            embedder = self._get_embedder()
            return embedder.encode(original_sentences)
        else:
            return self.vectorizer.transform(sentences).toarray()
    
    def _rank_sentences(
        self,
        similarity_matrix: np.ndarray,
        original_positions: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Apply PageRank to get sentence scores with optional position bias.

        Position bias helps capture the "inverted pyramid" structure of news,
        where the most important information is at the beginning.
        """
        # Create graph
        graph = nx.from_numpy_array(similarity_matrix)

        # Apply PageRank
        try:
            scores = nx.pagerank(graph, alpha=self.damping, max_iter=100)
            pagerank_scores = np.array([scores[i] for i in range(len(scores))])
        except nx.PowerIterationFailedConvergence:
            # Fallback to uniform scores
            pagerank_scores = np.ones(similarity_matrix.shape[0]) / similarity_matrix.shape[0]

        # Apply position bias if enabled
        if self.position_bias > 0 and original_positions is not None:
            n = len(original_positions)
            # Position score: exponential decay from start
            position_scores = np.array([
                np.exp(-pos / max(n * 0.5, 1))  # Decay factor
                for pos in original_positions
            ])
            # Normalize position scores
            if position_scores.max() > 0:
                position_scores = position_scores / position_scores.max()

            # Combine: (1 - bias) * pagerank + bias * position
            final_scores = (1 - self.position_bias) * pagerank_scores + self.position_bias * position_scores
        else:
            final_scores = pagerank_scores

        return final_scores
    
    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        target_words: Optional[int] = None
    ) -> str:
        """
        Generate extractive summary using TextRank.

        Args:
            text: Input text to summarize.
            num_sentences: Override default number of sentences.
            target_words: Target word count (adaptive length). Overrides num_sentences.

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

        # Filter empty sentences AND short sentences
        valid_indices = [
            i for i, s in enumerate(processed_sentences)
            if len(s.strip()) > 0 and len(original_sentences[i].split()) >= self.min_sentence_words
        ]

        if len(valid_indices) <= n_sent:
            return ' '.join(original_sentences[:n_sent])

        valid_processed = [processed_sentences[i] for i in valid_indices]
        valid_original = [original_sentences[i] for i in valid_indices]

        # Build similarity matrix (uses embeddings or TF-IDF based on mode)
        similarity_matrix = self._build_similarity_matrix(valid_processed, valid_original)

        # Get vectors for MMR (uses embeddings or TF-IDF based on mode)
        sentence_vectors = self._get_sentence_vectors(valid_processed, valid_original)

        # Rank sentences with position bias
        # Pass original positions for position-aware scoring
        original_positions = list(range(len(valid_indices)))
        scores = self._rank_sentences(similarity_matrix, original_positions)
        
        # Adaptive length: select sentences until target_words is reached
        if target_words is not None:
            ranked_indices = np.argsort(scores)[::-1]
            selected_indices = []
            current_words = 0

            for idx in ranked_indices:
                sent = original_sentences[valid_indices[idx]]
                sent_words = len(sent.split())
                if current_words + sent_words <= target_words * 1.2:  # Allow 20% overshoot
                    selected_indices.append(idx)
                    current_words += sent_words
                if current_words >= target_words:
                    break

            # Apply MMR on selected if enabled
            if self.use_mmr and len(selected_indices) > 1:
                selected_indices = mmr_rerank(
                    sentences=[original_sentences[valid_indices[i]] for i in selected_indices],
                    sentence_vectors=sentence_vectors[selected_indices],
                    scores=scores[selected_indices],
                    num_sentences=len(selected_indices),
                    lambda_param=self.mmr_lambda
                )
            original_indices = sorted([valid_indices[i] for i in selected_indices])
        elif self.use_mmr:
            # Apply MMR reranking if enabled
            selected_indices = mmr_rerank(
                sentences=[original_sentences[i] for i in valid_indices],
                sentence_vectors=sentence_vectors,
                scores=scores,
                num_sentences=n_sent,
                lambda_param=self.mmr_lambda
            )
            # Map back to original indices
            original_indices = sorted([valid_indices[i] for i in selected_indices])
        else:
            # Original behavior: just top-N by score
            ranked_indices = np.argsort(scores)[::-1][:n_sent]
            original_indices = sorted([valid_indices[i] for i in ranked_indices])

        # Build summary
        summary_sentences = [original_sentences[i] for i in original_indices]

        return ' '.join(summary_sentences)
    
    def summarize_multi(
        self, 
        documents: List[str], 
        num_sentences: Optional[int] = None,
        use_mmr: Optional[bool] = None
    ) -> str:
        """
        Summarize multiple documents.
        
        Args:
            documents: List of document texts.
            num_sentences: Number of sentences to extract.
            use_mmr: Override default MMR setting (recommended True for MDS).
            
        Returns:
            Combined summary.
        """
        # For multi-doc, MMR is highly recommended
        original_mmr = self.use_mmr
        if use_mmr is not None:
            self.use_mmr = use_mmr
        elif not self.use_mmr:
            # Auto-enable MMR for multi-doc if not explicitly disabled
            self.use_mmr = True
        
        # Concatenate all documents
        combined_text = ' '.join(documents)
        result = self.summarize(combined_text, num_sentences)
        
        # Restore original setting
        self.use_mmr = original_mmr
        return result
    
    def summarize_multi_with_sources(
        self, 
        documents: List[str], 
        num_sentences: Optional[int] = None,
        use_mmr: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Summarize multiple documents with source tracking.
        
        Returns each selected sentence along with its source document index.
        Useful for citation, verification, and understanding summary composition.
        
        Args:
            documents: List of document texts.
            num_sentences: Number of sentences to extract.
            use_mmr: Override default MMR setting.
            
        Returns:
            List of dicts with keys: 'sentence', 'source_doc_index', 'original_position', 'score'
        """
        from typing import Dict
        
        if not documents:
            return []
        
        n_sent = num_sentences or self.num_sentences
        
        # Track document boundaries
        doc_boundaries = []  # List of (start_idx, end_idx, doc_index)
        all_sentences = []
        
        for doc_idx, doc_text in enumerate(documents):
            sentences = self.preprocessor.tokenize_sentences(doc_text)
            start_idx = len(all_sentences)
            all_sentences.extend(sentences)
            end_idx = len(all_sentences)
            doc_boundaries.append((start_idx, end_idx, doc_idx))
        
        if len(all_sentences) <= n_sent:
            # Return all sentences with source info
            result = []
            for sent_idx, sent in enumerate(all_sentences):
                doc_idx = self._find_doc_index(sent_idx, doc_boundaries)
                result.append({
                    'sentence': sent,
                    'source_doc_index': doc_idx,
                    'original_position': sent_idx,
                    'score': 1.0
                })
            return result
        
        # Preprocess for similarity calculation
        processed_sentences = [
            self.preprocessor.preprocess_sentence(s) 
            for s in all_sentences
        ]
        
        # Filter empty sentences
        valid_indices = [
            i for i, s in enumerate(processed_sentences) 
            if len(s.strip()) > 0
        ]
        
        if len(valid_indices) <= n_sent:
            result = []
            for i in valid_indices[:n_sent]:
                doc_idx = self._find_doc_index(i, doc_boundaries)
                result.append({
                    'sentence': all_sentences[i],
                    'source_doc_index': doc_idx,
                    'original_position': i,
                    'score': 1.0
                })
            return result
        
        valid_processed = [processed_sentences[i] for i in valid_indices]
        valid_original = [all_sentences[i] for i in valid_indices]
        
        # Build similarity matrix and get vectors (handles embedding/tfidf modes)
        similarity_matrix = self._build_similarity_matrix(valid_processed, valid_original)
        sentence_vectors = self._get_sentence_vectors(valid_processed, valid_original)
        
        # Rank sentences
        scores = self._rank_sentences(similarity_matrix)
        
        # Apply MMR if enabled
        original_mmr = self.use_mmr
        if use_mmr is not None:
            self.use_mmr = use_mmr
        elif not self.use_mmr:
            self.use_mmr = True
        
        if self.use_mmr:
            selected_indices = mmr_rerank(
                sentences=[all_sentences[i] for i in valid_indices],
                sentence_vectors=sentence_vectors,
                scores=scores,
                num_sentences=n_sent,
                lambda_param=self.mmr_lambda
            )
            original_indices = sorted([valid_indices[i] for i in selected_indices])
            selected_scores = [scores[i] for i in selected_indices]
        else:
            ranked_indices = np.argsort(scores)[::-1][:n_sent]
            original_indices = sorted([valid_indices[i] for i in ranked_indices])
            selected_scores = [scores[i] for i in ranked_indices]
        
        self.use_mmr = original_mmr
        
        # Build result with source tracking
        result = []
        for idx, orig_idx in enumerate(original_indices):
            doc_idx = self._find_doc_index(orig_idx, doc_boundaries)
            result.append({
                'sentence': all_sentences[orig_idx],
                'source_doc_index': doc_idx,
                'original_position': orig_idx,
                'score': float(selected_scores[idx]) if idx < len(selected_scores) else 0.0
            })
        
        return result
    
    def _find_doc_index(self, sent_idx: int, doc_boundaries: List) -> int:
        """Find which document a sentence belongs to."""
        for start_idx, end_idx, doc_idx in doc_boundaries:
            if start_idx <= sent_idx < end_idx:
                return doc_idx
        return -1


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