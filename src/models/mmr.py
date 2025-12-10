"""
Maximal Marginal Relevance (MMR) for Redundancy Reduction

Implements MMR algorithm to select diverse, non-redundant sentences
from a ranked list. Essential for Multi-Document Summarization.

Reference: Carbonell & Goldstein (1998) - "The Use of MMR, Diversity-Based
Reranking for Reordering Documents and Producing Summaries"
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


def mmr_rerank(
    sentences: List[str],
    sentence_vectors: np.ndarray,
    scores: np.ndarray,
    num_sentences: int = 3,
    lambda_param: float = 0.6,
    diversity_threshold: float = 0.7
) -> List[int]:
    """
    Apply MMR to rerank sentences for diversity.
    
    Args:
        sentences: List of original sentences.
        sentence_vectors: TF-IDF or embedding vectors for each sentence.
        scores: Relevance scores from TextRank/LexRank.
        num_sentences: Number of sentences to select.
        lambda_param: Trade-off between relevance and diversity (0-1).
                     Higher = more relevance, Lower = more diversity.
        diversity_threshold: Maximum similarity allowed between selected sentences.
        
    Returns:
        List of indices of selected sentences in order.
    """
    if len(sentences) <= num_sentences:
        return list(range(len(sentences)))
    
    n = len(sentences)
    selected_indices = []
    remaining_indices = list(range(n))
    
    # Precompute similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)
    
    # Normalize scores to [0, 1]
    if scores.max() > scores.min():
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized_scores = np.ones(n) / n
    
    for _ in range(num_sentences):
        if not remaining_indices:
            break
            
        best_idx = None
        best_mmr_score = -float('inf')
        
        for idx in remaining_indices:
            # Relevance component
            relevance = normalized_scores[idx]
            
            # Diversity component (max similarity to already selected)
            if selected_indices:
                max_similarity = max(
                    similarity_matrix[idx][sel_idx] 
                    for sel_idx in selected_indices
                )
            else:
                max_similarity = 0.0
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            # Optional: Hard threshold for diversity
            if selected_indices and max_similarity > diversity_threshold:
                mmr_score -= 1.0  # Heavy penalty for too similar
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return selected_indices


def mmr_summarize(
    sentences: List[str],
    sentence_vectors: np.ndarray,
    scores: np.ndarray,
    num_sentences: int = 3,
    lambda_param: float = 0.6,
    preserve_order: bool = True
) -> Tuple[str, List[int]]:
    """
    Generate summary using MMR reranking.
    
    Args:
        sentences: Original sentences.
        sentence_vectors: Vector representations.
        scores: Relevance scores.
        num_sentences: Target summary length.
        lambda_param: MMR lambda parameter.
        preserve_order: If True, return sentences in original document order.
        
    Returns:
        Tuple of (summary_text, selected_indices).
    """
    selected_indices = mmr_rerank(
        sentences=sentences,
        sentence_vectors=sentence_vectors,
        scores=scores,
        num_sentences=num_sentences,
        lambda_param=lambda_param
    )
    
    if preserve_order:
        selected_indices = sorted(selected_indices)
    
    summary_sentences = [sentences[i] for i in selected_indices]
    summary = ' '.join(summary_sentences)
    
    return summary, selected_indices


class MMRSummarizer:
    """
    Wrapper class for MMR-based summarization.
    Can be used standalone or integrated with TextRank/LexRank.
    """
    
    def __init__(
        self,
        lambda_param: float = 0.6,
        diversity_threshold: float = 0.7,
        preserve_order: bool = True
    ):
        """
        Initialize MMR summarizer.
        
        Args:
            lambda_param: Trade-off parameter (0.5-0.7 recommended).
            diversity_threshold: Max similarity between selected sentences.
            preserve_order: Maintain original sentence order in output.
        """
        self.lambda_param = lambda_param
        self.diversity_threshold = diversity_threshold
        self.preserve_order = preserve_order
    
    def rerank(
        self,
        sentences: List[str],
        vectors: np.ndarray,
        scores: np.ndarray,
        num_sentences: int
    ) -> List[int]:
        """Apply MMR reranking to scored sentences."""
        return mmr_rerank(
            sentences=sentences,
            sentence_vectors=vectors,
            scores=scores,
            num_sentences=num_sentences,
            lambda_param=self.lambda_param,
            diversity_threshold=self.diversity_threshold
        )
    
    def summarize(
        self,
        sentences: List[str],
        vectors: np.ndarray,
        scores: np.ndarray,
        num_sentences: int
    ) -> str:
        """Generate summary with MMR reranking."""
        summary, _ = mmr_summarize(
            sentences=sentences,
            sentence_vectors=vectors,
            scores=scores,
            num_sentences=num_sentences,
            lambda_param=self.lambda_param,
            preserve_order=self.preserve_order
        )
        return summary


if __name__ == "__main__":
    # Demo
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    sentences = [
        "Indonesia announced a new vaccination program.",
        "The government launched the vaccination initiative yesterday.",
        "COVID cases have been declining in the region.",
        "The vaccination program targets 70% coverage.",
        "Indonesia's vaccination effort was announced by officials.",
    ]
    
    # Simulate TextRank scores (higher = more important)
    scores = np.array([0.9, 0.85, 0.7, 0.75, 0.82])
    
    # Vectorize
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences).toarray()
    
    print("=== Without MMR (just top-3 by score) ===")
    top_3 = np.argsort(scores)[::-1][:3]
    for i in top_3:
        print(f"  [{scores[i]:.2f}] {sentences[i]}")
    
    print("\n=== With MMR (diversity-aware) ===")
    mmr_indices = mmr_rerank(sentences, vectors, scores, num_sentences=3, lambda_param=0.5)
    for i in mmr_indices:
        print(f"  [{scores[i]:.2f}] {sentences[i]}")
