"""
Outlier Detection for Multi-Document Summarization

Detects documents that are statistically distant from the centroid
of the document cluster using cosine similarity.

Threshold Strategy: STRICT (>2σ)
- Documents beyond 2 standard deviations from the mean similarity
  to the centroid are flagged as outliers.
- Suitable for high-stakes fact-checking scenarios.

Methods supported:
1. TF-IDF + Cosine Similarity (fast, no GPU required)
2. IndoBERT Embeddings + Cosine Similarity (semantic, GPU recommended)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# Optional: IndoBERT embeddings
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EmbeddingMethod(Enum):
    """Embedding method for similarity calculation."""
    TFIDF = "tfidf"
    INDOBERT = "indobert"


@dataclass
class OutlierResult:
    """Result of outlier detection for a single document."""
    doc_index: int
    text: str
    similarity_to_centroid: float
    z_score: float
    is_outlier: bool
    
    def to_dict(self) -> Dict:
        return {
            "doc_index": self.doc_index,
            "text_preview": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "similarity_to_centroid": round(self.similarity_to_centroid, 4),
            "z_score": round(self.z_score, 4),
            "is_outlier": self.is_outlier
        }


@dataclass  
class OutlierAnalysis:
    """Complete outlier analysis results."""
    results: List[OutlierResult]
    mean_similarity: float
    std_similarity: float
    threshold_z: float
    num_outliers: int
    outlier_indices: List[int]
    
    def to_dict(self) -> Dict:
        return {
            "mean_similarity": round(self.mean_similarity, 4),
            "std_similarity": round(self.std_similarity, 4),
            "threshold_z": self.threshold_z,
            "num_documents": len(self.results),
            "num_outliers": self.num_outliers,
            "outlier_indices": self.outlier_indices,
            "documents": [r.to_dict() for r in self.results]
        }
    
    def get_valid_documents(self) -> List[str]:
        """Return texts of non-outlier documents."""
        return [r.text for r in self.results if not r.is_outlier]
    
    def get_outlier_documents(self) -> List[str]:
        """Return texts of outlier documents."""
        return [r.text for r in self.results if r.is_outlier]


class OutlierDetector:
    """
    Detect outlier documents in a collection using cosine similarity.
    
    Uses STRICT threshold (>2σ) by default for high-stakes fact-checking.
    
    Attributes:
        threshold_z: Z-score threshold for outlier detection.
        method: Embedding method (TF-IDF or IndoBERT).
        min_documents: Minimum documents required for analysis.
    """
    
    # Indonesian stopwords for TF-IDF
    INDONESIAN_STOPWORDS = [
        'yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'untuk', 'pada',
        'adalah', 'sebagai', 'dengan', 'juga', 'akan', 'sudah', 'atau',
        'ia', 'dia', 'mereka', 'kita', 'kami', 'anda', 'saya', 'tidak',
        'bisa', 'ada', 'oleh', 'sebuah', 'dalam', 'tersebut', 'dapat',
        'lebih', 'telah', 'hanya', 'karena', 'agar', 'seperti', 'saat',
        'bahwa', 'jika', 'maka', 'hal', 'sehingga', 'namun', 'tetapi'
    ]
    
    def __init__(
        self,
        threshold_z: float = 2.0,  # STRICT: 2 standard deviations
        method: Union[str, EmbeddingMethod] = EmbeddingMethod.TFIDF,
        min_documents: int = 3,
        indobert_model: str = "indobenchmark/indobert-base-p1"
    ):
        """
        Initialize the outlier detector.
        
        Args:
            threshold_z: Z-score threshold (2.0 = strict, 1.5 = moderate, 3.0 = loose).
            method: "tfidf" or "indobert" for embeddings.
            min_documents: Minimum number of documents for analysis.
            indobert_model: Model name for IndoBERT embeddings.
        """
        self.threshold_z = threshold_z
        self.method = EmbeddingMethod(method) if isinstance(method, str) else method
        self.min_documents = min_documents
        self.indobert_model = indobert_model
        
        # Initialize vectorizer for TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=self.INDONESIAN_STOPWORDS,
            min_df=1,
            max_df=0.95
        )
        
        # IndoBERT components (lazy loading)
        self._bert_tokenizer = None
        self._bert_model = None
        self._device = None
        
        print(f"[OutlierDetector] Initialized")
        print(f"  Method: {self.method.value}")
        print(f"  Threshold: {self.threshold_z}σ (STRICT)" if threshold_z == 2.0 else f"  Threshold: {self.threshold_z}σ")
    
    def _load_indobert(self):
        """Lazy load IndoBERT model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Use method='tfidf' instead.")
        
        if self._bert_model is None:
            print(f"[OutlierDetector] Loading IndoBERT: {self.indobert_model}")
            
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._bert_tokenizer = AutoTokenizer.from_pretrained(self.indobert_model)
            self._bert_model = AutoModel.from_pretrained(self.indobert_model)
            self._bert_model.to(self._device)
            self._bert_model.eval()
            
            print(f"[OutlierDetector] IndoBERT loaded on {self._device}")
    
    def _get_tfidf_embeddings(self, documents: List[str]) -> np.ndarray:
        """Get TF-IDF embeddings for documents."""
        # Fit and transform
        embeddings = self.tfidf_vectorizer.fit_transform(documents)
        return embeddings.toarray()
    
    @torch.no_grad()
    def _get_bert_embeddings(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        """Get IndoBERT embeddings using mean pooling."""
        self._load_indobert()
        
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Tokenize
            encoded = self._bert_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)
            
            # Forward pass
            outputs = self._bert_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Mean pooling over tokens
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _get_embeddings(self, documents: List[str]) -> np.ndarray:
        """Get embeddings based on configured method."""
        if self.method == EmbeddingMethod.TFIDF:
            return self._get_tfidf_embeddings(documents)
        elif self.method == EmbeddingMethod.INDOBERT:
            return self._get_bert_embeddings(documents)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def detect_outliers(self, documents: List[str]) -> OutlierAnalysis:
        """
        Detect outlier documents in a collection.
        
        Documents with cosine similarity to the centroid more than
        threshold_z standard deviations below the mean are flagged.
        
        Args:
            documents: List of document texts.
            
        Returns:
            OutlierAnalysis with detailed results.
        """
        n_docs = len(documents)
        
        # Validation
        if n_docs < self.min_documents:
            print(f"[OutlierDetector] Warning: Only {n_docs} documents. Need {self.min_documents}+ for analysis.")
            # Return all as valid
            results = [
                OutlierResult(
                    doc_index=i,
                    text=doc,
                    similarity_to_centroid=1.0,
                    z_score=0.0,
                    is_outlier=False
                )
                for i, doc in enumerate(documents)
            ]
            return OutlierAnalysis(
                results=results,
                mean_similarity=1.0,
                std_similarity=0.0,
                threshold_z=self.threshold_z,
                num_outliers=0,
                outlier_indices=[]
            )
        
        # Filter empty documents
        valid_docs = []
        valid_indices = []
        for i, doc in enumerate(documents):
            if doc and len(doc.strip()) > 10:
                valid_docs.append(doc)
                valid_indices.append(i)
        
        if len(valid_docs) < self.min_documents:
            print(f"[OutlierDetector] Warning: Only {len(valid_docs)} valid documents after filtering.")
            results = [
                OutlierResult(
                    doc_index=i,
                    text=doc,
                    similarity_to_centroid=1.0 if i in valid_indices else 0.0,
                    z_score=0.0,
                    is_outlier=i not in valid_indices
                )
                for i, doc in enumerate(documents)
            ]
            return OutlierAnalysis(
                results=results,
                mean_similarity=1.0,
                std_similarity=0.0,
                threshold_z=self.threshold_z,
                num_outliers=len(documents) - len(valid_docs),
                outlier_indices=[i for i in range(len(documents)) if i not in valid_indices]
            )
        
        print(f"[OutlierDetector] Analyzing {len(valid_docs)} documents...")
        
        # Get embeddings
        embeddings = self._get_embeddings(valid_docs)
        
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        
        # Calculate similarities to centroid
        similarities = cosine_similarity(embeddings, centroid).flatten()
        
        # Statistical analysis
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Handle edge case of zero std
        if std_sim < 1e-6:
            std_sim = 1e-6
        
        # Calculate Z-scores (negative = further from centroid)
        z_scores = (similarities - mean_sim) / std_sim
        
        # Identify outliers (documents with similarity significantly below mean)
        # Z-score < -threshold means similarity is threshold stdevs below mean
        outlier_mask = z_scores < -self.threshold_z
        
        # Build results
        results = []
        outlier_indices = []
        
        # Map back to original indices
        valid_idx = 0
        for orig_idx, doc in enumerate(documents):
            if orig_idx in valid_indices:
                sim = similarities[valid_idx]
                z = z_scores[valid_idx]
                is_outlier = outlier_mask[valid_idx]
                valid_idx += 1
            else:
                # Empty/invalid document - mark as outlier
                sim = 0.0
                z = -999.0
                is_outlier = True
            
            results.append(OutlierResult(
                doc_index=orig_idx,
                text=doc,
                similarity_to_centroid=float(sim),
                z_score=float(z),
                is_outlier=is_outlier
            ))
            
            if is_outlier:
                outlier_indices.append(orig_idx)
        
        print(f"[OutlierDetector] Found {len(outlier_indices)} outliers out of {n_docs} documents")
        
        return OutlierAnalysis(
            results=results,
            mean_similarity=float(mean_sim),
            std_similarity=float(std_sim),
            threshold_z=self.threshold_z,
            num_outliers=len(outlier_indices),
            outlier_indices=outlier_indices
        )
    
    def filter_outliers(self, documents: List[str]) -> Tuple[List[str], List[int]]:
        """
        Filter out outlier documents.
        
        Args:
            documents: List of document texts.
            
        Returns:
            Tuple of (filtered_documents, removed_indices).
        """
        analysis = self.detect_outliers(documents)
        
        filtered = analysis.get_valid_documents()
        removed = analysis.outlier_indices
        
        return filtered, removed
    
    def get_similarity_matrix(self, documents: List[str]) -> np.ndarray:
        """
        Get pairwise similarity matrix for documents.
        
        Useful for visualization and debugging.
        
        Args:
            documents: List of document texts.
            
        Returns:
            NxN similarity matrix.
        """
        embeddings = self._get_embeddings(documents)
        return cosine_similarity(embeddings)


# Factory functions for different threshold strategies
def create_strict_detector() -> OutlierDetector:
    """Create detector with STRICT threshold (>2σ)."""
    return OutlierDetector(threshold_z=2.0)


def create_moderate_detector() -> OutlierDetector:
    """Create detector with MODERATE threshold (>1.5σ)."""
    return OutlierDetector(threshold_z=1.5)


def create_loose_detector() -> OutlierDetector:
    """Create detector with LOOSE threshold (>3σ)."""
    return OutlierDetector(threshold_z=3.0)


if __name__ == "__main__":
    # Demo
    print("=" * 50)
    print("Outlier Detection Demo")
    print("=" * 50)
    
    # Sample documents (some related, one outlier)
    documents = [
        "Pemerintah Indonesia mengumumkan kebijakan baru tentang vaksinasi COVID-19 untuk masyarakat.",
        "Kementerian Kesehatan melaporkan peningkatan cakupan vaksinasi di seluruh provinsi.",
        "Program vaksinasi nasional mencapai target 70% populasi mendapat dosis lengkap.",
        "WHO memuji keberhasilan Indonesia dalam kampanye vaksinasi massal.",
        "Resep masakan rendang padang yang enak dan mudah dibuat di rumah.",  # OUTLIER
        "Studi terbaru menunjukkan efektivitas vaksin Sinovac mencapai 65% terhadap varian Delta."
    ]
    
    detector = create_strict_detector()
    analysis = detector.detect_outliers(documents)
    
    print(f"\nAnalysis Summary:")
    print(f"  Mean similarity: {analysis.mean_similarity:.4f}")
    print(f"  Std similarity: {analysis.std_similarity:.4f}")
    print(f"  Threshold: {analysis.threshold_z}σ")
    print(f"  Outliers found: {analysis.num_outliers}")
    
    print("\nDocument Results:")
    print("-" * 50)
    
    for result in analysis.results:
        status = "⚠️ OUTLIER" if result.is_outlier else "✓ VALID"
        print(f"\n[{result.doc_index}] {status}")
        print(f"    Similarity: {result.similarity_to_centroid:.4f}")
        print(f"    Z-score: {result.z_score:.4f}")
        print(f"    Text: {result.text[:60]}...")
    
    print("\n" + "=" * 50)
    print("Filtered documents (outliers removed):")
    for doc in analysis.get_valid_documents():
        print(f"  - {doc[:50]}...")
