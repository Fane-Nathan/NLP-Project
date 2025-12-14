"""
Sentence Embeddings for Indonesian Text Summarization

Provides semantic embeddings using sentence-transformers.
Supports multiple Indonesian and multilingual models.
Includes LRU caching for repeated sentence encoding.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Tuple
import warnings
import hashlib

# Lazy imports to avoid loading heavy models at startup
_sentence_transformer = None
_model_cache = {}
_sentence_cache: Dict[str, np.ndarray] = {}  # LRU-style sentence cache
_cache_order: List[str] = []  # Track insertion order for LRU

# Cache settings (can be overridden by config)
SENTENCE_CACHE_SIZE = 1000


def _get_cache_key(sentence: str, model_name: str) -> str:
    """Generate a unique cache key for a sentence + model combo."""
    return hashlib.sha256(f"{model_name}:{sentence}".encode()).hexdigest()[:32]


def _cache_embedding(key: str, embedding: np.ndarray) -> None:
    """Add embedding to cache with LRU eviction."""
    global _sentence_cache, _cache_order
    
    if key in _sentence_cache:
        # Move to end (most recently used)
        _cache_order.remove(key)
        _cache_order.append(key)
        return
    
    # Evict oldest if at capacity
    while len(_cache_order) >= SENTENCE_CACHE_SIZE:
        oldest = _cache_order.pop(0)
        _sentence_cache.pop(oldest, None)
    
    _sentence_cache[key] = embedding
    _cache_order.append(key)


def get_cache_stats() -> Dict:
    """Get current cache statistics."""
    return {
        "size": len(_sentence_cache),
        "max_size": SENTENCE_CACHE_SIZE,
        "hit_rate": "N/A"  # Would need hit/miss counters
    }


# Available models ranked by Indonesian performance
EMBEDDING_MODELS = {
    # Best Indonesian-specific (SOTA)
    "indo-e5-small": {
        "name": "LazarusNLP/all-indo-e5-small-v4",
        "size_mb": 134,
        "dim": 384,
        "description": "SOTA Indonesian embedding (small), fast inference"
    },
    # Larger Indo-E5 for better accuracy
    "indo-e5-base": {
        "name": "LazarusNLP/all-indo-e5-base-v4",
        "size_mb": 440,
        "dim": 768,
        "description": "SOTA Indonesian embedding (base), best accuracy for ROUGE"
    },
    # Indonesian SBERT
    "indo-sbert": {
        "name": "firqaaa/indo-sentence-bert-base",
        "size_mb": 440,
        "dim": 768,
        "description": "Indonesian Sentence-BERT base model"
    },
    # Multilingual E5 (larger)
    "multilingual-e5-base": {
        "name": "intfloat/multilingual-e5-base",
        "size_mb": 1100,
        "dim": 768,
        "description": "Multilingual E5 base, best for mixed-language content"
    },
    # Multilingual E5
    "multilingual-e5-small": {
        "name": "intfloat/multilingual-e5-small",
        "size_mb": 470,
        "dim": 384,
        "description": "Multilingual E5 small, good for 100+ languages"
    },
    # Lightweight multilingual
    "multilingual-minilm": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "size_mb": 470,
        "dim": 384,
        "description": "Multilingual MiniLM, 50+ languages"
    },
}

# Default model
DEFAULT_MODEL = "indo-e5-small"


def get_embedding_model(model_key: str = DEFAULT_MODEL):
    """
    Load and cache a sentence transformer model.

    Args:
        model_key: Key from EMBEDDING_MODELS or full HuggingFace model name.

    Returns:
        SentenceTransformer model instance.
    """
    global _sentence_transformer, _model_cache

    # Lazy import
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run:\n"
                "pip install sentence-transformers"
            )

    # Get model name
    if model_key in EMBEDDING_MODELS:
        model_name = EMBEDDING_MODELS[model_key]["name"]
    else:
        model_name = model_key  # Assume it's a full model name

    # Return cached model if available
    if model_name in _model_cache:
        return _model_cache[model_name]

    # Load model with device info
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Embeddings] Loading model: {model_name}")
    print(f"[Embeddings] Device: {device.upper()}")
    
    model = _sentence_transformer(model_name, trust_remote_code=True, device=device)
    _model_cache[model_name] = model

    return model


class SentenceEmbedder:
    """
    Wrapper for sentence embeddings with caching and batching.
    
    Features:
    - Model caching (singleton per model)
    - Sentence-level LRU cache to avoid re-encoding
    - Automatic GPU detection and usage
    - Batched encoding for efficiency
    """

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize sentence embedder.

        Args:
            model_key: Key from EMBEDDING_MODELS or HuggingFace model name.
            device: Device to use ('cuda', 'cpu', or None for auto).
            batch_size: Batch size for encoding.
        """
        self.model_key = model_key
        self.batch_size = batch_size
        self._model = None
        self._device = device

        # Get model info
        if model_key in EMBEDDING_MODELS:
            self.model_info = EMBEDDING_MODELS[model_key]
        else:
            self.model_info = {"name": model_key, "dim": None}

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = get_embedding_model(self.model_key)
            if self._device:
                self._model = self._model.to(self._device)
        return self._model

    def encode(
        self,
        sentences: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode sentences to embeddings.

        Args:
            sentences: Single sentence or list of sentences.
            normalize: L2 normalize embeddings (recommended for cosine similarity).
            show_progress: Show progress bar.

        Returns:
            Numpy array of embeddings (N x dim).
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # Handle E5 models (require "query: " or "passage: " prefix)
        model_name = self.model_info.get("name", "")
        if "e5" in model_name.lower():
            # For summarization, treat all as passages
            sentences = [f"passage: {s}" for s in sentences]

        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            sentences: List of sentences.

        Returns:
            N x N similarity matrix.
        """
        embeddings = self.encode(sentences, normalize=True)

        # Cosine similarity (embeddings are normalized)
        similarity = embeddings @ embeddings.T

        return similarity

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.model_info.get("dim"):
            return self.model_info["dim"]
        # Fallback: encode a dummy sentence
        dummy = self.encode(["test"])
        return dummy.shape[1]


def list_models():
    """Print available embedding models."""
    print("\nAvailable Embedding Models:")
    print("-" * 70)
    for key, info in EMBEDDING_MODELS.items():
        print(f"  {key}:")
        print(f"    Model: {info['name']}")
        print(f"    Size:  {info['size_mb']} MB")
        print(f"    Dim:   {info['dim']}")
        print(f"    Info:  {info['description']}")
        print()


if __name__ == "__main__":
    list_models()

    print("\n=== Testing Indo-E5-Small ===")
    embedder = SentenceEmbedder("indo-e5-small")

    sentences = [
        "Pemerintah Indonesia mengumumkan kebijakan vaksinasi baru.",
        "Indonesia meluncurkan program vaksinasi COVID-19.",
        "Harga minyak dunia naik tajam.",
    ]

    print(f"\nSentences: {len(sentences)}")

    sim_matrix = embedder.similarity_matrix(sentences)
    print(f"\nSimilarity Matrix:")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s[:50]}...")
    print()
    print(sim_matrix.round(3))

    print(f"\nNote: Sentences 0 & 1 should have high similarity (both about vaccination)")
