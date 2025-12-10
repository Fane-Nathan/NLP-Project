"""
Centralized Configuration for NLP Project

All configurable settings in one place for easy tuning.
"""
import torch


# ============================================================
# Embedding Settings
# ============================================================
EMBEDDING_MODEL = "indo-e5-small"  # Best for Indonesian text
EMBEDDING_BATCH_SIZE = 64  # Optimized for GPU (reduce if OOM)
EMBEDDING_CACHE_SIZE = 1000  # LRU cache for sentence embeddings

# ============================================================
# GPU/Device Settings
# ============================================================
USE_GPU = True
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# ============================================================
# Summarization Defaults
# ============================================================
DEFAULT_SUMMARIZER_MODE = "embedding"  # tfidf, embedding, or hybrid
DEFAULT_NUM_SENTENCES = 3
MMR_LAMBDA = 0.7  # Higher = more relevance, Lower = more diversity
POSITION_BIAS = 0.3  # Weight for early sentences (news inverted pyramid)

# ============================================================
# TTS Settings (Kokoro)
# ============================================================
TTS_SPEED = 1.1
TTS_VOLUME = 1.0
TTS_DEFAULT_PERSONA = "friday"

# ============================================================
# Knowledge Graph Settings
# ============================================================
KG_ENTITY_BOOST = 0.3  # Boost for sentences with KG entities

# ============================================================
# Hoax Detection
# ============================================================
HOAX_CONFIDENCE_THRESHOLD = 0.7


def log_config():
    """Print current configuration."""
    print("\n" + "=" * 50)
    print("NLP Project Configuration")
    print("=" * 50)
    print(f"  Device: {DEVICE}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Batch Size: {EMBEDDING_BATCH_SIZE}")
    print(f"  Cache Size: {EMBEDDING_CACHE_SIZE}")
    print(f"  Summarizer Mode: {DEFAULT_SUMMARIZER_MODE}")
    print("=" * 50)


if __name__ == "__main__":
    log_config()
