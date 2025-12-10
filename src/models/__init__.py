"""
Knowledge Graph Module for Trust-Driven Summarization Model (TDSM)

This module provides:
1. Entity Extraction - NER for Indonesian text using IndoBERT
2. Relation Extraction - Identifying relationships between entities
3. Temporal Anchoring - Extracting and normalizing temporal expressions
4. Knowledge Graph Construction - Building graph from documents
5. Fact Verification - Cross-referencing claims against the graph
6. Constrained Generation - Grounding summarization in KG facts

Architecture:
    Documents → EntityExtractor → RelationExtractor → TemporalAnchor
                                        ↓
                              KnowledgeGraph (NetworkX)
                                        ↓
                              FactVerifier → ConstrainedSummarizer
"""

from .entity_extractor import EntityExtractor, Entity, EntityType
from .relation_extractor import RelationExtractor, Relation, RelationType
from .temporal_anchor import TemporalAnchor, TemporalExpression, TemporalType
from .knowledge_graph import KnowledgeGraph, KGTriple
from .fact_verifier import FactVerifier, VerificationResult
from .constrained_summarizer import ConstrainedSummarizer

# Summarizers
from .textrank import TextRankSummarizer
from .lexrank import LexRankSummarizer
from .kg_enhanced_textrank import KGEnhancedTextRank

# Embeddings
from .embeddings import SentenceEmbedder, get_embedding_model, get_cache_stats

# Shared embedder singleton (GPU memory efficient)
_shared_embedder = None


def get_shared_embedder():
    """
    Get or create a shared SentenceEmbedder instance.
    
    Using a singleton ensures:
    - Only one GPU model load across all summarizers
    - Shared sentence cache for faster repeated encoding
    """
    global _shared_embedder
    if _shared_embedder is None:
        _shared_embedder = SentenceEmbedder()
    return _shared_embedder


__all__ = [
    # KG Components
    'EntityExtractor',
    'Entity',
    'EntityType',
    'RelationExtractor',
    'Relation',
    'RelationType',
    'TemporalAnchor',
    'TemporalExpression',
    'TemporalType',
    'KnowledgeGraph',
    'KGTriple',
    'FactVerifier',
    'VerificationResult',
    'ConstrainedSummarizer',
    # Summarizers
    'TextRankSummarizer',
    'LexRankSummarizer',
    'KGEnhancedTextRank',
    # Embeddings
    'SentenceEmbedder',
    'get_embedding_model',
    'get_shared_embedder',
    'get_cache_stats',
]
