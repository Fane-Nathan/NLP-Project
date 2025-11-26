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

__all__ = [
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
    'ConstrainedSummarizer'
]
