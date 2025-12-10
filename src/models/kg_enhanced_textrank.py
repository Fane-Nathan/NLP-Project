"""
Knowledge Graph Enhanced TextRank Summarization

Extends TextRank by boosting sentences that contain entities from
the Knowledge Graph. This ensures the summary includes key verified facts.

Key idea: Sentences containing KG entities are more likely to contain
verifiable, important information.
"""

import numpy as np
from typing import List, Optional, Dict, Set
from src.models.textrank import TextRankSummarizer
from src.models.mmr import mmr_rerank


class KGEnhancedTextRank(TextRankSummarizer):
    """
    Knowledge Graph Enhanced TextRank Summarizer.
    
    Boosts PageRank scores for sentences containing entities
    that appear in the knowledge graph.
    
    Benefits:
    - Prioritizes verified/important entities
    - Reduces hallucination risk (KG-backed facts)
    - Better integration of named entities in summary
    """
    
    def __init__(
        self, 
        num_sentences: int = 3,
        similarity_threshold: float = 0.1,
        damping: float = 0.85,
        use_mmr: bool = True,
        mmr_lambda: float = 0.6,
        kg_boost: float = 0.3,
        entity_boost: float = 0.1,
        similarity_mode: str = "embedding",  # Use embedding by default
        position_bias: float = 0.3,
        min_sentence_words: int = 5
    ):
        """
        Initialize KG-Enhanced TextRank.
        
        Args:
            num_sentences: Number of sentences to extract.
            similarity_threshold: Minimum similarity to create edge.
            damping: PageRank damping factor.
            use_mmr: Apply MMR reranking.
            mmr_lambda: MMR trade-off parameter.
            kg_boost: Boost factor for sentences with KG entities (0.0-1.0).
            entity_boost: Additional boost per entity mention.
            similarity_mode: 'tfidf', 'embedding', or 'hybrid'.
            position_bias: Weight for position-based scoring (0-1).
            min_sentence_words: Minimum words for a sentence to be considered.
        """
        super().__init__(
            num_sentences=num_sentences,
            similarity_threshold=similarity_threshold,
            damping=damping,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            similarity_mode=similarity_mode,
            position_bias=position_bias,
            min_sentence_words=min_sentence_words
        )
        
        self.kg_boost = kg_boost
        self.entity_boost = entity_boost
        self.kg = None
        self.kg_entities = set()
    
    def set_knowledge_graph(self, kg) -> None:
        """
        Set the knowledge graph for entity boosting.
        
        Args:
            kg: KnowledgeGraph instance.
        """
        self.kg = kg
        
        # Extract entity names for fast lookup
        self.kg_entities = set()
        for node_key in kg.graph.nodes():
            node_data = kg.graph.nodes[node_key]
            # Add normalized name
            normalized = node_data.get('normalized', '')
            if normalized:
                self.kg_entities.add(normalized.lower())
            # Add aliases
            for alias in node_data.get('aliases', set()):
                self.kg_entities.add(alias.lower())
        
        print(f"[KGEnhancedTextRank] Loaded {len(self.kg_entities)} entity names from KG")
    
    def _count_kg_entities(self, sentence: str) -> int:
        """Count how many KG entities appear in a sentence."""
        if not self.kg_entities:
            return 0
        
        count = 0
        sentence_lower = sentence.lower()
        
        for entity in self.kg_entities:
            if entity in sentence_lower:
                count += 1
        
        return count
    
    def _apply_kg_boost(
        self, 
        scores: np.ndarray, 
        sentences: List[str]
    ) -> np.ndarray:
        """
        Apply knowledge graph boosting to sentence scores.
        
        Sentences containing KG entities get higher scores.
        """
        if not self.kg_entities:
            return scores
        
        boosted_scores = scores.copy()
        
        for i, sentence in enumerate(sentences):
            entity_count = self._count_kg_entities(sentence)
            
            if entity_count > 0:
                # Base boost for having any KG entity
                boost = self.kg_boost
                # Additional boost per entity (diminishing returns)
                boost += self.entity_boost * min(entity_count, 5)
                
                # Apply boost (multiplicative)
                boosted_scores[i] *= (1.0 + boost)
        
        # Renormalize
        if boosted_scores.sum() > 0:
            boosted_scores /= boosted_scores.sum()
        
        return boosted_scores
    
    def summarize(
        self, 
        text: str, 
        num_sentences: Optional[int] = None
    ) -> str:
        """
        Generate KG-enhanced extractive summary.
        
        Overrides parent to apply KG entity boosting after PageRank.
        """
        if not text or len(text.strip()) < 10:
            return ""
        
        n_sent = num_sentences or self.num_sentences
        
        original_sentences = self.preprocessor.tokenize_sentences(text)
        
        if len(original_sentences) <= n_sent:
            return text
        
        processed_sentences = [
            self.preprocessor.preprocess_sentence(s) 
            for s in original_sentences
        ]
        
        valid_indices = [
            i for i, s in enumerate(processed_sentences) 
            if len(s.strip()) > 0
        ]
        
        if len(valid_indices) <= n_sent:
            return ' '.join(original_sentences[:n_sent])
        
        valid_processed = [processed_sentences[i] for i in valid_indices]
        valid_original = [original_sentences[i] for i in valid_indices]
        
        # Build similarity matrix (handles embedding/tfidf modes)
        similarity_matrix = self._build_similarity_matrix(valid_processed, valid_original)
        sentence_vectors = self._get_sentence_vectors(valid_processed, valid_original)
        
        # Rank sentences with PageRank
        scores = self._rank_sentences(similarity_matrix)
        
        # Apply KG entity boosting
        scores = self._apply_kg_boost(scores, valid_original)
        
        # Apply MMR reranking if enabled
        if self.use_mmr:
            selected_indices = mmr_rerank(
                sentences=valid_original,
                sentence_vectors=sentence_vectors,
                scores=scores,
                num_sentences=n_sent,
                lambda_param=self.mmr_lambda
            )
            original_indices = sorted([valid_indices[i] for i in selected_indices])
        else:
            ranked_indices = np.argsort(scores)[::-1][:n_sent]
            original_indices = sorted([valid_indices[i] for i in ranked_indices])
        
        # Build summary
        summary_sentences = [original_sentences[i] for i in original_indices]
        
        return ' '.join(summary_sentences)
    
    def summarize_with_kg(
        self, 
        text: str, 
        kg,
        num_sentences: Optional[int] = None
    ) -> str:
        """
        Convenience method to summarize with KG in one call.
        
        Args:
            text: Text to summarize.
            kg: KnowledgeGraph instance.
            num_sentences: Number of sentences.
            
        Returns:
            Summary string.
        """
        self.set_knowledge_graph(kg)
        return self.summarize(text, num_sentences)
    
    def get_entity_coverage(self, summary: str) -> Dict:
        """
        Analyze entity coverage in the summary.
        
        Returns statistics about how many KG entities are in the summary.
        """
        if not self.kg_entities:
            return {"kg_entities": 0, "found": 0, "coverage": 0.0}
        
        found = set()
        summary_lower = summary.lower()
        
        for entity in self.kg_entities:
            if entity in summary_lower:
                found.add(entity)
        
        return {
            "kg_entities": len(self.kg_entities),
            "found": len(found),
            "coverage": len(found) / len(self.kg_entities) if self.kg_entities else 0.0,
            "entities_in_summary": list(found)[:10]  # Top 10
        }


if __name__ == "__main__":
    print("=== KG Enhanced TextRank Demo ===\n")
    
    # Create sample KG (mock)
    class MockKG:
        def __init__(self):
            import networkx as nx
            self.graph = nx.DiGraph()
            # Add some entities
            entities = [
                ("PERSON:Joko Widodo", "Joko Widodo", {"Jokowi", "Presiden Joko Widodo"}),
                ("ORGANIZATION:Kementerian Kesehatan", "Kementerian Kesehatan", {"Kemkes"}),
                ("PERSON:Budi Gunadi", "Budi Gunadi", {"Menteri Kesehatan"}),
            ]
            for key, normalized, aliases in entities:
                self.graph.add_node(key, normalized=normalized, aliases=aliases)
    
    # Test with KG
    summarizer = KGEnhancedTextRank(num_sentences=3, kg_boost=0.5)
    summarizer.set_knowledge_graph(MockKG())
    
    text = """
    Pemerintah Indonesia mengumumkan kebijakan vaksinasi baru.
    Program ini ditargetkan menjangkau 70% populasi.
    Presiden Joko Widodo memberikan dukungan penuh terhadap program ini.
    Kementerian Kesehatan telah menyiapkan 100 juta dosis vaksin.
    Menteri Kesehatan Budi Gunadi menjelaskan tahapan vaksinasi.
    Masyarakat diimbau untuk mendaftar melalui aplikasi resmi.
    Vaksinasi gratis untuk seluruh warga negara Indonesia.
    """
    
    summary = summarizer.summarize(text)
    coverage = summarizer.get_entity_coverage(summary)
    
    print(f"Summary:\n{summary}\n")
    print(f"Entity Coverage: {coverage['coverage']:.1%}")
    print(f"Entities found: {coverage['entities_in_summary']}")
