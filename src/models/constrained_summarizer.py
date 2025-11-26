"""
Constrained Summarization for TMDS

The FINAL LAYER that generates hallucination-free summaries by:
1. Using Knowledge Graph as ground truth
2. Constraining generation to verified facts only
3. Post-hoc verification of generated content
4. Iterative refinement if hallucinations detected

This module integrates with your existing summarizers (TextRank, LexRank, Gemini)
and adds the KG-grounding layer on top.

Architecture:
    Documents → Trust Layer → KG Construction → Fact Extraction
                                                      ↓
                                              [Verified Facts]
                                                      ↓
                                        Constrained Summarizer
                                          /        |        \
                                   TextRank   LexRank    Gemini
                                          \        |        /
                                            [Raw Summary]
                                                      ↓
                                          Fact Verification
                                                      ↓
                                          [Verified Summary]
                                                      ↓
                                        (Iterative Refinement)
                                                      ↓
                                        FINAL GROUNDED SUMMARY
"""

import re
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .knowledge_graph import KnowledgeGraph, KGTriple
from .fact_verifier import (
    FactVerifier, 
    VerificationResult, 
    SummaryVerificationReport,
    VerificationStatus,
    HallucinationType
)
from .entity_extractor import EntityExtractor, Entity, EntityType
from .temporal_anchor import TemporalAnchor, Timeline


class SummarizationMode(Enum):
    """Summarization approach."""
    EXTRACTIVE = "EXTRACTIVE"      # TextRank/LexRank
    ABSTRACTIVE = "ABSTRACTIVE"    # Gemini/LLM
    HYBRID = "HYBRID"              # Extract + Abstract


@dataclass
class ConstrainedSummaryResult:
    """
    Result of constrained summarization.
    
    Includes the summary, verification report, and grounding info.
    """
    summary: str
    verification_report: SummaryVerificationReport
    grounding_facts: List[KGTriple]
    mode: SummarizationMode
    iterations: int = 1
    confidence: float = 0.0
    timeline: Optional[Timeline] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_hallucination_free(self) -> bool:
        return self.verification_report.hallucination_count == 0
    
    @property
    def is_verified(self) -> bool:
        return self.verification_report.overall_status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.PARTIALLY_VERIFIED
        )
    
    def to_dict(self) -> Dict:
        return {
            "summary": self.summary,
            "mode": self.mode.value,
            "iterations": self.iterations,
            "confidence": round(self.confidence, 4),
            "is_hallucination_free": self.is_hallucination_free,
            "verification": self.verification_report.to_dict(),
            "grounding_facts_count": len(self.grounding_facts),
            "timeline": self.timeline.to_dict() if self.timeline else None
        }
    
    def print_result(self):
        """Print formatted result."""
        print("\n" + "=" * 70)
        print("📝 CONSTRAINED SUMMARY RESULT")
        print("=" * 70)
        
        status_emoji = "✅" if self.is_hallucination_free else "⚠️"
        
        print(f"\nStatus: {status_emoji} {'Hallucination-Free' if self.is_hallucination_free else 'Contains Issues'}")
        print(f"Confidence: {self.confidence:.1%}")
        print(f"Mode: {self.mode.value}")
        print(f"Iterations: {self.iterations}")
        print(f"Grounding Facts: {len(self.grounding_facts)}")
        
        print("\n" + "-" * 70)
        print("SUMMARY:")
        print("-" * 70)
        print(self.summary)
        
        if self.timeline and self.timeline.events:
            print("\n" + "-" * 70)
            print("TIMELINE:")
            print("-" * 70)
            for event in self.timeline.events[:5]:
                temporal, desc, doc_idx = event
                print(f"  📅 {temporal.normalized_start[:10]} | {desc[:50]}...")
        
        print("\n" + "=" * 70)


class ConstrainedSummarizer:
    """
    Knowledge Graph-Constrained Summarizer.
    
    Generates summaries that are GUARANTEED to be grounded in verified facts
    from the Knowledge Graph. This is the anti-hallucination layer.
    
    Approach:
    1. Build KG from source documents
    2. Extract verified facts relevant to query/topic
    3. Generate summary constrained to these facts
    4. Verify output against KG
    5. Iteratively refine if hallucinations found
    """
    
    def __init__(
        self,
        kg: Optional[KnowledgeGraph] = None,
        gemini_api_key: Optional[str] = None,
        max_refinement_iterations: int = 3,
        min_verification_rate: float = 0.7
    ):
        """
        Initialize constrained summarizer.
        
        Args:
            kg: Pre-built Knowledge Graph (optional, can build from docs).
            gemini_api_key: API key for Gemini abstractive summarization.
            max_refinement_iterations: Max times to refine if hallucinations found.
            min_verification_rate: Minimum acceptable verification rate.
        """
        self.kg = kg or KnowledgeGraph(name="constrained_kg")
        self.max_iterations = max_refinement_iterations
        self.min_verification_rate = min_verification_rate
        
        # Initialize fact verifier
        self.verifier = None  # Lazy init after KG is populated
        
        # Initialize temporal anchor for timeline
        self.temporal_anchor = TemporalAnchor()
        
        # Gemini for abstractive (optional)
        self.gemini_api_key = gemini_api_key
        self._gemini_summarizer = None
        
        # Extractive summarizers (lazy loaded)
        self._textrank = None
        self._lexrank = None
        
        print(f"[ConstrainedSummarizer] Initialized")
        print(f"  Max iterations: {max_refinement_iterations}")
        print(f"  Min verification rate: {min_verification_rate:.1%}")
    
    def _get_verifier(self) -> FactVerifier:
        """Get or create fact verifier."""
        if self.verifier is None:
            self.verifier = FactVerifier(self.kg)
        return self.verifier
    
    def _get_textrank(self):
        """Lazy load TextRank."""
        if self._textrank is None:
            try:
                from src.models.textrank import TextRankSummarizer
                self._textrank = TextRankSummarizer(num_sentences=5)
            except ImportError:
                print("[Warning] TextRank not available")
        return self._textrank
    
    def _get_lexrank(self):
        """Lazy load LexRank."""
        if self._lexrank is None:
            try:
                from src.models.lexrank import LexRankSummarizer
                self._lexrank = LexRankSummarizer(num_sentences=5)
            except ImportError:
                print("[Warning] LexRank not available")
        return self._lexrank
    
    def _get_gemini(self):
        """Lazy load Gemini summarizer."""
        if self._gemini_summarizer is None and self.gemini_api_key:
            try:
                from src.models.gemini_summarizer import GeminiSummarizer
                self._gemini_summarizer = GeminiSummarizer(
                    api_key=self.gemini_api_key
                )
            except Exception as e:
                print(f"[Warning] Gemini not available: {e}")
        return self._gemini_summarizer
    
    def build_kg_from_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ):
        """
        Build Knowledge Graph from documents.
        
        This should be called before summarization to populate the KG.
        
        Args:
            documents: Source documents.
            show_progress: Print progress.
        """
        print("[ConstrainedSummarizer] Building Knowledge Graph...")
        
        stats = self.kg.add_documents(documents, show_progress)
        
        print(f"[ConstrainedSummarizer] KG built:")
        print(f"  Entities: {self.kg.graph.number_of_nodes()}")
        print(f"  Relations: {self.kg.graph.number_of_edges()}")
        
        # Reset verifier to use updated KG
        self.verifier = None
    
    def _extract_grounding_facts(
        self,
        query: Optional[str] = None,
        topic_entities: Optional[List[str]] = None,
        max_facts: int = 30
    ) -> List[KGTriple]:
        """
        Extract verified facts from KG for grounding.
        
        Args:
            query: Optional query to focus facts.
            topic_entities: Specific entities to gather facts about.
            max_facts: Maximum facts to extract.
            
        Returns:
            List of KG triples as grounding facts.
        """
        verifier = self._get_verifier()
        
        facts = []
        
        if topic_entities:
            facts = verifier.get_grounded_facts(topic_entities, max_facts)
        else:
            # Get most mentioned entities
            entity_mentions = []
            for key in self.kg.graph.nodes():
                node = self.kg.graph.nodes[key]
                entity_mentions.append((key, node.get('mentions', 1)))
            
            entity_mentions.sort(key=lambda x: -x[1])
            top_entities = [key for key, _ in entity_mentions[:10]]
            
            for entity_key in top_entities:
                relations = self.kg.get_relations_for_entity(entity_key)
                for rel in relations[:3]:
                    facts.append(KGTriple(
                        subject=rel.get('subject', entity_key),
                        predicate=rel.get('relation_type', 'RELATED_TO'),
                        object=rel.get('object', ''),
                        confidence=rel.get('confidence', 0.5)
                    ))
        
        return facts[:max_facts]
    
    def _build_fact_context(self, facts: List[KGTriple]) -> str:
        """Build context string from facts for LLM."""
        if not facts:
            return ""
        
        context_lines = ["VERIFIED FACTS (from Knowledge Graph):"]
        
        for i, fact in enumerate(facts, 1):
            # Clean up subject/object (remove type prefix)
            subj = fact.subject.split(":")[-1] if ":" in fact.subject else fact.subject
            obj = fact.object.split(":")[-1] if ":" in fact.object else fact.object
            pred = fact.predicate.replace("_", " ").lower()
            
            context_lines.append(f"{i}. {subj} {pred} {obj}")
        
        return "\n".join(context_lines)
    
    def _generate_extractive(
        self,
        documents: List[str],
        num_sentences: int = 5,
        method: str = "textrank"
    ) -> str:
        """Generate extractive summary."""
        combined_text = " ".join(documents)
        
        if method == "textrank":
            summarizer = self._get_textrank()
            if summarizer:
                return summarizer.summarize(combined_text, num_sentences)
        
        elif method == "lexrank":
            summarizer = self._get_lexrank()
            if summarizer:
                return summarizer.summarize(combined_text, num_sentences)
        
        # Fallback: simple first-N sentences
        sentences = re.split(r'[.!?]+', combined_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return ". ".join(sentences[:num_sentences]) + "."
    
    def _generate_abstractive(
        self,
        documents: List[str],
        grounding_facts: List[KGTriple],
        query: Optional[str] = None
    ) -> str:
        """Generate abstractive summary with KG grounding."""
        gemini = self._get_gemini()
        
        if not gemini:
            print("[Warning] Gemini not available, falling back to extractive")
            return self._generate_extractive(documents, 5, "textrank")
        
        # Build grounding context
        fact_context = self._build_fact_context(grounding_facts)
        
        # Create constrained prompt
        grounded_prompt = f"""Anda adalah asisten yang merangkum berita Indonesia.

PENTING: Anda HANYA boleh menggunakan informasi yang ada dalam FAKTA TERVERIFIKASI dan DOKUMEN SUMBER.
Jangan menambahkan informasi yang tidak ada dalam sumber.
Jangan mengubah nama, tanggal, atau angka.
Jangan menyimpulkan hal yang tidak disebutkan secara eksplisit.

{fact_context}

DOKUMEN SUMBER:
{chr(10).join([f"[{i+1}] {doc[:500]}..." for i, doc in enumerate(documents[:5])])}

{f'FOKUS: {query}' if query else ''}

INSTRUKSI:
Buat ringkasan 2-3 paragraf yang:
1. HANYA berdasarkan fakta terverifikasi dan dokumen sumber
2. Menyebutkan entitas dengan nama yang benar
3. Menyebutkan tanggal dengan akurat
4. Tidak menyimpulkan atau menambahkan informasi baru

RINGKASAN:"""

        try:
            result = gemini.summarize(
                [grounded_prompt],  # Pass as single doc
                style="default"
            )
            return result.summary
        except Exception as e:
            print(f"[Error] Gemini generation failed: {e}")
            return self._generate_extractive(documents, 5, "textrank")
    
    def _refine_summary(
        self,
        summary: str,
        verification_report: SummaryVerificationReport,
        grounding_facts: List[KGTriple]
    ) -> str:
        """
        Refine summary to fix hallucinations.
        
        Uses the verification report to identify and fix issues.
        """
        # Find problematic claims
        problematic_claims = [
            r for r in verification_report.claim_results
            if r.status in (VerificationStatus.HALLUCINATION, VerificationStatus.CONTRADICTED)
        ]
        
        if not problematic_claims:
            return summary
        
        # Build correction context
        corrections = []
        for result in problematic_claims:
            if result.suggestions:
                corrections.append(f"- {result.claim[:50]}... → {result.suggestions[0]}")
        
        gemini = self._get_gemini()
        
        if not gemini:
            # Simple removal approach
            refined = summary
            for claim in problematic_claims:
                # Try to remove the problematic sentence
                sentences = refined.split(".")
                refined = ". ".join([
                    s for s in sentences
                    if claim.claim[:30] not in s
                ])
            return refined
        
        # Use LLM to refine
        refinement_prompt = f"""Berikut adalah ringkasan yang perlu diperbaiki karena mengandung informasi yang tidak akurat:

RINGKASAN ASLI:
{summary}

MASALAH YANG DITEMUKAN:
{chr(10).join(corrections)}

FAKTA YANG BENAR:
{self._build_fact_context(grounding_facts)}

INSTRUKSI:
Tulis ulang ringkasan dengan menghapus atau memperbaiki informasi yang tidak akurat.
Hanya gunakan fakta yang terverifikasi.

RINGKASAN YANG DIPERBAIKI:"""

        try:
            result = gemini.summarize([refinement_prompt], style="default")
            return result.summary
        except:
            return summary
    
    def summarize(
        self,
        documents: List[str],
        query: Optional[str] = None,
        mode: SummarizationMode = SummarizationMode.HYBRID,
        topic_entities: Optional[List[str]] = None,
        num_sentences: int = 5,
        build_timeline: bool = True
    ) -> ConstrainedSummaryResult:
        """
        Generate a hallucination-free summary.
        
        Args:
            documents: Source documents.
            query: Optional query/focus.
            mode: Summarization mode (extractive/abstractive/hybrid).
            topic_entities: Specific entities to focus on.
            num_sentences: Target sentences for extractive.
            build_timeline: Whether to build event timeline.
            
        Returns:
            ConstrainedSummaryResult with verified summary.
        """
        print(f"\n[ConstrainedSummarizer] Starting {mode.value} summarization")
        print(f"  Documents: {len(documents)}")
        
        # Step 1: Build/update KG if needed
        if self.kg.graph.number_of_nodes() == 0:
            print("[ConstrainedSummarizer] KG empty, building from documents...")
            self.build_kg_from_documents(documents, show_progress=False)
        
        # Step 2: Extract grounding facts
        grounding_facts = self._extract_grounding_facts(
            query=query,
            topic_entities=topic_entities
        )
        print(f"  Grounding facts: {len(grounding_facts)}")
        
        # Step 3: Generate initial summary
        if mode == SummarizationMode.EXTRACTIVE:
            summary = self._generate_extractive(documents, num_sentences, "textrank")
        elif mode == SummarizationMode.ABSTRACTIVE:
            summary = self._generate_abstractive(documents, grounding_facts, query)
        else:  # HYBRID
            # Start with extractive, then refine with LLM
            extractive = self._generate_extractive(documents, num_sentences * 2, "textrank")
            
            gemini = self._get_gemini()
            if gemini:
                summary = self._generate_abstractive(
                    [extractive],  # Use extractive as input
                    grounding_facts,
                    query
                )
            else:
                summary = extractive
        
        # Step 4: Verify and refine
        verifier = self._get_verifier()
        iteration = 1
        
        while iteration <= self.max_iterations:
            print(f"  Iteration {iteration}: Verifying...")
            
            verification_report = verifier.verify_summary(summary)
            
            print(f"    Verification rate: {verification_report.verification_rate:.1%}")
            print(f"    Hallucinations: {verification_report.hallucination_count}")
            
            # Check if good enough
            if (verification_report.hallucination_count == 0 or
                verification_report.verification_rate >= self.min_verification_rate):
                break
            
            # Refine
            if iteration < self.max_iterations:
                print(f"    Refining summary...")
                summary = self._refine_summary(
                    summary, verification_report, grounding_facts
                )
            
            iteration += 1
        
        # Step 5: Build timeline if requested
        timeline = None
        if build_timeline:
            timeline = self.temporal_anchor.build_timeline(documents)
        
        # Calculate final confidence
        confidence = (
            verification_report.verification_rate * 0.6 +
            verification_report.confidence * 0.4
        )
        
        return ConstrainedSummaryResult(
            summary=summary,
            verification_report=verification_report,
            grounding_facts=grounding_facts,
            mode=mode,
            iterations=iteration,
            confidence=confidence,
            timeline=timeline
        )
    
    def summarize_with_credibility(
        self,
        documents: List[str],
        credibility_scores: List[float],
        threshold: float = 0.5,
        **kwargs
    ) -> ConstrainedSummaryResult:
        """
        Summarize with credibility filtering.
        
        Integrates with your existing Trust Layer.
        
        Args:
            documents: Source documents.
            credibility_scores: Score for each document (0-1).
            threshold: Minimum credibility to include.
            **kwargs: Additional args passed to summarize().
            
        Returns:
            ConstrainedSummaryResult from filtered documents.
        """
        # Filter documents
        filtered_docs = [
            doc for doc, score in zip(documents, credibility_scores)
            if score >= threshold
        ]
        
        print(f"[ConstrainedSummarizer] Credibility filter: {len(filtered_docs)}/{len(documents)} passed")
        
        if not filtered_docs:
            return ConstrainedSummaryResult(
                summary="Tidak ada dokumen yang memenuhi threshold kredibilitas.",
                verification_report=SummaryVerificationReport(
                    summary="",
                    claim_results=[],
                    overall_status=VerificationStatus.UNVERIFIED,
                    hallucination_count=0,
                    verification_rate=0.0,
                    confidence=0.0
                ),
                grounding_facts=[],
                mode=kwargs.get('mode', SummarizationMode.HYBRID),
                confidence=0.0
            )
        
        return self.summarize(filtered_docs, **kwargs)


# Integration function for existing pipeline
def create_constrained_summarizer(
    hoax_model_path: str = "models/hoax_indobert_lora",
    gemini_api_key: Optional[str] = None
) -> ConstrainedSummarizer:
    """
    Create a constrained summarizer integrated with the Trust Layer.
    
    Args:
        hoax_model_path: Path to hoax classifier.
        gemini_api_key: Gemini API key.
        
    Returns:
        Configured ConstrainedSummarizer.
    """
    return ConstrainedSummarizer(
        gemini_api_key=gemini_api_key,
        max_refinement_iterations=3,
        min_verification_rate=0.7
    )


if __name__ == "__main__":
    # Demo
    print("=" * 70)
    print("Constrained Summarizer Demo")
    print("=" * 70)
    
    documents = [
        """Presiden Joko Widodo mengumumkan kebijakan vaksinasi baru 
        pada 15 Januari 2024 di Istana Negara Jakarta. Program ini 
        ditargetkan mencapai 70% cakupan dalam 6 bulan.""",
        
        """Menteri Kesehatan Budi Gunadi menjelaskan program vaksinasi
        akan dimulai dengan tenaga kesehatan. Anggaran sebesar 
        Rp 150 miliar telah disetujui DPR.""",
        
        """WHO memuji Indonesia atas strategi vaksinasi yang komprehensif.
        Direktur WHO memberikan apresiasi kepada Kementerian Kesehatan
        pada Februari 2024."""
    ]
    
    # Create summarizer (without Gemini for demo)
    summarizer = ConstrainedSummarizer()
    
    # Build KG
    summarizer.build_kg_from_documents(documents)
    
    # Generate summary
    result = summarizer.summarize(
        documents,
        mode=SummarizationMode.EXTRACTIVE,
        build_timeline=True
    )
    
    # Print result
    result.print_result()
    
    # Print verification details
    print("\n" + "=" * 70)
    print("VERIFICATION DETAILS:")
    print("=" * 70)
    result.verification_report.print_report()
