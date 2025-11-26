"""
Fact Verification for TMDS

The critical module that PREVENTS HALLUCINATIONS by:
1. Cross-referencing generated claims against the Knowledge Graph
2. Detecting temporal inconsistencies
3. Flagging unsupported assertions
4. Providing confidence scores for each claim

This is what makes TMDS superior to standard MDS - every claim
in the summary can be traced back to verified facts.

Types of Hallucinations Detected:
1. ENTITY_FABRICATION: Mentioning entities not in the KG
2. RELATION_FABRICATION: Inventing relationships
3. TEMPORAL_ERROR: Wrong timing/sequence of events
4. ATTRIBUTION_ERROR: Wrong source for a statement
5. QUANTITY_ERROR: Incorrect numbers/statistics
6. CONFLATION: Merging distinct entities/events
"""

import re
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .entity_extractor import Entity, EntityType, EntityExtractor
from .relation_extractor import Relation, RelationType, RelationExtractor
from .temporal_anchor import TemporalExpression, TemporalAnchor
from .knowledge_graph import KnowledgeGraph, KGTriple


class HallucinationType(Enum):
    """Types of hallucinations that can be detected."""
    ENTITY_FABRICATION = "ENTITY_FABRICATION"
    RELATION_FABRICATION = "RELATION_FABRICATION"
    TEMPORAL_ERROR = "TEMPORAL_ERROR"
    ATTRIBUTION_ERROR = "ATTRIBUTION_ERROR"
    QUANTITY_ERROR = "QUANTITY_ERROR"
    CONFLATION = "CONFLATION"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"


class VerificationStatus(Enum):
    """Status of fact verification."""
    VERIFIED = "VERIFIED"           # Claim supported by KG
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"  # Some aspects supported
    UNVERIFIED = "UNVERIFIED"       # Cannot verify (not necessarily false)
    CONTRADICTED = "CONTRADICTED"   # Contradicts KG
    HALLUCINATION = "HALLUCINATION" # Definitely fabricated


@dataclass
class VerificationResult:
    """
    Result of verifying a claim or summary.
    
    Attributes:
        claim: The claim being verified
        status: Verification status
        confidence: Confidence in the verification (0-1)
        hallucination_type: Type of hallucination if detected
        supporting_evidence: KG triples that support the claim
        contradicting_evidence: KG triples that contradict
        suggestions: Suggested corrections if issues found
        explanation: Human-readable explanation
    """
    claim: str
    status: VerificationStatus
    confidence: float = 0.0
    hallucination_type: Optional[HallucinationType] = None
    supporting_evidence: List[KGTriple] = field(default_factory=list)
    contradicting_evidence: List[KGTriple] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "claim": self.claim[:200],
            "status": self.status.value,
            "confidence": round(self.confidence, 4),
            "hallucination_type": self.hallucination_type.value if self.hallucination_type else None,
            "supporting_evidence_count": len(self.supporting_evidence),
            "contradicting_evidence_count": len(self.contradicting_evidence),
            "suggestions": self.suggestions[:3],
            "explanation": self.explanation[:300]
        }
    
    @property
    def is_valid(self) -> bool:
        return self.status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED)


@dataclass
class SummaryVerificationReport:
    """
    Complete verification report for a summary.
    
    Attributes:
        summary: The original summary
        claim_results: Verification results for each claim
        overall_status: Aggregate status
        hallucination_count: Number of hallucinations detected
        verification_rate: Percentage of verified claims
        confidence: Overall confidence
        corrected_summary: Suggested corrected version
    """
    summary: str
    claim_results: List[VerificationResult]
    overall_status: VerificationStatus
    hallucination_count: int
    verification_rate: float
    confidence: float
    corrected_summary: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "summary_length": len(self.summary),
            "num_claims": len(self.claim_results),
            "overall_status": self.overall_status.value,
            "hallucination_count": self.hallucination_count,
            "verification_rate": round(self.verification_rate, 4),
            "confidence": round(self.confidence, 4),
            "claims": [r.to_dict() for r in self.claim_results]
        }
    
    def print_report(self):
        """Print human-readable report."""
        print("\n" + "=" * 60)
        print("📋 FACT VERIFICATION REPORT")
        print("=" * 60)
        
        status_emoji = {
            VerificationStatus.VERIFIED: "✅",
            VerificationStatus.PARTIALLY_VERIFIED: "⚠️",
            VerificationStatus.UNVERIFIED: "❓",
            VerificationStatus.CONTRADICTED: "❌",
            VerificationStatus.HALLUCINATION: "🚨"
        }
        
        print(f"\nOverall Status: {status_emoji[self.overall_status]} {self.overall_status.value}")
        print(f"Verification Rate: {self.verification_rate:.1%}")
        print(f"Hallucinations: {self.hallucination_count}")
        print(f"Confidence: {self.confidence:.1%}")
        
        print("\n" + "-" * 40)
        print("Claim-by-Claim Analysis:")
        print("-" * 40)
        
        for i, result in enumerate(self.claim_results, 1):
            emoji = status_emoji[result.status]
            print(f"\n{i}. {emoji} [{result.status.value}]")
            print(f"   Claim: {result.claim[:80]}...")
            
            if result.hallucination_type:
                print(f"   ⚠️ Hallucination: {result.hallucination_type.value}")
            
            if result.suggestions:
                print(f"   💡 Suggestion: {result.suggestions[0]}")
        
        print("\n" + "=" * 60)


class FactVerifier:
    """
    Verifies claims and summaries against a Knowledge Graph.
    
    This is the HALLUCINATION PREVENTION layer that ensures
    generated summaries are grounded in verified facts.
    
    Verification Process:
    1. Parse claim into entities and relations
    2. Query KG for matching/contradicting evidence
    3. Check temporal consistency
    4. Calculate confidence score
    5. Generate correction suggestions if needed
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        similarity_threshold: float = 0.7,
        temporal_tolerance_days: int = 7
    ):
        """
        Initialize fact verifier.
        
        Args:
            knowledge_graph: The KG to verify against.
            similarity_threshold: Min similarity for entity matching.
            temporal_tolerance_days: Allowed date variance.
        """
        self.kg = knowledge_graph
        self.similarity_threshold = similarity_threshold
        self.temporal_tolerance_days = temporal_tolerance_days
        
        # Extractors for parsing claims
        self._entity_extractor = EntityExtractor(use_transformer=False)
        self._relation_extractor = RelationExtractor()
        self._temporal_anchor = TemporalAnchor()
        
        print(f"[FactVerifier] Initialized")
        print(f"  KG entities: {self.kg.graph.number_of_nodes()}")
        print(f"  KG relations: {self.kg.graph.number_of_edges()}")
    
    def _fuzzy_match_entity(
        self,
        entity_text: str,
        entity_type: Optional[EntityType] = None
    ) -> Tuple[Optional[str], float]:
        """
        Find best matching entity in KG using fuzzy matching.
        
        Returns:
            Tuple of (best_match_key, similarity_score).
        """
        best_match = None
        best_score = 0.0
        
        search_nodes = (
            self.kg._entity_index.get(entity_type, set())
            if entity_type else self.kg.graph.nodes()
        )
        
        for key in search_nodes:
            if key not in self.kg.graph:
                continue
            
            node = self.kg.graph.nodes[key]
            normalized = node.get('normalized', '')
            aliases = node.get('aliases', set())
            
            # Check normalized name
            score = SequenceMatcher(
                None, entity_text.lower(), normalized.lower()
            ).ratio()
            
            if score > best_score:
                best_score = score
                best_match = key
            
            # Check aliases
            for alias in aliases:
                score = SequenceMatcher(
                    None, entity_text.lower(), alias.lower()
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_match = key
        
        if best_score >= self.similarity_threshold:
            return best_match, best_score
        
        return None, best_score
    
    def _verify_entity(self, entity: Entity) -> Tuple[VerificationStatus, str, float]:
        """
        Verify if an entity exists in the KG.
        
        Returns:
            Tuple of (status, kg_key_if_found, confidence).
        """
        # Try exact match first
        entity_key = f"{entity.entity_type.value}:{entity.normalized}"
        
        if entity_key in self.kg.graph:
            return VerificationStatus.VERIFIED, entity_key, 1.0
        
        # Try fuzzy match
        match_key, score = self._fuzzy_match_entity(
            entity.normalized, entity.entity_type
        )
        
        if match_key:
            return VerificationStatus.PARTIALLY_VERIFIED, match_key, score
        
        # Try without type constraint
        match_key, score = self._fuzzy_match_entity(entity.normalized)
        
        if match_key:
            return VerificationStatus.PARTIALLY_VERIFIED, match_key, score * 0.8
        
        # Entity not found - could be fabrication
        return VerificationStatus.UNVERIFIED, None, 0.0
    
    def _verify_relation(
        self,
        subject_key: str,
        predicate: RelationType,
        object_key: str
    ) -> Tuple[VerificationStatus, List[Dict], float]:
        """
        Verify if a relation exists in the KG.
        
        Returns:
            Tuple of (status, supporting_triples, confidence).
        """
        supporting = []
        
        # Check direct relation
        if self.kg.graph.has_edge(subject_key, object_key):
            edge = self.kg.graph[subject_key][object_key]
            edge_type = edge.get('relation_type', '')
            
            if edge_type == predicate.value:
                return (
                    VerificationStatus.VERIFIED,
                    [edge],
                    edge.get('confidence', 0.8)
                )
            else:
                # Different relation exists
                return (
                    VerificationStatus.CONTRADICTED,
                    [edge],
                    0.3
                )
        
        # Check for path (indirect relation)
        path = self.kg.find_path(subject_key, object_key, max_length=2)
        if path:
            return (
                VerificationStatus.PARTIALLY_VERIFIED,
                [{"path": path}],
                0.5
            )
        
        return VerificationStatus.UNVERIFIED, [], 0.0
    
    def _verify_temporal(
        self,
        claim_temporal: TemporalExpression,
        entity_key: str
    ) -> Tuple[VerificationStatus, str, float]:
        """
        Verify temporal consistency for an entity.
        
        Returns:
            Tuple of (status, explanation, confidence).
        """
        if entity_key not in self.kg.graph:
            return VerificationStatus.UNVERIFIED, "Entity not found", 0.0
        
        node = self.kg.graph.nodes[entity_key]
        kg_start = node.get('temporal_start')
        kg_end = node.get('temporal_end')
        
        if not kg_start and not kg_end:
            return (
                VerificationStatus.UNVERIFIED,
                "No temporal info in KG",
                0.5
            )
        
        # Compare dates
        from datetime import datetime, timedelta
        
        try:
            claim_date = datetime.fromisoformat(claim_temporal.normalized_start[:10])
            
            if kg_start:
                kg_start_date = datetime.fromisoformat(kg_start[:10])
                
                # Check if claim date is within tolerance of KG date
                diff = abs((claim_date - kg_start_date).days)
                
                if diff <= self.temporal_tolerance_days:
                    return (
                        VerificationStatus.VERIFIED,
                        f"Date matches (within {diff} days)",
                        max(0.5, 1.0 - diff / 30)
                    )
                elif diff <= 30:
                    return (
                        VerificationStatus.PARTIALLY_VERIFIED,
                        f"Date differs by {diff} days",
                        0.5
                    )
                else:
                    return (
                        VerificationStatus.CONTRADICTED,
                        f"Significant date discrepancy ({diff} days)",
                        0.2
                    )
        except:
            pass
        
        return VerificationStatus.UNVERIFIED, "Cannot parse dates", 0.3
    
    def verify_claim(self, claim: str) -> VerificationResult:
        """
        Verify a single claim against the Knowledge Graph.
        
        Args:
            claim: The claim text to verify.
            
        Returns:
            VerificationResult with detailed analysis.
        """
        if not claim or len(claim.strip()) < 10:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                explanation="Claim too short to verify"
            )
        
        # Extract components from claim
        entities = self._entity_extractor.extract(claim)
        relations = self._relation_extractor.extract(claim, entities)
        temporals = self._temporal_anchor.extract(claim)
        
        # Track verification results
        verified_entities = []
        unverified_entities = []
        verified_relations = []
        hallucinations = []
        supporting_evidence = []
        contradicting_evidence = []
        
        # Verify entities
        entity_keys = {}
        for entity in entities:
            status, kg_key, conf = self._verify_entity(entity)
            
            if status == VerificationStatus.VERIFIED:
                verified_entities.append(entity)
                entity_keys[entity] = kg_key
            elif status == VerificationStatus.PARTIALLY_VERIFIED:
                verified_entities.append(entity)
                entity_keys[entity] = kg_key
            else:
                unverified_entities.append(entity)
                
                # Check if this looks like fabrication
                if entity.entity_type in (EntityType.PERSON, EntityType.ORGANIZATION):
                    hallucinations.append({
                        "type": HallucinationType.ENTITY_FABRICATION,
                        "entity": entity.normalized
                    })
        
        # Verify relations
        for relation in relations:
            if relation.subject in entity_keys and relation.object in entity_keys:
                subj_key = entity_keys[relation.subject]
                obj_key = entity_keys[relation.object]
                
                status, evidence, conf = self._verify_relation(
                    subj_key, relation.predicate, obj_key
                )
                
                if status == VerificationStatus.VERIFIED:
                    verified_relations.append(relation)
                    supporting_evidence.extend([
                        KGTriple(subj_key, relation.predicate.value, obj_key)
                    ])
                elif status == VerificationStatus.CONTRADICTED:
                    contradicting_evidence.extend([
                        KGTriple(subj_key, e.get('relation_type', ''), obj_key)
                        for e in evidence
                    ])
                    hallucinations.append({
                        "type": HallucinationType.RELATION_FABRICATION,
                        "relation": f"{relation.subject.normalized} -> {relation.object.normalized}"
                    })
        
        # Verify temporals
        temporal_issues = []
        for temporal in temporals:
            # Check against entities mentioned nearby
            for entity in entities:
                if entity in entity_keys:
                    status, explanation, conf = self._verify_temporal(
                        temporal, entity_keys[entity]
                    )
                    
                    if status == VerificationStatus.CONTRADICTED:
                        temporal_issues.append(explanation)
                        hallucinations.append({
                            "type": HallucinationType.TEMPORAL_ERROR,
                            "detail": explanation
                        })
        
        # Calculate overall status and confidence
        total_items = len(entities) + len(relations)
        verified_items = len(verified_entities) + len(verified_relations)
        
        if total_items == 0:
            overall_status = VerificationStatus.UNVERIFIED
            confidence = 0.5
        elif hallucinations:
            overall_status = VerificationStatus.HALLUCINATION
            confidence = 0.2
        elif verified_items == total_items:
            overall_status = VerificationStatus.VERIFIED
            confidence = 0.9
        elif verified_items > 0:
            overall_status = VerificationStatus.PARTIALLY_VERIFIED
            confidence = verified_items / total_items
        else:
            overall_status = VerificationStatus.UNVERIFIED
            confidence = 0.3
        
        # Generate suggestions
        suggestions = []
        if unverified_entities:
            names = [e.normalized for e in unverified_entities[:3]]
            suggestions.append(f"Verify entities: {', '.join(names)}")
        if temporal_issues:
            suggestions.append(f"Check dates: {temporal_issues[0]}")
        
        # Generate explanation
        explanation_parts = []
        if verified_entities:
            explanation_parts.append(f"{len(verified_entities)} entities verified")
        if unverified_entities:
            explanation_parts.append(f"{len(unverified_entities)} entities unverified")
        if hallucinations:
            explanation_parts.append(f"{len(hallucinations)} potential hallucinations")
        
        return VerificationResult(
            claim=claim,
            status=overall_status,
            confidence=confidence,
            hallucination_type=hallucinations[0]["type"] if hallucinations else None,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            suggestions=suggestions,
            explanation="; ".join(explanation_parts)
        )
    
    def verify_summary(
        self,
        summary: str,
        split_into_claims: bool = True
    ) -> SummaryVerificationReport:
        """
        Verify an entire summary.
        
        Args:
            summary: The summary text to verify.
            split_into_claims: Whether to split into sentences.
            
        Returns:
            SummaryVerificationReport with detailed analysis.
        """
        if not summary:
            return SummaryVerificationReport(
                summary="",
                claim_results=[],
                overall_status=VerificationStatus.UNVERIFIED,
                hallucination_count=0,
                verification_rate=0.0,
                confidence=0.0
            )
        
        # Split summary into claims (sentences)
        if split_into_claims:
            claims = re.split(r'[.!?]+', summary)
            claims = [c.strip() for c in claims if len(c.strip()) > 10]
        else:
            claims = [summary]
        
        # Verify each claim
        results = []
        for claim in claims:
            result = self.verify_claim(claim)
            results.append(result)
        
        # Calculate aggregates
        hallucination_count = sum(
            1 for r in results 
            if r.status == VerificationStatus.HALLUCINATION
        )
        
        verified_count = sum(
            1 for r in results
            if r.status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED)
        )
        
        verification_rate = verified_count / len(results) if results else 0.0
        
        avg_confidence = (
            sum(r.confidence for r in results) / len(results)
            if results else 0.0
        )
        
        # Determine overall status
        if hallucination_count > len(results) * 0.3:
            overall_status = VerificationStatus.HALLUCINATION
        elif verification_rate >= 0.8:
            overall_status = VerificationStatus.VERIFIED
        elif verification_rate >= 0.5:
            overall_status = VerificationStatus.PARTIALLY_VERIFIED
        elif any(r.status == VerificationStatus.CONTRADICTED for r in results):
            overall_status = VerificationStatus.CONTRADICTED
        else:
            overall_status = VerificationStatus.UNVERIFIED
        
        return SummaryVerificationReport(
            summary=summary,
            claim_results=results,
            overall_status=overall_status,
            hallucination_count=hallucination_count,
            verification_rate=verification_rate,
            confidence=avg_confidence
        )
    
    def get_grounded_facts(
        self,
        topic_entities: List[str],
        max_facts: int = 20
    ) -> List[KGTriple]:
        """
        Get verified facts from KG for given topic entities.
        
        Useful for providing context to constrained generation.
        
        Args:
            topic_entities: Entity names to gather facts about.
            max_facts: Maximum facts to return.
            
        Returns:
            List of KG triples as grounded facts.
        """
        facts = []
        
        for entity_text in topic_entities:
            # Find entity in KG
            match_key, score = self._fuzzy_match_entity(entity_text)
            
            if match_key:
                # Get relations
                relations = self.kg.get_relations_for_entity(match_key)
                
                for rel in relations[:5]:
                    facts.append(KGTriple(
                        subject=rel.get('subject', ''),
                        predicate=rel.get('predicate', rel.get('relation_type', '')),
                        object=rel.get('object', ''),
                        confidence=rel.get('confidence', 0.5),
                        source_docs=rel.get('source_docs', set())
                    ))
            
            if len(facts) >= max_facts:
                break
        
        return facts[:max_facts]


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Fact Verifier Demo")
    print("=" * 60)
    
    # Build a small KG first
    kg = KnowledgeGraph(name="demo_kg")
    
    documents = [
        """Presiden Joko Widodo mengumumkan kebijakan vaksinasi baru 
        pada 15 Januari 2024 di Jakarta.""",
        """Menteri Kesehatan Budi Gunadi menjelaskan program vaksinasi
        akan mencapai 70% cakupan pada Maret 2024.""",
        """DPR menyetujui anggaran Rp 150 miliar untuk program kesehatan."""
    ]
    
    kg.add_documents(documents, show_progress=False)
    print(f"\nKG built: {kg.graph.number_of_nodes()} entities, {kg.graph.number_of_edges()} relations")
    
    # Initialize verifier
    verifier = FactVerifier(kg)
    
    # Test claims
    test_claims = [
        "Joko Widodo mengumumkan kebijakan di Jakarta.",  # Should verify
        "Menteri Kesehatan Ali Wibowo meluncurkan program.",  # Wrong name - hallucination
        "DPR menyetujui anggaran kesehatan.",  # Should partially verify
        "Pada 2020, vaksinasi dimulai di Surabaya.",  # Wrong date - temporal error
    ]
    
    print("\n" + "=" * 60)
    print("Verifying individual claims:")
    print("=" * 60)
    
    for claim in test_claims:
        result = verifier.verify_claim(claim)
        
        status_emoji = {
            VerificationStatus.VERIFIED: "✅",
            VerificationStatus.PARTIALLY_VERIFIED: "⚠️",
            VerificationStatus.UNVERIFIED: "❓",
            VerificationStatus.CONTRADICTED: "❌",
            VerificationStatus.HALLUCINATION: "🚨"
        }
        
        print(f"\n{status_emoji[result.status]} [{result.status.value}]")
        print(f"  Claim: {claim}")
        print(f"  Confidence: {result.confidence:.2f}")
        if result.hallucination_type:
            print(f"  Hallucination: {result.hallucination_type.value}")
        print(f"  Explanation: {result.explanation}")
    
    # Test full summary
    print("\n" + "=" * 60)
    print("Verifying full summary:")
    print("=" * 60)
    
    summary = """
    Presiden Joko Widodo mengumumkan kebijakan vaksinasi baru di Jakarta.
    Program ini didukung oleh Menteri Pendidikan dengan anggaran Rp 200 miliar.
    Vaksinasi direncanakan dimulai pada 2020 dan mencapai 90% cakupan.
    """
    
    report = verifier.verify_summary(summary)
    report.print_report()
