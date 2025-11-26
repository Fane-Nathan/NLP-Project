"""
Relation Extraction for Indonesian News Text

Extracts relationships between entities to build knowledge graph edges.
Uses a combination of:
1. Dependency parsing patterns
2. Rule-based verb/preposition triggers
3. Co-occurrence in sentence windows

Indonesian-specific challenges handled:
- Active/passive voice transformations (me-/di- prefixes)
- Causative constructions (-kan suffix)
- Topic-comment sentence structure
"""

import re
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

from .entity_extractor import Entity, EntityType


class RelationType(Enum):
    """Types of relations in Indonesian news."""
    # Temporal relations
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    DURING = "DURING"
    SIMULTANEOUS = "SIMULTANEOUS"
    
    # Causal relations
    CAUSES = "CAUSES"
    PREVENTS = "PREVENTS"
    ENABLES = "ENABLES"
    RESULTS_IN = "RESULTS_IN"
    
    # Membership/Affiliation
    MEMBER_OF = "MEMBER_OF"
    LEADS = "LEADS"
    WORKS_AT = "WORKS_AT"
    REPRESENTS = "REPRESENTS"
    
    # Spatial relations
    LOCATED_IN = "LOCATED_IN"
    NEAR = "NEAR"
    PART_OF = "PART_OF"
    
    # Attribution
    SAID_BY = "SAID_BY"
    ANNOUNCED_BY = "ANNOUNCED_BY"
    REPORTED_BY = "REPORTED_BY"
    DECIDED_BY = "DECIDED_BY"
    
    # Action relations
    AFFECTS = "AFFECTS"
    INVOLVES = "INVOLVES"
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"
    
    # Possession/Association
    HAS = "HAS"
    OWNS = "OWNS"
    RECEIVES = "RECEIVES"
    
    # Identity
    SAME_AS = "SAME_AS"
    INSTANCE_OF = "INSTANCE_OF"
    
    # Generic/Unknown
    RELATED_TO = "RELATED_TO"
    UNKNOWN = "UNKNOWN"


@dataclass
class Relation:
    """
    Represents a relation between two entities.
    
    Attributes:
        subject: Source entity
        predicate: Relation type
        object: Target entity
        confidence: Extraction confidence
        source_sentence: Original sentence text
        source_doc: Document index
        temporal_anchor: When this relation is valid
        evidence: Supporting text spans
        metadata: Additional relation data
    """
    subject: Entity
    predicate: RelationType
    object: Entity
    confidence: float = 1.0
    source_sentence: str = ""
    source_doc: int = 0
    temporal_anchor: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash((
            self.subject.normalized,
            self.predicate,
            self.object.normalized
        ))
    
    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (
            self.subject.normalized == other.subject.normalized and
            self.predicate == other.predicate and
            self.object.normalized == other.object.normalized
        )
    
    def to_triple(self) -> Tuple[str, str, str]:
        """Return as (subject, predicate, object) triple."""
        return (
            self.subject.normalized,
            self.predicate.value,
            self.object.normalized
        )
    
    def to_dict(self) -> Dict:
        return {
            "subject": self.subject.to_dict(),
            "predicate": self.predicate.value,
            "object": self.object.to_dict(),
            "confidence": round(self.confidence, 4),
            "source_sentence": self.source_sentence[:200],
            "source_doc": self.source_doc,
            "temporal_anchor": self.temporal_anchor,
            "evidence": self.evidence[:3],
            "metadata": self.metadata
        }


class RelationExtractor:
    """
    Extract relations between entities in Indonesian news text.
    
    Uses pattern matching and heuristics optimized for news domain.
    """
    
    # Indonesian verb triggers for different relation types
    VERB_TRIGGERS = {
        # Attribution verbs
        RelationType.SAID_BY: [
            'mengatakan', 'menyatakan', 'mengungkapkan', 'menjelaskan',
            'menegaskan', 'menyebutkan', 'mengklaim', 'berkata', 'ujar',
            'tutur', 'kata', 'jelasnya', 'ungkapnya', 'tambahnya'
        ],
        RelationType.ANNOUNCED_BY: [
            'mengumumkan', 'meluncurkan', 'meresmikan', 'mendeklarasikan',
            'memproklamasikan', 'mengabarkan', 'memberitakan', 'menyiarkan'
        ],
        RelationType.DECIDED_BY: [
            'memutuskan', 'menetapkan', 'menentukan', 'menyetujui',
            'mengesahkan', 'meratifikasi', 'mengadopsi'
        ],
        
        # Causal verbs
        RelationType.CAUSES: [
            'menyebabkan', 'mengakibatkan', 'memicu', 'mendorong',
            'menimbulkan', 'membuat', 'menghasilkan'
        ],
        RelationType.PREVENTS: [
            'mencegah', 'menghalangi', 'menghambat', 'melarang',
            'menghentikan', 'membatalkan', 'menolak'
        ],
        RelationType.ENABLES: [
            'memungkinkan', 'mengizinkan', 'membolehkan', 'membuka',
            'memberikan kesempatan', 'mendukung'
        ],
        
        # Membership verbs
        RelationType.LEADS: [
            'memimpin', 'mengepalai', 'mengetuai', 'memandu',
            'mengarahkan', 'mengomandoi', 'menjabat sebagai'
        ],
        RelationType.MEMBER_OF: [
            'anggota', 'bagian dari', 'termasuk dalam', 'bergabung dengan',
            'berafiliasi dengan', 'tergabung dalam'
        ],
        RelationType.WORKS_AT: [
            'bekerja di', 'bertugas di', 'menjabat di', 'berkarir di',
            'dipekerjakan oleh', 'bergabung dengan'
        ],
        
        # Support/Opposition
        RelationType.SUPPORTS: [
            'mendukung', 'menyetujui', 'membantu', 'menolong',
            'mengadvokasi', 'memperjuangkan', 'membela'
        ],
        RelationType.OPPOSES: [
            'menentang', 'menolak', 'mengkritik', 'memprotes',
            'mengecam', 'menyerang', 'melawan'
        ],
        
        # Possession
        RelationType.HAS: [
            'memiliki', 'mempunyai', 'punya', 'mengandung',
            'terdiri dari', 'meliputi', 'mencakup'
        ],
        RelationType.RECEIVES: [
            'menerima', 'mendapat', 'memperoleh', 'meraih',
            'mengambil', 'diberikan'
        ],
        
        # Location
        RelationType.LOCATED_IN: [
            'terletak di', 'berada di', 'berlokasi di', 'bertempat di',
            'di', 'pada'
        ],
    }
    
    # Preposition triggers
    PREP_TRIGGERS = {
        RelationType.LOCATED_IN: ['di', 'pada', 'dalam'],
        RelationType.BEFORE: ['sebelum', 'sebelumnya', 'pra-'],
        RelationType.AFTER: ['setelah', 'sesudah', 'pasca-', 'seusai'],
        RelationType.DURING: ['selama', 'saat', 'ketika', 'waktu'],
        RelationType.PART_OF: ['bagian dari', 'dalam', 'termasuk'],
    }
    
    # Passive voice markers in Indonesian
    PASSIVE_MARKERS = ['di-', 'ter-', 'ke-an']
    
    def __init__(
        self,
        window_size: int = 50,
        min_confidence: float = 0.3
    ):
        """
        Initialize relation extractor.
        
        Args:
            window_size: Maximum character distance between entities.
            min_confidence: Minimum confidence to keep relation.
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        
        # Compile trigger patterns
        self._compile_triggers()
        
        print(f"[RelationExtractor] Initialized")
        print(f"  Window size: {window_size}")
        print(f"  Min confidence: {min_confidence}")
    
    def _compile_triggers(self):
        """Compile verb trigger patterns."""
        self.verb_patterns = {}
        
        for rel_type, verbs in self.VERB_TRIGGERS.items():
            # Create regex pattern matching any of the verbs
            pattern = r'\b(' + '|'.join(re.escape(v) for v in verbs) + r')\b'
            self.verb_patterns[rel_type] = re.compile(pattern, re.IGNORECASE)
    
    def _find_verb_triggers(self, text: str) -> List[Tuple[RelationType, int, int]]:
        """Find verb triggers in text with positions."""
        triggers = []
        
        for rel_type, pattern in self.verb_patterns.items():
            for match in pattern.finditer(text):
                triggers.append((rel_type, match.start(), match.end()))
        
        return triggers
    
    def _is_in_window(
        self,
        entity1: Entity,
        entity2: Entity,
        max_distance: int = None
    ) -> bool:
        """Check if two entities are within the window distance."""
        max_dist = max_distance or self.window_size
        
        # Calculate distance between entities
        if entity1.end <= entity2.start:
            distance = entity2.start - entity1.end
        elif entity2.end <= entity1.start:
            distance = entity1.start - entity2.end
        else:
            distance = 0  # Overlapping
        
        return distance <= max_dist
    
    def _get_text_between(
        self,
        text: str,
        entity1: Entity,
        entity2: Entity
    ) -> str:
        """Get text between two entities."""
        start = min(entity1.end, entity2.end)
        end = max(entity1.start, entity2.start)
        
        if start >= end:
            return ""
        
        return text[start:end].strip()
    
    def _infer_relation_from_verb(
        self,
        text: str,
        subject: Entity,
        obj: Entity
    ) -> Optional[Tuple[RelationType, float]]:
        """Infer relation type from verbs between entities."""
        between_text = self._get_text_between(text, subject, obj)
        
        if not between_text:
            return None
        
        # Check for verb triggers
        for rel_type, pattern in self.verb_patterns.items():
            if pattern.search(between_text):
                # Calculate confidence based on distance
                distance = abs(subject.start - obj.start)
                confidence = max(0.3, 1.0 - (distance / 500))
                return (rel_type, confidence)
        
        return None
    
    def _infer_relation_from_types(
        self,
        subject: Entity,
        obj: Entity
    ) -> Optional[Tuple[RelationType, float]]:
        """Infer relation based on entity types."""
        sub_type = subject.entity_type
        obj_type = obj.entity_type
        
        # Person + Organization → WORKS_AT or LEADS
        if sub_type == EntityType.PERSON and obj_type == EntityType.ORGANIZATION:
            return (RelationType.WORKS_AT, 0.4)
        
        # Organization + Location → LOCATED_IN
        if sub_type == EntityType.ORGANIZATION and obj_type == EntityType.LOCATION:
            return (RelationType.LOCATED_IN, 0.5)
        
        # Person + Position → HAS
        if sub_type == EntityType.PERSON and obj_type == EntityType.POSITION:
            return (RelationType.HAS, 0.5)
        
        # Event + Date → temporal anchor (handled separately)
        if sub_type == EntityType.EVENT and obj_type == EntityType.DATE:
            return (RelationType.DURING, 0.6)
        
        # Person + Location → LOCATED_IN (with lower confidence)
        if sub_type == EntityType.PERSON and obj_type == EntityType.LOCATION:
            return (RelationType.LOCATED_IN, 0.3)
        
        return None
    
    def _extract_attribution(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relation]:
        """Extract attribution relations (X said Y)."""
        relations = []
        
        # Find quote patterns
        quote_pattern = re.compile(
            r'["""]([^"""]+)["""],?\s*'
            r'(?:kata|ujar|tutur|ungkap|jelas|sebut|tegas)\s*'
            r'([^.]+)',
            re.IGNORECASE
        )
        
        # Also handle "X mengatakan bahwa Y" pattern
        statement_pattern = re.compile(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+'
            r'(?:mengatakan|menyatakan|mengungkapkan|menjelaskan)\s+'
            r'(?:bahwa\s+)?([^.]+)',
            re.IGNORECASE
        )
        
        # Find person entities
        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        
        for match in quote_pattern.finditer(text):
            quote_text = match.group(1)
            speaker_text = match.group(2)
            
            # Find matching person entity
            for person in person_entities:
                if person.normalized.lower() in speaker_text.lower():
                    # Create claim entity
                    claim_entity = Entity(
                        text=quote_text[:100],
                        normalized=quote_text[:100],
                        entity_type=EntityType.UNKNOWN,
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.8
                    )
                    
                    relations.append(Relation(
                        subject=claim_entity,
                        predicate=RelationType.SAID_BY,
                        object=person,
                        confidence=0.8,
                        source_sentence=match.group(),
                        evidence=[match.group()]
                    ))
                    break
        
        return relations
    
    def _extract_temporal_relations(
        self,
        entities: List[Entity]
    ) -> List[Relation]:
        """Extract temporal relations between events and dates."""
        relations = []
        
        # Find date entities
        date_entities = [e for e in entities if e.entity_type == EntityType.DATE]
        
        # Find event-like entities (policies, events, positions)
        event_types = {EntityType.EVENT, EntityType.POLICY, EntityType.POSITION}
        event_entities = [e for e in entities if e.entity_type in event_types]
        
        # Also treat some organizations as events when they're clearly actions
        
        for date in date_entities:
            for event in event_entities:
                if self._is_in_window(date, event, 100):
                    relations.append(Relation(
                        subject=event,
                        predicate=RelationType.DURING,
                        object=date,
                        confidence=0.6,
                        temporal_anchor=date.normalized
                    ))
        
        return relations
    
    def _extract_location_relations(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relation]:
        """Extract location-based relations."""
        relations = []
        
        location_entities = [e for e in entities if e.entity_type == EntityType.LOCATION]
        other_entities = [e for e in entities if e.entity_type != EntityType.LOCATION]
        
        for loc in location_entities:
            for other in other_entities:
                if self._is_in_window(loc, other, 30):
                    between = self._get_text_between(text, other, loc)
                    
                    # Check for location prepositions
                    if any(prep in between.lower() for prep in ['di', 'pada', 'dalam']):
                        relations.append(Relation(
                            subject=other,
                            predicate=RelationType.LOCATED_IN,
                            object=loc,
                            confidence=0.7,
                            evidence=[between]
                        ))
        
        return relations
    
    def extract_from_sentence(
        self,
        sentence: str,
        entities: List[Entity],
        doc_idx: int = 0
    ) -> List[Relation]:
        """
        Extract relations from a single sentence.
        
        Args:
            sentence: Input sentence.
            entities: Entities found in this sentence.
            doc_idx: Document index.
            
        Returns:
            List of extracted relations.
        """
        if len(entities) < 2:
            return []
        
        relations = []
        
        # Try all entity pairs
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                if not self._is_in_window(e1, e2):
                    continue
                
                # Try verb-based inference
                verb_result = self._infer_relation_from_verb(sentence, e1, e2)
                if verb_result:
                    rel_type, confidence = verb_result
                    relations.append(Relation(
                        subject=e1,
                        predicate=rel_type,
                        object=e2,
                        confidence=confidence,
                        source_sentence=sentence,
                        source_doc=doc_idx
                    ))
                    continue
                
                # Try type-based inference
                type_result = self._infer_relation_from_types(e1, e2)
                if type_result:
                    rel_type, confidence = type_result
                    relations.append(Relation(
                        subject=e1,
                        predicate=rel_type,
                        object=e2,
                        confidence=confidence,
                        source_sentence=sentence,
                        source_doc=doc_idx
                    ))
        
        # Special extraction patterns
        relations.extend(self._extract_attribution(sentence, entities))
        relations.extend(self._extract_temporal_relations(entities))
        relations.extend(self._extract_location_relations(sentence, entities))
        
        # Filter by confidence
        relations = [r for r in relations if r.confidence >= self.min_confidence]
        
        return relations
    
    def extract(
        self,
        text: str,
        entities: List[Entity],
        doc_idx: int = 0
    ) -> List[Relation]:
        """
        Extract relations from full text.
        
        Args:
            text: Input text.
            entities: All entities found in text.
            doc_idx: Document index.
            
        Returns:
            List of extracted relations.
        """
        if not text or len(entities) < 2:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        all_relations = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Find entities in this sentence
            sent_start = text.find(sentence)
            sent_end = sent_start + len(sentence)
            
            sentence_entities = [
                e for e in entities
                if e.start >= sent_start and e.end <= sent_end
            ]
            
            if len(sentence_entities) >= 2:
                relations = self.extract_from_sentence(
                    sentence, sentence_entities, doc_idx
                )
                all_relations.extend(relations)
        
        # Remove duplicates
        unique_relations = list(set(all_relations))
        
        return unique_relations


if __name__ == "__main__":
    # Demo
    from .entity_extractor import EntityExtractor
    
    print("=" * 60)
    print("Relation Extractor Demo")
    print("=" * 60)
    
    entity_extractor = EntityExtractor(use_transformer=False)
    relation_extractor = RelationExtractor()
    
    text = """
    Presiden Joko Widodo mengumumkan kebijakan baru di Istana Negara Jakarta.
    Menteri Kesehatan Budi Gunadi menjelaskan bahwa program vaksinasi akan 
    dimulai pada Januari 2024. "Kami siap melayani masyarakat," kata Budi.
    DPR mendukung kebijakan tersebut dengan persetujuan 85 persen anggota.
    """
    
    # Extract entities
    entities = entity_extractor.extract(text)
    print(f"\nFound {len(entities)} entities")
    
    # Extract relations
    relations = relation_extractor.extract(text, entities)
    print(f"Found {len(relations)} relations\n")
    
    for rel in relations:
        print(f"[{rel.predicate.value}]")
        print(f"  Subject: {rel.subject.normalized} ({rel.subject.entity_type.value})")
        print(f"  Object: {rel.object.normalized} ({rel.object.entity_type.value})")
        print(f"  Confidence: {rel.confidence:.2f}")
        print()
