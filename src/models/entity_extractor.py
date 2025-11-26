"""
Entity Extraction for Indonesian News Text

Uses a combination of:
1. Rule-based extraction for Indonesian-specific patterns
2. Transformer-based NER (IndoBERT fine-tuned on NER)
3. Dictionary lookup for known entities (ministers, organizations, locations)

Handles Indonesian-specific challenges:
- Prefixes/suffixes in names (Bapak, Ibu, -nya)
- Titles and honorifics (Dr., Prof., H., Hj.)
- Organizational abbreviations (Kemkes, Kemenkeu, DPR, MPR)
"""

import re
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class EntityType(Enum):
    """Types of entities in Indonesian news."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    EVENT = "EVENT"
    POLICY = "POLICY"
    POSITION = "POSITION"  # Job titles, roles
    UNKNOWN = "UNKNOWN"


@dataclass
class Entity:
    """
    Represents an extracted entity with metadata.
    """
    text: str
    normalized: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    source_doc: int = 0
    temporal_anchor: Optional[str] = None
    aliases: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)
    language: str = "id"  # 'id' or 'en'
    
    def __hash__(self):
        return hash((self.normalized, self.entity_type))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.normalized == other.normalized and self.entity_type == other.entity_type
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "type": self.entity_type.value,
            "span": [self.start, self.end],
            "confidence": round(self.confidence, 4),
            "source_doc": self.source_doc,
            "temporal_anchor": self.temporal_anchor,
            "aliases": list(self.aliases),
            "metadata": self.metadata,
            "language": self.language
        }


class EntityExtractor:
    """
    Extract named entities from Indonesian and English news text.
    """
    
    # --- INDONESIAN PATTERNS ---
    ID_HONORIFICS = {
        'bapak', 'ibu', 'pak', 'bu', 'saudara', 'saudari',
        'dr', 'dr.', 'prof', 'prof.', 'ir', 'ir.',
        'h', 'h.', 'hj', 'hj.', 'kh', 'kh.',
        'drs', 'drs.', 'drg', 'drg.',
        'letjen', 'mayjen', 'brigjen', 'kolonel', 'komjen',
        'jenderal', 'laksamana', 'marsekal'
    }
    
    ID_ORG_PATTERNS = [
        r'(?:Kementerian|Kementeri)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        r'(?:PT|CV|UD|Yayasan|Universitas|Institut|Politeknik)\s+[A-Z][a-zA-Z\s]+',
        r'(?:Bank|Partai|Badan|Lembaga|Komisi|Dewan)\s+[A-Z][a-zA-Z\s]+',
        r'[A-Z]{2,5}(?:\s+RI)?',
    ]
    
    ID_POSITION_PATTERNS = [
        r'(?:Presiden|Wakil Presiden|Wapres)',
        r'(?:Menteri|Wakil Menteri)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        r'(?:Gubernur|Wakil Gubernur|Wagub)',
        r'(?:Bupati|Wakil Bupati|Wabup)',
        r'(?:Wali ?Kota|Wakil Wali ?Kota|Wawali)',
        r'(?:Ketua|Wakil Ketua)\s+(?:DPR|MPR|DPD|MK|MA|KPK|KPU|Umum)',
        r'(?:Direktur|Dirjen|Sekjen|Irjen)\s+[A-Z][a-zA-Z\s]+',
        r'(?:Kepala|Kapolri|Kapolda|Panglima|Kasad|Kasal|Kasau)',
    ]

    ID_DATE_PATTERNS = [
        r'\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}',
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}',
        r'(?:tahun\s+)?\d{4}',
        r'(?:kemarin|hari ini|besok|lusa)',
        r'(?:minggu|bulan|tahun)\s+(?:lalu|depan|ini)',
    ]

    # --- ENGLISH PATTERNS ---
    EN_HONORIFICS = {
        'mr', 'mr.', 'mrs', 'mrs.', 'ms', 'ms.', 'dr', 'dr.', 'prof', 'prof.',
        'sir', 'madam', 'general', 'colonel', 'major', 'captain'
    }

    EN_ORG_PATTERNS = [
        r'(?:Ministry|Department)\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        r'(?:University|Institute|College)\s+of\s+[A-Z][a-zA-Z\s]+',
        r'[A-Z][a-zA-Z\s]+\s+(?:Inc\.|Corp\.|Ltd\.|LLC|Group|Bank|Association)',
        r'(?:The\s+)?(?:United\s+Nations|World\s+Health\s+Organization|European\s+Union)',
        r'[A-Z]{2,5}',
    ]

    EN_POSITION_PATTERNS = [
        r'(?:President|Vice\s+President|VP)',
        r'(?:Prime\s+Minister|Minister)\s+of\s+[A-Z][a-z]+',
        r'(?:Governor|Mayor|Senator|Congressman|Representative)',
        r'(?:CEO|CFO|CTO|COO|Director|Chairman|Secretary)',
        r'(?:Head|Chief)\s+of\s+[A-Z][a-zA-Z\s]+',
    ]

    EN_DATE_PATTERNS = [
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'(?:yesterday|today|tomorrow)',
        r'(?:last|next)\s+(?:week|month|year)',
    ]
    
    # Common organization abbreviations (Shared/Mixed)
    ORG_ABBREVS = {
        'kemkes': 'Kementerian Kesehatan',
        'kemenkeu': 'Kementerian Keuangan',
        'who': 'World Health Organization',
        'un': 'United Nations',
        'pbb': 'Perserikatan Bangsa-Bangsa',
        'fbi': 'Federal Bureau of Investigation',
        'cia': 'Central Intelligence Agency',
        'nasa': 'National Aeronautics and Space Administration',
        # ... (keep existing ID abbrevs) ...
    }
    
    # Indonesian provinces and major cities
    LOCATIONS = {
        # ... (keep existing) ...
        'jakarta', 'surabaya', 'bandung', 'medan', 'semarang',
        'bali', 'yogyakarta', 'aceh', 'papua',
        'london', 'new york', 'tokyo', 'paris', 'singapore', 'washington',
        'beijing', 'moscow', 'berlin', 'canberra'
    }
    
    def __init__(
        self,
        use_transformer: bool = False,
        transformer_model: str = "cahya/bert-base-indonesian-NER"
    ):
        self.use_transformer = use_transformer
        self.transformer_model = transformer_model
        self._compile_patterns()
        self._ner_pipeline = None
        print(f"[EntityExtractor] Initialized (Multi-language support enabled)")

    def _compile_patterns(self):
        """Compile regex patterns for both languages."""
        # ID Patterns
        self.id_org_regex = [re.compile(p, re.IGNORECASE) for p in self.ID_ORG_PATTERNS]
        self.id_position_regex = [re.compile(p, re.IGNORECASE) for p in self.ID_POSITION_PATTERNS]
        self.id_date_regex = [re.compile(p, re.IGNORECASE) for p in self.ID_DATE_PATTERNS]
        
        # EN Patterns
        self.en_org_regex = [re.compile(p, re.IGNORECASE) for p in self.EN_ORG_PATTERNS]
        self.en_position_regex = [re.compile(p, re.IGNORECASE) for p in self.EN_POSITION_PATTERNS]
        self.en_date_regex = [re.compile(p, re.IGNORECASE) for p in self.EN_DATE_PATTERNS]
        
        # Shared
        self.money_regex = [re.compile(p, re.IGNORECASE) for p in [
            r'Rp\.?\s*[\d.,]+(?:\s*(?:ribu|juta|miliar|triliun))?',
            r'\$\s*[\d.,]+(?:\s*(?:thousand|million|billion))?',
            r'[\d.,]+\s*(?:rupiah|dollar|dolar|USD|IDR|EUR|GBP)',
        ]]
        self.percent_regex = [re.compile(p, re.IGNORECASE) for p in [r'[\d.,]+\s*(?:persen|%)']]
        
        # Person Regex (Language specific honorifics)
        self.id_person_regex = re.compile(
            r'(?:(?:' + '|'.join(self.ID_HONORIFICS) + r')\s+)?'
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})',
            re.IGNORECASE
        )
        self.en_person_regex = re.compile(
            r'(?:(?:' + '|'.join(self.EN_HONORIFICS) + r')\s+)?'
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})',
            re.IGNORECASE
        )

    def detect_language(self, text: str) -> str:
        """Detect language of text (id or en)."""
        if not LANGDETECT_AVAILABLE:
            # Fallback heuristic
            text_lower = text.lower()
            id_words = {'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk'}
            en_words = {'the', 'and', 'in', 'to', 'of', 'this', 'that', 'for'}
            
            id_count = sum(1 for w in text_lower.split() if w in id_words)
            en_count = sum(1 for w in text_lower.split() if w in en_words)
            
            return 'id' if id_count >= en_count else 'en'
            
        try:
            return detect(text)
        except:
            return 'id'

    def _extract_by_pattern(
        self,
        text: str,
        patterns: List[re.Pattern],
        entity_type: EntityType,
        doc_idx: int = 0,
        language: str = 'id'
    ) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                entity_text = match.group()
                normalized = self._normalize_entity(entity_text, entity_type, language)
                
                entities.append(Entity(
                    text=entity_text,
                    normalized=normalized,
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8,
                    source_doc=doc_idx,
                    language=language
                ))
        return entities

    def _normalize_entity(self, text: str, entity_type: EntityType, language: str = 'id') -> str:
        """Normalize entity text."""
        text = text.strip()
        
        if entity_type == EntityType.ORGANIZATION:
            text_lower = text.lower()
            if text_lower in self.ORG_ABBREVS:
                return self.ORG_ABBREVS[text_lower]
            return text
            
        elif entity_type == EntityType.PERSON:
            honorifics = self.ID_HONORIFICS if language == 'id' else self.EN_HONORIFICS
            for honorific in honorifics:
                if text.lower().startswith(honorific):
                    text = text[len(honorific):].strip()
                    break
            return text.title()
            
        elif entity_type == EntityType.LOCATION:
            return text.title()
            
        elif entity_type == EntityType.DATE:
            # TODO: Add English date normalization
            if language == 'id':
                return self._normalize_date_id(text)
            return text
            
        return text

    def _normalize_date_id(self, date_str: str) -> str:
        """Normalize Indonesian date."""
        # ... (Existing logic) ...
        return date_str

    def extract(
        self,
        text: str,
        doc_idx: int = 0,
        merge_overlapping: bool = True
    ) -> List[Entity]:
        """Extract all entities from text."""
        if not text or len(text.strip()) < 5:
            return []
        
        language = self.detect_language(text)
        all_entities = []
        
        # Select patterns based on language
        if language == 'id':
            org_regex = self.id_org_regex
            pos_regex = self.id_position_regex
            date_regex = self.id_date_regex
            person_regex = self.id_person_regex
        else:
            org_regex = self.en_org_regex
            pos_regex = self.en_position_regex
            date_regex = self.en_date_regex
            person_regex = self.en_person_regex
            
        # Extract
        all_entities.extend(self._extract_by_pattern(text, org_regex, EntityType.ORGANIZATION, doc_idx, language))
        all_entities.extend(self._extract_by_pattern(text, pos_regex, EntityType.POSITION, doc_idx, language))
        all_entities.extend(self._extract_by_pattern(text, date_regex, EntityType.DATE, doc_idx, language))
        all_entities.extend(self._extract_by_pattern(text, self.money_regex, EntityType.MONEY, doc_idx, language))
        all_entities.extend(self._extract_by_pattern(text, self.percent_regex, EntityType.PERCENT, doc_idx, language))
        
        # Locations (Shared)
        # ... (Location extraction logic) ...
        
        # Person names
        for match in person_regex.finditer(text):
            name = match.group(1) if match.lastindex else match.group()
            # Skip if it looks like an organization
            skip_words = ['kementerian', 'partai', 'bank', 'pt'] if language == 'id' else ['ministry', 'bank', 'corp', 'inc']
            if any(w in name.lower() for w in skip_words):
                continue
            
            all_entities.append(Entity(
                text=match.group(),
                normalized=self._normalize_entity(name, EntityType.PERSON, language),
                entity_type=EntityType.PERSON,
                start=match.start(),
                end=match.end(),
                confidence=0.6,
                source_doc=doc_idx,
                language=language
            ))

        # Remove duplicates and merge
        if merge_overlapping:
            all_entities = self._merge_entities(all_entities)
        
        all_entities.sort(key=lambda e: (e.start, -e.end))
        return all_entities

    # ... (Keep _merge_entities and other helpers) ...
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities, preferring higher confidence."""
        if not entities:
            return []
        
        # Sort by start position, then by length (longer first)
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        
        merged = []
        for entity in entities:
            # Check if this overlaps with any existing entity
            overlapping = False
            for i, existing in enumerate(merged):
                # Check overlap
                if not (entity.end <= existing.start or entity.start >= existing.end):
                    overlapping = True
                    # Keep the one with higher confidence, or longer span
                    if entity.confidence > existing.confidence:
                        merged[i] = entity
                    elif entity.confidence == existing.confidence:
                        if (entity.end - entity.start) > (existing.end - existing.start):
                            merged[i] = entity
                    break
            
            if not overlapping:
                merged.append(entity)
        
        return merged
    
    def extract_from_documents(
        self,
        documents: List[str]
    ) -> Tuple[List[List[Entity]], Dict[str, Entity]]:
        """
        Extract entities from multiple documents.
        
        Returns:
            Tuple of:
            - List of entity lists (one per document)
            - Dictionary of unique entities by normalized name
        """
        all_doc_entities = []
        unique_entities = {}
        
        for doc_idx, doc in enumerate(documents):
            doc_entities = self.extract(doc, doc_idx)
            all_doc_entities.append(doc_entities)
            
            # Track unique entities
            for entity in doc_entities:
                key = f"{entity.entity_type.value}:{entity.normalized}"
                if key not in unique_entities:
                    unique_entities[key] = entity
                else:
                    # Merge aliases
                    unique_entities[key].aliases.add(entity.text)
        
        return all_doc_entities, unique_entities


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Entity Extractor Demo")
    print("=" * 60)
    
    extractor = EntityExtractor(use_transformer=False)
    
    text = """
    Presiden Joko Widodo mengumumkan kebijakan baru pada 15 Januari 2024 
    di Istana Negara, Jakarta. Menteri Kesehatan Budi Gunadi Sadikin 
    menjelaskan bahwa anggaran Kemkes sebesar Rp 150 miliar akan dialokasikan 
    untuk program vaksinasi. DPR menyetujui kebijakan ini dengan dukungan 
    80 persen anggota. WHO memberikan apresiasi atas langkah pemerintah Indonesia.
    """
    
    entities = extractor.extract(text)
    
    print(f"\nFound {len(entities)} entities:\n")
    
    for entity in entities:
        print(f"[{entity.entity_type.value}] {entity.text}")
        print(f"  Normalized: {entity.normalized}")
        print(f"  Confidence: {entity.confidence:.2f}")
        print(f"  Span: [{entity.start}, {entity.end}]")
        print()
