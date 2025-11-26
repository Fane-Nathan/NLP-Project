"""
Knowledge Graph for Indonesian News TMDS

The core graph structure that stores:
1. Entities as nodes (with temporal attributes)
2. Relations as edges (with provenance tracking)
3. Temporal anchors for timeline construction
4. Source attribution for fact verification

This is the "ground truth" that constrains summarization to prevent hallucinations.

Technical Design:
- Backend: NetworkX (flexible, Python-native)
- Node attributes: entity data, temporal bounds, source documents
- Edge attributes: relation type, confidence, evidence
- Supports querying by time range, entity type, relation type
"""

import json
from typing import List, Dict, Optional, Set, Tuple, Any, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import networkx as nx

from .entity_extractor import Entity, EntityType, EntityExtractor
from .relation_extractor import Relation, RelationType, RelationExtractor
from .temporal_anchor import TemporalExpression, TemporalType, TemporalAnchor, Timeline


@dataclass
class KGTriple:
    """
    A knowledge graph triple (subject, predicate, object).
    
    The fundamental unit of knowledge representation.
    """
    subject: str           # Entity normalized name
    predicate: str         # Relation type
    object: str            # Entity normalized name
    confidence: float = 1.0
    source_docs: Set[int] = field(default_factory=set)
    temporal_start: Optional[str] = None
    temporal_end: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": round(self.confidence, 4),
            "source_docs": list(self.source_docs),
            "temporal": {
                "start": self.temporal_start,
                "end": self.temporal_end
            },
            "evidence_count": len(self.evidence)
        }
    
    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))
    
    def __eq__(self, other):
        if not isinstance(other, KGTriple):
            return False
        return (
            self.subject == other.subject and
            self.predicate == other.predicate and
            self.object == other.object
        )


class KnowledgeGraph:
    """
    Knowledge Graph for Indonesian News Documents.
    
    Provides:
    1. Entity storage with temporal bounds
    2. Relation storage with provenance
    3. Timeline construction
    4. Query capabilities for fact verification
    5. Conflict detection
    
    Architecture:
        Directed MultiGraph (NetworkX DiGraph with parallel edges)
        - Nodes: Entities (keyed by normalized name)
        - Edges: Relations (can have multiple between same nodes)
    """
    
    def __init__(self, name: str = "news_kg"):
        """
        Initialize empty knowledge graph.
        
        Args:
            name: Graph identifier.
        """
        self.name = name
        self.graph = nx.DiGraph()
        
        # Indexes for fast lookup
        self._entity_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self._relation_index: Dict[RelationType, Set[Tuple[str, str]]] = defaultdict(set)
        self._temporal_index: Dict[str, Set[str]] = defaultdict(set)  # date -> entities
        self._source_index: Dict[int, Set[str]] = defaultdict(set)  # doc_idx -> entities
        
        # Extractors (lazy initialized)
        self._entity_extractor = None
        self._relation_extractor = None
        self._temporal_anchor = None
        
        # Statistics
        self.stats = {
            "entities_added": 0,
            "relations_added": 0,
            "documents_processed": 0,
            "conflicts_detected": 0
        }
        
        print(f"[KnowledgeGraph] Initialized: {name}")
    
    def _get_entity_extractor(self) -> EntityExtractor:
        """Lazy load entity extractor."""
        if self._entity_extractor is None:
            self._entity_extractor = EntityExtractor(use_transformer=False)
        return self._entity_extractor
    
    def _get_relation_extractor(self) -> RelationExtractor:
        """Lazy load relation extractor."""
        if self._relation_extractor is None:
            self._relation_extractor = RelationExtractor()
        return self._relation_extractor
    
    def _get_temporal_anchor(self) -> TemporalAnchor:
        """Lazy load temporal anchor."""
        if self._temporal_anchor is None:
            self._temporal_anchor = TemporalAnchor()
        return self._temporal_anchor
    
    def _entity_key(self, entity: Entity) -> str:
        """Generate unique key for entity."""
        return f"{entity.entity_type.value}:{entity.normalized}"
    
    def add_entity(
        self,
        entity: Entity,
        doc_idx: int = 0,
        temporal: Optional[TemporalExpression] = None
    ) -> str:
        """
        Add entity to graph.
        
        Args:
            entity: Entity to add.
            doc_idx: Source document index.
            temporal: Optional temporal anchor for this entity.
            
        Returns:
            Entity key in graph.
        """
        key = self._entity_key(entity)
        
        if key in self.graph:
            # Update existing node
            node = self.graph.nodes[key]
            node['source_docs'].add(doc_idx)
            node['mentions'] += 1
            node['aliases'].update(entity.aliases)
            node['aliases'].add(entity.text)
            
            # Update confidence (running average)
            old_conf = node['confidence']
            old_count = node['mentions'] - 1
            node['confidence'] = (old_conf * old_count + entity.confidence) / node['mentions']
            
            # Update temporal bounds if provided
            if temporal:
                if node['temporal_start'] is None or temporal.normalized_start < node['temporal_start']:
                    node['temporal_start'] = temporal.normalized_start
                if node['temporal_end'] is None or temporal.normalized_end > node['temporal_end']:
                    node['temporal_end'] = temporal.normalized_end
        else:
            # Add new node
            self.graph.add_node(
                key,
                normalized=entity.normalized,
                entity_type=entity.entity_type.value,
                confidence=entity.confidence,
                source_docs={doc_idx},
                mentions=1,
                aliases={entity.text},
                temporal_start=temporal.normalized_start if temporal else None,
                temporal_end=temporal.normalized_end if temporal else None,
                metadata=entity.metadata,
                language=entity.language
            )
            self.stats["entities_added"] += 1
        
        # Update indexes
        self._entity_index[entity.entity_type].add(key)
        self._source_index[doc_idx].add(key)
        
        if temporal and temporal.normalized_start:
            date_key = temporal.normalized_start[:10]  # YYYY-MM-DD
            self._temporal_index[date_key].add(key)
        
        return key
    
    def add_relation(
        self,
        relation: Relation,
        doc_idx: int = 0
    ) -> Tuple[str, str]:
        """
        Add relation to graph.
        
        Args:
            relation: Relation to add.
            doc_idx: Source document index.
            
        Returns:
            Tuple of (subject_key, object_key).
        """
        # Ensure entities exist
        subj_key = self.add_entity(relation.subject, doc_idx)
        obj_key = self.add_entity(relation.object, doc_idx)
        
        rel_type = relation.predicate.value
        
        # Check for existing edge
        if self.graph.has_edge(subj_key, obj_key):
            edge = self.graph[subj_key][obj_key]
            
            # Check if same relation type
            if edge.get('relation_type') == rel_type:
                # Update existing relation
                edge['source_docs'].add(doc_idx)
                edge['mentions'] += 1
                edge['confidence'] = max(edge['confidence'], relation.confidence)
                if relation.evidence:
                    edge['evidence'].extend(relation.evidence[:2])
            else:
                # Different relation type - might be conflict or multiple relations
                # For now, keep higher confidence one
                if relation.confidence > edge.get('confidence', 0):
                    self._update_edge(
                        subj_key, obj_key, relation, doc_idx
                    )
                    self.stats["conflicts_detected"] += 1
        else:
            # Add new edge
            self.graph.add_edge(
                subj_key,
                obj_key,
                relation_type=rel_type,
                confidence=relation.confidence,
                source_docs={doc_idx},
                mentions=1,
                evidence=relation.evidence[:3] if relation.evidence else [],
                temporal_anchor=relation.temporal_anchor,
                source_sentence=relation.source_sentence[:200]
            )
            self.stats["relations_added"] += 1
        
        # Update relation index
        self._relation_index[relation.predicate].add((subj_key, obj_key))
        
        return subj_key, obj_key
    
    def _update_edge(
        self,
        subj_key: str,
        obj_key: str,
        relation: Relation,
        doc_idx: int
    ):
        """Update existing edge with new relation data."""
        self.graph[subj_key][obj_key].update({
            'relation_type': relation.predicate.value,
            'confidence': relation.confidence,
            'source_docs': {doc_idx},
            'mentions': 1,
            'evidence': relation.evidence[:3] if relation.evidence else [],
            'temporal_anchor': relation.temporal_anchor,
            'source_sentence': relation.source_sentence[:200]
        })
    
    def add_document(
        self,
        text: str,
        doc_idx: int = 0,
        extract_temporal: bool = True
    ) -> Dict[str, int]:
        """
        Process a document and add all extracted knowledge.
        
        Args:
            text: Document text.
            doc_idx: Document index.
            extract_temporal: Whether to extract temporal expressions.
            
        Returns:
            Statistics dict with counts.
        """
        if not text or len(text.strip()) < 10:
            return {"entities": 0, "relations": 0, "temporals": 0}
        
        # Extract entities
        entity_extractor = self._get_entity_extractor()
        entities = entity_extractor.extract(text, doc_idx)
        
        # Extract temporal expressions
        temporals = []
        if extract_temporal:
            temporal_anchor = self._get_temporal_anchor()
            temporals = temporal_anchor.extract(text, doc_idx)
        
        # Associate temporals with nearby entities
        entity_temporals = self._associate_temporals(entities, temporals)
        
        # Add entities
        for entity in entities:
            temporal = entity_temporals.get(entity, None)
            self.add_entity(entity, doc_idx, temporal)
        
        # Extract and add relations
        relation_extractor = self._get_relation_extractor()
        relations = relation_extractor.extract(text, entities, doc_idx)
        
        for relation in relations:
            self.add_relation(relation, doc_idx)
        
        self.stats["documents_processed"] += 1
        
        return {
            "entities": len(entities),
            "relations": len(relations),
            "temporals": len(temporals)
        }
    
    def _associate_temporals(
        self,
        entities: List[Entity],
        temporals: List[TemporalExpression]
    ) -> Dict[Entity, TemporalExpression]:
        """Associate temporal expressions with nearby entities."""
        associations = {}
        
        for entity in entities:
            closest = None
            min_distance = float('inf')
            
            for temporal in temporals:
                # Calculate distance
                if temporal.end_pos <= entity.start:
                    distance = entity.start - temporal.end_pos
                elif entity.end <= temporal.start_pos:
                    distance = temporal.start_pos - entity.end
                else:
                    distance = 0  # Overlapping
                
                if distance < min_distance and distance < 100:  # Max 100 chars
                    min_distance = distance
                    closest = temporal
            
            if closest:
                associations[entity] = closest
        
        return associations
    
    def add_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Process multiple documents.
        
        Args:
            documents: List of document texts.
            show_progress: Print progress.
            
        Returns:
            Aggregate statistics.
        """
        total_stats = {"entities": 0, "relations": 0, "temporals": 0}
        
        for doc_idx, doc in enumerate(documents):
            if show_progress and doc_idx % 10 == 0:
                print(f"[KG] Processing doc {doc_idx + 1}/{len(documents)}")
            
            stats = self.add_document(doc, doc_idx)
            for key in total_stats:
                total_stats[key] += stats[key]
        
        print(f"[KG] Finished: {total_stats}")
        return total_stats
    
    # === QUERY METHODS ===
    
    def get_entity(self, key: str) -> Optional[Dict]:
        """Get entity by key."""
        if key in self.graph:
            return dict(self.graph.nodes[key])
        return None
    
    def get_entities_by_type(
        self,
        entity_type: EntityType
    ) -> List[Tuple[str, Dict]]:
        """Get all entities of a specific type."""
        keys = self._entity_index.get(entity_type, set())
        return [(k, dict(self.graph.nodes[k])) for k in keys if k in self.graph]
    
    def get_entities_by_time(
        self,
        start_date: str,
        end_date: str
    ) -> List[Tuple[str, Dict]]:
        """Get entities within a time range."""
        results = []
        
        for key in self.graph.nodes():
            node = self.graph.nodes[key]
            node_start = node.get('temporal_start')
            node_end = node.get('temporal_end')
            
            if node_start is None and node_end is None:
                continue
            
            # Check overlap with query range
            if node_start and node_start <= end_date:
                if node_end is None or node_end >= start_date:
                    results.append((key, dict(node)))
        
        return results
    
    def get_relations_for_entity(
        self,
        entity_key: str,
        direction: str = "both"
    ) -> List[Dict]:
        """
        Get relations involving an entity.
        
        Args:
            entity_key: Entity key.
            direction: "outgoing", "incoming", or "both".
            
        Returns:
            List of relation dicts.
        """
        relations = []
        
        if direction in ("outgoing", "both"):
            for _, target, data in self.graph.out_edges(entity_key, data=True):
                relations.append({
                    "subject": entity_key,
                    "predicate": data.get('relation_type'),
                    "object": target,
                    **data
                })
        
        if direction in ("incoming", "both"):
            for source, _, data in self.graph.in_edges(entity_key, data=True):
                relations.append({
                    "subject": source,
                    "predicate": data.get('relation_type'),
                    "object": entity_key,
                    **data
                })
        
        return relations
    
    def get_relations_by_type(
        self,
        relation_type: RelationType
    ) -> List[Dict]:
        """Get all relations of a specific type."""
        pairs = self._relation_index.get(relation_type, set())
        relations = []
        
        for subj_key, obj_key in pairs:
            if self.graph.has_edge(subj_key, obj_key):
                data = dict(self.graph[subj_key][obj_key])
                relations.append({
                    "subject": subj_key,
                    "object": obj_key,
                    **data
                })
        
        return relations
    
    def find_path(
        self,
        source: str,
        target: str,
        max_length: int = 3
    ) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        try:
            path = nx.shortest_path(
                self.graph, source, target, weight=None
            )
            if len(path) <= max_length + 1:
                return path
        except nx.NetworkXNoPath:
            pass
        return None
    
    def get_neighborhood(
        self,
        entity_key: str,
        depth: int = 1
    ) -> nx.DiGraph:
        """Get subgraph around an entity."""
        nodes = {entity_key}
        frontier = {entity_key}
        
        for _ in range(depth):
            new_frontier = set()
            for node in frontier:
                new_frontier.update(self.graph.predecessors(node))
                new_frontier.update(self.graph.successors(node))
            nodes.update(new_frontier)
            frontier = new_frontier
        
        return self.graph.subgraph(nodes).copy()
    
    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[KGTriple]:
        """
        Query graph with SPARQL-like pattern.
        
        Args:
            subject: Subject pattern (None = any).
            predicate: Predicate pattern (None = any).
            obj: Object pattern (None = any).
            
        Returns:
            Matching triples.
        """
        results = []
        
        for source, target, data in self.graph.edges(data=True):
            # Check subject
            if subject and subject not in source:
                continue
            
            # Check predicate
            rel_type = data.get('relation_type', '')
            if predicate and predicate.upper() != rel_type.upper():
                continue
            
            # Check object
            if obj and obj not in target:
                continue
            
            results.append(KGTriple(
                subject=source,
                predicate=rel_type,
                object=target,
                confidence=data.get('confidence', 1.0),
                source_docs=data.get('source_docs', set()),
                evidence=data.get('evidence', [])
            ))
        
        return results
    
    # === CONFLICT DETECTION ===
    
    def detect_conflicts(self) -> List[Dict]:
        """
        Detect potential conflicts in the knowledge graph.
        
        Conflicts include:
        - Contradictory relations (X supports Y and X opposes Y)
        - Temporal impossibilities (event B before event A, but B caused A)
        - Role conflicts (X leads org at time T, but Y also leads org at T)
        """
        conflicts = []
        
        # Check for contradictory relations
        contradictions = [
            (RelationType.SUPPORTS, RelationType.OPPOSES),
            (RelationType.CAUSES, RelationType.PREVENTS),
            (RelationType.BEFORE, RelationType.AFTER),
        ]
        
        for rel1, rel2 in contradictions:
            pairs1 = self._relation_index.get(rel1, set())
            pairs2 = self._relation_index.get(rel2, set())
            
            # Check if same pair has contradictory relations
            for pair in pairs1 & pairs2:
                conflicts.append({
                    "type": "contradiction",
                    "entities": pair,
                    "relations": [rel1.value, rel2.value],
                    "severity": "HIGH"
                })
        
        # Check for temporal impossibilities
        # (Entity mentioned before it existed)
        for key in self.graph.nodes():
            node = self.graph.nodes[key]
            node_start = node.get('temporal_start')
            
            if not node_start:
                continue
            
            # Check incoming relations
            for source, _, data in self.graph.in_edges(key, data=True):
                source_node = self.graph.nodes.get(source, {})
                source_end = source_node.get('temporal_end')
                
                if source_end and source_end < node_start:
                    conflicts.append({
                        "type": "temporal_impossibility",
                        "entities": (source, key),
                        "reason": f"{source} ended before {key} started",
                        "severity": "MEDIUM"
                    })
        
        self.stats["conflicts_detected"] = len(conflicts)
        return conflicts
    
    # === EXPORT/IMPORT ===
    
    def to_dict(self) -> Dict:
        """Export graph to dictionary."""
        return {
            "name": self.name,
            "stats": self.stats,
            "nodes": [
                {"key": k, **dict(v)}
                for k, v in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **dict(d)}
                for u, v, d in self.graph.edges(data=True)
            ]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export graph to JSON string."""
        data = self.to_dict()
        
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(i) for i in obj]
            return obj
        
        return json.dumps(convert_sets(data), indent=indent, ensure_ascii=False)
    
    def save(self, filepath: str):
        """Save graph to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        print(f"[KG] Saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeGraph':
        """Load graph from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls(name=data.get('name', 'loaded_kg'))
        kg.stats = data.get('stats', kg.stats)
        
        # Reconstruct graph
        for node in data.get('nodes', []):
            key = node.pop('key')
            # Convert lists back to sets
            if 'source_docs' in node:
                node['source_docs'] = set(node['source_docs'])
            if 'aliases' in node:
                node['aliases'] = set(node['aliases'])
            kg.graph.add_node(key, **node)
        
        for edge in data.get('edges', []):
            source = edge.pop('source')
            target = edge.pop('target')
            if 'source_docs' in edge:
                edge['source_docs'] = set(edge['source_docs'])
            kg.graph.add_edge(source, target, **edge)
        
        print(f"[KG] Loaded from {filepath}")
        return kg
    
    def get_summary(self) -> str:
        """Get human-readable summary of graph."""
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        
        type_counts = defaultdict(int)
        for key in self.graph.nodes():
            entity_type = self.graph.nodes[key].get('entity_type', 'UNKNOWN')
            type_counts[entity_type] += 1
        
        summary = [
            f"Knowledge Graph: {self.name}",
            "=" * 40,
            f"Entities: {n_nodes}",
            f"Relations: {n_edges}",
            f"Documents processed: {self.stats['documents_processed']}",
            "",
            "Entity Types:",
        ]
        
        for etype, count in sorted(type_counts.items()):
            summary.append(f"  {etype}: {count}")
        
        return "\n".join(summary)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Knowledge Graph Demo")
    print("=" * 60)
    
    kg = KnowledgeGraph(name="demo_kg")
    
    documents = [
        """Presiden Joko Widodo mengumumkan kebijakan vaksinasi baru 
        pada 15 Januari 2024 di Istana Negara Jakarta. Menteri Kesehatan 
        Budi Gunadi menjelaskan program akan dimulai Februari 2024.""",
        
        """Pada kuartal III 2023, DPR telah menyetujui anggaran Kemkes 
        sebesar Rp 150 miliar untuk program vaksinasi nasional. Ketua DPR 
        Puan Maharani mendukung kebijakan tersebut.""",
        
        """WHO memuji Indonesia atas keberhasilan program vaksinasi yang 
        mencapai 70% cakupan pada Desember 2023. Direktur WHO Tedros 
        mengapresiasi kerja keras Kementerian Kesehatan."""
    ]
    
    # Build graph from documents
    kg.add_documents(documents)
    
    # Print summary
    print("\n" + kg.get_summary())
    
    # Query examples
    print("\n" + "=" * 40)
    print("Query Examples:")
    print("=" * 40)
    
    # Get all persons
    persons = kg.get_entities_by_type(EntityType.PERSON)
    print(f"\nPersons found: {len(persons)}")
    for key, data in persons[:5]:
        print(f"  - {data['normalized']}")
    
    # Get relations for an organization
    print("\nRelations involving 'Kementerian Kesehatan':")
    kemkes_key = "ORGANIZATION:Kementerian Kesehatan"
    if kemkes_key in kg.graph:
        relations = kg.get_relations_for_entity(kemkes_key)
        for rel in relations[:5]:
            print(f"  [{rel['predicate']}] {rel['subject']} -> {rel['object']}")
    
    # Detect conflicts
    conflicts = kg.detect_conflicts()
    print(f"\nConflicts detected: {len(conflicts)}")
    
    # Export
    print("\n" + "=" * 40)
    print("Exporting graph...")
    print(kg.to_json()[:500] + "...")
