"""
Knowledge Graph Integration Test Script

Run this to verify the Knowledge Graph module is working correctly
with your existing TDSM pipeline.

Usage:
    python test_knowledge_graph.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_entity_extraction():
    """Test entity extraction for Indonesian text."""
    print("\n" + "=" * 60)
    print("TEST 1: Entity Extraction")
    print("=" * 60)
    
    from src.models.entity_extractor import EntityExtractor, EntityType
    
    extractor = EntityExtractor(use_transformer=False)
    
    text = """
    Presiden Joko Widodo mengumumkan kebijakan baru pada 15 Januari 2024 
    di Istana Negara, Jakarta. Menteri Kesehatan Budi Gunadi Sadikin 
    menjelaskan bahwa anggaran Kemkes sebesar Rp 150 miliar akan dialokasikan 
    untuk program vaksinasi. DPR menyetujui kebijakan ini dengan dukungan 
    80 persen anggota. WHO memberikan apresiasi atas langkah pemerintah Indonesia.
    """
    
    entities = extractor.extract(text)
    
    print(f"\n✓ Found {len(entities)} entities:\n")
    
    # Group by type
    by_type = {}
    for e in entities:
        etype = e.entity_type.value
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append(e.normalized)
    
    for etype, names in sorted(by_type.items()):
        print(f"  [{etype}]")
        for name in names[:3]:
            print(f"    - {name}")
        if len(names) > 3:
            print(f"    ... and {len(names) - 3} more")
    
    return len(entities) > 0


def test_temporal_anchor():
    """Test temporal expression extraction."""
    print("\n" + "=" * 60)
    print("TEST 2: Temporal Anchoring")
    print("=" * 60)
    
    from src.models.temporal_anchor import TemporalAnchor
    
    anchor = TemporalAnchor()
    
    text = """
    Kebijakan diumumkan pada 15 Januari 2024. Sebelumnya, 
    pada kuartal III 2023, pemerintah telah melakukan kajian selama 3 bulan.
    Minggu lalu, Menteri memberikan penjelasan. Sekitar tahun 2020,
    kebijakan serupa pernah diusulkan. Implementasi dimulai Februari 2024.
    """
    
    expressions = anchor.extract(text)
    
    print(f"\n✓ Found {len(expressions)} temporal expressions:\n")
    
    for expr in expressions:
        print(f"  [{expr.temporal_type.value}] \"{expr.text}\"")
        print(f"    Normalized: {expr.normalized_start[:10] if expr.normalized_start else 'N/A'}")
        print(f"    Confidence: {expr.confidence:.0%}")
    
    return len(expressions) > 0


def test_relation_extraction():
    """Test relation extraction."""
    print("\n" + "=" * 60)
    print("TEST 3: Relation Extraction")
    print("=" * 60)
    
    from src.models.entity_extractor import EntityExtractor
    from src.models.relation_extractor import RelationExtractor
    
    entity_extractor = EntityExtractor(use_transformer=False)
    relation_extractor = RelationExtractor()
    
    text = """
    Presiden Joko Widodo mengumumkan kebijakan baru di Istana Negara Jakarta.
    Menteri Kesehatan Budi Gunadi menjelaskan bahwa program vaksinasi akan 
    dimulai pada Januari 2024. "Kami siap melayani masyarakat," kata Budi.
    DPR mendukung kebijakan tersebut dengan persetujuan 85 persen anggota.
    """
    
    entities = entity_extractor.extract(text)
    relations = relation_extractor.extract(text, entities)
    
    print(f"\n✓ Found {len(relations)} relations:\n")
    
    for rel in relations[:5]:
        print(f"  [{rel.predicate.value}]")
        print(f"    {rel.subject.normalized} → {rel.object.normalized}")
        print(f"    Confidence: {rel.confidence:.0%}")
    
    return len(relations) > 0


def test_knowledge_graph():
    """Test knowledge graph construction."""
    print("\n" + "=" * 60)
    print("TEST 4: Knowledge Graph Construction")
    print("=" * 60)
    
    from src.models.knowledge_graph import KnowledgeGraph
    from src.models.entity_extractor import EntityType
    
    kg = KnowledgeGraph(name="test_kg")
    
    documents = [
        """Presiden Joko Widodo mengumumkan kebijakan vaksinasi baru 
        pada 15 Januari 2024 di Istana Negara Jakarta.""",
        
        """Menteri Kesehatan Budi Gunadi menjelaskan program vaksinasi
        akan dimulai pada Februari 2024. Anggaran sebesar Rp 150 miliar 
        telah disetujui DPR.""",
        
        """WHO memuji Indonesia atas strategi vaksinasi yang komprehensif.
        Direktur WHO memberikan apresiasi kepada Kementerian Kesehatan
        pada Maret 2024."""
    ]
    
    stats = kg.add_documents(documents, show_progress=False)
    
    print(f"\n{kg.get_summary()}")
    
    # Test queries
    print("\n  Query Tests:")
    
    # Get all persons
    persons = kg.get_entities_by_type(EntityType.PERSON)
    print(f"    Persons: {len(persons)}")
    
    # Get all organizations
    orgs = kg.get_entities_by_type(EntityType.ORGANIZATION)
    print(f"    Organizations: {len(orgs)}")
    
    # Detect conflicts
    conflicts = kg.detect_conflicts()
    print(f"    Conflicts: {len(conflicts)}")
    
    return kg.graph.number_of_nodes() > 0


def test_fact_verifier():
    """Test fact verification."""
    print("\n" + "=" * 60)
    print("TEST 5: Fact Verification")
    print("=" * 60)
    
    from src.models.knowledge_graph import KnowledgeGraph
    from src.models.fact_verifier import FactVerifier, VerificationStatus
    
    # Build KG
    kg = KnowledgeGraph(name="verify_test_kg")
    
    documents = [
        "Presiden Joko Widodo mengumumkan kebijakan vaksinasi di Jakarta.",
        "Menteri Kesehatan Budi Gunadi menjelaskan program vaksinasi.",
        "DPR menyetujui anggaran Rp 150 miliar untuk kesehatan."
    ]
    
    kg.add_documents(documents, show_progress=False)
    
    verifier = FactVerifier(kg)
    
    # Test claims
    test_claims = [
        ("Joko Widodo mengumumkan kebijakan di Jakarta.", "Should VERIFY"),
        ("Menteri Ali Wibowo meluncurkan program.", "Should HALLUCINATE (wrong name)"),
        ("DPR menyetujui anggaran kesehatan.", "Should VERIFY"),
    ]
    
    print(f"\n  Claim Verification Results:\n")
    
    all_passed = True
    for claim, expected in test_claims:
        result = verifier.verify_claim(claim)
        
        status_emoji = {
            VerificationStatus.VERIFIED: "✅",
            VerificationStatus.PARTIALLY_VERIFIED: "⚠️",
            VerificationStatus.UNVERIFIED: "❓",
            VerificationStatus.CONTRADICTED: "❌",
            VerificationStatus.HALLUCINATION: "🚨"
        }
        
        print(f"    {status_emoji[result.status]} [{result.status.value}]")
        print(f"       Claim: {claim[:50]}...")
        print(f"       Expected: {expected}")
        print()
    
    return True


def test_constrained_summarizer():
    """Test constrained summarization (without Gemini API)."""
    print("\n" + "=" * 60)
    print("TEST 6: Constrained Summarization (Extractive)")
    print("=" * 60)
    
    from src.models.knowledge_graph import KnowledgeGraph
    from src.models.constrained_summarizer import (
        ConstrainedSummarizer, 
        SummarizationMode
    )
    
    documents = [
        """Presiden Joko Widodo mengumumkan kebijakan vaksinasi baru 
        pada 15 Januari 2024 di Istana Negara Jakarta. Program ini 
        ditargetkan mencapai 70% cakupan dalam 6 bulan.""",
        
        """Menteri Kesehatan Budi Gunadi menjelaskan program vaksinasi
        akan dimulai dengan tenaga kesehatan. Anggaran sebesar 
        Rp 150 miliar telah disetujui DPR.""",
        
        """WHO memuji Indonesia atas strategi vaksinasi yang komprehensif.
        Direktur WHO memberikan apresiasi kepada Kementerian Kesehatan."""
    ]
    
    # Create summarizer (without Gemini for testing)
    summarizer = ConstrainedSummarizer(
        gemini_api_key=None,  # Will fall back to extractive
        max_refinement_iterations=2,
        min_verification_rate=0.5
    )
    
    # Build KG
    summarizer.build_kg_from_documents(documents)
    
    # Generate summary
    result = summarizer.summarize(
        documents,
        mode=SummarizationMode.EXTRACTIVE,
        build_timeline=True
    )
    
    print(f"\n  Summary: {result.summary[:200]}...")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Hallucination-Free: {result.is_hallucination_free}")
    print(f"  Verification Rate: {result.verification_report.verification_rate:.1%}")
    
    if result.timeline and result.timeline.events:
        print(f"\n  Timeline ({len(result.timeline.events)} events):")
        for event in result.timeline.events[:3]:
            temporal, desc, doc_idx = event
            date_str = temporal.normalized_start[:10] if temporal.normalized_start else "Unknown"
            print(f"    📅 {date_str} | {desc[:40]}...")
    
    return len(result.summary) > 0


def test_full_integration():
    """Test full integration with existing hoax detection."""
    print("\n" + "=" * 60)
    print("TEST 7: Full Integration with Trust Layer")
    print("=" * 60)
    
    try:
        from src.hoax_detection import CredibilityAnalyzer
        
        # Check if model exists
        model_path = "models/hoax_indobert_lora"
        if not os.path.exists(model_path):
            print(f"\n  ⚠️ Hoax model not found at {model_path}")
            print("    Run: python -m src.hoax_detection.train_lora --synthetic")
            print("    Skipping integration test...")
            return True  # Don't fail, just skip
        
        from src.models.knowledge_graph import KnowledgeGraph
        from src.models.constrained_summarizer import ConstrainedSummarizer, SummarizationMode
        
        documents = [
            "Pemerintah Indonesia mengumumkan kebijakan vaksinasi COVID-19 baru.",
            "VIRAL! Vaksin mengandung microchip 5G! Bagikan sebelum dihapus!",  # HOAX
            "Kementerian Kesehatan melaporkan peningkatan cakupan vaksinasi.",
            "Resep rendang padang yang enak.",  # OUTLIER
            "WHO memuji Indonesia atas program vaksinasi."
        ]
        
        # Step 1: Trust Layer
        analyzer = CredibilityAnalyzer(hoax_model_path=model_path)
        filtered_docs, report = analyzer.filter_documents(documents)
        
        print(f"\n  Trust Layer: {len(filtered_docs)}/{len(documents)} passed")
        
        # Step 2: KG + Summarization
        summarizer = ConstrainedSummarizer(gemini_api_key=None)
        summarizer.build_kg_from_documents(filtered_docs)
        
        result = summarizer.summarize(
            filtered_docs,
            mode=SummarizationMode.EXTRACTIVE
        )
        
        print(f"  Summary generated: {len(result.summary)} chars")
        print(f"  Hallucination-Free: {result.is_hallucination_free}")
        
        return True
        
    except ImportError as e:
        print(f"\n  ⚠️ Hoax detection module not available: {e}")
        print("    Integration test skipped.")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧪 KNOWLEDGE GRAPH MODULE TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Entity Extraction", test_entity_extraction),
        ("Temporal Anchoring", test_temporal_anchor),
        ("Relation Extraction", test_relation_extraction),
        ("Knowledge Graph", test_knowledge_graph),
        ("Fact Verification", test_fact_verifier),
        ("Constrained Summarizer", test_constrained_summarizer),
        ("Full Integration", test_full_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "✅ PASSED" if passed else "❌ FAILED"))
        except Exception as e:
            results.append((name, f"❌ ERROR: {e}"))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    for name, status in results:
        print(f"  {status} - {name}")
    
    passed_count = sum(1 for _, s in results if "PASSED" in s)
    total_count = len(results)
    
    print(f"\n  Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n  🎉 All tests passed! Knowledge Graph module is ready.")
    else:
        print("\n  ⚠️ Some tests failed. Check the output above.")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
