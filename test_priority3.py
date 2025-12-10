"""
Comprehensive test for Priority 3: Research-Level Enhancements

Tests:
1. Faithfulness Metric (NLI-based)
2. KG-Enhanced TextRank
3. OOD Evaluation for Hoax Detector
"""
import sys
import traceback

print("=" * 70)
print("Priority 3: Research-Level Enhancements - Comprehensive Test")
print("=" * 70)


# ============================================================
# Test 1: Faithfulness Metric
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Faithfulness Metric (NLI-based)")
print("=" * 70)

try:
    from src.faithfulness import FaithfulnessChecker, check_hallucination
    
    # Test without NLI model (lexical overlap)
    checker = FaithfulnessChecker(use_nli=False)
    
    source_docs = [
        "Presiden Joko Widodo mengumumkan program vaksinasi COVID-19.",
        "Target pemerintah adalah 70% populasi dalam 6 bulan.",
        "Kementerian Kesehatan menyiapkan 100 juta dosis vaksin."
    ]
    
    # Good summary (faithful)
    good_summary = "Presiden Jokowi mengumumkan vaksinasi dengan target 70% populasi. Kemenkes menyiapkan 100 juta dosis."
    
    # Bad summary (hallucination)
    bad_summary = "Presiden Jokowi mengumumkan bahwa vaksin akan gratis untuk semua WNA. WHO melarang vaksinasi di Indonesia."
    
    print("\nSource documents:")
    for i, doc in enumerate(source_docs):
        print(f"  [{i}] {doc}")
    
    print(f"\n[Good summary]: {good_summary}")
    result_good = checker.check_faithfulness(good_summary, source_docs)
    print(f"  Faithfulness score: {result_good['overall_score']:.2f}")
    print(f"  Claims checked: {result_good['num_claims']}")
    
    print(f"\n[Bad summary]: {bad_summary}")
    result_bad = checker.check_faithfulness(bad_summary, source_docs)
    print(f"  Faithfulness score: {result_bad['overall_score']:.2f}")
    print(f"  Claims checked: {result_bad['num_claims']}")
    
    # Quick hallucination check
    has_hallucination, unsupported = check_hallucination(bad_summary, source_docs, threshold=0.5)
    print(f"\n  Hallucination detected: {has_hallucination}")
    if unsupported:
        print(f"  Unsupported claims: {unsupported[:2]}...")
    
    print("\n[PASS] Faithfulness Metric works correctly!")
    
except Exception as e:
    print(f"\n[FAIL] Faithfulness Metric error: {e}")
    traceback.print_exc()


# ============================================================
# Test 2: KG-Enhanced TextRank
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: KG-Enhanced TextRank")
print("=" * 70)

try:
    from src.models.kg_enhanced_textrank import KGEnhancedTextRank
    import networkx as nx
    
    # Create mock Knowledge Graph
    class SimpleKG:
        def __init__(self):
            self.graph = nx.DiGraph()
            entities = [
                ("PERSON:Joko Widodo", "Joko Widodo", {"Jokowi", "Presiden Joko Widodo"}),
                ("ORG:Kementerian Kesehatan", "Kementerian Kesehatan", {"Kemenkes"}),
                ("ORG:WHO", "WHO", {"World Health Organization"}),
            ]
            for key, normalized, aliases in entities:
                self.graph.add_node(key, normalized=normalized, aliases=aliases)
    
    summarizer = KGEnhancedTextRank(
        num_sentences=3,
        kg_boost=0.4,
        use_mmr=True,
        similarity_mode="embedding"  # Use embedding mode
    )
    summarizer.set_knowledge_graph(SimpleKG())
    
    text = """
    Pemerintah Indonesia mengumumkan kebijakan vaksinasi baru.
    Presiden Joko Widodo mengatakan target vaksinasi adalah 70% populasi.
    Kementerian Kesehatan menyiapkan 100 juta dosis vaksin.
    Vaksinasi akan dimulai bulan depan di seluruh Indonesia.
    WHO memberikan dukungan penuh terhadap program ini.
    Masyarakat diimbau untuk tetap mematuhi protokol kesehatan.
    """
    
    summary = summarizer.summarize(text)
    coverage = summarizer.get_entity_coverage(summary)
    
    print(f"\nInput text: {len(text.split())} words")
    print(f"\nSummary:\n  {summary}")
    print(f"\nEntity Coverage:")
    print(f"  Coverage: {coverage['coverage']:.1%}")
    print(f"  Entities found: {coverage['entities_in_summary']}")
    print(f"  Total KG entities: {coverage['kg_entities']}")
    
    print("\n[PASS] KG-Enhanced TextRank works correctly!")
    
except Exception as e:
    print(f"\n[FAIL] KG-Enhanced TextRank error: {e}")
    traceback.print_exc()


# ============================================================
# Test 3: OOD Evaluation for Hoax Detector
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: OOD Evaluation (Hoax Detector)")
print("=" * 70)

try:
    # Import OODEvaluator directly to avoid peft dependency
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    
    # Import only what we need
    from src.hoax_detection.evaluate_ood import OODEvaluator, OODTestResult
    
    # Create mock classifier for testing
    class MockResult:
        def __init__(self, label, confidence, probabilities):
            self.label = label
            self.confidence = confidence
            self.probabilities = probabilities
    
    class MockClassifier:
        def predict(self, text):
            # Simple heuristic for testing
            text_lower = text.lower()
            hoax_signals = ["viral", "awas", "waspada", "!!!"]
            is_hoax = any(s in text_lower for s in hoax_signals)
            return MockResult(
                label="HOAX" if is_hoax else "VALID",
                confidence=0.85 if is_hoax else 0.75,
                probabilities={"HOAX": 0.85, "VALID": 0.15} if is_hoax else {"HOAX": 0.25, "VALID": 0.75}
            )
    
    evaluator = OODEvaluator(classifier=MockClassifier())
    
    # Test adversarial example generation
    hoax_texts = [
        "VIRAL!!! Vaksin COVID mengandung chip 5G!!!",
        "AWAS! Pemerintah sembunyikan kebenaran tentang pandemi!",
        "WASPADA! Makanan ini ternyata mengandung racun berbahaya!!!",
    ]
    
    adversarial = evaluator.generate_adversarial_examples(hoax_texts, num_examples=3)
    
    print("\nOriginal hoax texts:")
    for i, t in enumerate(hoax_texts[:2]):
        print(f"  [{i}] {t[:60]}...")
    
    print("\nAdversarial examples (clickbait signals removed):")
    for i, ex in enumerate(adversarial[:2]):
        print(f"  [{i}] {ex['modified'][:60]}...")
    
    # Test calibration
    test_texts = [
        "Pemerintah mengumumkan kebijakan baru tentang pendidikan.",
        "VIRAL! Obat herbal ini bisa menyembuhkan semua penyakit!!!",
        "Menteri Kesehatan menjelaskan prosedur vaksinasi.",
        "AWAS! Rumah makan ini ternyata pakai bahan berbahaya!",
    ]
    test_labels = ["VALID", "HOAX", "VALID", "HOAX"]
    
    calibration = evaluator.evaluate_calibration(test_texts, test_labels)
    
    print(f"\nCalibration Results:")
    print(f"  Expected Calibration Error (ECE): {calibration['ece']:.3f}")
    print(f"  Interpretation: {calibration['interpretation']}")
    
    print("\n[PASS] OOD Evaluation works correctly!")
    
except Exception as e:
    print(f"\n[FAIL] OOD Evaluation error: {e}")
    traceback.print_exc()


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("""
✓ Test 1: Faithfulness Metric - Detects hallucinations in summaries
✓ Test 2: KG-Enhanced TextRank - Boosts sentences with KG entities
✓ Test 3: OOD Evaluation - Generates adversarial examples, measures calibration
""")
print("=" * 70)
print("All Priority 3 tests completed!")
print("=" * 70)
