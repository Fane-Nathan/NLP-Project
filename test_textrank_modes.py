"""
Test script to compare TextRank summarizer configurations.
Evaluates TF-IDF vs Embedding modes and MMR redundancy removal.
"""
import random
random.seed(42)

from src.models.textrank import TextRankSummarizer
from src.evaluation import Evaluator

def test_semantic_redundancy():
    """Test case with semantic redundancy (same meaning, different words)."""
    docs = [
        'Presiden Joko Widodo mengumumkan kebijakan vaksinasi. Target 70 persen populasi.',
        'Kepala negara memberitahukan program imunisasi. Sasaran tujuh puluh persen rakyat.',  # Semantic duplicate!
        'Kementerian Kesehatan menyiapkan 100 juta dosis. Prioritas tenaga medis.',
        'WHO memuji langkah Indonesia dalam pandemi. Tedros memberikan apresiasi.',
    ]

    reference = 'Presiden mengumumkan program vaksinasi dengan target 70 persen populasi. Kemenkes menyiapkan 100 juta dosis untuk tenaga medis. WHO memuji Indonesia.'

    combined_text = ' '.join(docs)
    evaluator = Evaluator()

    print('=' * 70)
    print('Test 1: Semantic Redundancy Detection')
    print('=' * 70)
    print('TF-IDF cannot detect: "Presiden" vs "Kepala negara"')
    print('Embeddings SHOULD detect these are semantically similar!')
    print()

    configs = [
        ('TF-IDF only', 'tfidf', False, False),
        ('TF-IDF + MMR (tfidf)', 'tfidf', True, False),
        ('TF-IDF + MMR (embed)', 'tfidf', True, True),
        ('Embedding only', 'embedding', False, False),
        ('Embedding + MMR', 'embedding', True, True),
    ]

    results = []
    for name, mode, mmr, mmr_embed in configs:
        tr = TextRankSummarizer(
            num_sentences=3,
            similarity_mode=mode,
            use_mmr=mmr,
            mmr_use_embedding=mmr_embed,
            mmr_lambda=0.5
        )
        summary = tr.summarize(combined_text)
        scores = evaluator.evaluate_single(reference, summary)
        r1 = scores['rouge1']['fmeasure']
        r2 = scores['rouge2']['fmeasure']
        rl = scores['rougeL']['fmeasure']
        
        results.append((name, r1, r2, rl, summary))
        print(f'{name:25s}: R1={r1:.3f}, R2={r2:.3f}, RL={rl:.3f}')
        print(f'  -> {summary[:100]}...')
        print()
    
    return results


def test_multi_document():
    """Test multi-document summarization with source tracking."""
    docs = [
        'Jakarta melaporkan kenaikan kasus COVID-19 sebesar 15 persen minggu ini.',
        'Pemerintah DKI Jakarta memperketat protokol kesehatan di tempat umum.',
        'Rumah sakit di Jakarta mulai kewalahan dengan lonjakan pasien.',
        'Gubernur Anies mengimbau warga untuk tetap di rumah.',
    ]
    
    reference = 'Kasus COVID Jakarta naik 15 persen. Pemerintah perketat protokol kesehatan. RS kewalahan. Gubernur imbau warga di rumah.'
    
    evaluator = Evaluator()
    
    print('=' * 70)
    print('Test 2: Multi-Document Summarization with Source Tracking')
    print('=' * 70)
    print()
    
    tr = TextRankSummarizer(
        num_sentences=3,
        similarity_mode='embedding',
        use_mmr=True,
        mmr_use_embedding=True,
        mmr_lambda=0.6
    )
    
    # Test multi-document with source tracking
    results = tr.summarize_multi_with_sources(docs)
    
    # Extract summary text
    summary = ' '.join([r['sentence'] for r in results])
    
    scores = evaluator.evaluate_single(reference, summary)
    r1 = scores['rouge1']['fmeasure']
    r2 = scores['rouge2']['fmeasure']
    rl = scores['rougeL']['fmeasure']
    
    print(f'Embedding + MMR (multi-doc): R1={r1:.3f}, R2={r2:.3f}, RL={rl:.3f}')
    print(f'Summary: {summary}')
    print()
    print('Source Tracking:')
    for r in results:
        print(f"  Doc {r['source_doc_index']}: {r['sentence'][:50]}...")
    print()


def test_lambda_variations():
    """Test different MMR lambda values."""
    docs = [
        'Presiden Joko Widodo melakukan kunjungan kerja ke Papua.',
        'Jokowi bertemu dengan tokoh adat Papua di Jayapura.',
        'Pemerintah berkomitmen percepat pembangunan infrastruktur Papua.',
        'Bandara baru di Papua akan selesai tahun depan.',
    ]
    
    reference = 'Presiden Jokowi kunjungi Papua dan bertemu tokoh adat. Pemerintah percepat pembangunan infrastruktur termasuk bandara baru.'
    
    combined_text = ' '.join(docs)
    evaluator = Evaluator()
    
    print('=' * 70)
    print('Test 3: MMR Lambda Variations (0.3 to 0.9)')
    print('=' * 70)
    print('Lambda 0.3 = more diversity, Lambda 0.9 = more relevance')
    print()
    
    for lam in [0.3, 0.5, 0.7, 0.9]:
        tr = TextRankSummarizer(
            num_sentences=3,
            similarity_mode='embedding',
            use_mmr=True,
            mmr_use_embedding=True,
            mmr_lambda=lam
        )
        summary = tr.summarize(combined_text)
        scores = evaluator.evaluate_single(reference, summary)
        r1 = scores['rouge1']['fmeasure']
        
        print(f'Lambda={lam}: R1={r1:.3f} -> {summary[:70]}...')
    print()


if __name__ == '__main__':
    print('\n' + '=' * 70)
    print('TextRank Summarizer - Mode Comparison Tests')
    print('=' * 70 + '\n')
    
    test_semantic_redundancy()
    test_multi_document()
    test_lambda_variations()
    
    print('=' * 70)
    print('All tests completed!')
    print('=' * 70)
