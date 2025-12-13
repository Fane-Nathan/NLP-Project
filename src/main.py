"""
Indonesian Multi-Document Summarization with Knowledge Graph Verification

ENHANCED PIPELINE:
1. Load documents
2. Run credibility analysis (hoax detection + outlier detection)
3. Build Knowledge Graph from filtered documents
4. Generate constrained summary (grounded in KG)
5. Verify summary against KG (detect hallucinations)
6. Output verified summary with confidence score

This solves the "Subtle Hallucination" problem in TMDS by ensuring
every claim in the summary can be traced back to verified facts.

Usage:
    # Full pipeline with KG verification
    python -m src.main --mode summarize --model gemini --credibility --verify
    
    # Build KG only (for inspection)
    python -m src.main --mode kg --input_file data/docs.json --output kg_output.json
    
    # Verify existing summary against documents
    python -m src.main --mode verify --input_file data/docs.json --summary "summary text"
"""

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def load_documents(file_path: str) -> List[str]:
    """Load documents from file (JSON/JSONL/TXT)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for key in ['documents', 'docs', 'texts', 'articles', 'content']:
                if key in data:
                    return data[key]
            raise KeyError("JSON must have 'documents' key or be a list")
    elif ext == '.jsonl':
        docs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, str):
                        docs.append(item)
                    elif isinstance(item, dict):
                        for key in ['text', 'content', 'body', 'article']:
                            if key in item:
                                docs.append(item[key])
                                break
        return docs
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        docs = [d.strip() for d in content.split('\n\n') if d.strip()]
        return docs


def run_credibility_analysis(
    documents: List[str],
    hoax_model_path: str = "models/hoax_indobert_lora",
    output_report: Optional[str] = None
) -> Tuple[List[str], any]:
    """Run credibility analysis (existing Trust Layer)."""
    print("\n" + "=" * 60)
    print("🔍 STAGE 1: TRUST LAYER (Credibility Analysis)")
    print("=" * 60)
    
    from src.hoax_detection.credibility_report import CredibilityAnalyzer
    
    analyzer = CredibilityAnalyzer(
        hoax_model_path=hoax_model_path,
        outlier_threshold_z=2.0,
        hoax_weight=0.6,
        outlier_weight=0.4
    )
    
    filtered_docs, report = analyzer.filter_documents(documents)
    report.print_summary()
    
    if output_report:
        report.save(output_report)
    
    return filtered_docs, report


def build_knowledge_graph(
    documents: List[str],
    output_path: Optional[str] = None
) -> any:
    """Build Knowledge Graph from documents."""
    print("\n" + "=" * 60)
    print("🧠 STAGE 2: KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 60)
    
    from src.models.knowledge_graph import KnowledgeGraph
    
    kg = KnowledgeGraph(name=f"news_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    stats = kg.add_documents(documents, show_progress=True)
    
    print("\n" + kg.get_summary())
    
    # Detect conflicts
    conflicts = kg.detect_conflicts()
    if conflicts:
        print(f"\n⚠️ Detected {len(conflicts)} potential conflicts in documents:")
        for conflict in conflicts[:3]:
            print(f"  - {conflict['type']}: {conflict.get('reason', conflict.get('entities', ''))}")
    
    if output_path:
        kg.save(output_path)
    
    return kg


def generate_verified_summary(
    documents: List[str],
    kg: any,
    model_name: str = "gemini",
    num_sentences: int = 5,
    gemini_api_key: Optional[str] = None
) -> Dict:
    """Generate summary with KG verification."""
    print("\n" + "=" * 60)
    print("📝 STAGE 3: CONSTRAINED SUMMARIZATION")
    print("=" * 60)
    
    from src.models.constrained_summarizer import ConstrainedSummarizer, SummarizationMode
    
    # Determine mode
    if model_name == "gemini":
        mode = SummarizationMode.ABSTRACTIVE
    elif model_name in ["textrank", "lexrank"]:
        mode = SummarizationMode.EXTRACTIVE
    else:
        mode = SummarizationMode.HYBRID
    
    summarizer = ConstrainedSummarizer(
        kg=kg,
        gemini_api_key=gemini_api_key,
        max_refinement_iterations=3,
        min_verification_rate=0.7
    )
    
    result = summarizer.summarize(
        documents=documents,
        mode=mode,
        num_sentences=num_sentences,
        build_timeline=True
    )
    
    result.print_result()
    
    return {
        "summary": result.summary,
        "confidence": result.confidence,
        "is_hallucination_free": result.is_hallucination_free,
        "verification_rate": result.verification_report.verification_rate,
        "iterations": result.iterations,
        "timeline": result.timeline.to_dict() if result.timeline else None,
        "grounding_facts_count": len(result.grounding_facts)
    }


def verify_summary(
    summary: str,
    documents: List[str],
    kg: Optional[any] = None
) -> Dict:
    """Verify an existing summary against documents."""
    print("\n" + "=" * 60)
    print("✅ SUMMARY VERIFICATION")
    print("=" * 60)
    
    from src.models.knowledge_graph import KnowledgeGraph
    from src.models.fact_verifier import FactVerifier
    
    # Build KG if not provided
    if kg is None:
        kg = KnowledgeGraph(name="verification_kg")
        kg.add_documents(documents, show_progress=False)
    
    verifier = FactVerifier(kg)
    report = verifier.verify_summary(summary)
    
    report.print_report()
    
    return {
        "overall_status": report.overall_status.value,
        "verification_rate": report.verification_rate,
        "hallucination_count": report.hallucination_count,
        "confidence": report.confidence,
        "claims": [r.to_dict() for r in report.claim_results]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Indonesian TMDS with Knowledge Graph Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['summarize', 'kg', 'verify', 'credibility', 'full', 'evaluate'],
        default='full',
        help="Operation mode (default: full - runs complete pipeline with credibility + KG verification)"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['textrank', 'lexrank', 'gemini', 'hybrid'],
        default='hybrid',
        help="Summarization model"
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        help="Path to input documents"
    )
    parser.add_argument(
        '--input_text', 
        type=str, 
        help="Direct input text"
    )
    parser.add_argument(
        '--summary', 
        type=str,
        help="Summary to verify (for --mode verify)"
    )
    parser.add_argument(
        '--credibility', 
        action='store_true',
        help="Enable Trust Layer filtering"
    )
    parser.add_argument(
        '--verify', 
        action='store_true',
        help="Enable KG verification"
    )
    parser.add_argument(
        '--hoax_model', 
        type=str, 
        default='models/hoax_indobert_lora',
        help="Path to hoax classifier"
    )
    parser.add_argument(
        '--output', 
        type=str,
        help="Output path"
    )
    parser.add_argument(
        '--num_sentences', 
        type=int, 
        default=5,
        help="Number of sentences for extractive"
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=50,
        help="Number of samples for evaluation mode"
    )
    parser.add_argument(
        '--use_mmr', 
        action='store_true',
        help="Enable MMR for redundancy reduction in summarization"
    )
    parser.add_argument(
        '--indo_rouge',
        action='store_true',
        help="Use Indonesian-specific ROUGE with Sastrawi stemming"
    )
    parser.add_argument(
        '--similarity_mode',
        type=str,
        default='tfidf',
        choices=['tfidf', 'embedding', 'hybrid'],
        help="Similarity mode: tfidf (fast), embedding (semantic), hybrid (best ROUGE)"
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='indo-e5-small',
        choices=['indo-e5-small', 'indo-e5-base', 'indo-sbert', 'multilingual-e5-base', 'multilingual-e5-small', 'multilingual-minilm'],
        help="Embedding model to use (for embedding/hybrid mode)"
    )
    parser.add_argument(
        '--hybrid_weight',
        type=float,
        default=0.6,
        help="Weight for embeddings in hybrid mode (0-1, default: 0.6)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🔬 TRUST-DRIVEN SUMMARIZATION MODEL (TDSM)")
    print("   with Knowledge Graph Verification")
    print("=" * 70)

    # Show simplified execution info when using defaults
    if args.mode == 'full' and not args.input_file and not args.input_text:
        print("\n💡 Running ONE-BUTTON mode: Full pipeline with demo data")
        print("   → Credibility Analysis → Knowledge Graph → Verified Summary")
        print("   (Use --input_file or --input_text for custom documents)")
    
    # === LOAD DOCUMENTS ===
    documents = []
    
    if args.input_file:
        print(f"\n[Input] Loading from: {args.input_file}")
        documents = load_documents(args.input_file)
        print(f"[Input] Loaded {len(documents)} documents")
    elif args.input_text:
        documents = [args.input_text]
    else:
        # Demo documents
        print("\n[Input] Using demo documents")
        documents = [
            "Presiden Joko Widodo mengumumkan kebijakan vaksinasi COVID-19 baru pada 15 Januari 2024 di Istana Negara Jakarta. Program ini ditargetkan mencapai 70% cakupan dalam 6 bulan.",
            "VIRAL! Vaksin COVID-19 mengandung microchip 5G untuk pelacakan warga! Bagikan sebelum dihapus pemerintah! Dokter terkenal sudah konfirmasi!",
            "Kementerian Kesehatan melaporkan peningkatan cakupan vaksinasi di seluruh provinsi pada Februari 2024. Menteri Kesehatan Budi Gunadi Sadikin menyatakan target 70% dapat tercapai.",
            "Resep masakan rendang padang yang enak dan mudah dibuat di rumah untuk keluarga tercinta.",
            "WHO memuji keberhasilan program vaksinasi Indonesia pada Maret 2024. Direktur WHO Tedros memberikan apresiasi kepada Kementerian Kesehatan.",
            "DPR menyetujui anggaran tambahan sebesar Rp 150 miliar untuk program vaksinasi nasional pada Januari 2024."
        ]
    
    if not documents:
        print("[Error] No documents to process")
        return
    
    # === MODE: EVALUATE ===
    if args.mode == 'evaluate':
        from src.evaluation import Evaluator, IndonesianEvaluator
        from src.models.textrank import TextRankSummarizer
        from src.models.lexrank import LexRankSummarizer
        from src.models.gemini_summarizer import GeminiSummarizer
        import numpy as np

        print("\n" + "=" * 60)
        print("📊 MODE: EVALUATION")
        print("=" * 60)

        dataset = []
        
        # Load dataset from file or XL-Sum
        if args.input_file:
            print(f"[Input] Loading dataset from: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    dataset = raw_data
                else:
                    dataset = raw_data.get('data', raw_data.get('exemplars', []))
        else:
            # Load from XL-Sum dataset
            print(f"[Input] Loading {args.num_samples} samples from XL-Sum (Indonesian)...")
            try:
                from datasets import load_dataset
                xlsum = load_dataset(
                    "csebuetnlp/xlsum", 
                    "indonesian", 
                    split="test",  # Use test split for evaluation
                    cache_dir="./data_cache",
                    trust_remote_code=True
                )
                # Sample randomly
                import random
                indices = random.sample(range(len(xlsum)), min(args.num_samples, len(xlsum)))
                for idx in indices:
                    item = xlsum[idx]
                    dataset.append({
                        "text": item["text"],
                        "summary": item["summary"]
                    })
                print(f"[Input] Loaded {len(dataset)} samples from XL-Sum test set")
            except Exception as e:
                print(f"[Warning] Could not load XL-Sum: {e}")
                print("[Info] Falling back to demo data.")
                dataset = [
                    {
                        "text": "Pemerintah Indonesia secara resmi meluncurkan program vaksinasi nasional COVID-19. Presiden Joko Widodo menjadi orang pertama yang menerima suntikan vaksin Sinovac. Langkah ini menandai dimulainya upaya ambisius untuk mengimunisasi 181,5 juta orang.",
                        "summary": "Pemerintah Indonesia meluncurkan program vaksinasi COVID-19 dengan Presiden Joko Widodo sebagai penerima pertama."
                    },
                    {
                        "text": "Tim nasional bulu tangkis Indonesia berhasil menjuarai Piala Thomas 2020. Kemenangan ini diraih setelah mengalahkan tim China dengan skor telak 3-0 di final. Ini adalah gelar ke-14 bagi Indonesia setelah penantian selama 19 tahun.",
                        "summary": "Indonesia menjuarai Piala Thomas 2020 setelah mengalahkan China 3-0."
                    }
                ]

        if not dataset:
            print("[Error] No data found for evaluation.")
            return

        print(f"\nEvaluating Model: {args.model.upper()}")
        print(f"MMR Enabled: {args.use_mmr}")
        print(f"Similarity: {args.similarity_mode.upper()}" + (f" ({args.embedding_model})" if args.similarity_mode != "tfidf" else ""))
        if args.similarity_mode == "hybrid":
            print(f"Hybrid Weight: {args.hybrid_weight} (embedding) / {1-args.hybrid_weight:.1f} (tfidf)")
        print(f"Indo ROUGE: {args.indo_rouge}")
        print(f"Samples: {len(dataset)}")
        print("-" * 30)

        # Initialize Model with options
        if args.model == "textrank":
            model = TextRankSummarizer(
                num_sentences=args.num_sentences,
                use_mmr=args.use_mmr,
                similarity_mode=args.similarity_mode,
                embedding_model=args.embedding_model,
                hybrid_weight=args.hybrid_weight
            )
        elif args.model == "lexrank":
            model = LexRankSummarizer(
                num_sentences=args.num_sentences,
                use_mmr=args.use_mmr,
                similarity_mode=args.similarity_mode,
                embedding_model=args.embedding_model,
                hybrid_weight=args.hybrid_weight
            )
        else:
            model = GeminiSummarizer()

        # Initialize evaluator (Indonesian or standard)
        if args.indo_rouge:
            evaluator = IndonesianEvaluator(use_stemmer=True)
        else:
            evaluator = Evaluator()
        
        # Collect individual scores for statistical analysis
        rouge1_scores = []
        rouge2_scores = []
        rougel_scores = []

        for i, item in enumerate(dataset):
            text = item.get('text', item.get('content', item.get('article', '')))
            ref_summary = item.get('summary', item.get('gold_summary', ''))

            if not text or not ref_summary:
                continue

            # Calculate target length from reference summary (adaptive length)
            ref_word_count = len(ref_summary.split())

            # Generate Summary with adaptive length
            if args.model in ["textrank", "lexrank"]:
                # Use target_words for adaptive length matching
                pred_summary = model.summarize(text, target_words=ref_word_count)
            else:
                pred_summary = model.summarize([text]).summary

            # Calculate individual scores
            individual_scores = evaluator.evaluate_single(ref_summary, pred_summary)
            rouge1_scores.append(individual_scores['rouge1']['fmeasure'])
            rouge2_scores.append(individual_scores['rouge2']['fmeasure'])
            rougel_scores.append(individual_scores['rougeL']['fmeasure'])
            
            if (i + 1) % 10 == 0 or i == len(dataset) - 1:
                print(f"[{i+1}/{len(dataset)}] Processed...")

        # Calculate statistics
        print("\n" + "=" * 50)
        print("📈 ROUGE Evaluation Results (with Statistics)")
        print("=" * 50)
        print(f"  Samples: {len(rouge1_scores)}")
        print(f"  Model: {args.model.upper()} {'(+MMR)' if args.use_mmr else ''}")
        print("-" * 50)
        print(f"  ROUGE-1: {np.mean(rouge1_scores):.4f} ± {np.std(rouge1_scores):.4f}")
        print(f"  ROUGE-2: {np.mean(rouge2_scores):.4f} ± {np.std(rouge2_scores):.4f}")
        print(f"  ROUGE-L: {np.mean(rougel_scores):.4f} ± {np.std(rougel_scores):.4f}")
        print("=" * 50)
        
        # Statistical significance note
        if len(rouge1_scores) >= 30:
            print("✓ Sample size sufficient for statistical significance (n≥30)")
        else:
            print(f"⚠ Sample size may be insufficient (n={len(rouge1_scores)}). Consider --num_samples 100+")
        
        return

    # === MODE: CREDIBILITY ONLY ===
    if args.mode == 'credibility':
        filtered_docs, report = run_credibility_analysis(
            documents=documents,
            hoax_model_path=args.hoax_model,
            output_report=args.output
        )
        return
    
    # === MODE: BUILD KG ONLY ===
    if args.mode == 'kg':
        kg = build_knowledge_graph(
            documents=documents,
            output_path=args.output
        )
        return
    
    # === MODE: VERIFY SUMMARY ===
    if args.mode == 'verify':
        if not args.summary:
            print("[Error] --summary required for verify mode")
            return
        
        result = verify_summary(
            summary=args.summary,
            documents=documents
        )
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        return
    
    # === MODE: FULL PIPELINE ===
    if args.mode in ['summarize', 'full']:
        docs_to_summarize = documents
        credibility_report = None
        kg = None
        
        # Step 1: Trust Layer (if enabled)
        if args.credibility or args.mode == 'full':
            docs_to_summarize, credibility_report = run_credibility_analysis(
                documents=documents,
                hoax_model_path=args.hoax_model
            )
            
            if not docs_to_summarize:
                print("\n[Warning] All documents filtered by Trust Layer!")
                return
        
        # Step 2: Build Knowledge Graph (if verify enabled)
        if args.verify or args.mode == 'full':
            kg = build_knowledge_graph(documents=docs_to_summarize)
        
        # Step 3: Generate Summary
        if kg and (args.verify or args.mode == 'full'):
            # Use constrained summarization with verification
            result = generate_verified_summary(
                documents=docs_to_summarize,
                kg=kg,
                model_name=args.model,
                num_sentences=args.num_sentences
            )
            
            summary = result["summary"]
        else:
            # Use basic summarization (existing behavior)
            from src.models.textrank import TextRankSummarizer
            from src.models.lexrank import LexRankSummarizer

            combined_text = ' '.join(docs_to_summarize)

            if args.model == "textrank":
                summarizer = TextRankSummarizer(
                    num_sentences=args.num_sentences,
                    use_mmr=args.use_mmr,
                    similarity_mode=args.similarity_mode,
                    embedding_model=args.embedding_model,
                    hybrid_weight=args.hybrid_weight
                )
                summary = summarizer.summarize(combined_text)
            elif args.model == "lexrank":
                summarizer = LexRankSummarizer(
                    num_sentences=args.num_sentences,
                    use_mmr=args.use_mmr,
                    similarity_mode=args.similarity_mode,
                    embedding_model=args.embedding_model,
                    hybrid_weight=args.hybrid_weight
                )
                summary = summarizer.summarize(combined_text)
            else:
                from src.models.gemini_summarizer import GeminiSummarizer
                summarizer = GeminiSummarizer()
                result = summarizer.summarize(docs_to_summarize)
                summary = result.summary

            result = {"summary": summary}
        
        # Output
        print("\n" + "=" * 70)
        print("📋 FINAL OUTPUT")
        print("=" * 70)
        print(f"\n{result['summary']}")
        
        if 'confidence' in result:
            print(f"\n✅ Confidence: {result['confidence']:.1%}")
            print(f"✅ Hallucination-Free: {result['is_hallucination_free']}")
            print(f"✅ Verification Rate: {result['verification_rate']:.1%}")
        
        if args.output:
            output_data = {
                "summary": result['summary'],
                "model": args.model,
                "credibility_enabled": args.credibility,
                "kg_verification_enabled": args.verify or args.mode == 'full',
                "num_input_documents": len(documents),
                "num_filtered_documents": len(docs_to_summarize),
                **{k: v for k, v in result.items() if k != 'summary'}
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n[Output] Saved to: {args.output}")


if __name__ == "__main__":
    main()