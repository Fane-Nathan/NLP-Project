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
        choices=['summarize', 'kg', 'verify', 'credibility', 'full'],
        required=True,
        help="Operation mode"
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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 TRUST-DRIVEN SUMMARIZATION MODEL (TDSM)")
    print("   with Knowledge Graph Verification")
    print("=" * 70)
    
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
                summarizer = TextRankSummarizer(num_sentences=args.num_sentences)
                summary = summarizer.summarize(combined_text)
            elif args.model == "lexrank":
                summarizer = LexRankSummarizer(num_sentences=args.num_sentences)
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