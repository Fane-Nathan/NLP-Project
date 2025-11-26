"""
Indonesian Multi-Document Summarization with Credibility Analysis

This is the main entry point for the summarization system.
Integrates hoax detection and outlier analysis BEFORE summarization.

Pipeline:
1. Load documents
2. [NEW] Run credibility analysis (parallel hoax detection + outlier detection)
3. [NEW] Generate credibility report
4. Filter documents based on credibility
5. Summarize filtered documents
6. Output results

Usage:
    # Summarize with credibility check
    python -m src.main --mode summarize --model textrank --input_file data/docs.json --credibility
    
    # Evaluate on test set
    python -m src.main --mode evaluate --model lexrank
    
    # Train hoax detector first
    python -m src.hoax_detection.train_lora --data_path data/turnbackhoax.csv
"""

import argparse
import os
import json
from typing import List, Dict, Optional

def load_documents(file_path: str) -> List[str]:
    """
    Load documents from file.
    
    Supports:
    - JSON: {"documents": ["doc1", "doc2", ...]}
    - JSON Lines: One document per line
    - Plain text: Documents separated by blank lines
    
    Args:
        file_path: Path to document file.
        
    Returns:
        List of document strings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys
            for key in ['documents', 'docs', 'texts', 'articles', 'content']:
                if key in data:
                    return data[key]
            raise KeyError("JSON must have 'documents' key or be a list")
        else:
            raise ValueError("Invalid JSON format")
            
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
                        # Try to find text field
                        for key in ['text', 'content', 'body', 'article']:
                            if key in item:
                                docs.append(item[key])
                                break
        return docs
        
    else:  # Plain text
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines
        docs = [d.strip() for d in content.split('\n\n') if d.strip()]
        return docs


def run_credibility_analysis(
    documents: List[str],
    hoax_model_path: str = "models/hoax_indobert_lora",
    output_report: Optional[str] = None
) -> tuple:
    """
    Run credibility analysis on documents.
    
    Args:
        documents: List of document texts.
        hoax_model_path: Path to trained hoax classifier.
        output_report: Optional path to save credibility report.
        
    Returns:
        Tuple of (filtered_documents, credibility_report).
    """
    print("\n" + "=" * 60)
    print("🔍 CREDIBILITY ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    from src.hoax_detection import CredibilityAnalyzer, CredibilityReport
    
    analyzer = CredibilityAnalyzer(
        hoax_model_path=hoax_model_path,
        outlier_threshold_z=2.0,  # STRICT
        hoax_weight=0.6,
        outlier_weight=0.4,
        credibility_threshold=0.5
    )
    
    # Run analysis
    filtered_docs, report = analyzer.filter_documents(documents)
    
    # Print summary
    report.print_summary()
    
    # Save report if requested
    if output_report:
        report.save(output_report)
    
    return filtered_docs, report


def summarize_documents(
    documents: List[str],
    model_name: str = "textrank",
    num_sentences: int = 5
) -> str:
    """
    Summarize documents using specified model.
    
    Args:
        documents: List of document texts.
        model_name: "textrank" or "lexrank".
        num_sentences: Number of sentences in summary.
        
    Returns:
        Summary string.
    """
    print(f"\n[Summarization] Using {model_name.upper()} model...")
    
    # Initialize model
    if model_name == "textrank":
        from src.models.textrank import TextRankSummarizer
        summarizer = TextRankSummarizer(num_sentences=num_sentences)
    elif model_name == "lexrank":
        from src.models.lexrank import LexRankSummarizer
        summarizer = LexRankSummarizer(num_sentences=num_sentences)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Combine documents
    combined_text = ' '.join(documents)
    
    # Generate summary
    summary = summarizer.summarize(combined_text, num_sentences)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Indonesian Multi-Document Summarization with Credibility Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See README.md for usage examples."
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['summarize', 'evaluate', 'credibility'], 
        required=True,
        help="Mode of operation"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['textrank', 'lexrank'],
        default='textrank',
        help="Summarization model to use"
    )
    parser.add_argument(
        '--input_text', 
        type=str, 
        help="Direct input text (single document)"
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        help="Path to input file (JSON/JSONL/TXT)"
    )
    parser.add_argument(
        '--credibility', 
        action='store_true',
        help="Enable credibility analysis before summarization"
    )
    parser.add_argument(
        '--hoax_model', 
        type=str, 
        default='models/hoax_indobert_lora',
        help="Path to trained hoax classifier"
    )
    parser.add_argument(
        '--report', 
        type=str,
        help="Path to save credibility report (JSON)"
    )
    parser.add_argument(
        '--num_sentences', 
        type=int, 
        default=5,
        help="Number of sentences in summary"
    )
    parser.add_argument(
        '--output', 
        type=str,
        help="Path to save output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Indonesian Multi-Document Summarization System")
    print("with Research-Grade Credibility Analysis")
    print("=" * 60)
    
    # === LOAD DOCUMENTS ===
    documents = []
    
    if args.input_file:
        print(f"\n[Input] Loading from: {args.input_file}")
        documents = load_documents(args.input_file)
        print(f"[Input] Loaded {len(documents)} documents")
        
    elif args.input_text:
        print(f"\n[Input] Using provided text")
        documents = [args.input_text]
        
    else:
        # Demo documents
        print("\n[Input] Using demo documents (no input provided)")
        documents = [
            "Pemerintah Indonesia mengumumkan kebijakan baru tentang vaksinasi COVID-19 untuk seluruh masyarakat. Program ini ditargetkan menjangkau 70% populasi dalam waktu enam bulan ke depan.",
            "VIRAL! Vaksin COVID-19 ternyata mengandung microchip 5G untuk melacak warga! Bagikan sebelum dihapus pemerintah! Dokter terkenal sudah konfirmasi!",
            "Kementerian Kesehatan melaporkan peningkatan signifikan cakupan vaksinasi di seluruh provinsi. Data terbaru menunjukkan lebih dari 50 juta warga telah divaksinasi.",
            "Resep masakan rendang padang yang enak dan mudah dibuat di rumah untuk keluarga tercinta.",
            "WHO memberikan dukungan penuh terhadap program vaksinasi Indonesia dan memuji strategi pemerintah dalam menangani pandemi."
        ]
    
    if not documents:
        print("[Error] No documents to process")
        return
    
    # === MODE: CREDIBILITY ONLY ===
    if args.mode == 'credibility':
        filtered_docs, report = run_credibility_analysis(
            documents=documents,
            hoax_model_path=args.hoax_model,
            output_report=args.report
        )
        
        # Print detailed results
        print("\n" + "=" * 60)
        print("📄 DOCUMENT DETAILS")
        print("=" * 60)
        
        for doc in report.documents:
            status = "❌ EXCLUDED" if doc.doc_index in report.documents_excluded else "✅ APPROVED"
            print(f"\n[Doc {doc.doc_index}] {status}")
            print(f"  Level: {doc.credibility_level.value}")
            print(f"  Score: {doc.credibility_score:.3f}")
            print(f"  Hoax: {doc.hoax_label} ({doc.hoax_probability:.1%})")
            print(f"  Outlier: {'Yes' if doc.is_outlier else 'No'}")
            print(f"  Text: {doc.text[:80]}...")
        
        return
    
    # === MODE: SUMMARIZE ===
    if args.mode == 'summarize':
        docs_to_summarize = documents
        report = None
        
        # Run credibility analysis if enabled
        if args.credibility:
            docs_to_summarize, report = run_credibility_analysis(
                documents=documents,
                hoax_model_path=args.hoax_model,
                output_report=args.report
            )
            
            if not docs_to_summarize:
                print("\n[Warning] All documents were filtered out by credibility analysis!")
                print("[Warning] No summary generated.")
                return
            
            print(f"\n[Filter] {len(docs_to_summarize)} of {len(documents)} documents passed credibility check")
        
        # Generate summary
        summary = summarize_documents(
            documents=docs_to_summarize,
            model_name=args.model,
            num_sentences=args.num_sentences
        )
        
        # Output results
        print("\n" + "=" * 60)
        print("📝 GENERATED SUMMARY")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        
        # Save if requested
        if args.output:
            result = {
                "summary": summary,
                "model": args.model,
                "num_input_documents": len(documents),
                "num_filtered_documents": len(docs_to_summarize),
                "credibility_enabled": args.credibility
            }
            
            if report:
                result["credibility_report"] = report.to_dict()
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n[Output] Saved to: {args.output}")
        
        return
    
    # === MODE: EVALUATE ===
    if args.mode == 'evaluate':
        print("\n[Evaluate] Evaluation pipeline requires labeled test data.")
        print("[Evaluate] Provide a test file with 'reference' and 'documents' fields.")
        print("[Evaluate] Not yet fully implemented.")
        return


if __name__ == "__main__":
    main()