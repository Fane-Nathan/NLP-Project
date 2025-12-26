
import json
import logging
import sys
import os
import random
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.getcwd())

from src.models.knowledge_graph import KnowledgeGraph
from src.models.fact_verifier import FactVerifier, VerificationStatus
from src.models.textrank import TextRankSummarizer

# Configure logging
logging.basicConfig(level=logging.ERROR)

def evaluate_verification(input_file: str, num_samples: int = 50):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for items with body_text
    data = [item for item in data if item.get('body_text')]
    
    # Sample
    if len(data) > num_samples:
        data = random.sample(data, num_samples)
    
    print(f"Evaluating Verification on {len(data)} samples...")
    print("Building Knowledge Graph and Generating Summaries...")
    
    kg = KnowledgeGraph(name="eval_kg")
    verifier = FactVerifier(kg)
    summarizer = TextRankSummarizer()
    
    verification_rates = []
    hallucination_free_counts = 0
    
    # We will simulate the process:
    # 1. Build KG from the document (Simulate "Perfect" KG from source)
    # 2. Generate Summary
    # 3. Verify Summary against KG
    
    for item in tqdm(data):
        text = item.get('body_text', '')
        if not text:
            continue
            
        # 1. Build KG for this document
        # In a real scenario, KG is built from multiple docs. 
        # Here we verify if the summary is faithful to the SOURCE document.
        # We clear the KG for each document to test individual summary faithfulness.
        kg = KnowledgeGraph(name=f"kg_{item.get('id')}")
        kg.add_documents([text], show_progress=False)
        verifier.kg = kg # specific ID
        
        # 2. Generate Summary (Extractive - should be high verification)
        summary = summarizer.summarize(text, num_sentences=3)
        
        # 3. Verify
        report = verifier.verify_summary(summary)
        
        verification_rates.append(report.verification_rate)
        
        # Hallucination Free if NO hallucinations detected
        if report.hallucination_count == 0:
            hallucination_free_counts += 1
            
    avg_verification_rate = np.mean(verification_rates)
    hallucination_free_rate = hallucination_free_counts / len(data)
    
    print("\n" + "="*50)
    print("VERIFICATION EVALUATION RESULTS")
    print("="*50)
    print(f"Algorithm: TextRank (Extractive)")
    print(f"Samples: {len(data)}")
    print(f"Verification Rate: {avg_verification_rate:.4f} ({avg_verification_rate*100:.1f}%)")
    print(f"Hallucination-Free Rate: {hallucination_free_rate:.4f} ({hallucination_free_rate*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    evaluate_verification("data/komdigi_hoaks.json", num_samples=30)
