
import json
import logging
import sys
import os
import random
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.getcwd())

from src.hoax_detection.outlier_detector import OutlierDetector, create_strict_detector, create_moderate_detector

def evaluate_trust_layer(input_file: str, num_valid_samples: int = 50, num_outlier_samples: int = 20):
    print(f"Loading valid data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # "Valid" documents - real news from the dataset
    valid_docs = [item.get('body_text', '') for item in data if item.get('body_text')]
    if len(valid_docs) > num_valid_samples:
        valid_docs = random.sample(valid_docs, num_valid_samples)
        
    # "Outlier" documents - completely unrelated text
    # We simulate these with obviously different topics (Recipes, Sports, K-Pop) which are distinct from "Hoax News"
    outlier_templates = [
        "Resep nasi goreng spesial. Bahan-bahan: nasi putih, telur, bawang merah, kecap manis. Cara membuat: tumis bumbu...",
        "Pertandingan sepak bola antara Manchester United melawan Liverpool berakhir dengan skor 2-0. Gol dicetak oleh...",
        "Daftar lagu terbaru dari album K-Pop yang sedang hits minggu ini. Blackpink merilis single baru...",
        "Tips merawat tanaman hias di dalam ruangan agar tetap segar dan tidak layu. Siram air secukupnya...",
        "Review laptop gaming terbaru dengan prosesor Intel Core i9 dan kartu grafis RTX 4090. Performanya sangat kencang..."
    ]
    
    outliers = []
    for _ in range(num_outlier_samples):
        outliers.append(random.choice(outlier_templates))
        
    print(f"Evaluating Trust Layer (Outlier Detection)...")
    print(f"Valid Doc Count: {len(valid_docs)}")
    print(f"Outlier Doc Count: {len(outliers)}")
    
    # Combined stream
    all_docs = valid_docs + outliers
    # Shuffle to simulate incoming stream
    random.shuffle(all_docs)
    
    # Use Moderate detector for better sensitivity on small batches
    detector = create_moderate_detector()
    
    # Run Detection
    analysis = detector.detect_outliers(all_docs)
    
    # Debug: Print Outlier Z-scores
    print("\n[DEBUG] Outlier Analysis:")
    for result in analysis.results[:10]: # Print first 10
        label = "OUTLIER" if result.text in outliers else "VALID"
        print(f"  [{label}] Z: {result.z_score:.2f} (Sim: {result.similarity_to_centroid:.2f}) -> Pred: {result.is_outlier}")
    
    # Calculate Metrics
    # True Positives: Outliers correctly identified as outliers
    # False Positives: Valid docs incorrectly identified as outliers
    # True Negatives: Valid docs correctly identified as valid
    # False Negatives: Outliers incorrectly identified as valid
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for result in analysis.results:
        is_actually_outlier = result.text in outliers
        predicted_outlier = result.is_outlier
        
        if is_actually_outlier and predicted_outlier:
            tp += 1
        elif not is_actually_outlier and predicted_outlier:
            fp += 1
        elif not is_actually_outlier and not predicted_outlier:
            tn += 1
        elif is_actually_outlier and not predicted_outlier:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(all_docs)
    
    print("\n" + "="*50)
    print("TRUST LAYER EVALUATION RESULTS (Outlier Detection)")
    print("="*50)
    print(f"Total Documents: {len(all_docs)}")
    print(f"Strict Threshold: {detector.threshold_z} sigma")
    print(f"Correctly Filtered Outliers: {tp}/{len(outliers)} ({recall*100:.1f}%)")
    print(f"Correctly Retained Valid Docs: {tn}/{len(valid_docs)} ({(tn/len(valid_docs))*100:.1f}%)")
    print(f"False Positives (Valid flagged as Outlier): {fp}")
    print(f"False Negatives (Outlier missed): {fn}")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_trust_layer("data/komdigi_hoaks.json")
