"""
Enhanced Hoax Detection Dataset Preparation

Downloads and combines multiple Indonesian hoax/fake news datasets:
1. HuggingFace: nlp-brin-id/fakenews-mafindo (2018-2023)
2. Local: data/cleaned/*.xlsx files
3. Local: data/hoax_dataset.csv

Output: data/combined_hoax_dataset.csv

Usage:
    python src/hoax_detection/prepare_enhanced_data.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# Try to import datasets, make it optional
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("[DataPrep] Warning: 'datasets' library not installed. Run: pip install datasets")


# Constants
OUTPUT_PATH = "data/combined_hoax_dataset.csv"
CLEANED_DATA_DIR = "data/cleaned"
EXISTING_HOAX_CSV = "data/hoax_dataset.csv"

# Label mapping: normalize various label formats to binary 0/1
LABEL_MAP = {
    # From fakenews-mafindo
    "HOAX": 1,
    "NON-HOAX": 0,
    "hoax": 1,
    "non-hoax": 0,
    # Fine-grained labels (all map to HOAX)
    "Fabricated Content": 1,
    "False Connection": 1,
    "False Context": 1,
    "Impostor Content": 1,
    "Manipulated Content": 1,
    "Misleading Content": 1,
    "Satire": 1,  # Treat satire as hoax for simplicity
    "CekFakta": 1,
    # Valid/real news
    "valid": 0,
    "real": 0,
    "factual": 0,
    "true": 0,
    # Common variations
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
}


def download_huggingface_dataset() -> pd.DataFrame:
    """
    Download the fakenews-mafindo dataset from HuggingFace.
    
    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    if not HAS_DATASETS:
        print("[DataPrep] Skipping HuggingFace download (datasets library not available)")
        return pd.DataFrame()
    
    print("[DataPrep] Downloading nlp-brin-id/fakenews-mafindo from HuggingFace...")
    
    try:
        ds = load_dataset("nlp-brin-id/fakenews-mafindo")
        
        # Convert to DataFrame
        all_rows = []
        for split in ds.keys():
            print(f"  Processing split: {split} ({len(ds[split])} samples)")
            for item in ds[split]:
                # The dataset has 'content' and 'label' columns
                text = item.get("content") or item.get("text") or item.get("narasi", "")
                label = item.get("label") or item.get("classification", "")
                
                if text and len(text.strip()) > 50:
                    all_rows.append({
                        "text": text.strip(),
                        "label": label,
                        "source": "huggingface-mafindo"
                    })
        
        df = pd.DataFrame(all_rows)
        print(f"[DataPrep] Downloaded {len(df)} samples from HuggingFace")
        return df
        
    except Exception as e:
        print(f"[DataPrep] Warning: Could not download HuggingFace dataset: {e}")
        return pd.DataFrame()


def load_local_excel_datasets() -> pd.DataFrame:
    """
    Load cleaned Excel datasets from data/cleaned directory.
    
    Returns:
        Combined DataFrame from all Excel files.
    """
    print(f"[DataPrep] Loading local Excel datasets from {CLEANED_DATA_DIR}...")
    
    all_dfs = []
    cleaned_dir = Path(CLEANED_DATA_DIR)
    
    if not cleaned_dir.exists():
        print(f"[DataPrep] Warning: {CLEANED_DATA_DIR} not found")
        return pd.DataFrame()
    
    for excel_file in cleaned_dir.glob("*.xlsx"):
        try:
            print(f"  Loading: {excel_file.name}")
            df = pd.read_excel(excel_file)
            
            # Find text column
            text_col = None
            for col in ["content", "text", "narasi", "artikel", "berita", "isi"]:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                # Use first column that looks like text
                for col in df.columns:
                    if df[col].dtype == object:
                        text_col = col
                        break
            
            # Find label column
            label_col = None
            for col in ["label", "status", "keterangan", "kategori"]:
                if col in df.columns:
                    label_col = col
                    break
            
            if text_col:
                # Determine label based on filename if no label column
                if label_col:
                    labels = df[label_col]
                elif "hoax" in excel_file.name.lower() or "turnback" in excel_file.name.lower():
                    labels = ["hoax"] * len(df)
                else:
                    labels = ["valid"] * len(df)  # Assume legitimate news sources
                
                temp_df = pd.DataFrame({
                    "text": df[text_col].astype(str),
                    "label": labels,
                    "source": f"local-{excel_file.stem}"
                })
                
                # Filter empty texts
                temp_df = temp_df[temp_df["text"].str.len() > 50]
                all_dfs.append(temp_df)
                print(f"    Added {len(temp_df)} samples")
                
        except Exception as e:
            print(f"  Error loading {excel_file.name}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"[DataPrep] Loaded {len(combined)} samples from local Excel files")
        return combined
    
    return pd.DataFrame()


def load_existing_hoax_csv() -> pd.DataFrame:
    """
    Load the existing hoax_dataset.csv if available.
    
    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    print(f"[DataPrep] Loading existing {EXISTING_HOAX_CSV}...")
    
    if not os.path.exists(EXISTING_HOAX_CSV):
        print(f"[DataPrep] {EXISTING_HOAX_CSV} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(EXISTING_HOAX_CSV)
        
        # Find text column
        text_col = None
        for col in ["content", "text", "narasi", "artikel"]:
            if col in df.columns:
                text_col = col
                break
        
        # Find label column  
        label_col = None
        for col in ["label", "status"]:
            if col in df.columns:
                label_col = col
                break
        
        if text_col and label_col:
            result = pd.DataFrame({
                "text": df[text_col].astype(str),
                "label": df[label_col],
                "source": "local-hoax_dataset"
            })
            result = result[result["text"].str.len() > 50]
            print(f"[DataPrep] Loaded {len(result)} samples from existing CSV")
            return result
        else:
            print(f"[DataPrep] Could not find text/label columns in {EXISTING_HOAX_CSV}")
            print(f"  Available columns: {list(df.columns)}")
            
    except Exception as e:
        print(f"[DataPrep] Error loading {EXISTING_HOAX_CSV}: {e}")
    
    return pd.DataFrame()


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all labels to binary 0 (valid) / 1 (hoax).
    
    Args:
        df: DataFrame with 'label' column.
        
    Returns:
        DataFrame with normalized integer labels.
    """
    print("[DataPrep] Normalizing labels to binary 0/1...")
    
    def map_label(label) -> int:
        if pd.isna(label):
            return 0  # Default to valid if unknown
        
        label_str = str(label).strip()
        
        # Try direct mapping
        if label_str in LABEL_MAP:
            return LABEL_MAP[label_str]
        
        # Try lowercase
        label_lower = label_str.lower()
        if label_lower in LABEL_MAP:
            return LABEL_MAP[label_lower]
        
        # Heuristics
        if any(kw in label_lower for kw in ["hoax", "fake", "false", "mislead", "fabri"]):
            return 1
        if any(kw in label_lower for kw in ["valid", "real", "true", "factual"]):
            return 0
        
        # Default
        return 0
    
    df["label"] = df["label"].apply(map_label)
    return df


def deduplicate_and_balance(df: pd.DataFrame, max_imbalance_ratio: float = 3.0) -> pd.DataFrame:
    """
    Remove duplicates and optionally balance the dataset.
    
    Args:
        df: Input DataFrame.
        max_imbalance_ratio: Maximum ratio between majority and minority class.
        
    Returns:
        Cleaned and balanced DataFrame.
    """
    print("[DataPrep] Deduplicating...")
    
    original_len = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"  Removed {original_len - len(df)} duplicates")
    
    # Check balance
    label_counts = df["label"].value_counts()
    print(f"  Label distribution: {dict(label_counts)}")
    
    if len(label_counts) == 2:
        majority_class = label_counts.idxmax()
        minority_class = label_counts.idxmin()
        ratio = label_counts[majority_class] / label_counts[minority_class]
        
        if ratio > max_imbalance_ratio:
            print(f"  Imbalance ratio {ratio:.1f}x exceeds threshold, downsampling majority class...")
            
            # Downsample majority class
            max_majority = int(label_counts[minority_class] * max_imbalance_ratio)
            majority_df = df[df["label"] == majority_class].sample(n=max_majority, random_state=42)
            minority_df = df[df["label"] == minority_class]
            
            df = pd.concat([majority_df, minority_df], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            print(f"  After balancing: {len(df)} samples")
            print(f"  New distribution: {dict(df['label'].value_counts())}")
    
    return df


def prepare_combined_dataset():
    """
    Main function to prepare the combined dataset.
    """
    print("=" * 60)
    print("Enhanced Hoax Detection Dataset Preparation")
    print("=" * 60)
    
    all_dfs = []
    
    # 1. Download HuggingFace dataset
    hf_df = download_huggingface_dataset()
    if len(hf_df) > 0:
        all_dfs.append(hf_df)
    
    # 2. Load local Excel files
    excel_df = load_local_excel_datasets()
    if len(excel_df) > 0:
        all_dfs.append(excel_df)
    
    # 3. Load existing CSV
    csv_df = load_existing_hoax_csv()
    if len(csv_df) > 0:
        all_dfs.append(csv_df)
    
    if not all_dfs:
        print("[DataPrep] ERROR: No data sources found!")
        return
    
    # Combine all
    print("\n[DataPrep] Combining all datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total before processing: {len(combined)}")
    
    # Normalize labels
    combined = normalize_labels(combined)
    
    # Deduplicate and balance
    combined = deduplicate_and_balance(combined)
    
    # Final stats
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(combined)}")
    print(f"Label distribution:")
    for label, count in combined["label"].value_counts().items():
        pct = count / len(combined) * 100
        label_name = "HOAX" if label == 1 else "VALID"
        print(f"  {label_name} ({label}): {count} ({pct:.1f}%)")
    
    print(f"\nSources:")
    for source, count in combined["source"].value_counts().items():
        print(f"  {source}: {count}")
    
    # Save
    print(f"\n[DataPrep] Saving to {OUTPUT_PATH}...")
    
    # Keep only text and label for training
    output_df = combined[["text", "label"]].copy()
    output_df.columns = ["content", "label"]  # Rename for compatibility
    output_df.to_csv(OUTPUT_PATH, index=False)
    
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"[DataPrep] Done! Saved {len(output_df)} samples ({file_size:.1f} MB)")
    print(f"\nTo train the model, run:")
    print(f"  python -m src.hoax_detection.train_lora --data_path {OUTPUT_PATH}")
    
    return output_df


if __name__ == "__main__":
    prepare_combined_dataset()
