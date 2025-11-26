import pandas as pd
import re
import os
from sklearn.utils import shuffle

def clean_hoax_text(text):
    if not isinstance(text, str):
        return ""
    # Remove tags like [PENIPUAN], [SALAH], [HOAX]
    cleaned = re.sub(r'^\[.*?\]\s*', '', text)
    return cleaned.strip()

def create_hoax_dataset(output_path="data/hoax_dataset.csv"):
    print("🔄 Generating High-Quality Training Data...")
    
    # --- 1. Load HOAX Data ---
    hoax_path = "data/raw/turnbackhoax_2020_2025.csv"
    if not os.path.exists(hoax_path):
        print(f"❌ Error: Could not find {hoax_path}")
        return

    print(f"   Loading Hoax data from: {hoax_path}")
    df_hoax = pd.read_csv(hoax_path)
    
    # *** FIX IS HERE: Explicitly use 'judul' ***
    col_hoax = 'judul' 
    
    if col_hoax not in df_hoax.columns:
        print(f"❌ Error: Column '{col_hoax}' not found. Available: {df_hoax.columns.tolist()}")
        return

    # Clean the titles
    hoax_texts = df_hoax[col_hoax].apply(clean_hoax_text).tolist()
    hoax_labels = [1] * len(hoax_texts)
    print(f"   -> Found {len(hoax_texts)} Hoax samples (using column: '{col_hoax}')")

    # --- 2. Load VALID Data ---
    valid_files = [
        "data/raw/cnnindonesia_news_RAW.csv",
        "data/raw/detikcom_news_RAW.csv",
        "data/raw/kompas_news_RAW.csv"
    ]
    
    valid_texts = []
    print("   Loading Valid data...")
    
    for file_path in valid_files:
        if os.path.exists(file_path):
            try:
                df_temp = pd.read_csv(file_path)
                # Smarter detection for valid news columns
                possible_cols = ['title', 'judul', 'headline']
                col_valid = next((c for c in possible_cols if c in df_temp.columns), None)
                
                if col_valid:
                    batch_texts = df_temp[col_valid].dropna().unique().tolist()
                    valid_texts.extend(batch_texts)
                    print(f"     + Loaded {len(batch_texts)} from {os.path.basename(file_path)} (col: {col_valid})")
                else:
                    print(f"     ! Skipped {os.path.basename(file_path)}: No title column found.")
            except Exception as e:
                print(f"     ! Error reading {file_path}: {e}")

    # --- 3. Balance & Save ---
    import random
    random.shuffle(valid_texts)
    
    # Balance 1:1
    limit = len(hoax_texts)
    balanced_valid_texts = valid_texts[:limit]
    valid_labels = [0] * len(balanced_valid_texts)
    
    print(f"   -> Selected {len(balanced_valid_texts)} Valid samples for balancing.")

    all_texts = hoax_texts + balanced_valid_texts
    all_labels = hoax_labels + valid_labels
    
    final_df = pd.DataFrame({'content': all_texts, 'label': all_labels})
    final_df = shuffle(final_df, random_state=42).reset_index(drop=True)
    
    # Validation Print
    print("\n🔍 Data Check (First 3 Hoax Samples):")
    print(final_df[final_df['label'] == 1].head(3)['content'].values)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(final_df)} rows to {output_path}")

if __name__ == "__main__":
    create_hoax_dataset()