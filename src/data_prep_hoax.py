import pandas as pd
from datasets import load_dataset
import random
import os

def create_hoax_dataset(output_path="data/hoax_dataset.csv"):
    print("Generating Training Data for Trust Layer...")
    
    # 1. Load HOAX Data (TurnBackHoax via Hugging Face)
    # This dataset contains articles debunked as hoaxes
    print("Loading Hoax Source (Rifky/indonesian-hoax-news)...")
    try:
        hoax_ds = load_dataset("Rifky/indonesian-hoax-news", split="train")
        # Extract title/content. We'll combine them for better context.
        # Assuming columns might be 'title' or 'content'. Adjust if needed based on inspection.
        # This dataset usually has 'title' and 'url'. The title often contains the claim.
        hoax_texts = [item['title'] for item in hoax_ds]
        hoax_labels = [1] * len(hoax_texts)
        print(f" -> Found {len(hoax_texts)} Hoax samples.")
    except Exception as e:
        print(f"Error loading Hoax data: {e}")
        return

    # 2. Load TRUSTED Data (BBC XL-Sum)
    # We treat BBC News as the 'Ground Truth' / Non-Hoax
    print("Loading Trusted Source (BBC XL-Sum)...")
    trusted_ds = load_dataset("csebuetnlp/xlsum", "indonesian", split="train", trust_remote_code=True)
    
    # Balance the dataset: Take same amount of trusted news as hoax news
    # to prevent the model from just guessing "Safe" all the time.
    sample_size = min(len(hoax_texts), len(trusted_ds))
    trusted_slice = trusted_ds.select(range(sample_size))
    
    trusted_texts = [item['text'] for item in trusted_slice]
    trusted_labels = [0] * len(trusted_texts)
    print(f" -> Found {len(trusted_texts)} Trusted samples.")

    # 3. Combine & Shuffle
    all_texts = hoax_texts + trusted_texts
    all_labels = hoax_labels + trusted_labels
    
    df = pd.DataFrame({'text': all_texts, 'label': all_labels})
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved balanced dataset to {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    create_hoax_dataset()