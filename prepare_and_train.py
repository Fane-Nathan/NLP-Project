"""
Prepare combined dataset and run LoRA fine-tuning for hoax detection.
"""
import pandas as pd
import os

def prepare_dataset():
    """Combine all cleaned datasets into one training file."""
    print("=" * 60)
    print("STEP 1: Preparing Combined Dataset")
    print("=" * 60)
    
    # Load all cleaned datasets
    print("\nLoading cleaned datasets...")
    hoax_df = pd.read_excel('data/cleaned/dataset_turnbackhoax_cleaned.xlsx')
    kompas_df = pd.read_excel('data/cleaned/dataset_kompas_cleaned.xlsx')
    cnn_df = pd.read_excel('data/cleaned/dataset_cnn_cleaned.xlsx')
    detik_df = pd.read_excel('data/cleaned/dataset_detik_cleaned.xlsx')
    
    print(f"  - TurnBackHoax (hoax=1): {len(hoax_df)} samples")
    print(f"  - Kompas (hoax=0): {len(kompas_df)} samples")
    print(f"  - CNN (hoax=0): {len(cnn_df)} samples")
    print(f"  - Detik (hoax=0): {len(detik_df)} samples")
    
    # Combine all - use Clean Narasi as text, hoax as label
    all_dfs = []
    for df, source in [(hoax_df, 'turnbackhoax'), (kompas_df, 'kompas'), 
                        (cnn_df, 'cnn'), (detik_df, 'detik')]:
        subset = df[['Clean Narasi', 'hoax']].copy()
        subset.columns = ['content', 'label']
        subset['source'] = source
        all_dfs.append(subset)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean: remove NaN and empty strings
    combined_df = combined_df.dropna(subset=['content'])
    combined_df = combined_df[combined_df['content'].str.len() > 20]
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nCombined dataset: {len(combined_df)} samples")
    print(f"  - Real news (0): {len(combined_df[combined_df['label']==0])}")
    print(f"  - Hoax news (1): {len(combined_df[combined_df['label']==1])}")
    
    # Save
    output_path = 'data/hoax_dataset.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    return output_path

def run_training(data_path):
    """Run the LoRA fine-tuning."""
    print("\n" + "=" * 60)
    print("STEP 2: Running LoRA Fine-Tuning")
    print("=" * 60)
    
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "-m", "src.hoax_detection.train_lora",
        "--data_path", data_path,
        "--epochs", "3",
        "--batch_size", "4",
        "--max_length", "256"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    # Step 1: Prepare dataset
    data_path = prepare_dataset()
    
    # Step 2: Run training
    run_training(data_path)
