"""
LoRA Fine-Tuning Script for IndoBERT Hoax Detection

Optimized for 4GB VRAM (GTX 1650 / RTX 3050)
Dataset: TurnBackHoax (Mafindo) - ~2,000 samples

Usage:
    python -m src.hoax_detection.train_lora --data_path data/turnbackhoax.csv
    
Features:
    - Graceful Ctrl+C save (saves checkpoint before exit)
    - LoRA with configurable rank
    - Mixed precision (FP16) training
    - Automatic checkpointing
"""

import os
import argparse
import json
import signal
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

import pandas as pd

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    BitsAndBytesConfig  # QLoRA 4-bit quantization
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training  # For QLoRA
)

# Constants
MODEL_NAME = "indobenchmark/indobert-base-p1"
MODEL_REVISION = "c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"  # Pinned for supply chain security
MAX_LENGTH = 256  # Optimized for GTX 1650 (good balance of speed/quality)
LORA_R = 16       # Higher rank = more capacity for generalization
LORA_ALPHA = 32   # Scaling factor (typically 2x rank)
LORA_DROPOUT = 0.05  # Lower dropout during training


@dataclass
class HoaxDatasetConfig:
    """Configuration for hoax dataset loading."""
    text_column: str = "content"
    label_column: str = "label"
    label_map: Dict[str, int] = field(default_factory=lambda: {"valid": 0, "hoax": 1})


class HoaxDataset(Dataset):
    """
    PyTorch Dataset for TurnBackHoax data.
    
    Attributes:
        texts: List of document texts.
        labels: List of integer labels (0=valid, 1=hoax).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer, 
        max_length: int = MAX_LENGTH
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Dynamic padding via DataCollator
            return_tensors=None
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label
        }


def load_turnbackhoax_data(
    file_path: str,
    config: HoaxDatasetConfig = HoaxDatasetConfig()
) -> Tuple[List[str], List[int]]:
    """
    Load and preprocess TurnBackHoax dataset.
    
    Args:
        file_path: Path to CSV/JSON file.
        config: Dataset configuration.
        
    Returns:
        Tuple of (texts, labels).
    """
    # Detect file format
    if os.path.isdir(file_path):
        print(f"[Data] Loading all files from directory: {file_path}")
        dfs = []
        for filename in os.listdir(file_path):
            full_path = os.path.join(file_path, filename)
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                print(f"  - Loading {filename}...")
                dfs.append(pd.read_excel(full_path))
            elif filename.endswith('.csv'):
                print(f"  - Loading {filename}...")
                dfs.append(pd.read_csv(full_path))
            elif filename.endswith('.parquet'):
                print(f"  - Loading {filename}...")
                dfs.append(pd.read_parquet(full_path))
        
        if not dfs:
            raise ValueError(f"No supported files found in {file_path}")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"[Data] Combined {len(dfs)} files. Total rows: {len(df)}")
        
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Validate columns
    if config.text_column not in df.columns:
        # Try common alternatives
        alternatives = ['isi_berita', 'text', 'content', 'article', 'body', 'narasi', 'Clean Narasi']
        for alt in alternatives:
            if alt in df.columns:
                config.text_column = alt
                break
        else:
            raise KeyError(f"Text column not found. Available: {df.columns.tolist()}")
    
    if config.label_column not in df.columns:
        alternatives = ['label', 'class', 'kategori', 'hoax']
        for alt in alternatives:
            if alt in df.columns:
                config.label_column = alt
                break
        else:
            raise KeyError(f"Label column not found. Available: {df.columns.tolist()}")
    
    # Clean data
    df = df.dropna(subset=[config.text_column, config.label_column])
    df = df[df[config.text_column].str.len() > 10]  # Remove empty/short entries
    
    texts = df[config.text_column].tolist()
    
    # Convert labels
    raw_labels = df[config.label_column].tolist()
    
    # Auto-detect label format
    if isinstance(raw_labels[0], (int, float)):
        labels = [int(l) for l in raw_labels]
    else:
        # String labels - map to integers
        unique_labels = set(raw_labels)
        print(f"[Data] Detected labels: {unique_labels}")
        
        # Common label mappings for Indonesian hoax datasets
        hoax_indicators = {'hoax', 'fake', 'palsu', 'bohong', '1', 'true'}
        valid_indicators = {'valid', 'real', 'fakta', 'benar', '0', 'false'}
        
        labels = []
        for l in raw_labels:
            l_lower = str(l).lower().strip()
            if l_lower in hoax_indicators or l_lower == '1':
                labels.append(1)
            elif l_lower in valid_indicators or l_lower == '0':
                labels.append(0)
            else:
                # Default mapping based on position
                labels.append(1 if l_lower in list(unique_labels)[:len(unique_labels)//2] else 0)
    
    print(f"[Data] Loaded {len(texts)} samples")
    print(f"[Data] Label distribution: Valid={labels.count(0)}, Hoax={labels.count(1)}")
    
    return texts, labels


def create_synthetic_data(num_samples: int = 500) -> Tuple[List[str], List[int]]:
    """
    Create synthetic training data for demonstration/testing.
    
    This generates Indonesian-style fake news patterns for initial testing
    when the real dataset is not available.
    """
    import random
    
    # Hoax patterns (Indonesian fake news characteristics)
    hoax_templates = [
        "VIRAL! {topic} ternyata {claim}. Bagikan sebelum dihapus!",
        "BREAKING: Pemerintah {action} mulai besok. Warga harus {reaction}!",
        "Rahasia {entity} terungkap! {claim} yang disembunyikan selama ini.",
        "AWAS! {topic} berbahaya bagi kesehatan. Dokter {location} mengkonfirmasi.",
        "Terbukti! {claim}. Video ini membuktikan segalanya.",
        "{entity} akhirnya mengakui {claim}. Media mainstream tidak memberitakan!",
        "Cek fakta: {claim} adalah BENAR menurut sumber terpercaya (tidak disebutkan).",
    ]
    
    valid_templates = [
        "Menteri {ministry} mengumumkan kebijakan baru terkait {topic}.",
        "Hasil penelitian {institution} menunjukkan {finding}.",
        "Pemerintah {location} meluncurkan program {program} untuk masyarakat.",
        "Berdasarkan data BPS, {statistic} pada kuartal ini.",
        "Konferensi pers {entity} membahas perkembangan {topic}.",
        "Laporan tahunan {institution} mencatat {finding}.",
    ]
    
    # Fill-in components
    topics = ["vaksin COVID-19", "ekonomi digital", "pendidikan", "kesehatan", "teknologi AI"]
    claims = ["palsu", "berbahaya", "menguntungkan elit", "disembunyikan pemerintah"]
    entities = ["WHO", "pemerintah", "perusahaan farmasi", "media besar"]
    locations = ["Jakarta", "Surabaya", "Bandung", "Indonesia"]
    institutions = ["Universitas Indonesia", "LIPI", "Kemenkes", "Bank Indonesia"]
    
    texts = []
    labels = []
    
    for _ in range(num_samples // 2):
        # Generate hoax
        template = random.choice(hoax_templates)
        text = template.format(
            topic=random.choice(topics),
            claim=random.choice(claims),
            entity=random.choice(entities),
            action=random.choice(["melarang", "mewajibkan", "menghapus"]),
            reaction=random.choice(["waspada", "bersiap", "protes"]),
            location=random.choice(locations)
        )
        texts.append(text)
        labels.append(1)  # Hoax
        
        # Generate valid
        template = random.choice(valid_templates)
        text = template.format(
            ministry=random.choice(["Kesehatan", "Keuangan", "Pendidikan"]),
            topic=random.choice(topics),
            institution=random.choice(institutions),
            finding=random.choice(["peningkatan 5%", "penurunan angka", "stabilitas"]),
            location=random.choice(locations),
            program=random.choice(["bantuan sosial", "digitalisasi", "pelatihan"]),
            statistic=random.choice(["inflasi 3.2%", "pertumbuhan ekonomi 5.1%"]),
            entity=random.choice(entities)
        )
        texts.append(text)
        labels.append(0)  # Valid
    
    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return list(texts), list(labels)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics for Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted'),
        "f1_hoax": f1_score(labels, predictions, pos_label=1)
    }


def train_lora_model(
    data_path: Optional[str] = None,
    output_dir: str = "models/hoax_indobert_lora",
    num_epochs: int = 5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    use_synthetic: bool = False
) -> str:
    """
    Fine-tune IndoBERT with LoRA for hoax detection.
    
    Args:
        data_path: Path to training data (CSV/JSON).
        output_dir: Directory to save the model.
        num_epochs: Number of training epochs.
        batch_size: Micro-batch size (effective = batch_size * gradient_accumulation).
        gradient_accumulation_steps: Steps before weight update.
        learning_rate: LoRA learning rate.
        use_synthetic: Use synthetic data for testing.
        
    Returns:
        Path to saved model.
    """
    print("=" * 60)
    print("IndoBERT + LoRA Hoax Detection Training")
    print("=" * 60)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("[WARNING] No GPU detected. Training will be slow.")
    
    # Load data
    print("\n[1/5] Loading data...")
    if use_synthetic or data_path is None:
        print("[Data] Using synthetic data for demonstration")
        texts, labels = create_synthetic_data(500)
    else:
        texts, labels = load_turnbackhoax_data(data_path)
    
    # Train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"[Data] Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Load tokenizer
    print("\n[2/5] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    
    # Load base model (FP32 base, trainer handles mixed precision)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "VALID", 1: "HOAX"},
        label2id={"VALID": 0, "HOAX": 1},
        use_safetensors=True,
        revision=MODEL_REVISION
    )
    
    # Note: gradient checkpointing disabled for FP16 compatibility
    
    # Configure LoRA
    print("\n[3/5] Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["classifier"],
        bias="lora_only",
        inference_mode=False
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # Create datasets
    train_dataset = HoaxDataset(train_texts, train_labels, tokenizer)
    val_dataset = HoaxDataset(val_texts, val_labels, tokenizer)
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments (4GB GPU optimized)
    print("\n[4/5] Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,  # Larger eval batch
        gradient_accumulation_steps=4,  # Effective batch = 16
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,  # Faster warmup
        lr_scheduler_type="cosine",  # Better scheduler
        
        # Speed optimizations for regular LoRA
        fp16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=False,  # Disabled for FP16 compatibility
        
        # Faster data loading
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        
        # Less frequent evaluation (much faster!)
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=100,
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        report_to="none",
        push_to_hub=False,
        seed=42
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Setup graceful Ctrl+C handler
    def save_on_interrupt(signum, frame):
        print("\n\n[!] Interrupt received! Saving model before exit...")
        try:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[✓] Model saved to {output_dir}")
        except Exception as e:
            print(f"[!] Error saving: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_on_interrupt)
    
    # Check for existing checkpoints to resume from
    resume_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Get the latest checkpoint by step number
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_checkpoint = os.path.join(output_dir, latest)
            print(f"\n[Resume] Found checkpoint: {latest}")
    
    # Train
    print("\n[5/5] Training... (Press Ctrl+C to save and exit)")
    print("-" * 40)
    
    if resume_checkpoint:
        print(f"[Resume] Resuming from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    
    # Save model
    print("\n[Save] Saving LoRA adapters...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config = {
        "base_model": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_length": MAX_LENGTH,
        "labels": {0: "VALID", 1: "HOAX"}
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Final evaluation
    print("\n" + "=" * 40)
    print("Final Evaluation")
    print("=" * 40)
    results = trainer.evaluate()
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n✓ Model saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IndoBERT + LoRA for hoax detection")
    parser.add_argument("--data_path", type=str, help="Path to training data (CSV/JSON)")
    parser.add_argument("--output_dir", type=str, default="models/hoax_indobert_lora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    
    args = parser.parse_args()
    
    train_lora_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_synthetic=args.synthetic
    )
