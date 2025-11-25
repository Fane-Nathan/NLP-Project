"""
LoRA Fine-Tuning Script for mT5 (Multitask)

Optimized for 4GB VRAM (GTX 1650)
Tasks:
1. Classification (Hoax/Valid)
2. Summarization (Abstractive)
3. Explanation (Optional, if data available)

Usage:
    python -m src.hoax_detection.train_mt5 --data_path data/turnbackhoax.csv --task classify
"""

import os
import argparse
import json
import random
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

# Constants
MODEL_NAME = "google/mt5-small"  # Fallback to small for 4GB VRAM
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

@dataclass
class MT5DatasetConfig:
    text_column: str = "content"
    label_column: str = "label"
    summary_column: str = "summary"

class MT5MultitaskDataset(Dataset):
    """
    Dataset for mT5 multitask training.
    Formats inputs as: "{task}: {text}"
    Formats targets as: "{target_text}"
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_target_length: int = MAX_TARGET_LENGTH
    ):
        """
        Args:
            data: List of dicts with 'task', 'input_text', 'target_text'
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        task_prefix = f"{item['task']}: "
        input_text = task_prefix + str(item['input_text'])
        target_text = str(item['target_text'])
        
        # Tokenize input
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding=False # Dynamic padding
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                truncation=True,
                padding=False
            )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def load_data_for_task(
    file_path: str, 
    task: str,
    config: MT5DatasetConfig = MT5DatasetConfig()
) -> List[Dict]:
    """Load and format data for a specific task."""
    
    formatted_data = []
    
    if task == "classify":
        # Load Hoax Data (reuse logic from train_lora.py roughly)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
            
        # Normalize columns (simplified)
        text_col = config.text_column if config.text_column in df.columns else 'content'
        label_col = config.label_column if config.label_column in df.columns else 'label'
        
        if text_col not in df.columns:
             # Try to find text column
            for c in ['isi_berita', 'text', 'narasi']:
                if c in df.columns:
                    text_col = c
                    break
        
        df = df.dropna(subset=[text_col, label_col])
        
        for _, row in df.iterrows():
            label_raw = row[label_col]
            # Map label to text
            if str(label_raw).lower() in ['1', 'hoax', 'fake', 'true']: # In some datasets 'true' means 'true hoax'
                 target = "HOAX"
            else:
                 target = "VALID"
                 
            formatted_data.append({
                "task": "classify",
                "input_text": row[text_col],
                "target_text": target
            })
            
    elif task == "summarize":
        # Load Summarization Data (e.g. XL-Sum)
        # For now, assuming a CSV/JSON with 'text' and 'summary'
        if file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
             df = pd.read_csv(file_path)
             
        text_col = 'text'
        sum_col = 'summary'
        
        for _, row in df.iterrows():
            formatted_data.append({
                "task": "summarize",
                "input_text": row[text_col],
                "target_text": row[sum_col]
            })
            
    return formatted_data

def train_mt5(
    data_path: str,
    output_dir: str = "models/mt5_lora",
    task: str = "classify",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3
):
    print(f"Training mT5 for task: {task}")
    
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # 2. Prepare Data
    if data_path == "synthetic":
        # Generate synthetic data for testing
        data = []
        for _ in range(100):
            if task == "classify":
                data.append({
                    "task": "classify",
                    "input_text": "Ini adalah berita palsu contoh " + str(random.random()),
                    "target_text": "HOAX"
                })
                data.append({
                    "task": "classify",
                    "input_text": "Ini adalah berita benar contoh " + str(random.random()),
                    "target_text": "VALID"
                })
            elif task == "summarize":
                 data.append({
                    "task": "summarize",
                    "input_text": "Panjang lebar berita ini " * 10,
                    "target_text": "Ringkasan berita."
                })
    else:
        data = load_data_for_task(data_path, task)
        
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    
    train_dataset = MT5MultitaskDataset(train_data, tokenizer)
    val_dataset = MT5MultitaskDataset(val_data, tokenizer)
    
    # 3. LoRA Config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q", "v"], 
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Training Args
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        predict_with_generate=True,
        save_total_limit=2
    )
    
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="synthetic")
    parser.add_argument("--task", type=str, default="classify")
    parser.add_argument("--output_dir", type=str, default="models/mt5_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_mt5(
        data_path=args.data_path, 
        output_dir=args.output_dir, 
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
