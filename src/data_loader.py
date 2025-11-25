import pandas as pd
from datasets import load_dataset
import random
from typing import List, Dict
from datetime import datetime, timedelta

class NewsDataLoader:
    """
    Handles loading of Indonesian News Data (XL-Sum via BBC Indonesia) and 
    simulates Multi-Document clusters for the project pipeline.
    """
    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = cache_dir
        print("Loading XL-Sum Dataset (BBC Indonesia)...")
        # TRUST_REMOTE_CODE=True is required for script-based datasets
        self.dataset = load_dataset(
            "csebuetnlp/xlsum", 
            "indonesian", 
            split="train", 
            cache_dir=cache_dir,
            trust_remote_code=True 
        )
        print(f"✓ Loaded {len(self.dataset)} articles.")

    def get_random_cluster(self, cluster_size: int = 5) -> Dict:
        """
        Simulates a multi-document cluster by picking random articles.
        """
        # Safety check if dataset is smaller than requested cluster
        real_size = min(len(self.dataset), cluster_size)
        indices = random.sample(range(len(self.dataset)), real_size)
        
        articles = [self.dataset[i] for i in indices]
        
        # Simulate 'Timeline' metadata
        base_date = datetime.now()
        cluster_data = []
        
        for i, art in enumerate(articles):
            fake_date = base_date - timedelta(days=random.randint(0, 10))
            cluster_data.append({
                'id': art['id'],
                'source': 'BBC Indonesia',  # Source is now BBC
                'date': fake_date.strftime("%Y-%m-%d"),
                'text': art['text'],        # XL-Sum uses 'text'
                'gold_summary': art['summary'], # XL-Sum uses 'summary'
                'url': art['url']
            })
            
        # Sort by date (Crucial for 'Timeline' feature)
        cluster_data.sort(key=lambda x: x['date'])
        
        return cluster_data