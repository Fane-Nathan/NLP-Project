import pandas as pd
from datasets import load_dataset
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class NewsDataLoader:
    """
    Handles loading of Indonesian News Data (XL-Sum via BBC Indonesia) and 
    creates Multi-Document clusters for the project pipeline.
    
    Supports:
    - Random clustering (legacy, for quick testing)
    - Semantic clustering (TF-IDF similarity-based, recommended for MDS)
    """
    
    # Indonesian stopwords for TF-IDF
    STOPWORDS = [
        'yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'untuk', 'pada',
        'adalah', 'sebagai', 'dengan', 'juga', 'akan', 'sudah', 'atau',
        'ia', 'dia', 'mereka', 'kita', 'kami', 'anda', 'saya', 'tidak',
        'bisa', 'ada', 'oleh', 'sebuah', 'dalam', 'tersebut', 'dapat'
    ]
    
    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = cache_dir
        print("Loading XL-Sum Dataset (BBC Indonesia)...")
        # Pinned revision for supply chain security
        self.dataset = load_dataset(
            "csebuetnlp/xlsum", 
            "indonesian", 
            split="train", 
            cache_dir=cache_dir,
            trust_remote_code=True,
            revision="b34c730"  # Pinned to Sep 2023 stable version
        )
        print(f"✓ Loaded {len(self.dataset)} articles.")
        
        # Precompute TF-IDF vectors for semantic clustering
        self._tfidf_matrix = None
        self._vectorizer = None

    def get_random_cluster(self, cluster_size: int = 5) -> List[Dict]:
        """
        Simulates a multi-document cluster by picking random articles.
        
        Note: This is NOT recommended for evaluation as random articles
        are unlikely to be semantically related. Use get_semantic_cluster() instead.
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
                'source': 'BBC Indonesia',
                'date': fake_date.strftime("%Y-%m-%d"),
                'text': art['text'],
                'gold_summary': art['summary'],
                'url': art['url']
            })
            
        # Sort by date
        cluster_data.sort(key=lambda x: x['date'])
        
        return cluster_data
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for semantic clustering (lazy initialization)."""
        if self._tfidf_matrix is not None:
            return
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for semantic clustering. Install with: pip install scikit-learn")
        
        print("[DataLoader] Building TF-IDF index for semantic clustering...")
        
        # Extract all texts
        all_texts = [item['text'] for item in self.dataset]
        
        # Create TF-IDF vectorizer
        self._vectorizer = TfidfVectorizer(
            stop_words=self.STOPWORDS,
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        self._tfidf_matrix = self._vectorizer.fit_transform(all_texts)
        print(f"[DataLoader] TF-IDF index built: {self._tfidf_matrix.shape}")
    
    def get_semantic_cluster(
        self, 
        cluster_size: int = 5,
        seed_index: Optional[int] = None,
        similarity_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Creates a semantically coherent document cluster.
        
        Finds documents that are topically related using TF-IDF cosine similarity.
        This is the recommended method for Multi-Document Summarization evaluation.
        
        Args:
            cluster_size: Number of documents in the cluster.
            seed_index: Index of seed document (random if None).
            similarity_threshold: Minimum similarity to include in cluster.
            
        Returns:
            List of related documents.
        """
        # Build index if not already done
        self._build_tfidf_index()
        
        # Pick random seed if not specified
        if seed_index is None:
            seed_index = random.randint(0, len(self.dataset) - 1)
        
        # Get seed document vector
        seed_vector = self._tfidf_matrix[seed_index]
        
        # Calculate similarity to all other documents
        similarities = cosine_similarity(seed_vector, self._tfidf_matrix)[0]
        
        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Select top-k similar documents (excluding those below threshold)
        selected_indices = []
        for idx in sorted_indices:
            if len(selected_indices) >= cluster_size:
                break
            if similarities[idx] >= similarity_threshold:
                selected_indices.append(idx)
        
        # Build cluster data
        base_date = datetime.now()
        cluster_data = []
        
        for i, idx in enumerate(selected_indices):
            art = self.dataset[idx]
            fake_date = base_date - timedelta(days=i)
            cluster_data.append({
                'id': art['id'],
                'source': 'BBC Indonesia',
                'date': fake_date.strftime("%Y-%m-%d"),
                'text': art['text'],
                'gold_summary': art['summary'],
                'url': art['url'],
                'similarity_to_seed': float(similarities[idx])
            })
        
        # Sort by date
        cluster_data.sort(key=lambda x: x['date'])
        
        print(f"[DataLoader] Semantic cluster created: {len(cluster_data)} docs, "
              f"similarity range: {cluster_data[0].get('similarity_to_seed', 0):.3f} - "
              f"{cluster_data[-1].get('similarity_to_seed', 0):.3f}")
        
        return cluster_data
    
    def get_cluster(
        self, 
        cluster_size: int = 5, 
        method: str = "semantic"
    ) -> List[Dict]:
        """
        Get document cluster using specified method.
        
        Args:
            cluster_size: Number of documents.
            method: "semantic" (recommended) or "random".
            
        Returns:
            List of documents.
        """
        if method == "semantic":
            return self.get_semantic_cluster(cluster_size)
        else:
            return self.get_random_cluster(cluster_size)


if __name__ == "__main__":
    # Demo
    loader = NewsDataLoader()
    
    print("\n=== Random Cluster ===")
    random_cluster = loader.get_random_cluster(3)
    for doc in random_cluster:
        print(f"  - {doc['text'][:80]}...")
    
    print("\n=== Semantic Cluster ===")
    semantic_cluster = loader.get_semantic_cluster(3)
    for doc in semantic_cluster:
        print(f"  [{doc['similarity_to_seed']:.3f}] {doc['text'][:80]}...")
