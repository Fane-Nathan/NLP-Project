# src/data_loader_llama.py
from llama_index.core import Document
from src.data_loader import NewsDataLoader

class LlamaNewsAdapter:
    """
    Adapts the existing NewsDataLoader to return LlamaIndex Documents.
    """
    def __init__(self):
        self.loader = NewsDataLoader()

    def get_temporal_documents(self, cluster_size=5):
        # 1. Get the raw cluster (dicts) from your existing loader
        raw_cluster = self.loader.get_random_cluster(cluster_size)
        
        llama_documents = []
        for art in raw_cluster:
            # 2. Convert to LlamaIndex Document
            # We strictly embed the 'date' into the metadata
            doc = Document(
                text=art['text'],
                metadata={
                    "date": art['date'],
                    "source": art['source'],
                    "url": art['url']
                },
                excluded_llm_metadata_keys=["url"] # Don't distract LLM with URLs
            )
            llama_documents.append(doc)
            
        return llama_documents

if __name__ == "__main__":
    adapter = LlamaNewsAdapter()
    docs = adapter.get_temporal_documents()
    print(f"Converted {len(docs)} LlamaIndex documents.")
    print(f"Sample Metadata: {docs[0].metadata}")