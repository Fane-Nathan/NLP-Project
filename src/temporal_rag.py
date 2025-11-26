# src/temporal_rag.py
import os
from llama_index.core import VectorStoreIndex, Settings
from src.data_loader_llama import LlamaNewsAdapter
from src.models.gemini_summarizer import create_summarizer

# Use your existing Gemini wrapper for the final synthesis
# (LlamaIndex can do this too, but your custom prompt is already tuned)

class TemporalRAGPipeline:
    def __init__(self):
        print("Initializing Temporal RAG...")
        self.adapter = LlamaNewsAdapter()
        
        # Initialize your custom "Historian" summarizer
        self.historian = create_summarizer()

    def run_temporal_analysis(self, query: str):
        # 1. LOAD DATA
        print("📚 Loading Archives...")
        documents = self.adapter.get_temporal_documents(cluster_size=10)
        
        # 2. INDEXING (The Librarian work)
        # This turns text into searchable vectors
        print("🗂️  Indexing content...")
        index = VectorStoreIndex.from_documents(documents)
        
        # 3. RETRIEVAL
        # Find the top 5 most relevant chunks to the query
        print(f"🔍 Searching for: '{query}'...")
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)
        
        # 4. TEMPORAL SORTING (The "Magic" Step)
        # LlamaIndex returns nodes by relevance (similarity score).
        # We MUST re-sort them by DATE to create a timeline.
        print("⏳ Organizing timeline...")
        sorted_nodes = sorted(
            nodes, 
            key=lambda node: node.metadata.get('date', '0000-00-00')
        )
        
        # 5. PREPARE FOR HISTORIAN
        # Convert back to the format your GeminiSummarizer expects
        context_for_llm = []
        for node in sorted_nodes:
            context_for_llm.append({
                'content': node.text,
                'date': node.metadata['date'],
                'source': node.metadata['source']
            })
            
        # 6. SYNTHESIS
        print("✍️  Writing history...")
        # We assume you updated GeminiSummarizer to handle list-of-dicts (as discussed previously)
        # If not, it will fall back to string concatenation, but the order is now correct!
        result = self.historian.summarize(
            documents=context_for_llm,
            query=query,
            style="timeline" # Triggers your "Historian" prompt
        )
        
        return result.summary

if __name__ == "__main__":
    pipeline = TemporalRAGPipeline()
    history = pipeline.run_temporal_analysis("Jelaskan perkembangan berita ini.")
    
    print("\n" + "="*60)
    print("📜 FINAL HISTORICAL RECORD")
    print("="*60)
    print(history)