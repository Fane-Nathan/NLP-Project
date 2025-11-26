import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings
)
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini

# Load env vars
env_path = os.path.abspath(".env")
print(f"[LlamaIndex] Loading .env from: {env_path}")
load_dotenv(env_path)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class LlamaIndexManager:
    def __init__(self, storage_dir="data/llama_storage", model_name="models/gemini-2.5-flash-lite"):
        self.storage_dir = storage_dir
        
        # Configure Settings
        # Use Gemini for LLM
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[Warning] GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment variables.")
            print("Available keys:", [k for k in os.environ.keys() if "API" in k])
        else:
            print(f"[LlamaIndex] API Key found: {api_key[:5]}...")
        
        Settings.llm = Gemini(model="models/gemini-2.0-flash-lite", api_key=api_key)
        
        # Use local HF embeddings (free and fast)
        # BAAI/bge-m3 is great for multilingual (including Indonesian)
        print("[LlamaIndex] Loading embedding model (BAAI/bge-m3)...")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        
        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        if not os.path.exists(self.storage_dir):
            print(f"[LlamaIndex] Creating new storage at {self.storage_dir}")
            os.makedirs(self.storage_dir)
            # Create empty index
            return VectorStoreIndex([])
        
        try:
            print(f"[LlamaIndex] Loading index from {self.storage_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            return load_index_from_storage(storage_context)
        except Exception as e:
            print(f"[LlamaIndex] Error loading index: {e}. Creating new one.")
            return VectorStoreIndex([])

    def add_document(self, text, metadata=None):
        """Adds a text document to the index."""
        if not text:
            return
            
        doc = Document(text=text, metadata=metadata or {})
        self.index.insert(doc)
        self.persist()
        print("[LlamaIndex] Document added and persisted.")

    def query(self, query_text, similarity_top_k=3):
        """Retrieves relevant context."""
        if not query_text:
            return ""
            
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        nodes = retriever.retrieve(query_text)
        
        context = "\n".join([n.node.get_content() for n in nodes])
        return context

    def persist(self):
        """Saves the index to disk."""
        self.index.storage_context.persist(persist_dir=self.storage_dir)
