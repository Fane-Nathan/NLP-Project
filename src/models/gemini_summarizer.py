"""
Gemini API Summarizer for Indonesian Multi-Document Summarization

Uses the new google-genai SDK (2024+) with Gemini 2.5 Flash Lite.
"""

import os
from typing import List, Optional, Union
from dataclasses import dataclass

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class SummaryResult:
    """Result container for summarization output."""
    summary: str
    model: str
    input_docs: int
    input_tokens: int = 0
    output_tokens: int = 0
    style: str = "default"


class GeminiSummarizer:
    """
    Gemini-based abstractive summarizer for Indonesian news.
    Uses Gemini 2.5 Flash Lite for optimal speed and cost.
    """
    
    DEFAULT_MODEL = "gemini-2.5-flash-lite"
    
    SYSTEM_PROMPT = """Anda adalah asisten ahli untuk merangkum berita Indonesia dari berbagai sumber.

PANDUAN:
1. Akurasi: Jangan mengubah atau menambahkan fakta
2. Sintesis: Gabungkan informasi terkait, hindari pengulangan
3. Kronologi: Susun informasi secara kronologis jika relevan
4. Objektivitas: Gunakan bahasa netral
5. Bahasa: Gunakan Bahasa Indonesia yang baik dan benar

FORMAT: Ringkasan dalam 2-3 paragraf tanpa menyebutkan "Dokumen 1", "Dokumen 2", dll."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None
    ):
        """
        Initialize Gemini summarizer.
        
        Args:
            api_key: Gemini API key. Uses GEMINI_API_KEY env var if not provided.
            model: Model name (default: gemini-2.5-flash-lite).
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        # Set API key in environment if provided explicitly
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        
        # Initialize client (reads GEMINI_API_KEY from environment)
        self.client = genai.Client()
        self.model_name = model or self.DEFAULT_MODEL
        
        print(f"[GeminiSummarizer] Initialized")
        print(f"  Model: {self.model_name}")
    
    def summarize(
        self,
        documents: Union[str, List[str]],
        query: Optional[str] = None,
        style: str = "default"
    ) -> SummaryResult:
        """
        Summarize one or more documents.
        
        Args:
            documents: Single document or list of documents.
            query: Optional focus query.
            style: Summary style (default, brief, detailed).
            
        Returns:
            SummaryResult with summary and metadata.
        """
        # Normalize input
        if isinstance(documents, str):
            documents = [documents]
        
        if not documents:
            return SummaryResult(
                summary="Tidak ada dokumen untuk dirangkum.",
                model=self.model_name,
                input_docs=0
            )
        
        # Filter empty documents
        documents = [d.strip() for d in documents if d and d.strip()]
        
        # Build prompt
        style_instructions = {
            "default": "Buat ringkasan komprehensif dalam 2-3 paragraf.",
            "brief": "Buat ringkasan singkat dalam 1 paragraf (maksimal 100 kata).",
            "detailed": "Buat ringkasan detail (3-5 paragraf)."
        }
        
        docs_formatted = "\n\n---\n\n".join([
            f"[Dokumen {i+1}]\n{doc}" 
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""{self.SYSTEM_PROMPT}

INSTRUKSI: {style_instructions.get(style, style_instructions['default'])}

{f'FOKUS: {query}' if query else ''}

DOKUMEN SUMBER ({len(documents)} dokumen):

{docs_formatted}

RINGKASAN:"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            summary = response.text.strip()
            
            # Get token usage if available
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            return SummaryResult(
                summary=summary,
                model=self.model_name,
                input_docs=len(documents),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                style=style
            )
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[GeminiSummarizer] {error_msg}")
            return SummaryResult(
                summary=error_msg,
                model=self.model_name,
                input_docs=len(documents)
            )
    
    def explain_hoax(
        self, 
        text: str, 
        classification: str, 
        confidence: float
    ) -> str:
        """Generate explanation for hoax/valid classification."""
        prompt = f"""Analisis konten berikut yang diklasifikasikan sebagai {classification} dengan kepercayaan {confidence:.1%}.

KONTEN:
"{text[:1500]}"

TUGAS:
1. Jelaskan mengapa konten ini {classification.lower()}
2. Identifikasi indikator spesifik (red flags untuk hoax, kredibilitas untuk valid)
3. Berikan rekomendasi untuk pembaca

Jawab dalam Bahasa Indonesia, maksimal 3 paragraf."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Tidak dapat menghasilkan penjelasan: {str(e)}"


# Factory function
def create_summarizer(api_key: Optional[str] = None) -> GeminiSummarizer:
    """Create a Gemini summarizer with default settings."""
    return GeminiSummarizer(api_key=api_key)


if __name__ == "__main__":
    summarizer = GeminiSummarizer()
    
    docs = [
        "Pemerintah Indonesia mengumumkan kebijakan vaksinasi COVID-19 baru.",
        "Kementerian Kesehatan melaporkan peningkatan cakupan vaksinasi.",
        "WHO memberikan apresiasi atas keberhasilan program vaksinasi Indonesia."
    ]
    
    result = summarizer.summarize(docs)
    print(f"\nSummary:\n{result.summary}")