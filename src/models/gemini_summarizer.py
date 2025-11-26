"""
Gemini API Summarizer for Indonesian Multi-Document Summarization

Uses the new google-genai SDK (2024+) with Gemini 2.5 Flash Lite.
"""

import os
from typing import List, Optional, Union
from dataclasses import dataclass

try:
    from google import genai
    from google.genai import types
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
    
    SYSTEM_PROMPT = """You are a professional executive assistant.
    
    PERSONA:
    - Tone: Natural, fluid, and engaging.
    - Language: ALWAYS speak in ENGLISH.
    - Style: Narrative storytelling. Use paragraphs, not bullet points.
    - Constraint: No exaggeration (e.g., avoid "Yikes", "Deets", "Buzz"). Be grounded but conversational.

    GUIDELINES:
    1. Accuracy: Stick to the facts.
    2. Flow: Connect ideas logically like a human telling a story.
    3. Format: Use clean paragraphs.
    """

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
    
    def _build_prompt(self, documents: Union[List[str], List[dict]], query: Optional[str] = None, style: str = "default") -> str:
        """Build the prompt for Gemini."""
        style_instructions = {
            "default": "Tell me the story of this document in 2-3 fluid paragraphs.",
            "brief": "Give me a quick 1-paragraph overview of the situation.",
            "detailed": "Provide a detailed narrative analysis. Cover the key events and context.",
            "chat": "Explain this to me naturally, like a colleague briefing me. No bullet points.",
            "timeline": "Construct a chronological narrative based on the dates provided. Cite sources and dates explicitly."
        }
        
        # Handle the "Live Historian" format (list of dicts)
        if documents and isinstance(documents[0], dict):
            docs_formatted = []
            for i, doc in enumerate(documents):
                # Support 'body', 'text', or 'content' keys
                content = doc.get('content') or doc.get('text') or doc.get('body') or ""
                date = doc.get('date', 'Unknown Date')
                source = doc.get('source', 'Web Search')
                
                docs_formatted.append(
                    f"--- [Sumber {i+1} | Tanggal: {date} | Asal: {source}] ---\n{content}"
                )
            docs_text = "\n\n".join(docs_formatted)
        else:
            # Fallback for simple string lists
            docs_text = "\n\n---\n\n".join([
                f"[Document {i+1}]\n{doc}" 
                for i, doc in enumerate(documents)
            ])
        
        prompt = f"""{self.SYSTEM_PROMPT}

INSTRUCTIONS: {style_instructions.get(style, style_instructions['default'])}

{f'FOCUS: {query}' if query else ''}

SOURCE DOCUMENTS ({len(documents)} docs):

{docs_text}

SUMMARY:"""
        return prompt

    def summarize(
        self,
        documents: Union[str, List[str], List[dict]],
        query: Optional[str] = None,
        style: str = "default"
    ) -> SummaryResult:
        """
        Summarize one or more documents.
        
        Args:
            documents: Single document, list of strings, or list of dicts (with metadata).
            query: Optional focus query.
            style: Summary style (default, brief, detailed, timeline).
            
        Returns:
            SummaryResult with summary and metadata.
        """
        # Normalize input
        if isinstance(documents, str):
            documents = [documents]
        
        if not documents:
            return SummaryResult(
                summary="No documents to summarize.",
                model=self.model_name,
                input_docs=0
            )
        
        # Filter empty documents (if strings)
        if documents and isinstance(documents[0], str):
            documents = [d.strip() for d in documents if d and d.strip()]
        
        # Build prompt
        prompt = self._build_prompt(documents, query, style)

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
        prompt = f"""Analyze the following content classified as {classification} with {confidence:.1%} confidence.

CONTENT:
"{text[:1500]}"

TASK:
1. Explain why this is {classification.lower()}
2. Identify specific indicators (red flags or credibility markers)
3. Provide a recommendation for the reader

Answer in ENGLISH, max 3 paragraphs."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Cannot generate explanation: {str(e)}"

    def summarize_image(
        self, 
        image_base64: str, 
        prompt: Optional[str] = None,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Analyze and summarize image content using Gemini Vision.
        
        Args:
            image_base64: Base64-encoded image (data URI or raw).
            prompt: Custom analysis prompt.
            max_tokens: Maximum tokens (unused in Gemini 2.5, but kept for compat).
            
        Returns:
            Image analysis text or None if failed.
        """
        import base64
        
        # Clean base64 string if it has prefix
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
            
        try:
            image_bytes = base64.b64decode(image_base64)
            
            user_prompt = prompt or "Analyze and describe this image in detail. Focus on text and key visual elements."
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    user_prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                ]
            )
            return response.text.strip()
            
        except Exception as e:
            print(f"[GeminiSummarizer] Vision Error: {e}")
            return None

    def describe_image(
        self, 
        image_base64: str, 
        max_tokens: int = 500
    ) -> Optional[str]:
        """Get detailed description of image."""
        return self.summarize_image(
            image_base64,
            prompt="Describe this image in detail. Identify main objects, text, setting, and context.",
            max_tokens=max_tokens
        )


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