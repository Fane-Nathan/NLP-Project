"""
Groq LLM Summarizer for Indonesian Text

Uses Groq's fast inference API with Llama/Mixtral models
for abstractive summarization and vision analysis.

Requires GROQ_API_KEY environment variable.
"""

import os
from typing import List, Optional, Union
from dataclasses import dataclass

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    model: str
    tokens_used: int


class GroqSummarizer:
    """
    Abstractive summarizer using Groq API.
    
    Supports both text summarization and image analysis
    for multi-modal document processing.
    """
    
    # Default models
    TEXT_MODEL = "llama-3.1-8b-instant"
    VISION_MODEL = "llama-3.2-11b-vision-preview"  # Updated to 11b preview
    
    # System prompts
    SUMMARIZE_PROMPT = """You are JARVIS, a highly advanced AI assistant.

GUIDELINES:
1. Language: ALWAYS speak in English.
2. Style: Concise, professional, and helpful. Like J.A.R.V.I.S from Iron Man.
3. Context: You are analyzing screen content or text.
4. Format: Use markdown for structure, but keep it readable for TTS.

Task: Summarize the following text."""

    DESCRIBE_PROMPT = """You are JARVIS, a vision-capable AI assistant.

GUIDELINES:
1. Language: ALWAYS speak in English.
2. Style: Concise, professional, and observant.
3. Task: Describe the visual content in detail.
4. Focus: Main elements, text (if any), context, and key information.

Task: Analyze this image."""

    def __init__(
        self, 
        api_key: Optional[str] = None,
        text_model: Optional[str] = None,
        vision_model: Optional[str] = None
    ):
        """
        Initialize Groq summarizer.
        
        Args:
            api_key: Groq API key (or use GROQ_API_KEY env var).
            text_model: Model for text summarization.
            vision_model: Model for image analysis.
        """
        if not GROQ_AVAILABLE:
            raise ImportError("groq package not installed. Run: pip install groq")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("[GroqSummarizer] Warning: No API key found. Set GROQ_API_KEY.")
        
        self.text_model = text_model or self.TEXT_MODEL
        self.vision_model = vision_model or self.VISION_MODEL
        
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        
        print(f"[GroqSummarizer] Initialized")
        print(f"  Text model: {self.text_model}")
        print(f"  Vision model: {self.vision_model}")
    
    def summarize(
        self, 
        documents: Union[str, List[str]], 
        max_tokens: int = 500,
        custom_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Summarize one or more documents.
        
        Args:
            documents: Single text or list of texts.
            max_tokens: Maximum tokens in response.
            custom_prompt: Override system prompt.
            
        Returns:
            Summary text or None if failed.
        """
        if not self.client:
            print("[GroqSummarizer] Error: No API client available.")
            return None
        
        # Handle list input
        if isinstance(documents, list):
            text = "\n\n---\n\n".join(documents)
        else:
            text = documents
        
        if not text or len(text.strip()) < 10:
            return None
        
        system_prompt = custom_prompt or self.SUMMARIZE_PROMPT
        
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this:\n\n{text}"}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[GroqSummarizer] API Error: {e}")
            return None
    
    def summarize_text(self, text: str, max_tokens: int = 500) -> Optional[str]:
        """Convenience method for single text summarization."""
        return self.summarize(text, max_tokens)
    
    def summarize_image(
        self, 
        image_base64: str, 
        prompt: Optional[str] = None,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Analyze and summarize image content.
        
        Args:
            image_base64: Base64-encoded image (with data URI prefix).
            prompt: Custom analysis prompt.
            max_tokens: Maximum tokens in response.
            
        Returns:
            Image analysis text or None if failed.
        """
        if not self.client:
            print("[GroqSummarizer] Error: No API client available.")
            return None
        
        user_prompt = prompt or "Analyze and describe this image in detail."
        
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_base64}
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[GroqSummarizer] Vision API Error: {e}")
            return None
    
    def describe_image(
        self, 
        image_base64: str, 
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Get detailed description of image.
        
        Args:
            image_base64: Base64-encoded image.
            max_tokens: Maximum tokens.
            
        Returns:
            Description text.
        """
        return self.summarize_image(
            image_base64,
            prompt=self.DESCRIBE_PROMPT,
            max_tokens=max_tokens
        )

    def summarize_with_search(
        self,
        query: str,
        search_context: str,
        max_tokens: int = 300
    ) -> Optional[str]:
        """
        Generate an answer based on search results.
        
        Args:
            query: User's original query.
            search_context: Formatted search results.
            max_tokens: Max tokens for response.
        """
        if not self.client:
            return None

        system_prompt = """You are JARVIS. Answer the user's question based ONLY on the provided web search results.
GUIDELINES:
1. If the results contain the answer, summarize it clearly.
2. If the results are irrelevant, say "I couldn't find that information in the search results."
3. Be concise and professional.
4. Do not hallucinate facts not in the context."""

        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Query: {query}\n\nSearch Results:\n{search_context}"}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[GroqSummarizer] Search-Augmented Generation Error: {e}")
            return None


if __name__ == "__main__":
    # Demo
    summarizer = GroqSummarizer()
    
    if summarizer.client:
        text = """
        The Indonesian government announced a new policy regarding COVID-19 vaccination.
        The program targets 70% of the population within six months.
        The Ministry of Health has prepared over 100 million vaccine doses.
        """
        
        summary = summarizer.summarize(text)
        if summary:
            print("Summary:")
            print(summary)
    else:
        print("Set GROQ_API_KEY to test summarization.")