"""LLM Summarizer for Indonesian Multi-Document Summarization

Supports multiple LLM providers:
- Groq (primary, fast, generous free tier)
- Gemini (fallback)
"""

import os
from typing import List, Optional, Union
from dataclasses import dataclass

# Groq SDK (primary)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Gemini SDK (fallback)
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
    Multi-LLM abstractive summarizer for Indonesian news.
    Supports Groq (primary) and Gemini (fallback) with automatic switching.
    """
    
    # Model defaults
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast and capable
    GEMINI_MODEL = "gemini-2.5-flash-lite"
    
    SYSTEM_PROMPT = """You're a sharp, well-informed friend who's really good at spotting BS in the news.

    YOUR VIBE:
    - Talk naturally, like you're chatting with a friend - no corporate speak, no rigid structures
    - Be direct and honest. If something looks sketchy, say so. If it looks legit, say that too.
    - Keep it brief but insightful - quality over quantity
    - Always speak in English
    
    CRITICAL - VARY YOUR OPENINGS:
    - NEVER start with "So" or "So the article" - this is banned
    - Mix it up: sometimes start with the topic, sometimes with your take, sometimes with a key fact
    - Examples of good varied openings: "This piece covers...", "Looks like...", "Here's the deal with...", "The article dives into...", "Pretty interesting one..."
    
    WHAT TO AVOID:
    - Bullet points, numbered lists, headers
    - Starting with "So" (seriously, don't do it)
    - Formal transitions like "Furthermore" or "In conclusion"
    - Sounding like a report or a template
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        provider: str = "auto"  # "auto", "groq", or "gemini"
    ):
        """
        Initialize LLM summarizer.
        
        Args:
            api_key: Gemini API key (uses GEMINI_API_KEY env var if not provided).
            groq_api_key: Groq API key (uses GROQ_API_KEY env var if not provided).
            provider: "auto" (try Groq first), "groq", or "gemini".
        """
        self.provider = provider
        self.groq_client = None
        self.gemini_client = None
        self.active_provider = None
        
        # Initialize Groq if available
        groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if GROQ_AVAILABLE and groq_key:
            self.groq_client = Groq(api_key=groq_key)
            self.active_provider = "groq"
            print(f"[GeminiSummarizer] Groq initialized")
            print(f"  Model: {self.GROQ_MODEL}")
        
        # Initialize Gemini as fallback
        if GEMINI_AVAILABLE:
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            if os.environ.get("GEMINI_API_KEY"):
                self.gemini_client = genai.Client()
                if not self.active_provider:
                    self.active_provider = "gemini"
                print(f"[GeminiSummarizer] Gemini initialized (fallback)")
                print(f"  Model: {self.GEMINI_MODEL}")
        
        # Force provider if specified
        if provider == "groq" and self.groq_client:
            self.active_provider = "groq"
        elif provider == "gemini" and self.gemini_client:
            self.active_provider = "gemini"
        
        if not self.active_provider:
            raise ValueError("No LLM provider available. Set GROQ_API_KEY or GEMINI_API_KEY.")
        
        print(f"[GeminiSummarizer] Active provider: {self.active_provider}")
    
    @property
    def model_name(self):
        """Return current model name."""
        return self.GROQ_MODEL if self.active_provider == "groq" else self.GEMINI_MODEL
    
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
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call the active LLM provider with automatic fallback.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt (used by Groq).
            
        Returns:
            Generated text response.
        """
        providers_to_try = []
        
        # Build provider order
        if self.active_provider == "groq" and self.groq_client:
            providers_to_try.append("groq")
        if self.gemini_client:
            providers_to_try.append("gemini")
        if self.active_provider == "gemini" and self.groq_client:
            providers_to_try.append("groq")  # Try Groq as fallback for Gemini
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                if provider == "groq":
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    response = self.groq_client.chat.completions.create(
                        model=self.GROQ_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2048
                    )
                    return response.choices[0].message.content.strip()
                    
                elif provider == "gemini":
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    response = self.gemini_client.models.generate_content(
                        model=self.GEMINI_MODEL,
                        contents=full_prompt
                    )
                    return response.text.strip()
                    
            except Exception as e:
                error_str = str(e)
                last_error = e
                print(f"[GeminiSummarizer] {provider} error: {error_str[:100]}...")
                
                # Check for rate limit and try fallback
                if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
                    print(f"[GeminiSummarizer] Rate limited on {provider}, trying fallback...")
                    continue
                else:
                    # Non-rate-limit error, still try fallback
                    continue
        
        # All providers failed
        raise last_error or Exception("All LLM providers failed")

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
            summary = self._call_llm(prompt)
            
            return SummaryResult(
                summary=summary,
                model=self.model_name,
                input_docs=len(documents),
                input_tokens=0,
                output_tokens=0,
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
            return self._call_llm(prompt)
        except Exception as e:
            return f"Cannot generate explanation: {str(e)}"

    def extract_verification_queries(self, content: str, title: str = "") -> list:
        """
        Extract search queries for fact-checking an article.

        Args:
            content: Article content.
            title: Article title.

        Returns:
            List of 3 search queries for verification.
        """
        prompt = f"""Extract 3 specific search queries to fact-check this article.

TITLE: {title}
CONTENT: {content[:3000]}

Generate 3 queries that would help verify the key claims:
1. A query about the main subject/entity (company, person, product)
2. A query about a specific claim or statistic mentioned
3. A query to find official sources or announcements

Return ONLY the 3 queries, one per line, no numbering or explanation.
Keep each query concise (5-10 words max).

Queries:"""

        try:
            result = self._call_llm(prompt)
            queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
            # Limit to 3 queries, fallback to title if empty
            queries = queries[:3] if queries else [title if title else "news verification"]
            return queries
        except Exception as e:
            print(f"[QueryGen] Error: {e}")
            return [title if title else "news verification"]

    def verify_article(
        self,
        title: str,
        content: str,
        hoax_probability: float = None,
        search_results: list = None,
        source_trusted: bool = None,
        source_name: str = None,
        is_outlier: bool = False,
        outlier_similarity: float = 1.0
    ) -> dict:
        """
        Verify article trustworthiness using LLM analysis.

        Args:
            title: Article title.
            content: Article content.
            hoax_probability: Hoax detection model probability (0-1).
            search_results: List of search result dicts for corroboration.
            source_trusted: Whether the source domain is trusted (None=unknown/text input).
            source_name: Name of the source domain.
            is_outlier: Whether article is semantically distant from search results.
            outlier_similarity: Cosine similarity to search result centroid (0-1).

        Returns:
            Dict with verdict, summary, reasoning, and confidence.
        """
        # Build source credibility context
        source_context = ""
        if source_trusted is not None:
            if source_trusted:
                source_context = f"\n\nSOURCE CREDIBILITY: TRUSTED - {source_name} (established news outlet)"
            else:
                source_context = f"\n\nSOURCE CREDIBILITY: UNKNOWN/UNVERIFIED - {source_name} (not a recognized news source - BE SKEPTICAL)"
        elif source_name:
            source_context = f"\n\nSOURCE CREDIBILITY: UNKNOWN - {source_name} (could not verify source reputation)"

        # Build outlier context
        outlier_context = ""
        if is_outlier:
            outlier_context = f"\n\nOUTLIER DETECTION: WARNING - This article's content is SIGNIFICANTLY DIFFERENT from mainstream coverage (similarity: {outlier_similarity:.1%}). This may indicate fabricated or misleading content."
        elif outlier_similarity < 0.8:
            outlier_context = f"\n\nOUTLIER DETECTION: CAUTION - This article differs somewhat from mainstream coverage (similarity: {outlier_similarity:.1%})."

        # Build context from search results
        search_context = ""
        if search_results:
            search_context = "\n\nRELATED SOURCES FOUND:\n"
            for i, r in enumerate(search_results[:5]):
                search_context += f"{i+1}. {r.get('title', 'No title')} - {r.get('source', '')}\n"

        # Build hoax detection context
        hoax_context = ""
        if hoax_probability is not None:
            hoax_label = "HOAX" if hoax_probability >= 0.5 else "VALID"
            hoax_context = f"\n\nAI HOAX DETECTION: {hoax_label} ({hoax_probability:.1%} hoax probability)"

        prompt = f"""Hey, I need you to check out this news article and tell me what you think.

Just talk to me like a friend - tell me what the article is about, whether you think it's trustworthy, and why. Be casual and natural about it.

CRITICAL SOURCE EVALUATION RULES:
- If source is UNKNOWN/UNVERIFIED: Default to UNCERTAIN unless claims are corroborated by trusted sources
- If article is based on "leaks", "rumors", "tipsters", or "reports suggest": Be extra skeptical
- If article makes specific claims about future products/events without official confirmation: Lean toward UNCERTAIN
- Only mark as TRUSTABLE if: source is trusted OR claims are verified by multiple trusted sources

IMPORTANT - VARY YOUR OPENING:
- Do NOT start with "So" or "So the article"
- Do NOT start with "Honestly" every time
- Mix it up! Start with the main topic, a surprising fact, or your direct reaction.
- Examples: "This piece covers...", "I checked out this story and...", "Here's what I found about...", "The article claims that...", "Looks like this story is..."

ARTICLE TITLE: {title}
{source_context}
{outlier_context}

ARTICLE CONTENT:
"{content[:12000]}"
{hoax_context}
{search_context}

Give me your honest take in 2-3 natural paragraphs. Don't use templates or structured formats - just talk naturally. Make sure to mention the source credibility in your analysis.

At the very end, on a new line, just write one word for your overall verdict: TRUSTABLE, UNCERTAIN, or NOT TRUSTABLE

Go ahead:"""

        try:
            result_text = self._call_llm(prompt)
            
            # Extract verdict from natural text (look for TRUSTABLE/UNCERTAIN/NOT TRUSTABLE at end)
            import re
            
            # Clean up the text
            lines = result_text.strip().split('\n')
            
            # Try to find verdict in last few lines
            verdict = "UNCERTAIN"
            for line in reversed(lines[-5:]):
                line_upper = line.strip().upper()
                if "NOT TRUSTABLE" in line_upper:
                    verdict = "NOT TRUSTABLE"
                    break
                elif "TRUSTABLE" in line_upper and "NOT" not in line_upper:
                    verdict = "TRUSTABLE"
                    break
                elif "UNCERTAIN" in line_upper:
                    verdict = "UNCERTAIN"
                    break
            
            # Get the conversational analysis (everything except the verdict line)
            analysis = '\n'.join(lines[:-1]).strip() if len(lines) > 1 else result_text
            
            print(f"[LLM Verifier] Verdict: {verdict}")
            
            return {
                "verdict": verdict,
                "confidence": 0.8 if verdict == "TRUSTABLE" else 0.5,
                "summary": analysis,  # The natural conversational text
                "reasoning": analysis,  # Same - it's all in the natural text now
                "red_flags": [],
                "credibility_signals": []
            }
            
        except Exception as e:
            print(f"[LLM Verifier] Error: {e}")
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "summary": f"Error analyzing article: {str(e)}",
                "reasoning": "LLM analysis failed",
                "red_flags": [],
                "credibility_signals": []
            }

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

    def generate_search_query(self, text: str, language: str = 'auto') -> str:
        """
        Generate a targeted search query to verify claims in the text.
        Uses AI-agent approach for smarter, more relevant queries.
        
        Args:
            text: The input text containing claims.
            language: 'id' for Indonesian, 'en' for English, 'auto' to detect.
            
        Returns:
            An optimized search query string.
        """
        # Auto-detect language if needed
        if language == 'auto':
            id_keywords = {'yang', 'dan', 'di', 'untuk', 'dengan', 'pada', 'ini', 'adalah'}
            words = set(text.lower().split()[:50])
            language = 'id' if len(words & id_keywords) >= 4 else 'en'
        
        if language == 'id':
            # Indonesian: focus on local entities and news
            prompt = f"""Anda adalah agen peneliti fakta Indonesia.

TUGAS: Buat query pencarian untuk memverifikasi klaim utama dalam teks berikut.

TEKS:
"{text[:1500]}"

PANDUAN:
1. Identifikasi KLAIM UTAMA (siapa, apa, kapan, dimana).
2. Sertakan nama ENTITAS SPESIFIK (orang, organisasi, tempat, program).
3. Gunakan kata kunci bahasa Indonesia yang tepat.
4. Kembalikan HANYA query string. Tanpa tanda kutip, tanpa penjelasan.

QUERY:"""
        else:
            # English: focus on credible sources
            prompt = f"""You are a fact-checking research agent.

TASK: Generate a precise search query to verify the main claim in this article.

ARTICLE:
"{text[:1500]}"

REQUIREMENTS:
1. Extract the PRIMARY claim (who, what, when, where).
2. Include SPECIFIC entity names (people, organizations, locations).
3. Include dates or time references if mentioned.
4. Add credible source hints: site:reuters.com OR site:bbc.com OR site:apnews.com
5. Return ONLY the query string. No quotes, no explanations.

QUERY:"""

        try:
            result = self._call_llm(prompt)
            query = result.strip().strip('"').strip("'")
            print(f"[GeminiSummarizer] Generated query ({language}): {query[:80]}...")
            return query
        except Exception as e:
            print(f"[GeminiSummarizer] Query Generation Error: {e}")
            # Fallback: extract first meaningful sentence
            return text.split('\n')[0][:100]

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