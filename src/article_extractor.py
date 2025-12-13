"""
Article Extractor using Crawl4AI

Clean article content extraction for hoax detection.
Uses Crawl4AI's PruningContentFilter to remove garbage and extract main article text.

Usage:
    from src.article_extractor import extract_article
    content = await extract_article("https://example.com/news/article")
"""

import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Try to import crawl4ai (may not be installed)
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
    HAS_CRAWL4AI = True
except ImportError:
    HAS_CRAWL4AI = False
    print("[ArticleExtractor] Warning: crawl4ai not installed. Using fallback.")


@dataclass
class ArticleContent:
    """Extracted article content."""
    url: str
    title: str
    content: str  # Clean article text
    raw_length: int  # Original HTML length
    clean_length: int  # Cleaned content length
    success: bool
    error: Optional[str] = None


async def extract_article(
    url: str,
    timeout: int = 30000,
    wait_for_js: bool = True
) -> ArticleContent:
    """
    Extract clean article content from a URL using Crawl4AI.
    
    Args:
        url: Article URL to extract.
        timeout: Page load timeout in milliseconds.
        wait_for_js: Whether to wait for JavaScript to load.
        
    Returns:
        ArticleContent with cleaned text.
    """
    if not HAS_CRAWL4AI:
        return await _fallback_extract(url)
    
    try:
        # Browser configuration - headless, fast
        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )
        
        # Crawler configuration with content filtering
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,  # Always fetch fresh
            page_timeout=timeout,
            
            # Target article content specifically (common news site selectors)
            css_selector="article, main, .article, .content, .post, .story, .news-content, .read__content, .detail__body-text, [itemprop='articleBody']",
            
            # Remove garbage elements
            excluded_tags=["nav", "footer", "aside", "header", "script", "style", "noscript", "form", "button", "iframe"],
            remove_overlay_elements=True,
            
            # Smart content filtering - more aggressive threshold
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.6,  # Higher = more aggressive garbage removal
                    threshold_type="fixed",
                    min_word_threshold=10  # Minimum words per block
                ),
                options={
                    "ignore_links": True,  # Don't include link URLs
                    "ignore_images": True,  # Don't include image references
                }
            ),
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)
            
            if not result.success:
                return ArticleContent(
                    url=url,
                    title="",
                    content="",
                    raw_length=0,
                    clean_length=0,
                    success=False,
                    error=result.error_message or "Crawl failed"
                )
            
            # Get clean content from fit_markdown (filtered) or fall back to raw
            if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
                clean_content = result.markdown.fit_markdown
            elif hasattr(result.markdown, 'raw_markdown'):
                clean_content = result.markdown.raw_markdown
            else:
                clean_content = str(result.markdown) if result.markdown else ""
            
            # Extract title from the page
            title = ""
            if result.metadata and "title" in result.metadata:
                title = result.metadata["title"]
            
            # Debug output
            raw_len = len(result.html) if result.html else 0
            clean_len = len(clean_content)
            print(f"\n[ArticleExtractor] Extracted from {url}")
            print(f"  Raw HTML: {raw_len:,} chars")
            print(f"  Clean content: {clean_len:,} chars ({clean_len/max(raw_len,1)*100:.1f}% of raw)")
            print(f"  Title: {title[:60]}..." if len(title) > 60 else f"  Title: {title}")
            # Show content preview (first 300 chars)
            preview = clean_content[:300].replace('\n', ' ').strip()
            print(f"  Preview: {preview}...")
            
            return ArticleContent(
                url=url,
                title=title,
                content=clean_content,
                raw_length=raw_len,
                clean_length=clean_len,
                success=True
            )
            
    except Exception as e:
        print(f"[ArticleExtractor] Error: {e}")
        return ArticleContent(
            url=url,
            title="",
            content="",
            raw_length=0,
            clean_length=0,
            success=False,
            error=str(e)
        )


async def _fallback_extract(url: str) -> ArticleContent:
    """
    Fallback extraction using requests + BeautifulSoup when crawl4ai isn't available.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        response = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove garbage elements
        for tag in soup.find_all(["nav", "footer", "aside", "header", "script", "style", "noscript"]):
            tag.decompose()
        
        # Get title
        title = soup.title.string if soup.title else ""
        
        # Get article content - try common article containers
        article = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
        
        if article:
            content = article.get_text(separator=" ", strip=True)
        else:
            # Fallback to body
            body = soup.find("body")
            content = body.get_text(separator=" ", strip=True) if body else ""
        
        # Clean up extra whitespace
        import re
        content = re.sub(r'\s+', ' ', content).strip()
        
        return ArticleContent(
            url=url,
            title=title,
            content=content,
            raw_length=len(response.text),
            clean_length=len(content),
            success=True
        )
        
    except Exception as e:
        return ArticleContent(
            url=url,
            title="",
            content="",
            raw_length=0,
            clean_length=0,
            success=False,
            error=str(e)
        )


# Synchronous wrapper for use in Flask/non-async code
def extract_article_sync(url: str, timeout: int = 30000) -> ArticleContent:
    """
    Synchronous wrapper for extract_article.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(extract_article(url, timeout))


if __name__ == "__main__":
    # Test the extractor
    async def test():
        test_urls = [
            "https://www.kompas.com/cekfakta/read/2023/01/01/000000382/hoax-pesan-berantai",
            "https://www.bbc.com/news"
        ]
        
        for url in test_urls:
            print(f"\n{'='*60}")
            print(f"Testing: {url}")
            result = await extract_article(url)
            print(f"Success: {result.success}")
            if result.success:
                print(f"Content preview: {result.content[:300]}...")
            else:
                print(f"Error: {result.error}")
    
    asyncio.run(test())
