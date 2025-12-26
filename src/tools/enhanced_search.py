"""
Enhanced Search Tool with Crawl4AI Content Fetching

Combines web search (DuckDuckGo) with content extraction (Crawl4AI)
to get full article content from search results for better analysis.

Usage:
    from src.tools.enhanced_search import EnhancedSearcher
    searcher = EnhancedSearcher()
    results = await searcher.search_and_fetch("query", max_fetch=3)
"""

import asyncio
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

# Try importing crawl4ai
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
    HAS_CRAWL4AI = True
except ImportError:
    HAS_CRAWL4AI = False
    print("[EnhancedSearcher] Warning: crawl4ai not installed")

# DuckDuckGo search
try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    print("[EnhancedSearcher] Warning: ddgs not installed")


@dataclass
class SearchResult:
    """Search result with optional fetched content."""
    title: str
    url: str
    snippet: str
    domain: str
    
    # Fetched content (if available)
    content: Optional[str] = None
    content_length: int = 0
    fetch_success: bool = False
    
    # Credibility info
    is_trusted: bool = False
    source_name: str = ""


class EnhancedSearcher:
    """
    Enhanced web searcher that:
    1. Searches using DuckDuckGo
    2. Fetches full content from top results using Crawl4AI
    3. Analyzes source credibility
    """
    
    # Site-specific CSS selectors for Indonesian news sites
    SITE_SELECTORS = {
        'kompas.com': '.read__content .clearfix, .read__content',
        'cnnindonesia.com': '.detail-text, .detail__body-text',
        'detik.com': '.detail__body-text, .itp_bodycontent',
        'tribunnews.com': '.txt-article, .content__article',
        'liputan6.com': '.article-content-body__item-content',
        'tempo.co': '.detail-konten, .detail-in',
        'antaranews.com': '.post-content',
        'republika.co.id': '.artikel',
    }
    
    # Default article selectors (fallback)
    DEFAULT_SELECTORS = "article, main, .article, .content, .post, [itemprop='articleBody'], .article-body, .article-content"
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        print(f"[EnhancedSearcher] Initialized")
        print(f"  Crawl4AI: {'✓' if HAS_CRAWL4AI else '✗'}")
        print(f"  DDGS: {'✓' if HAS_DDGS else '✗'}")
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ""
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is Indonesian or English.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            'id' for Indonesian, 'en' for English/other.
        """
        # Indonesian common function words
        id_keywords = {'yang', 'dan', 'di', 'untuk', 'dengan', 'pada', 'ini', 'itu', 
                      'adalah', 'akan', 'dari', 'ke', 'tidak', 'juga', 'sudah',
                      'bisa', 'lebih', 'tersebut', 'dalam', 'oleh', 'bahwa'}
        
        # Take first 100 words
        words = set(text.lower().split()[:100])
        id_count = len(words & id_keywords)
        
        # If 5+ Indonesian keywords found, classify as Indonesian
        return 'id' if id_count >= 5 else 'en'
    
    def search(self, query: str, timelimit: Optional[str] = None, region: Optional[str] = None) -> List[Dict]:
        """
        Search using DuckDuckGo (synchronous).
        
        Args:
            query: Search query.
            timelimit: 'd' (day), 'w' (week), 'm' (month), or None.
            region: Region code ('id-id' for Indonesia, 'wt-wt' for worldwide).
            
        Returns:
            List of search result dicts.
        """
        if not HAS_DDGS:
            print("[EnhancedSearcher] DDGS not available")
            return []
        
        # Default to worldwide if no region specified
        search_region = region if region else 'wt-wt'
        print(f"[EnhancedSearcher] Searching: {query[:60]}... (region={search_region})")
        results = []
        
        try:
            with DDGS() as ddgs:
                raw_results = ddgs.text(
                    query,
                    region=search_region,
                    safesearch='moderate',
                    timelimit=timelimit,
                    max_results=self.max_results
                )
                
                seen_urls = set()
                if raw_results:
                    for r in raw_results:
                        url = r.get('href', '')
                        if url and url not in seen_urls:
                            results.append({
                                'title': r.get('title', ''),
                                'url': url,
                                'snippet': r.get('body', ''),
                                'domain': self._extract_domain(url)
                            })
                            seen_urls.add(url)
        except Exception as e:
            print(f"[EnhancedSearcher] Search error: {e}")
        
        print(f"[EnhancedSearcher] Found {len(results)} results")
        return results

    def search_sync(self, query: str, max_results: int = 5, region: Optional[str] = None) -> List[SearchResult]:
        """
        Search and return SearchResult objects (synchronous).

        Args:
            query: Search query.
            max_results: Maximum number of results.

            region: Region code ('id-id' for Indonesia, 'wt-wt' for worldwide).

        Returns:
            List of SearchResult objects.
        """
        raw_results = self.search(query, region=region)
        results = []
        for r in raw_results[:max_results]:
            results.append(SearchResult(
                title=r.get('title', ''),
                url=r.get('url', ''),
                snippet=r.get('snippet', ''),
                domain=r.get('domain', '')
            ))
        return results

    async def fetch_content(self, url: str, timeout: int = 15000) -> Optional[str]:
        """
        Fetch clean content from URL using Crawl4AI.
        
        Args:
            url: URL to fetch.
            timeout: Timeout in milliseconds.
            
        Returns:
            Clean article content or None.
        """
        if not HAS_CRAWL4AI:
            return None
        
        try:
            # Handle Kompas pagination - get full article
            fetch_url = url
            if 'kompas.com' in url and '?page=all' not in url and '&page=all' not in url:
                fetch_url = url + ('&' if '?' in url else '?') + 'page=all'
                print(f"[EnhancedSearcher] Using full-page URL for Kompas")
            
            # Get domain for site-specific selector
            domain = self._extract_domain(fetch_url)
            css_selector = self.DEFAULT_SELECTORS
            for site, selector in self.SITE_SELECTORS.items():
                if site in domain:
                    css_selector = selector
                    print(f"[EnhancedSearcher] Using site-specific selector for {site}")
                    break
            
            browser_config = BrowserConfig(headless=True, verbose=False)
            
            crawler_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=timeout,
                css_selector=css_selector,
                word_count_threshold=30,  # Skip blocks with < 30 words
                excluded_tags=["nav", "footer", "aside", "header", "script", "style", 
                              "noscript", "iframe", "form", "button", "svg"],
                exclude_external_links=True,
                exclude_social_media_links=True,
                remove_overlay_elements=True,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.5,
                        threshold_type="fixed",
                        min_word_threshold=15
                    ),
                    options={"ignore_links": True, "ignore_images": True}
                ),
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=fetch_url, config=crawler_config)
                
                if result.success:
                    # Try fit_markdown first (best quality), then raw_markdown
                    content = None
                    if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
                        content = result.markdown.fit_markdown
                    elif hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown:
                        content = result.markdown.raw_markdown
                    elif result.markdown:
                        content = str(result.markdown)
                    
                    if content and len(content) > 100:
                        print(f"[EnhancedSearcher] Fetched {len(content)} chars from {domain}")
                        return content
                    else:
                        print(f"[EnhancedSearcher] Content too short from {domain}: {len(content) if content else 0} chars")
                else:
                    print(f"[EnhancedSearcher] Crawl failed: {result.error_message}")
                    
        except Exception as e:
            print(f"[EnhancedSearcher] Fetch error for {url[:50]}: {e}")
        
        return None
    
    async def search_and_fetch(
        self, 
        query: str, 
        max_fetch: int = 3,
        timelimit: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search and fetch content from top results.
        
        Args:
            query: Search query.
            max_fetch: Max number of results to fetch full content for.
            timelimit: Time limit for search.
            
        Returns:
            List of SearchResult with fetched content.
        """
        # Step 1: Search
        raw_results = self.search(query, timelimit)
        
        if not raw_results:
            return []
        
        # Step 2: Fetch content from top results
        results = []
        fetch_count = 0
        
        for r in raw_results:
            domain = r.get('domain', '')
            
            result = SearchResult(
                title=r.get('title', ''),
                url=r.get('url', ''),
                snippet=r.get('snippet', ''),
                domain=domain
            )
            
            # Fetch content for top N results
            if fetch_count < max_fetch and HAS_CRAWL4AI:
                print(f"[EnhancedSearcher] Fetching: {domain}...")
                content = await self.fetch_content(r['url'])
                
                if content:
                    result.content = content[:5000]  # Limit content size
                    result.content_length = len(content)
                    result.fetch_success = True
                    fetch_count += 1
            
            results.append(result)
        
        print(f"[EnhancedSearcher] Fetched content from {fetch_count}/{len(results)} results")
        return results
    
    def search_and_fetch_sync(
        self,
        query: str,
        max_fetch: int = 3,
        timelimit: Optional[str] = None
    ) -> List[SearchResult]:
        """Synchronous wrapper for search_and_fetch."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.search_and_fetch(query, max_fetch, timelimit)
        )

    def fetch_url_content(self, url: str, timeout: int = 15000) -> Optional[str]:
        """Synchronous wrapper for fetch_content."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.fetch_content(url, timeout))


# Factory function
def create_enhanced_searcher(max_results: int = 10) -> EnhancedSearcher:
    """Create an enhanced searcher."""
    return EnhancedSearcher(max_results=max_results)


if __name__ == "__main__":
    # Demo
    async def demo():
        searcher = EnhancedSearcher(max_results=5)
        
        results = await searcher.search_and_fetch(
            "The Fed pangkas suku bunga",
            max_fetch=2
        )
        
        print(f"\n{'='*60}")
        print(f"Found {len(results)} results")
        
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r.title}")
            print(f"    Domain: {r.domain}")
            print(f"    Fetched: {'Yes' if r.fetch_success else 'No'}")
            if r.content:
                print(f"    Content: {r.content[:200]}...")
    
    asyncio.run(demo())
