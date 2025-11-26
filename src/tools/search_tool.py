from ddgs import DDGS
from typing import List, Dict, Optional
import json

class WebSearcher:
    """
    Free web search tool using DuckDuckGo's browser-mimicking API.
    No API key required.
    """
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def search(self, query: str, timelimit: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Performs a text search and returns a list of results.
        timelimit: 'd' (day), 'w' (week), 'm' (month), 'y' (year), or None (no limit)
        """
        print(f"[WebSearcher] Searching for: {query} (Time: {timelimit})")
        results = []
        try:
            # DDGS() acts as a context manager to handle the session
            with DDGS() as ddgs:
                # .text() simulates a standard browser search
                raw_results = ddgs.text(
                    query,
                    region='wt-wt', # 'wt-wt' = No region (Global)
                    safesearch='moderate',
                    timelimit=timelimit,
                    max_results=self.max_results
                )
                
                # Deduplicate and format
                seen_urls = set()
                if raw_results:
                    for r in raw_results:
                        if r['href'] not in seen_urls:
                            results.append({
                                "title": r['title'],
                                "link": r['href'],
                                "snippet": r['body']
                            })
                            seen_urls.add(r['href'])
                            
        except Exception as e:
            print(f"[WebSearcher] Error: {e}")
            return []

        return results

    def get_formatted_results(self, query: str, timelimit: Optional[str] = None) -> str:
        """
        Returns search results formatted as a string for the LLM.
        """
        results = self.search(query, timelimit)
        if not results:
            return "No search results found."
            
        formatted = f"Search Results for '{query}':\n\n"
        for i, res in enumerate(results, 1):
            formatted += f"[{i}] {res['title']}\n"
            formatted += f"    Source: {res['link']}\n"
            formatted += f"    Context: {res['snippet']}\n\n"
            
        return formatted

# Demo
if __name__ == "__main__":
    searcher = WebSearcher()
    print(searcher.get_formatted_results("latest python features 2025"))
