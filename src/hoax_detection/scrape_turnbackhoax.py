"""
TurnBackHoax.id Scraper

Scrapes fresh hoax news data from turnbackhoax.id for training.
This is the official Indonesian hoax clarification site managed by Mafindo.

Usage:
    python src/hoax_detection/scrape_turnbackhoax.py --pages 50

Output: data/scraped_turnbackhoax.csv
"""

import os
import re
import time
import random
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
import pandas as pd


# Configuration
BASE_URL = "https://turnbackhoax.id"
OUTPUT_PATH = "data/scraped_turnbackhoax.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}


@dataclass
class HoaxArticle:
    """Scraped hoax article data."""
    url: str
    title: str
    content: str
    category: str
    date: str
    label: int  # 1 = hoax


def get_page_urls(page_num: int = 1) -> List[str]:
    """
    Get article URLs from a listing page.
    
    Args:
        page_num: Page number to scrape.
        
    Returns:
        List of article URLs.
    """
    # New TurnBackHoax uses /articles endpoint with pagination
    url = f"{BASE_URL}/articles?page={page_num}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        article_links = []
        
        # Find all links containing /articles/
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # Match article URLs like /articles/30652--salah-...
            if "/articles/" in href and href != "/articles":
                full_url = href if href.startswith("http") else f"{BASE_URL}{href}"
                if full_url not in article_links:
                    article_links.append(full_url)
        
        return list(set(article_links))[:30]  # Limit per page
        
    except Exception as e:
        print(f"  Error fetching page {page_num}: {e}")
        return []


def scrape_article(url: str) -> Optional[HoaxArticle]:
    """
    Scrape a single article.
    
    Args:
        url: Article URL.
        
    Returns:
        HoaxArticle or None if failed.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Get title
        title_elem = soup.find("h1", class_="entry-title") or soup.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Get content
        content_div = soup.find("div", class_="entry-content")
        if content_div:
            # Remove script/style tags
            for tag in content_div.find_all(["script", "style", "aside"]):
                tag.decompose()
            content = content_div.get_text(separator=" ", strip=True)
        else:
            content = ""
        
        # Clean content
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Determine category from URL
        if "/salah/" in url.lower() or "hoax" in url.lower():
            category = "HOAX"
            label = 1
        elif "/benar/" in url.lower():
            category = "VALID"
            label = 0
        else:
            category = "HOAX"  # Default for turnbackhoax
            label = 1
        
        # Get date
        date_elem = soup.find("time", class_="entry-date")
        date = date_elem.get("datetime", "")[:10] if date_elem else ""
        
        if len(content) < 100:
            return None
        
        return HoaxArticle(
            url=url,
            title=title,
            content=content[:5000],  # Limit length
            category=category,
            date=date,
            label=label
        )
        
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def scrape_turnbackhoax(max_pages: int = 30, delay: float = 1.0) -> pd.DataFrame:
    """
    Scrape multiple pages from TurnBackHoax.id.
    
    Args:
        max_pages: Maximum number of listing pages to scrape.
        delay: Delay between requests (be polite!).
        
    Returns:
        DataFrame with scraped articles.
    """
    print("=" * 60)
    print("TurnBackHoax.id Scraper")
    print("=" * 60)
    print(f"Target: {max_pages} pages")
    print()
    
    all_articles = []
    all_urls = set()
    
    # Collect article URLs
    print("[1/2] Collecting article URLs...")
    for page in range(1, max_pages + 1):
        print(f"  Page {page}/{max_pages}...", end=" ")
        urls = get_page_urls(page)
        new_urls = [u for u in urls if u not in all_urls]
        all_urls.update(new_urls)
        print(f"found {len(new_urls)} new articles")
        
        time.sleep(delay * random.uniform(0.5, 1.5))
        
        if len(urls) == 0:
            print("  No more pages found, stopping.")
            break
    
    print(f"\nTotal unique URLs: {len(all_urls)}")
    
    # Scrape articles
    print("\n[2/2] Scraping articles...")
    for i, url in enumerate(all_urls, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(all_urls)}")
        
        article = scrape_article(url)
        if article:
            all_articles.append({
                "url": article.url,
                "title": article.title,
                "content": article.content,
                "category": article.category,
                "date": article.date,
                "label": article.label
            })
        
        time.sleep(delay * random.uniform(0.3, 0.8))
    
    # Create DataFrame
    df = pd.DataFrame(all_articles)
    
    print(f"\n{'=' * 60}")
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Total articles scraped: {len(df)}")
    if len(df) > 0:
        print(f"Label distribution: {dict(df['label'].value_counts())}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape TurnBackHoax.id")
    parser.add_argument("--pages", type=int, default=30, help="Number of pages to scrape")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()
    
    df = scrape_turnbackhoax(max_pages=args.pages, delay=args.delay)
    
    if len(df) > 0:
        df.to_csv(args.output, index=False)
        file_size = os.path.getsize(args.output) / 1024
        print(f"\nSaved to {args.output} ({file_size:.1f} KB)")
    else:
        print("\nNo articles scraped.")


if __name__ == "__main__":
    main()
