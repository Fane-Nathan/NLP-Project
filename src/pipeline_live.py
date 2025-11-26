# src/pipeline_live.py
import json
from datetime import datetime
from ddgs import DDGS
from src.hoax_detection import HoaxClassifier
from src.models.gemini_summarizer import create_summarizer

class LiveHistorian:
    def __init__(self, verifier=None, historian=None):
        print("[System] Initializing Live Historian...")
        # 1. The Gatekeeper (Your Hoax Detector)
        if verifier:
            self.verifier = verifier
        else:
            self.verifier = HoaxClassifier(model_path="models/hoax_indobert_lora")
        
        # 2. The Historian (Your Gemini Summarizer)
        if historian:
            self.historian = historian
        else:
            self.historian = create_summarizer()

    def search_and_verify(self, query, max_results=10):
        print(f"\n[Scout] Searching the web for: '{query}'...")
        
        valid_intel = []
        
        # Use DuckDuckGo to get 'news' or 'text' results which often have dates
        with DDGS() as ddgs:
            # 'timelimit="m"' (past month) or 'y' (past year) helps get LATEST data
            raw_results = ddgs.text(
                query, 
                region='wt-wt', 
                safesearch='moderate', 
                timelimit='y', # Focus on the last year for "latest" context
                max_results=max_results
            )

        print(f"[Scout] Found {len(raw_results)} raw reports. Verifying...")

        for res in raw_results:
            text_content = f"{res['title']}. {res['body']}"
            
            # --- STEP 2: HOAX CHECK ---
            # Check if this specific search result is credible
            credibility = self.verifier.predict(text_content)
            
            # Relaxed threshold for demo: only reject if > 70% confident it's a hoax
            if credibility.hoax_probability > 0.7:
                print(f"  ❌ Rejected (Hoax {credibility.hoax_probability:.0%}): {res['title'][:40]}...")
                continue
                
            # --- STEP 3: NORMALIZE ---
            # We assume search engine results are 'current' relative to their pub date
            # If DDG doesn't give a date, we treat it as 'undated' or 'current context'
            valid_intel.append({
                'content': text_content,
                'date': res.get('date', 'Unknown Date'), # DDG sometimes provides this
                'source': res['title'],
                'url': res['href']
            })
            print(f"  ✅ Verified: {res['title'][:40]}...")

        return valid_intel

    def generate_live_history(self, query):
        # 1. Gather Intel
        intel = self.search_and_verify(query)
        
        if not intel:
            return "No credible intelligence found on this topic."

        print(f"\n[Historian] Synthesizing narrative from {len(intel)} verified sources...")
        
        # 2. Construct Narrative
        # We pass the verified web results to Gemini
        result = self.historian.summarize(
            documents=intel,
            query=query,
            style="timeline" # This triggers the chronological prompt
        )
        
        return result.summary

if __name__ == "__main__":
    agent = LiveHistorian()
    
    # Example Query: Something evolving right now or recently
    topic = "Perkembangan terbaru proyek IKN Nusantara 2025"
    
    narrative = agent.generate_live_history(topic)
    
    print("\n" + "="*60)
    print(f"📜 LIVE HISTORICAL RECORD: {topic}")
    print("="*60)
    print(narrative)
    print("="*60)
