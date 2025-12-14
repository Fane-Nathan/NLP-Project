"""
TDSM Full Workspace - News Verification with Kokoro TTS
Complete pipeline: URL Fetch → Trust Layer → Web Search → KG → Verdict → TTS
"""

import os
import re
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Third-party imports
from typing import Optional, List, Dict, Tuple

import requests
from flask import Flask, render_template, request, jsonify

# Configure Flask with templates folder
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Ensure proper UTF-8 encoding for JSON and templates (fixes emoji display)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'

# Global TTS instance (Kokoro)
tts_voice = None
tts_enabled = True

def init_tts():
    """Initialize Kokoro TTS."""
    global tts_voice
    try:
        from src.voice_kokoro import EchoVoice
        tts_voice = EchoVoice(server_mode=True)
        print("✓ Kokoro TTS initialized")
        return True
    except Exception as e:
        print(f"[Warning] Kokoro TTS not available: {e}")
        return False

@app.route('/stream_audio')
def stream_audio():
    """Stream audio bytes to the client"""
    if not tts_voice:
        return "TTS not initialized", 404
        
    def generate():
        for chunk in tts_voice.get_audio_stream():
            yield chunk

    from flask import Response
    return Response(generate(), mimetype='application/octet-stream')


@app.route('/api/tts/generate', methods=['POST'])
def generate_tts():
    """Generate audio for a single text chunk using edge-tts."""
    import asyncio
    import edge_tts
    import io
    from flask import Response
    
    data = request.json or {}
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Limit text length
    text = text[:2000]
    
    voice = "en-US-EmmaNeural"  # Friday voice
    
    async def generate_audio():
        """Generate audio using edge-tts."""
        audio_bytes = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice, rate="+5%", pitch="+3Hz")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.write(chunk["data"])
        return audio_bytes.getvalue()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_data = loop.run_until_complete(generate_audio())
        loop.close()
        
        if audio_data:
            return Response(audio_data, mimetype='audio/mpeg')
        else:
            return jsonify({'error': 'No audio generated'}), 500
            
    except Exception as e:
        print(f"[TTS] Error generating audio: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tts/sentences', methods=['POST'])
def get_sentences():
    """Split text into sentences for chunked TTS."""
    import re
    
    data = request.json or {}
    text = data.get('text', '')
    
    if not text:
        return jsonify({'sentences': []})
    
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Limit to first 10 sentences
    sentences = sentences[:10]
    
    return jsonify({'sentences': sentences})


@app.route('/')
def index():
    """Serve the workspace interface."""
    return render_template('workspace.html')

@app.route('/api/verify', methods=['POST'])
def verify():
    """Full verification pipeline: Fetch → Trust → Search → KG → Verdict."""
    try:
        data = request.json
        input_type = data.get('type', 'text')
        content = data.get('content', '').strip()

        if not content:
            return jsonify({'error': 'No content provided'}), 400

        # Step 1: Fetch content if URL
        documents = []
        source_url = None
        article_title = None  # Store title for search query

        if input_type == 'url':
            source_url = content
            fetched = fetch_url_content(content)
            if fetched:
                documents = [fetched['text']]
                article_title = fetched.get('title', '')  # Get title for search
            else:
                return jsonify({'error': 'Could not fetch URL content'}), 400
        else:
            # Split by paragraphs
            documents = [p.strip() for p in content.split('\n\n') if p.strip()]
            if not documents:
                documents = [content]

        # Document analysis
        doc_analysis = []
        for doc in documents:
            doc_analysis.append({
                'text': doc[:200] + ('...' if len(doc) > 200 else ''),
                'status': 'valid',
                'credibility_score': None,
                'hoax_probability': None
            })

        filtered_docs = documents
        hoax_detected = False
        avg_hoax_prob = 0

        # Step 1.5: Check INPUT source credibility
        input_source_trusted = None  # None = unknown/text input, True/False = verified
        input_source_name = None

        # Try to extract source from URL or from text content
        if source_url:
            try:
                from src.tools.source_credibility import SourceCredibilityAnalyzer
                src_analyzer = SourceCredibilityAnalyzer()
                src_analysis = src_analyzer.analyze_source(source_url, article_title or "")
                input_source_trusted = src_analysis.is_trusted
                input_source_name = src_analysis.source_name
                if src_analysis.is_suspicious:
                    input_source_trusted = False
                print(f"[InputSource] {input_source_name}: trusted={input_source_trusted}")
            except Exception as e:
                print(f"[InputSource] Error checking source: {e}")
        else:
            # Try to extract source from pasted text (e.g., "Source: XYZ")
            import re
            source_match = re.search(r'Source:\s*([^\n]+)', content, re.IGNORECASE)
            if source_match:
                input_source_name = source_match.group(1).strip()
                # Check if this source is in our trusted list
                try:
                    from src.tools.source_credibility import TRUSTED_SOURCES
                    source_lower = input_source_name.lower()
                    input_source_trusted = any(
                        trusted in source_lower or source_lower in trusted
                        for trusted in TRUSTED_SOURCES.keys()
                    )
                    if not input_source_trusted:
                        input_source_trusted = False  # Explicitly mark as untrusted
                    print(f"[InputSource] Extracted from text: {input_source_name}, trusted={input_source_trusted}")
                except Exception as e:
                    print(f"[InputSource] Error checking extracted source: {e}")

        # Step 2: Trust Layer (Credibility Analysis)
        try:
            from src.hoax_detection.credibility_report import CredibilityAnalyzer
            analyzer = CredibilityAnalyzer(
                hoax_model_path="models/hoax_indobert_lora",
                outlier_threshold_z=2.0,
                hoax_weight=0.6,
                outlier_weight=0.4
            )
            filtered_docs, report = analyzer.filter_documents(documents)

            if hasattr(report, 'documents') and report.documents:
                hoax_probs = []
                for i, doc_cred in enumerate(report.documents):
                    if i < len(doc_analysis):
                        doc_analysis[i]['credibility_score'] = doc_cred.credibility_score
                        doc_analysis[i]['hoax_probability'] = doc_cred.hoax_probability
                        hoax_probs.append(doc_cred.hoax_probability)

                        if doc_cred.hoax_label == "HOAX":
                            doc_analysis[i]['status'] = 'hoax'
                            hoax_detected = True
                        elif doc_cred.is_outlier:
                            doc_analysis[i]['status'] = 'outlier'

                if hoax_probs:
                    avg_hoax_prob = sum(hoax_probs) / len(hoax_probs)

        except ImportError:
            pass
        except Exception as e:
            print(f"Trust layer error: {e}")

        # Step 3: Web Search for corroboration (with content fetching)
        evidence = []
        search_supports = 0
        search_contradicts = 0
        fetched_contents = []  # Store fetched content for LLM analysis
        # fetched_contents = []  # Store fetched content for LLM analysis (moved inside the try block)

        try:
            from src.tools.enhanced_search import EnhancedSearcher
            searcher = EnhancedSearcher(max_results=15)

            # Generate TARGETED search queries for verification
            search_queries = []
            try:
                from src.models.gemini_summarizer import GeminiSummarizer
                query_gen = GeminiSummarizer()
                text_content = (article_title + "\n" + documents[0]) if article_title and documents else (documents[0] if documents else "")
                
                # Get 3 targeted queries (Partnerships, Specs, Exclusives)
                search_queries = query_gen.extract_verification_queries(text_content, article_title or "")
                print(f"[Verification] Generated queries: {search_queries}")
            except Exception as e:
                print(f"[Warning] Query generation failed: {e}")
                search_queries = [article_title.strip()] if article_title else ["news verification"]

            # Execute searches for ALL queries
            all_results = []
            seen_urls = set()
            
            # 1. Primary search (first query is usually the most relevant overall)
            if search_queries:
                primary_results = searcher.search_sync(search_queries[0], max_results=5)
                for r in primary_results:
                    if r.url not in seen_urls:
                        all_results.append(r)
                        seen_urls.add(r.url)
            
            # 2. Secondary searches (quick checks for specific claims)
            for q in search_queries[1:]:
                sub_results = searcher.search_sync(q, max_results=3)
                for r in sub_results:
                    if r.url not in seen_urls:
                        all_results.append(r)
                        seen_urls.add(r.url)

            # Limit total results
            results = all_results[:10]
            
            # Fetch content for top 3 unique results
            fetched_contents = []
            fetch_count = 0
            
            for res in results:
                if fetch_count >= 3: break
                
                # Skip if it's the source domain (simple check)
                if source_url and (source_url in res.url or res.url in source_url):
                    continue
                    
                print(f"[EnhancedSearcher] Fetching verification source: {res.url[:50]}...")
                content = searcher.fetch_url_content(res.url)
                if content:
                    fetched_contents.append({
                        'title': res.title,
                        'url': res.url,
                        'content': content  # fetch_url_content returns string directly
                    })
                    fetch_count += 1
            
            # Analyze source credibility of search results
            corroboration = None
            try:
                from src.tools.source_credibility import SourceCredibilityAnalyzer
                credibility_analyzer = SourceCredibilityAnalyzer()
                # Convert SearchResult objects to dicts for credibility analyzer
                results_dicts = [{'url': r.url, 'title': r.title, 'snippet': r.snippet} for r in results]
                corroboration = credibility_analyzer.analyze_search_results(
                    results_dicts,
                    original_title=article_title or "",
                    original_content=documents[0] if documents else ""
                )
            except Exception as e:
                print(f"[SourceCredibilityAnalyzer] Error: {e}")

            for r in results[:5]:
                # Check if source is trusted from credibility analysis
                is_trusted = False
                source_name = r.domain
                
                if corroboration:
                    # Find this source in the analysis
                    for src in corroboration.sources:
                        if src.url == r.url:
                            is_trusted = src.is_trusted
                            source_name = src.source_name
                            break
                
                evidence.append({
                    'title': r.title,
                    'url': r.url,
                    'source': source_name,
                    'snippet': r.snippet,
                    'is_trusted': is_trusted,
                    'has_content': r.fetch_success,
                    'content_preview': r.content[:200] + "..." if r.content and len(r.content) > 200 else r.content,
                    'supports': is_trusted,  # Trusted sources count as support
                    'contradicts': False
                })
                
                # Store fetched content for LLM analysis
                if r.content:
                    fetched_contents.append({
                        'source': source_name,
                        'url': r.url,
                        'content': r.content[:2000],
                        'is_trusted': is_trusted
                    })
            
            # Store corroboration result for LLM
            if corroboration:
                search_supports = corroboration.num_trusted
                search_contradicts = corroboration.num_suspicious

        except ImportError:
            pass
        except Exception as e:
            print(f"Web search error: {e}")

        # Step 3.5: Outlier Detection (compare input vs search results)
        input_is_outlier = False
        outlier_similarity = 1.0
        if fetched_contents and documents:
            try:
                from src.hoax_detection.outlier_detector import OutlierDetector
                # Combine input article with fetched search content
                all_docs = [documents[0]] + [fc['content'] for fc in fetched_contents if fc.get('content')]

                if len(all_docs) >= 2:  # Need at least input + 1 search result
                    print(f"[OutlierDetector] Comparing input article against {len(all_docs)-1} search results...")
                    outlier_detector = OutlierDetector(threshold_z=2.0, method="indobert")  # Semantic embeddings
                    outlier_analysis = outlier_detector.detect_outliers(all_docs)

                    # Check if input article (index 0) is an outlier
                    if outlier_analysis.results:
                        input_result = outlier_analysis.results[0]
                        input_is_outlier = input_result.is_outlier
                        outlier_similarity = input_result.similarity_to_centroid
                        print(f"[OutlierDetector] Input similarity: {outlier_similarity:.3f}, outlier: {input_is_outlier}")
            except Exception as e:
                print(f"[OutlierDetector] Error: {e}")

        # Step 4: Knowledge Graph (if available)
        kg_confidence = 0.5
        try:
            from src.models.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph(name="verify_kg")
            kg.add_documents(filtered_docs, show_progress=False)
            # KG adds confidence if entities are consistent
            kg_confidence = 0.7 if kg.graph.number_of_nodes() > 0 else 0.5
        except ImportError:
            pass

        # Step 5: LLM Verification (Gemini)
        llm_verdict = None
        try:
            from src.models.gemini_summarizer import GeminiSummarizer
            llm = GeminiSummarizer()
            
            llm_verdict = llm.verify_article(
                title=article_title or "Unknown",
                content=documents[0] if documents else "",
                hoax_probability=avg_hoax_prob if hoax_detected else None,
                search_results=evidence,
                source_trusted=input_source_trusted,
                source_name=input_source_name,
                is_outlier=input_is_outlier,
                outlier_similarity=outlier_similarity
            )
            print(f"[LLM Verifier] Summary: {llm_verdict.get('summary', '')[:100]}...")
        except Exception as e:
            print(f"[LLM Verifier] Skipped: {e}")

        # Step 6: Generate Final Verdict
        verdict, summary, confidence = generate_verdict(
            documents=documents,
            doc_analysis=doc_analysis,
            hoax_detected=hoax_detected,
            avg_hoax_prob=avg_hoax_prob,
            evidence=evidence,
            kg_confidence=kg_confidence
        )
        
        # Use LLM verdict if available
        if llm_verdict and llm_verdict.get('verdict'):
            verdict = llm_verdict.get('verdict', verdict)
            summary = llm_verdict.get('summary', summary)
            confidence = llm_verdict.get('confidence', confidence)

        return jsonify({
            'verdict': verdict,
            'summary': summary,
            'confidence': confidence,
            'documents': doc_analysis,
            'evidence': evidence,
            'llm_analysis': llm_verdict  # Include detailed LLM analysis
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# VERIFICATION STEP HELPERS (Extracted to reduce generate() branch count)
# =============================================================================

def _run_trust_layer(documents: List[str]) -> Tuple[bool, float, List[Dict]]:
    """Run hoax detection and credibility analysis."""
    hoax_detected = False
    avg_hoax_prob = 0.0
    doc_analysis = []
    
    try:
        from src.hoax_detection.credibility_report import CredibilityAnalyzer
        analyzer = CredibilityAnalyzer(
            hoax_model_path="models/hoax_indobert_lora",
            outlier_threshold_z=2.0
        )
        report = analyzer.analyze(documents)
        
        # Unload model to free VRAM
        analyzer.unload()
        del analyzer
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if report.documents:
            for doc_result in report.documents:
                doc_analysis.append({
                    'text': doc_result.text[:200] + '...',
                    'hoax_probability': doc_result.hoax_probability,
                    'hoax_label': doc_result.hoax_label,
                    'credibility_score': doc_result.credibility_score
                })
                if doc_result.hoax_label == 'HOAX':
                    hoax_detected = True
                avg_hoax_prob += doc_result.hoax_probability
            avg_hoax_prob /= len(report.documents)
    except Exception as e:
        print(f"Trust layer error: {e}")
    
    return hoax_detected, avg_hoax_prob, doc_analysis


def _run_web_search(article_title: Optional[str], documents: List[str]) -> Tuple[List[Dict], int]:
    """Search for corroborating sources."""
    evidence = []
    search_supports = 0
    
    try:
        from src.tools.enhanced_search import EnhancedSearcher
        from src.tools.source_credibility import SourceCredibilityAnalyzer
        
        searcher = EnhancedSearcher(max_results=15)
        search_query = article_title if article_title else generate_search_query(documents[0][:1000])
        results = searcher.search_and_fetch_sync(search_query, max_fetch=7)
        
        credibility_analyzer = SourceCredibilityAnalyzer()
        results_dicts = [{'url': r.url, 'title': r.title, 'snippet': r.snippet} for r in results]
        corroboration = credibility_analyzer.analyze_search_results(results_dicts)
        
        for r in results[:5]:
            is_trusted = False
            source_name = r.domain
            if corroboration:
                for src in corroboration.sources:
                    if src.url == r.url:
                        is_trusted = src.is_trusted
                        source_name = src.source_name
                        break
            evidence.append({
                'title': r.title,
                'url': r.url,
                'source': source_name,
                'is_trusted': is_trusted
            })
        
        search_supports = corroboration.num_trusted if corroboration else 0
    except Exception as e:
        print(f"Search error: {e}")
    
    return evidence, search_supports


def _run_knowledge_graph(documents: List[str]) -> Dict:
    """Build knowledge graph and return stats."""
    kg_stats = {'entities': 0, 'relations': 0}
    try:
        from src.models.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(name="verify_kg")
        kg.add_documents(documents, show_progress=False)
        kg_stats = kg.get_stats()
    except Exception as e:
        print(f"KG error: {e}")
    return kg_stats


def _run_llm_verdict(article_title: Optional[str], documents: List[str], 
                     hoax_detected: bool, avg_hoax_prob: float, evidence: List[Dict]) -> Optional[Dict]:
    """Get LLM verdict on the article."""
    try:
        from src.models.gemini_summarizer import GeminiSummarizer
        llm = GeminiSummarizer()
        return llm.verify_article(
            title=article_title or "Unknown",
            content=documents[0] if documents else "",
            hoax_probability=avg_hoax_prob if hoax_detected else None,
            search_results=evidence
        )
    except Exception as e:
        print(f"LLM error: {e}")
        return None


@app.route('/api/verify-stream', methods=['POST'])
def verify_stream():
    """
    SSE streaming verification endpoint.
    Streams progress events as the verification runs.
    """
    from flask import Response
    import json
    
    data = request.json
    input_type = data.get('type', 'text')
    content = data.get('content', '').strip()
    
    def generate():
        try:
            # Helper to send SSE event
            def send_event(step, status, message=""):
                event_data = json.dumps({'step': step, 'status': status, 'message': message})
                return f"data: {event_data}\n\n"
            
            if not content:
                yield send_event('error', 'error', 'No content provided')
                return
            
            # Step 1: FETCH
            yield send_event('fetch', 'active', 'Fetching article content...')
            
            documents = []
            article_title = None
            
            if input_type == 'url':
                fetched = fetch_url_content(content)
                if fetched:
                    documents = [fetched['text']]
                    article_title = fetched.get('title', '')
                else:
                    yield send_event('fetch', 'error', 'Could not fetch URL')
                    return
            else:
                documents = [p.strip() for p in content.split('\n\n') if p.strip()]
                if not documents:
                    documents = [content]
            
            yield send_event('fetch', 'done', f'Fetched {len(documents)} document(s)')
            
            # Step 2: TRUST LAYER
            yield send_event('trust', 'active', 'Running hoax detection...')
            hoax_detected, avg_hoax_prob, doc_analysis = _run_trust_layer(documents)
            yield send_event('trust', 'done', f'Hoax probability: {avg_hoax_prob:.1%}')
            
            # Step 3: WEB SEARCH
            yield send_event('search', 'active', 'Searching for corroboration...')
            evidence, search_supports = _run_web_search(article_title, documents)
            yield send_event('search', 'done', f'Found {len(evidence)} sources, {search_supports} trusted')
            
            # Step 4: KNOWLEDGE GRAPH
            yield send_event('kg', 'active', 'Building knowledge graph...')
            kg_stats = _run_knowledge_graph(documents)
            yield send_event('kg', 'done', f'{kg_stats.get("entities", 0)} entities found')
            
            # Step 5: LLM VERDICT
            yield send_event('verdict', 'active', 'Generating LLM verdict...')
            llm_verdict = _run_llm_verdict(article_title, documents, hoax_detected, avg_hoax_prob, evidence)
            
            # Final verdict determination
            verdict = "UNCERTAIN"
            summary = "Analysis complete."
            confidence = 0.5
            
            if llm_verdict:
                verdict = llm_verdict.get('verdict', 'UNCERTAIN')
                summary = llm_verdict.get('summary', 'Analysis complete.')
                confidence = llm_verdict.get('confidence', 0.5)
            elif hoax_detected:
                verdict = "NOT TRUSTABLE"
                summary = f"High hoax probability detected: {avg_hoax_prob:.1%}"
                confidence = avg_hoax_prob
            
            yield send_event('verdict', 'done', verdict)
            
            # Send final result
            final_result = {
                'step': 'complete',
                'status': 'done',
                'result': {
                    'verdict': verdict,
                    'summary': summary,
                    'confidence': confidence,
                    'documents': doc_analysis,
                    'evidence': evidence,
                    'llm_analysis': llm_verdict
                }
            }
            yield f"data: {json.dumps(final_result)}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'error', 'status': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        generate(), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
            'Connection': 'keep-alive'
        }
    )


def fetch_url_content(url: str) -> Optional[Dict]:
    """
    Fetch and extract clean article content from URL.
    Uses crawl4ai with PruningContentFilter for garbage-free extraction.
    Falls back to simple extraction if crawl4ai fails.
    """
    # Auto-expand Kompas.com pagination
    if 'kompas.com' in url and '?page=all' not in url:
        if '?' in url:
            url += '&page=all'
        else:
            url += '?page=all'
        print(f"[fetch_url] Kompas URL detected. Forcing single page: {url}")

    # Try crawl4ai first (better quality)
    try:
        from src.article_extractor import extract_article_sync
        result = extract_article_sync(url, timeout=30000)
        
        if result.success and result.clean_length > 100:
            return {
                'url': url,
                'text': result.content[:10000],  # Allow more text for full article analysis
                'title': result.title or 'Unknown'
            }
        # Fall through to fallback if crawl4ai returned empty content
    except Exception as e:
        print(f"[fetch_url] Crawl4AI error, using fallback: {e}")
    
    # Fallback: Simple HTML extraction
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Simple HTML text extraction
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ['script', 'style', 'nav', 'header', 'footer', 'aside']:
                    self.skip = True

            def handle_endtag(self, tag):
                if tag in ['script', 'style', 'nav', 'header', 'footer', 'aside']:
                    self.skip = False

            def handle_data(self, data):
                if not self.skip:
                    text = data.strip()
                    if text:
                        self.text.append(text)

        parser = TextExtractor()
        parser.feed(response.text)
        text = ' '.join(parser.text)

        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 100:
            return None

        return {
            'url': url,
            'text': text[:10000],  # Allow more text
            'title': extract_title(response.text)
        }

    except Exception as e:
        print(f"URL fetch error: {e}")
        return None


def extract_title(html: str) -> str:
    """Extract title from HTML."""
    match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    return match.group(1).strip() if match else 'Unknown'


def generate_search_query(text: str) -> str:
    """Generate a search query from text content."""
    # Extract key phrases (simple approach)
    words = text.split()[:20]
    # Remove common words
    stopwords = {'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'adalah', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    keywords = [w for w in words if w.lower() not in stopwords and len(w) > 3]
    return ' '.join(keywords[:10])


def generate_verdict(
    documents: List[str],
    doc_analysis: List[Dict],
    hoax_detected: bool,
    avg_hoax_prob: float,
    evidence: List[Dict],
    kg_confidence: float
) -> Tuple[str, str, float]:
    """Generate verification verdict."""

    # Calculate confidence
    confidence = 0.5

    # Hoax detection weight
    if hoax_detected:
        confidence = max(0.1, 1.0 - avg_hoax_prob)
        verdict = "FALSE"
        summary = f"This content has been flagged as potential misinformation with {avg_hoax_prob*100:.0f}% hoax probability. "
    elif avg_hoax_prob > 0.5:
        confidence = 0.4
        verdict = "UNCERTAIN"
        summary = f"The content shows mixed credibility signals. Hoax probability: {avg_hoax_prob*100:.0f}%. "
    else:
        confidence = min(0.9, 0.5 + kg_confidence * 0.3 + (1 - avg_hoax_prob) * 0.2)
        verdict = "TRUE"
        summary = "The content appears credible based on trust layer analysis. "

    # Add evidence context
    if evidence:
        summary += f"Found {len(evidence)} related sources for reference."

    # Trusted document count
    trusted = sum(1 for d in doc_analysis if d['status'] == 'valid')
    total = len(doc_analysis)

    if trusted < total:
        summary += f" {total - trusted} of {total} segments were filtered due to low credibility."

    return verdict, summary, confidence


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


# === TTS API Endpoints ===

@app.route('/api/tts/status')
def tts_status():
    """Check if Kokoro TTS is available."""
    global tts_voice
    return jsonify({
        'available': tts_voice is not None,
        'enabled': tts_enabled,
        'voice': tts_voice.persona if tts_voice else None
    })


@app.route('/api/tts/speak', methods=['POST'])
def tts_speak():
    """Speak text using Kokoro TTS."""
    global tts_voice, tts_enabled
    if not tts_voice or not tts_enabled:
        return jsonify({'success': False, 'error': 'TTS not available'})

    data = request.json
    text = data.get('text', '').strip()

    if text:
        tts_voice.speak(text)
        return jsonify({'success': True})

    return jsonify({'success': False, 'error': 'No text provided'})


@app.route('/api/tts/toggle', methods=['POST'])
def tts_toggle():
    """Toggle TTS on/off."""
    global tts_enabled
    data = request.json
    tts_enabled = data.get('enabled', True)
    return jsonify({'success': True, 'enabled': tts_enabled})


@app.route('/api/tts/voice', methods=['POST'])
def tts_voice_change():
    """Change TTS voice persona."""
    global tts_voice
    if not tts_voice:
        return jsonify({'success': False, 'error': 'TTS not available'})

    data = request.json
    voice = data.get('voice', 'friday')

    if tts_voice.set_persona(voice):
        return jsonify({'success': True, 'voice': voice})

    return jsonify({'success': False, 'error': 'Invalid voice'})


def run_server(host: str = None, port: int = 5000, debug: bool = False):
    """
    Run the Flask server.
    
    Args:
        host: Host to bind to. Defaults to BIND_HOST env var or '127.0.0.1' (localhost only).
              Set BIND_HOST=0.0.0.0 for Docker/LAN access.
        port: Port to bind to.
        debug: Enable debug mode.
    """
    import os
    if host is None:
        host = os.environ.get('BIND_HOST', '127.0.0.1')
    
    print(f"\n{'='*60}")
    print("TDSM News Verification Workspace")
    print(f"{'='*60}")

    # Initialize Kokoro TTS
    init_tts()

    print(f"\n→ Open http://localhost:{port} in your browser")
    print("\nFeatures:")
    print("  • URL verification (paste article links)")
    print("  • Text verification (paste content directly)")
    print("  • Clipboard reading")
    print("  • Kokoro TTS (local neural voice)")
    print("  • Bookmarklet for quick verification")
    print()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
