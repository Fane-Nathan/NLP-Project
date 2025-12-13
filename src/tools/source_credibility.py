"""
Source Credibility Analyzer

Analyzes search results to determine:
1. Source reliability (known trusted vs unknown sources)
2. Corroboration (multiple trusted sources reporting the same claim)
3. Contradiction detection (conflicting information across sources)

Used to enhance hoax detection by cross-referencing with external sources.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import re


# Trusted Indonesian news sources (major outlets with editorial standards)
TRUSTED_SOURCES = {
    # Major national news
    'kompas.com': {'tier': 1, 'name': 'Kompas', 'type': 'mainstream'},
    'detik.com': {'tier': 1, 'name': 'Detik', 'type': 'mainstream'},
    'tempo.co': {'tier': 1, 'name': 'Tempo', 'type': 'mainstream'},
    'liputan6.com': {'tier': 1, 'name': 'Liputan 6', 'type': 'mainstream'},
    'cnnindonesia.com': {'tier': 1, 'name': 'CNN Indonesia', 'type': 'mainstream'},
    'cnbcindonesia.com': {'tier': 1, 'name': 'CNBC Indonesia', 'type': 'business'},
    'republika.co.id': {'tier': 1, 'name': 'Republika', 'type': 'mainstream'},
    'antaranews.com': {'tier': 1, 'name': 'Antara', 'type': 'wire'},
    'bisnis.com': {'tier': 1, 'name': 'Bisnis Indonesia', 'type': 'business'},
    'kontan.co.id': {'tier': 1, 'name': 'Kontan', 'type': 'business'},
    
    # Fact-checking organizations
    'cekfakta.com': {'tier': 1, 'name': 'Cek Fakta', 'type': 'factcheck'},
    'turnbackhoax.id': {'tier': 1, 'name': 'Turn Back Hoax', 'type': 'factcheck'},
    'mafindo.or.id': {'tier': 1, 'name': 'MAFINDO', 'type': 'factcheck'},
    
    # Government/official sources
    'kemkes.go.id': {'tier': 1, 'name': 'Kemenkes', 'type': 'government'},
    'covid19.go.id': {'tier': 1, 'name': 'COVID-19 Satgas', 'type': 'government'},
    'kominfo.go.id': {'tier': 1, 'name': 'Kominfo', 'type': 'government'},
    
    # International trusted sources
    'bbc.com': {'tier': 1, 'name': 'BBC', 'type': 'international'},
    'reuters.com': {'tier': 1, 'name': 'Reuters', 'type': 'wire'},
    'apnews.com': {'tier': 1, 'name': 'AP News', 'type': 'wire'},
    'aljazeera.com': {'tier': 2, 'name': 'Al Jazeera', 'type': 'international'},
    
    # Secondary trusted (tier 2)
    'suara.com': {'tier': 2, 'name': 'Suara', 'type': 'mainstream'},
    'merdeka.com': {'tier': 2, 'name': 'Merdeka', 'type': 'mainstream'},
    'okezone.com': {'tier': 2, 'name': 'Okezone', 'type': 'mainstream'},
    'tribunnews.com': {'tier': 2, 'name': 'Tribun', 'type': 'mainstream'},
    'jpnn.com': {'tier': 2, 'name': 'JPNN', 'type': 'mainstream'},
    'sindonews.com': {'tier': 2, 'name': 'Sindo News', 'type': 'mainstream'},
}

# Known unreliable/suspicious patterns
SUSPICIOUS_PATTERNS = [
    r'blogspot\.com',
    r'wordpress\.com',
    r'\.tk$',
    r'\.ml$',
    r'\.ga$',
    r'bit\.ly',
    r'tinyurl\.com',
    r'whatsapp',
    r'telegram',
    r'facebook\.com/story',
]


@dataclass
class SourceAnalysis:
    """Analysis of a single source."""
    url: str
    domain: str
    title: str
    is_trusted: bool
    trust_tier: int  # 0=unknown, 1=highly trusted, 2=moderately trusted
    source_name: str
    source_type: str
    is_suspicious: bool
    supports_claim: Optional[bool] = None  # True=supports, False=contradicts, None=neutral


@dataclass
class CorroborationResult:
    """Result of cross-referencing multiple sources."""
    num_sources: int
    num_trusted: int
    num_tier1: int
    num_tier2: int
    num_suspicious: int
    
    # Corroboration signals
    corroboration_score: float  # 0-1, how well the claim is corroborated
    has_factcheck: bool  # Source from a fact-checking org
    has_wire_service: bool  # Source from Reuters/AP/Antara
    has_government: bool  # Source from government site
    
    # Detailed analysis
    sources: List[SourceAnalysis]
    summary: str
    recommendation: str  # "TRUSTABLE", "UNCERTAIN", "NOT_TRUSTABLE"


class SourceCredibilityAnalyzer:
    """
    Analyzer for source credibility in search results.
    
    Used to determine if search results support or contradict
    the claim being verified.
    """
    
    def __init__(self):
        self.trusted_sources = TRUSTED_SOURCES
        self.suspicious_patterns = [re.compile(p, re.I) for p in SUSPICIOUS_PATTERNS]
        print("[SourceCredibilityAnalyzer] Initialized")
        print(f"  Trusted sources: {len(self.trusted_sources)}")
    
    def _extract_domain(self, url: str) -> str:
        """Extract base domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ""
    
    def _is_suspicious(self, url: str) -> bool:
        """Check if URL matches suspicious patterns."""
        for pattern in self.suspicious_patterns:
            if pattern.search(url):
                return True
        return False
    
    def analyze_source(self, url: str, title: str = "") -> SourceAnalysis:
        """Analyze a single source URL."""
        domain = self._extract_domain(url)
        
        # Check if trusted
        source_info = None
        for trusted_domain, info in self.trusted_sources.items():
            if trusted_domain in domain:
                source_info = info
                break
        
        is_trusted = source_info is not None
        trust_tier = source_info['tier'] if source_info else 0
        source_name = source_info['name'] if source_info else domain
        source_type = source_info['type'] if source_info else 'unknown'
        
        # Check for suspicious patterns
        is_suspicious = self._is_suspicious(url)
        
        return SourceAnalysis(
            url=url,
            domain=domain,
            title=title,
            is_trusted=is_trusted,
            trust_tier=trust_tier,
            source_name=source_name,
            source_type=source_type,
            is_suspicious=is_suspicious
        )
    
    def analyze_search_results(
        self, 
        results: List[Dict],
        original_title: str = "",
        original_content: str = ""
    ) -> CorroborationResult:
        """
        Analyze search results for corroboration.
        
        Args:
            results: List of search result dicts with 'url', 'title', 'snippet'.
            original_title: Title of the original article.
            original_content: Content of the original article.
            
        Returns:
            CorroborationResult with analysis.
        """
        if not results:
            return CorroborationResult(
                num_sources=0,
                num_trusted=0,
                num_tier1=0,
                num_tier2=0,
                num_suspicious=0,
                corroboration_score=0.0,
                has_factcheck=False,
                has_wire_service=False,
                has_government=False,
                sources=[],
                summary="No search results to analyze.",
                recommendation="UNCERTAIN"
            )
        
        # Analyze each source
        sources = []
        for r in results:
            url = r.get('url', '')
            title = r.get('title', '')
            
            analysis = self.analyze_source(url, title)
            sources.append(analysis)
        
        # Count statistics
        num_trusted = sum(1 for s in sources if s.is_trusted)
        num_tier1 = sum(1 for s in sources if s.trust_tier == 1)
        num_tier2 = sum(1 for s in sources if s.trust_tier == 2)
        num_suspicious = sum(1 for s in sources if s.is_suspicious)
        
        # Check for special source types
        has_factcheck = any(s.source_type == 'factcheck' for s in sources)
        has_wire_service = any(s.source_type == 'wire' for s in sources)
        has_government = any(s.source_type == 'government' for s in sources)
        
        # Calculate corroboration score
        # Higher if: more trusted sources, tier 1 sources, wire services, fact-checks
        corroboration_score = 0.0
        
        if num_trusted > 0:
            # Base score from trusted sources
            corroboration_score += min(0.4, num_trusted * 0.1)
            
            # Bonus for tier 1 sources
            corroboration_score += min(0.3, num_tier1 * 0.1)
            
            # Bonus for special sources
            if has_factcheck:
                corroboration_score += 0.2
            if has_wire_service:
                corroboration_score += 0.1
            if has_government:
                corroboration_score += 0.1
        
        # Penalty for suspicious sources
        if num_suspicious > 0:
            corroboration_score -= num_suspicious * 0.1
        
        # Clamp to 0-1
        corroboration_score = max(0.0, min(1.0, corroboration_score))
        
        # Generate summary
        summary_parts = []
        if num_trusted > 0:
            summary_parts.append(f"{num_trusted} trusted source(s) found")
            if num_tier1 > 0:
                summary_parts.append(f"including {num_tier1} major outlet(s)")
        else:
            summary_parts.append("No trusted sources found")
        
        if has_factcheck:
            summary_parts.append("fact-check organization referenced")
        if has_wire_service:
            summary_parts.append("wire service (Reuters/AP/Antara) referenced")
        if num_suspicious > 0:
            summary_parts.append(f"{num_suspicious} suspicious source(s) detected")
        
        summary = "; ".join(summary_parts) + "."
        
        # Determine recommendation
        if has_factcheck or (num_tier1 >= 2 and corroboration_score >= 0.5):
            recommendation = "TRUSTABLE"
        elif num_trusted == 0 or (num_suspicious > num_trusted) or corroboration_score < 0.2:
            recommendation = "NOT_TRUSTABLE"
        else:
            recommendation = "UNCERTAIN"
        
        # Debug output
        print(f"[SourceCredibilityAnalyzer] Analysis complete:")
        print(f"  Sources: {len(sources)} total, {num_trusted} trusted, {num_suspicious} suspicious")
        print(f"  Corroboration: {corroboration_score:.1%}")
        print(f"  Recommendation: {recommendation}")
        
        return CorroborationResult(
            num_sources=len(sources),
            num_trusted=num_trusted,
            num_tier1=num_tier1,
            num_tier2=num_tier2,
            num_suspicious=num_suspicious,
            corroboration_score=corroboration_score,
            has_factcheck=has_factcheck,
            has_wire_service=has_wire_service,
            has_government=has_government,
            sources=sources,
            summary=summary,
            recommendation=recommendation
        )


# Factory function
def create_analyzer() -> SourceCredibilityAnalyzer:
    """Create a source credibility analyzer."""
    return SourceCredibilityAnalyzer()


if __name__ == "__main__":
    # Demo
    analyzer = SourceCredibilityAnalyzer()
    
    # Sample search results
    results = [
        {'url': 'https://www.kompas.com/news/article123', 'title': 'Breaking: Major Event'},
        {'url': 'https://www.detik.com/news/456', 'title': 'Event Coverage'},
        {'url': 'https://www.reuters.com/world/event', 'title': 'Reuters Report'},
        {'url': 'https://randomsite.blogspot.com/post', 'title': 'SHOCKING NEWS!!!'},
        {'url': 'https://turnbackhoax.id/article/789', 'title': 'Fact Check: Event'},
    ]
    
    result = analyzer.analyze_search_results(results)
    
    print(f"\nSummary: {result.summary}")
    print(f"Recommendation: {result.recommendation}")
    print(f"\nSources:")
    for s in result.sources:
        trust = "✓ TRUSTED" if s.is_trusted else ("⚠ SUSPICIOUS" if s.is_suspicious else "? UNKNOWN")
        print(f"  {s.source_name}: {trust}")
