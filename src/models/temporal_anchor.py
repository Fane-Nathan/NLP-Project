"""
Temporal Anchoring for Indonesian News Text

Handles the critical "when" dimension of knowledge graphs:
1. Extracting temporal expressions (dates, times, durations)
2. Normalizing to standard formats (ISO 8601)
3. Resolving relative expressions ("kemarin", "minggu lalu")
4. Building event timelines

This is THE key module for preventing temporal hallucinations in TMDS.
"""

import re
from enum import Enum
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta


class TemporalType(Enum):
    """Types of temporal expressions."""
    POINT = "POINT"           # Specific datetime: 2024-01-15
    INTERVAL = "INTERVAL"     # Date range: Jan-Mar 2024
    DURATION = "DURATION"     # Time span: 3 months
    RELATIVE = "RELATIVE"     # Relative to now: yesterday, last week
    FUZZY = "FUZZY"           # Approximate: sekitar 2020, beberapa bulan
    RECURRING = "RECURRING"   # Periodic: setiap tahun, per bulan
    UNKNOWN = "UNKNOWN"


class TemporalGranularity(Enum):
    """Granularity of temporal expression."""
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    QUARTER = "QUARTER"
    YEAR = "YEAR"
    DECADE = "DECADE"
    UNKNOWN = "UNKNOWN"


@dataclass
class TemporalExpression:
    """
    Represents a temporal expression with normalization.
    
    Attributes:
        text: Original text span
        temporal_type: Type classification
        granularity: Level of precision
        normalized_start: ISO format start datetime
        normalized_end: ISO format end datetime (same as start for points)
        confidence: Extraction confidence
        is_approximate: Whether expression is fuzzy/approximate
        relative_anchor: For relative expressions, what it's relative to
        source_doc: Document index
        metadata: Additional data
    """
    text: str
    temporal_type: TemporalType
    granularity: TemporalGranularity
    normalized_start: str
    normalized_end: str
    confidence: float = 1.0
    is_approximate: bool = False
    relative_anchor: Optional[str] = None
    source_doc: int = 0
    start_pos: int = 0
    end_pos: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "type": self.temporal_type.value,
            "granularity": self.granularity.value,
            "normalized": {
                "start": self.normalized_start,
                "end": self.normalized_end
            },
            "confidence": round(self.confidence, 4),
            "is_approximate": self.is_approximate,
            "source_doc": self.source_doc
        }
    
    def overlaps(self, other: 'TemporalExpression') -> bool:
        """Check if two temporal expressions overlap."""
        try:
            self_start = datetime.fromisoformat(self.normalized_start)
            self_end = datetime.fromisoformat(self.normalized_end)
            other_start = datetime.fromisoformat(other.normalized_start)
            other_end = datetime.fromisoformat(other.normalized_end)
            
            return not (self_end < other_start or other_end < self_start)
        except:
            return False
    
    def is_before(self, other: 'TemporalExpression') -> bool:
        """Check if this expression is before another."""
        try:
            self_end = datetime.fromisoformat(self.normalized_end)
            other_start = datetime.fromisoformat(other.normalized_start)
            return self_end < other_start
        except:
            return False


@dataclass
class Timeline:
    """
    A chronologically ordered sequence of events.
    
    Attributes:
        events: List of (temporal_expression, event_description, doc_idx)
        start_date: Earliest date in timeline
        end_date: Latest date in timeline
    """
    events: List[Tuple[TemporalExpression, str, int]] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    def add_event(
        self,
        temporal: TemporalExpression,
        description: str,
        doc_idx: int = 0
    ):
        """Add event and maintain chronological order."""
        self.events.append((temporal, description, doc_idx))
        self._sort_events()
        self._update_bounds()
    
    def _sort_events(self):
        """Sort events by start date."""
        self.events.sort(key=lambda x: x[0].normalized_start)
    
    def _update_bounds(self):
        """Update start and end dates."""
        if self.events:
            self.start_date = self.events[0][0].normalized_start
            self.end_date = self.events[-1][0].normalized_end
    
    def to_dict(self) -> Dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "num_events": len(self.events),
            "events": [
                {
                    "date": e[0].normalized_start,
                    "description": e[1][:100],
                    "doc_idx": e[2]
                }
                for e in self.events
            ]
        }


class TemporalAnchor:
    """
    Extracts and normalizes temporal expressions from Indonesian text.
    
    Handles Indonesian-specific temporal patterns:
    - Full dates: 15 Januari 2024
    - Month-year: Januari 2024
    - Relative: kemarin, minggu lalu, bulan depan
    - Fuzzy: sekitar 2020, beberapa bulan lalu
    - Fiscal/quarters: kuartal III 2024, Q3 2024
    """
    
    # Indonesian month names
    MONTHS = {
        'januari': 1, 'februari': 2, 'maret': 3, 'april': 4,
        'mei': 5, 'juni': 6, 'juli': 7, 'agustus': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'desember': 12,
        # Short forms
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'mei': 5, 'jun': 6,
        'jul': 7, 'agu': 8, 'agst': 8, 'sep': 9, 'sept': 9,
        'okt': 10, 'nov': 11, 'des': 12
    }
    
    # Indonesian day names (for context, rarely used for anchoring)
    DAYS = {
        'senin': 0, 'selasa': 1, 'rabu': 2, 'kamis': 3,
        'jumat': 4, 'sabtu': 5, 'minggu': 6
    }
    
    # Relative time expressions
    RELATIVE_PATTERNS = {
        # Days
        'hari ini': (0, 'day'),
        'kemarin': (-1, 'day'),
        'kemarin dulu': (-2, 'day'),
        'besok': (1, 'day'),
        'lusa': (2, 'day'),
        
        # Weeks
        'minggu ini': (0, 'week'),
        'minggu lalu': (-1, 'week'),
        'minggu kemarin': (-1, 'week'),
        'minggu depan': (1, 'week'),
        'pekan ini': (0, 'week'),
        'pekan lalu': (-1, 'week'),
        
        # Months
        'bulan ini': (0, 'month'),
        'bulan lalu': (-1, 'month'),
        'bulan kemarin': (-1, 'month'),
        'bulan depan': (1, 'month'),
        
        # Years
        'tahun ini': (0, 'year'),
        'tahun lalu': (-1, 'year'),
        'tahun kemarin': (-1, 'year'),
        'tahun depan': (1, 'year'),
    }
    
    # Duration patterns
    DURATION_UNITS = {
        'detik': 'second',
        'menit': 'minute',
        'jam': 'hour',
        'hari': 'day',
        'minggu': 'week',
        'bulan': 'month',
        'tahun': 'year',
    }
    
    # Fuzzy modifiers
    FUZZY_MODIFIERS = [
        'sekitar', 'kira-kira', 'kurang lebih', 'hampir',
        'lebih dari', 'kurang dari', 'mendekati', 'awal',
        'pertengahan', 'akhir', 'beberapa'
    ]
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize temporal anchor.
        
        Args:
            reference_date: Reference date for relative expressions.
                           Defaults to current date.
        """
        self.reference_date = reference_date or datetime.now()
        
        # Compile patterns
        self._compile_patterns()
        
        print(f"[TemporalAnchor] Initialized")
        print(f"  Reference date: {self.reference_date.strftime('%Y-%m-%d')}")
    
    def _compile_patterns(self):
        """Compile regex patterns for temporal extraction."""
        month_names = '|'.join(self.MONTHS.keys())
        
        # Full date: 15 Januari 2024
        self.full_date_pattern = re.compile(
            rf'(\d{{1,2}})\s+({month_names})\s+(\d{{4}})',
            re.IGNORECASE
        )
        
        # Month year: Januari 2024
        self.month_year_pattern = re.compile(
            rf'({month_names})\s+(\d{{4}})',
            re.IGNORECASE
        )
        
        # Year only: tahun 2024 or just 2024
        self.year_pattern = re.compile(
            r'(?:tahun\s+)?(\d{4})\b',
            re.IGNORECASE
        )
        
        # Numeric date: 15/01/2024 or 15-01-2024
        self.numeric_date_pattern = re.compile(
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
        )
        
        # ISO date: 2024-01-15
        self.iso_date_pattern = re.compile(
            r'(\d{4})-(\d{2})-(\d{2})'
        )
        
        # Quarter: Q1 2024, kuartal I 2024
        self.quarter_pattern = re.compile(
            r'(?:Q|kuartal)\s*([IViv1-4])\s*(\d{4})',
            re.IGNORECASE
        )
        
        # Duration: 3 bulan, 2 tahun
        duration_units = '|'.join(self.DURATION_UNITS.keys())
        self.duration_pattern = re.compile(
            rf'(\d+)\s+({duration_units})',
            re.IGNORECASE
        )
        
        # Relative expressions
        relative_patterns = '|'.join(
            re.escape(p) for p in self.RELATIVE_PATTERNS.keys()
        )
        self.relative_pattern = re.compile(
            rf'({relative_patterns})',
            re.IGNORECASE
        )
        
        # Fuzzy prefix
        fuzzy_mods = '|'.join(re.escape(m) for m in self.FUZZY_MODIFIERS)
        self.fuzzy_prefix_pattern = re.compile(
            rf'({fuzzy_mods})\s+',
            re.IGNORECASE
        )
    
    def _normalize_to_iso(
        self,
        year: int,
        month: int = 1,
        day: int = 1,
        hour: int = 0,
        minute: int = 0,
        second: int = 0
    ) -> str:
        """Convert to ISO 8601 format."""
        try:
            dt = datetime(year, month, day, hour, minute, second)
            return dt.isoformat()
        except ValueError:
            # Handle invalid dates (e.g., Feb 30)
            return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
    
    def _get_end_of_period(
        self,
        start: datetime,
        granularity: TemporalGranularity
    ) -> datetime:
        """Get end datetime for a period based on granularity."""
        if granularity == TemporalGranularity.DAY:
            return start.replace(hour=23, minute=59, second=59)
        elif granularity == TemporalGranularity.WEEK:
            return start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        elif granularity == TemporalGranularity.MONTH:
            next_month = start.replace(day=28) + timedelta(days=4)
            return next_month.replace(day=1) - timedelta(seconds=1)
        elif granularity == TemporalGranularity.QUARTER:
            quarter_end_month = ((start.month - 1) // 3 + 1) * 3
            end = start.replace(month=quarter_end_month, day=28) + timedelta(days=4)
            return end.replace(day=1) - timedelta(seconds=1)
        elif granularity == TemporalGranularity.YEAR:
            return start.replace(month=12, day=31, hour=23, minute=59, second=59)
        else:
            return start
    
    def _resolve_relative(
        self,
        expression: str,
        reference: datetime
    ) -> Tuple[datetime, datetime, TemporalGranularity]:
        """Resolve relative temporal expression."""
        expr_lower = expression.lower().strip()
        
        if expr_lower in self.RELATIVE_PATTERNS:
            offset, unit = self.RELATIVE_PATTERNS[expr_lower]
            
            if unit == 'day':
                start = reference + timedelta(days=offset)
                granularity = TemporalGranularity.DAY
            elif unit == 'week':
                # Go to start of current week, then apply offset
                start_of_week = reference - timedelta(days=reference.weekday())
                start = start_of_week + timedelta(weeks=offset)
                granularity = TemporalGranularity.WEEK
            elif unit == 'month':
                start = reference.replace(day=1) + relativedelta(months=offset)
                granularity = TemporalGranularity.MONTH
            elif unit == 'year':
                start = reference.replace(month=1, day=1) + relativedelta(years=offset)
                granularity = TemporalGranularity.YEAR
            else:
                start = reference
                granularity = TemporalGranularity.UNKNOWN
            
            end = self._get_end_of_period(start, granularity)
            return start, end, granularity
        
        return reference, reference, TemporalGranularity.UNKNOWN
    
    def _parse_quarter(self, quarter_str: str, year: int) -> Tuple[int, int]:
        """Parse quarter to start month."""
        quarter_map = {
            'i': 1, 'ii': 4, 'iii': 7, 'iv': 10,
            '1': 1, '2': 4, '3': 7, '4': 10
        }
        quarter = quarter_map.get(quarter_str.lower(), 1)
        return quarter, quarter + 2
    
    def extract(
        self,
        text: str,
        doc_idx: int = 0
    ) -> List[TemporalExpression]:
        """
        Extract all temporal expressions from text.
        
        Args:
            text: Input text.
            doc_idx: Document index.
            
        Returns:
            List of temporal expressions.
        """
        if not text:
            return []
        
        expressions = []
        
        # Check for fuzzy prefixes
        fuzzy_regions = set()
        for match in self.fuzzy_prefix_pattern.finditer(text):
            # Mark region after fuzzy modifier as approximate
            fuzzy_regions.add(match.end())
        
        # Full dates: 15 Januari 2024
        for match in self.full_date_pattern.finditer(text):
            day = int(match.group(1))
            month = self.MONTHS.get(match.group(2).lower(), 1)
            year = int(match.group(3))
            
            is_fuzzy = any(
                abs(match.start() - fr) < 20 for fr in fuzzy_regions
            )
            
            try:
                start_dt = datetime(year, month, day)
                expressions.append(TemporalExpression(
                    text=match.group(),
                    temporal_type=TemporalType.POINT,
                    granularity=TemporalGranularity.DAY,
                    normalized_start=start_dt.isoformat(),
                    normalized_end=start_dt.replace(
                        hour=23, minute=59, second=59
                    ).isoformat(),
                    confidence=0.95 if not is_fuzzy else 0.7,
                    is_approximate=is_fuzzy,
                    source_doc=doc_idx,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
            except ValueError:
                continue
        
        # Month-year: Januari 2024
        for match in self.month_year_pattern.finditer(text):
            month = self.MONTHS.get(match.group(1).lower(), 1)
            year = int(match.group(2))
            
            # Check if this is part of a full date (already extracted)
            if any(
                e.start_pos <= match.start() < e.end_pos
                for e in expressions
            ):
                continue
            
            is_fuzzy = any(
                abs(match.start() - fr) < 20 for fr in fuzzy_regions
            )
            
            start_dt = datetime(year, month, 1)
            end_dt = self._get_end_of_period(start_dt, TemporalGranularity.MONTH)
            
            expressions.append(TemporalExpression(
                text=match.group(),
                temporal_type=TemporalType.INTERVAL,
                granularity=TemporalGranularity.MONTH,
                normalized_start=start_dt.isoformat(),
                normalized_end=end_dt.isoformat(),
                confidence=0.85 if not is_fuzzy else 0.6,
                is_approximate=is_fuzzy,
                source_doc=doc_idx,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Quarters: Q1 2024
        for match in self.quarter_pattern.finditer(text):
            quarter_str = match.group(1)
            year = int(match.group(2))
            
            start_month, end_month = self._parse_quarter(quarter_str, year)
            start_dt = datetime(year, start_month, 1)
            end_dt = datetime(year, end_month, 1)
            end_dt = self._get_end_of_period(end_dt, TemporalGranularity.MONTH)
            
            expressions.append(TemporalExpression(
                text=match.group(),
                temporal_type=TemporalType.INTERVAL,
                granularity=TemporalGranularity.QUARTER,
                normalized_start=start_dt.isoformat(),
                normalized_end=end_dt.isoformat(),
                confidence=0.9,
                source_doc=doc_idx,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Year only: 2024
        for match in self.year_pattern.finditer(text):
            year = int(match.group(1))
            
            # Skip if already covered by more specific pattern
            if any(
                e.start_pos <= match.start() < e.end_pos
                for e in expressions
            ):
                continue
            
            # Validate year range (1900-2100)
            if not (1900 <= year <= 2100):
                continue
            
            is_fuzzy = any(
                abs(match.start() - fr) < 20 for fr in fuzzy_regions
            )
            
            start_dt = datetime(year, 1, 1)
            end_dt = datetime(year, 12, 31, 23, 59, 59)
            
            expressions.append(TemporalExpression(
                text=match.group(),
                temporal_type=TemporalType.INTERVAL,
                granularity=TemporalGranularity.YEAR,
                normalized_start=start_dt.isoformat(),
                normalized_end=end_dt.isoformat(),
                confidence=0.7 if not is_fuzzy else 0.5,
                is_approximate=is_fuzzy,
                source_doc=doc_idx,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Relative expressions
        for match in self.relative_pattern.finditer(text):
            start_dt, end_dt, granularity = self._resolve_relative(
                match.group(), self.reference_date
            )
            
            expressions.append(TemporalExpression(
                text=match.group(),
                temporal_type=TemporalType.RELATIVE,
                granularity=granularity,
                normalized_start=start_dt.isoformat(),
                normalized_end=end_dt.isoformat(),
                confidence=0.8,
                relative_anchor=self.reference_date.isoformat(),
                source_doc=doc_idx,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Duration expressions
        for match in self.duration_pattern.finditer(text):
            amount = int(match.group(1))
            unit = self.DURATION_UNITS.get(match.group(2).lower(), 'day')
            
            expressions.append(TemporalExpression(
                text=match.group(),
                temporal_type=TemporalType.DURATION,
                granularity=TemporalGranularity[unit.upper()],
                normalized_start="",  # Duration doesn't have fixed start
                normalized_end="",
                confidence=0.85,
                source_doc=doc_idx,
                start_pos=match.start(),
                end_pos=match.end(),
                metadata={"amount": amount, "unit": unit}
            ))
        
        # Sort by position
        expressions.sort(key=lambda e: e.start_pos)
        
        return expressions
    
    def build_timeline(
        self,
        documents: List[str],
        event_descriptions: Optional[List[str]] = None
    ) -> Timeline:
        """
        Build a timeline from multiple documents.
        
        Args:
            documents: List of document texts.
            event_descriptions: Optional descriptions for each doc.
            
        Returns:
            Timeline object with chronologically ordered events.
        """
        timeline = Timeline()
        
        for doc_idx, doc in enumerate(documents):
            temporals = self.extract(doc, doc_idx)
            
            description = (
                event_descriptions[doc_idx] if event_descriptions
                else doc[:100] + "..."
            )
            
            # Add most specific temporal expression as anchor
            if temporals:
                # Prefer points over intervals over others
                best_temporal = min(
                    temporals,
                    key=lambda t: (
                        0 if t.temporal_type == TemporalType.POINT else
                        1 if t.temporal_type == TemporalType.INTERVAL else
                        2
                    )
                )
                timeline.add_event(best_temporal, description, doc_idx)
        
        return timeline
    
    def set_reference_date(self, date: datetime):
        """Update reference date for relative expressions."""
        self.reference_date = date
        print(f"[TemporalAnchor] Reference updated: {date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Temporal Anchor Demo")
    print("=" * 60)
    
    anchor = TemporalAnchor()
    
    text = """
    Presiden mengumumkan kebijakan pada 15 Januari 2024. Sebelumnya, 
    pada kuartal III 2023, pemerintah telah melakukan kajian selama 3 bulan.
    Minggu lalu, Menteri memberikan penjelasan tambahan. Sekitar tahun 2020,
    kebijakan serupa pernah diusulkan. Implementasi direncanakan pada Februari 2024.
    """
    
    expressions = anchor.extract(text)
    
    print(f"\nFound {len(expressions)} temporal expressions:\n")
    
    for expr in expressions:
        print(f"[{expr.temporal_type.value}] {expr.text}")
        print(f"  Granularity: {expr.granularity.value}")
        print(f"  Normalized: {expr.normalized_start[:10] if expr.normalized_start else 'N/A'}")
        print(f"  Confidence: {expr.confidence:.2f}")
        print(f"  Approximate: {expr.is_approximate}")
        print()
    
    # Build timeline
    docs = [
        "Kebijakan diumumkan pada 15 Januari 2024.",
        "Kajian dilakukan pada September 2023.",
        "Evaluasi awal pada Maret 2023.",
        "Implementasi dimulai Februari 2024."
    ]
    
    timeline = anchor.build_timeline(docs)
    
    print("\n" + "=" * 60)
    print("Timeline:")
    print("=" * 60)
    
    for event in timeline.events:
        temporal, desc, doc_idx = event
        print(f"  {temporal.normalized_start[:10]} | Doc {doc_idx}: {desc[:50]}")
