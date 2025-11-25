"""
Credibility Report Generator

Combines hoax classification and outlier detection into a unified
credibility analysis with detailed reporting.

Architecture: Parallel Processing
- Hoax classification runs independently (IndoBERT + LoRA)
- Outlier detection runs independently (TF-IDF/BERT + Cosine Similarity)
- Results are combined using weighted scoring

Output: Comprehensive credibility report for each document
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .classifier import HoaxClassifier, ClassificationResult
from .outlier_detector import OutlierDetector, OutlierAnalysis, OutlierResult


class CredibilityLevel(Enum):
    """Credibility assessment levels."""
    HIGH = "HIGH"           # Valid news, fits document cluster
    MEDIUM = "MEDIUM"       # Some concerns but not definitive
    LOW = "LOW"             # Likely hoax OR outlier
    CRITICAL = "CRITICAL"   # Both hoax AND outlier - highest risk


@dataclass
class DocumentCredibility:
    """Credibility assessment for a single document."""
    doc_index: int
    text: str
    
    # Hoax classification results
    hoax_label: str
    hoax_confidence: float
    hoax_probability: float
    
    # Outlier detection results
    is_outlier: bool
    similarity_to_centroid: float
    outlier_z_score: float
    
    # Combined assessment
    credibility_level: CredibilityLevel
    credibility_score: float  # 0.0 (not credible) to 1.0 (highly credible)
    flags: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "doc_index": self.doc_index,
            "text_preview": self.text[:150] + "..." if len(self.text) > 150 else self.text,
            "hoax_analysis": {
                "label": self.hoax_label,
                "confidence": round(self.hoax_confidence, 4),
                "hoax_probability": round(self.hoax_probability, 4)
            },
            "outlier_analysis": {
                "is_outlier": self.is_outlier,
                "similarity_to_centroid": round(self.similarity_to_centroid, 4),
                "z_score": round(self.outlier_z_score, 4)
            },
            "assessment": {
                "credibility_level": self.credibility_level.value,
                "credibility_score": round(self.credibility_score, 4),
                "flags": self.flags,
                "recommendation": self.recommendation
            }
        }


@dataclass
class CredibilityReport:
    """
    Complete credibility report for a document collection.
    
    This is the main output of the credibility analysis system.
    """
    timestamp: str
    total_documents: int
    
    # Individual assessments
    documents: List[DocumentCredibility]
    
    # Aggregate statistics
    high_credibility_count: int
    medium_credibility_count: int
    low_credibility_count: int
    critical_credibility_count: int
    
    # Filtered results
    documents_for_summarization: List[int]  # Indices of approved documents
    documents_excluded: List[int]           # Indices of excluded documents
    
    # Overall assessment
    collection_risk_level: str
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            "report_metadata": {
                "timestamp": self.timestamp,
                "total_documents": self.total_documents,
                "analyzer_version": "1.0.0"
            },
            "aggregate_statistics": {
                "high_credibility": self.high_credibility_count,
                "medium_credibility": self.medium_credibility_count,
                "low_credibility": self.low_credibility_count,
                "critical_credibility": self.critical_credibility_count,
                "documents_approved": len(self.documents_for_summarization),
                "documents_excluded": len(self.documents_excluded)
            },
            "collection_assessment": {
                "risk_level": self.collection_risk_level,
                "summary": self.summary
            },
            "approved_document_indices": self.documents_for_summarization,
            "excluded_document_indices": self.documents_excluded,
            "document_details": [doc.to_dict() for doc in self.documents]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, filepath: str):
        """Save report to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        print(f"[Report] Saved to {filepath}")
    
    def get_approved_texts(self, documents: List[str]) -> List[str]:
        """Get texts of approved documents."""
        return [documents[i] for i in self.documents_for_summarization]
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "=" * 60)
        print("📊 CREDIBILITY REPORT")
        print("=" * 60)
        print(f"Generated: {self.timestamp}")
        print(f"Total Documents: {self.total_documents}")
        print()
        print("📈 DISTRIBUTION:")
        print(f"  ✅ HIGH credibility:     {self.high_credibility_count}")
        print(f"  ⚠️  MEDIUM credibility:  {self.medium_credibility_count}")
        print(f"  🔶 LOW credibility:      {self.low_credibility_count}")
        print(f"  🚨 CRITICAL:             {self.critical_credibility_count}")
        print()
        print(f"📋 FILTERING RESULTS:")
        print(f"  Approved for summarization: {len(self.documents_for_summarization)}")
        print(f"  Excluded from summarization: {len(self.documents_excluded)}")
        print()
        print(f"⚡ COLLECTION RISK: {self.collection_risk_level}")
        print(f"📝 {self.summary}")
        print("=" * 60)


class CredibilityAnalyzer:
    """
    Main credibility analysis engine.
    
    Combines hoax classification and outlier detection in parallel,
    then produces a unified credibility report.
    
    Attributes:
        hoax_classifier: IndoBERT + LoRA hoax detector.
        outlier_detector: Cosine similarity outlier detector.
        hoax_weight: Weight for hoax score in combined calculation.
        outlier_weight: Weight for outlier score in combined calculation.
        credibility_threshold: Minimum score for document approval.
    """
    
    def __init__(
        self,
        hoax_model_path: str = "models/hoax_indobert_lora",
        outlier_threshold_z: float = 2.0,  # STRICT
        hoax_weight: float = 0.6,
        outlier_weight: float = 0.4,
        credibility_threshold: float = 0.5
    ):
        """
        Initialize the credibility analyzer.
        
        Args:
            hoax_model_path: Path to trained hoax classifier.
            outlier_threshold_z: Z-score threshold for outliers.
            hoax_weight: Weight for hoax classification (0-1).
            outlier_weight: Weight for outlier detection (0-1).
            credibility_threshold: Minimum score to approve document.
        """
        print("[CredibilityAnalyzer] Initializing...")
        
        # Validate weights
        if abs(hoax_weight + outlier_weight - 1.0) > 0.01:
            raise ValueError("hoax_weight + outlier_weight must equal 1.0")
        
        self.hoax_weight = hoax_weight
        self.outlier_weight = outlier_weight
        self.credibility_threshold = credibility_threshold
        
        # Initialize components
        print("[CredibilityAnalyzer] Loading hoax classifier...")
        self.hoax_classifier = HoaxClassifier(model_path=hoax_model_path)
        
        print("[CredibilityAnalyzer] Initializing outlier detector...")
        self.outlier_detector = OutlierDetector(threshold_z=outlier_threshold_z)
        
        print("[CredibilityAnalyzer] Ready.")
    
    def _calculate_credibility_score(
        self,
        hoax_prob: float,
        is_outlier: bool,
        similarity: float
    ) -> float:
        """
        Calculate combined credibility score.
        
        Score formula:
        - Hoax component: (1 - hoax_probability) * hoax_weight
        - Outlier component: similarity_to_centroid * outlier_weight
          (with penalty if is_outlier)
        
        Returns:
            Float between 0.0 (not credible) and 1.0 (highly credible).
        """
        # Hoax component (higher hoax_prob = lower credibility)
        hoax_score = (1.0 - hoax_prob) * self.hoax_weight
        
        # Outlier component
        if is_outlier:
            # Heavy penalty for outliers
            outlier_score = similarity * 0.3 * self.outlier_weight
        else:
            outlier_score = similarity * self.outlier_weight
        
        return hoax_score + outlier_score
    
    def _determine_credibility_level(
        self,
        hoax_label: str,
        is_outlier: bool,
        credibility_score: float
    ) -> CredibilityLevel:
        """Determine credibility level based on combined analysis."""
        is_hoax = hoax_label == "HOAX"
        
        if is_hoax and is_outlier:
            return CredibilityLevel.CRITICAL
        elif is_hoax or is_outlier:
            return CredibilityLevel.LOW
        elif credibility_score >= 0.7:
            return CredibilityLevel.HIGH
        else:
            return CredibilityLevel.MEDIUM
    
    def _generate_flags(
        self,
        hoax_label: str,
        hoax_prob: float,
        is_outlier: bool,
        z_score: float
    ) -> List[str]:
        """Generate warning flags for a document."""
        flags = []
        
        if hoax_label == "HOAX":
            if hoax_prob >= 0.9:
                flags.append("HIGH_CONFIDENCE_HOAX")
            elif hoax_prob >= 0.7:
                flags.append("LIKELY_HOAX")
            else:
                flags.append("POSSIBLE_HOAX")
        
        if is_outlier:
            if z_score < -3.0:
                flags.append("EXTREME_OUTLIER")
            else:
                flags.append("TOPIC_OUTLIER")
        
        if hoax_prob >= 0.5 and hoax_prob < 0.7:
            flags.append("UNCERTAIN_CLASSIFICATION")
        
        return flags
    
    def _generate_recommendation(
        self,
        level: CredibilityLevel,
        flags: List[str]
    ) -> str:
        """Generate action recommendation."""
        if level == CredibilityLevel.CRITICAL:
            return "EXCLUDE: High-risk content. Do not include in summarization."
        elif level == CredibilityLevel.LOW:
            if "TOPIC_OUTLIER" in flags:
                return "EXCLUDE: Off-topic content. Does not match document cluster."
            else:
                return "EXCLUDE: Potential misinformation detected."
        elif level == CredibilityLevel.MEDIUM:
            return "INCLUDE WITH CAUTION: Verify facts before publishing."
        else:
            return "INCLUDE: Content appears credible."
    
    def analyze(self, documents: List[str]) -> CredibilityReport:
        """
        Perform full credibility analysis on a document collection.
        
        This is the main entry point. It runs hoax classification and
        outlier detection in parallel, then combines the results.
        
        Args:
            documents: List of document texts to analyze.
            
        Returns:
            CredibilityReport with detailed analysis.
        """
        n_docs = len(documents)
        print(f"\n[CredibilityAnalyzer] Analyzing {n_docs} documents...")
        
        # === PARALLEL ANALYSIS ===
        
        # 1. Hoax Classification
        print("[CredibilityAnalyzer] Running hoax classification...")
        hoax_results = self.hoax_classifier.predict_batch(documents)
        
        # 2. Outlier Detection
        print("[CredibilityAnalyzer] Running outlier detection...")
        outlier_analysis = self.outlier_detector.detect_outliers(documents)
        
        # === COMBINE RESULTS ===
        print("[CredibilityAnalyzer] Combining results...")
        
        doc_credibilities = []
        approved_indices = []
        excluded_indices = []
        
        level_counts = {
            CredibilityLevel.HIGH: 0,
            CredibilityLevel.MEDIUM: 0,
            CredibilityLevel.LOW: 0,
            CredibilityLevel.CRITICAL: 0
        }
        
        for i, (hoax_res, outlier_res) in enumerate(zip(hoax_results, outlier_analysis.results)):
            # Calculate combined score
            cred_score = self._calculate_credibility_score(
                hoax_prob=hoax_res.hoax_probability,
                is_outlier=outlier_res.is_outlier,
                similarity=outlier_res.similarity_to_centroid
            )
            
            # Determine level
            level = self._determine_credibility_level(
                hoax_label=hoax_res.label,
                is_outlier=outlier_res.is_outlier,
                credibility_score=cred_score
            )
            
            # Generate flags and recommendation
            flags = self._generate_flags(
                hoax_label=hoax_res.label,
                hoax_prob=hoax_res.hoax_probability,
                is_outlier=outlier_res.is_outlier,
                z_score=outlier_res.z_score
            )
            
            recommendation = self._generate_recommendation(level, flags)
            
            # Create document credibility record
            doc_cred = DocumentCredibility(
                doc_index=i,
                text=documents[i],
                hoax_label=hoax_res.label,
                hoax_confidence=hoax_res.confidence,
                hoax_probability=hoax_res.hoax_probability,
                is_outlier=outlier_res.is_outlier,
                similarity_to_centroid=outlier_res.similarity_to_centroid,
                outlier_z_score=outlier_res.z_score,
                credibility_level=level,
                credibility_score=cred_score,
                flags=flags,
                recommendation=recommendation
            )
            
            doc_credibilities.append(doc_cred)
            level_counts[level] += 1
            
            # Filtering decision
            if level in [CredibilityLevel.HIGH, CredibilityLevel.MEDIUM]:
                approved_indices.append(i)
            else:
                excluded_indices.append(i)
        
        # === COLLECTION-LEVEL ASSESSMENT ===
        
        # Determine overall risk
        critical_ratio = level_counts[CredibilityLevel.CRITICAL] / n_docs if n_docs > 0 else 0
        low_ratio = level_counts[CredibilityLevel.LOW] / n_docs if n_docs > 0 else 0
        
        if critical_ratio > 0.3:
            collection_risk = "CRITICAL"
            summary = f"High-risk collection: {level_counts[CredibilityLevel.CRITICAL]} critical documents detected. Manual review strongly recommended."
        elif critical_ratio > 0.1 or low_ratio > 0.4:
            collection_risk = "HIGH"
            summary = f"Elevated risk: {len(excluded_indices)} documents excluded. Verify remaining sources."
        elif low_ratio > 0.2:
            collection_risk = "MODERATE"
            summary = f"Some concerns detected: {len(excluded_indices)} documents excluded. Proceed with caution."
        else:
            collection_risk = "LOW"
            summary = f"Collection appears credible. {len(approved_indices)} of {n_docs} documents approved."
        
        # === BUILD REPORT ===
        report = CredibilityReport(
            timestamp=datetime.now().isoformat(),
            total_documents=n_docs,
            documents=doc_credibilities,
            high_credibility_count=level_counts[CredibilityLevel.HIGH],
            medium_credibility_count=level_counts[CredibilityLevel.MEDIUM],
            low_credibility_count=level_counts[CredibilityLevel.LOW],
            critical_credibility_count=level_counts[CredibilityLevel.CRITICAL],
            documents_for_summarization=approved_indices,
            documents_excluded=excluded_indices,
            collection_risk_level=collection_risk,
            summary=summary
        )
        
        print("[CredibilityAnalyzer] Analysis complete.")
        return report
    
    def filter_documents(
        self, 
        documents: List[str]
    ) -> Tuple[List[str], CredibilityReport]:
        """
        Filter documents and return both filtered list and report.
        
        Convenience method that returns approved documents directly.
        
        Args:
            documents: List of document texts.
            
        Returns:
            Tuple of (approved_documents, credibility_report).
        """
        report = self.analyze(documents)
        approved_docs = report.get_approved_texts(documents)
        
        return approved_docs, report


# Factory function
def create_analyzer(
    hoax_model_path: str = "models/hoax_indobert_lora",
    strict: bool = True
) -> CredibilityAnalyzer:
    """
    Create a credibility analyzer with standard settings.
    
    Args:
        hoax_model_path: Path to trained hoax model.
        strict: Use strict (2σ) outlier threshold.
        
    Returns:
        Configured CredibilityAnalyzer.
    """
    return CredibilityAnalyzer(
        hoax_model_path=hoax_model_path,
        outlier_threshold_z=2.0 if strict else 1.5
    )


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Credibility Analyzer Demo")
    print("=" * 60)
    
    # Sample documents
    documents = [
        "Pemerintah Indonesia mengumumkan kebijakan baru tentang vaksinasi COVID-19 untuk masyarakat luas.",
        "VIRAL! Vaksin COVID-19 mengandung microchip 5G untuk pelacakan! Bagikan sebelum dihapus pemerintah!",
        "Kementerian Kesehatan melaporkan peningkatan signifikan cakupan vaksinasi di seluruh provinsi.",
        "AWAS! Air minum kemasan merek terkenal mengandung racun berbahaya! Dokter terkenal konfirmasi!",
        "Resep masakan rendang padang yang enak dan mudah dibuat di rumah untuk keluarga.",  # Off-topic
        "WHO memuji keberhasilan Indonesia dalam kampanye vaksinasi massal melawan COVID-19."
    ]
    
    analyzer = CredibilityAnalyzer()
    report = analyzer.analyze(documents)
    
    # Print summary
    report.print_summary()
    
    # Print individual results
    print("\n📄 DOCUMENT DETAILS:")
    print("-" * 60)
    
    for doc in report.documents:
        level_emoji = {
            CredibilityLevel.HIGH: "✅",
            CredibilityLevel.MEDIUM: "⚠️",
            CredibilityLevel.LOW: "🔶",
            CredibilityLevel.CRITICAL: "🚨"
        }
        
        print(f"\n[Doc {doc.doc_index}] {level_emoji[doc.credibility_level]} {doc.credibility_level.value}")
        print(f"  Text: {doc.text[:60]}...")
        print(f"  Hoax: {doc.hoax_label} ({doc.hoax_probability:.2%})")
        print(f"  Outlier: {'Yes' if doc.is_outlier else 'No'} (sim: {doc.similarity_to_centroid:.3f})")
        print(f"  Score: {doc.credibility_score:.3f}")
        print(f"  Flags: {doc.flags}")
        print(f"  → {doc.recommendation}")
    
    # Export JSON
    print("\n" + "=" * 60)
    print("Exporting report to JSON...")
    print(report.to_json())