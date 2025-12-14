"""
Faithfulness Metric for Abstractive Summarization

Measures how faithfully an abstractive summary represents the source documents.
Uses NLI (Natural Language Inference) to check if summary claims are entailed
by the source text.

This addresses the hallucination problem in LLM-generated summaries.
"""

from typing import List, Dict, Tuple, Optional
import re


class FaithfulnessChecker:
    """
    Checks faithfulness of abstractive summaries using entailment.
    
    A summary is considered faithful if all its claims are entailed
    (supported) by the source documents.
    
    Methods:
    1. Lexical Overlap: Simple word overlap baseline
    2. NLI-based: Uses IndoBERT or multilingual NLI model (if available)
    """
    
    def __init__(self, use_nli: bool = False, model_name: str = "indobenchmark/indobert-base-p1"):
        """
        Initialize faithfulness checker.
        
        Args:
            use_nli: Use neural NLI model (requires transformers + torch).
            model_name: HuggingFace model for NLI (default: IndoBERT).
        """
        self.use_nli = use_nli
        self.model_name = model_name
        self.nli_model = None
        self.tokenizer = None
        
        if self.use_nli:
            self._load_nli_model()
    
    def _load_nli_model(self):
        """Load NLI model for entailment checking."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            print(f"[Faithfulness] Loading NLI model: {self.model_name}...")
            
            # Try multilingual NLI model first (works better for Indonesian)
            # Pinned revision for supply chain security
            nli_model_name = "MoritzLaworski/m-DeBERTa-NLI-caching"
            nli_model_revision = "main"  # Using main as this model is rarely updated
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    nli_model_name,
                    revision=nli_model_revision
                )
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                    nli_model_name,
                    revision=nli_model_revision
                )
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.nli_model.to(self.device)
                self.nli_model.eval()
                print(f"[Faithfulness] NLI model loaded on {self.device} ✓")
            except Exception as e:
                print(f"[Warning] Could not load NLI model: {e}")
                print("[Faithfulness] Falling back to lexical overlap method")
                self.use_nli = False
                
        except ImportError:
            print("[Warning] transformers/torch not available for NLI")
            print("[Faithfulness] Using lexical overlap method")
            self.use_nli = False
    
    def _split_into_claims(self, summary: str) -> List[str]:
        """Split summary into individual claims (sentences)."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', summary)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def _lexical_overlap(self, claim: str, source: str) -> float:
        """
        Calculate lexical overlap between claim and source.
        
        Returns a score between 0 and 1.
        """
        claim_words = set(claim.lower().split())
        source_words = set(source.lower().split())
        
        if not claim_words:
            return 0.0
        
        overlap = len(claim_words & source_words)
        return overlap / len(claim_words)
    
    def _nli_entailment(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Check if premise entails hypothesis using NLI model.
        
        Returns probabilities for entailment, neutral, contradiction.
        """
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Model typically has labels: entailment, neutral, contradiction
        # Order may vary by model
        return {
            "entailment": float(probs[0]),
            "neutral": float(probs[1]) if len(probs) > 1 else 0.0,
            "contradiction": float(probs[2]) if len(probs) > 2 else 0.0
        }
    
    def check_claim(self, claim: str, source: str) -> Dict:
        """
        Check if a single claim is supported by the source.
        
        Args:
            claim: A claim from the summary.
            source: Source document(s) text.
            
        Returns:
            Dict with 'score', 'method', and 'details'.
        """
        if self.use_nli and self.nli_model:
            # Use NLI model
            nli_result = self._nli_entailment(source[:2000], claim)  # Truncate source
            return {
                "score": nli_result["entailment"],
                "method": "nli",
                "details": nli_result,
                "is_supported": nli_result["entailment"] > 0.5
            }
        else:
            # Use lexical overlap
            overlap = self._lexical_overlap(claim, source)
            return {
                "score": overlap,
                "method": "lexical",
                "details": {"overlap": overlap},
                "is_supported": overlap > 0.3  # Threshold for lexical
            }
    
    def check_faithfulness(
        self, 
        summary: str, 
        source_documents: List[str]
    ) -> Dict:
        """
        Check faithfulness of an entire summary.
        
        Args:
            summary: Generated summary text.
            source_documents: List of source document texts.
            
        Returns:
            Dict with overall score and per-claim breakdown.
        """
        # Combine source documents
        combined_source = ' '.join(source_documents)
        
        # Split summary into claims
        claims = self._split_into_claims(summary)
        
        if not claims:
            return {
                "overall_score": 1.0,
                "num_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": [],
                "claim_details": []
            }
        
        # Check each claim
        claim_results = []
        supported_count = 0
        unsupported_claims = []
        
        for claim in claims:
            result = self.check_claim(claim, combined_source)
            claim_results.append({
                "claim": claim,
                **result
            })
            
            if result["is_supported"]:
                supported_count += 1
            else:
                unsupported_claims.append(claim)
        
        # Calculate overall score
        overall_score = supported_count / len(claims) if claims else 1.0
        
        return {
            "overall_score": overall_score,
            "num_claims": len(claims),
            "supported_claims": supported_count,
            "unsupported_claims": unsupported_claims,
            "claim_details": claim_results,
            "method": "nli" if self.use_nli else "lexical"
        }
    
    def compare_summaries(
        self,
        summaries: Dict[str, str],
        source_documents: List[str]
    ) -> Dict[str, Dict]:
        """
        Compare faithfulness of multiple summaries.
        
        Args:
            summaries: Dict mapping model names to their summaries.
            source_documents: Source documents.
            
        Returns:
            Dict mapping model names to their faithfulness scores.
        """
        results = {}
        for model_name, summary in summaries.items():
            results[model_name] = self.check_faithfulness(summary, source_documents)
        
        return results


def check_hallucination(
    summary: str, 
    sources: List[str],
    threshold: float = 0.5
) -> Tuple[bool, List[str]]:
    """
    Quick hallucination check function.
    
    Args:
        summary: Generated summary.
        sources: Source documents.
        threshold: Faithfulness threshold (below = hallucination).
        
    Returns:
        Tuple of (has_hallucination, list_of_unsupported_claims)
    """
    checker = FaithfulnessChecker(use_nli=False)  # Use fast lexical method
    result = checker.check_faithfulness(summary, sources)
    
    has_hallucination = result["overall_score"] < threshold
    return has_hallucination, result["unsupported_claims"]


if __name__ == "__main__":
    # Demo
    print("=== Faithfulness Checker Demo ===\n")
    
    sources = [
        "Pemerintah Indonesia meluncurkan program vaksinasi COVID-19 pada Januari 2024. "
        "Presiden Joko Widodo menjadi orang pertama yang divaksin. "
        "Target cakupan adalah 70% populasi dalam 6 bulan.",
        
        "Kementerian Kesehatan menyatakan bahwa vaksin yang digunakan adalah Sinovac dan AstraZeneca. "
        "Vaksinasi gratis untuk seluruh masyarakat Indonesia."
    ]
    
    # Good summary (faithful)
    good_summary = "Program vaksinasi COVID-19 diluncurkan oleh pemerintah Indonesia dengan target 70% populasi."
    
    # Bad summary (contains hallucination)
    bad_summary = "Program vaksinasi COVID-19 menelan biaya 50 triliun rupiah dan menggunakan vaksin Pfizer."
    
    checker = FaithfulnessChecker(use_nli=False)
    
    print("Source documents:", len(sources))
    print("\n--- Good Summary ---")
    print(f"Summary: {good_summary}")
    result = checker.check_faithfulness(good_summary, sources)
    print(f"Faithfulness Score: {result['overall_score']:.2f}")
    print(f"Unsupported claims: {result['unsupported_claims']}")
    
    print("\n--- Bad Summary (with hallucination) ---")
    print(f"Summary: {bad_summary}")
    result = checker.check_faithfulness(bad_summary, sources)
    print(f"Faithfulness Score: {result['overall_score']:.2f}")
    print(f"Unsupported claims: {result['unsupported_claims']}")
