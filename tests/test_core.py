"""
Unit Tests for TDSM News Verification System
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# TEST: Embeddings Module
# =============================================================================

class TestEmbeddings:
    """Tests for the embeddings module."""
    
    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        from src.models.embeddings import _get_cache_key
        
        key1 = _get_cache_key("test sentence", "model1")
        key2 = _get_cache_key("test sentence", "model1")
        key3 = _get_cache_key("different sentence", "model1")
        
        # Same input should produce same key
        assert key1 == key2
        # Different input should produce different key
        assert key1 != key3
        # Key should be 32 characters (SHA256 truncated)
        assert len(key1) == 32
    
    def test_embedding_models_exist(self):
        """Test that embedding model configs are defined."""
        from src.models.embeddings import EMBEDDING_MODELS, DEFAULT_MODEL
        
        assert len(EMBEDDING_MODELS) > 0
        assert DEFAULT_MODEL in EMBEDDING_MODELS


# =============================================================================
# TEST: Gemini Summarizer
# =============================================================================

class TestGeminiSummarizer:
    """Tests for GeminiSummarizer (mock API calls)."""
    
    def test_summarizer_initialization(self):
        """Test that summarizer can be initialized."""
        # Skip if no API key
        if not os.environ.get('GOOGLE_API_KEY') and not os.environ.get('GEMINI_API_KEY'):
            pytest.skip("No Gemini API key available")
        
        from src.models.gemini_summarizer import GeminiSummarizer
        summarizer = GeminiSummarizer()
        assert summarizer is not None


# =============================================================================
# TEST: Web Workspace
# =============================================================================

class TestWebWorkspace:
    """Tests for Flask web application."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.web_workspace import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_index_route(self, client):
        """Test that index page loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'TDSM' in response.data or b'News Verification' in response.data
    
    def test_verify_endpoint_exists(self, client):
        """Test that verify endpoint exists."""
        response = client.post('/api/verify', json={'type': 'text', 'content': ''})
        # Should return 400 for empty content, not 404
        assert response.status_code in [200, 400]
    
    def test_tts_status_endpoint(self, client):
        """Test TTS status endpoint."""
        response = client.get('/api/tts/status')
        assert response.status_code == 200


# =============================================================================
# TEST: Hoax Classifier
# =============================================================================

class TestHoaxClassifier:
    """Tests for hoax detection classifier."""
    
    def test_classifier_structure(self):
        """Test that classifier module structure is correct."""
        from src.hoax_detection import classifier
        assert hasattr(classifier, 'HoaxClassifier')


# =============================================================================
# TEST: Outlier Detector
# =============================================================================

class TestOutlierDetector:
    """Tests for outlier detection."""
    
    def test_outlier_detector_exists(self):
        """Test that OutlierDetector class exists."""
        from src.hoax_detection.outlier_detector import OutlierDetector
        assert OutlierDetector is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
