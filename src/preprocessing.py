"""
Text Preprocessing for Indonesian News Summarization

Handles text cleaning, tokenization, and stemming for Indonesian text.
Accounts for Indonesian morphology (prefixes/suffixes like me-, ber-, -kan, -i).
"""

import re
from typing import List, Union

# Optional dependencies
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    StemmerFactory = None

try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    sent_tokenize = None
    word_tokenize = None
    nltk = None

# Ensure NLTK data is downloaded if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')


class TextPreprocessor:
    """
    Handles text preprocessing for Indonesian news summarization.
    Includes cleaning, tokenization, and stemming.
    """
    
    def __init__(self, use_stemmer: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_stemmer: Whether to apply Sastrawi stemming (computationally expensive).
        """
        self.use_stemmer = use_stemmer
        if self.use_stemmer:
            if SASTRAWI_AVAILABLE:
                factory = StemmerFactory()
                self.stemmer = factory.create_stemmer()
            else:
                print("Warning: Sastrawi not found. Stemming disabled.")
                self.use_stemmer = False
        
        # Comprehensive Indonesian stopwords
        self.stopwords = set([
            'yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'untuk', 'pada', 
            'adalah', 'sebagai', 'dengan', 'juga', 'akan', 'sudah', 'atau', 
            'ia', 'dia', 'mereka', 'kita', 'kami', 'anda', 'saya', 'tidak',
            'bisa', 'ada', 'oleh', 'sebuah', 'dalam', 'tersebut', 'dapat',
            'lebih', 'telah', 'hanya', 'karena', 'agar', 'seperti', 'saat',
            'bahwa', 'jika', 'maka', 'hal', 'sehingga', 'namun', 'tetapi',
            'lalu', 'kemudian', 'secara', 'hingga', 'antara', 'setelah',
            'sebelum', 'masih', 'belum', 'pun', 'begitu', 'lain', 'sama'
        ])

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: remove extra whitespace, normalize quotes, etc.
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove HTML tags if any
        text = re.sub(r'<.*?>', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        """
        text = self.clean_text(text)
        
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        else:
            # Simple fallback: split by .!? followed by whitespace
            # Lookbehind is used to keep the delimiter if possible, but re.split consumes it.
            # Let's use a simpler regex that works well enough for testing.
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def preprocess_sentence(self, sentence: str) -> str:
        """
        Preprocess a single sentence: lowercasing, removing punctuation/numbers,
        and stemming.
        """
        # Lowercase
        sentence = sentence.lower()
        
        # Remove punctuation and numbers (keep only letters and spaces)
        sentence = re.sub(r'[^a-z\s]', '', sentence)
        
        if self.use_stemmer:
            sentence = self.stemmer.stem(sentence)
            
        return sentence

    def preprocess_document(self, text: str) -> List[str]:
        """
        Full pipeline: clean -> sent_tokenize -> preprocess each sentence.
        Returns a list of preprocessed sentence strings.
        """
        sentences = self.tokenize_sentences(text)
        processed_sentences = [self.preprocess_sentence(s) for s in sentences]
        return processed_sentences
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove Indonesian stopwords from token list."""
        return [t for t in tokens if t.lower() not in self.stopwords]