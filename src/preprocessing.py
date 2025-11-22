import re
from typing import List, Union
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        
        # Basic Indonesian stopwords (can be expanded)
        self.stopwords = set([
            'yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'untuk', 'pada', 
            'adalah', 'sebagai', 'dengan', 'juga', 'akan', 'sudah', 'atau', 
            'ia', 'dia', 'mereka', 'kita', 'kami', 'anda', 'saya'
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
        
        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        """
        text = self.clean_text(text)
        return sent_tokenize(text)

    def preprocess_sentence(self, sentence: str) -> str:
        """
        Preprocess a single sentence: lowercasing, removing punctuation/numbers (optional),
        and stemming.
        """
        # Lowercase
        sentence = sentence.lower()
        
        # Remove punctuation and numbers (keep only letters and spaces)
        # Note: For some tasks we might want to keep numbers.
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
