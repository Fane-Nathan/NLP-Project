# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **dual-purpose NLP project** combining two distinct systems:

1. **JARVIS Voice Assistant** - A local AI assistant with screen analysis and ultra-fast voice interaction using Kokoro TTS
2. **Indonesian Multi-Document Summarization** - NLP research system for synthesizing Indonesian news articles

## Python Environment

**Critical**: This project requires **Python 3.11** for Kokoro TTS compatibility. Do not upgrade to Python 3.12+.

```bash
# Create environment
py -3.11 -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Applications

### JARVIS Voice Assistant
```bash
python src/assistant.py
```

**Hotkeys:**
- `Ctrl+Alt+S` - Summarize Screen (Vision + OCR)
- `Ctrl+Alt+D` - Describe Screen (Vision only)
- `Ctrl+Alt+V` - Toggle Voice (Friday ↔ Jarvis)
- `Esc` - Exit

### Indonesian Summarization System
```bash
# Summarize with automatic cluster loading
python src/main.py --mode summarize --model textrank
python src/main.py --mode summarize --model lexrank
python src/main.py --mode summarize --model groq

# Summarize manual input
python src/main.py --mode summarize --model groq --input_text "path/to/file.txt"

# Evaluation mode (not yet implemented)
python src/main.py --mode evaluate --model textrank
```
### Hoax Detection & Credibility Analysis
```bash
# Train hoax classifier (first time only)
python -m src.hoax_detection.train_lora --data_path data/hoax_dataset.csv --epochs 3

# Run credibility analysis
python -m src.main --mode credibility --input_text "Your text" --hoax_model models/hoax_indobert_lora

# Summarize with credibility filtering
python -m src.main --mode summarize --model textrank --input_file data/docs.json --credibility
```

**Architecture:**
- IndoBERT + LoRA (99.5% F1-Score)
- Outlier Detection (2σ strict threshold)
- Combined scoring (60% hoax, 40% outlier)

### JARVIS Voice Assistant Stack

```
src/assistant.py         # Main entry point, keyboard hotkey handling
├── src/screen_capture.py    # MSS-based screen capture
├── src/ocr_engine.py         # Tesseract OCR wrapper
├── src/monitor.py            # Proactive window monitoring
├── src/voice_kokoro.py       # Kokoro TTS wrapper with voice personas
└── src/models/llm_groq.py    # Groq API (Vision + Text summarization)
```

**Flow:**
1. User triggers hotkey (Ctrl+Alt+S/D)
2. `assistant.py` → captures screen via `ScreenCapturer`
3. Converts to base64 → sends to `GroqSummarizer.summarize_image()` (Vision API)
4. If Vision fails → falls back to `OCREngine.extract_text()` → `summarize_text()`
5. Result cleaned (markdown removal) → spoken via `JarvisVoice.speak()`

### Indonesian Summarization Stack

```
src/main.py               # CLI entry point
├── src/data_loader.py        # XL-Sum dataset loader (BBC Indonesia)
├── src/preprocessing.py      # Indonesian text preprocessing (Sastrawi stemmer)
├── src/evaluation.py         # ROUGE metrics evaluation
└── src/models/
    ├── textrank.py           # Graph-based extractive (placeholder)
    ├── lexrank.py            # Eigenvector-based extractive (placeholder)
    └── llm_groq.py           # Groq LLM abstractive (shared with JARVIS)
```

**Flow:**
1. `NewsDataLoader` loads XL-Sum Indonesian dataset, simulates multi-doc clusters
2. `TextPreprocessor` cleans text, applies Sastrawi stemming for Indonesian morphology
3. Model (`TextRank`/`LexRank`/`Groq`) generates summary
4. Output displayed to console

## Key Technical Constraints

### Indonesian NLP
- **Morphology**: Indonesian has complex affixation (me-, ber-, -kan, -i). The `TextPreprocessor` uses **Sastrawi** stemmer to handle this.
- **Stopwords**: Custom Indonesian stopword list in `preprocessing.py:31-35`.
- **Dataset**: Uses `csebuetnlp/xlsum` (Indonesian split) from Hugging Face with `trust_remote_code=True`.
- **TextRank/LexRank**: Currently placeholder implementations - only return first N sentences.

### JARVIS Voice System
- **TTS**: Kokoro-82M model (~0.27x RTF, 4x faster than real-time).
- **Voices**: `af_heart` (Friday - warm female), `am_adam` (Jarvis - authoritative male).
- **Voice Configuration**: Edit `src/voice_kokoro.py` for volume/speed adjustments.
- **Vision Model**: Groq's `meta-llama/llama-4-scout-17b-16e-instruct` for multimodal.
- **Text Model**: Groq's `llama-3.3-70b-versatile` for text-only summarization.

### GPU Requirements
- NVIDIA GPU recommended (tested on GTX 1650, ~500MB VRAM).
- PyTorch with CUDA 12.1 support required (see `requirements.txt`).

## Environment Variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

**Note**: `.env` is gitignored. The `GroqSummarizer` will raise an error if the API key is missing.

## Data & Caching

- **Dataset Cache**: XL-Sum downloads to `./data_cache/` on first run.
- **Hugging Face Cache**: Model checkpoints cached in `./checkpoints/hf_cache/`.
- **Data Cache Directory**: Git-ignored to avoid committing large files.

## Important Code Patterns

### Indonesian Text Preprocessing
Always use the `TextPreprocessor` when working with Indonesian text:
```python
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(use_stemmer=True)
sentences = preprocessor.tokenize_sentences(text)
processed = preprocessor.preprocess_document(text)
```

### Multi-Document Cluster Handling
The `NewsDataLoader.get_random_cluster()` returns a **sorted timeline**:
```python
loader = NewsDataLoader()
cluster = loader.get_random_cluster(cluster_size=4)
# Each item has: id, source, date, text, gold_summary, url
# Sorted chronologically by date
```

### Groq API Dual Mode
The `GroqSummarizer` supports both Vision and Text:
```python
summarizer = GroqSummarizer()

# Vision mode (for JARVIS)
summary = summarizer.summarize_image(image_base64)

# Text mode (for Indonesian summarization)
summary = summarizer.summarize_text(text)
```

### Voice Output (JARVIS)
Always clean text before TTS to remove markdown:
```python
def clean_text_for_speech(text):
    text = re.sub(r'[*_#`]', '', text)  # Remove markdown
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links
    return text.strip()

voice.speak(clean_text_for_speech(result))
```

## Development Workflow

1. **Testing JARVIS**: Run `src/assistant.py` and test hotkeys on different screen content.
2. **Testing Summarization**: Use `src/main.py` with different models and observe output quality.
3. **Implementing TextRank/LexRank**: Current implementations in `src/models/` are placeholders - need graph construction and ranking logic.
4. **Evaluation**: The `Evaluator` class is ready but evaluation pipeline in `main.py` is incomplete.

## Code Style (from .agent/rules/code-style-guide.md)

- **Type Hints**: Strictly use (`List`, `Dict`, `Tuple` from `typing`)
- **Docstrings**: Google-style for all functions/classes
- **Modularity**: Separate preprocessing, models, and execution
- **Error Handling**: Robust checks for empty strings, encoding errors (common in scraped news)

## Common Issues

1. **NLTK punkt not found**: The `TextPreprocessor.__init__()` auto-downloads it, but manual download: `python -m nltk.downloader punkt`
2. **Groq API errors**: Check `.env` file and API key validity
3. **Kokoro model download**: ~350MB on first run, downloads to HuggingFace cache
4. **Trust Remote Code**: XL-Sum dataset requires `trust_remote_code=True` parameter
5. **GPU not detected**: Run `check_gpu.py` to verify PyTorch CUDA setup

## Testing

Currently no automated test suite. The `tests/` directory exists but is empty.

## Project State

**Completed:**
- JARVIS assistant with Vision + OCR + TTS integration
- Data loading pipeline for Indonesian news
- Preprocessing pipeline with Sastrawi stemming
- Groq API integration for both Vision and Text modes
- ROUGE evaluation framework

**TODO (Implicit from code):**
- Complete TextRank graph construction and ranking algorithm
- Complete LexRank eigenvector centrality implementation
- Implement full evaluation pipeline in `main.py`
- Add mT5/IndoBERT transformer-based models (mentioned in code-style-guide.md but not implemented)
- Create test suite
