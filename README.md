# TDSM - Trust-Driven Summarization Model

**Indonesian Multi-Document Summarization with Knowledge Graph Verification**

A comprehensive NLP system for summarizing Indonesian news articles with built-in hoax detection, knowledge graph verification, and web search corroboration.

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd NLP-Project
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

# 2. Configure API keys (create .env file)
# GEMINI_API_KEY=your_key
# GROQ_API_KEY=your_key

# 3. Run web interface
python -m src.web_app
# Open http://localhost:5000
```

---

## ✨ Key Features

| Feature                       | Description                                                         |
| ----------------------------- | ------------------------------------------------------------------- |
| **Trust Layer**               | Filters unreliable sources using IndoBERT hoax detection (99.5% F1) |
| **Knowledge Graph**           | Extracts entities, relations, and temporal anchors from documents   |
| **Constrained Summarization** | Hybrid extractive-abstractive with KG grounding                     |
| **Web Corroboration**         | Searches external sources to verify article claims                  |
| **Multi-Model Support**       | TextRank, LexRank, Gemini LLM, and Hybrid modes                     |
| **Voice Assistant**           | Optional Kokoro TTS for voice-enabled interaction                   |

---

## 🎯 Three Ways to Run

### 1. Web Interface (Recommended)

```bash
python -m src.web_app
```

Interactive UI at http://localhost:5000 with URL fetching, live summarization, and trust analysis.

### 2. Command-Line

```bash
python -m src.main --mode full --model hybrid --credibility --verify
```

### 3. Voice Assistant

```bash
python -m src.assistant
```

Use hotkeys: `Ctrl+Alt+S` (Summarize), `Ctrl+Alt+H` (Hoax Check), `Esc` (Exit)

---

## 📋 Requirements

- **Python 3.11** (required for Kokoro TTS)
- **API Keys**: Gemini + Groq (see [GUIDE_BOOK.md](GUIDE_BOOK.md))
- **GPU**: NVIDIA recommended (~500MB VRAM)

---

## 📚 Documentation

| Document                                     | Purpose                                         |
| -------------------------------------------- | ----------------------------------------------- |
| [GUIDE_BOOK.md](GUIDE_BOOK.md)               | Detailed installation, usage, and CLI reference |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Complete file/folder explanation                |

---

## 🔬 Technical Stack

- **NLP**: spaCy, NLTK, Sastrawi (Indonesian stemmer)
- **ML**: PyTorch, Transformers, IndoBERT
- **LLM**: Groq (Llama 3.3), Gemini 2.0
- **Web**: Flask, Crawl4AI, DuckDuckGo Search
- **TTS**: Kokoro-82M (local, ultra-fast)

---

## 📧 Authors

- Felix Nathaniel

---

_TDSM v1.0 - Trust-Driven Summarization Model with Knowledge Graph Verification_
