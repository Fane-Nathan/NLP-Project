# 📚 TDSM Software Guide Book

## Trust-Driven Summarization Model - User & Developer Manual

---

## 🔑 Prerequisites

### Required API Keys

This software requires **two API keys** to function:

| API Key          | Purpose                                       | Get It From                                            |
| ---------------- | --------------------------------------------- | ------------------------------------------------------ |
| `GEMINI_API_KEY` | LLM summarization, knowledge graph extraction | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY`   | Fast vision-based screen analysis             | [Groq Console](https://console.groq.com/keys)          |

### System Requirements

- **Python**: 3.11 (required for Kokoro TTS compatibility)
- **GPU**: NVIDIA GPU recommended (runs on GTX 1650, ~500MB VRAM)
- **OS**: Windows (tested), Linux, macOS

---

## 🛠️ Installation

### Step 1: Clone & Navigate

```bash
git clone <repository-url>
cd NLP-Project
```

### Step 2: Create Virtual Environment

```bash
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3.11 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Playwright (for web scraping)

```bash
playwright install chromium
```

### Step 5: Configure API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Running the Software

### Option 1: Web Interface (Recommended)

Run the web playground for an interactive experience:

```bash
python -m src.web_app
```

Then open your browser to: **http://localhost:5000**

**Features:**

- Paste Indonesian news articles or fetch from URL
- Select summarization model (Hybrid, TextRank, LexRank, Gemini)
- Enable/disable Trust Layer and KG Verification
- View trust analysis and verification results

---

### Option 2: Command-Line Interface (CLI)

Run the full pipeline from terminal:

```bash
# Full pipeline with demo data (one-button mode)
python -m src.main

# Full pipeline with custom file
python -m src.main --input_file data/docs.json

# Summarize only (no verification)
python -m src.main --mode summarize --model textrank --num_sentences 5

# Build Knowledge Graph only
python -m src.main --mode kg --input_file data/docs.json --output kg_output.json

# Verify an existing summary
python -m src.main --mode verify --input_file data/docs.json --summary "Your summary text here"

# Run credibility analysis only
python -m src.main --mode credibility --input_file data/docs.json

# Evaluate model performance on XL-Sum
python -m src.main --mode evaluate --model textrank --num_samples 100 --indo_rouge
```

#### CLI Arguments Reference

| Argument            | Options                                                        | Description                                     |
| ------------------- | -------------------------------------------------------------- | ----------------------------------------------- |
| `--mode`            | `full`, `summarize`, `kg`, `verify`, `credibility`, `evaluate` | Operation mode                                  |
| `--model`           | `textrank`, `lexrank`, `gemini`, `hybrid`                      | Summarization model                             |
| `--input_file`      | path                                                           | Path to input documents (JSON/JSONL/TXT)        |
| `--input_text`      | text                                                           | Direct input text                               |
| `--num_sentences`   | int                                                            | Number of sentences for extractive (default: 5) |
| `--credibility`     | flag                                                           | Enable Trust Layer filtering                    |
| `--verify`          | flag                                                           | Enable KG verification                          |
| `--output`          | path                                                           | Output file path                                |
| `--use_mmr`         | flag                                                           | Enable MMR for redundancy reduction             |
| `--indo_rouge`      | flag                                                           | Use Indonesian-specific ROUGE evaluation        |
| `--similarity_mode` | `tfidf`, `embedding`, `hybrid`                                 | Similarity computation mode                     |

---

### Option 3: Voice Assistant (Terminal)

Run the voice-enabled screen assistant:

```bash
python -m src.assistant
```

#### Hotkeys

| Shortcut     | Action                             |
| ------------ | ---------------------------------- |
| `Ctrl+Alt+S` | Summarize current screen           |
| `Ctrl+Alt+D` | Describe screen in detail          |
| `Ctrl+Alt+V` | Toggle voice persona (Friday/Echo) |
| `Ctrl+Alt+H` | Check for hoax/misinformation      |
| `Ctrl+Alt+G` | Add to Knowledge Graph             |
| `Ctrl+Alt+U` | Run unified pipeline               |
| `Ctrl+Alt+W` | Web research mode                  |
| `Esc`        | Exit assistant                     |

---

## 📂 Project Structure

```
NLP-Project/
├── src/
│   ├── web_app.py          # Web interface (Flask)
│   ├── main.py             # CLI entry point
│   ├── assistant.py        # Voice assistant
│   ├── models/             # Summarization models
│   │   ├── textrank.py
│   │   ├── lexrank.py
│   │   ├── gemini_summarizer.py
│   │   ├── knowledge_graph.py
│   │   └── constrained_summarizer.py
│   ├── hoax_detection/     # Trust Layer components
│   └── tools/              # Web search, extraction
├── models/                 # Pre-trained model weights
│   └── hoax_indobert_lora/ # Hoax detection model
├── data/                   # Sample datasets
├── requirements.txt        # Python dependencies
└── .env                    # API keys (create this)
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue                      | Solution                                         |
| -------------------------- | ------------------------------------------------ |
| `GEMINI_API_KEY not found` | Ensure `.env` file exists with valid key         |
| `torch not found`          | Run `pip install torch torchaudio`               |
| `playwright not installed` | Run `playwright install chromium`                |
| `ModuleNotFoundError`      | Ensure virtual environment is activated          |
| `CUDA out of memory`       | Reduce `EMBEDDING_BATCH_SIZE` in `src/config.py` |

### Verify Environment

```bash
python verify_env.py
```

---

## 📝 Input File Formats

The CLI accepts multiple input formats:

**JSON:**

```json
{
  "documents": ["Article 1 text...", "Article 2 text..."]
}
```

**JSONL:**

```jsonl
{"text": "Article 1 text..."}
{"text": "Article 2 text..."}
```

**TXT:** Separate documents with blank lines.

---

## 🎯 Quick Start Commands

```bash
# 1. Activate environment
.\venv\Scripts\activate

# 2. Run web interface
python -m src.web_app

# 3. Or run CLI for quick demo
python -m src.main
```

---

_TDSM v1.0 - Trust-Driven Summarization Model with Knowledge Graph Verification_
