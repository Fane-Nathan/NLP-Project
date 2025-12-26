# 📂 Project Structure

Complete explanation of every file and folder in the TDSM workspace.

---

## 📁 Root Directory

| File                   | Purpose                                                     |
| ---------------------- | ----------------------------------------------------------- |
| `README.md`            | Project overview and quick start                            |
| `GUIDE_BOOK.md`        | Detailed user & developer manual                            |
| `PROJECT_STRUCTURE.md` | This file - workspace documentation                         |
| `requirements.txt`     | Python dependencies                                         |
| `pyproject.toml`       | Project metadata and tool configs                           |
| `.env`                 | API keys (GEMINI_API_KEY, GROQ_API_KEY) - **not committed** |
| `.gitignore`           | Files excluded from version control                         |
| `Dockerfile`           | Container configuration for deployment                      |
| `verify_env.py`        | Script to verify environment setup                          |

---

## 📁 src/ - Source Code

### Core Applications

| File           | Purpose                                             |
| -------------- | --------------------------------------------------- |
| `web_app.py`   | **Flask web interface** - main UI at localhost:5000 |
| `main.py`      | **CLI entry point** - command-line pipeline         |
| `assistant.py` | **Voice assistant** - hotkey-driven screen analysis |
| `config.py`    | Global configuration settings                       |

### src/models/ - Summarization Models

| File                        | Purpose                                             |
| --------------------------- | --------------------------------------------------- |
| `textrank.py`               | Graph-based extractive summarization                |
| `lexrank.py`                | Eigenvector-based extractive summarization          |
| `gemini_summarizer.py`      | LLM-based abstractive (Groq + Gemini fallback)      |
| `constrained_summarizer.py` | **Hybrid summarizer** with KG grounding             |
| `knowledge_graph.py`        | Entity-relation extraction and KG building          |
| `fact_verifier.py`          | Verifies claims against Knowledge Graph             |
| `mmr.py`                    | Maximal Marginal Relevance for redundancy reduction |

### src/hoax_detection/ - Trust Layer

| File                    | Purpose                                               |
| ----------------------- | ----------------------------------------------------- |
| `hoax_classifier.py`    | IndoBERT + LoRA hoax detection model                  |
| `outlier_detector.py`   | Statistical outlier detection                         |
| `credibility_report.py` | **Main Trust Layer** - combines hoax + outlier scores |
| `train_lora.py`         | Training script for hoax classifier                   |

### src/tools/ - Utilities

| File                    | Purpose                                         |
| ----------------------- | ----------------------------------------------- |
| `enhanced_search.py`    | DuckDuckGo search + Crawl4AI content extraction |
| `search_tool.py`        | Basic web search wrapper                        |
| `source_credibility.py` | Source domain credibility scoring               |

### src/ - Other Files

| File                | Purpose                                     |
| ------------------- | ------------------------------------------- |
| `preprocessing.py`  | Indonesian text cleaning, Sastrawi stemmer  |
| `data_loader.py`    | XL-Sum dataset loader                       |
| `evaluation.py`     | ROUGE metrics evaluation                    |
| `voice_kokoro.py`   | Kokoro TTS wrapper (ultra-fast local voice) |
| `screen_capture.py` | MSS-based screen capture                    |
| `ocr_engine.py`     | Tesseract OCR wrapper                       |
| `monitor.py`        | Active window monitoring                    |

---

## 📁 models/ - Pre-trained Weights

| Folder                | Purpose                                     |
| --------------------- | ------------------------------------------- |
| `hoax_indobert_lora/` | Fine-tuned IndoBERT LoRA for hoax detection |

> ⚠️ This folder is **git-ignored** due to large file sizes. Download separately or train with `train_lora.py`.

---

## 📁 data/ - Datasets

| File/Folder            | Purpose                        |
| ---------------------- | ------------------------------ |
| `raw/`                 | Original scraped news articles |
| `cleaned/`             | Preprocessed datasets (xlsx)   |
| `summary/`             | Pre-generated summaries (csv)  |
| `rifky_hoax.csv`       | Indonesian hoax dataset        |
| `ultimate_dataset.csv` | Combined training data         |

> ⚠️ Data folder is **git-ignored**. Download from project resources.

---

## 📁 tests/ - Test Suite

| File                        | Purpose                         |
| --------------------------- | ------------------------------- |
| `grand_integration_demo.py` | End-to-end system demonstration |

---

## 📁 notebooks/ - Jupyter Notebooks

Contains experimental notebooks for:

- Model exploration
- Dataset analysis
- Evaluation experiments

---

## 🔧 Configuration Files

| File             | Purpose                       |
| ---------------- | ----------------------------- |
| `.bandit`        | Security linter config        |
| `.dockerignore`  | Docker build exclusions       |
| `pyproject.toml` | pytest, black, isort settings |

---

## 🚫 Git-Ignored (Not in Repository)

These folders are excluded from version control:

| Folder         | Reason                                  |
| -------------- | --------------------------------------- |
| `.venv/`       | Virtual environment (recreate with pip) |
| `data/`        | Large datasets (download separately)    |
| `models/`      | Model weights (download or train)       |
| `data_cache/`  | Hugging Face cache                      |
| `checkpoints/` | Training checkpoints                    |
| `.env`         | Contains API secrets                    |

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│         (Web App / CLI / Voice Assistant)                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    Trust Layer                           │
│         (Hoax Detection + Outlier Filter)                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 Knowledge Graph                          │
│     (Entity Extraction + Relation Mining + Timeline)     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│            Constrained Summarization                     │
│   (Hybrid: Extractive + Abstractive + KG Grounding)      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Verification & Corroboration                │
│        (Fact Verification + Web Search Grounding)        │
└─────────────────────────────────────────────────────────┘
```

---

_TDSM v1.0 - Trust-Driven Summarization Model_
