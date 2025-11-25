# Hoax Detection Module 🔍

Research-grade credibility analysis for Indonesian news documents using parallel hoax classification and outlier detection.

## Overview

This module provides a pre-summarization credibility filtering system that:

1. **Hoax Classification** (IndoBERT + LoRA)
   - Fine-tuned on TurnBackHoax dataset
   - Detects Indonesian fake news patterns
   - Outputs confidence scores

2. **Outlier Detection** (Cosine Similarity)
   - Identifies off-topic documents
   - Uses TF-IDF or BERT embeddings
   - Strict 2σ threshold for high-stakes filtering

3. **Combined Scoring**
   - Parallel architecture (independent analysis)
   - Weighted combination (60% hoax, 40% outlier)
   - Four credibility levels: HIGH, MEDIUM, LOW, CRITICAL

## Architecture

```
                    ┌─────────────────────┐
                    │   Input Documents   │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
               ▼                               ▼
    ┌─────────────────────┐      ┌─────────────────────┐
    │  Hoax Classifier    │      │  Outlier Detector   │
    │  (IndoBERT + LoRA)  │      │  (TF-IDF + Cosine)  │
    └──────────┬──────────┘      └──────────┬──────────┘
               │                               │
               │   PARALLEL EXECUTION          │
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Combined Scoring   │
                    │  & Level Assignment │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Credibility Report  │
                    │ + Filtered Docs     │
                    └─────────────────────┘
```

## Quick Start

### 1. Generate Training Data (if no TurnBackHoax available)

```bash
python -m src.data.download_data --generate --samples 2000
```

### 2. Train the Hoax Classifier

```bash
# With synthetic data
python -m src.hoax_detection.train_lora --synthetic

# With real TurnBackHoax data
python -m src.hoax_detection.train_lora --data_path data/turnbackhoax.csv
```

### 3. Run Credibility Analysis

```python
from src.hoax_detection import CredibilityAnalyzer

# Initialize analyzer
analyzer = CredibilityAnalyzer(
    hoax_model_path="models/hoax_indobert_lora",
    outlier_threshold_z=2.0,  # STRICT
    hoax_weight=0.6,
    outlier_weight=0.4
)

# Analyze documents
documents = [
    "Pemerintah mengumumkan vaksinasi untuk masyarakat.",
    "VIRAL! Vaksin berbahaya! Bagikan sebelum dihapus!",
    "Resep rendang padang yang enak."  # Off-topic
]

report = analyzer.analyze(documents)
report.print_summary()

# Get filtered documents for summarization
approved_docs, _ = analyzer.filter_documents(documents)
```

### 4. CLI Usage

```bash
# Run credibility analysis only
python -m src.main --mode credibility --input_file data/docs.json --report output/report.json

# Summarize with credibility filtering
python -m src.main --mode summarize --model textrank --input_file data/docs.json --credibility
```

## Components

### HoaxClassifier (`classifier.py`)

IndoBERT + LoRA classifier for detecting Indonesian hoaxes.

```python
from src.hoax_detection import HoaxClassifier

classifier = HoaxClassifier(model_path="models/hoax_indobert_lora")
result = classifier.predict("VIRAL! Berita mengejutkan ini harus kamu baca!")

print(result.label)           # "HOAX"
print(result.confidence)      # 0.92
print(result.hoax_probability)  # 0.92
```

### OutlierDetector (`outlier_detector.py`)

Statistical outlier detection using cosine similarity.

```python
from src.hoax_detection import OutlierDetector

detector = OutlierDetector(threshold_z=2.0)  # STRICT
analysis = detector.detect_outliers(documents)

for result in analysis.results:
    print(f"Doc {result.doc_index}: outlier={result.is_outlier}")
```

### CredibilityAnalyzer (`credibility_report.py`)

Complete pipeline combining both approaches.

```python
from src.hoax_detection import CredibilityAnalyzer

analyzer = CredibilityAnalyzer()
report = analyzer.analyze(documents)

# Access results
print(f"Approved: {report.documents_for_summarization}")
print(f"Excluded: {report.documents_excluded}")
print(f"Risk Level: {report.collection_risk_level}")

# Export
report.save("report.json")
```

## Credibility Levels

| Level | Description | Action |
|-------|-------------|--------|
| **HIGH** | Valid content, fits document cluster | Include in summary |
| **MEDIUM** | Minor concerns but acceptable | Include with caution |
| **LOW** | Either hoax OR outlier detected | Exclude from summary |
| **CRITICAL** | Both hoax AND outlier - highest risk | Exclude, flag for review |

## Training Configuration

Optimized for **4GB GPU** (GTX 1650 / RTX 3050):

| Parameter | Value | Reason |
|-----------|-------|--------|
| LoRA Rank | r=8 | Minimal memory overhead |
| Batch Size | 4 | Fits in 4GB VRAM |
| Gradient Accumulation | 4 | Effective batch of 16 |
| Max Length | 256 | Balance speed/quality |
| FP16 | Enabled | 50% memory reduction |
| Gradient Checkpointing | Enabled | Further memory savings |

## File Structure

```
src/hoax_detection/
├── __init__.py           # Module exports
├── classifier.py         # HoaxClassifier (IndoBERT + LoRA)
├── outlier_detector.py   # OutlierDetector (Cosine Similarity)
├── credibility_report.py # CredibilityAnalyzer & Report
└── train_lora.py         # LoRA fine-tuning script

models/
└── hoax_indobert_lora/   # Saved LoRA adapters
    ├── adapter_config.json
    ├── adapter_model.bin
    └── config.json

data/
├── turnbackhoax.csv      # TurnBackHoax dataset (manual download)
└── synthetic_turnbackhoax.csv  # Generated synthetic data
```

## Evaluation Metrics

The system evaluates:

1. **Hoax Detection**
   - Accuracy, Precision, Recall, F1
   - Separate F1 for HOAX and VALID classes

2. **Outlier Detection**
   - Precision, Recall at threshold
   - Mean similarity to centroid

3. **Pipeline Quality**
   - Documents correctly filtered
   - False positive/negative rates

## Thresholds

| Strategy | Z-Score | Use Case |
|----------|---------|----------|
| **STRICT** | >2σ | High-stakes fact-checking (default) |
| MODERATE | >1.5σ | General news aggregation |
| LOOSE | >3σ | Diverse topic coverage |

## References

- **TurnBackHoax Dataset**: [Mafindo](https://turnbackhoax.id/)
- **IndoBERT**: [IndoBenchmark](https://github.com/indobenchmark/indonlu)
- **LoRA**: [Microsoft Research](https://github.com/microsoft/LoRA)

## License

MIT License - See project root for details.
