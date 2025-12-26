---
base_model: indobenchmark/indobert-base-p1
library_name: peft
tags:
  - base_model:adapter:indobenchmark/indobert-base-p1
  - lora
  - transformers
  - hoax-detection
  - indonesian
  - text-classification
language:
  - id
license: mit
---

# IndoBERT + LoRA Hoax Detection Model

A fine-tuned Indonesian hoax/fake news detection model using LoRA (Low-Rank Adaptation) on IndoBERT.

## Model Details

### Model Description

This model classifies Indonesian news articles as either **VALID** (legitimate news) or **HOAX** (misinformation/fake news). It was developed as part of the TDSM (Trust-Driven Summarization with Multi-document) project for Indonesian NLP.

- **Developed by:** Felix Nathaniel, Dennison Seodibjo, and Wilbert Devos Kyenil
- **Model type:** Text Classification (Binary)
- **Language(s):** Indonesian (id)
- **License:** MIT
- **Finetuned from model:** [indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)

### Model Sources

- **Repository:** [Fane-Nathan/NLP-Project](https://github.com/Fane-Nathan/NLP-Project)

## Uses

### Direct Use

Classify Indonesian text as hoax or valid news:

```python
from src.hoax_detection.classifier import HoaxClassifier

classifier = HoaxClassifier("models/hoax_indobert_lora")
result = classifier.predict("Berita ini sangat mencurigakan...")
print(result.label)  # "HOAX" or "VALID"
print(result.hoax_probability)  # 0.0 - 1.0
```

### Downstream Use

Integrated into the TDSM pipeline as the **Trust Layer** gatekeeper, filtering potentially unreliable sources before summarization.

### Out-of-Scope Use

- Non-Indonesian text
- Formal academic papers or technical documentation
- Satire/parody content (may be misclassified)

## Bias, Risks, and Limitations

- Trained primarily on social media and news article patterns
- May have biases toward certain topics or writing styles in the training data
- Should not be used as the sole determinant of content credibility

### Recommendations

Use in combination with other verification methods (fact-checking, source verification, outlier detection) for robust credibility assessment.

## Training Details

### Training Data

Combined dataset from:

- TurnBackHoax (Mafindo) fact-checking database
- Komdigi hoax repository
- Additional curated Indonesian news sources

### Training Procedure

#### Training Hyperparameters

- **Training regime:** FP16 mixed precision
- **LoRA rank (r):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Learning rate:** 2e-4
- **Batch size:** 4 (effective 16 with gradient accumulation)
- **Epochs:** 5
- **Optimizer:** AdamW (fused)
- **Scheduler:** Cosine annealing

## Technical Specifications

### Model Architecture and Objective

- **Base:** BERT (IndoBERT-base-p1, 124M parameters)
- **Adaptation:** LoRA applied to query, key, value, and dense layers
- **Trainable parameters:** ~1.2M (< 1% of base model)
- **Classification head:** 2-class softmax

### Compute Infrastructure

#### Hardware

- NVIDIA GTX 1650 (4GB VRAM) - optimized for low-memory training
- Intel Core i5/Ryzen 5 class CPU

#### Software

- Python 3.11
- PyTorch 2.0+
- Transformers 4.40+
- PEFT 0.13.0 / 0.18.0

## Model Card Authors

Felix Nathaniel, Dennison Seodibjo, and Wilbert Devos Kyenil

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/Fane-Nathan/NLP-Project/issues).

### Framework versions

- PEFT 0.13.0 / 0.18.0
- Transformers 4.40+
- PyTorch 2.0+
