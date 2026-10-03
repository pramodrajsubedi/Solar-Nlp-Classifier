# Space Physics NLP Classifier

A scientific text classification system fine-tuned on arXiv abstracts to classify space physics literature into four domain categories — deployed as a live REST API.

| | |
|---|---|
| **REST API** | https://solar-nlp-classifier.onrender.com/docs |
| **HuggingFace Model** | https://huggingface.co/prsubedi/solar-nlp-classifier |

---

## What it does

| Layer | Description |
|---|---|
| Data collection | Fetches arXiv abstracts via the arXiv API across 4 space science categories |
| Preprocessing | Deduplication, length filtering, stratified train/val/test split |
| Baseline | TF-IDF + Logistic Regression benchmark |
| Fine-tuning | SciBERT (`allenai/scibert_scivocab_uncased`) fine-tuned for sequence classification |
| REST API | FastAPI endpoint returning label, confidence, and per-class scores |
| Deployment | Dockerized, deployed on Render with CI/CD via GitHub Actions |

---

## Model Performance

5-fold stratified cross-validation on 1,091 arXiv abstracts.

| Metric | Score |
|---|---|
| Test Accuracy | **0.86** |
| Macro F1 | **0.86** |
| Val Accuracy | 0.886 |

**Per-class results:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Earth Planetary | 0.86 | 0.68 | 0.76 | 44 |
| Geophysics | 0.94 | 0.89 | 0.92 | 57 |
| Solar Stellar | 0.88 | 0.88 | 0.88 | 58 |
| Space Physics | 0.79 | 0.95 | 0.86 | 60 |

> SciBERT outperformed DistilBERT by +3% accuracy and +6% on Geophysics F1, validating the use of a science-domain pre-trained model over a general-purpose one.

---

## REST API

Built with FastAPI, containerized with Docker, deployed on Render.

```bash
curl -X POST https://solar-nlp-classifier.onrender.com/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Solar flare observed in active region 12345. X-class flare with strong EUV emission and associated CME detected by LASCO."
  }'
```

**Response:**
```json
{
  "label": "Solar Stellar",
  "confidence": 0.7664,
  "all_scores": {
    "Earth Planetary": 0.0469,
    "Geophysics": 0.0380,
    "Solar Stellar": 0.7664,
    "Space Physics": 0.1487
  }
}
```

---

## MLOps

| Component | Tool |
|---|---|
| API framework | FastAPI |
| Containerization | Docker |
| Model hosting | HuggingFace Hub |
| Deployment | Render (free tier) |
| CI/CD | GitHub Actions — runs tests and builds Docker image on every push |

---

## Dataset

| Property | Value |
|---|---|
| Source | arXiv API |
| Total records | 1,091 |
| Classes | 4 |
| Train / Val / Test | 784 / 88 / 219 |
| Text format | Title + Abstract |

**Class distribution:**

| Class | arXiv Category | Count |
|---|---|---|
| Space Physics | physics.space-ph | 300 |
| Solar Stellar | astro-ph.SR | 286 |
| Geophysics | physics.geo-ph | 285 |
| Earth Planetary | astro-ph.EP | 220 |

> Labels are author-assigned arXiv taxonomy tags — not inferred or manually annotated.

---

## Architecture

```
arXiv API
    │
    ▼
fetch_data.py         ← Fetches abstracts across 4 categories
    │
    ▼
solar_events.csv      ← 1,091 labeled text records
    │
    ▼
train.ipynb           ← SciBERT fine-tuning on T4 GPU (Google Colab)
    │
    ▼
solar_nlp_model/      ← Saved model + tokenizer + label encoder
    │
    ├── HuggingFace Hub   ← Model hosted at prsubedi/solar-nlp-classifier
    │
    ▼
api/main.py           ← FastAPI REST endpoint
    │
    ▼
Dockerfile            ← Containerized deployment
    │
    ▼
Render                ← Live public API
```

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | allenai/scibert_scivocab_uncased |
| Epochs | 5 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max token length | 512 |
| Hardware | Google Colab T4 GPU |
| Training time | ~4 minutes |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Warmup steps | 50 |

---

## Why SciBERT over DistilBERT

| Model | Test Accuracy | Macro F1 | Confidence (sample) |
|---|---|---|---|
| DistilBERT | 0.83 | 0.83 | 0.667 |
| **SciBERT** | **0.86** | **0.86** | **0.766** |

SciBERT was pre-trained on 1.14M scientific papers from Semantic Scholar. For domain-specific scientific text, it consistently outperforms general-purpose models trained on Wikipedia and books.

---

## Stack

| Tool | Role |
|---|---|
| Python 3.10+ | Core language |
| HuggingFace Transformers | SciBERT fine-tuning and inference |
| PyTorch | Training backend |
| scikit-learn | Label encoding, train/test split, evaluation |
| FastAPI | REST API with Pydantic validation |
| Docker | Containerized deployment |
| GitHub Actions | CI/CD — automated tests and Docker build on every push |
| Render | Live cloud deployment |
| HuggingFace Hub | Model hosting |
| Google Colab | T4 GPU training environment |
| arXiv API | Training data source |

---

## Author

**Pramod Raj Subedi**
[LinkedIn](https://www.linkedin.com/in/pramodrajsubedi/)
