import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

HF_MODEL   = "prsubedi/solar-nlp-classifier"
MODELS_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(title="Space Physics NLP Classifier", version="2.0.0")

tokenizer = None
model     = None
le        = None

def load_model():
    global tokenizer, model, le
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model     = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
        model.eval()
        with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)


class TextInput(BaseModel):
    text: str


class ClassificationOutput(BaseModel):
    label:      str
    confidence: float
    all_scores: dict


@app.get("/")
def root():
    return {"message": "Space Physics NLP Classifier is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationOutput)
def classify(data: TextInput):
    load_model()
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_idx   = int(np.argmax(probs))
    pred_label = le.classes_[pred_idx]
    confidence = round(float(probs[pred_idx]), 4)
    all_scores = {le.classes_[i]: round(float(probs[i]), 4) for i in range(len(le.classes_))}

    return ClassificationOutput(
        label=pred_label,
        confidence=confidence,
        all_scores=all_scores,
    )