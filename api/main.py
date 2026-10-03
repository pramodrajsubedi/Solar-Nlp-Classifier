from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Space Physics NLP Classifier", version="2.0.0")


class TextInput(BaseModel):
    text: str


class ClassificationOutput(BaseModel):
    label:      str
    confidence: float
    message:    str


@app.get("/")
def root():
    return {"message": "Space Physics NLP Classifier is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationOutput)
def classify(data: TextInput):
    return ClassificationOutput(
        label="Solar Stellar",
        confidence=0.7664,
        message="Full model inference available locally. Render free tier does not support 1.5GB SciBERT. See GitHub for full implementation.",
    )