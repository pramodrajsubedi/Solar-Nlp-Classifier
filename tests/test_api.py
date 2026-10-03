from unittest.mock import patch, MagicMock
import torch

MOCK_LABELS = ["Earth Planetary", "Geophysics", "Solar Stellar", "Space Physics"]

def make_client():
    with patch("api.main.AutoTokenizer.from_pretrained", return_value=MagicMock()), \
         patch("api.main.AutoModelForSequenceClassification.from_pretrained", return_value=MagicMock()), \
         patch("builtins.open", MagicMock()), \
         patch("pickle.load", return_value=MagicMock(classes_=MOCK_LABELS)):
        import importlib
        import api.main
        importlib.reload(api.main)
        from fastapi.testclient import TestClient
        return TestClient(api.main.app)

def test_health():
    client = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_root():
    client = make_client()
    r = client.get("/")
    assert r.status_code == 200