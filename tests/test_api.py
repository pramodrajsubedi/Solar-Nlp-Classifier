from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_classify_returns_valid_schema():
    r = client.post("/classify", json={
        "text": "Solar flare observed with strong X-ray emission from active region."
    })
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ["Earth Planetary", "Geophysics", "Solar Stellar", "Space Physics"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["all_scores"]) == 4