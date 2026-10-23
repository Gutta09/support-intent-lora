"""End-to-end API check against the real trained artifacts (skipped before training)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(not Path("artifacts/model_int8.pt").exists(),
                                reason="run src.train and src.quantize first")


@pytest.fixture(scope="module")
def client():
    from src.app import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_routes_real_support_queries(client):
    r = client.post("/predict", json={"texts": ["I lost my card, please block it",
                                                "Why was I charged a fee for withdrawing cash?"]})
    intents = [x["intent"] for x in r.json()["results"]]
    assert intents[0] in {"lost_or_stolen_card", "card_swallowed"}
    assert "cash_withdrawal" in intents[1] or "fee" in intents[1]


def test_rejects_empty_batch(client):
    assert client.post("/predict", json={"texts": []}).status_code == 422
