"""Support-ticket intent API: route a customer message, or escalate when unsure."""
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

ARTIFACTS = Path(os.getenv("ARTIFACTS", "artifacts"))
ESCALATE_BELOW = float(os.getenv("ESCALATE_BELOW", "0.5"))

app = FastAPI(title="support-intent-lora")


class Query(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


def route(probs, labels, threshold=ESCALATE_BELOW):
    conf, idx = probs.max(-1)
    return [{"intent": labels[i], "confidence": round(c, 4), "escalate": c < threshold}
            for c, i in zip(conf.tolist(), idx.tolist())]


@lru_cache(maxsize=1)
def model_bundle():
    from transformers import AutoTokenizer

    from src.quantize import set_quant_engine
    set_quant_engine()
    tok = AutoTokenizer.from_pretrained(ARTIFACTS / "lora")
    labels = json.loads((ARTIFACTS / "lora" / "labels.json").read_text())
    model = torch.load(ARTIFACTS / "model_int8.pt", weights_only=False).eval()
    return tok, model, labels


@lru_cache(maxsize=2048)
def _predict_one(text):
    return predict_batch((text,))[0]


@torch.inference_mode()
def predict_batch(texts):
    tok, model, labels = model_bundle()
    enc = tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
    return route(model(**enc).logits.softmax(-1), labels)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(q: Query):
    if len(q.texts) == 1:
        return {"results": [_predict_one(q.texts[0])]}
    return {"results": predict_batch(tuple(q.texts))}
