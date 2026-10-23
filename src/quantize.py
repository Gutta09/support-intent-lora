"""Merge the LoRA adapter, quantize Linear layers to int8, and benchmark both.

Measures what a support team would care about: accuracy, model size,
and latency for a single query and for a batch of 32.
"""
import io
import json
import time
from pathlib import Path

import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

from src.data import load_banking77


def set_quant_engine():
    # x86 ships fbgemm; Apple Silicon / ARM needs qnnpack
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"


def quantize_dynamic(model):
    set_quant_engine()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.inference_mode()
def evaluate(model, tok, texts, labels, batch=64):
    correct = 0
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
        correct += (model(**enc).logits.argmax(-1) == torch.tensor(labels[i:i + batch])).sum().item()
    return correct / len(texts)


@torch.inference_mode()
def latency_ms(model, tok, texts, batch, runs=20):
    enc = tok(texts[:batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
    for _ in range(3):
        model(**enc)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(**enc)
    return (time.perf_counter() - t0) / runs * 1000


def main(adapter="artifacts/lora"):
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(adapter)
    num_labels = len(json.loads(Path(adapter, "labels.json").read_text()))
    fp32 = AutoPeftModelForSequenceClassification.from_pretrained(
        adapter, num_labels=num_labels).merge_and_unload().eval()
    int8 = quantize_dynamic(fp32)
    ds, _ = load_banking77()
    texts, labels = ds["test"]["text"], ds["test"]["label"]

    rows = {}
    for name, m in [("fp32", fp32), ("int8", int8)]:
        rows[name] = {"accuracy": round(evaluate(m, tok, texts, labels), 4),
                      "size_mb": round(size_mb(m), 1),
                      "latency_ms_batch1": round(latency_ms(m, tok, texts, 1), 2),
                      "latency_ms_batch32": round(latency_ms(m, tok, texts, 32), 2)}
        print(name, rows[name])
    torch.save(int8, "artifacts/model_int8.pt")
    Path("results/quantization.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
