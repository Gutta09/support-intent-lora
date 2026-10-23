# support-intent-lora

Routes customer-support messages to one of 77 banking intents, and hands anything it isn't sure about to a human.

A DistilBERT model is fine-tuned with **LoRA** (Hugging Face Transformers + PEFT) on
[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets), which has 13,083 real customer queries
such as "I am still waiting on my card?". The model is then **quantized to int8** and benchmarked
against fp32, and a **Keras / TensorFlow** baseline sets the score it has to beat. A FastAPI service
serves predictions, with caching and batching.

## Results

All numbers come from the files in `results/`, on the full 3,080-query test set, run on an Apple M-series laptop (CPU for the benchmarks).

| Model | Test accuracy | Trainable params | Train time |
|---|---|---|---|
| Keras TF-IDF bigrams + dense net (TensorFlow backend) | 88.3% | all (≈5M) | 0.1 min |
| **DistilBERT + LoRA (r=16, attention layers)** | **91.2%** | **1.24M of 68.3M (1.8%)** | 3.5 min, 3 epochs |

Accuracy by epoch for LoRA: 80.0% → 87.6% → 91.2%. It was still climbing at epoch 3, so more epochs
would likely help.

### Quantization: fp32 vs dynamic int8 (`torch.ao.quantization.quantize_dynamic`, Linear layers)

| | Accuracy | Size | Latency, 1 query | Latency, batch of 32 |
|---|---|---|---|---|
| fp32 (LoRA merged) | 91.17% | 268 MB | 7.0 ms | 55 ms |
| int8 | 90.75% | 139 MB | 5.4 ms | 197 ms |

What this shows:
- **Size halves** and accuracy drops by just 0.4 points. The reduction stops at about half because the embedding table is not quantized, and only the Linear layers are.
- **Single-query latency improves 23%.**
- **Batched int8 is 3.5× slower than fp32 on this machine.** ARM uses the `qnnpack` quantized
  backend, and in this run it lost to fp32's batched matrix multiply. On x86 (`fbgemm`) the result may differ, but I have not
  measured that. For high-throughput batch scoring here, fp32 is the better choice. Int8 wins on memory and single requests.
- **Batching alone is the biggest latency gain.** fp32 drops from 7.0 ms per query one at a time to 1.7 ms per query in a batch of 32.

## How the service decides

`POST /predict` with `{"texts": [...]}` returns an intent and a confidence for each message.
Anything below `ESCALATE_BELOW` (default 0.5) gets `"escalate": true`, so it goes to a person instead of an automated answer.

```
"I lost my card, please block it"               -> lost_or_stolen_card     0.84
"Why was I charged a fee for withdrawing cash?" -> cash_withdrawal_charge  1.00
"hello there"                                   -> card_delivery_estimate  0.23  escalate
```

- **Caching:** repeated single messages are served from an LRU cache.
- **Batching:** multi-message requests run in one forward pass.

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.train --epochs 3      # LoRA fine-tune  -> artifacts/lora, results/lora.json
.venv/bin/python -m src.quantize              # int8 + benchmark -> artifacts/model_int8.pt
.venv/bin/python -m src.keras_baseline        # TensorFlow/Keras baseline
.venv/bin/python -m pytest -q                 # 8 tests
.venv/bin/uvicorn src.app:app --port 8000
```

Docker: `docker build -t support-intent-lora . && docker run -p 8000:8000 support-intent-lora` (run this after training, because the image copies `artifacts/`). Verified: the 1.9 GB CPU image serves `/predict`.

## Layout

```
src/data.py            Banking77 loader (downloads the PolyAI CSVs on first run)
src/train.py           LoRA fine-tuning with the HF Trainer
src/quantize.py        merge adapter, int8 dynamic quantization, accuracy/size/latency benchmark
src/keras_baseline.py  Keras 3 on TensorFlow: TF-IDF bigrams -> dense classifier
src/app.py             FastAPI service with an escalation threshold, caching and batching
tests/                 routing, quantization, LoRA parameter share, and API tests
```

## Notes

- **Keras validation fix:** the Banking77 training CSV is grouped by intent. Keras's `validation_split` takes the *last* 10%, so the
  first baseline run validated on classes it had never seen (5.6% val accuracy) and early stopping misfired.
  The baseline now shuffles before splitting.
- **Not done yet:** no hyperparameter search, and no cloud deployment.
