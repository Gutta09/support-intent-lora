"""Banking77: 13k real customer-support queries across 77 intents (PolyAI).

Source CSVs: github.com/PolyAI-LDN/task-specific-datasets (banking_data/).
"""
import csv
import urllib.request
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"


def _rows(split):
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL.format(split), path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_banking77(train_limit=None, seed=42):
    train, test = _rows("train"), _rows("test")
    labels = sorted({r["category"] for r in train})
    feats = Features({"text": Value("string"), "label": ClassLabel(names=labels)})

    def build(rows):
        return Dataset.from_dict({"text": [r["text"] for r in rows],
                                  "label": [labels.index(r["category"]) for r in rows]},
                                 features=feats)

    ds = DatasetDict(train=build(train), test=build(test))
    if train_limit:
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(train_limit))
    return ds, labels


def tokenize(ds, tokenizer, max_length=64):
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )
