"""Fine-tune DistilBERT on Banking77 with LoRA (PEFT): only the adapters train."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from src.data import MODEL_NAME, load_banking77, tokenize


def lora_model(num_labels, r=16):
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=32, lora_dropout=0.1,
                     target_modules=["q_lin", "k_lin", "v_lin", "out_lin"])
    return get_peft_model(base, cfg)


def trainable_share(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def accuracy(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, -1) == labels).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--out", default="artifacts/lora")
    args = ap.parse_args()

    ds, labels = load_banking77(args.train_limit)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenize(ds, tok)
    model = lora_model(len(labels))
    trainable, total = trainable_share(model)
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="artifacts/run", num_train_epochs=args.epochs,
                               per_device_train_batch_size=32, per_device_eval_batch_size=64,
                               learning_rate=2e-3, eval_strategy="epoch", save_strategy="no",
                               logging_steps=50, report_to=[], seed=42),
        train_dataset=enc["train"], eval_dataset=enc["test"],
        data_collator=DataCollatorWithPadding(tok), compute_metrics=accuracy,
    )
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    metrics = trainer.evaluate()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "labels.json").write_text(json.dumps(labels))
    result = {"test_accuracy": metrics["eval_accuracy"], "trainable_params": trainable,
              "total_params": total, "epochs": args.epochs, "train_minutes": round(minutes, 1),
              "train_examples": len(enc["train"])}
    Path("results/lora.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
