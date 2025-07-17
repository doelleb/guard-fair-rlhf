import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_PATH = "./models/llama3_rm/last_checkpoint"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, use_fast=False)
model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH).to(device)
model.eval()

rb = load_dataset("allenai/reward-bench-2", "default", split="test")

print("\nSample raw scores (first 5 examples):")
def score_raw(text):
    if isinstance(text, (list, tuple)):
        text = text[-1]
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=tokenizer.model_max_length,
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits.squeeze()
    if logits.dim() > 0:
        logits = logits[0]
    return logits.item()

samples = rb.select(range(5))
for ex in samples:
    sc_chosen = score_raw(ex["chosen"])
    sc_rej    = [score_raw(r) for r in ex["rejected"]]
    print(f" CHOSEN: {sc_chosen:.4f}   REJ max: {max(sc_rej):.4f}   DIFF: {(sc_chosen - max(sc_rej)):.4f}")

# 5) Scoring helper using raw logits
def score(text: str) -> float:
    if isinstance(text, (list, tuple)):
        text = text[-1]
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=tokenizer.model_max_length,
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits.squeeze()
    if logits.dim() > 0:
        logits = logits[0]
    return logits.item()

# 6) Loop through RewardBench examples
correct = 0
total   = 0

for ex in rb:
    sc_chosen = score(ex["chosen"])
    sc_rej    = [score(r) for r in ex["rejected"]]
    if sc_chosen > max(sc_rej):
        correct += 1
    total += 1

# 7) Report
acc = correct / total * 100
print(f"\nRewardBench accuracy: {correct}/{total} = {acc:.2f}%")
