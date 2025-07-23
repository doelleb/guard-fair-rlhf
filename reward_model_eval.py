# plots for reward model 
import fair_reward_model_disc as frm 


from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoModel
)
from transformers.utils import PaddingStrategy
import math
from datasets import load_dataset
import numpy as np

import matplotlib.pyplot as plt

import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_reward_score(text: str) -> float:
    enc = tokenizer(
        text,
        truncation=True,
        padding="longest",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    # move to GPU
    enc = {k: v.to(device) for k, v in enc.items()}
    # cast only attention_mask to fp16
    if "attention_mask" in enc:
        enc["attention_mask"] = enc["attention_mask"].half()
    with torch.no_grad():
        out = model(**enc).logits.squeeze()
    return out.cpu().item()


config = AutoConfig.from_pretrained("models/llama3_rm/llama32_checkpoint/config.json")

model = AutoModelForSequenceClassification.from_pretrained("models/llama3_rm/llama32_checkpoint", config=config).to(device)


tokenizer = AutoTokenizer.from_pretrained("models/llama3_rm/llama32_checkpoint")


ds_helpful  = load_dataset(
    "Anthropic/hh-rlhf",
    split="train[5000:6000]",
    data_dir="helpful-base"
)
ds_harmless = load_dataset(
    "Anthropic/hh-rlhf",
    split="train[5000:6000]",
    data_dir="harmless-base"
)

helpful_scores  = np.array([get_reward_score(t) for t in ds_helpful["chosen"]])
harmless_scores = np.array([get_reward_score(t) for t in ds_harmless["chosen"]])

# Create histogram plot
plt.figure(figsize=(10, 6))
plt.hist(helpful_scores, bins=30, alpha=0.7, label="Helpful", color='blue', edgecolor='black')
plt.hist(harmless_scores, bins=30, alpha=0.7, label="Harmless", color='red', edgecolor='black')
plt.title("Reward Distribution: Helpful vs Harmless (Anthropic HH-RLHF)")
plt.xlabel("Reward Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reward_model_eval_fairness.png", dpi=300, bbox_inches='tight')





