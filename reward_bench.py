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

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("allenai/reward-bench-2")

print(ds)