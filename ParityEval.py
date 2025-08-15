#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO Policy Evaluation (HH-RLHF reward as safety metric) for multiple models
"""

import os, math, json, random
from typing import List
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

# =========================
# HARD-CODED PATHS
# =========================
MODELS = {
    "baseline_no_curiosity": r"C:\Users\advay\Desktop\Algoverse\ppo_runs\model_baseline_nocuri",
    "fair_no_curiosity":     r"C:\Users\advay\Desktop\Algoverse\ppo_runs\model_fair_nocuri",
    "fair_curiosity":        r"C:\Users\advay\Desktop\Algoverse\ppo_runs\model_fair_curi"
}

OUT_DIR = r"C:\Users\advay\Desktop\Algoverse\ppo_runs\ppo_eval_hhrlhf"
SAFETY_MODEL_REPO = "OpenAssistant/reward-model-deberta-v3-large-v2"

# Generation settings
MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.7
TOP_P          = 0.9
BATCH_SIZE     = 8
SEED           = 42

# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def batched(items, bs: int):
    for i in range(0, len(items), bs):
        yield items[i:i+bs]

def load_causal_model(path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    mdl = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    ).to(device).eval()
    return tok, mdl

@torch.no_grad()
def generate_batch(tokenizer, model, prompts: List[str], max_new_tokens=512, temperature=0.7, top_p=0.9):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
    )
    texts = []
    for i in range(out.size(0)):
        in_len = (enc.input_ids[i] != tokenizer.pad_token_id).sum().item()
        new_tokens = out[i][in_len:]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return texts

# =========================
# Safety scorer
# =========================
class HHRewardScorer:
    def __init__(self, repo: str, device: torch.device):
        self.tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            repo,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(device).eval()

    @torch.no_grad()
    def score(self, texts: List[str]) -> List[float]:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.mdl.device)
        logits = self.mdl(**enc).logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        sel = logits[:, 0].float()  # positive logit
        return torch.sigmoid(sel).cpu().numpy().tolist()

# =========================
# Load HH-RLHF prompts
# =========================
def load_hhrlhf_prompts(n_helpful: int = 100, n_harmless: int = 100) -> pd.DataFrame:
    ds_helpful = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    helpful_prompts = ds_helpful.shuffle(SEED).select(range(n_helpful))
    
    ds_harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    harmless_prompts = ds_harmless.shuffle(SEED).select(range(n_harmless))

    all_prompts = []
    for row in helpful_prompts:
        all_prompts.append((row["chosen"], "helpful"))
    for row in harmless_prompts:
        all_prompts.append((row["chosen"], "harmless"))

    df = pd.DataFrame(all_prompts, columns=["prompt", "category"])
    df.insert(0, "id", range(len(df)))
    return df

# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load prompts
    df = load_hhrlhf_prompts(100, 100)
    print(f"[Info] Using {len(df)} prompts: {df['category'].value_counts().to_dict()}")
    prompts = df["prompt"].tolist()

    # Safety scorer
    scorer = HHRewardScorer(SAFETY_MODEL_REPO, device)

    # Store results
    all_results = df.copy()

    for label, path in MODELS.items():
        print(f"[Load] Actor: {label} ({path})")
        tok, mdl = load_causal_model(path, device)

        print(f"[Gen] {label} generating...")
        gens = []
        for batch in tqdm(batched(prompts, BATCH_SIZE), total=math.ceil(len(prompts)/BATCH_SIZE)):
            gens.extend(generate_batch(tok, mdl, batch, MAX_NEW_TOKENS, TEMPERATURE, TOP_P))

        print(f"[Score] Scoring {label} generations...")
        scores = scorer.score(gens)

        all_results[f"response_{label}"] = gens
        all_results[f"safety_{label}"] = scores

        # Per-category stats
        cat = all_results.groupby("category")[f"safety_{label}"].mean().reset_index()
        cat_path = os.path.join(OUT_DIR, f"per_category_means_{label}.csv")
        cat.to_csv(cat_path, index=False, encoding="utf-8")
        print(f"[Save] {cat_path}")

        gap = float(cat[f"safety_{label}"].max() - cat[f"safety_{label}"].min())
        report = {"parity_gap": gap}
        gap_path = os.path.join(OUT_DIR, f"parity_gap_{label}.txt")
        with open(gap_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2))
        print(f"[Parity Gap] {label}: {json.dumps(report, indent=2)}")

    # Save all generations + scores
    per_sample_path = os.path.join(OUT_DIR, "per_sample_results_all_models.csv")
    all_results.to_csv(per_sample_path, index=False, encoding="utf-8")
    print(f"[Save] {per_sample_path}")

    print("[Done] HH-RLHF safety evaluation complete for all models.")

if __name__ == "__main__":
    main()
