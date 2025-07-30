#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install transformers==4.42.4 trl==0.9.4 accelerate==0.31.0 datasets==2.20.0 peft==0.11.1 bitsandbytes==0.43.1 evaluate==0.4.2 wandb==0.17.7 sentencepiece==0.2.0 protobuf==5.27.2 numpy==1.25.2 matplotlib
#pip install transformers==4.42.4 trl==0.9.4 accelerate==0.31.0 datasets==2.20.0 peft==0.11.1 bitsandbytes==0.42.0 evaluate==0.4.2 wandb==0.17.7 sentencepiece==0.2.0 protobuf==5.27.2 numpy==1.25.2 matplotlib
# todo: add curiosity, stop it from overusing ram


import os
import re
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    BitsAndBytesConfig
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import trange

# === Constants and Paths ===
ACTOR_MODEL_NAME = "meta-llama/Llama-3.2-1B"
REWARD_BACKBONE = "microsoft/deberta-v3-base"
OUTPUT_DIR = "./rlhf-demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SFT_MAX_SAMPLES = 8         # originally 512
RM_MAX_SAMPLES = 16         # originally 1000
PPO_UPDATES = 2             # originally 100
GEN_MAX_NEW_TOKENS = 32     # originally 128
MAX_PROMPT_LEN = 128        # originally 512


# === Device Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# === Load Dataset ===
hh = load_dataset("Anthropic/hh-rlhf", split=f"train[:{RM_MAX_SAMPLES}]")

def extract_prompt_from_conv(conv: str):
    human_blocks = re.findall(r"Human:(.*?)(?:Assistant:|$)", conv, flags=re.S)
    if not human_blocks:
        return None
    last_human = human_blocks[-1].strip()
    return f"Human: {last_human}\n\nAssistant:"

prompts = [extract_prompt_from_conv(ex["chosen"]) for ex in hh if extract_prompt_from_conv(ex["chosen"]) is not None]
print(f"Extracted {len(prompts)} prompts.")

# === Reward Model Training ===
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_BACKBONE, use_fast=True)

def rm_tokenize(batch):
    pos = reward_tokenizer(batch["chosen"], truncation=True, max_length=512, padding=True)
    neg = reward_tokenizer(batch["rejected"], truncation=True, max_length=512, padding=True)
    return {
        "input_ids_pos": pos["input_ids"],
        "attention_mask_pos": pos["attention_mask"],
        "input_ids_neg": neg["input_ids"],
        "attention_mask_neg": neg["attention_mask"],
    }

tokenized_rm = hh.map(rm_tokenize, batched=True, remove_columns=hh.column_names)

class PairwiseCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, features):
        pos = [{"input_ids": f["input_ids_pos"], "attention_mask": f["attention_mask_pos"]} for f in features]
        neg = [{"input_ids": f["input_ids_neg"], "attention_mask": f["attention_mask_neg"]} for f in features]
        batch_pos = self.tok.pad(pos, return_tensors="pt")
        batch_neg = self.tok.pad(neg, return_tensors="pt")
        return {
            "input_ids_pos": batch_pos["input_ids"],
            "attention_mask_pos": batch_pos["attention_mask"],
            "input_ids_neg": batch_neg["input_ids"],
            "attention_mask_neg": batch_neg["attention_mask"],
        }

class PairwiseRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        pos_out = model(input_ids=inputs["input_ids_pos"], attention_mask=inputs["attention_mask_pos"]).logits.squeeze(-1)
        neg_out = model(input_ids=inputs["input_ids_neg"], attention_mask=inputs["attention_mask_neg"]).logits.squeeze(-1)
        loss = -nn.functional.logsigmoid(pos_out - neg_out).mean()
        return (loss, {"pos": pos_out, "neg": neg_out}) if return_outputs else loss

reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_BACKBONE, num_labels=1).to(device)

args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "rm_hh"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-5,
    logging_steps=5,
    save_steps=99999,  # never save to save disk
    bf16=False,
    report_to=[],
    remove_unused_columns=False,
)


rm_trainer = PairwiseRewardTrainer(
    model=reward_model,
    args=args,
    train_dataset=tokenized_rm,
    data_collator=PairwiseCollator(reward_tokenizer),
)

rm_trainer.train()
#rm_trainer.save_model(os.path.join(OUTPUT_DIR, "rm_hh"))
#reward_tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "rm_hh"))
reward_model.eval().to(device)
print("Reward model trained on HH-RLHF.")

# === PPO Training ===
#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.bfloat16,
#)

bnb_config = None

actor_tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_NAME, use_fast=False)
actor_tokenizer.pad_token = actor_tokenizer.eos_token

actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ACTOR_MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ACTOR_MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


ppo_ds = Dataset.from_dict({"query": prompts})

ppo_config = PPOConfig(
    model_name=os.path.join(OUTPUT_DIR, "sft"),
    learning_rate=1e-5,
    mini_batch_size=1,
    batch_size=2,
    gradient_accumulation_steps=1,
    target_kl=0.1,
    ppo_epochs=1,
    seed=42,
)


ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=actor_model,
    ref_model=ref_model,
    tokenizer=actor_tokenizer,
    dataset=ppo_ds,
)

def compute_reward(prompts, responses):
    with torch.no_grad():
        toks = reward_tokenizer(responses, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        scores = reward_model(**toks).logits.squeeze(-1)
        return scores.detach().float().cpu().tolist()

gen_kwargs = dict(
    max_new_tokens=GEN_MAX_NEW_TOKENS,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    pad_token_id=actor_tokenizer.eos_token_id
)

for step in trange(PPO_UPDATES):
    batch = next(iter(ppo_trainer.dataloader))
    queries = batch["query"]
    query_tensors = actor_tokenizer(queries, padding=True, truncation=True, max_length=MAX_PROMPT_LEN, return_tensors="pt").to(actor_model.device)
    response_tensors = ppo_trainer.generate(query_tensors["input_ids"], **gen_kwargs)
    responses = actor_tokenizer.batch_decode(response_tensors[:, query_tensors["input_ids"].shape[-1]:], skip_special_tokens=True)
    rewards = compute_reward(queries, responses)
    stats = ppo_trainer.step(query_tensors["input_ids"], response_tensors, torch.tensor(rewards).to(actor_model.device))
    ppo_trainer.log_stats(stats, batch, rewards)

#ppo_trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "ppo_hh"))
#actor_tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "ppo_hh"))
print("PPO w/ HH-RLHF complete.")

pipe = pipeline(
    "text-generation",
    model=os.path.join(OUTPUT_DIR, "ppo_hh"),
    tokenizer=actor_tokenizer,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

prompt = "Human: Explain what RLHF is and why Anthropic built HH-RLHF.\n\nAssistant:"
print(pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"])

# === Plot PPO Metrics ===
stats = pd.DataFrame(ppo_trainer.stats_history).sort_values(by="ppo/epoch", ignore_index=True)

plt.figure(figsize=(10, 4))
plt.plot(stats["ppo/epoch"], stats["env/reward_mean"], label="Reward (mean)")
plt.fill_between(stats["ppo/epoch"], stats["env/reward_mean"] - stats["env/reward_std"], stats["env/reward_mean"] + stats["env/reward_std"], alpha=0.2)
plt.title("Reward vs PPO Steps")
plt.xlabel("PPO Epoch")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(stats["ppo/epoch"], stats["objective/kl"], color="orange", label="KL Divergence")
plt.axhline(y=ppo_config.target_kl, color="red", linestyle="--", label="Target KL")
plt.title("KL Divergence per PPO Step")
plt.xlabel("PPO Epoch")
plt.ylabel("KL")
plt.grid(True)
plt.legend()
plt.show()

if "env/response_length_mean" in stats.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(stats["ppo/epoch"], stats["env/response_length_mean"], color="green", label="Response Length")
    plt.title("Response Length vs PPO Steps")
    plt.xlabel("PPO Epoch")
    plt.ylabel("Mean Length (tokens)")
    plt.grid(True)
    plt.legend()
    plt.show()
