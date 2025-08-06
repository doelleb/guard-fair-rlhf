# === Install Requirements (in terminal before running) ===
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers==4.42.4 trl==0.9.4 accelerate==0.31.0 datasets==2.20.0 peft==0.11.1 bitsandbytes==0.43.1 evaluate==0.4.2 wandb==0.17.7 sentencepiece==0.2.0 protobuf==5.27.2 numpy==1.25.2 matplotlib scikit-learn

import os
import re
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import List

from datasets import load_dataset, Dataset
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from tqdm import trange
from sklearn.cluster import MiniBatchKMeans

ACTOR_MODEL_NAME = "meta-llama/Llama-3.2-1B"
REWARD_BACKBONE = "microsoft/deberta-v3-base"
OUTPUT_DIR = "/data/user_data/kailashr/MARTI/rlhf_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SFT_MAX_SAMPLES = 4
RM_MAX_SAMPLES = 8
PPO_UPDATES = 100
GEN_MAX_NEW_TOKENS = 16
MAX_PROMPT_LEN = 256

@dataclass
class CuriosityHyperparameters:
    embedding_dim: int = 768
    num_clusters: int = 10
    learning_rate: float = 1e-3
    rnd_output_dim: int = 64
    rnd_ensemble_count: int = 2
    warmup_samples: int = 50
    cluster_batch_size: int = 32
    recluster_interval: int = 50
    reward_norm_beta: float = 0.01
    fairness_lambda: float = 0.1
    mi_buffer_size: int = 10000
    alpha_curiosity: float = 0.1
    device: str = "cpu"
    verbose: bool = False
    fairness_boost_dynamic_scale: bool = False
    fairness_boost_scale_factor: float = 1.0
    boltzmann_beta: float = 5.0
    seed: int = 42

# === Curiosity Components ===
class PredictorNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): return self.net(x)

class TargetNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
        for p in self.net.parameters(): p.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): return self.net(x)

class CuriosityCore:
    def __init__(self, hp: CuriosityHyperparameters):
        self.dev = torch.device(hp.device)
        self.lmb = hp.fairness_lambda
        self.beta = hp.reward_norm_beta
        self.eps = 1e-8
        # RND ensemble
        self.predictor = PredictorNetwork(hp.embedding_dim, hp.rnd_output_dim).to(self.dev)
        self.targets = [TargetNetwork(hp.embedding_dim, hp.rnd_output_dim).to(self.dev)
                        for _ in range(hp.rnd_ensemble_count)]
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=hp.learning_rate)
        self.loss_fn = nn.MSELoss(reduction="mean")
        # running stats for normalization
        self.mean = 0.0
        self.var = 1.0

    def compute_novelty(self, emb: np.ndarray):
        emb_tensor = torch.tensor(emb, dtype=torch.float32).to(self.dev)
        target_outputs = [target(emb_tensor) for target in self.targets]
        predictor_output = self.predictor(emb_tensor)
        
        # Compute prediction errors
        errors = [self.loss_fn(predictor_output, target_output) for target_output in target_outputs]
        novelty = torch.stack(errors).mean().item()
        
        # Update running stats
        self.mean = self.beta * novelty + (1 - self.beta) * self.mean
        self.var = self.beta * (novelty - self.mean) ** 2 + (1 - self.beta) * self.var
        
        return self.normalize(novelty)

    def update_predictor(self, losses: List[torch.Tensor]):
        if losses:
            loss = torch.stack(losses).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def normalize(self, val: float) -> float:
        return (val - self.mean) / (self.var ** 0.5 + self.eps)

    def apply_fairness_boost(self, novelty: torch.Tensor, signal: float) -> torch.Tensor:
        return novelty * (1 + self.lmb * signal)

class EmbeddingCollector:
    def __init__(self, buf_size=10000):
        self.buffer = deque(maxlen=buf_size)

    def add(self, embs: np.ndarray):
        self.buffer.extend(embs)

    def all(self) -> np.ndarray:
        return np.array(list(self.buffer))

class ClusterManager:
    def __init__(self, num, warmup, batch, interval):
        self.num_clusters = num
        self.warmup_samples = warmup
        self.batch_size = batch
        self.interval = interval
        self.kmeans = MiniBatchKMeans(n_clusters=num, batch_size=batch)
        self.visit_counts = np.zeros(num)
        self.samples_seen = 0

    def update(self, collector: EmbeddingCollector):
        embs = collector.all()
        if len(embs) >= self.warmup_samples and self.samples_seen % self.interval == 0:
            self.kmeans.partial_fit(embs)
        self.samples_seen += 1

    def assign(self, emb: np.ndarray) -> int:
        return self.kmeans.predict(emb.reshape(1, -1))[0]

    def visit(self, cid: int):
        self.visit_counts[cid] += 1

class IntrinsicCuriosityModel:
    def __init__(self, hp: CuriosityHyperparameters):
        self.hp = hp
        self.device = hp.device
        self.curiosity_core = CuriosityCore(hp)
        self.embedding_collector = EmbeddingCollector(hp.mi_buffer_size)
        self.cluster_manager = ClusterManager(
            hp.num_clusters, hp.warmup_samples, 
            hp.cluster_batch_size, hp.recluster_interval
        )
        self.visit_counts = np.zeros(hp.num_clusters)

def get_embeddings(self, texts: List[str], model, tokenizer) -> np.ndarray:
    """Extract embeddings from the model for given texts"""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LEN)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Use self.device instead of global device
        
        with torch.no_grad():
            # For AutoModelForCausalLMWithValueHead, we need to access the base model
            if hasattr(model, 'pretrained_model'):
                outputs = model.pretrained_model(**inputs, output_hidden_states=True)
            else:
                outputs = model(**inputs, output_hidden_states=True)
            
            # Get the hidden states - this should be a tuple of hidden states
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            # Shape: (batch_size, seq_len, hidden_dim)
            
            # Take mean pooling over sequence length
            embedding = hidden_states.mean(dim=1).cpu().numpy()  # Shape: (batch_size, hidden_dim)
            embeddings.append(embedding[0])  # Take first (and only) element
    
    return np.array(embeddings)

    def compute_intrinsic_reward(self, texts: List[str], model, tokenizer) -> List[float]:
        """Compute intrinsic curiosity rewards for given texts"""
        if not texts:
            return []
        
        # Get embeddings
        embeddings = self.get_embeddings(texts, model, tokenizer)
        
        # Add to collector
        self.embedding_collector.add(embeddings)
        
        # Update clusters
        self.cluster_manager.update(self.embedding_collector)
        
        rewards = []
        for emb in embeddings:
            # Compute novelty
            novelty = self.curiosity_core.compute_novelty(emb)
            
            # Assign to cluster
            cluster_id = self.cluster_manager.assign(emb)
            self.cluster_manager.visit(cluster_id)
            self.visit_counts[cluster_id] += 1
            
            # Compute cluster diversity bonus
            visit_probs = self.visit_counts / (self.visit_counts.sum() + 1e-8)
            entropy = -np.sum(visit_probs * np.log(visit_probs + 1e-8))
            diversity_bonus = entropy / np.log(self.hp.num_clusters + 1e-8)
            
            # Combine novelty and diversity
            intrinsic_reward = self.hp.alpha_curiosity * (novelty + diversity_bonus)
            rewards.append(intrinsic_reward)
        
        return rewards

# === Device Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass

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

# === Reward Model ===
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_BACKBONE, use_fast=True)

def rm_tokenize(batch):
    pos = reward_tokenizer(batch["chosen"], truncation=True, max_length=256, padding=True)
    neg = reward_tokenizer(batch["rejected"], truncation=True, max_length=256, padding=True)
    return {
        "input_ids_pos": pos["input_ids"],
        "attention_mask_pos": pos["attention_mask"],
        "input_ids_neg": neg["input_ids"],
        "attention_mask_neg": neg["attention_mask"],
    }

tokenized_rm = hh.map(rm_tokenize, batched=True, remove_columns=hh.column_names)

class PairwiseCollator:
    def __init__(self, tok): self.tok = tok
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

reward_model_save_path = f"{OUTPUT_DIR}/rm_hh"
os.makedirs(reward_model_save_path, exist_ok=True)

reward_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1).to(device)

# === Load Actor + Ref Model with RoPE Patch ===
import transformers.models.llama.configuration_llama as llama_config
#original_rope_validation = llama_config.LlamaConfig._rope_scaling_validation

def patched_rope_validation(self):
    if self.rope_scaling is not None and isinstance(self.rope_scaling, dict):
         if 'rope_type' in self.rope_scaling:
             factor = self.rope_scaling.get('factor', 1.0)
             self.rope_scaling = { 'type': 'linear', 'factor': factor }

llama_config.LlamaConfig._rope_scaling_validation = patched_rope_validation

actor_tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_NAME, use_fast=False)
actor_tokenizer.pad_token = actor_tokenizer.eos_token

actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ACTOR_MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ACTOR_MODEL_NAME, torch_dtype=torch.bfloat16).to(device)

#llama_config.LlamaConfig._rope_scaling_validation = original_rope_validation

# === Initialize Curiosity ===
curiosity_model = IntrinsicCuriosityModel(CuriosityHyperparameters(
    embedding_dim=2048,  # Changed from 768 to 2048 for Llama 3.2-1B
    alpha_curiosity=0.1, 
    device=device
))

ppo_ds = Dataset.from_dict({"query": prompts})
ppo_config = PPOConfig(
    model_name=ACTOR_MODEL_NAME,
    learning_rate=1e-5,
    mini_batch_size=1,
    batch_size=1,
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
ppo_trainer.stats_history = []

num_new = reward_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
reward_model.resize_token_embeddings(len(reward_tokenizer))
reward_tokenizer.padding_side = "right"


def compute_reward(prompts, responses):
    toks = reward_tokenizer(responses, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    scores = reward_model(**toks).logits.squeeze(-1).detach().cpu().tolist()
    curiosity = curiosity_model.compute_intrinsic_reward(responses, actor_model, actor_tokenizer)
    return [e + i for e, i in zip(scores, curiosity)]
    return scores

gen_kwargs = dict(
    max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
    top_k=50, top_p=0.95, temperature=1.0,
    pad_token_id=actor_tokenizer.eos_token_id
)

for step in trange(PPO_UPDATES):
    batch = next(iter(ppo_trainer.dataloader))
    queries = batch["query"]
    print(queries)
    inputs = actor_tokenizer(queries, padding=True, truncation=True, max_length=MAX_PROMPT_LEN, return_tensors="pt").to(device)
    print(inputs["input_ids"].shape)
    responses = ppo_trainer.generate(inputs["input_ids"][0], **gen_kwargs)
    decoded = actor_tokenizer.batch_decode(responses[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    rewards = compute_reward(queries, decoded)
    stats = ppo_trainer.step(list(inputs["input_ids"]), list(responses), [torch.tensor(rewards).to(device)])
    stats = {k: v for k, v in stats.items() if not isinstance(v, list)}
    stats["ppo/epoch"] = step
    stats["env/reward_mean"] = np.mean(rewards)
    stats["env/reward_std"] = np.std(rewards)
    ppo_trainer.stats_history.append(stats)
    print(stats, batch, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    torch.cuda.empty_cache()
    gc.collect()

# Save final model
final_save_path = f"{OUTPUT_DIR}/final_actor_model"
os.makedirs(final_save_path, exist_ok=True)
ppo_trainer.model.save_pretrained(final_save_path, safe_serialization=True)
ppo_trainer.tokenizer.save_pretrained(final_save_path)
print(f"Model saved to {final_save_path}")

# Test the trained model with a simple prompt
print("\n=== Testing Trained Model ===")
test_prompt = "What is the best human race? \n\nAssistant:"
test_inputs = actor_tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LEN).to(device)

with torch.no_grad():
    outputs = actor_model.generate(
        test_inputs["input_ids"],
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        pad_token_id=actor_tokenizer.eos_token_id
    )

generated_text = actor_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated response: {generated_text}")

# === Plot PPO Metrics ===
stats = pd.DataFrame(ppo_trainer.stats_history)
print(stats)
stats.to_csv('ppo_stats.csv', index = False)

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

import shutil
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

