# pip install transformers==4.41.1 accelerate==0.30.1 peft==0.11.1 bitsandbytes==0.43.0 datasets==2.19.1 wandb trl==0.8.1

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
import wandb

# Initialize Weights & Biases
wandb.init(project="fair-rlhf-llama3.2", name="baseline_rlhf_llama3.2")

# ===== CONFIGURATION =====
config = PPOConfig(
    model_name="meta-llama/Llama-3.2-1B",  # Replace w/ LLaMA 3.2 1B if available
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    log_with="wandb",
    seed=42
)

set_seed(config.seed)

# ===== LOAD MODEL & TOKENIZER =====
tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Patch the config before loading
model_config = AutoConfig.from_pretrained(config.model_name)

# Force rope_scaling to be compatible
model_config.rope_scaling = {
    "type": "linear",
    "factor": 2.0  # You can adjust the factor if needed
}

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    config=model_config,
    torch_dtype=torch.float16,
    device_map="auto"
)


# ===== REWARD MODEL MOCKUP (replace later with actual reward model) =====
def reward_fn(sample, response):
    """ Placeholder reward model function. Replace with fairness-aware reward model. """
    text = sample + response
    return torch.tensor([len(response) / 100.0])  # proxy reward: encourage longer responses

# ===== MUTUAL INFORMATION REGULARIZER PLACEHOLDER =====
def mutual_information_penalty(samples, rewards, categories):
    """ 
    Placeholder MI penalty. In practice, estimate I(r ; c) using MINE or kernel-based estimator.
    """
    # TODO: Replace with real MI estimator.
    penalty = torch.zeros_like(rewards)
    return penalty

# ===== DATASET =====
dataset = load_dataset("HuggingFaceH4/hh-rlhf", split="train[:100]")

# Format data as (prompt, category)
prompts = [{"prompt": d["chosen"], "category": "helpfulness"} for d in dataset]

# ===== PPO TRAINER =====
ppo_trainer = PPOTrainer(config, model, tokenizer)

# ===== TRAINING LOOP =====
for epoch in range(2):  # Try 2 epochs for demo
    for sample in prompts:
        prompt = sample["prompt"]
        category = sample["category"]

        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(model.device)

        # Generate
        response = ppo_trainer.generate(input_ids, max_new_tokens=64)
        response_text = tokenizer.decode(response[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # Reward
        reward = reward_fn(prompt, response_text)
        mi_penalty = mutual_information_penalty(prompt, reward, category)
        fair_reward = reward - mi_penalty  # Apply MI fairness penalty

        # PPO Step
        ppo_trainer.step([prompt], [response_text], fair_reward)

        # Logging
        wandb.log({
            "reward_raw": reward.item(),
            "reward_fair": fair_reward.item(),
            "epoch": epoch
        })

        print(f"[Epoch {epoch}] Prompt: {prompt[:50]}... | Response: {response_text[:50]}... | Reward: {reward.item():.3f}")

# ===== SAVE MODEL =====
output_dir = "llama3.2_fair_rlhf"
ppo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
