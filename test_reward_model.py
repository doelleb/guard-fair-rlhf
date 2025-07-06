# is our reward model from train_reward_model_base.py actually unbiased? 


# TODO: 
# 1. load the reward model 
# 2. load the dataset 
# 3. test the reward model on the dataset 
# take the hh-rlhf dataset, take 100 helpful examples, 100 harmless examples, 
# run reward model on each of the examples and plot the helpful distribution and plot the harmless distribution of rewards 
# ideally, for base model, these are going to be very different because model is biased 
# then, once we add our fairness, we should see that the distributions are much closer 

#!pip install --upgrade datasets
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# To use own pretrained reward model, uncomment and set the path:
# MODEL_NAME = "/path/to/your/reward-model"
MODEL_NAME  = "distilbert-base-uncased"
NUM_SAMPLES = 100
MAX_LENGTH  = 512


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_model = (
    AutoModelForSequenceClassification
    .from_pretrained(MODEL_NAME, num_labels=1)
    .to(device)
)
reward_model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ds    = load_dataset("Dahoas/full-hh-rlhf", split="train")
pairs = ds.shuffle(seed=42).select(range(NUM_SAMPLES))

def get_reward_score(text: str) -> float:
    toks = tokenizer(
        text,
        truncation=True,
        max_length=min(MAX_LENGTH, tokenizer.model_max_length),
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        return reward_model(**toks).logits.squeeze().cpu().item()

helpful_scores  = np.array([get_reward_score(ex["chosen"])  for ex in pairs])
harmless_scores = np.array([get_reward_score(ex["rejected"]) for ex in pairs])

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

sns.kdeplot(
    helpful_scores,
    label="Helpful",
    fill=True,
    alpha=0.5,
    linewidth=2
)
sns.kdeplot(
    harmless_scores,
    label="Harmless",
    fill=True,
    alpha=0.5,
    linewidth=2
)

plt.xlabel("Reward score")
plt.ylabel("Density")
plt.title("Reward Distribution: Helpful vs. Harmless")
plt.legend()
plt.tight_layout()
plt.show()
