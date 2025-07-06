# is our reward model from train_reward_model_base.py actually unbiased? 


# TODO: 
# 1. load the reward model 
# 2. load the dataset 
# 3. test the reward model on the dataset 
# take the hh-rlhf dataset, take 100 helpful examples, 100 harmless examples, 
# run reward model on each of the examples and plot the helpful distribution and plot the harmless distribution of rewards 
# ideally, for base model, these are going to be very different because model is biased 
# then, once we add our fairness, we should see that the distributions are much closer 
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
reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
reward_model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ds    = load_dataset("Dahoas/full-hh-rlhf", split="train")
pairs = ds.shuffle(seed=42).select(range(NUM_SAMPLES))

def get_reward_score(text: str) -> float:
    toks = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = reward_model(**toks).logits
    return logits.squeeze().cpu().item()

chosen_scores   = np.array([get_reward_score(ex["chosen"])   for ex in pairs])
rejected_scores = np.array([get_reward_score(ex["rejected"]) for ex in pairs])

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

raw_min = min(chosen_scores.min(), rejected_scores.min())
raw_max = max(chosen_scores.max(), rejected_scores.max())
span    = raw_max - raw_min
xmin = raw_min - 0.05 * span
xmax = raw_max + 0.05 * span
x = np.linspace(xmin, xmax, 200)

ax = axes[0]
k1 = gaussian_kde(chosen_scores)
k2 = gaussian_kde(rejected_scores)
ax.plot(x, k1(x), label="Chosen (better)", linewidth=1.5)
ax.fill_between(x, k1(x), alpha=0.3)
ax.plot(x, k2(x), label="Rejected (worse)", linewidth=1.5)
ax.fill_between(x, k2(x), alpha=0.3)
ax.set_title("(a) DistilBERT rewards on HH-RLHF (zoomed)")
ax.set_xlabel("Reward score")
ax.set_ylabel("Density")
ax.set_xlim(xmin, xmax)
ax.legend()

ax = axes[1]
ax.hist(chosen_scores,   bins=20, density=True, alpha=0.6, label="Chosen",   edgecolor="white", range=(xmin, xmax))
ax.hist(rejected_scores, bins=20, density=True, alpha=0.6, label="Rejected", edgecolor="white", range=(xmin, xmax))
ax.set_title("(b) Histogram view (zoomed)")
ax.set_xlabel("Reward score")
ax.legend()

plt.tight_layout()
plt.show()
