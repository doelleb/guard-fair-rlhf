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
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt

# ---- CONFIG ----
MODEL_NAME = "distilbert-base-uncased"
NUM_SAMPLES = 100
MAX_LENGTH = 512  # tokenizer max length

# ---- 1. Load the reward model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1
)
reward_model.to(device)
reward_model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---- 2. Load the dataset ----
ds = load_dataset("Dahoas/full-hh-rlhf", split="train", cache_dir=None)
helpful_ds = ds.filter(lambda ex: ex["label"] == 1).shuffle(seed=42).select(range(NUM_SAMPLES))
harmless_ds = ds.filter(lambda ex: ex["label"] == 0).shuffle(seed=42).select(range(NUM_SAMPLES))

# ---- 3. Score function ----
def get_reward_score(text: str) -> float:
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # for num_labels=1, logits is shape [1,1]
        score = outputs.logits.squeeze().cpu().item()
    return score

# ---- 4. Compute scores ----
helpful_scores = [get_reward_score(ex["text"]) for ex in helpful_ds]
harmless_scores = [get_reward_score(ex["text"]) for ex in harmless_ds]

# ---- 5. Plot distributions ----
plt.figure()
plt.hist(helpful_scores, bins=20)
plt.title("Reward Distribution on Helpful Examples")
plt.xlabel("Reward Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(harmless_scores, bins=20)
plt.title("Reward Distribution on Harmless Examples")
plt.xlabel("Reward Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
