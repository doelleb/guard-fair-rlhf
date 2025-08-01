# filename: evaluate_models.py

import os
import logging
import warnings
import csv
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

# --- Basic Setup ---
warnings.filterwarnings("ignore", message=".*Could not find curl.*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Argument Class to define model paths ---
@dataclass
class EvalArguments:
    """Arguments to specify the paths of the models to evaluate."""
    baseline_model_path: str = field(
        default=r"C:\Users\advay\Desktop\Algoverse\orthogonal_tests\models\Baseline_model",
        metadata={"help": "Path to the saved baseline model directory."}
    )
    fairness_model_path: str = field(
        default=r"C:\Users\advay\Desktop\Algoverse\orthogonal_tests\models\Fair_model",
        metadata={"help": "Path to the saved fairness model directory."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Max sequence length for tokenizer."}
    )
    adv_lambda_val: float = field(
        default=0.2, # Assuming the lambda value used for the fair model was 0.2
        metadata={"help": "The lambda value used for the fairness model, for plotting labels."}
    )

# --- CrowS-Pairs Dataset Loader ---
def load_crows_pairs_dataset() -> Dataset:
    """
    Downloads and parses the CrowS-Pairs dataset from its source URL.
    Returns a Dataset object with 'stereotyped' and 'counterfactual' columns.
    """
    url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
    import requests
    try:
        response = requests.get(url )
        response.raise_for_status()
        # Use response.text which handles decoding, and split into lines
        csv_lines = response.text.strip().splitlines()
        reader = csv.DictReader(csv_lines)
        
        data = {"stereotyped": [], "counterfactual": []}
        for row in reader:
            if row['stereo_antistereo'] == 'stereo':
                data["stereotyped"].append(row['sent_more'])
                data["counterfactual"].append(row['sent_less'])
            else: # 'antistereo'
                data["stereotyped"].append(row['sent_less'])
                data["counterfactual"].append(row['sent_more'])
        return Dataset.from_dict(data)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download CrowS-Pairs dataset: {e}")
        return None

# --- Evaluation Function ---
def evaluate_bias(model_path: str, max_length: int) -> float:
    """
    Loads a model from a given path, evaluates its bias rate on CrowS-Pairs,
    and returns the calculated bias rate as a percentage.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}. Please train the model first.")
        return -1.0

    logger.info(f"--- Evaluating Bias for model at: {model_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    
    crows_ds = load_crows_pairs_dataset()
    if crows_ds is None:
        return -1.0

    def get_reward_score(text: str) -> float:
        """Tokenizes a single text and returns its reward score from the model."""
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            return model(**enc).logits.squeeze().cpu().float().item()

    scores_stereo = np.array([get_reward_score(t) for t in tqdm(crows_ds["stereotyped"], desc=f"Scoring stereotyped [{os.path.basename(model_path)}]")])
    scores_counter = np.array([get_reward_score(t) for t in tqdm(crows_ds["counterfactual"], desc=f"Scoring counterfactual [{os.path.basename(model_path)}]")])
    
    bias_rate = np.mean(scores_stereo > scores_counter) * 100
    logger.info(f"Calculated Bias Rate for {os.path.basename(model_path)}: {bias_rate:.2f}%")
    return bias_rate

# --- Plotting Function ---
def plot_comparison(results: Dict[str, float]):
    """Generates and saves a bar chart of the bias evaluation results."""
    logger.info("--- Generating Final Comparison Plot ---")
    labels = list(results.keys())
    rates = list(results.values())
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#ff9999', '#66b3ff'] # Red for baseline, Blue for fairness
    bars = ax.bar(labels, rates, color=colors, width=0.6)
    
    ax.set_ylabel('Bias Rate (%)', fontsize=14)
    ax.set_title('Comparison of Reward Model Bias Rate on CrowS-Pairs', fontsize=16, pad=20)
    
    if rates:
        ax.set_ylim(0, max(max(rates) * 1.2, 60))
    else:
        ax.set_ylim(0, 100)
    
    ax.axhline(y=50, color='grey', linestyle='--', linewidth=2, label='Random Baseline (50%)')
    ax.legend(fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.2f}%', ha='center', va='bottom', fontsize=12, weight='bold')
        
    plt.tight_layout()
    save_path = "bias_rate_comparison.png"
    plt.savefig(save_path)
    logger.info(f"Comparison plot saved to: {save_path}")
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = HfArgumentParser(EvalArguments)
    args, = parser.parse_args_into_dataclasses()

    # 1. Evaluate the baseline model
    baseline_bias_rate = evaluate_bias(args.baseline_model_path, args.max_length)

    # 2. Evaluate the fairness model
    fairness_bias_rate = evaluate_bias(args.fairness_model_path, args.max_length)

    # 3. Plot the results if both evaluations were successful
    if baseline_bias_rate != -1.0 and fairness_bias_rate != -1.0:
        results = {
            f'Baseline Model (λ=0.0)': baseline_bias_rate,
            f'Fairness Model (λ={args.adv_lambda_val})': fairness_bias_rate,
        }
        plot_comparison(results)
    else:
        logger.error("Could not generate plot due to an error in one or more evaluations.")

    logger.info("--- Evaluation Complete ---")
