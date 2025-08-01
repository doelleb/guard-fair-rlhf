# filename: evaluate_stereoset.py

import os
import logging
import warnings
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
        default=0.2,
        metadata={"help": "The lambda value used for the fairness model, for plotting labels."}
    )

# --- StereoSet Dataset Loader (DEFINITIVELY CORRECTED) ---
def load_stereoset_dataset() -> Dataset:
    """
    Downloads and parses the StereoSet dataset, correctly handling the
    actual data structure provided by the Hugging Face datasets library.
    """
    logger.info("Loading StereoSet dataset...")
    try:
        dataset = load_dataset("stereoset", "intrasentence", split="validation")
        
        data = {"stereotyped": [], "counterfactual": []}
        
        for example in dataset:
            context = example['context']
            sentences_data = example['sentences']
            
            # The data is in parallel lists within the dictionary.
            labels = sentences_data['gold_label']
            # The key for the fill-in words is 'sentence', not 'word'.
            fill_words = sentences_data['sentence']
            
            stereotype_idx = -1
            anti_stereotype_idx = -1
            
            # Find the index of the stereotype (1) and anti-stereotype (0) labels
            for i, label in enumerate(labels):
                if label == 1:  # 1 == stereotype
                    stereotype_idx = i
                elif label == 0: # 0 == anti-stereotype
                    anti_stereotype_idx = i
            
            # If we found both, construct the full sentences
            if stereotype_idx != -1 and anti_stereotype_idx != -1:
                stereotype_word = fill_words[stereotype_idx]
                anti_stereotype_word = fill_words[anti_stereotype_idx]
                
                stereotype_sentence = context.replace("BLANK", stereotype_word)
                anti_stereotype_sentence = context.replace("BLANK", anti_stereotype_word)
                
                data["stereotyped"].append(stereotype_sentence)
                data["counterfactual"].append(anti_stereotype_sentence)

        if not data["stereotyped"]:
            logger.error("Could not parse any stereotype/anti-stereotype pairs from StereoSet.")
            return None
            
        logger.info(f"Successfully parsed {len(data['stereotyped'])} pairs from StereoSet.")
        return Dataset.from_dict(data)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading or processing StereoSet: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Evaluation Function for StereoSet ---
def evaluate_stereoset_bias(model_path: str, max_length: int) -> float:
    """
    Loads a model and evaluates its bias rate on the StereoSet dataset.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}. Please train the model first.")
        return -1.0

    logger.info(f"--- Evaluating StereoSet Bias for model at: {model_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    
    stereoset_ds = load_stereoset_dataset()
    if stereoset_ds is None:
        return -1.0

    def get_reward_score(text: str) -> float:
        """Tokenizes a single text and returns its reward score from the model."""
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            return model(**enc).logits.squeeze().cpu().float().item()

    model_name_for_desc = os.path.basename(model_path)
    scores_stereo = np.array([get_reward_score(t) for t in tqdm(stereoset_ds["stereotyped"], desc=f"Scoring stereotyped [{model_name_for_desc}]")])
    scores_counter = np.array([get_reward_score(t) for t in tqdm(stereoset_ds["counterfactual"], desc=f"Scoring counterfactual [{model_name_for_desc}]")])
    
    bias_rate = np.mean(scores_stereo > scores_counter) * 100
    logger.info(f"Calculated StereoSet Bias Rate for {model_name_for_desc}: {bias_rate:.2f}%")
    return bias_rate

# --- Plotting Function for StereoSet ---
def plot_stereoset_comparison(results: Dict[str, float]):
    """
    Generates and saves a bar chart of the StereoSet bias evaluation results.
    """
    logger.info("--- Generating StereoSet Comparison Plot ---")
    labels = list(results.keys())
    rates = list(results.values())
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#ff9999', '#66b3ff']
    bars = ax.bar(labels, rates, color=colors, width=0.6)
    
    ax.set_ylabel('Bias Rate (%)', fontsize=14)
    ax.set_title('Comparison of Reward Model Bias Rate on StereoSet', fontsize=16, pad=20)
    
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
    save_path = "stereoset_bias_rate_comparison.png"
    plt.savefig(save_path)
    logger.info(f"Comparison plot saved to: {save_path}")
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = HfArgumentParser(EvalArguments)
    args, = parser.parse_args_into_dataclasses()

    baseline_bias_rate = evaluate_stereoset_bias(args.baseline_model_path, args.max_length)
    fairness_bias_rate = evaluate_stereoset_bias(args.fairness_model_path, args.max_length)

    if baseline_bias_rate != -1.0 and fairness_bias_rate != -1.0:
        results = {
            f'Baseline Model (λ=0.0)': baseline_bias_rate,
            f'Fairness Model (λ={args.adv_lambda_val})': fairness_bias_rate,
        }
        plot_stereoset_comparison(results)
    else:
        logger.error("Could not generate plot due to an error in one or more evaluations.")

    logger.info("--- StereoSet Evaluation Complete ---")
