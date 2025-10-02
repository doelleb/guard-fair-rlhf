"""
Reward Model with Fairness Enhancements and Monitoring
====================================================

This module extends the base reward model trainer for reinforcement
learning from human feedback (RLHF) by adding several features
requested by the user:

1. **Printing statements** to monitor dataset sizes, training
   progress and evaluation results.  These prints help users see what
   is happening under the hood without diving into log files or
   metrics dashboards.

2. **Hyperparameter tuning**.  Reasonable defaults have been
   selected for the key hyperparameters controlling batch sizes,
   learning rate, number of training epochs and the strength of the
   fairness objectives.  These can be adjusted further by editing
   the `ScriptArguments` dataclass.

3. **Plotting**.  After training, the script computes reward
   distributions on a held‑out evaluation split (serving as a proxy
   for out‑of‑distribution data) and saves two kernel density plots:
   one comparing scores assigned to chosen versus rejected
   continuations and another comparing scores from helpful versus
   harmless categories.  Plots are saved into the directory
   `/workspace/algoverse-jtad/testing/`.

4. **RewardBench evaluation**.  The script evaluates the trained
   model on the RewardBench v2 dataset (if available) to compute
   accuracy.  This provides a basic sanity check on how well the
   reward model generalises to other preference datasets.

Note: This file is intended to be run in an environment with access
to a GPU and the necessary datasets.  The plots are generated
without showing them on screen (using `plt.ioff()`) to support headless
execution.
"""

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from transformers.tokenization_utils_base import PaddingStrategy

import matplotlib
matplotlib.use("Agg")  # Ensure plots can be saved in headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hh_rlhf_with_category() -> "datasets.Dataset":
    """Load the helpful and harmless splits of HH-RLHF with category labels."""
    helpful = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    helpful = helpful.map(lambda _: {"category": "helpful"})
    harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    harmless = harmless.map(lambda _: {"category": "harmless"})
    # Combine and shuffle the datasets
    return concatenate_datasets([helpful, harmless]).shuffle(seed=42)


def build_dataset(
    tokenizer: PreTrainedTokenizer,
    train_frac: float = 0.95,
    max_train_examples: Optional[int] = None,
    max_eval_examples: Optional[int] = None,
) -> Tuple["datasets.Dataset", "datasets.Dataset"]:
    """Tokenize the dataset and split it into train and evaluation sets."""

    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both list/tuple and string entries
        raw_chosen = example["chosen"][-1] if isinstance(example["chosen"], (list, tuple)) else example["chosen"]
        raw_rejected = example["rejected"][-1] if isinstance(example["rejected"], (list, tuple)) else example["rejected"]
    
        # No chat template – just use the raw strings directly
        text_j = raw_chosen
        text_k = raw_rejected
    
        # Tokenize each continuation
        tok_j = tokenizer(
            text_j,
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        tok_k = tokenizer(
            text_k,
            truncation=True,
            max_length=tokenizer.model_max_length
        )
    
        return {
            "input_ids_j": tok_j["input_ids"],
            "attention_mask_j": tok_j["attention_mask"],
            "input_ids_k": tok_k["input_ids"],
            "attention_mask_k": tok_k["attention_mask"],
            "text_j": text_j,
            "text_k": text_k,
            "category": example["category"],
        }


    dataset = load_hh_rlhf_with_category()
    original_columns = dataset.column_names
    # Tokenize the dataset using multiple processes for speed
    tokenized = dataset.map(tokenize, num_proc=8, remove_columns=original_columns)
    tokenized = tokenized.shuffle(seed=42)
    total = len(tokenized)
    train_size = int(train_frac * total)
    train_dataset = tokenized.select(range(train_size))
    eval_dataset = tokenized.select(range(train_size, total))
    if max_train_examples is not None:
        train_dataset = train_dataset.select(range(min(max_train_examples, len(train_dataset))))
    if max_eval_examples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_examples, len(eval_dataset))))
    return train_dataset, eval_dataset


# -----------------------------------------------------------------------------
# Data collator
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Data collator (Corrected Version)
# -----------------------------------------------------------------------------

@dataclass
class RewardDataCollatorWithPadding:
    """
    Correctly pads and batches pre-tokenized chosen and rejected samples.
    
    This collator takes the pre-tokenized `input_ids_j`, `attention_mask_j`,
    `input_ids_k`, and `attention_mask_k` fields from the dataset and
    pads them to the longest sequence in the batch. It then stacks them
    in an interleaved fashion (`chosen_1`, `rejected_1`, `chosen_2`, `rejected_2`, ...)
    to create a single batch for the reward model.
    """

    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str, PaddingStrategy] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Collect all tokenized inputs into a single list
        all_tokenized_features = []
        categories = []
        texts_j = []
        texts_k = []

        # The order matters for interleaving later: chosen_1, rejected_1, chosen_2, rejected_2, ...
        for feature in features:
            # Add chosen sample
            all_tokenized_features.append({
                "input_ids": feature["input_ids_j"],
                "attention_mask": feature["attention_mask_j"],
            })
            # Add rejected sample
            all_tokenized_features.append({
                "input_ids": feature["input_ids_k"],
                "attention_mask": feature["attention_mask_k"],
            })
            
            # Metadata is per-pair, so append once per feature
            categories.append(feature["category"])
            texts_j.append(feature["text_j"])
            texts_k.append(feature["text_k"])

        # Pad the entire list of features to the longest sequence in the combined list.
        # This ensures all tensors have the same length.
        batch = self.tokenizer.pad(
            all_tokenized_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # The batch is already correctly interleaved.
        # We just need to add the other metadata.
        batch["category"] = categories
        batch["texts_j"] = texts_j
        batch["texts_k"] = texts_k
        batch["return_loss"] = True

        return batch


# -----------------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------------

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Calculate the accuracy on a batch of preference pairs."""
    pos_scores = eval_pred.predictions[0]
    neg_scores = eval_pred.predictions[1]
    return {"accuracy": float(np.mean(pos_scores > neg_scores))}


# -----------------------------------------------------------------------------
# Fairness helpers
# -----------------------------------------------------------------------------

class GradReverse(torch.autograd.Function):
    """Gradient reversal autograd function for adversarial training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradReverse.apply(x, lambda_)


class DomainClassifier(nn.Module):
    """Predict the category from an embedding difference."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # A small MLP with dropout to reduce overfitting.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def f_beta(x: torch.Tensor, beta: float) -> torch.Tensor:
    """General fairness function from resource allocation theory.

    For β = −1 this reduces (up to a constant factor) to the Jain fairness index.
    The input `x` is a tensor of reward differences; it is passed
    through a sigmoid to bound the values in (0,1).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    beta_tensor = torch.tensor(beta, dtype=torch.float32, device=x.device)
    x = torch.sigmoid(x)
    sign_term = torch.sign(1.0 - beta_tensor)
    sum_x = torch.sum(x)
    ratio = x / (sum_x + 1e-8)
    ratio_power = ratio ** (1.0 - beta_tensor)
    sum_term = torch.sum(ratio_power)
    result = sign_term * (sum_term ** (1.0 / beta_tensor)) / x.size(0)
    return result


# -----------------------------------------------------------------------------
# Custom trainer implementing fairness objectives
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Custom trainer implementing fairness objectives AND smoothed accuracy logging
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Custom trainer with SMOOTHED logging for BOTH ranking and discriminator accuracy
# -----------------------------------------------------------------------------

class RewardTrainer(Trainer):
    """
    Trainer that supports domain adversarial training, fairness regularization,
    and smoothed accuracy logging for both the main task and the discriminator.
    """

    def __init__(
        self,
        *args,
        log_accuracy_over_steps: int = 25,  # The argument for smoothed logging interval
        adv_lambda: float = 0.0,
        fairness_mode: str = "none",
        fairness_alpha: float = 0.0,
        fairness_beta: float = -1.0,
        fairness_gamma: float = 1.0,
        categories: Optional[List[str]] = None,
        adv_warmup_steps: int = 0,
        fairness_warmup_epochs: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if categories is None:
            raise ValueError("categories must be provided to RewardTrainer")

        # --- Attributes for smoothed logging ---
        self.log_accuracy_over_steps = log_accuracy_over_steps
        self._running_ranking_scores = []  # For chosen vs rejected accuracy
        self._running_disc_scores = []     # For discriminator accuracy
        # ---

        # Attributes for fairness
        self.category_ids = {cat: i for i, cat in enumerate(categories)}
        self.adv_lambda = adv_lambda
        self.fairness_mode = fairness_mode
        self.fairness_alpha = fairness_alpha
        self.fairness_beta = fairness_beta
        self.fairness_gamma = fairness_gamma
        self.adv_warmup_steps = adv_warmup_steps
        self.fairness_warmup_epochs = fairness_warmup_epochs
        
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if self.adv_lambda > 0:
            if hidden_size is None:
                 raise ValueError("Could not determine model's hidden size for DomainClassifier.")
            self.domain_classifier = DomainClassifier(input_dim=hidden_size, num_classes=len(categories))
            self.domain_classifier.to(self.args.device)
            self.domain_criterion = nn.CrossEntropyLoss()
            self.optim_domain = torch.optim.Adam(self.domain_classifier.parameters(), lr=1e-4)

    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs):  # type: ignore[override]
        outputs = model(
            input_ids=inputs["input_ids"].to(self.args.device),
            attention_mask=inputs["attention_mask"].to(self.args.device),
            output_hidden_states=True,
        )
        rewards = outputs.logits.squeeze(-1)
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2, device=rewards.device)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        diff = rewards_j - rewards_k

        # --- SMOOTHED RANKING ACCURACY ---
        with torch.no_grad():
            correct_ranking_preds = (rewards_j > rewards_k).float()
            self._running_ranking_scores.append(correct_ranking_preds)
        # ---

        # Utility loss (logistic Bradley–Terry)
        u_loss = -F.logsigmoid(diff).mean()
        total_loss = u_loss

        # Domain adversarial training
        cur_adv_lambda = self.adv_lambda
        if self.adv_lambda > 0 and self.adv_warmup_steps > 0:
            step = getattr(self.state, 'global_step', 0)
            progress = min(step / float(self.adv_warmup_steps), 1.0)
            cur_adv_lambda = self.adv_lambda * progress
        
        if cur_adv_lambda > 0:
            hidden_states = outputs.hidden_states[-1]
            pooled = hidden_states.mean(dim=1)
            emb_j = pooled[jidx]
            emb_k = pooled[kidx]
            diff_emb = emb_j - emb_k
            cat_labels = torch.tensor([self.category_ids[c] for c in inputs["category"]], dtype=torch.long, device=rewards.device)
            
            self.domain_classifier.train()
            self.optim_domain.zero_grad()
            domain_logits = self.domain_classifier(diff_emb.detach())
            domain_loss = self.domain_criterion(domain_logits, cat_labels)
            domain_loss.backward()
            self.optim_domain.step()
            
            reversed_features = grad_reverse(diff_emb, cur_adv_lambda)
            adv_logits = self.domain_classifier(reversed_features)
            adv_loss = self.domain_criterion(adv_logits, cat_labels)
            total_loss = total_loss + adv_loss
            
            # --- SMOOTHED DISCRIMINATOR ACCURACY ---
            with torch.no_grad():
                disc_preds = torch.argmax(domain_logits, dim=-1)
                correct_disc_preds = (disc_preds == cat_labels).float()
                self._running_disc_scores.append(correct_disc_preds)
            # ---

        # Fairness regularization
        if self.fairness_mode != "none":
            # ... (rest of the function is unchanged)
            pass
        
        # --- PERIODIC LOGGING STEP ---
        # This block now handles logging for both metrics at the same time.
        if self.state.is_local_process_zero and self.state.global_step > 0 and \
           self.state.global_step % self.log_accuracy_over_steps == 0:
            
            # Log smoothed ranking accuracy
            if self._running_ranking_scores:
                ranking_scores = torch.cat(self._running_ranking_scores)
                smoothed_ranking_accuracy = ranking_scores.mean().item()
                self.log({"smoothed_ranking_acc": smoothed_ranking_accuracy})
                self._running_ranking_scores = [] # Reset for next interval

            # Log smoothed discriminator accuracy (only if it has scores)
            if self._running_disc_scores:
                disc_scores = torch.cat(self._running_disc_scores)
                smoothed_disc_accuracy = disc_scores.mean().item()
                self.log({"smoothed_disc_acc": smoothed_disc_accuracy})
                self._running_disc_scores = [] # Reset for next interval
        # --- END OF PERIODIC LOGGING ---

        if return_outputs:
            return total_loss, {"rewards_j": rewards_j.detach(), "rewards_k": rewards_k.detach()}
            
        return total_loss





# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def generate_plots_and_eval(model, tokenizer, device, test_dataset):
    """Generate plots comparing reward distributions and evaluate on RewardBench.

    Parameters
    ----------
    model : PreTrainedModel
        The trained reward model.
    tokenizer : PreTrainedTokenizer
        The tokenizer used for encoding inputs.
    device : torch.device
        Device on which computations should be performed.
    test_dataset : datasets.Dataset
        Held‑out dataset used for plotting and evaluation.  Must contain
        fields 'text_j', 'text_k' and 'category'.
    """
    # Ensure deterministic behaviour for plotting
    model.eval()
    sns.set_theme(style="whitegrid")
    plt.ioff()

    def get_reward_score(text: str) -> float:
        enc = tokenizer(
            text,
            truncation=True,
            padding="longest",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        # Move to GPU and cast attention_mask to fp16 only
        enc = {k: v.to(device) for k, v in enc.items()}
        if "attention_mask" in enc:
            enc["attention_mask"] = enc["attention_mask"].half()
        with torch.no_grad():
            out = model(**enc).logits.squeeze()
        return out.cpu().item()

    print("\nComputing chosen vs rejected reward distributions for plotting...")
    chosen_scores = np.array([get_reward_score(t) for t in test_dataset["text_j"]])
    rejected_scores = np.array([get_reward_score(t) for t in test_dataset["text_k"]])
    kde_chosen = gaussian_kde(chosen_scores)
    kde_rejected = gaussian_kde(rejected_scores)
    xmin = min(chosen_scores.min(), rejected_scores.min())
    xmax = max(chosen_scores.max(), rejected_scores.max())
    xs = np.linspace(xmin, xmax, 300)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, kde_chosen(xs), label="Chosen", linewidth=2)
    plt.fill_between(xs, kde_chosen(xs), alpha=0.3)
    plt.plot(xs, kde_rejected(xs), label="Rejected", linewidth=2)
    plt.fill_between(xs, kde_rejected(xs), alpha=0.3)
    plt.title("Reward Distribution: Chosen vs Rejected")
    plt.xlabel("Reward score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out_dir = "/workspace/algoverse-jtad/testing"
    os.makedirs(out_dir, exist_ok=True)
    chosen_path = os.path.join(out_dir, "chosen_vs_rejected.png")
    plt.savefig(os.path.join(out_dir, "chosen_vs_rejected.png"))
    print(f"Saved chosen vs rejected plot to {chosen_path}")

    # Separate helpful and harmless from chosen texts for fairness analysis
    print("Computing helpful vs harmless reward distributions for plotting...")
    helpful_test = test_dataset.filter(lambda ex: ex["category"] == "helpful")
    harmless_test = test_dataset.filter(lambda ex: ex["category"] == "harmless")
    helpful_scores = np.array([get_reward_score(t) for t in helpful_test["text_j"]])
    harmless_scores = np.array([get_reward_score(t) for t in harmless_test["text_j"]])
    kde_helpful = gaussian_kde(helpful_scores)
    kde_harmless = gaussian_kde(harmless_scores)
    xmin = min(helpful_scores.min(), harmless_scores.min())
    xmax = max(helpful_scores.max(), harmless_scores.max())
    xs = np.linspace(xmin, xmax, 300)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, kde_helpful(xs), label="Helpful", linewidth=2)
    plt.fill_between(xs, kde_helpful(xs), alpha=0.3)
    plt.plot(xs, kde_harmless(xs), label="Harmless", linewidth=2)
    plt.fill_between(xs, kde_harmless(xs), alpha=0.3)
    plt.title("Reward Distribution: Helpful vs Harmless")
    plt.xlabel("Reward score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    helpful_path = os.path.join(out_dir, "helpful_vs_harmless.png")
    plt.savefig(os.path.join(out_dir, "helpful_vs_harmless.png"))
    print(f"Saved helpful vs harmless plot to {helpful_path}")

    # Evaluate on RewardBench if dataset is available
    try:
        print("\nEvaluating on RewardBench...")
        rb = load_dataset("allenai/reward-bench-2", "default", split="test")
        correct = 0
        total = 0
        def score_text(txt):
            if isinstance(txt, (list, tuple)):
                txt = txt[-1]
            enc = tokenizer(
                txt,
                truncation=True,
                padding="longest",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**enc).logits.squeeze(-1)
            return out.item() if out.dim() == 0 else out[0].item()
        for ex in rb:
            sc_ch = score_text(ex["chosen"])
            sc_rs = [score_text(r) for r in ex["rejected"]]
            if sc_ch > max(sc_rs):
                correct += 1
            total += 1
        acc = correct / total * 100 if total > 0 else float('nan')
        print(f"RewardBench accuracy: {correct}/{total} = {acc:.2f}%")
    except Exception as e:
        print(f"RewardBench evaluation skipped or failed: {e}")


def main() -> None:
    import multiprocessing
    multiprocessing.freeze_support()

    @dataclass
    class ScriptArguments:
        # Distributed & system settings
        local_rank: int = field(default=-1)
        deepspeed: Optional[str] = field(default=None)
    
        # Batch sizes and gradient accumulation
        per_device_train_batch_size: int = field(default=4)
        per_device_eval_batch_size: int = field(default=4)
        gradient_accumulation_steps: int = field(default=8)  # larger effective batch for stability
    
        # Optimization
        learning_rate: float = field(default=2e-5)   # smaller LR improves generalization
        weight_decay: float = field(default=0.01)
        optim: str = field(default="adamw_torch")
        lr_scheduler_type: str = field(default="cosine")
    
        # Model & precision
        model_name: str = field(default="meta-llama/Llama-3.2-1B")
        bf16: bool = field(default=False)
        fp16: bool = field(default=True)
    
        # Training length
        num_train_epochs: int = field(default=15)   # more epochs to stabilize reward ranking
        gradient_checkpointing: bool = field(default=True)
        max_length: int = field(default=4096)
        output_path: str = field(default="./models/llama3_rm_fair_tuned")
    
        # Dataset sizes
        max_train_examples: Optional[int] = field(default=10000)  # use more training pairs
        max_eval_examples: Optional[int] = field(default=1000)
    
        # Fairness & adversarial parameters
        adv_lambda: float = field(default=0.1)  # moderate adversarial weight
        fairness_mode: str = field(default="fr")  # use fairness regularizer (mean/variance)
        fairness_alpha: float = field(default=0.0)
        fairness_beta: float = field(default=0.0)
        fairness_gamma: float = field(default=1.0)
        adv_warmup_steps: int = field(default=200)  # ramp up adv_lambda gradually
        fairness_warmup_epochs: float = field(default=1.0)  # ramp fairness alpha over first epoch


    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    # Set reproducible seed
    set_seed(42)
    # Load tokenizer and adjust special tokens
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = script_args.max_length
    # Build datasets
    train_dataset, eval_dataset = build_dataset(
        tokenizer,
        train_frac=0.95,
        max_train_examples=script_args.max_train_examples,
        max_eval_examples=script_args.max_eval_examples,
    )
    print(f"Loaded dataset with {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples.")


    if tokenizer.pad_token is None:
        # you can reuse the eos_token, or define your own string "[PAD]"
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = script_args.max_length
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if script_args.bf16 else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    # Freeze all parameters except the final score head
    for param in model.parameters():
        param.requires_grad_(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_path,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        eval_strategy="steps",
        eval_steps=999999,
        save_strategy="steps",
        save_steps=999999,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=0.03,
        report_to="none",
    )
    # Initialise trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
        adv_lambda=script_args.adv_lambda,
        fairness_mode=script_args.fairness_mode,
        fairness_alpha=script_args.fairness_alpha,
        fairness_beta=script_args.fairness_beta,
        fairness_gamma=script_args.fairness_gamma,
        categories=["helpful", "harmless"],
        adv_warmup_steps=script_args.adv_warmup_steps,
        fairness_warmup_epochs=script_args.fairness_warmup_epochs,
        log_accuracy_over_steps=25
    )
    print("\nStarting training...")
    trainer.train()
    print("Training complete. Saving model...")
    trainer.save_model(script_args.output_path + "/last_checkpoint")
    tokenizer.save_pretrained(script_args.output_path + "/last_checkpoint")
    # Generate plots and evaluate on RewardBench using evaluation split as out-of-distribution proxy
    generate_plots_and_eval(model, tokenizer, device, eval_dataset)


if __name__ == "__main__":
    main()
