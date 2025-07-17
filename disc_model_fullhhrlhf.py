# Recoded Fair Curiosity Model

import multiprocessing
import random
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoModel
)
from transformers.utils import PaddingStrategy
import math
import numpy as np
import torch.nn.functional as F
from collections import deque


import warnings
warnings.filterwarnings(
    "ignore",
    message="MiniBatchKMeans is known to have a memory leak on Windows with MKL"
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


from transformers import TrainerCallback



from transformers import PreTrainedTokenizer, PreTrainedModel

def get_llm_embeddings(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:

    model = model.to(device).eval()
    x = torch.randint(0, tokenizer.vocab_size, (1, 8)).to(device)

    tokenizer.model_max_length = model.config.max_position_embeddings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        embs = last_hidden.mean(dim=1).cpu().numpy()
        all_embs.append(embs)

    return np.vstack(all_embs)

from datasets import load_dataset, concatenate_datasets

def load_hh_rlhf_with_category() -> "datasets.Dataset":
    helpful = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="helpful-base",
        split="train",
    )
    helpful = helpful.map(lambda _: {"category": "helpful"})      # 43 835 rows
    harmless = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="harmless-base",
        split="train",
    )
    harmless = harmless.map(lambda _: {"category": "harmless"})   # 42 537 rows
    combined = concatenate_datasets([helpful, harmless]).shuffle(seed=42)
    return combined


if __name__ == "__main__":
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define and parse arguments.
    @dataclass
    class ScriptArguments:
        local_rank: Optional[int] = field(
            default=-1, metadata={"help": "Used for multi-gpu"})

        deepspeed: Optional[str] = field(
            default=None,
            metadata={
                "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn\"t fit on a single GPU."
            },
        )
        per_device_train_batch_size: Optional[int] = field(default=8)
        per_device_eval_batch_size: Optional[int] = field(default=8)
        gradient_accumulation_steps: Optional[int] = field(default=4)        
        learning_rate: Optional[float] = field(default=3e-5)
        weight_decay: Optional[float] = field(default=0.001)

        model_name: Optional[str] = field(
            default="meta-llama/Llama-3.2-1B",
            metadata={
                "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
            },
        )
        bf16: Optional[bool] = field(
            default=False,
            metadata={"help": "Use bfloat16 if supported"}
        )
        fp16: Optional[bool] = field(
            default=True,
            metadata={"help": "Use float16 mixed precision"}
        )

        num_train_epochs: Optional[int] = field(
            default=15,
            metadata={"help": "The number of training epochs for the reward model."},
        )
        output_path: Optional[str] = field(
            default="./models/llama3_rm",
            metadata={"help": "The dir for output model"},
        )
        gradient_checkpointing: Optional[bool] = field(
            default=True,
            metadata={"help": "Enables gradient checkpointing."},
        )
        optim: Optional[str] = field(
            default="adamw_torch",
            metadata={"help": "The optimizer to use."},
        )
        lr_scheduler_type: Optional[str] = field(
            default="cosine",
            metadata={"help": "The lr scheduler"},
        )
        max_length: Optional[int] = field(default=4096)

        save_every_steps: Optional[int] = field(
            default=999999,
            metadata={"help": "Save the model every x steps"},
        )
        eval_every_steps: Optional[int] = field(
            default=999999,
            metadata={"help": "Eval the model every x steps"},
        )
        curiosity_start_epoch: int = field(
            default=2,
            metadata={"help": "Epoch at which to begin adding curiosity loss"},
        )
        curiosity_ramp_epochs: int = field(
            default=3,
            metadata={"help": "Number of epochs over which to linearly ramp curiosity weight"}
        )


    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    tokenizer_name = script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if script_args.bf16 else None,
    )

    # 1) Resize embeddings *while still on CPU*
    tokenizer.model_max_length = model.config.max_position_embeddings
    model.config.use_cache = not script_args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # 2) Now move the fully-initialized model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)



    output_name = script_args.output_path

    def build_dataset(tokenizer, train_size: int = 10000, eval_size: int = 2000):
        def tokenize(sample):
            # 1) If chosen/rejected come in as lists, grab the *last* element
            raw_chosen   = sample["chosen"][-1]  if isinstance(sample["chosen"],  (list, tuple)) else sample["chosen"]
            raw_rejected = sample["rejected"][-1] if isinstance(sample["rejected"], (list, tuple)) else sample["rejected"]

            # 2) If that element is a dict, extract its \'content\'; otherwise assume it\'s already a string
            text_j = tokenizer.apply_chat_template(
                raw_chosen, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "") if isinstance(raw_chosen, dict) else raw_chosen
            text_k = tokenizer.apply_chat_template(
                raw_rejected, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "") if isinstance(raw_rejected, dict) else raw_rejected

            # 3) Tokenize each side
            tok_j = tokenizer(text_j, truncation=True, max_length=tokenizer.model_max_length)
            tok_k = tokenizer(text_k, truncation=True, max_length=tokenizer.model_max_length)

            # 4) Store the pieces back on the sample
            return {
                "input_ids_j":      tok_j["input_ids"],
                "attention_mask_j": tok_j["attention_mask"],
                "input_ids_k":      tok_k["input_ids"],
                "attention_mask_k": tok_k["attention_mask"],
                "text_j":           text_j,
                "text_k":           text_k,
                "category":         sample["category"],
            }

        # 1) Load only the train split
        ds = load_hh_rlhf_with_category()

        # 2) Remember original columns so we can drop them after tokenization
        original_columns = ds.column_names

        # 3) Map (tokenize) — this will return only the fields we produce in the dict above
        ds = ds.map(
            tokenize,
            num_proc=8,
            remove_columns=original_columns,
        )

        # 4) Shuffle & split off a small eval set
        ds = ds.shuffle(seed=42)
        total_size = len(ds)
        train_size = int(0.95 * total_size)
        test_size = total_size - train_size

        train_dataset = ds.select(range(train_size))
        test_dataset  = ds.select(range(train_size, total_size))

        return train_dataset, test_dataset



    train_dataset, test_dataset = build_dataset(tokenizer)
    #train_dataset = train_dataset.select(range(10))
    #test_dataset  = test_dataset.select(range(10)) #temporary for testing purposes only


    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        eval_strategy="steps",
        eval_steps=script_args.eval_every_steps,
        save_strategy="steps",
        save_steps=script_args.save_every_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=0.03,
        report_to='wandb',
        fp16=script_args.fp16,
    )

    num_proc = 24
    original_columns = train_dataset.column_names

    from dataclasses import dataclass
    from typing import Any, Dict, List, Optional, Union
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import PaddingStrategy
    import torch

    def pairwise_moments(rew_pairs: torch.Tensor, cats: List[str], eps=1e-8):

        device = rew_pairs.device
        cat_tensor = torch.tensor([c == "helpful" for c in cats], device=device)

        def _stats(mask):
            """mask : bool tensor (B,)"""
            vals = rew_pairs[mask]          # (N,2)  or empty
            if vals.numel() == 0:           # guard against empty category in batch
                return [torch.zeros(2, device=device) for _ in range(4)]

            mean = vals.mean(dim=0)                         # (2,)
            centred = vals - mean
            var  = (centred ** 2).mean(dim=0)
            std  = (var + eps).sqrt()
            skew = ((centred / std) ** 3).mean(dim=0)
            kurt = ((centred / std) ** 4).mean(dim=0) - 3.0
            return torch.cat([mean, var, skew, kurt], dim=0)
            #   return [mean, var, skew, kurt]                  # each (2,)

        helpful_stats  = _stats( cat_tensor ) # H
        harmless_stats = _stats(~cat_tensor )    # R
        #print(helpful_stats.shape)
        #print(harmless_stats.shape)

        feat_dict = {
            "helpful": helpful_stats,
            "harmless": harmless_stats,
        }
        return feat_dict 


    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: AutoTokenizer
        padding: Union[bool, str, PaddingStrategy] = "longest"
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            #print(features)
            categories = [f["category"] for f in features]
            # 1) Extract texts
            texts_j = [f["text_j"] for f in features]
            texts_k = [f["text_k"] for f in features]

            # 2) Build an interleaved list [j0, k0, j1, k1, ...]
            interleaved = []
            for j, k in zip(texts_j, texts_k):
                interleaved.append(j)
                interleaved.append(k)

            # 3) Tokenize *all* at once so they share the same pad/truncate
            batch = self.tokenizer(
                interleaved,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length or self.tokenizer.model_max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            # 4) Return exactly the tensors the trainer expects
            return {
                "input_ids":      batch["input_ids"],      # shape: (2*batch_size, seq_len)
                "attention_mask": batch["attention_mask"], # same shape
                "return_loss":    True,
                "texts_j":        texts_j,
                "texts_k":        texts_k,
                "category":       categories,
            }



    def compute_metrics(eval_pred):
        result = {}
        pos_predictions_scores = eval_pred.predictions[0]
        neg_predictions_scores = eval_pred.predictions[1]
        result["accuracy"] = np.sum(
            pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
        return result
    
    from transformers import Trainer
    import torch
    import torch.nn as nn
    import torch.nn.functional as F



    # train a discriminator to predict the category of a reward model output 
    # mutual information constraint done in an adversarial manner 
    class Discriminator(nn.Module):
        def __init__(self, input_dim=8, num_categories=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_categories)
            )
        
        def forward(self, x):
            #print(x.shape)
            return self.net(x)

    class RewardTrainer(Trainer):
        def __init__(self, *args, adv_lambda = 0.6, categories=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.discriminator = Discriminator().to(self.args.device)
            self.optim_d = torch.optim.Adam(self.discriminator.parameters())
            self.criterion_d = nn.CrossEntropyLoss()
            self.adv_lambda = adv_lambda

            if categories is not None: 
                # create a dictionary mapping each category to an index 
                self.category_ids = {cat: i for i, cat in enumerate(categories)}
            else: 
                assert False, "categories must be provided"





        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

            if hasattr(self, "_disc_loss") and self._disc_loss is not None:
                self.optim_d.zero_grad()
                self._disc_loss.backward()
                self.optim_d.step()
                # Reset buffer
                self._disc_loss = None

            rewards = model(
                input_ids=inputs["input_ids"].to(self.args.device), attention_mask=inputs["attention_mask"].to(self.args.device)
            )[0]
            bsz = rewards.size(0)
            jidx = torch.arange(0, bsz, 2)
            kidx = jidx + 1

            # j is preferred, k is not 
            rewards_j = rewards[jidx]
            rewards_k = rewards[kidx]
            loss_bt = -nn.functional.logsigmoid(rewards_j - rewards_k).mean() 

            # Adversarial loss
            categories = torch.tensor([self.category_ids[cat] for cat in inputs["category"]])
            reward_pair_batch = torch.stack([rewards_j, rewards_k], dim = 1).squeeze()

            # for each minibatch, calculate the pairwise moments 
            disc_features = pairwise_moments(reward_pair_batch, inputs["category"])
            # --- quick hack to turn any list outputs into a single tensor ---
            for cat, feat in disc_features.items():
                if isinstance(feat, list):
                    disc_features[cat] = torch.cat(feat, dim=0)

            loss_adv = 0 
            loss_d = 0

            for cat in disc_features: 
                #print(cat, disc_features[cat])

                # ----- Adversarial loss (freeze discriminator weights) -----
                # Temporarily freeze discriminator parameters so gradients do NOT flow into them
                was_training = self.discriminator.training
                self.discriminator.eval()
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)

                adv_logits = self.discriminator(disc_features[cat].to(self.args.device))  # grads w.r.t. disc_features kept
                #print(f"cat: {cat}, adv_logits: {adv_logits}")
                probs = adv_logits.softmax(dim=-1)
                loss_adv += -(probs * torch.log(probs)).sum() 
                

                # Restore discriminator param state
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)
                if was_training:
                    self.discriminator.train()
                d_logits = self.discriminator(disc_features[cat].to(self.args.device).detach())
                
                loss_d_single = self.criterion_d(
                    d_logits.unsqueeze(0),
                    torch.tensor([self.category_ids[cat]], device=d_logits.device)
                )
                # Buffer the loss for later update in training_step
                if not hasattr(self, "_disc_loss") or self._disc_loss is None:
                    self._disc_loss = 0.0
                self._disc_loss = self._disc_loss + loss_d_single # detach so it\'s a separate graph

            

            # Reward model loss: Maximize discriminator uncertainty
            # d_logits = self.discriminator(reward_pair_batch)  # Forward with gradient
            # loss_adv = -d_logits.softmax(dim=1).log().mean()  # Negative log-likelihood
            
            # Total loss -- want to minimize negative entropy (ie. maximize entropy) of discriminator 
            total_loss = loss_bt - self.adv_lambda * loss_adv/len(disc_features)

           # print(f"loss_bt={loss_bt.item():.3f}, loss_adv={loss_adv.item():.3f}, loss_d = {self._disc_loss.item():.3f}")
            
            if return_outputs:
                return total_loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
            return total_loss



    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer),
        categories=["helpful", "harmless"],
    )


    print("Starting reward-model training…")
    trainer.train()


    print("Saving last checkpoint of the model")
    trainer.save_model(output_name + "/last_checkpoint")
    tokenizer.save_pretrained(output_name + "/last_checkpoint")

    model.eval()
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    from scipy.stats import gaussian_kde
    from datasets import load_dataset
    plt.ioff()
    # --- helper: score with fp16 mask only ---
    def get_reward_score(text: str) -> float:
        enc = tokenizer(
            text,
            truncation=True,
            padding="longest",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        # move to GPU
        enc = {k: v.to(device) for k, v in enc.items()}
        # cast only attention_mask to fp16
        if "attention_mask" in enc:
            enc["attention_mask"] = enc["attention_mask"].half()
        with torch.no_grad():
            out = model(**enc).logits.squeeze()
        return out.cpu().item()

    # --- Plot 1: Chosen vs Rejected on your trained model ---
    chosen_scores   = np.array([get_reward_score(t) for t in test_dataset["text_j"]])
    rejected_scores = np.array([get_reward_score(t) for t in test_dataset["text_k"]])

    kde_chosen   = gaussian_kde(chosen_scores)
    kde_rejected = gaussian_kde(rejected_scores)
    xmin = min(chosen_scores.min(), rejected_scores.min())
    xmax = max(chosen_scores.max(), rejected_scores.max())
    xs   = np.linspace(xmin, xmax, 300)

    plt.figure(figsize=(6,4))
    plt.plot(xs, kde_chosen(xs),   label="Chosen",   linewidth=2)
    plt.fill_between(xs, kde_chosen(xs),   alpha=0.3)
    plt.plot(xs, kde_rejected(xs), label="Rejected", linewidth=2)
    plt.fill_between(xs, kde_rejected(xs), alpha=0.3)
    plt.title("Reward Distribution: Chosen vs Rejected")
    plt.xlabel("Reward score"); plt.ylabel("Density")
    plt.legend(); plt.tight_layout(); plt.show()


    helpful_test = test_dataset.filter(lambda ex: ex["category"] == "helpful")
    harmless_test = test_dataset.filter(lambda ex: ex["category"] == "harmless")

    helpful_scores  = np.array([get_reward_score(t) for t in helpful_test["text_j"]])
    harmless_scores = np.array([get_reward_score(t) for t in harmless_test["text_j"]])

    kde_helpful  = gaussian_kde(helpful_scores)
    kde_harmless = gaussian_kde(harmless_scores)
    xmin = min(helpful_scores.min(), harmless_scores.min())
    xmax = max(helpful_scores.max(), harmless_scores.max())
    xs   = np.linspace(xmin, xmax, 300)

    plt.figure(figsize=(6,4))
    plt.plot(xs, kde_helpful(xs),   label="Helpful",   linewidth=2)
    plt.fill_between(xs, kde_helpful(xs),   alpha=0.3)
    plt.plot(xs, kde_harmless(xs),  label="Harmless",  linewidth=2)
    plt.fill_between(xs, kde_harmless(xs),  alpha=0.3)
    plt.title("Reward Distribution: Helpful vs Harmless")
    plt.xlabel("Reward score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
