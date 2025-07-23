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
import seaborn as sns
from sklearn.metrics import mutual_info_score

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

    tokenizer.model_max_length = model.config.n_positions
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
        per_device_train_batch_size: Optional[int] = field(default=16)
        per_device_eval_batch_size: Optional[int] = field(default=16)
        gradient_accumulation_steps: Optional[int] = field(default=2)
        learning_rate: Optional[float] = field(default=3e-5)
        weight_decay: Optional[float] = field(default=0.001)

        model_name: Optional[str] = field(
            default="erwanf/gpt2-mini",
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
            default=5,
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

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if script_args.bf16 else None,
    )

    # 1) Resize embeddings *while still on CPU*
    tokenizer.model_max_length = model.config.n_positions
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

            # 2) If that element is a dict, extract its 'content'; otherwise assume it's already a string
            text_j = raw_chosen["content"]   if isinstance(raw_chosen, dict)  else raw_chosen
            text_k = raw_rejected["content"] if isinstance(raw_rejected, dict) else raw_rejected

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

        print(ds[0])
        print() 
        print(ds[1])




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
        train_dataset = ds.select(range(train_size))
        eval_dataset  = ds.select(range(train_size, train_size + eval_size))

        return train_dataset, eval_dataset



    train_dataset, eval_dataset = build_dataset(tokenizer, train_size=10000, eval_size=2500)


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
        """
        rew_pairs : (B, 2) tensor        ← [chosen, rejected] per sample
        cats      : length-B python list ← 'helpful' / 'harmless'

        returns   : 1-D tensor
                [ mean_H(2) , mean_R(2) ,
                    var_H(2)  , var_R(2)  ,
                    skew_H(2) , skew_R(2) ,
                    kurt_H(2) , kurt_R(2) ]      → length  2⋅4⋅2 = 16
        """
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
        def __init__(self, input_dim=16, num_categories=2):
            super().__init__()
            self.in_dim = input_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_categories),
            )
        def forward(self, x):
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
            # 0) First, if we buffered a discriminator loss from the *previous* step, update D now.
            if getattr(self, "_disc_loss", None) is not None:
                self.optim_d.zero_grad()
                self._disc_loss.backward()
                self.optim_d.step()
                self._disc_loss = None

            # 1) Forward pass through reward model
            outputs = model(
                input_ids=inputs["input_ids"].to(self.args.device),
                attention_mask=inputs["attention_mask"].to(self.args.device)
            )
            rewards = outputs.logits.squeeze(-1)  # (2*B,)

            # 2) Split chosen/rejected
            jidx = torch.arange(0, rewards.size(0), 2, device=rewards.device)
            kidx = jidx + 1
            r_j = rewards[jidx]
            r_k = rewards[kidx]

            # Bradley–Terry loss
            loss_bt = -F.logsigmoid(r_j - r_k).mean()

            # 3) Build discriminator features (pairwise moments per category)
            reward_pair_batch = torch.stack([r_j, r_k], dim=1)  # (B, 2)
            feat_dict = pairwise_moments(reward_pair_batch, inputs["category"])  # dict[str -> tensor]
            feat_dict = {k: v.to(rewards.device) for k, v in feat_dict.items()}

            # Lazy‑init discriminator with correct feature dim
            first_feat_dim = next(iter(feat_dict.values())).numel()
            if not hasattr(self, "_feat_dim_init"):
                # Recreate discriminator if wrong size
                if getattr(self.discriminator, "in_dim", None) != first_feat_dim:
                    self.discriminator = Discriminator(
                        input_dim=first_feat_dim,
                        num_categories=len(self.category_ids)
                    ).to(self.args.device)
                self._feat_dim_init = True

            # 4) Adversarial term = maximize entropy => minimize (-entropy)
            adv_entropy = 0.0
            disc_loss = 0.0
            for cat, feat in feat_dict.items():
                feat = feat.unsqueeze(0)  # (1, D)

                # Freeze D, compute entropy (grads flow to reward model params through feat)
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                logits_adv = self.discriminator(feat)
                probs = logits_adv.softmax(dim=-1)
                entropy = -(probs * probs.log()).sum()
                adv_entropy = adv_entropy + entropy

                # Unfreeze D, train it to *predict* the category (feat detached)
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)
                logits_d = self.discriminator(feat.detach())
                target = torch.tensor([self.category_ids[cat]], device=logits_d.device)
                disc_loss = disc_loss + self.criterion_d(logits_d, target)

            # buffer discriminator loss for next step
            self._disc_loss = disc_loss

            # 5) Total reward-model loss
            total_loss = loss_bt - self.adv_lambda * adv_entropy / len(feat_dict)

            # Optional debug print
            if self.state.global_step % 10 == 0:
                d_val = disc_loss.detach().item() if isinstance(disc_loss, torch.Tensor) else float(disc_loss)
                print(f"loss_bt={loss_bt.item():.3f}, adv_ent={adv_entropy.item():.3f}, loss_d={d_val:.3f}")

            if return_outputs:
                return total_loss, {"rewards_j": r_j, "rewards_k": r_k}
            return total_loss


        # ---------------------------------------------------------------
        # Custom training_step: run standard Trainer step then update D
        # ---------------------------------------------------------------

        # def training_step(self, model, inputs):
        #     """Run the regular training step for the reward model, then update the discriminator with the buffered loss."""
        #     # First let the parent class handle forward, backward, and (maybe) optimizer step for the reward model
        #     loss = super().training_step(model, inputs)

        #     # After gradients have been propagated through the reward model we can safely update the discriminator
        #     if hasattr(self, "_disc_loss") and self._disc_loss is not None:
        #         self.optim_d.zero_grad()
        #         self._disc_loss.backward()
        #         self.optim_d.step()
        #         # Reset buffer
        #         self._disc_loss = None

        #     return loss
    class AdvLambdaAblation:
        """
        Sweep a list of `adv_lambda` values, train a fresh reward model for each,
        then plot Bradley–Terry loss vs. mutual information I(r ; c).
        """
        def __init__(self,
                    script_args,
                    tokenizer,
                    train_ds,
                    eval_ds,
                    sweep_vals=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
            self.cfg        = script_args
            self.tokenizer  = tokenizer
            self.train_ds   = train_ds
            self.eval_ds    = eval_ds
            self.sweep_vals = sweep_vals
            self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sns.set_theme(style="whitegrid")

        # -------------------- evaluation helpers ---------------------------------
        @staticmethod
        def _bt_loss(pos, neg):
            """scalar BT loss for a batch (numpy arrays)."""
            return np.mean(-np.logaddexp(0.0, -(pos - neg)))

        # ---------- helper: evaluate one trained model ----------
        def _eval_one(self, model, batch_size=32):
            """
            Return (BT_loss, MI) on self.eval_ds without using Trainer.predict,
            so no extra keys hit model.forward().
            """
            model.eval()
            pos_scores, neg_scores = [], []
            chosen_rewards, cats   = [], []

            for i in range(0, len(self.eval_ds), batch_size):
                batch = self.eval_ds[i : i + batch_size]

                # ------ chosen & rejected texts ------
                tj = batch["text_j"]
                tk = batch["text_k"]
                enc = self.tokenizer(
                    tj + tk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).to(self.device)

                with torch.no_grad():
                    out = model(**enc).logits.squeeze().cpu().numpy()

                # split back into j / k
                n = len(tj)
                r_j, r_k = out[:n], out[n:]
                pos_scores.extend(r_j)
                neg_scores.extend(r_k)

                # for MI we only need chosen side
                chosen_rewards.extend(r_j)
                cats.extend(batch["category"])

            # ---- metrics ----
            bt_loss = self._bt_loss(np.array(pos_scores), np.array(neg_scores))
            r_disc  = np.digitize(
                chosen_rewards,
                np.histogram_bin_edges(chosen_rewards, bins=50)
            )
            mi = mutual_info_score(cats, r_disc)
            return bt_loss, mi


        # -------------------- main loop ------------------------------------------
        def run(self, plot_path="runs/adv_lambda_tradeoff.png"):
            results = []
            for lam in self.sweep_vals:
                print(f"\n▶ Ablation: adv_lambda = {lam}")
                set_seed(42)  # reproducible run‑to‑run
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.cfg.model_name, num_labels=1,
                    torch_dtype=torch.bfloat16 if self.cfg.bf16 else None
                ).to(self.device)
                model.resize_token_embeddings(len(self.tokenizer))

                targs = TrainingArguments(
                    output_dir=f"./runs/adv_{lam}",
                    per_device_train_batch_size=self.cfg.per_device_train_batch_size,
                    per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
                    num_train_epochs=self.cfg.num_train_epochs,
                    learning_rate=self.cfg.learning_rate,
                    weight_decay=self.cfg.weight_decay,
                    gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
                    gradient_checkpointing=self.cfg.gradient_checkpointing,
                    optim=self.cfg.optim,
                    lr_scheduler_type=self.cfg.lr_scheduler_type,
                    fp16=self.cfg.fp16, bf16=self.cfg.bf16,
                    eval_strategy="no", save_strategy="no",
                    remove_unused_columns=False,
                    logging_steps=200,
                )

                self.trainer = RewardTrainer(
                    model=model, args=targs,
                    train_dataset=self.train_ds, eval_dataset=self.eval_ds,
                    data_collator=RewardDataCollatorWithPadding(tokenizer=self.tokenizer),
                    categories=["helpful", "harmless"],
                    adv_lambda=lam,

                )
                self.trainer.train()
                bt, mi = self._eval_one(model)
                results.append({"λ": lam, "BT_loss": bt, "MI": mi})
                print(f"   → BT_loss={bt:.4f} | I(r;c)={mi:.4f}")
                del model, self.trainer; torch.cuda.empty_cache()

            # ------------ plot fairness vs. loss ---------------------------------
            os.makedirs("runs", exist_ok=True)
            np.save("runs/ablation_results.npy", results)  # raw numbers if you need them
            lambdas  = [r["λ"] for r in results]
            bt_loss  = [r["BT_loss"] for r in results]
            mi_vals  = [r["MI"] for r in results]

            plt.figure(figsize=(6,4))
            ax = sns.scatterplot(x=mi_vals, y=bt_loss, hue=lambdas, palette="viridis", s=80)
            for l, x, y in zip(lambdas, mi_vals, bt_loss):
                ax.text(x, y, f"λ={l}", fontsize=9, weight="bold", ha="left", va="bottom")
            ax.set_xlabel("Mutual Information")
            ax.set_ylabel("Bradley-Terry Loss")
            ax.set_title("Fairness vs Accuracy trade-off across adv_lambda")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            print(f"\nSaved trade-off plot → {plot_path}")

    # ──────────── END PATCH ────────────────────────────────────────────────────

    import os

    # trainer = RewardTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     compute_metrics=compute_metrics,
    #     data_collator=RewardDataCollatorWithPadding(
    #         tokenizer=tokenizer,
    #     ),
    #     fair_curiosity_model=fair_curiosity_model,
    #     script_args=script_args,
    #     callbacks = [KMeansRefitCallback()],
    #     standardize_curiosity=True,
    #     cur_buf_size=5000,

    # )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer),
        categories=["helpful", "harmless"],
    )


    print("Starting reward-model training…")
    trainer.train()
    AblationRunner = AdvLambdaAblation(
    script_args,
    tokenizer,
    train_dataset,
    eval_dataset,
    sweep_vals=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0),   # tweak as you like
)
AblationRunner.run(plot_path="runs/adv_lambda_tradeoff.png")


print("Saving last checkpoint of the model")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")
