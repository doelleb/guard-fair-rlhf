# Standard Curiosity Model

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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoModel
)
from transformers.utils import PaddingStrategy


# === Utils ===
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# === Hyperparameters ===
@dataclass
class Hyperparameters:
    embedding_dim: int
    num_clusters: int = 10
    learning_rate: float = 1e-3
    rnd_output_dim: int = 64
    rnd_ensemble_count: int = 2
    warmup_samples: int = 50
    cluster_batch_size: int = 32
    recluster_interval: int = 50
    reward_norm_beta: float = 0.01
    mi_buffer_size: int = 10000
    alpha_curiosity: float = 1.0
    device: str = "cpu"
    verbose: bool = False
    seed: int = 42

# === RND / Curiosity Core ===
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
    def __init__(self, hp: Hyperparameters):
        self.dev       = torch.device(hp.device)
        self.beta      = hp.reward_norm_beta
        self.eps       = 1e-8
        # RND ensemble
        self.predictor = PredictorNetwork(hp.embedding_dim, hp.rnd_output_dim).to(self.dev)
        self.targets   = [TargetNetwork(hp.embedding_dim, hp.rnd_output_dim).to(self.dev)
                          for _ in range(hp.rnd_ensemble_count)]
        self.opt       = torch.optim.Adam(self.predictor.parameters(), lr=hp.learning_rate)
        self.loss_fn   = nn.MSELoss(reduction="mean")
        # running stats for normalization
        self.mean = 0.0
        self.var  = 1.0
        self.hp = hp # Store hp as an instance variable

    def compute_novelty(self, emb: np.ndarray):
        x = torch.from_numpy(emb).float().unsqueeze(0).to(self.dev)
        with torch.no_grad():
            outs = [t(x) for t in self.targets]
        pred = self.predictor(x)
        losses = [self.loss_fn(pred, o) for o in outs]
        return torch.stack(losses).mean(), pred, outs[0]

    def update_predictor(self, losses: List[torch.Tensor]):
        self.opt.zero_grad()
        torch.stack(losses).mean().backward()
        self.opt.step()

    def normalize(self, val: float) -> float:
        self.mean = (1 - self.beta)*self.mean + self.beta*val
        self.var  = (1 - self.beta)*self.var  + self.beta*(val - self.mean)**2
        std = math.sqrt(max(self.var, self.eps))
        return float((val - self.mean) / (std + self.eps))

# === Embedding & Cluster Management ===
class EmbeddingCollector:
    def __init__(self, buf_size=10000):
        self.buf = deque(maxlen=buf_size)

    def add(self, embs: np.ndarray):
        for e in embs.reshape(-1, embs.shape[-1]):
            self.buf.append(e)

    def all(self) -> np.ndarray:
        return np.array(self.buf)

class ClusterManager:
    def __init__(self, num, warmup, batch, interval):
        self.num       = num
        self.warmup    = warmup
        self.batch     = batch
        self.interval  = interval
        self.km        = MiniBatchKMeans(n_clusters=num, random_state=0, n_init=10)
        self.fitted    = False
        self.visits    = np.zeros(num)
        self.total     = 0

    def update(self, collector: EmbeddingCollector):
        data = collector.all()
        if data.size == 0:
            return
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # initial fit
        if (not self.fitted) and (len(data) >= max(self.warmup, self.num)):
            self.km.fit(data)
            self.fitted = True
        elif self.fitted:
            # partial-fit
            if len(data) >= self.batch:
                b = data[-self.batch:]
                if b.ndim == 1: b = b.reshape(1, -1)
                self.km.partial_fit(b)
            # full refit at intervals
            if (self.total % self.interval == 0) and self.total > 0 and len(data) >= self.num:
                self.km = MiniBatchKMeans(n_clusters=self.num, random_state=0, n_init=10)
                self.km.fit(data)

    def assign(self, emb: np.ndarray) -> int:
        if not self.fitted:
            return -1
        return int(self.km.predict(emb.reshape(1, -1))[0])

    def visit(self, cid: int):
        if cid >= 0:
            self.visits[cid] += 1

# === The Standard Curiosity Model ===
class StandardCuriosityModel:
    def __init__(self, hp: Hyperparameters):
        if not hp.verbose:
            logging.getLogger().setLevel(logging.WARNING)
        set_seed(hp.seed)

        self.hp    = hp
        self.core  = CuriosityCore(hp)
        self.coll  = EmbeddingCollector(hp.mi_buffer_size)
        self.clust = ClusterManager(hp.num_clusters, hp.warmup_samples,
                                    hp.cluster_batch_size, hp.recluster_interval)
        self.mi_buf        = deque(maxlen=hp.mi_buffer_size)
        self.loss_buffer   = []
        self.step          = 0

    def observe(self, embs: np.ndarray):
        for emb in embs.reshape(-1, embs.shape[-1]):
            self.step += 1
            # 1) Compute RND novelty
            nov, pr, tgt = self.core.compute_novelty(emb)
            self.loss_buffer.append(self.core.loss_fn(pr, tgt))
            if len(self.loss_buffer) >= self.hp.cluster_batch_size:
                self.core.update_predictor(self.loss_buffer)
                self.loss_buffer.clear()

            # 2) Cluster & record visits
            self.coll.add(emb.reshape(1, -1))
            self.clust.total = self.step
            self.clust.update(self.coll)

            cid = self.clust.assign(emb)
            self.clust.visit(cid)

            # 3) Buffer for MI
            r = self.core.normalize(nov.item())
            self.mi_buf.append((r, cid))

    def intrinsic_reward(self, emb: np.ndarray) -> float:
        nov, _, _ = self.core.compute_novelty(emb)
        return self.hp.alpha_curiosity * self.core.normalize(nov.item())

    def sample_cluster(self) -> int:
        # Grab current visit counts
        visits = self.clust.visits
        # Invert: under-visited clusters get higher weight
        inv = visits.max() - visits
        # Turn into a valid probability distribution
        if inv.sum() > 0:
            probs = inv / inv.sum()
        else:
            # if somehow all equal, fall back to uniform
            probs = np.ones_like(inv) / len(inv)
        # Sample a cluster according to this distribution
        return int(np.random.choice(len(probs), p=probs))


    def get_cluster_visits(self) -> np.ndarray:
        return self.clust.visits.copy()

    def get_mutual_information_estimate(self) -> float:
        if len(self.mi_buf) < 2:
            return 0.0
        R = np.array([r for r, _ in self.mi_buf]).reshape(-1, 1)
        C = np.array([c for _, c in self.mi_buf])
        return float(mutual_info_regression(R, C, discrete_features=False)[0])

# === Embedding / Simulation / Plotting ===
def get_llm_embeddings(texts: List[str], model_name="gpt2", device="cpu", batch_size=32):
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name).to(device)

    tok.model_max_length = mod.config.n_positions

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_embs = []
    for i in range(0, len(texts), batch_size):
        b = texts[i : i+batch_size]
        inp = tok(b, return_tensors="pt", truncation=True, max_length=tok.model_max_length, padding="max_length").to(device)
        with torch.no_grad():
            out = mod(**inp)
        em = out.last_hidden_state.mean(dim=1).cpu().numpy()
        if em.ndim == 1:
            em = em.reshape(1, -1)
        all_embs.append(em)
    return np.vstack(all_embs)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Define and parse arguments.
    @dataclass
    class ScriptArguments:
        """
        These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
        """
        local_rank: Optional[int] = field(
            default=-1, metadata={"help": "Used for multi-gpu"})

        deepspeed: Optional[str] = field(
            default=None,
            metadata={
                "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn\"t fit on a single GPU."
            },
        )
        per_device_train_batch_size: Optional[int] = field(default=1)
        per_device_eval_batch_size: Optional[int] = field(default=1)
        gradient_accumulation_steps: Optional[int] = field(default=64)
        learning_rate: Optional[float] = field(default=2e-6)
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
            default=1,
            metadata={"help": "The number of training epochs for the reward model."},
        )
        train_set_path: Optional[str] = field(
            default="hendrydong/preference_700K",
            metadata={"help": "The dir of the subset of the training data to use"},
        )
        eval_set_path: Optional[str] = field(
            default="hendrydong/preference_700K",
            metadata={"help": "The dir of the subset of the eval data to use"},
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    tokenizer.model_max_length = model.config.n_positions
    model.config.use_cache = not script_args.gradient_checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    train_path = script_args.train_set_path
    eval_path = script_args.eval_set_path
    output_name = script_args.output_path

    def build_dataset(tokenizer, train_path, eval_path):
        def tokenize(sample):
            def unwrap(x):
                while isinstance(x, (list, tuple)) and len(x) > 0:
                    x = x[0]
                return x

            pos = unwrap(sample["chosen"])
            neg = unwrap(sample["rejected"])

            pos = pos if isinstance(pos, str) else str(pos)
            neg = neg if isinstance(neg, str) else str(neg)

            tok_pos = tokenizer(pos, truncation=True, max_length=tokenizer.model_max_length)
            tok_neg = tokenizer(neg, truncation=True, max_length=tokenizer.model_max_length)

            sample["input_ids_j"]      = tok_pos["input_ids"]
            sample["attention_mask_j"] = tok_pos["attention_mask"]
            sample["input_ids_k"]      = tok_neg["input_ids"]
            sample["attention_mask_k"] = tok_neg["attention_mask"]
            # Store original texts for curiosity model
            sample["text_j"] = pos
            sample["text_k"] = neg
            return sample

        ds = load_dataset(train_path, split="train[:500]")
        ds = ds.map(tokenize, num_proc=8)

        train_dataset = ds
        eval_dataset  = ds.select(range(100))
        return train_dataset, eval_dataset

    train_dataset, eval_dataset = build_dataset(tokenizer, train_path, eval_path)
    print(f"Loaded {len(train_dataset)} train examples, {len(eval_dataset)} eval examples")
    print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

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
        report_to='wandb'
    )

    num_proc = 24
    original_columns = train_dataset.column_names

    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: AutoTokenizer
        padding: Union[bool, str, PaddingStrategy] = "max_length"
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            merged_features = []
            texts_j = []
            texts_k = []

            for feature in features:
                merged_features.append(
                    {
                        "input_ids": feature["input_ids_j"],
                        "attention_mask": feature["attention_mask_j"],
                    }
                )
                merged_features.append(
                    {
                        "input_ids": feature["input_ids_k"],
                        "attention_mask": feature["attention_mask_k"],
                    }
                )
                texts_j.append(feature["text_j"])
                texts_k.append(feature["text_k"])

            ctx = self.tokenizer.model_max_length
            batch = self.tokenizer.pad(
                merged_features,
                padding="longest",
                return_tensors=self.return_tensors,
            )

            if batch["input_ids"].size(1) > ctx:
                batch["input_ids"]      = batch["input_ids"][:, -ctx:]
                batch["attention_mask"] = batch["attention_mask"][:, -ctx:]
            
            batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "return_loss": True,
                "texts_j": texts_j,
                "texts_k": texts_k,
            }
            return batch

    def compute_metrics(eval_pred):
        result = {}
        pos_predictions_scores = eval_pred.predictions[0]
        neg_predictions_scores = eval_pred.predictions[1]
        result["accuracy"] = np.sum(
            pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
        return result

    class RewardTrainer(Trainer):
        def __init__(self, *args, curiosity_model=None, tokenizer=None, device=None, script_args=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.curiosity_model = curiosity_model
            self.processing_class = tokenizer # Use processing_class instead of tokenizer
            self.device = device
            self.script_args = script_args # Store script_args

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            rewards = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )[0]
            bsz = rewards.size(0)
            jidx = torch.arange(0, bsz, 2)
            kidx = jidx + 1

            rewards_j = rewards[jidx]
            rewards_k = rewards[kidx]

            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()

            # Add curiosity loss if the model is provided
            if self.curiosity_model and self.processing_class and self.device and self.script_args:
                texts_j = inputs["texts_j"]
                texts_k = inputs["texts_k"]

                # Get embeddings for chosen and rejected texts
                # Use the tokenizer and model to get embeddings
                embs_j = get_llm_embeddings(texts_j, model_name=self.script_args.model_name, device=self.device)
                embs_k = get_llm_embeddings(texts_k, model_name=self.script_args.model_name, device=self.device)

                curiosity_loss = 0.0
                for i in range(len(embs_j)):
                    # Observe and get intrinsic reward for chosen text
                    self.curiosity_model.observe(embs_j[i].reshape(1, -1))
                    curiosity_loss += self.curiosity_model.intrinsic_reward(embs_j[i])

                    # Observe and get intrinsic reward for rejected text
                    self.curiosity_model.observe(embs_k[i].reshape(1, -1))
                    curiosity_loss += self.curiosity_model.intrinsic_reward(embs_k[i])
                
                # Average curiosity loss over the batch
                curiosity_loss /= (len(embs_j) * 2)
                loss += curiosity_loss

            if return_outputs:
                return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
            return loss


    # Define the trainer
    # Initialize standard curiosity model
    # Infer embedding_dim from the model
    embedding_dim = model.config.hidden_size
    
    hp_standard = Hyperparameters(
        embedding_dim=embedding_dim, num_clusters=3,
        recluster_interval=100, warmup_samples=100,
        mi_buffer_size=1000, verbose=True, device=device
    )
    standard_curiosity_model = StandardCuriosityModel(hp_standard)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=script_args.max_length
        ),
        curiosity_model=standard_curiosity_model,
        device=device,
        script_args=script_args
    )


    print("Starting reward-model training…")
    trainer.train()


    print("Saving last checkpoint of the model")
    trainer.save_model(output_name + "/last_checkpoint")
    tokenizer.save_pretrained(output_name + "/last_checkpoint")

    import seaborn as sns
    from scipy.stats import gaussian_kde

    model.eval()
    sns.set_theme(style="whitegrid")

    # grab 100 HH-RLHF pairs
    pairs = load_dataset("Dahoas/full-hh-rlhf", split="train") \
            .shuffle(seed=42).select(range(100))

    def get_reward_score(text: str) -> float:
        # clamp to the model’s actual context size
        max_len = tokenizer.model_max_length
        toks = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            return model(**toks).logits.squeeze().cpu().item()

    chosen_scores   = np.array([get_reward_score(ex["chosen"])   for ex in pairs])
    rejected_scores = np.array([get_reward_score(ex["rejected"]) for ex in pairs])

    plt.figure(figsize=(8, 5))

    # plot KDEs
    sns.kdeplot(
        chosen_scores,
        label="Helpful",
        fill=True,
        alpha=0.5,
        linewidth=2
    )
    sns.kdeplot(
        rejected_scores,
        label="Harmless",
        fill=True,
        alpha=0.5,
        linewidth=2
    )

    plt.xlabel("Rewards")
    plt.ylabel("Density")
    plt.title("Reward Distribution: Helpful vs. Harmless (Standard Curiosity)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_distribution_standard_curiosity.png")
    plt.show()

