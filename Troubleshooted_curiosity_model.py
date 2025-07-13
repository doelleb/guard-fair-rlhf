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

N_CLUSTERS = 100
KMEANS = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    batch_size=1024,
    reassignment_ratio=0.01,
)


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
    cluster_batch_size: int =100
    recluster_interval: int = 50
    reward_norm_beta: float = 0.01
    fairness_lambda: float = 0.2
    mi_buffer_size: int = 10000
    alpha_curiosity: float = 0.1
    device: str = "cuda"
    verbose: bool = False
    fairness_boost_dynamic_scale: bool = False
    fairness_boost_scale_factor: float = 1.0
    boltzmann_beta: float = 5.0
    seed: int = 42

from transformers import TrainerCallback

class KMeansRefitCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        train_dataloader = kwargs["train_dataloader"]
        model = kwargs["model"]
        emb_list = []

        # collect up to 50 batches of embeddings
        for _, batch in zip(range(50), train_dataloader):
            inputs = {
                "input_ids":      batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            emb = (
                outputs.hidden_states[-1]
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            emb_list.append(emb)

        # stack into (n_samples, dim)
        emb_array = np.vstack(emb_list)

        # only call partial_fit if we have at least as many samples as clusters
        n_clusters = KMEANS.n_clusters
        if emb_array.shape[0] >= n_clusters:
            KMEANS.partial_fit(emb_array)
        else:
            if state.is_local_process_zero:
                print(
                    f"[KMeansRefitCallback] Skipping partial_fit: "
                    f"got {emb_array.shape[0]} samples, need ≥ {n_clusters}"
                )

        return control




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
        self.lmb       = hp.fairness_lambda
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

    def apply_fairness_boost(self, novelty: torch.Tensor, signal: float) -> torch.Tensor:
        boost = self.lmb * signal
        if self.eps and self.lmb:
            boost *= (math.sqrt(self.var) * self.hp.fairness_boost_scale_factor) if self.hp.fairness_boost_dynamic_scale else 1.0
        return novelty + boost

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
        self.km        = MiniBatchKMeans(n_clusters=num, random_state=0, n_init=1, batch_size=6144)
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

# === The Fairness‐Imbued Curiosity Model ===
class FairnessImbuedCuriosityModel:
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
        # --- NEW: store **fairness signals** per cluster ---
        self.fairness_signals = np.zeros(hp.num_clusters)
        self.group_visits = {
            "helpful": np.zeros(hp.num_clusters),
            "harmless": np.zeros(hp.num_clusters)
        }
    def observe(self, embs: np.ndarray, group: str):
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
            if cid >= 0:
                if group == "helpful":
                    self.group_visits["helpful"][cid] += 1
                elif group == "harmless":
                    self.group_visits["harmless"][cid] += 1

            # 3) Compute flipped fairness signal
            if cid >= 0:
                total = (self.group_visits["helpful"][cid]
                    + self.group_visits["harmless"][cid]
                    + self.core.eps)
                if group == "helpful":
                    # boost when helpful has visited fewer times than harmless
                    fairness_signal = (
                        self.group_visits["harmless"][cid]
                    - self.group_visits["helpful"][cid]
                    ) / total
                elif group == "harmless":
                    # boost when harmless has visited fewer times than helpful
                    fairness_signal = (
                        self.group_visits["helpful"][cid]
                    - self.group_visits["harmless"][cid]
                    ) / total
                else:
                    fairness_signal = 0.0
            else:
                fairness_signal = 0.0

            # 4) (Optional) Clamp the boost magnitude
            #    so it never exceeds ±0.5
            fairness_signal = max(-0.5, min(0.5, fairness_signal))

            # 5) Record for sampling stats
            if cid >= 0:
                self.fairness_signals[cid] = fairness_signal

            # 6) Apply boost & buffer for MI
            boosted = self.core.apply_fairness_boost(nov, fairness_signal)
            r = self.core.normalize(boosted.item())
            self.mi_buf.append((r, cid))


    def intrinsic_reward(self, emb: np.ndarray, group: str) -> float:
        # 1) Compute raw novelty
        nov, _, _ = self.core.compute_novelty(emb)
        # 2) Find cluster
        cid = self.clust.assign(emb)
        # 3) Compute flipped fairness signal
        fairness_signal = 0.0
        if cid >= 0:
            total = (self.group_visits["helpful"][cid]
                + self.group_visits["harmless"][cid]
                + self.core.eps)
            if group == "helpful":
                # boost when helpful has visited less than harmless
                fairness_signal = (
                    self.group_visits["harmless"][cid]
                - self.group_visits["helpful"][cid]
                ) / total
            elif group == "harmless":
                # boost when harmless has visited less than helpful
                fairness_signal = (
                    self.group_visits["helpful"][cid]
                - self.group_visits["harmless"][cid]
                ) / total

        # 4) Apply fairness boost (now encourages under-visited)
        boosted = self.core.apply_fairness_boost(nov, fairness_signal)
        # 5) Normalize and scale by curiosity weight
        return self.hp.alpha_curiosity * self.core.normalize(boosted.item())


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

from transformers import PreTrainedTokenizer, PreTrainedModel

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
            }

        # 1) Load only the train split
        ds = load_dataset("Dahoas/full-hh-rlhf", split="train")

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



    train_dataset, eval_dataset = build_dataset(tokenizer, train_size=2000, eval_size=500)


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
    import torch
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import PaddingStrategy


    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: AutoTokenizer
        padding: Union[bool, str, PaddingStrategy] = "longest"
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    import torch.nn.functional as F

    class RewardTrainer(Trainer):
        def __init__(
            self,
            *args,
            fair_curiosity_model=None,
            script_args=None,
            standardize_curiosity: bool = False,
            cur_buf_size: int = 10000,
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.fair_curiosity_model     = fair_curiosity_model
            self.script_args              = script_args
            self.standardize_curiosity    = standardize_curiosity
            # buffer for z-scoring the per-step curiosity distances
            from collections import deque
            self._cur_buf = deque(maxlen=cur_buf_size)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # ------------------- ranking loss -------------------
            # forward pass to get reward logits
            rewards = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits.squeeze(-1)  # shape: (2*batch_size,)

            # split into chosen vs rejected
            bsz  = rewards.size(0)
            jidx = torch.arange(0, bsz, 2, device=rewards.device)
            kidx = jidx + 1
            r_j  = rewards[jidx]
            r_k  = rewards[kidx]

            # pure ranking loss
            ranking_loss = -F.logsigmoid(r_j - r_k).mean()
            loss = ranking_loss

            # --------------- curiosity/fairness ---------------
            if (
                self.fair_curiosity_model and
                int(self.state.epoch) >= self.script_args.curiosity_start_epoch
            ):
                # 1) batch‐encode all prompts
                texts = inputs["texts_j"] + inputs["texts_k"]
                enc   = self.data_collator.tokenizer(
                            texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        ).to(next(model.parameters()).device)

                # 2) get last‐hidden‐mean embeddings
                out   = model(**enc, output_hidden_states=True, return_dict=True)
                embs  = out.hidden_states[-1].mean(dim=1)  # shape: (2*B, D)
                emb_np = embs.detach().cpu().numpy()

                # 3) per-dimension standardize
                emb_np = (emb_np - emb_np.mean(0)) / (emb_np.std(0) + 1e-8)

                # 4) centroid distances scaled by √D
                cids    = KMEANS.predict(emb_np)
                cents   = KMEANS.cluster_centers_[cids]
                D       = emb_np.shape[1]
                rwd_np  = np.linalg.norm(emb_np - cents, axis=1) / math.sqrt(D)

                # 5) optional z-score over running buffer
                if self.standardize_curiosity:
                    self._cur_buf.extend(rwd_np)
                    buf   = np.array(self._cur_buf)
                    mu    = buf.mean()
                    sigma = buf.std() + 1e-8
                    rwd_np = (rwd_np - mu) / sigma

                # 6) mean it and convert to tensor
                curiosity_r = torch.from_numpy(rwd_np).to(rewards.device).float().mean()

                # 7) linear ramp weight
                epochs_since = int(self.state.epoch) - self.script_args.curiosity_start_epoch + 1
                w            = min(1.0, epochs_since / float(self.script_args.curiosity_ramp_epochs))

                print(f"rank={ranking_loss.item():.3f}, curious={curiosity_r.item():.3f}")
                loss = ranking_loss + w * curiosity_r

            if return_outputs:
                return loss, {"rewards_j": r_j, "rewards_k": r_k}
            return loss



    embedding_dim = model.config.hidden_size
    
    hp_fair = Hyperparameters(
        embedding_dim=embedding_dim, num_clusters=3, fairness_lambda=1.0,
        recluster_interval=100, warmup_samples=100,
        mi_buffer_size=1000, verbose=True,
        fairness_boost_dynamic_scale=False, fairness_boost_scale_factor=0.5,
        boltzmann_beta=5.0,
        device=str(device)
    )

    fair_curiosity_model = FairnessImbuedCuriosityModel(hp_fair)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer,
        ),
        fair_curiosity_model=fair_curiosity_model,
        script_args=script_args,
        callbacks = [KMeansRefitCallback()],
        standardize_curiosity=True,
        cur_buf_size=5000,

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
    chosen_scores   = np.array([get_reward_score(t) for t in train_dataset["text_j"]])
    rejected_scores = np.array([get_reward_score(t) for t in train_dataset["text_k"]])

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


    # --- Plot 2: Helpful vs Harmless using Anthropic HH-RLHF splits ---
    ds_helpful  = load_dataset(
        "Anthropic/hh-rlhf",
        split="train[:1000]",
        data_dir="helpful-base"
    )
    ds_harmless = load_dataset(
        "Anthropic/hh-rlhf",
        split="train[:1000]",
        data_dir="harmless-base"
    )

    helpful_scores  = np.array([get_reward_score(t) for t in ds_helpful["chosen"]])
    harmless_scores = np.array([get_reward_score(t) for t in ds_harmless["chosen"]])

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
    plt.title("Reward Distribution: Helpful vs Harmless (Anthropic HH-RLHF)")
    plt.xlabel("Reward score"); plt.ylabel("Density")
    plt.legend(); plt.tight_layout(); plt.show()
