#!/usr/bin/env python3
import random
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

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
    fairness_lambda: float = 0.1
    mi_buffer_size: int = 10000
    alpha_curiosity: float = 1.0
    device: str = "cpu"
    verbose: bool = False
    fairness_boost_dynamic_scale: bool = False
    fairness_boost_scale_factor: float = 1.0
    boltzmann_beta: float = 5.0
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
            boost *= (math.sqrt(self.var) * hp.fairness_boost_scale_factor) if hp.fairness_boost_dynamic_scale else 1.0
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

            # 3) Compute **fairness signal** = how under-visited this cluster is
            visits = self.clust.visits
            if visits.sum() > 0 and cid >= 0:
                avg = visits.mean()
                fairness_signal = (avg - visits[cid]) / (avg + self.core.eps)
            else:
                fairness_signal = 0.0

            # 4) Record it for sampling
            if cid >= 0:
                self.fairness_signals[cid] = fairness_signal

            # 5) Apply boost & buffer for MI
            boosted = self.core.apply_fairness_boost(nov, fairness_signal)
            r = self.core.normalize(boosted.item())
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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_embs = []
    for i in range(0, len(texts), batch_size):
        b = texts[i : i+batch_size]
        inp = tok(b, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = mod(**inp)
        em = out.last_hidden_state.mean(dim=1).cpu().numpy()
        if em.ndim == 1:
            em = em.reshape(1, -1)
        all_embs.append(em)
    return np.vstack(all_embs)

def run_simulation(model, embs: np.ndarray, steps: int):
    rewards, visits, mis = [], [], []
    # warm-up
    idxs = np.random.choice(len(embs), min(model.hp.warmup_samples, len(embs)), replace=False)
    model.observe(embs[idxs])
    for i in range(steps):
        cid = model.sample_cluster()
        # pick a random embedding from that cluster
        ids = [j for j, e in enumerate(embs) if model.clust.assign(e) == cid]
        idx = random.choice(ids) if ids else random.randrange(len(embs))
        emb = embs[idx].reshape(1, -1)

        model.observe(emb)
        rewards.append(model.intrinsic_reward(emb.flatten()))
        visits.append(model.get_cluster_visits())
        mis.append(model.get_mutual_information_estimate())

        if (i + 1) % 100 == 0:
            logging.info(f"Step {i+1}/{steps} — MI {mis[-1]:.4f}")
    return rewards, visits, mis

def plot_results(fr, fv, fm, nr, nv, nm, labels, embs, hp):
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
    # 1) Reward curves
    axes[0].plot(fr, label="Fair"); axes[0].plot(nr, label="Normal")
    axes[0].set_title("Intrinsic Reward"); axes[0].legend(); axes[0].grid(True)

    # 2) Final visits
    fv0, nv0 = fv[-1], nv[-1]
    x = np.arange(len(fv0))
    axes[1].bar(x - 0.2, fv0, 0.4, label="Fair")
    axes[1].bar(x + 0.2, nv0, 0.4, label="Normal")
    axes[1].set_title("Cluster Visits"); axes[1].legend(); axes[1].grid(axis="y")

    # 3) MI curves
    axes[2].plot(fm, label="Fair MI"); axes[2].plot(nm, label="Normal MI")
    axes[2].set_title("Mutual Information"); axes[2].legend(); axes[2].grid(True)

    # 4) t-SNE
    tsne = TSNE(n_components=2, random_state=hp.seed, perplexity=min(30, len(embs)-1))
    e2 = tsne.fit_transform(embs)
    sc = axes[3].scatter(e2[:,0], e2[:,1], c=labels, cmap="tab10", alpha=0.7)
    axes[3].set_title("t-SNE by Category")
    plt.colorbar(sc, ax=axes[3], label="Category")
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(42)

    # Prepare a small synthetic dataset
    NUM = 500
    TEXTS = [
        # Cat 0
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries globally.",
        "Machine learning algorithms are at the core of modern data analysis.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require vast amounts of data for training.",
        # Cat 1
        "The sun rises in the east and sets in the west.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        "The Earth revolves around the Sun.",
        "Gravity is a fundamental force of nature.",
        # Cat 2
        "A cat purrs when it is content.",
        "Dogs are often called man's best friend.",
        "Birds build nests to lay their eggs.",
        "Fish swim in water using their fins.",
        "Lions are apex predators in their ecosystem."
    ]
    while len(TEXTS) < NUM:
        TEXTS += TEXTS
    TEXTS = TEXTS[:NUM]
    LABELS = np.tile(np.array([0]*5 + [1]*5 + [2]*5), math.ceil(NUM/15))[:NUM]

    logging.info("Generating embeddings…")
    EMBS = get_llm_embeddings(TEXTS, model_name="gpt2", device="cpu")
    EMB_DIM = EMBS.shape[1]
    logging.info(f"Embeddings: {EMBS.shape}")

    # Fairness-Imbued run
    hp = Hyperparameters(
        embedding_dim=EMB_DIM, num_clusters=3, fairness_lambda=1.0,
        recluster_interval=NUM//10, warmup_samples=NUM//5,
        mi_buffer_size=1000, verbose=True, seed=42, device="cpu",
        fairness_boost_dynamic_scale=True, fairness_boost_scale_factor=0.5,
        boltzmann_beta=5.0
    )
    fair_model = FairnessImbuedCuriosityModel(hp)
    fr, fv, fm = run_simulation(fair_model, EMBS, 1000)

    # Normal run
    hp0 = Hyperparameters(
        embedding_dim=EMB_DIM, num_clusters=3, fairness_lambda=0.0,
        recluster_interval=NUM//10, warmup_samples=NUM//5,
        mi_buffer_size=1000, verbose=True, seed=42, device="cpu"
    )
    norm_model = FairnessImbuedCuriosityModel(hp0)
    nr, nv, nm = run_simulation(norm_model, EMBS, 1000)

    logging.info("Plotting results…")
    plot_results(fr, fv, fm, nr, nv, nm, LABELS, EMBS, hp)
    logging.info("Done.")
