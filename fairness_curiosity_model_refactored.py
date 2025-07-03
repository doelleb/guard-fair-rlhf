
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math
from collections import deque
import logging
import random
from dataclasses import dataclass
from typing import List, Union

# Configure logging (initial setup, will be adjusted by verbose flag)
logging.basicConfig(level=logging.INFO, format=
    "%(asctime)s - %(levelname)s - %(message)s")

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.cuda.manual_seed_all(seed)

@dataclass
class Hyperparameters:
    embedding_dim: int = 64
    num_clusters: int = 10
    learning_rate: float = 0.001
    rnd_output_dim: int = 64
    warmup_samples: int = 50
    cluster_batch_size: int = 32
    recluster_interval: int = 50
    reward_norm_beta: float = 0.01
    fairness_lambda: float = 0.1
    mi_buffer_size: int = 10000
    alpha_curiosity: float = 1.0
    device: str = "cpu"
    verbose: bool = False
    seed: int = 42
    fairness_boost_dynamic_scale: bool = False
    fairness_boost_scale_factor: float = 1.0
    policy_type: str = "uniform"
    boltzmann_beta: float = 1.0

class PredictorNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(PredictorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0) # Reduced gain for predictor
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TargetNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(TargetNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for param in self.net.parameters():
            param.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0) # Reduced gain for target
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EmbeddingCollector:
    def __init__(self, mi_buffer_size: int = 10000):
        self.accumulated_embeddings = deque(maxlen=mi_buffer_size)

    def add_embeddings(self, embeddings: np.ndarray):
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        for emb in embeddings:
            self.accumulated_embeddings.append(emb)

    def get_all_embeddings(self) -> np.ndarray:
        if not self.accumulated_embeddings:
            return np.array([])
        return np.array(list(self.accumulated_embeddings))

class ClusterManager:
    def __init__(self, num_clusters: int, warmup_samples: int, cluster_batch_size: int, recluster_interval: int):
        self.num_clusters = num_clusters
        self.warmup_samples = warmup_samples
        self.cluster_batch_size = cluster_batch_size
        self.recluster_interval = recluster_interval
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        self.clusters_fitted = False
        self.cluster_visits = np.zeros(num_clusters)
        self.total_samples_processed = 0
        self.previous_centroids = None

    def update_clusters(self, embeddings_collector: EmbeddingCollector):
        all_current_embeddings = embeddings_collector.get_all_embeddings()

        if not self.clusters_fitted and len(all_current_embeddings) >= max(self.warmup_samples, self.num_clusters):
            logging.info(f"Fitting initial KMeans with {len(all_current_embeddings)} samples.")
            self.kmeans.fit(all_current_embeddings)
            self.clusters_fitted = True
            self.previous_centroids = self.kmeans.cluster_centers_
        elif self.clusters_fitted:
            if len(embeddings_collector.accumulated_embeddings) >= self.cluster_batch_size:
                batch = np.array(list(embeddings_collector.accumulated_embeddings)[-self.cluster_batch_size:])
                self.kmeans.partial_fit(batch)

            if self.total_samples_processed % self.recluster_interval == 0 and self.total_samples_processed > 0:
                if len(all_current_embeddings) >= self.num_clusters:
                    logging.info(f"Re-fitting KMeans with {len(all_current_embeddings)} samples at step {self.total_samples_processed}.")
                    if self.previous_centroids is not None:
                        centroid_drift = np.linalg.norm(self.kmeans.cluster_centers_ - self.previous_centroids)
                        logging.info(f"Cluster centroid drift: {centroid_drift:.4f}")
                    self.kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0, n_init=10)
                    self.kmeans.fit(all_current_embeddings)
                    self.previous_centroids = self.kmeans.cluster_centers_

    def get_cluster_id(self, embedding: np.ndarray) -> int:
        if not self.clusters_fitted:
            return -1
        return self.kmeans.predict(embedding.reshape(1, -1))[0]

    def increment_visit(self, cluster_id: int, total_samples_processed: int):
        if total_samples_processed > self.warmup_samples and cluster_id != -1:
            self.cluster_visits[cluster_id] += 1
            logging.debug(f"Incremented visit for cluster {cluster_id}. Total visits: {self.cluster_visits[cluster_id]}")

    def get_cluster_visit_variance(self) -> float:
        if np.sum(self.cluster_visits) == 0:
            return 0.0
        normalized_visits = self.cluster_visits / np.sum(self.cluster_visits)
        variance = np.var(normalized_visits)
        logging.debug(f"Cluster visit variance: {variance}")
        return variance

class CuriosityCore:

    def __init__(self, embedding_dim: int, rnd_output_dim: int, learning_rate: float, 
                 reward_norm_beta: float, fairness_lambda: float, mi_buffer_size: int, 
                 alpha_curiosity: float, device: str, 
                 fairness_boost_dynamic_scale: bool, fairness_boost_scale_factor: float):
        self.embedding_dim = embedding_dim
        self.rnd_output_dim = rnd_output_dim
        self.device = torch.device(device)
        self.alpha_curiosity = alpha_curiosity
        self.fairness_lambda = fairness_lambda
        self.fairness_boost_dynamic_scale = fairness_boost_dynamic_scale
        self.fairness_boost_scale_factor = fairness_boost_scale_factor

        self.predictor_net = PredictorNetwork(embedding_dim, rnd_output_dim).to(self.device)
        self.target_net = TargetNetwork(embedding_dim, rnd_output_dim).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=learning_rate)
        self.rnd_loss_fn = nn.MSELoss(reduction="mean")

        self.running_mean = 0.0
        self.running_var = 1.0
        self.reward_norm_beta = reward_norm_beta
        self.epsilon = 1e-8

        self.mi_buffer = deque(maxlen=mi_buffer_size)
        self.current_mi_estimate = 0.0
        self.mi_update_interval = 100

    def compute_novelty(self, embedding: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the RND novelty for a given embedding.

        Args:
            embedding (np.ndarray): The input embedding.

        Returns:
            tuple: A tuple containing:
                - novelty (torch.Tensor): The computed novelty score.
                - pred_out (torch.Tensor): Output of the predictor network.
                - target_out (torch.Tensor): Output of the target network.
        """
        embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(self.device)
        self.predictor_net.train()
        self.target_net.eval()
        
        with torch.no_grad():
            target_out = self.target_net(embedding_tensor)
        pred_out = self.predictor_net(embedding_tensor)
        novelty = self.rnd_loss_fn(pred_out, target_out)
        return novelty, pred_out, target_out

    def update_predictor(self, novelty_losses: List[torch.Tensor]):
        """Updates the RND predictor network based on a list of novelty losses.

        Args:
            novelty_losses (list): List of scalar novelty loss tensors.
        """
        loss_batch = torch.stack(novelty_losses)
        self.rnd_optimizer.zero_grad()
        loss_batch.mean().backward()
        self.rnd_optimizer.step()
        logging.debug(f"Predictor network updated. Mean loss: {loss_batch.mean().item():.4f}")

    def normalize_reward(self, novelty: float) -> float:
        """Normalizes the intrinsic reward using a running mean and variance.

        Args:
            novelty (float): The raw novelty score.

        Returns:
            float: The normalized and clipped reward.
        """
        self.running_mean = (1 - self.reward_norm_beta) * self.running_mean + self.reward_norm_beta * novelty
        self.running_var = (1 - self.reward_norm_beta) * self.running_var + self.reward_norm_beta * (novelty - self.running_mean)**2

        self.running_var = max(self.running_var, self.epsilon)
        if self.running_var < self.epsilon:
            normalized_reward = (novelty - self.running_mean) / self.epsilon
        else:
            normalized_reward = (novelty - self.running_mean) / (math.sqrt(self.running_var) + self.epsilon)
        clipped_reward = np.clip(normalized_reward, -3.0, 3.0) # Narrower clipping range
        logging.debug(f"Reward normalized: raw={novelty:.4f}, normalized={clipped_reward:.4f}")
        return clipped_reward

    def apply_fairness_boost(self, novelty: torch.Tensor, fairness_signal: float) -> torch.Tensor:
        """Applies a fairness boost to the novelty score.

        Args:
            novelty (torch.Tensor): The raw novelty score tensor.
            fairness_signal (float): The fairness constraint signal.

        Returns:
            torch.Tensor: The novelty score with fairness boost applied.
        """
        boost_amount = self.fairness_lambda * fairness_signal
        if self.fairness_boost_dynamic_scale:
            boost_amount *= math.sqrt(self.running_var) * self.fairness_boost_scale_factor
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Dynamic boost applied. Current std: {math.sqrt(self.running_var):.4f}, Boost amount: {boost_amount:.4f}")

        boosted_novelty = novelty + boost_amount
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Fairness boost: signal={fairness_signal:.2f}, lambda={self.fairness_lambda:.2f}, boost_amount={boost_amount:.4f}, original_novelty={novelty.item():.4f}, boosted_novelty={boosted_novelty.item():.4f}")
        return boosted_novelty

class FairnessImbuedCuriosityModel:
    """Integrates RND, clustering, and fairness boosting for intrinsic motivation.

    Args:
        hp (Hyperparameters): Hyperparameters for the model.
    """
    def __init__(self, hp: Hyperparameters):
        self.hp = hp
        if not self.hp.verbose:
            logging.getLogger().setLevel(logging.WARNING)
        logging.info(f"Initializing FairnessImbuedCuriosityModel with hyperparameters: {self.hp}")

        self.embedding_collector = EmbeddingCollector(hp.mi_buffer_size)
        self.cluster_manager = ClusterManager(hp.num_clusters, hp.warmup_samples, hp.cluster_batch_size, hp.recluster_interval)
        self.curiosity_core = CuriosityCore(hp.embedding_dim, hp.rnd_output_dim, hp.learning_rate,
                                            hp.reward_norm_beta, hp.fairness_lambda, hp.mi_buffer_size,
                                            hp.alpha_curiosity, hp.device,
                                            hp.fairness_boost_dynamic_scale, hp.fairness_boost_scale_factor)
        self.total_samples_processed = 0
        self.novelty_losses_buffer = deque(maxlen=hp.cluster_batch_size) # Buffer for RND predictor updates

    def observe(self, embeddings: np.ndarray, fairness_signals: np.ndarray):
        """Observes new embeddings and fairness signals, updates models, and computes rewards.

        Args:
            embeddings (np.ndarray): A batch of new embeddings.
            fairness_signals (np.ndarray): Corresponding fairness signals for the embeddings.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if fairness_signals.ndim == 0:
            fairness_signals = fairness_signals.reshape(1)

        for i, embedding in enumerate(embeddings):
            fairness_signal = fairness_signals[i]
            self.total_samples_processed += 1

            # 1. Compute Novelty
            novelty, pred_out, target_out = self.curiosity_core.compute_novelty(embedding)
            self.novelty_losses_buffer.append(self.curiosity_core.rnd_loss_fn(pred_out, target_out))

            # 2. Update RND Predictor (periodically or after a batch)
            if len(self.novelty_losses_buffer) >= self.hp.cluster_batch_size:
                self.curiosity_core.update_predictor(list(self.novelty_losses_buffer))
                self.novelty_losses_buffer.clear()

            # 3. Collect Embeddings for Clustering
            self.embedding_collector.add_embeddings(embedding)

            # 4. Update Clusters and Increment Visits
            self.cluster_manager.total_samples_processed = self.total_samples_processed # Sync total samples
            self.cluster_manager.update_clusters(self.embedding_collector)
            cluster_id = self.cluster_manager.get_cluster_id(embedding)
            self.cluster_manager.increment_visit(cluster_id, self.total_samples_processed)

            # 5. Apply Fairness Boost and Normalize Reward
            boosted_novelty = self.curiosity_core.apply_fairness_boost(novelty, fairness_signal)
            normalized_reward = self.curiosity_core.normalize_reward(boosted_novelty.item())

            # 6. Store for MI Estimation
            if cluster_id != -1:
                self.curiosity_core.mi_buffer.append((normalized_reward, cluster_id))

            # 7. Update MI Estimate (periodically)
            if self.total_samples_processed % self.curiosity_core.mi_update_interval == 0 and len(self.curiosity_core.mi_buffer) > 0:
                self.curiosity_core.current_mi_estimate = self._estimate_mutual_information()

    def intrinsic_reward(self, embedding: np.ndarray, fairness_signal: float) -> float:
        """Computes the intrinsic reward for a given embedding and fairness signal.

        Args:
            embedding (np.ndarray): The input embedding.
            fairness_signal (float): The fairness constraint signal.

        Returns:
            float: The final intrinsic reward.
        """
        novelty, _, _ = self.curiosity_core.compute_novelty(embedding)
        boosted_novelty = self.curiosity_core.apply_fairness_boost(novelty, fairness_signal)
        reward = self.curiosity_core.normalize_reward(boosted_novelty.item())
        return self.hp.alpha_curiosity * reward

    def get_cluster_visits(self) -> np.ndarray:
        """Returns the current cluster visit counts.

        Returns:
            np.ndarray: Array of cluster visit counts.
        """
        return self.cluster_manager.cluster_visits

    def get_mutual_information_estimate(self) -> float:
        """Returns the current mutual information estimate.

        Returns:
            float: The mutual information estimate.
        """
        if len(self.curiosity_core.mi_buffer) < self.curiosity_core.mi_update_interval: # Ensure enough data for a meaningful estimate
            logging.warning("Insufficient data in MI buffer for estimation.")
            return 0.0
        return self.curiosity_core.current_mi_estimate

    def _estimate_mutual_information(self) -> float:
        """Estimates mutual information I(Reward; Cluster) using collected data.
        This is a simplified estimation for demonstration.
        """
        if len(self.curiosity_core.mi_buffer) == 0:
            return 0.0

        rewards = np.array([item[0] for item in self.curiosity_core.mi_buffer])
        cluster_ids = np.array([item[1] for item in self.curiosity_core.mi_buffer])

        num_reward_bins = 10
        reward_bins = np.linspace(rewards.min(), rewards.max(), num_reward_bins + 1)
        reward_binned = np.digitize(rewards, reward_bins) - 1
        reward_binned = np.clip(reward_binned, 0, num_reward_bins - 1) # Ensure indices are within bounds

        joint_counts = np.zeros((num_reward_bins, self.hp.num_clusters))
        for i in range(len(rewards)):
            joint_counts[reward_binned[i], cluster_ids[i]] += 1
        
        p_rc = joint_counts / np.sum(joint_counts) + self.curiosity_core.epsilon # Add epsilon for stability

        p_r = np.sum(p_rc, axis=1, keepdims=True) # Sum over clusters
        p_c = np.sum(p_rc, axis=0, keepdims=True) # Sum over rewards

        with np.errstate(divide='ignore', invalid='ignore'): # Ignore log of zero warnings
            mi = np.sum(p_rc * np.log(p_rc / (p_r * p_c)))
        
        logging.info(f"Mutual Information estimated: {mi:.4f}")
        return mi

    def get_cluster_centroids(self) -> np.ndarray:
        """Returns the current cluster centroids.

        Returns:
            np.ndarray: Array of cluster centroids.
        """
        if not self.cluster_manager.clusters_fitted:
            return np.array([])
        return self.cluster_manager.kmeans.cluster_centers_

    def get_cluster_rewards(self) -> np.ndarray:
        """Estimates the average intrinsic reward for each cluster.

        Returns:
            np.ndarray: Array of average intrinsic rewards per cluster.
        """
        cluster_rewards = np.zeros(self.hp.num_clusters)
        cluster_counts = np.zeros(self.hp.num_clusters)

        for reward, cluster_id in self.curiosity_core.mi_buffer:
            if cluster_id != -1:
                cluster_rewards[cluster_id] += reward
                cluster_counts[cluster_id] += 1
        
        # Avoid division by zero
        cluster_rewards = np.divide(cluster_rewards, cluster_counts, 
                                    out=np.zeros_like(cluster_rewards), where=cluster_counts!=0)
        return cluster_rewards

    def get_policy_distribution(self) -> np.ndarray:
        """Calculates the policy distribution over clusters based on intrinsic rewards.

        Returns:
            np.ndarray: Probability distribution over clusters.
        """
        if self.hp.policy_type == "uniform":
            return np.ones(self.hp.num_clusters) / self.hp.num_clusters
        elif self.hp.policy_type == "boltzmann":
            cluster_rewards = self.get_cluster_rewards()
            # Apply softmax to convert rewards to probabilities
            exp_rewards = np.exp(self.hp.boltzmann_beta * cluster_rewards)
            policy_dist = exp_rewards / np.sum(exp_rewards)
            return policy_dist
        else:
            raise ValueError(f"Unknown policy type: {self.hp.policy_type}")

    def sample_cluster(self) -> int:
        """Samples a cluster ID based on the current policy distribution.

        Returns:
            int: The sampled cluster ID.
        """
        policy_dist = self.get_policy_distribution()
        # Handle cases where policy_dist might have NaNs or be all zeros
        if np.isnan(policy_dist).any() or np.sum(policy_dist) == 0:
            logging.warning("Policy distribution contains NaN or is all zeros. Falling back to uniform sampling.")
            policy_dist = np.ones(self.hp.num_clusters) / self.hp.num_clusters
        
        # Ensure probabilities sum to 1 (due to potential floating point issues)
        policy_dist = policy_dist / np.sum(policy_dist)

        return np.random.choice(self.hp.num_clusters, p=policy_dist)


