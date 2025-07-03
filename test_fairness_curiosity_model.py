
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from fairness_curiosity_model_refactored import FairnessImbuedCuriosityModel, Hyperparameters, set_seed
import logging
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_llm_embeddings(texts: list[str], model_name: str = "gpt2", device: str = "cpu", batch_size: int = 32) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        if embeddings.ndim == 1: # Handle single embedding case
            embeddings = embeddings.reshape(1, -1)
        embeddings_list.append(embeddings)
    return np.vstack(embeddings_list)


def run_simulation(model, all_embeddings, all_fairness_signals, num_steps):
    rewards = []
    cluster_visits = []
    mi_estimates = []

    # Initial observation to warm up the model and get initial cluster assignments
    # Observe a small random subset to avoid biasing initial clusters too much
    initial_indices = np.random.choice(len(all_embeddings), min(model.hp.warmup_samples, len(all_embeddings)), replace=False)
    model.observe(all_embeddings[initial_indices], all_fairness_signals[initial_indices])

    for i in range(num_steps):
        # Sample a cluster based on the model\'s current policy
        sampled_cluster_id = model.sample_cluster()

        # Find an embedding belonging to the sampled cluster
        # This is a simplified approach; in a real RL setting, the agent would interact with the environment
        # and receive an embedding corresponding to the chosen action/state.
        
        # Get cluster IDs for all embeddings
        all_cluster_ids = np.array([model.cluster_manager.get_cluster_id(emb) for emb in all_embeddings])
        cluster_embeddings_indices = np.where(all_cluster_ids == sampled_cluster_id)[0]
        
        if len(cluster_embeddings_indices) == 0:
            logging.warning(f"No embeddings found for sampled cluster {sampled_cluster_id}. Sampling randomly.")
            idx = np.random.randint(0, len(all_embeddings))
        else:
            idx = np.random.choice(cluster_embeddings_indices)

        current_embedding = all_embeddings[idx]
        current_fairness_signal = all_fairness_signals[idx]

        model.observe(np.array([current_embedding]), np.array([current_fairness_signal]))        
        reward = model.intrinsic_reward(current_embedding, current_fairness_signal)
        rewards.append(reward)
        cluster_visits.append(model.get_cluster_visits().copy())
        mi_estimates.append(model.get_mutual_information_estimate())

        if (i + 1) % 100 == 0:
            logging.info(f"Step {i+1}/{num_steps} - MI: {mi_estimates[-1]:.4f}, Visits: {model.get_cluster_visits()}")

    return rewards, cluster_visits, mi_estimates

def plot_results(fair_rewards, fair_visits, fair_mi, normal_rewards, normal_visits, normal_mi, category_labels, embeddings, hp, title_prefix=""):    

    num_plots = 4 # Rewards, Visits, MI, t-SNE
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))
    fig.suptitle(f'{title_prefix} Curiosity Model Comparison', fontsize=16)

    # Plot 1: Intrinsic Rewards Over Time
    axes[0].plot(fair_rewards, label='Fairness-Imbued Curiosity', alpha=0.7)
    axes[0].plot(normal_rewards, label='Normal Curiosity', alpha=0.7)
    axes[0].set_title('Intrinsic Reward Over Time')
    axes[0].set_xlabel('Simulation Step')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Cluster Visit Distribution (End State)
    final_fair_visits = fair_visits[-1] if fair_visits else np.zeros(hp.num_clusters)
    final_normal_visits = normal_visits[-1] if normal_visits else np.zeros(hp.num_clusters)

    bar_width = 0.35
    index = np.arange(len(final_fair_visits))

    axes[1].bar(index, final_fair_visits, bar_width, label='Fairness-Imbued', alpha=0.7)
    axes[1].bar(index + bar_width, final_normal_visits, bar_width, label='Normal', alpha=0.7)
    axes[1].set_title('Final Cluster Visit Distribution')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Visits')
    axes[1].set_xticks(index + bar_width / 2)
    axes[1].set_xticklabels([f'Cluster {i}' for i in range(len(final_fair_visits))])
    axes[1].legend()
    axes[1].grid(axis='y')

    # Plot 3: Mutual Information Estimate Over Time
    axes[2].plot(fair_mi, label='Fairness-Imbued MI', alpha=0.7)
    axes[2].plot(normal_mi, label='Normal Curiosity MI', alpha=0.7)
    axes[2].set_title('Mutual Information Estimate (Reward vs. Cluster) Over Time')
    axes[2].set_xlabel('Simulation Step (MI Update Interval)')
    axes[2].set_ylabel('Mutual Information')
    axes[2].legend()
    axes[2].grid(True)

    # Plot 4: t-SNE Visualization of Embeddings with Cluster Assignments
    if embeddings.shape[0] > 1 and embeddings.shape[1] > 2:
        try:
            tsne = TSNE(n_components=2, random_state=hp.seed, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = tsne.fit_transform(embeddings)

            scatter = axes[3].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=category_labels, cmap='viridis', alpha=0.6)
            axes[3].set_title('t-SNE Visualization of Embeddings by Category')
            axes[3].set_xlabel('t-SNE Component 1')
            axes[3].set_ylabel('t-SNE Component 2')
            plt.colorbar(scatter, ax=axes[3], label='Category Label')
            axes[3].grid(True)
        except Exception as e:
            logging.warning(f"Could not perform t-SNE visualization: {e}")
            axes[3].set_title('t-SNE Visualization (Failed)')
            axes[3].text(0.5, 0.5, 'Visualization failed due to data constraints or error.', 
                         horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)
    else:
        axes[3].set_title('t-SNE Visualization (Not Enough Data)')
        axes[3].text(0.5, 0.5, 'Not enough data for t-SNE visualization.', 
                     horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{title_prefix.replace(" ", "_").lower()}curiosity_comparison.png')
    plt.show()


if __name__ == '__main__':
    set_seed(42)
    logging.getLogger().setLevel(logging.INFO)

    LLM_MODEL_NAME = "gpt2"
    DEVICE = "cpu"
    NUM_SAMPLES = 500 # Increased for better clustering and MI estimation
    NUM_SIMULATION_STEPS = 1000 # Increased simulation steps
    NUM_CATEGORIES = 3 # Number of distinct categories/clusters

    logging.info(f"Loading dataset and generating embeddings using {LLM_MODEL_NAME}...")
    # Using a larger, more diverse set of texts to generate more meaningful embeddings
    # This is still a controlled set to ensure distinct categories for fairness signals
    texts = [
        "The quick brown fox jumps over the lazy dog.", # Category 0
        "Artificial intelligence is transforming industries globally.", # Category 0
        "Machine learning algorithms are at the core of modern data analysis.", # Category 0
        "Natural language processing enables computers to understand human language.", # Category 0
        "Deep learning models require vast amounts of data for training.", # Category 0

        "The sun rises in the east and sets in the west.", # Category 1
        "Water boils at 100 degrees Celsius at sea level.", # Category 1
        "Photosynthesis is the process used by plants to convert light energy into chemical energy.", # Category 1
        "The Earth revolves around the Sun.", # Category 1
        "Gravity is a fundamental force of nature.", # Category 1

        "A cat purrs when it is content.", # Category 2
        "Dogs are often called man's best friend.", # Category 2
        "Birds build nests to lay their eggs.", # Category 2
        "Fish swim in water using their fins.", # Category 2
        "Lions are apex predators in their ecosystem.", # Category 2
    ]
    # Extend texts to reach NUM_SAMPLES, ensuring variety
    while len(texts) < NUM_SAMPLES:
        texts.extend(texts)
    texts = texts[:NUM_SAMPLES]

    # Assign category labels based on the text groups
    category_labels = np.array([0]*5 + [1]*5 + [2]*5)
    while len(category_labels) < NUM_SAMPLES:
        category_labels = np.append(category_labels, category_labels)
    category_labels = category_labels[:NUM_SAMPLES]

    embeddings = get_llm_embeddings(texts, model_name=LLM_MODEL_NAME, device=DEVICE)
    EMBEDDING_DIM = embeddings.shape[1]
    logging.info(f"Generated {len(embeddings)} embeddings with dimension {EMBEDDING_DIM}.")

    # Define fairness signals based on categories. For example, prioritize Category 0, penalize Category 2.
    # This simulates a scenario where some categories are 'under-explored' or 'undesirable' from a fairness perspective.
    fairness_signals_for_simulation = np.array([
        1.0 if label == 0 else (-1.0 if label == 2 else 0.0) 
        for label in category_labels
    ])

    # Fairness-Imbued Model Hyperparameters
    hp_fair = Hyperparameters(
        embedding_dim=EMBEDDING_DIM,
        num_clusters=NUM_CATEGORIES,
        fairness_lambda=1.0, # Increased lambda to make fairness boost more impactful
        recluster_interval=NUM_SAMPLES // 10, # Recluster more frequently
        warmup_samples=NUM_SAMPLES // 5,
        mi_buffer_size=NUM_SIMULATION_STEPS, # Buffer size should be at least simulation steps for MI over time
        verbose=True, # Enable verbose logging for debugging
        seed=42,
        device=DEVICE,
        fairness_boost_dynamic_scale=True, # Enable dynamic scaling
        fairness_boost_scale_factor=0.5, # Adjust scale factor
        policy_type="boltzmann", # Enable policy-driven sampling
        boltzmann_beta=5.0 # Increased beta for stronger policy influence
    )
    fair_model = FairnessImbuedCuriosityModel(hp_fair)

    logging.info("Running simulation for Fairness-Imbued Curiosity Model...")
    fair_rewards, fair_visits, fair_mi = run_simulation(
        fair_model, embeddings, fairness_signals_for_simulation, NUM_SIMULATION_STEPS
    )
    logging.info("Fairness-Imbued Curiosity Model simulation complete.")

    # Normal Curiosity Model Hyperparameters (fairness_lambda=0)
    hp_normal = Hyperparameters(
        embedding_dim=EMBEDDING_DIM,
        num_clusters=NUM_CATEGORIES,
        fairness_lambda=0.0, # No fairness boost
        recluster_interval=NUM_SAMPLES // 10,
        warmup_samples=NUM_SAMPLES // 5,
        mi_buffer_size=NUM_SIMULATION_STEPS,
        verbose=True,
        seed=42,
        device=DEVICE,
        policy_type="boltzmann", # Also use boltzmann for normal to compare policy behavior
        boltzmann_beta=5.0
    )
    normal_model = FairnessImbuedCuriosityModel(hp_normal)

    # For normal model, fairness signals are all zero as there's no fairness component
    normal_fairness_signals = np.zeros_like(fairness_signals_for_simulation)

    logging.info("Running simulation for Normal Curiosity Model...")
    normal_rewards, normal_visits, normal_mi = run_simulation(
        normal_model, embeddings, normal_fairness_signals, NUM_SIMULATION_STEPS
    )
    logging.info("Normal Curiosity Model simulation complete.")

    logging.info("Plotting results...")
    plot_results(
        fair_rewards, fair_visits, fair_mi,
        normal_rewards, normal_visits, normal_mi,
        category_labels, embeddings,
        hp=hp_fair,
        title_prefix="Fairness vs. Normal"
    )
    logging.info("Comparison plots generated and saved.")

    logging.info("Script finished.")




