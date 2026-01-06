"""
Visualize clusters in 2D space with criteria coloring.

This script creates a scatter plot similar to the reference image:
- Each criterion is colored differently
- Data points are projected to 2D using PCA
- Cluster boundaries are shown
- Dominant criterion in each cluster is highlighted
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter
from datasets import load_from_disk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.insert(0, project_root)

from src.cluster_preferences import load_clusters
from src.logistic_regression import load_precomputed_gradients_b

CRITERIA = ["helpfulness", "truthfulness", "instruction_following", "honesty"]
CRITERIA_COLORS = {
    "helpfulness": "#2E8B57",  # Green
    "truthfulness": "#8B008B",  # Dark purple
    "instruction_following": "#4169E1",  # Blue
    "honesty": "#FFD700",  # Yellow/Gold
}


def visualize_clusters(
    clusters_path: str,
    dataset_path: str,
    precompute_file: str = None,
    output_path: str = "pre_compute/cluster_visualization.png",
    downsample_ratio: float = None,
    seed: int = 42,
    max_samples: int = None,
    method: str = "pca",  # "pca" or "tsne"
    verbose: bool = True,
):
    """
    Visualize clusters in 2D space with criteria coloring.
    
    Args:
        clusters_path: Path to clusters .npy file
        dataset_path: Path to dataset directory
        precompute_file: Path to precomputed gradients file (for feature extraction)
        output_path: Path to save the visualization
        downsample_ratio: Downsample ratio used during precompute
        seed: Random seed used during precompute
        max_samples: Maximum samples used during precompute
        method: Dimensionality reduction method ("pca" or "tsne")
        verbose: Whether to print progress
    """
    # Load clusters
    clusters = load_clusters(clusters_path)
    
    if verbose:
        print("=" * 60)
        print("Visualizing Clusters")
        print("=" * 60)
        print(f"Loaded {len(clusters)} clusters")
        print(f"Total preferences: {sum(len(c) for c in clusters)}")
        print()
    
    # Load dataset and get criteria
    train_ds = load_from_disk(os.path.join(dataset_path, "train"))
    
    if "criterion_used" not in train_ds.column_names:
        raise ValueError("Dataset must have 'criterion_used' field")
    
    # Reapply downsample/shuffle to match precompute
    original_size = len(train_ds)
    
    if downsample_ratio is not None and downsample_ratio < 1.0:
        train_ds = train_ds.shuffle(seed=seed)
        n_samples = int(len(train_ds) * downsample_ratio)
        train_ds = train_ds.select(range(n_samples))
        if verbose:
            print(f"Applied downsample (ratio={downsample_ratio}, seed={seed})")
    
    if max_samples is not None and max_samples < len(train_ds):
        train_ds = train_ds.select(range(max_samples))
    
    all_criteria = train_ds["criterion_used"]
    
    # Get all data point indices from clusters
    all_indices = []
    cluster_labels = []  # Which cluster each point belongs to
    for cluster_idx, cluster in enumerate(clusters):
        for idx in cluster:
            all_indices.append(int(idx))
            cluster_labels.append(cluster_idx)
    
    all_indices = np.array(all_indices)
    cluster_labels = np.array(cluster_labels)
    
    # Get criteria for all data points
    data_criteria = [all_criteria[int(idx)] for idx in all_indices]
    
    if verbose:
        print(f"Total data points: {len(all_indices)}")
        crit_counts = Counter(data_criteria)
        for crit in CRITERIA:
            count = crit_counts.get(crit, 0)
            pct = count / len(data_criteria) * 100 if len(data_criteria) > 0 else 0
            print(f"  {crit}: {count} ({pct:.1f}%)")
        print()
    
    # Load features (gradients) for dimensionality reduction
    if precompute_file is None:
        # Try to find precompute file automatically based on downsample_ratio
        if downsample_ratio == 0.04:
            precompute_file = "Llama-3.2-1B-Instruct_200_0.04.pt"
        elif downsample_ratio == 0.025:
            precompute_file = "Llama-3.2-1B-Instruct_200_0.025.pt"
        else:
            raise ValueError(f"Precompute file not found. Please specify --precompute_file")
    
    if verbose:
        print(f"Loading features from: {precompute_file}")
    
    gradients_np, _, _, projection_matrix, _ = load_precomputed_gradients_b(precompute_file)
    
    # Extract features for our data points
    # Map cluster indices to gradient indices
    # The gradients are computed for the first N samples in the downsampled dataset
    # So we need to filter clusters to only include indices that have gradients
    max_gradient_idx = len(gradients_np) - 1
    
    # Filter to only include indices that have gradients
    valid_mask = all_indices <= max_gradient_idx
    valid_indices = all_indices[valid_mask]
    valid_cluster_labels = cluster_labels[valid_mask]
    valid_data_criteria = [data_criteria[i] for i in range(len(data_criteria)) if valid_mask[i]]
    
    if verbose:
        print(f"Gradient count: {len(gradients_np)}")
        print(f"Total cluster indices: {len(all_indices)}")
        print(f"Valid indices (<= {max_gradient_idx}): {len(valid_indices)}")
        if len(valid_indices) < len(all_indices):
            print(f"Warning: {len(all_indices) - len(valid_indices)} cluster indices exceed gradient range")
    
    # Extract features for valid indices
    features = np.stack([gradients_np[int(idx)] for idx in valid_indices], axis=0)
    
    # Update variables to use only valid data
    all_indices = valid_indices
    cluster_labels = valid_cluster_labels
    data_criteria = valid_data_criteria
    
    if verbose:
        print(f"Feature shape: {features.shape}")
    
    # Apply dimensionality reduction
    if verbose:
        print(f"\nApplying {method.upper()} dimensionality reduction...")
    
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        embeddings = reducer.fit_transform(features)
        if verbose:
            print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(reducer.explained_variance_ratio_):.2%}")
    elif method == "tsne":
        # Use perplexity based on data size
        perplexity = min(30, len(features) - 1)
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perplexity, n_iter=1000)
        embeddings = reducer.fit_transform(features)
        if verbose:
            print(f"t-SNE completed with perplexity={perplexity}")
    elif method == "umap":
        if not HAS_UMAP:
            raise ValueError("UMAP not installed. Install with: pip install umap-learn")
        reducer = UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        embeddings = reducer.fit_transform(features)
        if verbose:
            print(f"UMAP completed")
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: pca, tsne, umap")
    
    # Create the plot - show clusters clearly
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use different markers for each cluster to make them visually distinct
    cluster_markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
    cluster_colors_light = ['#90EE90', '#DA70D6', '#87CEEB', '#FFE4B5']  # Light colors for cluster backgrounds
    
    # First, draw cluster regions (convex hulls) as subtle backgrounds
    for cluster_idx in range(len(clusters)):
        cluster_mask = cluster_labels == cluster_idx
        if np.any(cluster_mask):
            cluster_embeddings = embeddings[cluster_mask]
            
            # Draw convex hull to show cluster boundary
            try:
                if len(cluster_embeddings) >= 3:
                    hull = ConvexHull(cluster_embeddings)
                    hull_points = cluster_embeddings[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.fill(hull_points[:, 0], hull_points[:, 1], 
                           color=cluster_colors_light[cluster_idx], 
                           alpha=0.2, 
                           edgecolor='gray',
                           linewidth=1.5,
                           zorder=0)
            except:
                pass
    
    # Plot points: color by criteria, shape by cluster
    for cluster_idx in range(len(clusters)):
        cluster_mask = cluster_labels == cluster_idx
        if np.any(cluster_mask):
            cluster_embeddings = embeddings[cluster_mask]
            cluster_criteria_subset = [data_criteria[i] for i in range(len(data_criteria)) if cluster_mask[i]]
            
            # Plot each criterion in this cluster
            for crit in CRITERIA:
                crit_mask = np.array([c == crit for c in cluster_criteria_subset])
                if np.any(crit_mask):
                    ax.scatter(
                        cluster_embeddings[crit_mask, 0],
                        cluster_embeddings[crit_mask, 1],
                        c=CRITERIA_COLORS[crit],
                        marker=cluster_markers[cluster_idx],
                        label=f'{crit}' if cluster_idx == 0 else '',  # Only label once per criterion
                        alpha=0.7,
                        s=50,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=1,
                    )
    
    # Add cluster labels at centroids
    for cluster_idx in range(len(clusters)):
        cluster_mask = cluster_labels == cluster_idx
        if np.any(cluster_mask):
            cluster_embeddings = embeddings[cluster_mask]
            center = cluster_embeddings.mean(axis=0)
            
            # Get dominant criterion
            cluster_criteria = [data_criteria[i] for i in range(len(data_criteria)) if cluster_mask[i]]
            crit_counts = Counter(cluster_criteria)
            dominant_crit = crit_counts.most_common(1)[0][0]
            dominant_count = crit_counts[dominant_crit]
            total = len(cluster_criteria)
            purity = dominant_count / total * 100 if total > 0 else 0
            
            ax.text(center[0], center[1], f'C{cluster_idx}', 
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black'),
                   ha='center', va='center', zorder=2)
    
    # Set axis labels based on method
    if method == "pca":
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
    elif method == "tsne":
        ax.set_xlabel("t-SNE Component 1", fontsize=12)
        ax.set_ylabel("t-SNE Component 2", fontsize=12)
    else:
        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
    
    ax.set_title("2D scatter colored by TRUE label", fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Set reasonable axis limits
    x_range = embeddings[:, 0].max() - embeddings[:, 0].min()
    y_range = embeddings[:, 1].max() - embeddings[:, 1].min()
    x_margin = x_range * 0.1
    y_margin = y_range * 0.1
    ax.set_xlim(embeddings[:, 0].min() - x_margin, embeddings[:, 0].max() + x_margin)
    ax.set_ylim(embeddings[:, 1].min() - y_margin, embeddings[:, 1].max() + y_margin)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f"\nVisualization saved to: {output_path}")
        print("=" * 60)
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize clusters in 2D space with criteria coloring"
    )
    parser.add_argument(
        "--clusters_file",
        type=str,
        required=True,
        help="Path to clusters .npy file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="notebooks/ultrafeedback_synthetic/mixed_criteria",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--precompute_file",
        type=str,
        default=None,
        help="Path to precomputed gradients file (for feature extraction)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="pre_compute/cluster_visualization.png",
        help="Path to save the visualization",
    )
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=None,
        help="Downsample ratio used during precompute",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples used during precompute",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=["pca", "tsne", "umap"],
        help="Dimensionality reduction method (default: tsne, better for cluster separation)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    visualize_clusters(
        clusters_path=args.clusters_file,
        dataset_path=args.dataset_path,
        precompute_file=args.precompute_file,
        output_path=args.output_file,
        downsample_ratio=args.downsample_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        method=args.method,
        verbose=args.verbose,
    )
